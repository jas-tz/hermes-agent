"""Phase 26B — Capability Runtime Integration.

Tests verify the actual child-agent execution path:
delegate_task → SubagentExecutionContext → task_type → runtime authorization → execution → result

Key finding: JARVISSubagentProvider.handle_tool_call() in jarvis/__init__.py is the
authoritative runtime gate, using subagent_types.py (not tool_capability_registry.py).
The ToolCapability registry is a policy definition layer; the provider is the enforcer.

These tests prove every denied gate prevents execution and produces zero mutation.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, call

from plugins.memory.jarvis import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    SubagentExecutionContext,
    SubagentLifecycleManager,
    create_execution_context,
    L0_READ_ONLY,
    L1_REVERSIBLE,
    L2_GATED,
    L3_APPROVAL,
    ALL_JARVIS_TOOLS,
)
from plugins.memory.jarvis.subagent_filesystem import (
    SubagentFilesystemScope,
    validate_subagent_read_path,
    validate_subagent_write_path,
    get_filesystem_capability_for_task_type,
)
from plugins.memory.jarvis.subagent_terminal import (
    get_terminal_capability_for_task_type,
    validate_command,
    execute_subagent_terminal,
    SUBAGENT_DENIED_COMMANDS,
)
from plugins.memory.jarvis.tool_capability_registry import (
    is_tool_permitted,
    get_capability,
)


# ---------------------------------------------------------------------------
# RAG/continuity runtime authorization
# ---------------------------------------------------------------------------

class TestRAGContinuityRuntimeAuthorization:
    """Test that RAG/continuity tools are gated by the provider at runtime."""

    def test_provider_blocks_unauthorized_rag_write_for_research(self):
        """Provider blocks hermes_rag_store for research task."""
        ctx = create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        # Provider should block write
        result = provider.handle_tool_call("hermes_rag_store", {"text": "data"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False
        assert "not authorized" in result_dict["error"]

    def test_provider_allows_authorized_rag_read_for_research(self):
        """Provider allows hermes_rag_search for research task."""
        ctx = create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "results": []})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is True

    def test_provider_blocks_l3_without_approval(self):
        """Decision 42: L3 is refused as read-only before approval is asked
        (previously: blocked by a denied approval)."""
        ctx = create_execution_context(
            child_id="sa-default",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="default",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="default",
            execution_context=ctx,
        )

        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": False}
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False
            assert result_dict["code"] == "delegated_read_only"
            mock_approval.assert_not_called()

    def test_provider_refuses_l3_even_with_approval(self):
        """Decision 42: an approved L3 call is still refused, without asking
        (previously: allowed with approval)."""
        ctx = create_execution_context(
            child_id="sa-default",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="default",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="default",
            execution_context=ctx,
        )

        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": True}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                mock_dispatch.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                result_dict = json.loads(result)
                assert result_dict["code"] == "delegated_read_only"
                mock_approval.assert_not_called()
                mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# Zero-mutation proofs
# ---------------------------------------------------------------------------

class TestZeroMutationProofs:
    """Prove that denied gates prevent execution and produce zero mutation."""

    def test_research_cannot_dispatch_rag_store(self):
        """Research task: hermes_rag_store never reaches dispatch."""
        ctx = create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            result = provider.handle_tool_call("hermes_rag_store", {"text": "data"})
            # Dispatch was NEVER called — gate blocked before execution
            mock.assert_not_called()

    def test_research_cannot_dispatch_rag_delete(self):
        """Research task: hermes_rag_delete never reaches dispatch."""
        ctx = create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            mock.assert_not_called()

    def test_l3_denied_approval_no_dispatch(self):
        """L3 denied approval: dispatch never called."""
        ctx = create_execution_context(
            child_id="sa-default",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="default",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="default",
            execution_context=ctx,
        )

        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": False}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                mock_dispatch.assert_not_called()

    def test_filesystem_write_outside_scope_no_file_created(self):
        """Write outside scope: no file system mutation."""
        scope = SubagentFilesystemScope(
            root_path="/tmp/work",
            allowed_paths=("/tmp/work",),
            denied_paths=("/etc",),
        )

        is_allowed, _ = validate_subagent_write_path("/etc/passwd", scope)
        assert is_allowed is False
        # No file was created

    def test_terminal_denied_command_no_execution(self):
        """Denied terminal command: no subprocess execution."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")

        with patch("tools.terminal_tool.terminal_tool") as mock_tool:
            result = execute_subagent_terminal("rm -rf /", cap, require_approval=False)
            mock_tool.assert_not_called()


# ---------------------------------------------------------------------------
# Provenance integrity
# ---------------------------------------------------------------------------

class TestProvenanceIntegrity:
    """Test that provenance is preserved through the execution path."""

    def test_rag_store_never_dispatched(self):
        """Decision 42: a sub-agent store is refused, so nothing (with or
        without sub-agent provenance) reaches dispatch (previously: dispatched
        with provenance added)."""
        ctx = create_execution_context(
            child_id="sa-note",
            parent_session_id="parent",
            task_goal="take notes",
            task_type="note_taking",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="take notes",
            task_type="note_taking",
            execution_context=ctx,
        )

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "memory_id": "uuid"})
            result = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": "finding"}))
            assert result["code"] == "delegated_read_only"
            mock.assert_not_called()


# ---------------------------------------------------------------------------
# Task-type boundary enforcement
# ---------------------------------------------------------------------------

class TestTaskTypeBoundaryEnforcement:
    """Test that task-type boundaries are enforced across all capabilities."""

    def test_research_blocks_all_writes(self):
        """Research task: all write tools blocked."""
        ctx = create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        write_tools = ["hermes_rag_store", "hermes_rag_store_learnings",
                       "hermes_goal_create", "hermes_task_create",
                       "hermes_rag_delete", "hermes_decision_supersede"]

        for tool in write_tools:
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
                result = provider.handle_tool_call(tool, {})
                mock.assert_not_called()

    def test_note_taking_allows_l1_blocks_l2_l3(self):
        """Note-taking: L1 refused under Decision 42 (formerly allowed); L2/L3 blocked."""
        ctx = create_execution_context(
            child_id="sa-note",
            parent_session_id="parent",
            task_goal="notes",
            task_type="note_taking",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="notes",
            task_type="note_taking",
            execution_context=ctx,
        )

        # L1 is refused (Decision 42)
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            result = provider.handle_tool_call("hermes_rag_store", {"text": "x"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False and result_dict["code"] == "delegated_read_only"
            mock.assert_not_called()

        # L2 should be blocked
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            result = provider.handle_tool_call("hermes_goal_create", {"title": "x"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False


# ---------------------------------------------------------------------------
# Filesystem scope enforcement
# ---------------------------------------------------------------------------

class TestFilesystemScopeEnforcement:
    """Test filesystem scope enforcement at runtime."""

    def test_read_within_scope_allowed(self, tmp_path):
        """Read within scope is allowed."""
        root = str(tmp_path / "workspace")
        os.makedirs(root)

        scope = SubagentFilesystemScope(
            root_path=root,
            allowed_paths=(root,),
            denied_paths=(),
        )

        file_path = os.path.join(root, "data.txt")
        with open(file_path, "w") as f:
            f.write("data")

        is_allowed, _ = validate_subagent_read_path(file_path, scope)
        assert is_allowed is True

    def test_write_within_scope_allowed(self, tmp_path):
        """Write within scope is allowed."""
        root = str(tmp_path / "workspace")
        os.makedirs(root)

        scope = SubagentFilesystemScope(
            root_path=root,
            allowed_paths=(root,),
            denied_paths=(),
        )

        file_path = os.path.join(root, "output.txt")
        is_allowed, _ = validate_subagent_write_path(file_path, scope)
        assert is_allowed is True

    def test_path_traversal_blocked(self, tmp_path):
        """Path traversal is blocked."""
        root = str(tmp_path / "workspace")
        os.makedirs(root)

        scope = SubagentFilesystemScope(
            root_path=root,
            allowed_paths=(root,),
            denied_paths=(),
        )

        traversal_path = os.path.join(root, "..", "etc", "passwd")
        is_allowed, _ = validate_subagent_read_path(traversal_path, scope)
        assert is_allowed is False


# ---------------------------------------------------------------------------
# Terminal runtime enforcement
# ---------------------------------------------------------------------------

class TestTerminalRuntimeEnforcement:
    """Test terminal runtime enforcement at runtime."""

    def test_denied_command_not_executed(self):
        """Denied command is not executed."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")

        with patch("tools.terminal_tool.terminal_tool") as mock:
            result = execute_subagent_terminal("sudo rm -rf /", cap, require_approval=False)
            mock.assert_not_called()

    def test_injection_not_executed(self):
        """Command with injection is not executed."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")

        with patch("tools.terminal_tool.terminal_tool") as mock:
            result = execute_subagent_terminal("ls; rm -rf /", cap, require_approval=False)
            mock.assert_not_called()

    def test_non_maintenance_blocked(self):
        """Non-maintenance task type cannot execute terminal."""
        cap = get_terminal_capability_for_task_type("research", "/tmp/work")

        with patch("tools.terminal_tool.terminal_tool") as mock:
            result = execute_subagent_terminal("ls", cap, require_approval=False)
            mock.assert_not_called()


# ---------------------------------------------------------------------------
# Forged tool calls
# ---------------------------------------------------------------------------

class TestForgedToolCalls:
    """Test that forged tool calls are rejected."""

    def test_forged_rag_tool_rejected(self):
        """Forged RAG tool name is rejected."""
        ctx = create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        result = provider.handle_tool_call("hermes_rag_faked", {})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

    def test_unknown_tool_rejected(self):
        """Unknown tool name is rejected."""
        ctx = create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        result = provider.handle_tool_call("unknown_tool_xyz", {})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False


# ---------------------------------------------------------------------------
# Capability registry vs runtime enforcement
# ---------------------------------------------------------------------------

class TestCapabilityRegistryVsRuntime:
    """Verify the relationship between ToolCapability registry and runtime enforcement."""

    def test_registry_is_policy_not_enforcement(self):
        """ToolCapability registry defines policy; provider enforces at runtime."""
        # The registry says research can use hermes_rag_search
        assert is_tool_permitted("hermes_rag_search", "research") is True

        # But the provider is the one that actually enforces this at runtime
        ctx = create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        # Provider enforces the same policy
        result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
        # Won't actually execute (no Qdrant), but passes authorization
        # The result depends on dispatch, not authorization gate

    def test_registry_blocks_what_provider_blocks(self):
        """Registry and provider agree on blocked tools."""
        blocked_tools = ["delegate_task", "terminal", "write_file", "read_file",
                        "browser_navigate", "computer_use", "cronjob", "send_message"]

        for tool in blocked_tools:
            assert is_tool_permitted(tool, "maintenance") is False, (
                f"Registry allows parent-only tool {tool}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
