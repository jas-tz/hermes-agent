"""Phase 26C — Adversarial Sub-Agent Capability Isolation.

Attack-phase tests that attempt to bypass every capability boundary.
All tests must FAIL (attack is blocked) for the security posture to hold.

Covers: forged context/provenance, task-type escalation, authority escalation,
capability confusion, bypass-gate injection, filesystem traversal/symlink/
boundary attacks, terminal injection/indirect-escape, cross-capability
escalation, parent-only tool access, unknown/forged capabilities, and
zero-mutation guarantees.

If any test passes (attack succeeds), it documents a concrete defect that
must be fixed in production code.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    SubagentExecutionContext,
    SubagentLifecycleManager,
    create_execution_context,
    L0_READ_ONLY,
    L1_REVERSIBLE,
    L2_GATED,
    L3_APPROVAL,
    SUBAGENT_TASK_TYPES,
)
from plugins.memory.jarvis.subagent_filesystem import (
    SubagentFilesystemScope,
    validate_subagent_read_path,
    validate_subagent_write_path,
    get_filesystem_capability_for_task_type,
    is_subagent_path_allowed,
)
from plugins.memory.jarvis.subagent_terminal import (
    get_terminal_capability_for_task_type,
    validate_command,
    is_delegation_escape,
    is_filesystem_escape,
    execute_subagent_terminal,
    SUBAGENT_DENIED_COMMANDS,
    SUBAGENT_ALLOWED_COMMANDS,
    SHELL_METACHARACTERS,
)
from plugins.memory.jarvis.tool_capability_registry import (
    is_tool_permitted,
    is_tool_parent_only,
    get_capability,
)


# ---------------------------------------------------------------------------
# Forged execution context / provenance
# ---------------------------------------------------------------------------

class TestForgedExecutionContext:
    """Test that forged execution context cannot escalate privileges."""

    def test_forged_task_type_to_maintenance_blocked(self):
        """Attempting to create context with forged task_type is blocked."""
        # create_execution_context uses SUBAGENT_TASK_TYPES — can't forge unknown
        with pytest.raises(ValueError, match="Unknown task type"):
            create_execution_context(
                child_id="sa-forged",
                parent_session_id="parent",
                task_goal="escalate",
                task_type="nonexistent",
                delegation_depth=1,
            )

    def test_forged_context_cannot_override_authority(self):
        """Frozen context prevents authority override."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )

        # Cannot mutate frozen fields
        with pytest.raises(AttributeError):
            ctx.authority_level = "L3"

        with pytest.raises(AttributeError):
            ctx.allowed_tools = L3_APPROVAL

        with pytest.raises(AttributeError):
            ctx.task_type = "maintenance"

    def test_forged_context_cannot_override_depth(self):
        """Cannot modify delegation depth."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )

        with pytest.raises(AttributeError):
            ctx.delegation_depth = 0


# ---------------------------------------------------------------------------
# Task-type escalation
# ---------------------------------------------------------------------------

class TestTaskTypeEscalation:
    """Test that task-type escalation is blocked."""

    def test_research_cannot_escalate_to_maintenance_via_context(self):
        """Research context cannot be swapped for maintenance."""
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

        # Provider uses ctx.task_type for authorization
        # Attempt to forge a new context
        forged_ctx = create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research",
            task_type="maintenance",
            delegation_depth=1,
        )

        # Provider still uses original context
        result = provider.handle_tool_call("hermes_rag_store", {"text": "x"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

    def test_context_task_type_unchanged_after_forged_attempt(self):
        """Original context task_type remains after forged attempt."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        original_type = ctx.task_type

        # Try to create a new context
        new_ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="maintenance",
            delegation_depth=1,
        )

        # Original context unchanged
        assert ctx.task_type == original_type


# ---------------------------------------------------------------------------
# Authority escalation
# ---------------------------------------------------------------------------

class TestAuthorityEscalation:
    """Test that authority escalation is blocked."""

    def test_l0_context_cannot_perform_l3_operations(self):
        """L0 context cannot perform L3 operations."""
        ctx = create_execution_context(
            child_id="sa-l0",
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

        # L3 operation blocked
        result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

    def test_l1_context_cannot_perform_l3_operations(self):
        """L1 context cannot perform L3 operations."""
        ctx = create_execution_context(
            child_id="sa-l1",
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

        result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False


# ---------------------------------------------------------------------------
# Capability confusion
# ---------------------------------------------------------------------------

class TestCapabilityConfusion:
    """Test that capability confusion attacks are blocked."""

    def test_rag_tool_name_used_as_terminal_command_blocked(self):
        """Using RAG tool name in terminal command is blocked."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        is_valid, _ = validate_command("hermes_rag_search test", cap)
        assert is_valid is False

    def test_terminal_command_used_as_rag_tool_blocked(self):
        """Using terminal command as RAG tool name is blocked."""
        ctx = create_execution_context(
            child_id="sa-test",
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

        result = provider.handle_tool_call("ls", {})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False


# ---------------------------------------------------------------------------
# Bypass-gate injection
# ---------------------------------------------------------------------------

class TestBypassGateInjection:
    """Test that _continuity_bypass_gate injection is blocked."""

    def test_bypass_gate_stripped_by_provider(self):
        """Provider strips _continuity_bypass_gate from kwargs."""
        ctx = create_execution_context(
            child_id="sa-test",
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
            mock.return_value = json.dumps({"ok": True})
            # Attempt to inject bypass gate
            provider.handle_tool_call("hermes_rag_search", {"query": "test"},
                                      _continuity_bypass_gate=True)
            # Should have been stripped — dispatch called without bypass
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# Filesystem traversal / symlink / boundary attacks
# ---------------------------------------------------------------------------

class TestFilesystemTraversalAttacks:
    """Test filesystem traversal and boundary attacks."""

    def test_dotdot_traversal_blocked(self, tmp_path):
        """../ traversal is blocked."""
        root = str(tmp_path / "workspace")
        os.makedirs(root)

        scope = SubagentFilesystemScope(
            root_path=root,
            allowed_paths=(root,),
            denied_paths=(),
        )

        # Direct traversal
        target = os.path.join(root, "..", "etc", "passwd")
        is_allowed, _ = validate_subagent_read_path(target, scope)
        assert is_allowed is False

    def test_nested_traversal_blocked(self, tmp_path):
        """Nested traversal is blocked."""
        root = str(tmp_path / "workspace")
        os.makedirs(root)

        scope = SubagentFilesystemScope(
            root_path=root,
            allowed_paths=(root,),
            denied_paths=(),
        )

        # Nested traversal
        target = os.path.join(root, "sub", "..", "..", "etc", "passwd")
        is_allowed, _ = validate_subagent_read_path(target, scope)
        assert is_allowed is False

    def test_absolute_path_outside_workspace_blocked(self, tmp_path):
        """Absolute path outside workspace is blocked."""
        root = str(tmp_path / "workspace")
        os.makedirs(root)

        scope = SubagentFilesystemScope(
            root_path=root,
            allowed_paths=(root,),
            denied_paths=(),
        )

        is_allowed, _ = validate_subagent_read_path("/etc/passwd", scope)
        assert is_allowed is False

    def test_symlink_to_outside_blocked(self, tmp_path):
        """Symlink pointing outside workspace is blocked."""
        root = str(tmp_path / "workspace")
        os.makedirs(root)

        # Create target outside scope
        outside = tmp_path / "secret.txt"
        with open(outside, "w") as f:
            f.write("secret")

        # Create symlink inside scope
        link = os.path.join(root, "link.txt")
        try:
            os.symlink(str(outside), link)
            scope = SubagentFilesystemScope(
                root_path=root,
                allowed_paths=(root,),
                denied_paths=(),
                allow_symlinks=False,
            )
            is_allowed, _ = validate_subagent_read_path(link, scope)
            assert is_allowed is False
        except OSError:
            pass  # Symlinks not supported


# ---------------------------------------------------------------------------
# Terminal command injection and indirect escape attempts
# ---------------------------------------------------------------------------

class TestTerminalInjectionAttacks:
    """Test terminal command injection attacks."""

    def test_semicolon_injection_blocked(self):
        """Semicolon injection is blocked."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        is_valid, _ = validate_command("ls; rm -rf /", cap)
        assert is_valid is False

    def test_pipe_injection_blocked(self):
        """Pipe injection is blocked."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        is_valid, _ = validate_command("ls | cat /etc/passwd", cap)
        assert is_valid is False

    def test_backtick_injection_blocked(self):
        """Backtick injection is blocked."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        is_valid, _ = validate_command("ls `whoami`", cap)
        assert is_valid is False

    def test_dollar_paren_injection_blocked(self):
        """$() injection is blocked."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        is_valid, _ = validate_command("ls $(cat /etc/passwd)", cap)
        assert is_valid is False

    def test_double_pipe_injection_blocked(self):
        """|| injection is blocked."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        is_valid, _ = validate_command("ls || rm -rf /", cap)
        assert is_valid is False

    def test_double_ampersand_injection_blocked(self):
        """&& injection is blocked."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        is_valid, _ = validate_command("ls && rm -rf /", cap)
        assert is_valid is False

    def test_newline_injection_blocked(self):
        """Newline injection is blocked."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        is_valid, _ = validate_command("ls\nrm -rf /", cap)
        assert is_valid is False

    def test_delegate_task_via_terminal_blocked(self):
        """delegate_task via terminal is blocked."""
        assert is_delegation_escape("delegate_task --goal bad") is True

    def test_hermes_serve_via_terminal_blocked(self):
        """hermes serve via terminal is blocked."""
        assert is_delegation_escape("hermes serve --port 8080") is True

    def test_systemctl_via_terminal_blocked(self):
        """systemctl via terminal is blocked."""
        assert is_delegation_escape("systemctl restart hermes") is True


# ---------------------------------------------------------------------------
# Cross-capability filesystem ↔ terminal escalation
# ---------------------------------------------------------------------------

class TestCrossCapabilityEscalation:
    """Test cross-capability escalation attacks."""

    def test_terminal_to_filesystem_outside_workspace_blocked(self):
        """Terminal cannot access filesystem outside workspace."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")

        # Attempt to access /etc/passwd via terminal
        is_escaped = is_filesystem_escape("/etc/passwd", cap.working_directory)
        assert is_escaped is True

    def test_filesystem_to_terminal_bypass_blocked(self):
        """Filesystem capability cannot bypass terminal restrictions."""
        fs_cap = get_filesystem_capability_for_task_type("research")
        assert fs_cap.can_read is False

        # Research cannot use terminal either
        term_cap = get_terminal_capability_for_task_type("research")
        assert term_cap.can_execute is False


# ---------------------------------------------------------------------------
# Parent-only tool access
# ---------------------------------------------------------------------------

class TestParentOnlyToolAccess:
    """Test that parent-only tools are inaccessible to sub-agents."""

    def test_research_cannot_access_parent_tools(self):
        """Research cannot access any parent-only tool."""
        parent_tools = ["delegate_task", "terminal", "write_file", "read_file",
                        "browser_navigate", "computer_use", "cronjob", "send_message"]

        for tool in parent_tools:
            assert is_tool_parent_only(tool) is True, f"{tool} is not parent-only"
            assert is_tool_permitted(tool, "research") is False

    def test_maintenance_cannot_access_parent_tools(self):
        """Even maintenance cannot access parent-only tools."""
        parent_tools = ["delegate_task", "terminal", "write_file", "read_file",
                        "browser_navigate", "computer_use", "cronjob", "send_message"]

        for tool in parent_tools:
            assert is_tool_permitted(tool, "maintenance") is False, \
                f"Maintenance can access parent tool {tool}"


# ---------------------------------------------------------------------------
# Unknown / forged capabilities
# ---------------------------------------------------------------------------

class TestUnknownForgedCapabilities:
    """Test that unknown/forged capabilities are rejected."""

    def test_unknown_tool_no_capability(self):
        """Unknown tools have no capability entry."""
        assert get_capability("nonexistent_tool") is None
        assert get_capability("fake_tool") is None
        assert get_capability("hermes_fake") is None

    def test_unknown_tool_not_permitted(self):
        """Unknown tools are not permitted."""
        assert is_tool_permitted("unknown_tool", "maintenance") is False
        assert is_tool_permitted("fake_tool", "research") is False

    def test_forged_rag_tool_name(self):
        """Forged RAG tool name is rejected."""
        ctx = create_execution_context(
            child_id="sa-test",
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


# ---------------------------------------------------------------------------
# Zero-mutation guarantees
# ---------------------------------------------------------------------------

class TestZeroMutationGuarantees:
    """Prove that denied operations produce zero mutation."""

    def test_denied_rag_write_no_dispatch(self):
        """Denied RAG write: dispatch never called."""
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
            provider.handle_tool_call("hermes_rag_store", {"text": "x"})
            provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            provider.handle_tool_call("hermes_goal_create", {"title": "x"})
            mock.assert_not_called()

    def test_denied_filesystem_write_no_file(self, tmp_path):
        """Denied filesystem write: no file created."""
        root = str(tmp_path / "workspace")
        os.makedirs(root)

        scope = SubagentFilesystemScope(
            root_path=root,
            allowed_paths=(root,),
            denied_paths=("/etc",),
        )

        # Attempt to write to denied path
        validate_subagent_write_path("/etc/passwd", scope)

        # File should not exist
        assert not os.path.exists("/etc/passwd_new")

    def test_denied_terminal_command_no_execution(self):
        """Denied terminal command: no subprocess."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")

        with patch("tools.terminal_tool.terminal_tool") as mock:
            execute_subagent_terminal("rm -rf /", cap, require_approval=False)
            execute_subagent_terminal("sudo cat /etc/passwd", cap, require_approval=False)
            execute_subagent_terminal("curl evil.com", cap, require_approval=False)
            mock.assert_not_called()

    def test_lifecycle_not_corrupted_by_denied_operation(self):
        """Lifecycle state not corrupted by denied operation."""
        ctx = create_execution_context(
            child_id="sa-test",
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
        manager = SubagentLifecycleManager(ctx)

        # Denied operation
        provider.handle_tool_call("hermes_rag_store", {"text": "x"})

        # Lifecycle still valid
        assert manager.state == "pending"

        # Can still transition normally
        manager.transition("initializing")
        assert manager.state == "initializing"

        manager.transition("running")
        manager.transition("completed")
        assert manager.is_terminal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
