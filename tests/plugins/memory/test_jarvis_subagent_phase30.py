"""Phase 30 — Authorization Boundary Hardening Tests.

Tests for JARVIS sub-agent single-authority enforcement:
- Capability registry is authoritative
- Parent-only tools are denied
- L3 approval flows through capability registry
- Terminal policy is enforced
- Filesystem scope is enforced
- Provenance is isolated
- Authorization is independent of observability failures
"""

import json
import pytest
from unittest.mock import MagicMock, patch

import plugins.memory.jarvis.__init__ as jarvis_init
from plugins.memory.jarvis.__init__ import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    create_execution_context,
)
from plugins.memory.jarvis.subagent_terminal import (
    SUBAGENT_DENIED_COMMANDS,
)
from plugins.memory.jarvis.subagent_filesystem import (
    DEFAULT_SUBAGENT_ROOT,
)
from plugins.memory.jarvis.tool_capability_registry import (
    is_tool_permitted,
    requires_approval,
    is_tool_parent_only,
    get_capability,
)
from plugins.memory.jarvis.subagent_security import (
    get_recorder,
    reset_recorder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_recorder_fixture():
    """Reset recorder before each test."""
    reset_recorder()
    yield
    reset_recorder()


def _make_provider(task_type="research", depth=1):
    """Create a JARVISSubagentProvider with a frozen execution context."""
    ctx = create_execution_context(
        child_id="sa-test-001",
        parent_session_id="parent-session-001",
        task_goal="Test research goal",
        task_type=task_type,
        delegation_depth=depth,
    )
    provider = JARVISSubagentProvider(
        goal="Test research goal",
        context="test context",
        depth=depth,
        task_type=task_type,
        execution_context=ctx,
    )
    return provider


# ====================================================================
# 1. Direct Dispatch Bypass Attempts
# ====================================================================

class TestDispatchBypassAttempts:
    """Test that no tool can bypass authorization by direct dispatch."""

    def test_unknown_tool_rejected(self):
        """Unknown tool names are rejected before dispatch."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call("unknown_tool", {})
        data = json.loads(result)
        assert data["ok"] is False

    def test_malformed_tool_name_rejected(self):
        """Malformed tool names are rejected."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call("'; DROP TABLE--", {})
        data = json.loads(result)
        assert data["ok"] is False

    def test_tool_with_special_chars_rejected(self):
        """Tool names with special characters are rejected."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("hermes_rag_search; rm -rf /", {})
        data = json.loads(result)
        assert data["ok"] is False


# ====================================================================
# 2. Unauthorized Task/Tool Combinations
# ====================================================================

class TestUnauthorizedTaskToolCombinations:
    """Test that task-type authorization is enforced."""

    def test_research_cannot_write_rag(self):
        """Research task type cannot write to RAG."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        data = json.loads(result)
        assert data["ok"] is False

    def test_research_cannot_create_goal(self):
        """Research task type cannot create goals."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call("hermes_goal_create", {"title": "test"})
        data = json.loads(result)
        assert data["ok"] is False

    def test_research_cannot_supersede_decision(self):
        """Research task type cannot supersede decisions."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call("hermes_decision_supersede", {"id": "d_12345678", "decision": "test"})
        data = json.loads(result)
        assert data["ok"] is False

    def test_summarize_cannot_write(self):
        """Summarize task type cannot write."""
        provider = _make_provider(task_type="summarize")
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        data = json.loads(result)
        assert data["ok"] is False

    def test_note_taking_can_search_rag(self):
        """Note-taking task type CAN search RAG."""
        provider = _make_provider(task_type="note_taking")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            data = json.loads(result)
            assert data["ok"] is True

    def test_research_cannot_execute_terminal(self):
        """Research task type CANNOT execute terminal."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call("terminal", {"command": "ls"})
        data = json.loads(result)
        assert data["ok"] is False


# ====================================================================
# 3. Forged Capability Names
# ====================================================================

class TestForgedCapabilityNames:
    """Test that forged capability names are detected."""

    def test_forged_tool_not_in_registry(self):
        """A tool name not in the capability registry is forged."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("forged_tool", {})
        data = json.loads(result)
        assert data["ok"] is False

    def test_forged_parent_only_tool_as_subagent(self):
        """A parent-only tool called by sub-agent is detected."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("delegate_task", {})
        data = json.loads(result)
        assert data["ok"] is False
        assert "parent-only" in data["error"]

    def test_forged_tool_similar_to_real(self):
        """A tool name similar to a real tool but not exact is forged."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("hermes_rag_store!", {})
        data = json.loads(result)
        assert data["ok"] is False


# ====================================================================
# 4. L2 Tool Misuse
# ====================================================================

class TestL2ToolMisuse:
    """Test that L2 tools are properly gated."""

    def test_l2_tool_from_low_task_rejected(self):
        """L2 tool from L0 task type is rejected."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call("hermes_goal_create", {"title": "test"})
        data = json.loads(result)
        assert data["ok"] is False

    def test_l2_tool_from_matching_task_refused(self):
        """Decision 42: an L2 write is refused even for an L2 task type (was: allowed)."""
        provider = _make_provider(task_type="task_execution")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True})
            result = provider.handle_tool_call("hermes_goal_create", {"title": "test"})
            data = json.loads(result)
            assert data["ok"] is False and data["code"] == "delegated_read_only"
            mock_dispatch.assert_not_called()


# ====================================================================
# 5. L3 Approval Bypass
# ====================================================================

class TestL3ApprovalBypass:
    """Test that L3 approval cannot be bypassed."""

    def test_l3_tool_requires_approval(self):
        """L3 tool requires explicit approval."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test_123"})
        data = json.loads(result)
        assert data["ok"] is False or "approved" in data.get("error", "")

    def test_l3_tool_refused_even_with_approval(self):
        """Decision 42: approval cannot authorize a delegated L3 write (was: succeeded)."""
        provider = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": True}) as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock_dispatch.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test_123"})
                data = json.loads(result)
                assert data["ok"] is False and data["code"] == "delegated_read_only"
                mock_approval.assert_not_called()
                mock_dispatch.assert_not_called()

    def test_l3_approval_denied(self):
        """L3 tool is rejected; Decision 42 refuses it before approval (was: "not approved")."""
        provider = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": False}) as mock_approval:
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test_123"})
            data = json.loads(result)
            assert data["ok"] is False
            assert data["code"] == "delegated_read_only"
            mock_approval.assert_not_called()

    def test_l3_approval_exception_denied(self):
        """L3 tool with approval exception is denied."""
        provider = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", side_effect=Exception("Approval failed")):
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test_123"})
            data = json.loads(result)
            assert data["ok"] is False


# ====================================================================
# 6. Parent Approval Replay
# ====================================================================

class TestParentApprovalReplay:
    """Test that parent approval cannot be replayed by child."""

    def test_bypass_gate_stripped(self):
        """_continuity_bypass_gate is stripped from kwargs."""
        provider = _make_provider(task_type="research")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
            result = provider.handle_tool_call(
                "hermes_rag_search",
                {"query": "test"},
                _continuity_bypass_gate=True,
            )
            data = json.loads(result)
            assert data["ok"] is True

    def test_parent_only_tool_denied_even_with_bypass(self):
        """Parent-only tools are denied even with bypass gate."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call(
            "delegate_task",
            {"goal": "test"},
            _continuity_bypass_gate=True,
        )
        data = json.loads(result)
        assert data["ok"] is False
        assert "parent-only" in data["error"]


# ====================================================================
# 7. Terminal Policy Bypass
# ====================================================================

class TestTerminalPolicyBypass:
    """Test that terminal policy cannot be bypassed."""

    def test_denied_command_rejected(self):
        """Denied commands are rejected."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("terminal", {"command": "rm -rf /"})
        data = json.loads(result)
        assert data["ok"] is False

    def test_allowed_command_denied_at_registry(self):
        """Allowed commands are denied at capability registry (parent-only)."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("terminal", {"command": "ls"})
        data = json.loads(result)
        assert data["ok"] is False

    def test_non_maintenance_cannot_use_terminal(self):
        """Non-maintenance task types cannot use terminal."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call("terminal", {"command": "ls"})
        data = json.loads(result)
        assert data["ok"] is False

    def test_shell_metacharacter_rejected_by_registry(self):
        """Shell metacharacters are rejected (terminal is parent-only)."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("terminal", {"command": "ls; rm -rf /"})
        data = json.loads(result)
        assert data["ok"] is False


# ====================================================================
# 8. Filesystem Scope Bypass
# ====================================================================

class TestFilesystemScopeBypass:
    """Test that filesystem scope cannot be bypassed."""

    def test_write_outside_scope_rejected(self):
        """Write outside sub-agent scope is rejected."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call(
            "write_file",
            {"path": "/etc/passwd", "content": "test"},
        )
        data = json.loads(result)
        assert data["ok"] is False

    def test_read_outside_scope_rejected(self):
        """Read outside sub-agent scope is rejected."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call(
            "read_file",
            {"path": "/etc/shadow"},
        )
        data = json.loads(result)
        assert data["ok"] is False

    def test_write_research_task_rejected(self):
        """Write from research task type is rejected."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call(
            "write_file",
            {"path": "/tmp/test.txt", "content": "test"},
        )
        data = json.loads(result)
        assert data["ok"] is False


# ====================================================================
# 9. Provenance Spoofing
# ====================================================================

class TestProvenanceSpoofing:
    """Test that provenance cannot be spoofed."""

    def test_provenance_forced_to_subagent(self):
        """Decision 42: a store with spoofed provenance is refused outright
        (was: dispatched with provenance forced to sub-agent origin)."""
        provider = _make_provider(task_type="note_taking")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True, "memory_id": "test"})
            result = provider.handle_tool_call(
                "hermes_rag_store",
                {"text": "test", "provenance": {"source": "parent", "context_id": "parent:123"}},
            )
            assert json.loads(result)["code"] == "delegated_read_only"
            mock_dispatch.assert_not_called()

    def test_args_deep_copied(self):
        """Args are deep-copied to prevent parent mutation."""
        provider = _make_provider(task_type="note_taking")
        original_args = {"text": "test", "nested": {"key": "value"}}
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True})
            provider.handle_tool_call("hermes_rag_store", original_args)
            assert "provenance" not in original_args
            assert original_args["text"] == "test"

    def test_parent_provenance_cannot_be_inherited(self):
        """Parent provenance cannot be inherited by child."""
        provider = _make_provider(task_type="note_taking")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True, "memory_id": "test"})
            result = provider.handle_tool_call(
                "hermes_rag_store",
                {"text": "test", "provenance": {"source": "user_confirmed"}},
            )
            data = json.loads(result)
            # Decision 42: refused, so no parent provenance can be used (was: ok, rewritten)
            assert data["ok"] is False and data["code"] == "delegated_read_only"
            mock_dispatch.assert_not_called()


# ====================================================================
# 10. Recorder/Audit Failure Isolation
# ====================================================================

class TestRecorderFailureIsolation:
    """Test that recorder/audit failures don't affect authorization."""

    def test_recorder_failure_doesnt_block_allow(self):
        """Recorder failure doesn't block ALLOW decisions."""
        provider = _make_provider(task_type="research")
        with patch("plugins.memory.jarvis.subagent_security.record_authorization_allowed") as mock_record:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock_record.side_effect = Exception("Recorder failed")
                mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                data = json.loads(result)
                assert data["ok"] is True

    def test_recorder_failure_doesnt_block_deny(self):
        """Recorder failure doesn't block DENY decisions."""
        provider = _make_provider(task_type="research")
        with patch("plugins.memory.jarvis.subagent_security.record_authorization_denied") as mock_record:
            mock_record.side_effect = Exception("Recorder failed")
            result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
            data = json.loads(result)
            assert data["ok"] is False

    def test_audit_store_failure_doesnt_affect_auth(self):
        """Audit store failure doesn't affect authorization."""
        provider = _make_provider(task_type="research")
        recorder = get_recorder()
        with patch.object(recorder, "record") as mock_audit:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock_audit.side_effect = Exception("Audit store failed")
                mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                data = json.loads(result)
                assert data["ok"] is True

    def test_both_recorder_and_audit_fail(self):
        """Both recorder and audit store failing doesn't affect auth."""
        provider = _make_provider(task_type="research")
        recorder = get_recorder()
        with patch("plugins.memory.jarvis.subagent_security.record_authorization_allowed") as mock_record:
            with patch.object(recorder, "record") as mock_audit:
                with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                    mock_record.side_effect = Exception("Recorder failed")
                    mock_audit.side_effect = Exception("Audit failed")
                    mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
                    result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                    data = json.loads(result)
                    assert data["ok"] is True


# ====================================================================
# 11. Capability Registry Is Authoritative
# ====================================================================

class TestCapabilityRegistryAuthoritative:
    """Test that capability registry is the authoritative source."""

    def test_tool_in_registry_and_task_allowed(self):
        """Tool in registry and task type is allowed."""
        provider = _make_provider(task_type="research")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            data = json.loads(result)
            assert data["ok"] is True

    def test_tool_in_registry_but_not_task(self):
        """Tool in registry but not for task type is denied."""
        provider = _make_provider(task_type="research")
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        data = json.loads(result)
        assert data["ok"] is False

    def test_tool_not_in_registry(self):
        """Tool not in registry is denied."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("nonexistent_tool", {})
        data = json.loads(result)
        assert data["ok"] is False

    def test_parent_only_tool_denied_by_registry(self):
        """Parent-only tool is denied by registry."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("delegate_task", {})
        data = json.loads(result)
        assert data["ok"] is False
        assert "parent-only" in data["error"]

    def test_requires_approval_from_registry(self):
        """Approval requirement comes from registry."""
        provider = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"})
        data = json.loads(result)
        assert "not authorized" not in data.get("error", "")


# ====================================================================
# 12. Lifecycle State in Events
# ====================================================================

class TestLifecycleStateInEvents:
    """Test that lifecycle state is correctly reported in events."""

    def test_lifecycle_state_from_context(self):
        """Lifecycle state comes from execution context."""
        provider = _make_provider(task_type="research")
        assert provider._execution_context.lifecycle_state == "pending"

    def test_lifecycle_state_empty_when_no_context(self):
        """Lifecycle state is empty when no context."""
        provider = JARVISSubagentProvider(goal="test", depth=1, task_type="research")
        assert provider._execution_context is None


# ====================================================================
# 13. Sub-Agent Identity Validation
# ====================================================================

class TestSubAgentIdentityValidation:
    """Test that sub-agent identity is validated."""

    def test_child_id_from_context(self):
        """Child ID comes from execution context."""
        provider = _make_provider(task_type="research")
        assert provider._execution_context.child_id == "sa-test-001"

    def test_parent_session_id_from_context(self):
        """Parent session ID comes from execution context."""
        provider = _make_provider(task_type="research")
        assert provider._execution_context.parent_session_id == "parent-session-001"

    def test_task_type_matches_context(self):
        """Task type matches execution context."""
        provider = _make_provider(task_type="research")
        assert provider._execution_context.task_type == "research"


# ====================================================================
# 14. Default Task Type Behavior
# ====================================================================

class TestDefaultTaskType:
    """Test default task type behavior."""

    def test_default_task_type_allows_all(self):
        """Default task type allows all tools."""
        provider = _make_provider(task_type="default")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            data = json.loads(result)
            assert data["ok"] is True


# ====================================================================
# 15. ALLOW/DENY Independence from Observability
# ====================================================================

class TestAllowDenyIndependence:
    """Test that ALLOW/DENY is independent of observability failures."""

    def test_allow_with_recorder_failure(self):
        """ALLOW decision is unchanged when recorder fails."""
        provider = _make_provider(task_type="research")

        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
            result1 = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            data1 = json.loads(result1)

        with patch("plugins.memory.jarvis.subagent_security.record_authorization_allowed") as mock:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock.side_effect = Exception("fail")
                mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
                result2 = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                data2 = json.loads(result2)

        assert data1["ok"] == data2["ok"]

    def test_deny_with_recorder_failure(self):
        """DENY decision is unchanged when recorder fails."""
        provider = _make_provider(task_type="research")

        result1 = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        data1 = json.loads(result1)

        with patch("plugins.memory.jarvis.subagent_security.record_authorization_denied") as mock:
            mock.side_effect = Exception("fail")
            result2 = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
            data2 = json.loads(result2)

        assert data1["ok"] == data2["ok"]

    def test_allow_with_all_observability_failed(self):
        """ALLOW works when ALL observability fails."""
        provider = _make_provider(task_type="research")

        with patch("plugins.memory.jarvis.subagent_security.record_authorization_allowed") as mock_allowed:
            with patch("plugins.memory.jarvis.subagent_security.record_tool_execution_completed") as mock_exec:
                with patch.object(get_recorder(), "record") as mock_audit:
                    with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                        mock_allowed.side_effect = Exception("fail")
                        mock_exec.side_effect = Exception("fail")
                        mock_audit.side_effect = Exception("fail")
                        mock_dispatch.return_value = json.dumps({"ok": True, "results": []})

                        result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                        data = json.loads(result)
                        assert data["ok"] is True

    def test_deny_with_all_observability_failed(self):
        """DENY works when ALL observability fails."""
        provider = _make_provider(task_type="research")

        with patch("plugins.memory.jarvis.subagent_security.record_authorization_denied") as mock_denied:
            with patch("plugins.memory.jarvis.subagent_security.record_forged_capability") as mock_forged:
                with patch.object(get_recorder(), "record") as mock_audit:
                    mock_denied.side_effect = Exception("fail")
                    mock_forged.side_effect = Exception("fail")
                    mock_audit.side_effect = Exception("fail")

                    result = provider.handle_tool_call("forged_tool", {})
                    data = json.loads(result)
                    assert data["ok"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
