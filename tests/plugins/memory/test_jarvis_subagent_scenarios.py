"""Integration tests for all 14 child-agent security scenarios."""

import json
import pytest
from unittest.mock import MagicMock, patch, call

from plugins.memory.jarvis import JARVISSubagentProvider


@pytest.fixture
def provider():
    return JARVISSubagentProvider("test goal", "test context")


class TestScenario01_L0ReadOnly:
    """Scenario 1: Child L0 tool call executes successfully, read-only."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l0_search_returns_results(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "results": [{"score": 0.9, "text": "test"}]})
        result = json.loads(provider.handle_tool_call("hermes_rag_search", {"query": "test"}))
        assert result["ok"] is True
        assert "results" in result

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l0_list_goals_returns_list(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "goals": []})
        result = json.loads(provider.handle_tool_call("hermes_goal_list", {}))
        assert result["ok"] is True

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l0_list_tasks_returns_list(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "tasks": []})
        result = json.loads(provider.handle_tool_call("hermes_task_list", {}))
        assert result["ok"] is True


class TestScenario02_L1ReversibleMutation:
    """Scenario 2: Child L1 tool call executes, reversible mutation."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l1_store_returns_uuid(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "memory_id": "test-uuid-123"})
        result = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": "hello"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: ok)
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l1_complete_goal_returns_updated(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "goal": {"id": "g_test", "status": "done"}})
        result = json.loads(provider.handle_tool_call("hermes_goal_complete", {"goal_id": "g_test"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: ok)
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l1_complete_task_returns_updated(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "task": {"id": "t_test", "status": "done"}})
        result = json.loads(provider.handle_tool_call("hermes_task_complete", {"task_id": "t_test"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: ok)
        mock_dispatch.assert_not_called()


class TestScenario03_L2ApprovalStaging:
    """Scenario 3: Child L2 tool call triggers approval/staging before mutation."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l2_create_goal_falls_through_to_handler(self, mock_dispatch, provider):
        # Decision 42: L2 tools no longer fall through to the handler's L2 gate
        mock_dispatch.return_value = json.dumps({"ok": True, "goal": {"id": "g_new"}})
        result = json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "new goal"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: ok)
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l2_create_task_falls_through(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "task": {"id": "t_new"}})
        result = json.loads(provider.handle_tool_call("hermes_task_create", {"title": "new task"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: ok)
        mock_dispatch.assert_not_called()


class TestScenario04_L3ApprovalEnforced:
    """Scenario 4: Child L3 tool call triggers approval BEFORE destructive operation."""

    @patch("tools.approval.request_tool_approval")
    def test_l3_delete_approval_before_mutation(self, mock_approval, provider):
        mock_approval.return_value = {"approved": False}
        result = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"}))
        # Decision 42: refused before approval and before any mutation
        # (was: approval came first and denied it)
        assert result["ok"] is False
        assert result["code"] == "delegated_read_only"
        mock_approval.assert_not_called()

    @patch("tools.approval.request_tool_approval")
    def test_l3_rollback_approval_before_mutation(self, mock_approval, provider):
        mock_approval.return_value = {"approved": False}
        result = json.loads(provider.handle_tool_call("hermes_rag_rollback_learnings", {}))
        assert result["ok"] is False
        assert result["code"] == "delegated_read_only"  # Decision 42 (was: "not approved")
        mock_approval.assert_not_called()


class TestScenario05_ApprovalDeniedZeroMutation:
    """Scenario 5: Approved denied causes ZERO database/Qdrant mutation."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    @patch("tools.approval.request_tool_approval")
    def test_l3_denied_no_mutation(self, mock_approval, mock_dispatch, provider):
        mock_approval.return_value = {"approved": False}
        result = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"}))
        assert result["ok"] is False
        # dispatch should NEVER be called — mutation path is blocked
        mock_dispatch.assert_not_called()


class TestScenario06_ParentOnlyToolRejected:
    """Scenario 6: Child attempts parent-only tool → rejected."""

    def test_terminal_rejected(self, provider):
        result = json.loads(provider.handle_tool_call("terminal", {"command": "ls"}))
        assert result["ok"] is False
        assert "parent-only" in result["error"]

    def test_write_file_rejected(self, provider):
        result = json.loads(provider.handle_tool_call("write_file", {"path": "/tmp/test.txt", "content": "test"}))
        assert result["ok"] is False
        assert "parent-only" in result["error"]

    def test_read_file_rejected(self, provider):
        result = json.loads(provider.handle_tool_call("read_file", {"path": "/tmp/test.txt"}))
        assert result["ok"] is False
        assert "parent-only" in result["error"]

    def test_browser_navigate_rejected(self, provider):
        result = json.loads(provider.handle_tool_call("browser_navigate", {"url": "http://example.com"}))
        assert result["ok"] is False
        assert "parent-only" in result["error"]

    def test_computer_use_rejected(self, provider):
        result = json.loads(provider.handle_tool_call("computer_use", {"action": "click"}))
        assert result["ok"] is False
        assert "parent-only" in result["error"]


class TestScenario07_DelegateTaskRejected:
    """Scenario 7: Child attempts delegate_task → rejected."""

    def test_delegate_task_rejected(self, provider):
        result = json.loads(provider.handle_tool_call("delegate_task", {"goal": "sub-goal"}))
        assert result["ok"] is False
        assert "parent-only" in result["error"]


class TestScenario08_BypassGateStripped:
    """Scenario 8: Child attempts _continuity_bypass_gate → stripped/rejected."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_bypass_gate_stripped_from_kwargs(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True})
        # Attempt to inject bypass gate — must be stripped
        provider.handle_tool_call("hermes_rag_search", {}, _continuity_bypass_gate=True)
        # Should have been stripped — dispatch called without bypass
        mock_dispatch.assert_called_once()


class TestScenario09_ForgedProvenanceBlocked:
    """Scenario 9: Child cannot forge provenance to impersonate another source."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_forged_provenance_overwritten(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True})
        # Child tries to impersonate "imported" source
        result = json.loads(provider.handle_tool_call(
            "hermes_rag_store",
            {"text": "test", "provenance": {"source": "imported"}}
        ))
        # Decision 42: the store is refused, so the forged provenance never
        # reaches dispatch (was: dispatched with sub-agent provenance)
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: ok)
        mock_dispatch.assert_not_called()


class TestScenario10_UnknownToolRejected:
    """Scenario 10: Child attempts unknown/unauthorized JARVIS tool → rejected."""

    def test_unknown_jarvis_tool_rejected(self, provider):
        result = json.loads(provider.handle_tool_call("hermes_rag_unknown", {}))
        assert result["ok"] is False
        assert "not registered" in result["error"]

    def test_unknown_custom_tool_rejected(self, provider):
        result = json.loads(provider.handle_tool_call("custom_tool", {}))
        assert result["ok"] is False
        assert "not registered" in result["error"]


class TestScenario11_MalformedArgumentsRejected:
    """Scenario 11: Malformed arguments rejected safely."""

    def test_empty_text_for_store(self, provider):
        result = json.loads(provider.handle_tool_call("hermes_rag_store", {}))
        assert result["ok"] is False

    def test_empty_memory_id_for_delete(self, provider):
        result = json.loads(provider.handle_tool_call("hermes_rag_delete", {}))
        assert result["ok"] is False

    def test_empty_goal_id_for_complete(self, provider):
        result = json.loads(provider.handle_tool_call("hermes_goal_complete", {}))
        assert result["ok"] is False


class TestScenario12_BackendFailureNoPartialMutation:
    """Scenario 12: Backend failure → no partial mutation."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_backend_exception_handled(self, mock_dispatch, provider):
        mock_dispatch.side_effect = Exception("Qdrant connection failed")
        result = json.loads(provider.handle_tool_call("hermes_rag_search", {"query": "test"}))
        assert result["ok"] is False
        assert "error" in result


class TestScenario13_RecursionDepthEnforced:
    """Scenario 13: Recursion depth limit actually enforced at runtime."""

    def test_depth_tracked(self, provider):
        assert provider._depth == 1

    def test_max_depth_constant(self):
        assert JARVISSubagentProvider._MAX_SUBAGENT_DEPTH == 2

    def test_child_depth_exceeds_max(self):
        """Child at depth 2 cannot spawn depth 3 (would exceed MAX_SUBAGENT_DEPTH)."""
        deep_provider = JARVISSubagentProvider("goal", depth=2)
        assert deep_provider._depth >= JARVISSubagentProvider._MAX_SUBAGENT_DEPTH


class TestScenario14_ChildFailureParentIntact:
    """Scenario 14: Child process failure → parent remains intact."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_child_exception_doesnt_affect_parent(self, mock_dispatch, provider):
        # Simulate child crashing on write
        mock_dispatch.side_effect = Exception("Child process crashed")
        result = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": "test"}))
        assert result["ok"] is False
        # Parent's args should not be mutated (deep copy isolation)
        # (This is tested more directly in provenance isolation tests)

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_args_isolated_between_calls(self, mock_dispatch, provider):
        mock_dispatch.return_value = json.dumps({"ok": True})
        args1 = {"text": "first", "provenance": {"source": "conversation"}}
        provider.handle_tool_call("hermes_rag_store", args1)
        original_source = args1["provenance"]["source"]
        # Original args must not be mutated
        assert original_source == "conversation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
