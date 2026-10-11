"""Tests for Phase 22B — Runtime Task-Type Enforcement."""

import json
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    get_task_type,
    is_tool_allowed_for_task,
)


@pytest.fixture
def research_provider():
    return JARVISSubagentProvider("research goal", task_type="research")


@pytest.fixture
def summarize_provider():
    return JARVISSubagentProvider("summarize goal", task_type="summarize")


@pytest.fixture
def note_taking_provider():
    return JARVISSubagentProvider("note goal", task_type="note_taking")


@pytest.fixture
def task_execution_provider():
    return JARVISSubagentProvider("execute goal", task_type="task_execution")


@pytest.fixture
def maintenance_provider():
    return JARVISSubagentProvider("maintenance goal", task_type="maintenance")


@pytest.fixture
def default_provider():
    """Provider without task_type — uses default whitelist."""
    return JARVISSubagentProvider("default goal")


class TestPhase22BResearchTaskRestrictions:
    """Scenario 13a: research (L0) cannot access L1/L2/L3 tools."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_research_can_search(self, mock_dispatch, research_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
        result = json.loads(research_provider.handle_tool_call("hermes_rag_search", {"query": "test"}))
        assert result["ok"] is True

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_research_cannot_store_l1(self, mock_dispatch, research_provider):
        """research attempting L1 tool — must be rejected."""
        result = json.loads(research_provider.handle_tool_call("hermes_rag_store", {"text": "test"}))
        assert result["ok"] is False
        assert "not authorized for task type" in result["error"]
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_research_cannot_create_goal_l2(self, mock_dispatch, research_provider):
        """research attempting L2 tool — must be rejected."""
        result = json.loads(research_provider.handle_tool_call("hermes_goal_create", {"title": "new"}))
        assert result["ok"] is False
        assert "not authorized for task type" in result["error"]
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_research_cannot_delete_l3(self, mock_dispatch, research_provider):
        """research attempting L3 tool — must be rejected."""
        result = json.loads(research_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"}))
        assert result["ok"] is False
        assert "not authorized for task type" in result["error"]
        mock_dispatch.assert_not_called()


class TestPhase22BSummarizeTaskRestrictions:
    """summarize (L0) same restrictions as research."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_summarize_can_search(self, mock_dispatch, summarize_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
        result = json.loads(summarize_provider.handle_tool_call("hermes_rag_search", {"query": "test"}))
        assert result["ok"] is True

    def test_summarize_cannot_store(self, summarize_provider):
        result = json.loads(summarize_provider.handle_tool_call("hermes_rag_store", {"text": "test"}))
        assert result["ok"] is False
        assert "not authorized" in result["error"]


class TestPhase22BNoteTakingTaskRestrictions:
    """note_taking (L1) can read + reversible writes, but not L2/L3."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_note_taking_can_search(self, mock_dispatch, note_taking_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "results": []})
        result = json.loads(note_taking_provider.handle_tool_call("hermes_rag_search", {"query": "test"}))
        assert result["ok"] is True

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_note_taking_cannot_store(self, mock_dispatch, note_taking_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "memory_id": "test-uuid"})
        result = json.loads(note_taking_provider.handle_tool_call("hermes_rag_store", {"text": "test"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: allowed)
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_note_taking_cannot_complete_goal(self, mock_dispatch, note_taking_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "goal": {"id": "g_test", "status": "done"}})
        result = json.loads(note_taking_provider.handle_tool_call("hermes_goal_complete", {"goal_id": "g_test"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: allowed)
        mock_dispatch.assert_not_called()

    def test_note_taking_cannot_create_goal_l2(self, note_taking_provider):
        """note_taking attempting L2 tool — must be rejected."""
        result = json.loads(note_taking_provider.handle_tool_call("hermes_goal_create", {"title": "new"}))
        assert result["ok"] is False
        assert "not authorized for task type" in result["error"]

    def test_note_taking_cannot_delete_l3(self, note_taking_provider):
        result = json.loads(note_taking_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"}))
        assert result["ok"] is False
        assert "not authorized" in result["error"]


class TestPhase22BTaskExecutionRestrictions:
    """task_execution (L2) can read, write, create — but not L3."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_task_execution_cannot_create_goal(self, mock_dispatch, task_execution_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "goal": {"id": "g_new"}})
        result = json.loads(task_execution_provider.handle_tool_call("hermes_goal_create", {"title": "new"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: allowed)
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_task_execution_cannot_create_task(self, mock_dispatch, task_execution_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "task": {"id": "t_new"}})
        result = json.loads(task_execution_provider.handle_tool_call("hermes_task_create", {"title": "new"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: allowed)
        mock_dispatch.assert_not_called()

    def test_task_execution_cannot_delete_l3(self, task_execution_provider):
        """task_execution attempting L3 tool — must be rejected."""
        result = json.loads(task_execution_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"}))
        assert result["ok"] is False
        assert "not authorized for task type" in result["error"]

    def test_task_execution_cannot_rollback_l3(self, task_execution_provider):
        result = json.loads(task_execution_provider.handle_tool_call("hermes_rag_rollback_learnings", {}))
        assert result["ok"] is False
        assert "not authorized" in result["error"]


class TestPhase22BMaintenanceTaskRestrictions:
    """maintenance (L3) can do everything — with approval for L3 ops."""

    @patch("tools.approval.request_tool_approval")
    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_maintenance_cannot_delete_even_with_approval(self, mock_dispatch, mock_approval, maintenance_provider):
        mock_approval.return_value = {"approved": True}
        mock_dispatch.return_value = json.dumps({"ok": True})
        result = json.loads(maintenance_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: allowed)
        mock_approval.assert_not_called()
        mock_dispatch.assert_not_called()

    @patch("tools.approval.request_tool_approval")
    def test_maintenance_denied_delete_no_mutation(self, mock_approval, maintenance_provider):
        mock_approval.return_value = {"approved": False}
        result = json.loads(maintenance_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"}))
        assert result["ok"] is False
        # Decision 42: refused before approval (was: "not approved")
        assert result["code"] == "delegated_read_only"
        mock_approval.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_maintenance_cannot_create_goal(self, mock_dispatch, maintenance_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "goal": {"id": "g_new"}})
        result = json.loads(maintenance_provider.handle_tool_call("hermes_goal_create", {"title": "new"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: allowed)
        mock_dispatch.assert_not_called()


class TestPhase22BForgedToolRejected:
    """Scenario: forged/unknown tool name."""

    def test_unknown_tool_rejected(self, research_provider):
        result = json.loads(research_provider.handle_tool_call("unknown_tool", {}))
        assert result["ok"] is False

    def test_parent_only_tool_terminal_rejected(self, research_provider):
        result = json.loads(research_provider.handle_tool_call("terminal", {"command": "ls"}))
        assert result["ok"] is False

    def test_forged_hermes_tool_rejected(self, research_provider):
        result = json.loads(research_provider.handle_tool_call("hermes_rag_fake", {}))
        assert result["ok"] is False


class TestPhase22BEscalationAttempts:
    """Scenario: task-type escalation attempt."""

    def test_research_emitting_l3_tool_name(self, research_provider):
        """Child with task_type='research' emits hermes_rag_delete — rejected."""
        result = json.loads(research_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"}))
        assert result["ok"] is False
        assert "not authorized" in result["error"]

    def test_note_taking_emitting_l3_tool_name(self, note_taking_provider):
        result = json.loads(note_taking_provider.handle_tool_call("hermes_rag_rollback_learnings", {}))
        assert result["ok"] is False
        assert "not authorized" in result["error"]


class TestPhase22BZeroMutationOnReject:
    """Scenario: unauthorized call produces zero mutation."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_research_store_no_mutation(self, mock_dispatch, research_provider):
        result = json.loads(research_provider.handle_tool_call("hermes_rag_store", {"text": "data"}))
        assert result["ok"] is False
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_research_delete_no_mutation(self, mock_dispatch, research_provider):
        result = json.loads(research_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"}))
        assert result["ok"] is False
        mock_dispatch.assert_not_called()


class TestPhase22BAuthorizedCallsWork:
    """Scenario: authorized calls continue working."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_research_search_works(self, mock_dispatch, research_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "results": [{"score": 0.9}]})
        result = json.loads(research_provider.handle_tool_call("hermes_rag_search", {"query": "test"}))
        assert result["ok"] is True

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_note_taking_store_refused(self, mock_dispatch, note_taking_provider):
        mock_dispatch.return_value = json.dumps({"ok": True, "memory_id": "uuid"})
        result = json.loads(note_taking_provider.handle_tool_call("hermes_rag_store", {"text": "note"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: allowed)
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_task_execution_create_refused(self, mock_dispatch, task_execution_provider):
        mock_dispatch.return_value = json.dumps({"ok": True})
        result = json.loads(task_execution_provider.handle_tool_call("hermes_task_create", {"title": "new"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: allowed)
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_default_provider_is_read_only(self, mock_dispatch, default_provider):
        """Default provider (no task_type) no longer has the full L0-L3
        whitelist: Decision 42 limits it to reads (was: store allowed)."""
        mock_dispatch.return_value = json.dumps({"ok": True, "memory_id": "uuid"})
        result = json.loads(default_provider.handle_tool_call("hermes_rag_store", {"text": "test"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"  # Decision 42 (was: allowed)
        mock_dispatch.assert_not_called()


class TestPhase22BBypassGateStillStripped:
    """Phase 22B preserves _continuity_bypass_gate stripping."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_bypass_gate_stripped(self, mock_dispatch, research_provider):
        mock_dispatch.return_value = json.dumps({"ok": True})
        research_provider.handle_tool_call("hermes_rag_search", {"query": "test"},
                                           _continuity_bypass_gate=True)
        mock_dispatch.assert_called_once()


class TestPhase22BProvenanceIsolationPreserved:
    """Phase 22B preserves provenance isolation."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_args_deepcopied(self, mock_dispatch, research_provider):
        mock_dispatch.return_value = json.dumps({"ok": True})
        original = {"query": "test", "extra": {"nested": "value"}}
        original_copy = json.loads(json.dumps(original))
        research_provider.handle_tool_call("hermes_rag_search", original)
        assert original == original_copy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
