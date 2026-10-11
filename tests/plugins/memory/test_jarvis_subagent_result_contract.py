"""Tests for Phase 23D — Sub-Agent Result Contract & Parent Integration."""

import json
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
    SubagentTaskType,
)
from tools.delegate_tool import (
    register_lifecycle_manager,
    unregister_lifecycle_manager,
    transition_lifecycle,
    _get_lifecycle_state_for_subagent,
    _build_result_contract,
    RESULT_STATUSES,
)


class TestResultContractSchema:
    """Test the structured result contract."""

    def test_build_result_contract_completed(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="Research findings: honey production increased",
            subagent_id="sa-test-123",
            child_session_id="child-session-456",
            task_goal="research honey production",
            lifecycle_state="completed",
            api_calls=5,
            duration_seconds=12.3,
            model="test-model",
            exit_reason="completed",
            cost_usd=0.001,
            cost_status="estimated",
        )
        assert contract["status"] == "completed"
        assert contract["subagent_id"] == "sa-test-123"
        assert contract["child_session_id"] == "child-session-456"
        assert contract["task_goal"] == "research honey production"
        assert contract["lifecycle_state"] == "completed"
        assert contract["api_calls"] == 5
        assert contract["duration_seconds"] == 12.3

    def test_build_result_contract_failed(self):
        contract = _build_result_contract(
            task_index=0,
            status="failed",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="failing task",
            lifecycle_state="failed",
            error="Connection timeout",
        )
        assert contract["status"] == "failed"
        assert contract["error"] == "Connection timeout"
        assert contract["summary"] is None

    def test_build_result_contract_cancelled(self):
        contract = _build_result_contract(
            task_index=0,
            status="cancelled",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="long task",
            lifecycle_state="cancelled",
        )
        assert contract["status"] == "cancelled"
        assert contract["lifecycle_state"] == "cancelled"

    def test_build_result_contract_timed_out(self):
        contract = _build_result_contract(
            task_index=0,
            status="timed_out",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="slow task",
            lifecycle_state="timed_out",
            duration_seconds=60.0,
        )
        assert contract["status"] == "timed_out"
        assert contract["lifecycle_state"] == "timed_out"

    def test_result_statuses_constant(self):
        expected = {"completed", "failed", "cancelled", "timed_out", "error", "interrupted"}
        assert RESULT_STATUSES == expected


class TestResultLifecycleAgreement:
    """Test that lifecycle state and result status agree."""

    def test_completed_lifecycle_matches_completed_result(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")

        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="done",
            subagent_id="sa-test",
            child_session_id="child",
            task_goal="test",
            lifecycle_state=manager.state,
        )
        assert contract["lifecycle_state"] == "completed"
        assert contract["status"] == "completed"

    def test_failed_lifecycle_matches_failed_result(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")

        contract = _build_result_contract(
            task_index=0,
            status="failed",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child",
            task_goal="test",
            lifecycle_state=manager.state,
            error="failed",
        )
        assert contract["lifecycle_state"] == "failed"
        assert contract["status"] == "failed"

    def test_timed_out_lifecycle_matches_timed_out_result(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("timed_out")

        contract = _build_result_contract(
            task_index=0,
            status="timed_out",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child",
            task_goal="test",
            lifecycle_state=manager.state,
        )
        assert contract["lifecycle_state"] == "timed_out"
        assert contract["status"] == "timed_out"


class TestGetLifecycleStateForSubagent:
    """Test the lifecycle state lookup helper."""

    def test_returns_state_when_registered(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        register_lifecycle_manager("sa-test", manager)

        assert _get_lifecycle_state_for_subagent("sa-test") == "initializing"

        unregister_lifecycle_manager("sa-test")

    def test_returns_none_when_not_registered(self):
        assert _get_lifecycle_state_for_subagent("unknown") is None

    def test_returns_none_for_none_id(self):
        assert _get_lifecycle_state_for_subagent(None) is None


class TestResultContainsChildIdentity:
    """Test that result preserves sub-agent identity/context."""

    def test_result_has_subagent_id(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="done",
            subagent_id="sa-unique-id",
            child_session_id="child-session",
            task_goal="test",
            lifecycle_state="completed",
        )
        assert contract["subagent_id"] == "sa-unique-id"

    def test_result_has_child_session_id(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="done",
            subagent_id="sa-test",
            child_session_id="child-session-123",
            task_goal="test",
            lifecycle_state="completed",
        )
        assert contract["child_session_id"] == "child-session-123"

    def test_result_has_task_goal(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="done",
            subagent_id="sa-test",
            child_session_id="child",
            task_goal="research honey production",
            lifecycle_state="completed",
        )
        assert contract["task_goal"] == "research honey production"


class TestChildOutputNotCanonicalMemory:
    """Test that child output does not automatically become canonical."""

    def test_child_output_not_in_rag_without_explicit_store(self):
        """Child findings must not automatically persist to RAG."""
        # This tests that the result contract does not trigger RAG writes
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

        # Research task cannot store to RAG (no hermes_rag_store access)
        # So child output cannot become canonical without explicit parent action
        result = provider.handle_tool_call("hermes_rag_store", {"text": "findings"})
        import json
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

    def test_child_findings_not_in_continuity(self):
        """Child findings must not automatically modify parent continuity."""
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

        # Research cannot create goals/tasks
        result = provider.handle_tool_call("hermes_goal_create", {"title": "new goal"})
        import json
        result_dict = json.loads(result)
        assert result_dict["ok"] is False


class TestChildFailureLeavesParentIntact:
    """Test that child failure doesn't corrupt parent context."""

    def test_child_failure_preserves_parent_fields(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent-session",
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
        manager.transition("initializing")
        manager.transition("running")

        # Simulate child failing on unauthorized tool
        result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
        import json
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

        # Transition to failed
        manager.transition("failed")

        # Parent context fields preserved
        assert ctx.parent_session_id == "parent-session"
        assert ctx.task_type == "research"
        assert ctx.authority_level == "L0"
        assert ctx.allowed_tools == L0_READ_ONLY
        assert ctx.provenance_context_id == "subagent:research"


class TestExistingAuthorizationIntact:
    """Test that existing Phase 20–22 authorization remains intact."""

    def test_research_cannot_write(self):
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
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        import json
        assert json.loads(result)["ok"] is False

    def test_note_taking_cannot_write(self):
        ctx = create_execution_context(
            child_id="sa-test",
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
        # Decision 42: note-taking can no longer store (was: L1-allowed)
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True, "memory_id": "uuid"})
            result = provider.handle_tool_call("hermes_rag_store", {"text": "note"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False and result_dict["code"] == "delegated_read_only"
            mock_dispatch.assert_not_called()

    def test_maintenance_requires_approval_for_delete(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="maintenance",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="maintenance",
            execution_context=ctx,
        )
        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": False}
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False


class TestProvenanceIsolation:
    """Test that provenance remains isolated."""

    def test_child_provenance_contains_subagent_goal(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research honey",
            task_type="research",
            delegation_depth=1,
        )
        assert "subagent:research" in ctx.provenance_context_id

    def test_child_provenance_does_not_contain_parent_session(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent-session-123",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        assert "parent-session" not in ctx.provenance_context_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
