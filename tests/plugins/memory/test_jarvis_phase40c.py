"""Phase 40C — Contract Unification tests."""

import pytest
import json

from plugins.memory.jarvis.capabilities.execution_result import (
    CapabilityExecutionResult,
    CapabilityExecutionStatus,
)
from tools.delegate_tool import _build_result_contract


class TestExecutedProperty:
    def test_success_executed(self):
        result = CapabilityExecutionResult(status=CapabilityExecutionStatus.SUCCESS, task_count=1, success_count=1)
        assert result.executed is True

    def test_partial_executed(self):
        result = CapabilityExecutionResult(status=CapabilityExecutionStatus.PARTIAL, task_count=2, success_count=1, failure_count=1)
        assert result.executed is True

    def test_failure_executed(self):
        result = CapabilityExecutionResult(status=CapabilityExecutionStatus.FAILURE, task_count=1, failure_count=1)
        assert result.executed is True

    def test_skipped_not_executed(self):
        result = CapabilityExecutionResult(status=CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED)
        assert result.executed is False

    def test_not_executed(self):
        result = CapabilityExecutionResult(status=CapabilityExecutionStatus.NOT_EXECUTED)
        assert result.executed is False


class TestBuildResultContractWithCapabilityExecution:
    def test_with_capability_execution(self):
        """capability_execution present when param given."""
        exec_result = {"status": "success", "task_count": 2, "success_count": 2}
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="All capabilities succeeded",
            subagent_id="sa-1",
            child_session_id="cs-1",
            task_goal="test goal",
            lifecycle_state="completed",
            capability_execution=exec_result,
        )
        assert contract["capability_execution"] == exec_result
        # All standard fields present
        assert contract["task_index"] == 0
        assert contract["status"] == "completed"
        assert contract["summary"] == "All capabilities succeeded"
        assert contract["subagent_id"] == "sa-1"
        assert contract["child_session_id"] == "cs-1"
        assert contract["task_goal"] == "test goal"
        assert contract["lifecycle_state"] == "completed"

    def test_without_capability_execution(self):
        """capability_execution absent when param None (backward compat)."""
        contract = _build_result_contract(
            task_index=1,
            status="completed",
            summary="Normal result",
            subagent_id="sa-2",
            child_session_id="cs-2",
            task_goal="test goal",
            lifecycle_state="completed",
        )
        assert "capability_execution" not in contract

    def test_existing_fields_preserved(self):
        """All existing fields present and correct."""
        contract = _build_result_contract(
            task_index=3,
            status="failed",
            summary="Something went wrong",
            subagent_id="sa-3",
            child_session_id="cs-3",
            task_goal="fail goal",
            lifecycle_state="failed",
            api_calls=5,
            duration_seconds=10.5,
            model="gpt-4",
            exit_reason="max_iterations",
            error="Too many iterations",
            tool_trace=[{"tool": "search"}],
            tokens={"input": 100, "output": 200},
            cost_usd=0.01,
            cost_status="estimated",
        )
        assert contract["task_index"] == 3
        assert contract["status"] == "failed"
        assert contract["summary"] == "Something went wrong"
        assert contract["subagent_id"] == "sa-3"
        assert contract["child_session_id"] == "cs-3"
        assert contract["task_goal"] == "fail goal"
        assert contract["lifecycle_state"] == "failed"
        assert contract["api_calls"] == 5
        assert contract["duration_seconds"] == 10.5
        assert contract["model"] == "gpt-4"
        assert contract["exit_reason"] == "max_iterations"
        assert contract["error"] == "Too many iterations"
        assert contract["tool_trace"] == [{"tool": "search"}]
        assert contract["tokens"] == {"input": 100, "output": 200}
        assert contract["cost_usd"] == 0.01
        assert contract["cost_status"] == "estimated"

    def test_capability_execution_not_exposed_raw_output(self):
        """capability_execution.to_dict() caps task summaries and truncates."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.SUCCESS,
            task_count=15,
            success_count=15,
            summary="x" * 1000,
        )
        d = result.to_dict()
        # Summary truncated to 500
        assert len(d["summary"]) <= 500
        # Tasks capped at 10
        assert len(d["tasks"]) <= 10

    def test_contract_json_serializable(self):
        """Contract is JSON-serializable (no non-serializable objects)."""
        exec_result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.SUCCESS,
            task_count=1,
            success_count=1,
            summary="test",
        ).to_dict()
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="test",
            subagent_id="sa-1",
            child_session_id="cs-1",
            task_goal="goal",
            lifecycle_state="completed",
            capability_execution=exec_result,
        )
        json_str = json.dumps(contract)
        assert json_str is not None
        # capability_execution is a plain dict
        loaded = json.loads(json_str)
        assert loaded["capability_execution"]["status"] == "success"


class TestUnificationSemantics:
    def test_all_succeeded_behavior(self):
        """all_succeeded is True only for SUCCESS status."""
        success = CapabilityExecutionResult(status=CapabilityExecutionStatus.SUCCESS)
        partial = CapabilityExecutionResult(status=CapabilityExecutionStatus.PARTIAL)
        failure = CapabilityExecutionResult(status=CapabilityExecutionStatus.FAILURE)
        skipped = CapabilityExecutionResult(status=CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED)
        not_exec = CapabilityExecutionResult(status=CapabilityExecutionStatus.NOT_EXECUTED)

        assert success.all_succeeded is True
        assert partial.all_succeeded is False
        assert failure.all_succeeded is False
        assert skipped.all_succeeded is False
        assert not_exec.all_succeeded is False

    def test_has_failures(self):
        success = CapabilityExecutionResult(status=CapabilityExecutionStatus.SUCCESS)
        partial = CapabilityExecutionResult(status=CapabilityExecutionStatus.PARTIAL, failure_count=1)
        failure = CapabilityExecutionResult(status=CapabilityExecutionStatus.FAILURE, failure_count=1)

        assert success.has_failures is False
        assert partial.has_failures is True
        assert failure.has_failures is True

    def test_approval_required(self):
        skipped = CapabilityExecutionResult(status=CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED)
        success = CapabilityExecutionResult(status=CapabilityExecutionStatus.SUCCESS)

        assert skipped.approval_required is True
        assert success.approval_required is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
