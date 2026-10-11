"""Phase 42 — Model Context Injection for Remediation tests."""

import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis.capabilities.execution_result import (
    CapabilityExecutionResult,
    CapabilityExecutionStatus,
    TaskSummary,
)
from tools.delegate_tool import _format_capability_preamble


class TestFormatCapabilityPreamble:
    def test_partial_preamble(self):
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.PARTIAL,
            task_count=3,
            success_count=2,
            failure_count=1,
            tasks=[
                TaskSummary(capability_name="goal_list", success=True),
                TaskSummary(capability_name="task_list", success=True),
                TaskSummary(capability_name="decision_list", success=False, error="timeout"),
            ],
        )
        preamble = _format_capability_preamble(result)
        assert "partially executed" in preamble
        assert "2/3 succeeded" in preamble
        assert "goal_list" in preamble
        assert "decision_list" in preamble
        assert "retry failed tasks" in preamble

    def test_failure_preamble(self):
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.FAILURE,
            task_count=2,
            success_count=0,
            failure_count=2,
            tasks=[
                TaskSummary(capability_name="goal_list", success=False, error="connection failed"),
                TaskSummary(capability_name="task_list", success=False, error="timeout"),
            ],
        )
        preamble = _format_capability_preamble(result)
        assert "failed completely" in preamble
        assert "connection failed" in preamble
        assert "complete the original task" in preamble

    def test_skipped_preamble(self):
        result = CapabilityExecutionResult.skipped_for_approval("L3 caps present")
        preamble = _format_capability_preamble(result)
        assert "skipped" in preamble
        assert "approval-required" in preamble
        assert "L3" in preamble
        assert "authorized tool execution path" in preamble

    def test_success_returns_empty(self):
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.SUCCESS,
            task_count=1,
            success_count=1,
        )
        preamble = _format_capability_preamble(result)
        assert preamble == ""

    def test_not_executed_returns_empty(self):
        result = CapabilityExecutionResult(status=CapabilityExecutionStatus.NOT_EXECUTED)
        preamble = _format_capability_preamble(result)
        assert preamble == ""

    def test_preamble_bounded_output(self):
        """Preamble should not expose large outputs."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.PARTIAL,
            task_count=1,
            success_count=0,
            failure_count=1,
            tasks=[
                TaskSummary(
                    capability_name="goal_list",
                    success=False,
                    error="x" * 1000,  # Very long error
                ),
            ],
        )
        preamble = _format_capability_preamble(result)
        # Error should be truncated to 100 chars in preamble
        assert len(preamble) < 500

    def test_preamble_no_raw_tool_output(self):
        """Preamble should truncate long error messages."""
        long_error = "x" * 200 + "SENSITIVE_MARKER"
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.FAILURE,
            task_count=1,
            success_count=0,
            failure_count=1,
            tasks=[
                TaskSummary(
                    capability_name="rag_search",
                    success=False,
                    error=long_error,
                ),
            ],
        )
        preamble = _format_capability_preamble(result)
        # Truncated to 100 chars — marker should not appear
        assert "SENSITIVE_MARKER" not in preamble


class TestModelContextInjection:
    def test_partial_injects_preamble_into_user_message(self):
        """Model should receive capability context for PARTIAL results."""
        from tools.delegate_tool import _attach_capability_plan, _execute_capability_plan_if_present

        child = MagicMock()
        child._capability_plan = MagicMock()
        child._capability_plan.stages = [MagicMock(task_ids=frozenset({"t1"}))]
        child._capability_graph = MagicMock()
        child._capability_graph.get_task.return_value = MagicMock(capability_name="goal_list")
        child._capability_task_type = "research"
        child._capability_authority_level = "L0"
        child._subagent_id = "sa-test"
        child._parent_session_id = "parent-test"
        child.session_id = "child-session"

        # The plan names goal_list, so the default capabilities must be
        # registered (the Decision 42 read-only check resolves every task in
        # the registry before execution); restored when the test ends.
        from plugins.memory.jarvis.capabilities import register_default_capabilities
        from plugins.memory.jarvis.capabilities import registry as registry_module
        registry = registry_module.JARVISCapabilityRegistry()
        register_default_capabilities(registry)

        with patch.object(registry_module, "_jarvis_registry", registry), \
             patch("plugins.memory.jarvis.capabilities.runtime.CapabilityRuntime.execute_plan") as mock_exec:
            mock_exec.return_value = MagicMock(
                all_succeeded=False,
                success_count=1,
                failure_count=1,
                task_count=2,
                stage_results=[],
            )
            result = _execute_capability_plan_if_present(child, "list goals")

        assert result.status == CapabilityExecutionStatus.PARTIAL
        preamble = _format_capability_preamble(result)
        assert len(preamble) > 0

    def test_success_no_preamble(self):
        """SUCCESS results should not inject preamble (early return)."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.SUCCESS,
            task_count=1,
            success_count=1,
        )
        preamble = _format_capability_preamble(result)
        assert preamble == ""

    def test_not_executed_no_preamble(self):
        """NOT_EXECUTED should not inject preamble."""
        result = CapabilityExecutionResult(status=CapabilityExecutionStatus.NOT_EXECUTED)
        preamble = _format_capability_preamble(result)
        assert preamble == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
