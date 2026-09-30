"""Phase 44 — Capability Execution Resilience tests."""

import time
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis.capabilities import register_default_capabilities
from plugins.memory.jarvis.capabilities.base import (
    Capability,
    CapabilityCategory,
    CapabilityContext,
    CapabilityResult,
    CapabilityRequirements,
)
from plugins.memory.jarvis.capabilities.registry import (
    get_jarvis_registry,
    reset_jarvis_registry,
)
from plugins.memory.jarvis.capabilities.runtime import CapabilityRuntime
from plugins.memory.jarvis.planner.task_graph import (
    ExecutionPlan,
    ExecutionStage,
    Task,
    TaskGraph,
)


@pytest.fixture(autouse=True)
def _reset_registries():
    reset_jarvis_registry()
    yield
    reset_jarvis_registry()


def _build_registry():
    registry = get_jarvis_registry()
    register_default_capabilities(registry)
    return registry


class SlowCapability(Capability):
    """A capability that takes a configurable time to execute."""

    def __init__(self, sleep_time: float = 10.0):
        self._sleep_time = sleep_time

    def get_name(self) -> str:
        return "slow_capability"

    def get_description(self) -> str:
        return "A slow capability for testing timeouts"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset(),
            requires_approval=False,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        time.sleep(self._sleep_time)
        return CapabilityResult(success=True, data={"result": "completed"})


class FastCapability(Capability):
    """A capability that completes quickly."""

    def get_name(self) -> str:
        return "fast_capability"

    def get_description(self) -> str:
        return "A fast capability"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset(),
            requires_approval=False,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult(success=True, data={"result": "done"})


class TestTaskTimeout:
    def test_normal_execution_unchanged(self):
        """Normal execution without timeout remains unchanged."""
        registry = _build_registry()
        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        result = runtime.execute_plan(plan, task_graph=graph)

        assert result.all_succeeded
        assert result.task_count == 1
        assert result.success_count == 1
        assert result.failure_count == 0

    def test_task_timeout_becomes_failed_task(self):
        """A slow capability exceeding task_timeout_seconds becomes failed."""
        registry = _build_registry()
        registry.register(SlowCapability)

        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="slow_capability"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        result = runtime.execute_plan(
            plan,
            task_graph=graph,
            task_timeout_seconds=0.5,
            plan_timeout_seconds=10.0,
        )

        assert result.all_succeeded is False
        assert result.task_count == 1
        assert result.failure_count == 1

        # Check error message
        task_result = result.stage_results[0].task_results[0]
        error_lower = (task_result.result.error or "").lower()
        assert "timed out" in error_lower
        assert "seconds" in error_lower

    def test_another_task_continues_after_timeout(self):
        """Other tasks continue after one task times out."""
        registry = _build_registry()
        registry.register(SlowCapability)
        registry.register(FastCapability)

        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="slow_capability"))
        graph.add_task(Task(id="t2", capability_name="fast_capability"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1", "t2"}))])

        result = runtime.execute_plan(
            plan,
            task_graph=graph,
            task_timeout_seconds=0.5,
            plan_timeout_seconds=10.0,
        )

        # Should be partial (one failed, one succeeded)
        assert result.all_succeeded is False
        assert result.task_count == 2
        assert result.failure_count >= 1
        # Fast task should have succeeded
        fast_result = [tr for tr in result.stage_results[0].task_results if tr.capability_name == "fast_capability"]
        assert len(fast_result) == 1
        assert fast_result[0].result.success



class TestPlanTimeout:
    def test_plan_timeout_skips_remaining_stages(self):
        """plan_timeout_seconds causes remaining stages to be skipped."""
        registry = _build_registry()
        registry.register(FastCapability)

        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="fast_capability"))
        graph.add_task(Task(id="t2", capability_name="fast_capability"))
        graph.add_task(Task(id="t3", capability_name="fast_capability"))
        graph.add_task(Task(id="t4", capability_name="fast_capability"))
        plan = ExecutionPlan(stages=[
            ExecutionStage(stage_index=0, task_ids=frozenset({"t1"})),
            ExecutionStage(stage_index=1, task_ids=frozenset({"t2"})),
            ExecutionStage(stage_index=2, task_ids=frozenset({"t3"})),
            ExecutionStage(stage_index=3, task_ids=frozenset({"t4"})),
        ])

        # With very short plan timeout, later stages should be skipped
        result = runtime.execute_plan(
            plan,
            task_graph=graph,
            task_timeout_seconds=1.0,
            plan_timeout_seconds=0.0001,
        )

        # Only first stage should have executed
        assert len(result.stage_results) == 1


class TestTimeoutResultBounded:
    def test_timeout_error_bounded(self):
        """Timeout error message is bounded in length."""
        registry = _build_registry()
        registry.register(SlowCapability)

        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="slow_capability"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        result = runtime.execute_plan(
            plan,
            task_graph=graph,
            task_timeout_seconds=0.3,
            plan_timeout_seconds=10.0,
        )

        task_result = result.stage_results[0].task_results[0]
        # Error message should be bounded
        assert task_result.result.error is not None
        assert len(task_result.result.error) < 100


class TestDefaultTimeouts:
    def test_default_timeout_values(self):
        """Default timeout values preserve backward compatibility."""
        registry = _build_registry()
        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        # Default timeouts (30s task, 120s plan) should not affect fast tasks
        result = runtime.execute_plan(plan, task_graph=graph)

        assert result.all_succeeded
        assert result.failure_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
