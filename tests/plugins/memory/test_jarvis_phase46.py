"""Phase 46 — Timeout Validation and Non-blocking Shutdown tests."""

import time
import pytest
from unittest.mock import MagicMock

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


class TestTimeoutValidation:
    def test_negative_task_timeout_raises_value_error(self):
        """Negative task_timeout_seconds raises ValueError."""
        registry = _build_registry()
        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        with pytest.raises(ValueError, match="task_timeout_seconds must be > 0"):
            runtime.execute_plan(
                plan,
                task_graph=graph,
                task_timeout_seconds=-1.0,
            )

    def test_zero_task_timeout_raises_value_error(self):
        """Zero task_timeout_seconds raises ValueError."""
        registry = _build_registry()
        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        with pytest.raises(ValueError, match="task_timeout_seconds must be > 0"):
            runtime.execute_plan(
                plan,
                task_graph=graph,
                task_timeout_seconds=0,
            )

    def test_negative_plan_timeout_raises_value_error(self):
        """Negative plan_timeout_seconds raises ValueError."""
        registry = _build_registry()
        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        with pytest.raises(ValueError, match="plan_timeout_seconds must be >= 0"):
            runtime.execute_plan(
                plan,
                task_graph=graph,
                plan_timeout_seconds=-1.0,
            )

    def test_zero_plan_timeout_valid(self):
        """Zero plan_timeout_seconds is valid (immediate break)."""
        registry = _build_registry()
        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        graph.add_task(Task(id="t2", capability_name="task_list"))
        plan = ExecutionPlan(stages=[
            ExecutionStage(stage_index=0, task_ids=frozenset({"t1"})),
            ExecutionStage(stage_index=1, task_ids=frozenset({"t2"})),
        ])

        result = runtime.execute_plan(
            plan,
            task_graph=graph,
            task_timeout_seconds=30.0,
            plan_timeout_seconds=0,
        )

        # With plan_timeout=0, should break immediately (no stages run)
        # elapsed >= 0 is true at first iteration → break
        assert len(result.stage_results) == 0


class TestNonBlockingShutdown:
    def test_subsequent_task_continues_after_timeout(self):
        """A timed-out task does not block subsequent tasks from starting."""
        registry = _build_registry()
        registry.register(SlowCapability)
        registry.register(FastCapability)

        runtime = CapabilityRuntime(registry=registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="slow_capability"))
        graph.add_task(Task(id="t2", capability_name="fast_capability"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1", "t2"}))])

        start = time.monotonic()
        result = runtime.execute_plan(
            plan,
            task_graph=graph,
            task_timeout_seconds=0.3,
            plan_timeout_seconds=5.0,
        )
        elapsed = time.monotonic() - start

        # Both tasks should have been attempted
        assert result.task_count == 2
        # Fast task should succeed
        fast_result = [tr for tr in result.stage_results[0].task_results if tr.capability_name == "fast_capability"]
        assert len(fast_result) == 1
        assert fast_result[0].result.success
        # Slow task should fail
        slow_result = [tr for tr in result.stage_results[0].task_results if tr.capability_name == "slow_capability"]
        assert len(slow_result) == 1
        assert not slow_result[0].result.success
        # Total time should be ~0.3s (timeout) + fast task, not 10s
        assert elapsed < 5.0


class TestTimeoutErrorMessage:
    def test_timeout_error_message_clear(self):
        """Timeout error message is clear and bounded."""
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
        error = task_result.result.error
        assert error is not None
        assert "timed out" in error.lower()
        assert "seconds" in error.lower()
        assert len(error) < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
