"""Phase 38 — Capability Runtime tests."""

import pytest
from unittest.mock import patch

from plugins.memory.jarvis.capabilities.base import (
    Capability,
    CapabilityCategory,
    CapabilityContext,
    CapabilityInputType,
    CapabilityOutputType,
    CapabilityRequirements,
    CapabilityResult,
    get_registry,
    reset_registry,
)
from plugins.memory.jarvis.capabilities.registry import (
    JARVISCapabilityRegistry,
    get_jarvis_registry,
    reset_jarvis_registry,
)
from plugins.memory.jarvis.capabilities.runtime import (
    CapabilityRuntime,
    CapabilityRuntimeError,
    PlanExecutionResult,
    StageExecutionResult,
    TaskExecutionResult,
    execute_plan,
)
from plugins.memory.jarvis.planner.task_graph import (
    Task,
    TaskGraph,
    ExecutionPlan,
    ExecutionStage,
    execution_plan_from_graph,
)


@pytest.fixture(autouse=True)
def _reset_registries():
    reset_registry()
    reset_jarvis_registry()
    yield
    reset_registry()
    reset_jarvis_registry()


# ---------------------------------------------------------------------------
# Test capabilities
# ---------------------------------------------------------------------------

class SimpleReadCap(Capability):
    def get_name(self) -> str:
        return "simple_read"

    def get_description(self) -> str:
        return "A simple read capability"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_search"}),
            input_type=CapabilityInputType.QUERY,
            output_type=CapabilityOutputType.STRUCTURED,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        query = context.parameters.get("query", "default")
        return CapabilityResult.success_result(
            data={"query": query, "results": ["r1", "r2"]},
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class DependentCap(Capability):
    """Depends on simple_read."""

    def get_name(self) -> str:
        return "dependent_cap"

    def get_description(self) -> str:
        return "Depends on simple_read"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.SYNTHESIS

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_search"}),
            dependencies=frozenset({"simple_read"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.TEXT,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        dep = context.dependency_results.get("simple_read", {})
        return CapabilityResult.success_result(
            data={"synthesized": dep},
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class FailingCap(Capability):
    def get_name(self) -> str:
        return "failing_cap"

    def get_description(self) -> str:
        return "Always fails"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.ANALYSIS

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_search"}),
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return False, "Intentional failure for testing"

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult.failure_result(error="Should not reach here")


class ExceptionCap(Capability):
    def get_name(self) -> str:
        return "exception_cap"

    def get_description(self) -> str:
        return "Raises exception"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.ANALYSIS

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_search"}),
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        raise RuntimeError("Boom!")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_registry():
    registry = get_jarvis_registry()
    registry.register(SimpleReadCap)
    registry.register(DependentCap)
    registry.register(FailingCap)
    registry.register(ExceptionCap)
    return registry


def _build_linear_graph():
    """simple_read -> dependent_cap"""
    g = TaskGraph()
    g.add_task(Task(id="t1", capability_name="simple_read", parameters={"query": "test"}))
    g.add_task(Task(id="t2", capability_name="dependent_cap"))
    g.add_edge("t1", "t2")
    return g


def _build_parallel_graph():
    """Two independent tasks"""
    g = TaskGraph()
    g.add_task(Task(id="t1", capability_name="simple_read"))
    g.add_task(Task(id="t2", capability_name="simple_read"))
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCapabilityRuntime:
    def test_execute_empty_plan(self):
        runtime = CapabilityRuntime()
        plan = ExecutionPlan(stages=[])
        graph = TaskGraph()
        result = runtime.execute_plan(plan, graph)
        assert result.all_succeeded is True
        assert result.task_count == 0

    def test_execute_single_task(self):
        _build_registry()
        runtime = CapabilityRuntime()
        g = TaskGraph()
        g.add_task(Task(id="t1", capability_name="simple_read", parameters={"query": "hello"}))
        plan = execution_plan_from_graph(g)
        result = runtime.execute_plan(plan, g)
        assert result.all_succeeded is True
        assert result.task_count == 1
        assert result.success_count == 1
        assert result.failure_count == 0
        # Check the task result
        task_result = result.stage_results[0].task_results[0]
        assert task_result.result.success is True
        assert task_result.result.data["query"] == "hello"

    def test_execute_with_dependency(self):
        _build_registry()
        runtime = CapabilityRuntime()
        g = _build_linear_graph()
        plan = execution_plan_from_graph(g)
        result = runtime.execute_plan(plan, g)
        assert result.all_succeeded is True
        assert result.task_count == 2
        # Dependent task should have received dependency results
        dep_task_result = result.stage_results[1].task_results[0]
        assert dep_task_result.capability_name == "dependent_cap"
        assert dep_task_result.result.success is True
        # The dependency data should be present
        synthesized = dep_task_result.result.data["synthesized"]
        assert synthesized is not None
        assert synthesized["query"] == "test"
        assert synthesized["results"] == ["r1", "r2"]

    def test_execute_parallel_tasks(self):
        _build_registry()
        runtime = CapabilityRuntime()
        g = _build_parallel_graph()
        plan = execution_plan_from_graph(g)
        result = runtime.execute_plan(plan, g)
        assert result.all_succeeded is True
        assert result.task_count == 2
        assert len(result.stage_results) == 1
        assert len(result.stage_results[0].task_results) == 2

    def test_execute_failing_task(self):
        _build_registry()
        runtime = CapabilityRuntime()
        g = TaskGraph()
        g.add_task(Task(id="t1", capability_name="failing_cap"))
        plan = execution_plan_from_graph(g)
        result = runtime.execute_plan(plan, g)
        assert result.all_succeeded is False
        assert result.failure_count == 1
        assert result.success_count == 0
        # Stage should report partial failure
        assert result.stage_results[0].all_succeeded is False
        assert result.stage_results[0].any_succeeded is False

    def test_execute_exception_task(self):
        _build_registry()
        runtime = CapabilityRuntime()
        g = TaskGraph()
        g.add_task(Task(id="t1", capability_name="exception_cap"))
        plan = execution_plan_from_graph(g)
        result = runtime.execute_plan(plan, g)
        assert result.all_succeeded is False
        assert result.failure_count == 1
        assert "Boom!" in (result.stage_results[0].task_results[0].result.error or "")

    def test_execute_mixed_success_and_failure(self):
        _build_registry()
        runtime = CapabilityRuntime()
        g = TaskGraph()
        g.add_task(Task(id="t1", capability_name="simple_read"))
        g.add_task(Task(id="t2", capability_name="failing_cap"))
        plan = execution_plan_from_graph(g)
        result = runtime.execute_plan(plan, g)
        assert result.all_succeeded is False
        assert result.success_count == 1
        assert result.failure_count == 1
        assert result.stage_results[0].all_succeeded is False
        assert result.stage_results[0].any_succeeded is True

    def test_execute_unknown_capability(self):
        """Task references a capability not in registry."""
        _build_registry()
        runtime = CapabilityRuntime()
        g = TaskGraph()
        g.add_task(Task(id="t1", capability_name="nonexistent_cap"))
        plan = execution_plan_from_graph(g)
        result = runtime.execute_plan(plan, g)
        assert result.all_succeeded is False
        assert result.failure_count == 1
        assert "not found in registry" in (result.stage_results[0].task_results[0].result.error or "")

    def test_execute_preserves_execution_id(self):
        _build_registry()
        runtime = CapabilityRuntime()
        g = TaskGraph()
        g.add_task(Task(id="t1", capability_name="simple_read"))
        plan = execution_plan_from_graph(g)
        result = runtime.execute_plan(plan, g, parent_session_id="parent-xyz", subagent_id="sa-abc")
        task_result = result.stage_results[0].task_results[0]
        assert task_result.result.execution_id == "exec-t1"

    def test_execute_timing_recorded(self):
        _build_registry()
        runtime = CapabilityRuntime()
        g = TaskGraph()
        g.add_task(Task(id="t1", capability_name="simple_read"))
        plan = execution_plan_from_graph(g)
        result = runtime.execute_plan(plan, g)
        assert result.total_execution_time_seconds >= 0
        for sr in result.stage_results:
            for tr in sr.task_results:
                assert tr.execution_time_seconds >= 0

    def test_execute_plan_convenience_function(self):
        _build_registry()
        g = TaskGraph()
        g.add_task(Task(id="t1", capability_name="simple_read"))
        plan = execution_plan_from_graph(g)
        result = execute_plan(plan, g)
        assert result.all_succeeded is True


class TestRuntimeWithPlanner:
    """Integration: planner produces plan, runtime executes it."""

    def test_plan_then_execute(self):
        from plugins.memory.jarvis.planner.sub_agent_planner import SubAgentPlanner
        registry = get_jarvis_registry()
        registry.register(SimpleReadCap)
        registry.register(DependentCap)
        planner = SubAgentPlanner(registry=registry)
        plan = planner.plan("test research", task_type="research", authority_level="L0")
        # Build a matching graph for execution
        # (In real usage, planner would return the graph too)
        runtime = CapabilityRuntime(registry=registry)
        # Construct graph from plan
        g = TaskGraph()
        for stage in plan.stages:
            for tid in stage.task_ids:
                # Extract capability name from task ID
                inner = tid[len("task_"):]
                for name in registry.list_capabilities():
                    if inner.startswith(name + "_"):
                        g.add_task(Task(id=tid, capability_name=name))
                        break
        result = runtime.execute_plan(plan, g)
        assert result.task_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
