"""Phase 40B — Result & Plan Contract Hardening tests."""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis.planner.plan_result import (
    CapabilityPlanResult,
    plan_with_result,
)
from plugins.memory.jarvis.capabilities.execution_result import (
    CapabilityExecutionResult,
    CapabilityExecutionStatus,
    TaskSummary,
)
from plugins.memory.jarvis.planner.sub_agent_planner import SubAgentPlanner
from plugins.memory.jarvis.capabilities.registry import (
    get_jarvis_registry,
    reset_jarvis_registry,
)
from plugins.memory.jarvis.capabilities import register_default_capabilities


@pytest.fixture(autouse=True)
def _reset_registries():
    reset_jarvis_registry()
    yield
    reset_jarvis_registry()


def _make_child_with_plan(plan, graph):
    child = MagicMock()
    child._capability_plan = plan
    child._capability_graph = graph
    child._capability_task_type = "research"
    child._capability_authority_level = "L0"
    child._subagent_id = "sa-test-001"
    child._parent_session_id = "parent-001"
    child.session_id = "child-session-001"
    return child


# ---------------------------------------------------------------------------
# CapabilityPlanResult tests
# ---------------------------------------------------------------------------

class TestCapabilityPlanResult:
    def test_empty_plan(self):
        from plugins.memory.jarvis.planner.task_graph import ExecutionPlan, TaskGraph
        result = CapabilityPlanResult(plan=ExecutionPlan(stages=[]), graph=TaskGraph())
        assert result.is_empty is True
        assert result.stage_count == 0
        assert result.task_count == 0

    def test_plan_with_stages(self):
        from plugins.memory.jarvis.planner.task_graph import ExecutionPlan, ExecutionStage
        plan = ExecutionPlan(stages=[
            ExecutionStage(stage_index=0, task_ids=frozenset({"t1"})),
            ExecutionStage(stage_index=1, task_ids=frozenset({"t2", "t3"})),
        ])
        result = CapabilityPlanResult(plan=plan, graph=MagicMock())
        assert result.is_empty is False
        assert result.stage_count == 2
        assert result.task_count == 3

    def test_immutable(self):
        from plugins.memory.jarvis.planner.task_graph import ExecutionPlan, TaskGraph
        result = CapabilityPlanResult(plan=ExecutionPlan(stages=[]), graph=TaskGraph())
        with pytest.raises(AttributeError):
            result.plan = None


# ---------------------------------------------------------------------------
# plan_with_result tests
# ---------------------------------------------------------------------------

class TestPlanWithResult:
    def test_returns_plan_and_graph(self):
        registry = get_jarvis_registry()
        register_default_capabilities(registry)
        planner = SubAgentPlanner(registry=registry)
        result = plan_with_result(planner, "list goals", task_type="research", authority_level="L0")
        assert result.plan is not None
        assert result.graph is not None
        assert result.is_empty is False

    def test_no_applicable_capabilities(self):
        registry = get_jarvis_registry()
        # Empty registry
        planner = SubAgentPlanner(registry=registry)
        result = plan_with_result(planner, "some goal", task_type="research")
        assert result.is_empty is True

    def test_backward_compatible_with_plan(self):
        """SubAgentPlanner.plan() still returns ExecutionPlan."""
        registry = get_jarvis_registry()
        register_default_capabilities(registry)
        planner = SubAgentPlanner(registry=registry)
        plan = planner.plan("list goals", task_type="research", authority_level="L0")
        # plan is an ExecutionPlan
        assert hasattr(plan, 'stages')

    @pytest.mark.parametrize("goal,task_type,authority_level", [
        ("list goals", "research", "L0"),
        ("list goals and tasks and decisions", "research", "L0"),
        ("store a memory, create a goal and record a decision", "default", "L2"),
    ])
    def test_plan_task_ids_resolve_in_returned_graph(self, goal, task_type, authority_level):
        """Every plan task id names a task in the returned graph, and the
        graph holds no other tasks. (The graph used to be rebuilt with fresh
        random ids, so none resolved.)"""
        registry = get_jarvis_registry()
        register_default_capabilities(registry)
        result = plan_with_result(SubAgentPlanner(registry=registry), goal,
                                  task_type=task_type, authority_level=authority_level)
        plan_ids = {task_id for stage in result.plan.stages for task_id in stage.task_ids}
        assert plan_ids, "expected a non-empty plan"
        assert all(result.graph.get_task(task_id) is not None for task_id in plan_ids)
        assert {t.id for t in result.graph.list_tasks()} == plan_ids

    def test_inconsistent_planner_output_is_rejected(self):
        """plan_with_result refuses a plan whose task ids are not in its graph."""
        from plugins.memory.jarvis.planner.task_graph import ExecutionPlan, ExecutionStage, TaskGraph

        planner = MagicMock()
        planner.plan_with_graph.return_value = (
            ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"ghost"}))]),
            TaskGraph(),
        )
        with pytest.raises(ValueError, match="ghost"):
            plan_with_result(planner, "list goals")


# ---------------------------------------------------------------------------
# CapabilityExecutionResult tests
# ---------------------------------------------------------------------------

class TestCapabilityExecutionResult:
    def test_success(self):
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.SUCCESS,
            task_count=3,
            success_count=3,
            failure_count=0,
        )
        assert result.all_succeeded is True
        assert result.has_failures is False
        assert result.approval_required is False
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["success_count"] == 3

    def test_partial(self):
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.PARTIAL,
            task_count=3,
            success_count=2,
            failure_count=1,
        )
        assert result.all_succeeded is False
        assert result.has_failures is True
        d = result.to_dict()
        assert d["status"] == "partial"

    def test_failure(self):
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.FAILURE,
            task_count=2,
            success_count=0,
            failure_count=2,
        )
        assert result.all_succeeded is False
        assert result.has_failures is True
        d = result.to_dict()
        assert d["status"] == "failure"

    def test_approval_skipped(self):
        result = CapabilityExecutionResult.skipped_for_approval("L3 cap present")
        assert result.approval_required is True
        assert result.status == CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED
        d = result.to_dict()
        assert d["status"] == "skipped_approval_required"

    def test_failed_factory(self):
        result = CapabilityExecutionResult.failed("runtime boom")
        assert result.status == CapabilityExecutionStatus.FAILURE
        assert "runtime boom" in result.summary

    def test_not_executed(self):
        result = CapabilityExecutionResult(status=CapabilityExecutionStatus.NOT_EXECUTED)
        assert result.all_succeeded is False
        assert result.has_failures is False

    def test_bounded_output(self):
        """Output is capped to prevent bloating the parent contract."""
        long_error = "x" * 1000
        task = TaskSummary(
            capability_name="test",
            success=False,
            error=long_error,
            output_preview="y" * 1000,
        )
        d = task.to_dict()
        assert len(d["error"]) <= 200
        assert len(d.get("output_preview", "")) <= 200

    def test_task_list_capped(self):
        """Task list capped at 10 summaries."""
        tasks = [TaskSummary(capability_name=f"cap_{i}", success=True) for i in range(20)]
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.SUCCESS,
            tasks=tasks,
        )
        d = result.to_dict()
        assert len(d["tasks"]) <= 10

    def test_from_plan_execution_result(self):
        mock_result = MagicMock()
        mock_result.all_succeeded = True
        mock_result.success_count = 2
        mock_result.failure_count = 0
        mock_result.task_count = 2
        stage_result = MagicMock()
        tr = MagicMock()
        tr.capability_name = "goal_list"
        tr.result.success = True
        tr.result.data = {"count": 5, "ok": True}
        tr.result.error = None
        stage_result.task_results = [tr]
        mock_result.stage_results = [stage_result]

        exec_result = CapabilityExecutionResult.from_plan_execution_result(mock_result)
        assert exec_result.status == CapabilityExecutionStatus.SUCCESS
        assert exec_result.success_count == 2

    def test_serialization_safe(self):
        """to_dict() is JSON-safe."""
        import json
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.PARTIAL,
            task_count=2,
            success_count=1,
            failure_count=1,
            tasks=[TaskSummary(capability_name="a", success=True),
                   TaskSummary(capability_name="b", success=False, error="fail")],
            summary="partial execution",
        )
        d = result.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert json_str is not None


# ---------------------------------------------------------------------------
# Integration: delegate_tool.py functions
# ---------------------------------------------------------------------------

class TestDelegateToolIntegration:
    def test_non_jarvis_child_unchanged(self):
        from tools.delegate_tool import _attach_capability_plan, _execute_capability_plan_if_present
        parent = MagicMock()
        parent._memory_manager = MagicMock()
        provider = MagicMock()
        provider.__class__.__name__ = "OtherProvider"
        parent._memory_manager._providers = [provider]
        child = MagicMock(spec=[])  # spec=[] prevents auto-attribute creation
        _attach_capability_plan(parent, child, "goal", "research")
        result = _execute_capability_plan_if_present(child, "goal")
        assert result.status == CapabilityExecutionStatus.NOT_EXECUTED

    def test_jarvis_no_plan_unchanged(self):
        from tools.delegate_tool import _execute_capability_plan_if_present
        child = MagicMock()
        child._capability_plan = None
        child._capability_graph = None
        result = _execute_capability_plan_if_present(child, "goal")
        assert result.status == CapabilityExecutionStatus.NOT_EXECUTED

    def test_execution_returns_result_object(self):
        """_execute_capability_plan_if_present returns CapabilityExecutionResult."""
        from tools.delegate_tool import _execute_capability_plan_if_present
        from plugins.memory.jarvis.planner.task_graph import TaskGraph, Task, ExecutionPlan, ExecutionStage

        registry = get_jarvis_registry()
        register_default_capabilities(registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        child = _make_child_with_plan(plan, graph)

        with patch("plugins.memory.jarvis.capabilities.runtime.CapabilityRuntime.execute_plan") as mock_exec:
            mock_exec.return_value = MagicMock(
                all_succeeded=True,
                success_count=1,
                failure_count=0,
                task_count=1,
                stage_results=[],
            )
            result = _execute_capability_plan_if_present(child, "list goals")

        assert isinstance(result, CapabilityExecutionResult)

    def test_l3_skipped(self):
        """L3 approval-required capability → result indicates skipped."""
        from tools.delegate_tool import _execute_capability_plan_if_present
        from plugins.memory.jarvis.planner.task_graph import TaskGraph, Task, ExecutionPlan, ExecutionStage

        registry = get_jarvis_registry()
        register_default_capabilities(registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="rag_delete"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        child = _make_child_with_plan(plan, graph)
        result = _execute_capability_plan_if_present(child, "delete memory")

        assert result.approval_required is True
        assert result.status == CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED

    def test_partial_execution_status(self):
        """Partial success → status=partial."""
        from tools.delegate_tool import _execute_capability_plan_if_present
        from plugins.memory.jarvis.planner.task_graph import TaskGraph, Task, ExecutionPlan, ExecutionStage

        registry = get_jarvis_registry()
        register_default_capabilities(registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        graph.add_task(Task(id="t2", capability_name="task_list"))
        plan = ExecutionPlan(stages=[
            ExecutionStage(stage_index=0, task_ids=frozenset({"t1", "t2"}))
        ])

        child = _make_child_with_plan(plan, graph)

        with patch("plugins.memory.jarvis.capabilities.runtime.CapabilityRuntime.execute_plan") as mock_exec:
            mock_exec.return_value = MagicMock(
                all_succeeded=False,
                success_count=1,
                failure_count=1,
                task_count=2,
                stage_results=[],
            )
            result = _execute_capability_plan_if_present(child, "list all")

        assert result.status == CapabilityExecutionStatus.PARTIAL
        assert result.has_failures is True


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_children_no_state_leak(self):
        """Two children execute capability plans independently."""
        from tools.delegate_tool import _execute_capability_plan_if_present
        from plugins.memory.jarvis.planner.task_graph import TaskGraph, Task, ExecutionPlan, ExecutionStage

        registry = get_jarvis_registry()
        register_default_capabilities(registry)

        def make_child(sid):
            graph = TaskGraph()
            graph.add_task(Task(id="t1", capability_name="goal_list"))
            plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])
            child = _make_child_with_plan(plan, graph)
            child._subagent_id = sid
            return child

        results = []

        def run_child(sid):
            child = make_child(sid)
            result = _execute_capability_plan_if_present(child, "list goals")
            results.append((sid, result.status))

        with patch("plugins.memory.jarvis.capabilities.runtime.CapabilityRuntime.execute_plan") as mock_exec:
            mock_exec.return_value = MagicMock(
                all_succeeded=True,
                success_count=1,
                failure_count=0,
                task_count=1,
                stage_results=[],
            )
            t1 = threading.Thread(target=run_child, args=("sa-1",))
            t2 = threading.Thread(target=run_child, args=("sa-2",))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        assert len(results) == 2
        assert results[0][1] == CapabilityExecutionStatus.SUCCESS
        assert results[1][1] == CapabilityExecutionStatus.SUCCESS

    def test_registry_safe_under_concurrent_reads(self):
        """Registry is read-only during capability execution."""
        registry = get_jarvis_registry()
        register_default_capabilities(registry)

        def read_caps():
            for _ in range(10):
                caps = registry.list_capabilities()
                assert len(caps) == 15

        threads = [threading.Thread(target=read_caps) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


# ---------------------------------------------------------------------------
# Failure isolation tests
# ---------------------------------------------------------------------------

class TestFailureIsolation:
    def test_planner_exception_isolated(self):
        """Planner failure doesn't break delegate_tool."""
        from tools.delegate_tool import _attach_capability_plan
        parent = MagicMock()
        parent._memory_manager = MagicMock()
        provider = MagicMock()
        provider.__class__.__name__ = "HermesRAGProvider"
        parent._memory_manager._providers = [provider]

        class FakeChild:
            _capability_plan = None
            _capability_graph = None
            _execution_context = MagicMock()
            _execution_context.authority_level = "L0"
            _execution_context.task_type = "research"

        child = FakeChild()

        with patch("plugins.memory.jarvis.planner.sub_agent_planner.SubAgentPlanner.plan",
                   side_effect=Exception("Planner boom")):
            _attach_capability_plan(parent, child, "goal", "research")

        # Should not raise, plan should be None
        assert child._capability_plan is None

    def test_runtime_exception_isolated(self):
        """Runtime failure returns failure result, doesn't raise."""
        from tools.delegate_tool import _execute_capability_plan_if_present
        from plugins.memory.jarvis.planner.task_graph import TaskGraph, Task, ExecutionPlan, ExecutionStage

        registry = get_jarvis_registry()
        register_default_capabilities(registry)

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        child = _make_child_with_plan(plan, graph)

        with patch("plugins.memory.jarvis.capabilities.runtime.CapabilityRuntime.execute_plan",
                   side_effect=Exception("Runtime boom")):
            result = _execute_capability_plan_if_present(child, "list goals")

        assert result.status == CapabilityExecutionStatus.FAILURE
        assert "Runtime boom" in result.summary


# ---------------------------------------------------------------------------
# Lifecycle unchanged tests
# ---------------------------------------------------------------------------

class TestLifecycleUnchanged:
    def test_no_new_lifecycle_states(self):
        """Existing lifecycle states remain."""
        from plugins.memory.jarvis.subagent_types import TERMINAL_STATES, VALID_TRANSITIONS
        existing_states = {"pending", "initializing", "running", "completed", "failed", "cancelled", "timed_out"}
        for s in existing_states:
            assert s in VALID_TRANSITIONS or s in TERMINAL_STATES, f"State {s} not in lifecycle"

    def test_capability_states_not_lifecycle(self):
        """CapabilityExecutionStatus is not a lifecycle state."""
        assert CapabilityExecutionStatus.SUCCESS.value != "running"
        assert CapabilityExecutionStatus.PARTIAL.value != "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
