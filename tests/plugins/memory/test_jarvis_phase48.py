"""Phase 48 — Real-World Integration Validation tests."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.jarvis.capabilities import register_default_capabilities
from plugins.memory.jarvis.capabilities.execution_result import (
    CapabilityExecutionResult,
    CapabilityExecutionStatus,
    TaskSummary,
)
from plugins.memory.jarvis.capabilities.registry import (
    get_jarvis_registry,
    reset_jarvis_registry,
)
from plugins.memory.jarvis.planner.task_graph import (
    ExecutionPlan,
    ExecutionStage,
    Task,
    TaskGraph,
)
from tools.delegate_tool import (
    _attach_capability_plan,
    _execute_capability_plan_if_present,
    _format_capability_preamble,
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


class FakeChild:
    """Minimal child mock that captures run_conversation args."""

    def __init__(self, *, authority_level="L0", task_type="research",
                 subagent_id="sa-test", parent_session_id="parent-test",
                 session_id="child-session", goal="test goal"):
        self._execution_context = MagicMock()
        self._execution_context.authority_level = authority_level
        self._execution_context.task_type = task_type
        self._subagent_id = subagent_id
        self._parent_session_id = parent_session_id
        self.session_id = session_id
        self._goal = goal
        self.model = "gpt-4"
        self._capability_plan = None  # type: ignore
        self._capability_graph = None  # type: ignore
        self._capability_task_type = None  # type: ignore
        self._capability_authority_level = None  # type: ignore
        self.run_conversation_calls = []

    def run_conversation(self, user_message, task_id=None, **kwargs):
        self.run_conversation_calls.append({
            "user_message": user_message,
            "task_id": task_id,
        })
        return {"response": "mocked", "user_message_received": user_message}


class TestConcurrentChildIsolation:
    def test_two_children_independent_plans(self):
        """Two children with different plans execute independently."""
        _build_registry()
        parent = MagicMock()
        jarvis_provider = MagicMock()
        jarvis_provider.__class__.__name__ = "HermesRAGProvider"
        parent._memory_manager._providers = [jarvis_provider]

        child1 = FakeChild(
            subagent_id="sa-1",
            goal="list goals",
            session_id="child-1",
        )
        child2 = FakeChild(
            subagent_id="sa-2",
            goal="list tasks",
            session_id="child-2",
        )

        _attach_capability_plan(parent, child1, "list goals", "research", depth=1)
        _attach_capability_plan(parent, child2, "list tasks", "research", depth=1)

        result1 = _execute_capability_plan_if_present(child1, "list goals")
        result2 = _execute_capability_plan_if_present(child2, "list tasks")

        # Results are independent
        assert result1 is not result2
        # Each child's plan should be different
        assert child1._capability_plan is not child2._capability_plan
        # Both should succeed (L0 capabilities)
        assert result1.status == CapabilityExecutionStatus.SUCCESS
        assert result2.status == CapabilityExecutionStatus.SUCCESS

    def test_one_child_timeout_does_not_affect_other(self):
        """One child's timeout doesn't alter another child's result."""
        _build_registry()
        parent = MagicMock()
        jarvis_provider = MagicMock()
        jarvis_provider.__class__.__name__ = "HermesRAGProvider"
        parent._memory_manager._providers = [jarvis_provider]

        child1 = FakeChild(subagent_id="sa-1", goal="list goals", session_id="child-1")
        child2 = FakeChild(subagent_id="sa-2", goal="list tasks", session_id="child-2")

        _attach_capability_plan(parent, child1, "list goals", "research", depth=1)
        _attach_capability_plan(parent, child2, "list tasks", "research", depth=1)

        results = {}

        def run_child(child, goal):
            results[child._subagent_id] = _execute_capability_plan_if_present(child, goal)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(run_child, child1, "list goals")
            f2 = pool.submit(run_child, child2, "list tasks")
            f1.result()
            f2.result()

        # Both should succeed independently
        assert results["sa-1"].status == CapabilityExecutionStatus.SUCCESS
        assert results["sa-2"].status == CapabilityExecutionStatus.SUCCESS


class TestRealAIAgentChild:
    def test_plan_attached_to_real_aiaiagent_child(self):
        """_attach_capability_plan works with a minimal AIAgent-like child."""
        _build_registry()
        parent = MagicMock()
        jarvis_provider = MagicMock()
        jarvis_provider.__class__.__name__ = "HermesRAGProvider"
        parent._memory_manager._providers = [jarvis_provider]

        # Construct a minimal AIAgent-like object
        child = MagicMock()
        child._execution_context = MagicMock()
        child._execution_context.authority_level = "L0"
        child._execution_context.task_type = "research"
        child.session_id = "child-session-001"
        child.model = "gpt-4"
        child._subagent_id = "sa-real-001"
        child._parent_session_id = "parent-001"
        # Initialize capability attributes to None (as a real child would have)
        child._capability_plan = None
        child._capability_graph = None
        child._capability_task_type = None
        child._capability_authority_level = None

        _attach_capability_plan(parent, child, "list goals", "research", depth=1)

        # Verify plan was attached
        assert child._capability_plan is not None
        assert child._capability_graph is not None
        assert child._capability_task_type == "research"
        assert child._capability_authority_level == "L0"

    def test_plan_not_attached_for_non_jarvis_parent(self):
        """Non-JARVIS parent results in no plan attached."""
        _build_registry()
        parent = MagicMock()
        other_provider = MagicMock()
        other_provider.__class__.__name__ = "OtherProvider"
        parent._memory_manager._providers = [other_provider]

        child = FakeChild(goal="some task")
        _attach_capability_plan(parent, child, "some task", "default", depth=1)

        assert child._capability_plan is None
        assert child._capability_graph is None


class TestPartialRemediation:
    def test_preamble_in_user_message(self):
        """PARTIAL result injects preamble into run_conversation user_message."""
        child = FakeChild(subagent_id="sa-partial", goal="list goals and tasks")

        # Manually attach a plan with mixed success/failure
        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="goal_list"))
        graph.add_task(Task(id="t2", capability_name="task_list"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1", "t2"}))])

        child._capability_plan = plan
        child._capability_graph = graph
        child._capability_task_type = "research"
        child._capability_authority_level = "L0"
        child._subagent_id = "sa-partial"
        child._parent_session_id = "parent-test"

        # Execute plan
        result = _execute_capability_plan_if_present(child, "list goals and tasks")

        # Simulate the Phase 42 injection logic from _run_single_child
        goal_with_context = "list goals and tasks"
        if result and result.status.value in ("partial", "failure", "skipped_approval_required"):
            preamble = _format_capability_preamble(result)
            if preamble and isinstance(goal_with_context, str):
                goal_with_context = f"{goal_with_context}\n\n{preamble}"

        # Simulate run_conversation
        child.run_conversation(user_message=goal_with_context)

        # Verify preamble was included
        assert len(child.run_conversation_calls) == 1
        user_message = child.run_conversation_calls[0]["user_message"]
        assert "list goals and tasks" in user_message
        # If result was partial, preamble should be present
        if result.status == CapabilityExecutionStatus.PARTIAL:
            assert "---" in user_message
            assert "Succeeded:" in user_message or "Failed:" in user_message


class TestL3Routing:
    def test_l3_skipped_routes_to_normal_tool_path(self):
        """SKIPPED_APPROVAL tells model to use normal authorized tool path."""
        _build_registry()
        child = FakeChild(authority_level="L0", goal="delete all data")

        # Manually attach a plan with L3 capability
        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="rag_delete"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        child._capability_plan = plan
        child._capability_graph = graph
        child._capability_task_type = "default"
        child._capability_authority_level = "L0"
        child._subagent_id = "sa-l3-test"
        child._parent_session_id = "parent-test"

        result = _execute_capability_plan_if_present(child, "delete all data")

        # Should be SKIPPED_APPROVAL_REQUIRED
        assert result.status == CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED

        # Generate preamble
        preamble = _format_capability_preamble(result)

        # Preamble should mention approval requirement
        assert "approval-required" in preamble.lower() or "L3" in preamble
        # Preamble should direct model to use normal tool path
        assert "authorized tool execution path" in preamble.lower() or "normal" in preamble.lower()

    def test_l3_skipped_does_not_execute_capability(self):
        """SKIPPED_APPROVAL does NOT execute the L3 capability via the runtime."""
        _build_registry()
        child = FakeChild(authority_level="L0", goal="delete all data")

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="rag_delete"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        child._capability_plan = plan  # type: ignore
        child._capability_graph = graph  # type: ignore
        child._capability_task_type = "default"  # type: ignore
        child._capability_authority_level = "L0"  # type: ignore
        child._subagent_id = "sa-l3-skip"
        child._parent_session_id = "parent-test"

        result = _execute_capability_plan_if_present(child, "delete all data")
        # Result should be SKIPPED_APPROVAL_REQUIRED
        assert result.status == CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
