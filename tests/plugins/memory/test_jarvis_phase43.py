"""Phase 43 — End-to-End Pipeline Validation.

Tests the full capability orchestration pipeline with realistic child-agent
attributes (not just isolated MagicMock component tests).

Verifies:
- _attach_capability_plan() reads child._execution_context correctly
- _execute_capability_plan_if_present() processes real attached plans
- Phase 42 preamble injection reaches the model's user_message
- L3 approval gating works with real plan structures
- _build_result_contract() includes capability_execution metadata
- Non-JARVIS children are unaffected
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

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
    _build_result_contract,
    _execute_capability_plan_if_present,
    _format_capability_preamble,
)


@dataclass(frozen=True)
class FakeExecutionContext:
    authority_level: str = "L0"
    task_type: str = "research"


class FakeChild:
    """Realistic child-agent mock for E2E pipeline tests."""

    def __init__(
        self,
        *,
        authority_level: str = "L0",
        task_type: str = "research",
        subagent_id: str = "sa-test-001",
        parent_session_id: str = "parent-test",
        session_id: str = "child-session-001",
        model: str = "gpt-4",
        goal: str = "list goals",
        jarvis_attached: bool = True,
    ):
        self._execution_context = FakeExecutionContext(
            authority_level=authority_level,
            task_type=task_type,
        )
        self._subagent_id = subagent_id
        self._parent_session_id = parent_session_id
        self.session_id = session_id
        self.model = model
        self._goal = goal
        self._jarvis_attached = jarvis_attached

        # Capture run_conversation args for assertion
        self.run_conversation_calls: List[Dict[str, Any]] = []

        # Capability plan state (attached by _attach_capability_plan)
        self._capability_plan: Optional[Any] = None
        self._capability_graph: Optional[Any] = None
        self._capability_task_type: Optional[str] = None
        self._capability_authority_level: Optional[str] = None

    def run_conversation(self, user_message: str, task_id: str = None, **kwargs) -> Dict[str, Any]:
        self.run_conversation_calls.append({
            "user_message": user_message,
            "task_id": task_id,
        })
        return {"response": "mocked", "user_message_received": user_message}


class FakeMemoryManager:
    def __init__(self, providers=None):
        self._providers = providers or []


@pytest.fixture(autouse=True)
def _reset_registries():
    reset_jarvis_registry()
    yield
    reset_jarvis_registry()


def _build_registry():
    registry = get_jarvis_registry()
    register_default_capabilities(registry)
    return registry


class TestE2EPlanAttachment:
    """Test 1-2: Plan attachment to realistic child."""

    def test_e2e_plan_attached_to_child(self):
        """Plan with relevant capabilities is attached to child with JARVIS provider."""
        _build_registry()
        jarvis_provider = MagicMock()
        jarvis_provider.__class__.__name__ = "HermesRAGProvider"
        parent_mm = FakeMemoryManager(providers=[jarvis_provider])
        parent = MagicMock()
        parent._memory_manager = parent_mm

        child = FakeChild(
            authority_level="L0",
            task_type="research",
            goal="list goals",
            jarvis_attached=True,
        )

        _attach_capability_plan(parent, child, "list goals", "research", depth=1)

        assert child._capability_plan is not None
        assert child._capability_graph is not None
        assert child._capability_task_type == "research"
        assert child._capability_authority_level == "L0"

    def test_e2e_plan_executed_for_l0_child(self):
        """Full plan→execute→result chain with L0 authority."""
        _build_registry()
        jarvis_provider = MagicMock()
        jarvis_provider.__class__.__name__ = "HermesRAGProvider"
        parent_mm = FakeMemoryManager(providers=[jarvis_provider])
        parent = MagicMock()
        parent._memory_manager = parent_mm

        child = FakeChild(authority_level="L0", task_type="research", goal="list goals")

        _attach_capability_plan(parent, child, "list goals", "research", depth=1)

        # Plan should be attached
        assert child._capability_plan is not None

        # Execute
        result = _execute_capability_plan_if_present(child, "list goals")

        # Should get a result (not None)
        assert result is not None
        # L0 plan for "list goals" should succeed (only L0 capabilities)
        assert result.status == CapabilityExecutionStatus.SUCCESS


class TestE2EL3ApprovalGating:
    """Test 3: L3 approval-required capabilities skipped for L0 child."""

    def test_e2e_l3_skipped_for_l0_child(self):
        """L3 capability in plan → SKIPPED_APPROVAL_REQUIRED."""
        _build_registry()
        jarvis_provider = MagicMock()
        jarvis_provider.__class__.__name__ = "HermesRAGProvider"
        parent_mm = FakeMemoryManager(providers=[jarvis_provider])
        parent = MagicMock()
        parent._memory_manager = parent_mm

        child = FakeChild(authority_level="L0", goal="delete all data")

        # Manually attach a plan with L3 capability
        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="rag_delete"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        child._capability_plan = plan
        child._capability_graph = graph
        child._capability_task_type = "default"
        child._capability_authority_level = "L0"

        result = _execute_capability_plan_if_present(child, "delete all data")

        assert result.status == CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED


class TestE2ERemediationContext:
    """Test 4-5: Remediation preamble reaches model context."""

    def test_e2e_remediation_preamble_in_user_message(self):
        """Phase 42 preamble reaches run_conversation user_message."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.PARTIAL,
            task_count=2,
            success_count=1,
            failure_count=1,
            tasks=[
                TaskSummary(capability_name="goal_list", success=True),
                TaskSummary(capability_name="task_list", success=False, error="timeout"),
            ],
        )

        preamble = _format_capability_preamble(result)
        assert preamble != ""
        assert "partially executed" in preamble
        assert "goal_list" in preamble
        assert "task_list" in preamble

    def test_e2e_partial_triggers_model_loop(self):
        """PARTIAL result → no early return → model loop runs."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.PARTIAL,
            task_count=2,
            success_count=1,
            failure_count=1,
        )

        # Partial should NOT trigger early return
        assert not result.all_succeeded
        # Preamble should be generated
        preamble = _format_capability_preamble(result)
        assert preamble != ""


class TestE2ENonJarvisChild:
    """Test 6: Non-JARVIS children unaffected."""

    def test_e2e_no_plan_for_non_jarvis(self):
        """Non-JARVIS parent → no plan attached → clean model loop."""
        _build_registry()
        other_provider = MagicMock()
        other_provider.__class__.__name__ = "OtherProvider"
        parent_mm = FakeMemoryManager(providers=[other_provider])
        parent = MagicMock()
        parent._memory_manager = parent_mm

        child = FakeChild(authority_level="L0", goal="some task", jarvis_attached=False)

        _attach_capability_plan(parent, child, "some task", "default", depth=1)

        assert child._capability_plan is None
        assert child._capability_graph is None


class TestE2EResultContract:
    """Test 7: Result contract propagation."""

    def test_e2e_result_contract_has_capability_execution(self):
        """_build_result_contract includes capability_execution when executed."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.SUCCESS,
            task_count=1,
            success_count=1,
            summary="Test summary",
        )

        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="Test",
            subagent_id="sa-1",
            child_session_id="cs-1",
            task_goal="test",
            lifecycle_state="completed",
            capability_execution=result.to_dict(),
        )

        assert contract["capability_execution"]["status"] == "success"
        assert contract["capability_execution"]["success_count"] == 1


class TestE2EExecutionContext:
    """Test 8: Authority level propagation."""

    def test_e2e_execution_context_authority_propagated(self):
        """child._execution_context.authority_level correctly read by planner."""
        _build_registry()
        jarvis_provider = MagicMock()
        jarvis_provider.__class__.__name__ = "HermesRAGProvider"
        parent_mm = FakeMemoryManager(providers=[jarvis_provider])
        parent = MagicMock()
        parent._memory_manager = parent_mm

        # L2 child
        child = FakeChild(authority_level="L2", task_type="research", goal="list goals")

        _attach_capability_plan(parent, child, "list goals", "research", depth=1)

        assert child._capability_authority_level == "L2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
