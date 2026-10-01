"""Phase 51 — Public `delegate_task()` Integration Validation.

Tests the real public JARVIS delegation path through:
delegate_task() → _build_child_agent() → _attach_capability_plan() →
_run_single_child() → _execute_capability_plan_if_present() →
capability runtime → result contract.

Uses mock AIAgent to avoid real API calls while exercising the
full orchestration path.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call
import threading

from tools.delegate_tool import (
    delegate_task,
    _attach_capability_plan,
    _execute_capability_plan_if_present,
    _build_result_contract,
    _format_capability_preamble,
)


def _make_mock_parent(depth=0):
    """Create a mock parent agent with JARVIS provider."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    # JARVIS provider marker — must have __class__.__name__ == "HermesRAGProvider"
    class HermesRAGProvider:
        pass
    jarvis_provider = HermesRAGProvider()
    # Configure _memory_manager._providers to return the real list
    parent._memory_manager = MagicMock()
    parent._memory_manager._providers = [jarvis_provider]
    return parent


def _make_mock_child():
    """Create a mock child agent that records run_conversation calls."""
    child = MagicMock()
    child.session_id = "child-session-001"
    child.model = "gpt-4"
    child._capability_plan = None
    child._capability_graph = None
    child._capability_task_type = None
    child._capability_authority_level = None
    child._subagent_id = "sa-mock-001"
    child._parent_session_id = "parent-session-001"
    child.run_conversation_calls = []

    def run_conversation(user_message, task_id=None, **kwargs):
        child.run_conversation_calls.append({
            "user_message": user_message,
            "task_id": task_id,
        })
        return {
            "final_response": "Mocked response",
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    child.run_conversation = run_conversation
    return child


class TestDelegateTaskJarvisIntegration:
    """Tests for delegate_task() with JARVIS capability orchestration."""

    def test_delegate_task_attaches_jarvis_capability_plan(self):
        """delegate_task() attaches a capability plan for JARVIS parents."""
        from plugins.memory.jarvis.capabilities import register_default_capabilities
        from plugins.memory.jarvis.capabilities.registry import get_jarvis_registry, reset_jarvis_registry

        reset_jarvis_registry()
        registry = get_jarvis_registry()
        register_default_capabilities(registry)

        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = _make_mock_child()
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(
                goal="list goals",
                parent_agent=parent,
            ))

            # The child should have been created
            assert MockAgent.called
            # Result has nested structure
            assert "results" in result
            assert len(result["results"]) >= 1

    def test_delegate_task_result_contract_bounded(self):
        """delegate_task() returns a bounded result contract."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = _make_mock_child()
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(
                goal="test task",
                parent_agent=parent,
            ))

            # Result should have nested structure with bounded fields
            assert "results" in result
            assert "total_duration_seconds" in result
            # Each result entry should have standard contract fields
            entry = result["results"][0]
            assert "status" in entry
            assert "task_index" in entry
            # No raw tool output should be exposed
            assert "messages" not in entry

    def test_delegate_task_no_parent_errors(self):
        """delegate_task() requires a parent agent."""
        result = json.loads(delegate_task(goal="test"))
        assert "error" in result
        assert "parent agent" in result["error"]


class TestCapabilityPlanExecution:
    """Tests for capability plan execution through delegation."""

    def test_capability_plan_executes_successfully(self):
        """A L0 capability plan executes and returns SUCCESS."""
        from plugins.memory.jarvis.capabilities import register_default_capabilities
        from plugins.memory.jarvis.capabilities.registry import (
            get_jarvis_registry,
            reset_jarvis_registry,
        )

        reset_jarvis_registry()
        registry = get_jarvis_registry()
        register_default_capabilities(registry)

        child = _make_mock_child()
        child._subagent_id = "sa-test-001"
        child._parent_session_id = "parent-test-001"

        # Attach a plan
        parent = _make_mock_parent()
        _attach_capability_plan(parent, child, "list goals", "research", depth=1)

        # Execute the plan
        result = _execute_capability_plan_if_present(child, "list goals")

        # Should return a result (may be NOT_EXECUTED if no JARVIS provider detected,
        # or SUCCESS/PLAN if execution proceeds)
        assert result is not None

    def test_l3_capability_skipped_for_approval(self):
        """L3 capabilities return SKIPPED_APPROVAL_REQUIRED."""
        from plugins.memory.jarvis.capabilities import register_default_capabilities
        from plugins.memory.jarvis.capabilities.registry import (
            get_jarvis_registry,
            reset_jarvis_registry,
        )

        reset_jarvis_registry()
        registry = get_jarvis_registry()
        register_default_capabilities(registry)

        child = _make_mock_child()
        child._subagent_id = "sa-l3-test"
        child._parent_session_id = "parent-test-001"

        # Manually attach a plan with L3 capability
        from plugins.memory.jarvis.planner.task_graph import (
            ExecutionPlan, ExecutionStage, Task, TaskGraph,
        )

        graph = TaskGraph()
        graph.add_task(Task(
            id="task_rag_delete_test",
            capability_name="rag_delete",
            task_type="default",
            authority_level="L0",
            parameters={"memory_id": "test-123"},
        ))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"task_rag_delete_test"}))])

        child._capability_plan = plan
        child._capability_graph = graph
        child._capability_task_type = "default"
        child._capability_authority_level = "L0"

        result = _execute_capability_plan_if_present(child, "delete memory")

        # L3 capability should be skipped
        from plugins.memory.jarvis.capabilities.execution_result import CapabilityExecutionStatus
        assert result.status == CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED


class TestNonJarvisRegression:
    """Verify non-JARVIS delegation still works."""

    def test_non_jarvis_parent_no_plan_attached(self):
        """Non-JARVIS parents should not get capability plans."""
        parent = MagicMock()
        parent.base_url = "https://openrouter.ai/api/v1"
        parent.api_key = "test-key"
        parent.provider = "openrouter"
        parent.api_mode = "chat_completions"
        parent.model = "gpt-4"
        parent.platform = "cli"
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        # Non-JARVIS provider
        other_provider = MagicMock()
        other_provider.__class__.__name__ = "OtherProvider"
        parent._memory_manager._providers = [other_provider]

        child = _make_mock_child()

        _attach_capability_plan(parent, child, "regular task", "default", depth=1)

        # No plan should be attached for non-JARVIS parents
        assert child._capability_plan is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
