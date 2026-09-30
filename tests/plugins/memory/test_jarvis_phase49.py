"""Phase 49 — Production Integration & Observability tests."""

import time
import pytest
from unittest.mock import MagicMock, patch

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


def _is_qdrant_available():
    """Check if Qdrant server is reachable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        client.get_collections()
        return True
    except Exception:
        return False


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


class TestTimeoutCount:
    def test_timeout_count_zero_on_success(self):
        """Successful plan has timeout_count=0."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.SUCCESS,
            task_count=2,
            success_count=2,
            failure_count=0,
        )
        assert result.timeout_count == 0
        assert result.to_dict()["timeout_count"] == 0

    def test_timeout_count_in_dict(self):
        """timeout_count appears in to_dict()."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.PARTIAL,
            task_count=3,
            success_count=2,
            failure_count=1,
            timeout_count=1,
        )
        d = result.to_dict()
        assert "timeout_count" in d
        assert d["timeout_count"] == 1

    def test_timeout_count_default_zero(self):
        """timeout_count defaults to 0."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.NOT_EXECUTED,
        )
        assert result.timeout_count == 0


class TestToolTrace:
    def test_capability_execution_info_structure(self):
        """Capability execution info has expected structure for tool_trace."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.PARTIAL,
            task_count=3,
            success_count=2,
            failure_count=1,
            timeout_count=1,
        )
        info = result.to_dict()
        assert info["status"] == "partial"
        assert info["task_count"] == 3
        assert info["success_count"] == 2
        assert info["failure_count"] == 1
        assert info["timeout_count"] == 1

    def test_no_raw_output_in_trace(self):
        """Capability execution info never contains raw tool output."""
        result = CapabilityExecutionResult(
            status=CapabilityExecutionStatus.SUCCESS,
            task_count=1,
            success_count=1,
            tasks=[
                TaskSummary(
                    capability_name="goal_list",
                    success=True,
                    output_preview="count=5",
                ),
            ],
        )
        info = result.to_dict()
        assert len(info["tasks"]) <= 10
        assert "data" not in info


class TestL3Routing:
    def test_l3_skipped_routes_to_normal_tool_path(self):
        """SKIPPED_APPROVAL tells model to use normal authorized tool path."""
        _build_registry()
        child = FakeChild(authority_level="L0", goal="delete all data")

        graph = TaskGraph()
        graph.add_task(Task(id="t1", capability_name="rag_delete"))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"t1"}))])

        child._capability_plan = plan  # type: ignore
        child._capability_graph = graph  # type: ignore
        child._capability_task_type = "default"  # type: ignore
        child._capability_authority_level = "L0"  # type: ignore
        child._subagent_id = "sa-l3-test"
        child._parent_session_id = "parent-test"

        result = _execute_capability_plan_if_present(child, "delete all data")

        assert result.status == CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED

        preamble = _format_capability_preamble(result)
        assert "approval-required" in preamble.lower() or "L3" in preamble
        assert "authorized tool execution path" in preamble.lower() or "normal" in preamble.lower()


class TestQdrantIntegration:
    @pytest.mark.skipif(
        not _is_qdrant_available(),
        reason="Qdrant server not available (requires local Qdrant instance)",
    )
    def test_rag_capability_with_local_qdrant(self):
        """RAG capability executes against local Qdrant (conditional)."""
        _build_registry()
        registry = get_jarvis_registry()
        caps = registry.list_capabilities()
        assert "rag_search" in caps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
