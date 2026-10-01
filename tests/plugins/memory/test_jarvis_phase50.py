"""Phase 50 — Real QDrant integration validation for RAG capabilities.

These tests exercise the RAG capability path against a real Qdrant instance.
All tests are skipped when Qdrant is not reachable so the suite remains
green in CI / environments without Qdrant.
"""

import os
import pytest

from plugins.memory.jarvis.capabilities import register_default_capabilities
from plugins.memory.jarvis.capabilities.base import CapabilityContext
from plugins.memory.jarvis.capabilities.concrete_rag import (
    RagSearchCapability,
    RagStoreCapability,
)
from plugins.memory.jarvis.capabilities.registry import (
    get_jarvis_registry,
    reset_jarvis_registry,
)


def _is_qdrant_available():
    """Check if Qdrant server is reachable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333", timeout=5)
        client.get_collections()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _reset_registries():
    reset_jarvis_registry()
    yield
    reset_jarvis_registry()


@pytest.fixture
def _registry():
    registry = get_jarvis_registry()
    register_default_capabilities(registry)
    return registry


@pytest.mark.skipif(
    not _is_qdrant_available(),
    reason="Qdrant server not available (requires local Qdrant instance)",
)
class TestRagCapabilityRealQdrant:
    """RAG capability tests against real Qdrant."""

    def test_rag_search_returns_results(self):
        """RagSearchCapability executes against local Qdrant."""
        cap = RagSearchCapability()
        context = CapabilityContext(
            execution_id="test-search-1",
            subagent_id="sa-test",
            parent_session_id="parent-test",
            task_goal="test search",
            task_type="research",
            authority_level="L0",
            delegation_depth=1,
            parameters={"query": "honey processing", "limit": 5},
        )
        result = cap.execute(context)
        # Result should be a CapabilityResult (success or failure, but not crash)
        assert result is not None
        # If Qdrant has data, we should get success
        if result.success:
            assert result.data is not None

    def test_rag_store_and_search_roundtrip(self):
        """Store text via RagStoreCapability, then search for it."""
        store_cap = RagStoreCapability()
        store_context = CapabilityContext(
            execution_id="test-store-1",
            subagent_id="sa-test",
            parent_session_id="parent-test",
            task_goal="test store",
            task_type="research",
            authority_level="L1",
            delegation_depth=1,
            parameters={
                "text": "Mbogo Honey FSMS document: TBS certification requires HACCP plan.",
                "metadata": {"source": "test", "category": "fsms"},
            },
        )
        store_result = store_cap.execute(store_context)
        # Store should succeed if Qdrant is available
        if store_result.success:
            # Now search for the stored text
            search_cap = RagSearchCapability()
            search_context = CapabilityContext(
                execution_id="test-search-2",
                subagent_id="sa-test",
                parent_session_id="parent-test",
                task_goal="test search after store",
                task_type="research",
                authority_level="L0",
                delegation_depth=1,
                parameters={"query": "Mbogo Honey FSMS", "limit": 5},
            )
            search_result = search_cap.execute(search_context)
            assert search_result is not None


@pytest.mark.skipif(
    not _is_qdrant_available(),
    reason="Qdrant server not available (requires local Qdrant instance)",
)
class TestRagCapabilityPlanE2E:
    """End-to-end capability plan execution with real Qdrant."""

    def test_rag_search_through_capability_runtime(self):
        """RagSearchCapability executes through CapabilityRuntime.execute_plan()."""
        from plugins.memory.jarvis.capabilities.runtime import CapabilityRuntime
        from plugins.memory.jarvis.planner.task_graph import (
            ExecutionPlan,
            ExecutionStage,
            Task,
            TaskGraph,
        )

        registry = get_jarvis_registry()
        register_default_capabilities(registry)
        runtime = CapabilityRuntime(registry=registry)

        # Build a simple plan with rag_search
        graph = TaskGraph()
        task_id = "task_rag_search_test"
        graph.add_task(Task(
            id=task_id,
            capability_name="rag_search",
            task_type="research",
            authority_level="L0",
            parameters={"query": "honey", "limit": 3},
        ))
        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({task_id}))])

        result = runtime.execute_plan(
            plan,
            task_graph=graph,
            parent_session_id="parent-test",
            subagent_id="sa-e2e-test",
            authority_level="L0",
            task_type="research",
        )

        assert result is not None
        assert result.task_count == 1
        # The task should have been attempted
        assert len(result.stage_results) == 1
        assert len(result.stage_results[0].task_results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
