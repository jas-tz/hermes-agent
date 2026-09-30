"""Phase 37 — Concrete JARVIS Capabilities tests."""

import json
import pytest
from unittest.mock import patch

from plugins.memory.jarvis.capabilities.base import (
    CapabilityContext,
    get_registry,
    reset_registry,
)
from plugins.memory.jarvis.capabilities.registry import (
    JARVISCapabilityRegistry,
    CapabilityRegistrationError,
    get_jarvis_registry,
    reset_jarvis_registry,
)
from plugins.memory.jarvis.capabilities.concrete_rag import (
    RagSearchCapability,
    RagStoreCapability,
    RagDeleteCapability,
    RagRetrieveLearningsCapability,
)
from plugins.memory.jarvis.capabilities.concrete_continuity import (
    GoalListCapability,
    GoalCreateCapability,
    TaskListCapability,
    TaskCreateCapability,
    DecisionListCapability,
    DecisionRecordCapability,
    GoalCompleteCapability,
    TaskCompleteCapability,
)
from plugins.memory.jarvis.capabilities import (
    register_default_capabilities,
)


@pytest.fixture(autouse=True)
def _reset_registries():
    reset_registry()
    reset_jarvis_registry()
    yield
    reset_registry()
    reset_jarvis_registry()


def _make_context(execution_id="exec-001", parameters=None, authority_level="L0", task_type="research"):
    return CapabilityContext(
        execution_id=execution_id,
        subagent_id="sa-001",
        parent_session_id="parent-001",
        task_goal="Test goal",
        task_type=task_type,
        authority_level=authority_level,
        delegation_depth=1,
        parameters=parameters or {},
    )


# ---------------------------------------------------------------------------
# Concrete RAG capabilities
# ---------------------------------------------------------------------------

class TestRagSearchCapability:
    def test_get_name(self):
        assert RagSearchCapability().get_name() == "rag_search"

    def test_requirements(self):
        reqs = RagSearchCapability().get_requirements()
        assert reqs.min_authority_level == "L0"
        assert "hermes_rag_search" in reqs.required_tools
        assert reqs.input_type.value == "query"

    def test_validate_empty_query(self):
        cap = RagSearchCapability()
        ctx = _make_context(parameters={})
        valid, reason = cap.validate(ctx)
        assert valid is False
        assert "query" in reason.lower()

    def test_validate_with_query(self):
        cap = RagSearchCapability()
        ctx = _make_context(parameters={"query": "test"})
        valid, reason = cap.validate(ctx)
        assert valid is True

    @patch("plugins.memory.jarvis.capabilities.concrete_rag._dispatch")
    def test_execute_success(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "results": [{"id": "1", "text": "hello"}]}
        cap = RagSearchCapability()
        ctx = _make_context(parameters={"query": "test"})
        result = cap.execute(ctx)
        assert result.success is True
        assert result.data["results"][0]["text"] == "hello"
        assert result.execution_id == "exec-001"
        assert result.capability_name == "rag_search"
        mock_dispatch.assert_called_once_with("hermes_rag_search", {"query": "test", "limit": 5})

    @patch("plugins.memory.jarvis.capabilities.concrete_rag._dispatch")
    def test_execute_failure(self, mock_dispatch):
        err_msg = "Qdrant unavailable"
        mock_dispatch.return_value = {"ok": False, "error": err_msg}
        cap = RagSearchCapability()
        ctx = _make_context(parameters={"query": "test"})
        result = cap.execute(ctx)
        assert result.success is False
        assert err_msg in (result.error or "")


class TestRagStoreCapability:
    def test_requirements(self):
        reqs = RagStoreCapability().get_requirements()
        assert reqs.min_authority_level == "L1"
        assert "hermes_rag_store" in reqs.required_tools

    def test_validate_empty_text(self):
        cap = RagStoreCapability()
        ctx = _make_context(parameters={})
        valid, reason = cap.validate(ctx)
        assert valid is False

    @patch("plugins.memory.jarvis.capabilities.concrete_rag._dispatch")
    def test_execute_success(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "memory_id": "abc-123"}
        cap = RagStoreCapability()
        ctx = _make_context(parameters={"text": "important fact"})
        result = cap.execute(ctx)
        assert result.success is True
        assert result.data["memory_id"] == "abc-123"


class TestRagDeleteCapability:
    def test_requirements(self):
        reqs = RagDeleteCapability().get_requirements()
        assert reqs.min_authority_level == "L3"
        assert reqs.requires_approval is True
        assert "hermes_rag_delete" in reqs.required_tools

    def test_validate_empty_id(self):
        cap = RagDeleteCapability()
        ctx = _make_context(parameters={})
        valid, reason = cap.validate(ctx)
        assert valid is False

    @patch("plugins.memory.jarvis.capabilities.concrete_rag._dispatch")
    def test_execute_success(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "deleted": "abc"}
        cap = RagDeleteCapability()
        ctx = _make_context(parameters={"memory_id": "abc"})
        result = cap.execute(ctx)
        assert result.success is True


class TestRagRetrieveLearningsCapability:
    def test_requirements(self):
        reqs = RagRetrieveLearningsCapability().get_requirements()
        assert reqs.min_authority_level == "L0"
        assert "hermes_rag_retrieve_learnings" in reqs.required_tools

    @patch("plugins.memory.jarvis.capabilities.concrete_rag._dispatch")
    def test_execute(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "results": []}
        cap = RagRetrieveLearningsCapability()
        ctx = _make_context(parameters={"query": "learnings"})
        result = cap.execute(ctx)
        assert result.success is True


# ---------------------------------------------------------------------------
# Concrete Continuity capabilities
# ---------------------------------------------------------------------------

class TestGoalListCapability:
    def test_requirements(self):
        reqs = GoalListCapability().get_requirements()
        assert reqs.min_authority_level == "L0"
        assert "hermes_goal_list" in reqs.required_tools

    def test_validate_always_true(self):
        cap = GoalListCapability()
        ctx = _make_context()
        valid, reason = cap.validate(ctx)
        assert valid is True

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "goals": [], "count": 0}
        cap = GoalListCapability()
        ctx = _make_context()
        result = cap.execute(ctx)
        assert result.success is True
        mock_dispatch.assert_called_once()


class TestGoalCreateCapability:
    def test_requirements(self):
        reqs = GoalCreateCapability().get_requirements()
        assert reqs.min_authority_level == "L2"
        assert "hermes_goal_create" in reqs.required_tools

    def test_validate_empty_title(self):
        cap = GoalCreateCapability()
        ctx = _make_context(parameters={})
        valid, reason = cap.validate(ctx)
        assert valid is False

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "goal": {"id": "g1", "title": "Test goal"}}
        cap = GoalCreateCapability()
        ctx = _make_context(parameters={"title": "Test goal"})
        result = cap.execute(ctx)
        assert result.success is True
        assert result.data["goal"]["title"] == "Test goal"


class TestTaskListCapability:
    def test_requirements(self):
        reqs = TaskListCapability().get_requirements()
        assert reqs.min_authority_level == "L0"
        assert "hermes_task_list" in reqs.required_tools

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "tasks": [], "count": 0}
        cap = TaskListCapability()
        ctx = _make_context()
        result = cap.execute(ctx)
        assert result.success is True


class TestTaskCreateCapability:
    def test_requirements(self):
        reqs = TaskCreateCapability().get_requirements()
        assert reqs.min_authority_level == "L2"
        assert "hermes_task_create" in reqs.required_tools

    def test_validate_empty_title(self):
        cap = TaskCreateCapability()
        ctx = _make_context(parameters={})
        valid, reason = cap.validate(ctx)
        assert valid is False

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "task": {"id": "t1", "title": "Test"}}
        cap = TaskCreateCapability()
        ctx = _make_context(parameters={"title": "Test task"})
        result = cap.execute(ctx)
        assert result.success is True


class TestDecisionListCapability:
    def test_requirements(self):
        reqs = DecisionListCapability().get_requirements()
        assert reqs.min_authority_level == "L0"
        assert "hermes_decision_list" in reqs.required_tools

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "decisions": [], "count": 0}
        cap = DecisionListCapability()
        ctx = _make_context()
        result = cap.execute(ctx)
        assert result.success is True


class TestDecisionRecordCapability:
    def test_requirements(self):
        reqs = DecisionRecordCapability().get_requirements()
        assert reqs.min_authority_level == "L2"
        assert "hermes_decision_record" in reqs.required_tools

    def test_validate_empty_decision(self):
        cap = DecisionRecordCapability()
        ctx = _make_context(parameters={})
        valid, reason = cap.validate(ctx)
        assert valid is False

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "decision": {"id": "d1"}}
        cap = DecisionRecordCapability()
        ctx = _make_context(parameters={"decision": "Use pytest"})
        result = cap.execute(ctx)
        assert result.success is True


class TestGoalCompleteCapability:
    def test_requirements(self):
        reqs = GoalCompleteCapability().get_requirements()
        assert reqs.min_authority_level == "L1"
        assert "hermes_goal_complete" in reqs.required_tools

    def test_validate_empty_id(self):
        cap = GoalCompleteCapability()
        ctx = _make_context(parameters={})
        valid, reason = cap.validate(ctx)
        assert valid is False

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "goal": {"id": "g1", "status": "done"}}
        cap = GoalCompleteCapability()
        ctx = _make_context(parameters={"goal_id": "g1"})
        result = cap.execute(ctx)
        assert result.success is True


class TestTaskCompleteCapability:
    def test_requirements(self):
        reqs = TaskCompleteCapability().get_requirements()
        assert reqs.min_authority_level == "L1"
        assert "hermes_task_complete" in reqs.required_tools

    def test_validate_empty_id(self):
        cap = TaskCompleteCapability()
        ctx = _make_context(parameters={})
        valid, reason = cap.validate(ctx)
        assert valid is False

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "task": {"id": "t1", "status": "done"}}
        cap = TaskCompleteCapability()
        ctx = _make_context(parameters={"task_id": "t1"})
        result = cap.execute(ctx)
        assert result.success is True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegisterDefaultCapabilities:
    def test_register_all(self):
        registry = JARVISCapabilityRegistry()
        register_default_capabilities(registry)
        names = registry.list_capabilities()
        assert "rag_search" in names
        assert "rag_store" in names
        assert "rag_delete" in names
        assert "rag_retrieve_learnings" in names
        assert "goal_list" in names
        assert "goal_create" in names
        assert "goal_update" in names
        assert "goal_complete" in names
        assert "task_list" in names
        assert "task_create" in names
        assert "task_update" in names
        assert "task_complete" in names
        assert "decision_list" in names
        assert "decision_record" in names
        assert "decision_supersede" in names
        assert len(names) == 15

    def test_register_duplicate_raises(self):
        registry = JARVISCapabilityRegistry()
        register_default_capabilities(registry)
        with pytest.raises(CapabilityRegistrationError, match="already registered"):
            registry.register(RagSearchCapability)

    def test_all_registered_pass_validation(self):
        registry = JARVISCapabilityRegistry()
        register_default_capabilities(registry)
        for name in registry.list_capabilities():
            cap = registry.get_capability(name)
            assert cap is not None
            # Each capability's requirements should be valid
            reqs = cap.get_requirements()
            assert reqs.min_authority_level in ("L0", "L1", "L2", "L3")
            for tool in reqs.required_tools:
                assert tool in [
                    "hermes_rag_search", "hermes_rag_store", "hermes_rag_delete",
                    "hermes_rag_retrieve_learnings",
                    "hermes_goal_list", "hermes_goal_create", "hermes_goal_update",
                    "hermes_goal_complete",
                    "hermes_task_list", "hermes_task_create", "hermes_task_update",
                    "hermes_task_complete",
                    "hermes_decision_list", "hermes_decision_record",
                    "hermes_decision_supersede",
                ]


# ---------------------------------------------------------------------------
# Integration: registry + planner with concrete capabilities
# ---------------------------------------------------------------------------

class TestPlannerWithConcreteCapabilities:
    def test_plan_with_concrete_caps(self):
        from plugins.memory.jarvis.planner.sub_agent_planner import SubAgentPlanner
        registry = get_jarvis_registry()
        register_default_capabilities(registry)
        planner = SubAgentPlanner(registry=registry)
        plan = planner.plan("research something", task_type="research", authority_level="L0")
        # Should have at least one stage with L0 capabilities
        assert len(plan.stages) >= 1
        all_ids = plan.all_task_ids
        # All planned tasks should reference registered capabilities
        registered = registry.list_capabilities()
        for tid in all_ids:
            # Task IDs are "task_<capname>_<uuid>" — extract capability name
            # Remove prefix "task_" and suffix "_<uuid>"
            inner = tid[len("task_"):]
            # UUID is 6 chars at end; capability name has no underscores... but ours do
            # Match by checking if any registered capability name is a prefix
            matched = any(inner.startswith(name + "_") for name in registered)
            assert matched, f"Task {tid} does not match any registered capability"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
