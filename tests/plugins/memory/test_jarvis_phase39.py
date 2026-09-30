"""Phase 39 — Mutable Continuity Capabilities tests."""

import pytest
from unittest.mock import patch

from plugins.memory.jarvis.capabilities.base import (
    CapabilityContext,
    get_registry,
    reset_registry,
)
from plugins.memory.jarvis.capabilities.registry import (
    JARVISCapabilityRegistry,
    get_jarvis_registry,
    reset_jarvis_registry,
)
from plugins.memory.jarvis.capabilities.concrete_continuity import (
    GoalUpdateCapability,
    TaskUpdateCapability,
    DecisionSupersedeCapability,
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
# GoalUpdateCapability
# ---------------------------------------------------------------------------

class TestGoalUpdateCapability:
    def test_requirements(self):
        cap = GoalUpdateCapability()
        reqs = cap.get_requirements()
        assert reqs.min_authority_level == "L2"
        assert "hermes_goal_update" in reqs.required_tools
        assert cap.get_category().value == "data_storage"

    def test_validate_missing_goal_id(self):
        cap = GoalUpdateCapability()
        ctx = _make_context(parameters={"title": "New title"})
        valid, reason = cap.validate(ctx)
        assert valid is False
        assert "goal_id" in reason.lower()

    def test_validate_no_updatable_fields(self):
        cap = GoalUpdateCapability()
        ctx = _make_context(parameters={"goal_id": "g1"})
        valid, reason = cap.validate(ctx)
        assert valid is False
        assert "updatable field" in reason.lower()

    def test_validate_title_only(self):
        cap = GoalUpdateCapability()
        ctx = _make_context(parameters={"goal_id": "g1", "title": "New title"})
        valid, reason = cap.validate(ctx)
        assert valid is True

    def test_validate_status_only(self):
        cap = GoalUpdateCapability()
        ctx = _make_context(parameters={"goal_id": "g1", "status": "blocked"})
        valid, reason = cap.validate(ctx)
        assert valid is True

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute_title_update(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "goal": {"id": "g1", "title": "New title"}}
        cap = GoalUpdateCapability()
        ctx = _make_context(parameters={"goal_id": "g1", "title": "New title"})
        result = cap.execute(ctx)
        assert result.success is True
        assert result.data["goal"]["title"] == "New title"
        mock_dispatch.assert_called_once_with("hermes_goal_update", {"goal_id": "g1", "title": "New title"})

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute_status_only(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "goal": {"id": "g1", "status": "done"}}
        cap = GoalUpdateCapability()
        ctx = _make_context(parameters={"goal_id": "g1", "status": "done"})
        result = cap.execute(ctx)
        assert result.success is True
        mock_dispatch.assert_called_once_with("hermes_goal_update", {"goal_id": "g1", "status": "done"})

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute_failure(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": False, "error": "no such goal: bad-id"}
        cap = GoalUpdateCapability()
        ctx = _make_context(parameters={"goal_id": "bad-id", "title": "X"})
        result = cap.execute(ctx)
        assert result.success is False
        assert "no such goal" in (result.error or "")

    def test_registration(self):
        registry = JARVISCapabilityRegistry()
        registry.register(GoalUpdateCapability)
        assert "goal_update" in registry.list_capabilities()


# ---------------------------------------------------------------------------
# TaskUpdateCapability
# ---------------------------------------------------------------------------

class TestTaskUpdateCapability:
    def test_requirements(self):
        reqs = TaskUpdateCapability().get_requirements()
        assert reqs.min_authority_level == "L2"
        assert "hermes_task_update" in reqs.required_tools

    def test_validate_missing_task_id(self):
        cap = TaskUpdateCapability()
        ctx = _make_context(parameters={"title": "New title"})
        valid, reason = cap.validate(ctx)
        assert valid is False
        assert "task_id" in reason.lower()

    def test_validate_no_updatable_fields(self):
        cap = TaskUpdateCapability()
        ctx = _make_context(parameters={"task_id": "t1"})
        valid, reason = cap.validate(ctx)
        assert valid is False
        assert "updatable field" in reason.lower()

    def test_validate_deadline_only(self):
        cap = TaskUpdateCapability()
        ctx = _make_context(parameters={"task_id": "t1", "deadline_at": "2026-12-31"})
        valid, reason = cap.validate(ctx)
        assert valid is True

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute_title_update(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "task": {"id": "t1", "title": "Updated"}}
        cap = TaskUpdateCapability()
        ctx = _make_context(parameters={"task_id": "t1", "title": "Updated"})
        result = cap.execute(ctx)
        assert result.success is True
        mock_dispatch.assert_called_once_with("hermes_task_update", {"task_id": "t1", "title": "Updated"})

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute_status_only(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "task": {"id": "t1", "status": "done"}}
        cap = TaskUpdateCapability()
        ctx = _make_context(parameters={"task_id": "t1", "status": "done"})
        result = cap.execute(ctx)
        assert result.success is True
        mock_dispatch.assert_called_once_with("hermes_task_update", {"task_id": "t1", "status": "done"})

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute_failure(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": False, "error": "no such task: bad-id"}
        cap = TaskUpdateCapability()
        ctx = _make_context(parameters={"task_id": "bad-id", "title": "X"})
        result = cap.execute(ctx)
        assert result.success is False

    def test_registration(self):
        registry = JARVISCapabilityRegistry()
        registry.register(TaskUpdateCapability)
        assert "task_update" in registry.list_capabilities()


# ---------------------------------------------------------------------------
# DecisionSupersedeCapability
# ---------------------------------------------------------------------------

class TestDecisionSupersedeCapability:
    def test_requirements(self):
        reqs = DecisionSupersedeCapability().get_requirements()
        assert reqs.min_authority_level == "L3"
        assert reqs.requires_approval is True
        assert "hermes_decision_supersede" in reqs.required_tools
        assert reqs.parallelizable is False

    def test_validate_missing_decision_id(self):
        cap = DecisionSupersedeCapability()
        ctx = _make_context(parameters={"decision": "New decision"})
        valid, reason = cap.validate(ctx)
        assert valid is False
        assert "decision_id" in reason.lower()

    def test_validate_missing_decision(self):
        cap = DecisionSupersedeCapability()
        ctx = _make_context(parameters={"decision_id": "d1"})
        valid, reason = cap.validate(ctx)
        assert valid is False
        assert "decision" in reason.lower()

    def test_validate_with_both(self):
        cap = DecisionSupersedeCapability()
        ctx = _make_context(parameters={"decision_id": "d1", "decision": "New decision"})
        valid, reason = cap.validate(ctx)
        assert valid is True

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute_success(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": True, "decision": {"id": "d2"}}
        cap = DecisionSupersedeCapability()
        ctx = _make_context(
            parameters={"decision_id": "d1", "decision": "Revised decision", "rationale": "Better info"},
            authority_level="L3",
        )
        result = cap.execute(ctx)
        assert result.success is True
        mock_dispatch.assert_called_once_with(
            "hermes_decision_supersede",
            {"decision_id": "d1", "decision": "Revised decision", "rationale": "Better info"},
        )

    @patch("plugins.memory.jarvis.capabilities.concrete_continuity._dispatch_continuity")
    def test_execute_failure(self, mock_dispatch):
        mock_dispatch.return_value = {"ok": False, "error": "no such decision: bad-id"}
        cap = DecisionSupersedeCapability()
        ctx = _make_context(parameters={"decision_id": "bad-id", "decision": "X"})
        result = cap.execute(ctx)
        assert result.success is False
        assert "no such decision" in (result.error or "")

    def test_registration(self):
        registry = JARVISCapabilityRegistry()
        registry.register(DecisionSupersedeCapability)
        assert "decision_supersede" in registry.list_capabilities()


# ---------------------------------------------------------------------------
# Integration: all mutable caps register + planner
# ---------------------------------------------------------------------------

class TestMutableCapsRegistration:
    def test_register_all_mutable(self):
        registry = JARVISCapabilityRegistry()
        registry.register(GoalUpdateCapability)
        registry.register(TaskUpdateCapability)
        registry.register(DecisionSupersedeCapability)
        names = registry.list_capabilities()
        assert "goal_update" in names
        assert "task_update" in names
        assert "decision_supersede" in names

    def test_register_mutable_with_existing(self):
        """Mutable caps register alongside Phase 37 caps."""
        from plugins.memory.jarvis.capabilities.concrete_continuity import (
            GoalCreateCapability, GoalListCapability,
        )
        registry = JARVISCapabilityRegistry()
        registry.register(GoalCreateCapability)
        registry.register(GoalListCapability)
        registry.register(GoalUpdateCapability)
        names = registry.list_capabilities()
        assert "goal_create" in names
        assert "goal_list" in names
        assert "goal_update" in names

    def test_mutable_caps_have_correct_tools(self):
        caps = [GoalUpdateCapability(), TaskUpdateCapability(), DecisionSupersedeCapability()]
        for cap in caps:
            reqs = cap.get_requirements()
            for tool in reqs.required_tools:
                assert tool in [
                    "hermes_goal_update",
                    "hermes_task_update",
                    "hermes_decision_supersede",
                ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
