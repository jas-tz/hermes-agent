"""Tests for Phase 23B — Sub-Agent Lifecycle State Machine."""

import pytest
from plugins.memory.jarvis.subagent_types import (
    SubagentExecutionContext,
    SubagentLifecycleManager,
    LIFECYCLE_STATES,
    TERMINAL_STATES,
    create_execution_context,
    is_valid_transition,
)


@pytest.fixture
def context():
    return create_execution_context(
        child_id="sa-test",
        parent_session_id="parent",
        task_goal="research honey",
        task_type="research",
        delegation_depth=1,
    )


@pytest.fixture
def manager(context):
    return SubagentLifecycleManager(context)


class TestInitialState:
    """Test initial lifecycle state."""

    def test_initial_state_is_pending(self, context):
        assert context.lifecycle_state == "pending"

    def test_lifecycle_states_set(self):
        expected = {"pending", "initializing", "running", "completed", "failed", "cancelled", "timed_out"}
        assert LIFECYCLE_STATES == expected

    def test_terminal_states_set(self):
        expected = {"completed", "failed", "cancelled", "timed_out"}
        assert TERMINAL_STATES == expected


class TestValidTransitions:
    """Test every valid lifecycle transition."""

    def test_pending_to_initializing(self, manager):
        new_ctx = manager.transition("initializing")
        assert new_ctx.lifecycle_state == "initializing"

    def test_pending_to_cancelled(self, manager):
        new_ctx = manager.transition("cancelled")
        assert new_ctx.lifecycle_state == "cancelled"

    def test_initializing_to_running(self, manager):
        manager.transition("initializing")
        new_ctx = manager.transition("running")
        assert new_ctx.lifecycle_state == "running"

    def test_initializing_to_failed(self, manager):
        manager.transition("initializing")
        new_ctx = manager.transition("failed")
        assert new_ctx.lifecycle_state == "failed"

    def test_initializing_to_cancelled(self, manager):
        manager.transition("initializing")
        new_ctx = manager.transition("cancelled")
        assert new_ctx.lifecycle_state == "cancelled"

    def test_running_to_completed(self, manager):
        manager.transition("initializing")
        manager.transition("running")
        new_ctx = manager.transition("completed")
        assert new_ctx.lifecycle_state == "completed"

    def test_running_to_failed(self, manager):
        manager.transition("initializing")
        manager.transition("running")
        new_ctx = manager.transition("failed")
        assert new_ctx.lifecycle_state == "failed"

    def test_running_to_cancelled(self, manager):
        manager.transition("initializing")
        manager.transition("running")
        new_ctx = manager.transition("cancelled")
        assert new_ctx.lifecycle_state == "cancelled"

    def test_running_to_timed_out(self, manager):
        manager.transition("initializing")
        manager.transition("running")
        new_ctx = manager.transition("timed_out")
        assert new_ctx.lifecycle_state == "timed_out"


class TestInvalidTransitions:
    """Test that invalid transitions are rejected."""

    def test_pending_cannot_go_to_running(self, manager):
        with pytest.raises(ValueError, match="Invalid lifecycle transition"):
            manager.transition("running")

    def test_pending_cannot_go_to_completed(self, manager):
        with pytest.raises(ValueError, match="Invalid lifecycle transition"):
            manager.transition("completed")

    def test_pending_cannot_go_to_failed(self, manager):
        with pytest.raises(ValueError, match="Invalid lifecycle transition"):
            manager.transition("failed")

    def test_running_cannot_go_to_initializing(self, manager):
        manager.transition("initializing")
        manager.transition("running")
        with pytest.raises(ValueError, match="Invalid lifecycle transition"):
            manager.transition("initializing")

    def test_initializing_cannot_go_to_timed_out(self, manager):
        manager.transition("initializing")
        with pytest.raises(ValueError, match="Invalid lifecycle transition"):
            manager.transition("timed_out")


class TestTerminalStateProtection:
    """Test that terminal states cannot transition further."""

    def test_completed_is_terminal(self, manager):
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.is_terminal

    def test_completed_cannot_transition(self, manager):
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        with pytest.raises(ValueError, match="Cannot transition from terminal state"):
            manager.transition("running")

    def test_failed_cannot_transition(self, manager):
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")
        with pytest.raises(ValueError, match="Cannot transition from terminal state"):
            manager.transition("completed")

    def test_cancelled_cannot_transition(self, manager):
        manager.transition("cancelled")
        with pytest.raises(ValueError, match="Cannot transition from terminal state"):
            manager.transition("running")

    def test_timed_out_cannot_transition(self, manager):
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("timed_out")
        with pytest.raises(ValueError, match="Cannot transition from terminal state"):
            manager.transition("completed")


class TestTransitionPreservesContextIntegrity:
    """Test that transitions don't modify immutable fields."""

    def test_task_type_unchanged(self, manager):
        original_task_type = manager.context.task_type
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.context.task_type == original_task_type

    def test_authority_level_unchanged(self, manager):
        original_authority = manager.context.authority_level
        manager.transition("initializing")
        manager.transition("running")
        assert manager.context.authority_level == original_authority

    def test_delegation_depth_unchanged(self, manager):
        original_depth = manager.context.delegation_depth
        manager.transition("initializing")
        assert manager.context.delegation_depth == original_depth

    def test_allowed_tools_unchanged(self, manager):
        original_tools = manager.context.allowed_tools
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.context.allowed_tools == original_tools

    def test_parent_session_unchanged(self, manager):
        original_parent = manager.context.parent_session_id
        manager.transition("initializing")
        manager.transition("running")
        assert manager.context.parent_session_id == original_parent

    def test_provenance_context_unchanged(self, manager):
        original_provenance = manager.context.provenance_context_id
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.context.provenance_context_id == original_provenance


class TestIsolatedTransitionObjects:
    """Test that transitions produce new immutable context objects."""

    def test_transition_returns_new_object(self, manager):
        ctx1 = manager.context
        ctx2 = manager.transition("initializing")
        assert ctx1 is not ctx2
        assert ctx1.lifecycle_state == "pending"
        assert ctx2.lifecycle_state == "initializing"

    def test_original_context_unchanged(self, manager):
        original = manager.context
        manager.transition("initializing")
        # The manager's internal reference is updated, but the original object is immutable
        assert original.lifecycle_state == "pending"


class TestManagerProperties:
    """Test SubagentLifecycleManager properties."""

    def test_state_property(self, manager):
        assert manager.state == "pending"
        manager.transition("initializing")
        assert manager.state == "initializing"

    def test_is_terminal_property(self, manager):
        assert not manager.is_terminal
        manager.transition("cancelled")
        assert manager.is_terminal


class TestIsolatedManagerPerChild:
    """Test that each child has its own isolated lifecycle manager."""

    def test_separate_managers(self, context):
        ctx2 = create_execution_context(
            child_id="sa-other",
            parent_session_id="parent",
            task_goal="other goal",
            task_type="research",
            delegation_depth=1,
        )
        m1 = SubagentLifecycleManager(context)
        m2 = SubagentLifecycleManager(ctx2)

        m1.transition("initializing")
        assert m1.state == "initializing"
        assert m2.state == "pending"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
