"""Tests for Phase 23C — Runtime Lifecycle Integration."""

import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    SubagentExecutionContext,
    SubagentLifecycleManager,
    create_execution_context,
    L0_READ_ONLY,
    L1_REVERSIBLE,
    L2_GATED,
    L3_APPROVAL,
)
from tools.delegate_tool import (
    register_lifecycle_manager,
    unregister_lifecycle_manager,
    transition_lifecycle,
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


class TestPhase23CRuntimeTransitions:
    """Test runtime lifecycle transitions in the delegation path."""

    def test_provider_creation_transitions_to_initializing(self, context):
        """When provider is created, lifecycle transitions pending -> initializing."""
        manager = SubagentLifecycleManager(context)
        # Simulate _attach_jarvis_subagent_provider creating the manager
        new_ctx = manager.transition("initializing")
        assert new_ctx.lifecycle_state == "initializing"

    def test_running_transition_before_execution(self, manager):
        """Before child execution, transition initializing -> running."""
        manager.transition("initializing")
        new_ctx = manager.transition("running")
        assert new_ctx.lifecycle_state == "running"

    def test_completed_transition_on_success(self, manager):
        """Successful child completion transitions running -> completed."""
        manager.transition("initializing")
        manager.transition("running")
        new_ctx = manager.transition("completed")
        assert new_ctx.lifecycle_state == "completed"
        assert new_ctx.is_terminal

    def test_failed_transition_on_exception(self, manager):
        """Child exception transitions running -> failed."""
        manager.transition("initializing")
        manager.transition("running")
        new_ctx = manager.transition("failed")
        assert new_ctx.lifecycle_state == "failed"
        assert new_ctx.is_terminal

    def test_cancelled_transition_on_parent_cancel(self, manager):
        """Parent cancellation transitions running -> cancelled."""
        manager.transition("initializing")
        manager.transition("running")
        new_ctx = manager.transition("cancelled")
        assert new_ctx.lifecycle_state == "cancelled"
        assert new_ctx.is_terminal

    def test_timed_out_transition_on_timeout(self, manager):
        """Timeout transitions running -> timed_out."""
        manager.transition("initializing")
        manager.transition("running")
        new_ctx = manager.transition("timed_out")
        assert new_ctx.lifecycle_state == "timed_out"
        assert new_ctx.is_terminal


class TestPhase23CLifecycleManagerRegistry:
    """Test lifecycle manager registration and lookup."""

    def test_register_and_transition(self, context):
        manager = SubagentLifecycleManager(context)
        manager.transition("initializing")
        register_lifecycle_manager("sa-test", manager)

        # Transition via the registry
        assert transition_lifecycle("sa-test", "running") is True
        assert manager.state == "running"

    def test_unregister_cleans_up(self, context):
        manager = SubagentLifecycleManager(context)
        register_lifecycle_manager("sa-test", manager)
        unregister_lifecycle_manager("sa-test")

        # Transition should fail after unregister
        assert transition_lifecycle("sa-test", "running") is False

    def test_transition_unknown_subagent(self):
        assert transition_lifecycle("unknown-sa", "running") is False


class TestPhase23CTerminalStateProtection:
    """Test that terminal states cannot transition further."""

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


class TestPhase23CContextIntegrity:
    """Test that lifecycle transitions don't modify immutable fields."""

    def test_task_type_unchanged(self, manager):
        original = manager.context.task_type
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.context.task_type == original

    def test_authority_level_unchanged(self, manager):
        original = manager.context.authority_level
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")
        assert manager.context.authority_level == original

    def test_depth_unchanged(self, manager):
        original = manager.context.delegation_depth
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.context.delegation_depth == original

    def test_allowed_tools_unchanged(self, manager):
        original = manager.context.allowed_tools
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.context.allowed_tools == original

    def test_parent_session_unchanged(self, manager):
        original = manager.context.parent_session_id
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.context.parent_session_id == original

    def test_provenance_unchanged(self, manager):
        original = manager.context.provenance_context_id
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.context.provenance_context_id == original


class TestPhase23CSecurityPreserved:
    """Test that Phase 20–22 security controls remain intact."""

    def test_provider_with_context_enforces_task_type(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )
        # Research cannot store
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        import json
        assert json.loads(result)["ok"] is False

    def test_lifecycle_transition_doesnt_affect_provider_security(self):
        """Lifecycle transitions don't weaken provider security."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")

        # Provider should still enforce task-type restrictions
        result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
        import json
        assert json.loads(result)["ok"] is False

        manager.transition("completed")

        # Security still enforced after completion
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        assert json.loads(result)["ok"] is False


class TestPhase23CChildFailureParentIntact:
    """Test that child failure doesn't corrupt parent context."""

    def test_child_failure_preserves_parent(self, context):
        manager = SubagentLifecycleManager(context)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")

        # Parent's context fields are preserved
        assert context.parent_session_id == "parent"
        assert context.task_type == "research"
        assert context.authority_level == "L0"
        assert context.allowed_tools == L0_READ_ONLY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
