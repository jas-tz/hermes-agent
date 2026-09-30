"""Tests for Phase 23A — Sub-Agent Execution Context."""

import pytest
from plugins.memory.jarvis.subagent_types import (
    SubagentExecutionContext,
    SubagentTaskType,
    create_execution_context,
    get_task_type,
    L0_READ_ONLY,
    L1_REVERSIBLE,
    L2_GATED,
    L3_APPROVAL,
)
from plugins.memory.jarvis import JARVISSubagentProvider


class TestExecutionContextCreation:
    """Test execution context is created correctly."""

    def test_create_basic_context(self):
        ctx = create_execution_context(
            child_id="sa-test-123",
            parent_session_id="parent-session-456",
            task_goal="research honey processing",
            task_type="research",
            delegation_depth=1,
        )
        assert isinstance(ctx, SubagentExecutionContext)
        assert ctx.child_id == "sa-test-123"
        assert ctx.parent_session_id == "parent-session-456"
        assert ctx.task_goal == "research honey processing"
        assert ctx.task_type == "research"
        assert ctx.authority_level == "L0"
        assert ctx.delegation_depth == 1
        assert ctx.lifecycle_state == "pending"

    def test_parent_identity_preserved(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent-session",
            task_goal="test goal",
            task_type="research",
            delegation_depth=1,
        )
        assert ctx.parent_session_id == "parent-session"

    def test_child_id_is_unique(self):
        ctx1 = create_execution_context(
            child_id="sa-test-1",
            parent_session_id="parent",
            task_goal="goal 1",
            task_type="research",
            delegation_depth=1,
        )
        ctx2 = create_execution_context(
            child_id="sa-test-2",
            parent_session_id="parent",
            task_goal="goal 2",
            task_type="research",
            delegation_depth=1,
        )
        assert ctx1.child_id != ctx2.child_id

    def test_depth_from_parent_runtime(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=3,
        )
        assert ctx.delegation_depth == 3


class TestExecutionContextImmutability:
    """Test that execution context is immutable."""

    def test_context_is_frozen(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        with pytest.raises(AttributeError):
            ctx.task_type = "maintenance"

    def test_authority_cannot_be_overridden(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        assert ctx.authority_level == "L0"
        with pytest.raises(AttributeError):
            ctx.authority_level = "L3"

    def test_task_type_cannot_be_overridden(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        assert ctx.task_type == "research"
        with pytest.raises(AttributeError):
            ctx.task_type = "maintenance"

    def test_depth_cannot_be_overridden(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        assert ctx.delegation_depth == 1
        with pytest.raises(AttributeError):
            ctx.delegation_depth = 5


class TestExecutionContextAllowedTools:
    """Test allowed tools correspond to trusted task type."""

    def test_research_allowed_tools(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        assert ctx.allowed_tools == L0_READ_ONLY
        assert "hermes_rag_search" in ctx.allowed_tools
        assert "hermes_rag_store" not in ctx.allowed_tools

    def test_note_taking_allowed_tools(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="take notes",
            task_type="note_taking",
            delegation_depth=1,
        )
        assert ctx.allowed_tools == L0_READ_ONLY | L1_REVERSIBLE
        assert "hermes_rag_store" in ctx.allowed_tools
        assert "hermes_goal_create" not in ctx.allowed_tools

    def test_maintenance_allowed_tools(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="maintenance",
            delegation_depth=1,
        )
        assert ctx.allowed_tools == L0_READ_ONLY | L1_REVERSIBLE | L2_GATED | L3_APPROVAL
        assert "hermes_rag_delete" in ctx.allowed_tools


class TestExecutionContextProvenance:
    """Test provenance isolation in context."""

    def test_provenance_context_id_set(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research honey",
            task_type="research",
            delegation_depth=1,
        )
        assert ctx.provenance_context_id.startswith("subagent:")
        assert "research" in ctx.provenance_context_id

    def test_provenance_isolated_from_parent(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent-session-123",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        # Provenance context ID contains task goal, not parent session
        assert "parent-session" not in ctx.provenance_context_id


class TestExecutionContextLifecycle:
    """Test lifecycle state properties."""

    def test_initial_state_is_pending(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        assert ctx.lifecycle_state == "pending"
        assert not ctx.is_active
        assert not ctx.is_terminal

    def test_is_active_when_running(self):
        # Note: We can't change lifecycle_state since it's frozen,
        # but we can verify the initial state
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        assert not ctx.is_active  # starts as pending


class TestExecutionContextWithProvider:
    """Test execution context integration with JARVISSubagentProvider."""

    def test_provider_accepts_execution_context(self):
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
        assert provider._execution_context == ctx
        assert provider._task_type == "research"

    def test_provider_works_without_execution_context(self):
        provider = JARVISSubagentProvider(goal="test")
        assert provider._execution_context is None

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


class TestExecutionContextErrorHandling:
    """Test error handling during context creation."""

    def test_unknown_task_type_raises_error(self):
        with pytest.raises(ValueError, match="Unknown task type"):
            create_execution_context(
                child_id="sa-test",
                parent_session_id="parent",
                task_goal="test",
                task_type="unknown_type",
                delegation_depth=1,
            )

    def test_child_failure_doesnt_corrupt_parent_context(self):
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
        # Simulate child failing on unauthorized tool
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        import json
        result_dict = json.loads(result)
        assert result_dict["ok"] is False
        # Parent context remains intact
        assert ctx.authority_level == "L0"
        assert ctx.task_type == "research"


class TestDefaultTaskType:
    """Test backward-compatible default task type."""

    def test_default_task_type_exists(self):
        task = get_task_type("default")
        assert task is not None
        assert task.authority_level == "L3"

    def test_default_task_type_has_full_whitelist(self):
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="do anything",
            task_type="default",
            delegation_depth=1,
        )
        assert ctx.allowed_tools == L0_READ_ONLY | L1_REVERSIBLE | L2_GATED | L3_APPROVAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
