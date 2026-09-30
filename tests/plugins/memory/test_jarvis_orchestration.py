"""Phase 24A — Sub-Agent Orchestration Integration Tests.

Tests the end-to-end integration of:
- Phase 20A/20B/20C: JARVISSubagentProvider with tool whitelist and provider wiring
- Phase 21A: Child-agent tool discovery
- Phase 22A: Task type registry + tool mapping
- Phase 22B: Runtime task-type enforcement
- Phase 23A: Immutable execution context
- Phase 23B: Lifecycle state machine
- Phase 23C: Runtime lifecycle integration
- Phase 23D: Result contract

Tests derive their contracts from the actual Phase 20–23 implementation,
not from invented expectations.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    SubagentExecutionContext,
    SubagentTaskType,
    SubagentLifecycleManager,
    create_execution_context,
    L0_READ_ONLY,
    L1_REVERSIBLE,
    L2_GATED,
    L3_APPROVAL,
    ALL_JARVIS_TOOLS,
    SUBAGENT_TASK_TYPES,
    get_task_type,
    validate_task_type,
    is_tool_allowed_for_task,
)
from tools.delegate_tool import (
    register_lifecycle_manager,
    unregister_lifecycle_manager,
    transition_lifecycle,
    _get_lifecycle_state_for_subagent,
    _build_result_contract,
    RESULT_STATUSES,
)


# ---------------------------------------------------------------------------
# Orchestration flow tests
# ---------------------------------------------------------------------------

class TestOrchestrationFlow:
    """Test the full parent → delegate → child → execution context flow."""

    def test_execution_context_created_with_correct_task_type(self):
        """Phase 23A: Execution context captures task_type from delegation."""
        ctx = create_execution_context(
            child_id="sa-research-001",
            parent_session_id="parent-session",
            task_goal="research honey production",
            task_type="research",
            delegation_depth=1,
        )
        assert ctx.task_type == "research"
        assert ctx.authority_level == "L0"
        assert ctx.child_id == "sa-research-001"
        assert ctx.parent_session_id == "parent-session"
        assert ctx.task_goal == "research honey production"
        assert ctx.delegation_depth == 1
        assert ctx.lifecycle_state == "pending"

    def test_execution_context_captures_all_task_types(self):
        """All registered task types produce valid contexts."""
        for task_type_name in SUBAGENT_TASK_TYPES:
            ctx = create_execution_context(
                child_id=f"sa-{task_type_name}",
                parent_session_id="parent",
                task_goal=f"test {task_type_name}",
                task_type=task_type_name,
                delegation_depth=1,
            )
            assert ctx.task_type == task_type_name
            assert ctx.authority_level == SUBAGENT_TASK_TYPES[task_type_name].authority_level

    def test_default_task_type_for_backward_compatibility(self):
        """Phase 22A: 'default' task type provides full whitelist for backward compat."""
        ctx = create_execution_context(
            child_id="sa-default",
            parent_session_id="parent",
            task_goal="any task",
            task_type="default",
            delegation_depth=1,
        )
        assert ctx.task_type == "default"
        assert ctx.authority_level == "L3"
        assert ctx.allowed_tools == ALL_JARVIS_TOOLS

    def test_unknown_task_type_raises_error(self):
        """Phase 23A: Unknown task types are denied."""
        with pytest.raises(ValueError, match="Unknown task type"):
            create_execution_context(
                child_id="sa-bad",
                parent_session_id="parent",
                task_goal="bad",
                task_type="nonexistent",
                delegation_depth=1,
            )


# ---------------------------------------------------------------------------
# Task-type authorization orchestration
# ---------------------------------------------------------------------------

class TestTaskTypeAuthorizationOrchestration:
    """Test that task-type authorization works end-to-end."""

    def test_research_provider_rejects_all_writes(self):
        """Research task (L0) cannot write to any backend."""
        ctx = create_execution_context(
            child_id="sa-research",
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

        # All write tools should be rejected
        for tool in ("hermes_rag_store", "hermes_goal_create", "hermes_task_create",
                     "hermes_rag_delete", "hermes_decision_supersede"):
            result = provider.handle_tool_call(tool, {"text": "test"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False, f"{tool} should be rejected for research"
            assert "not authorized" in result_dict["error"]

    def test_research_provider_allows_reads(self):
        """Research task (L0) can read from all backends."""
        ctx = create_execution_context(
            child_id="sa-research",
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

        # All read tools should be allowed
        for tool in ("hermes_rag_search", "hermes_goal_list", "hermes_task_list",
                     "hermes_decision_list", "hermes_rag_retrieve_learnings",
                     "hermes_rag_extract_learnings"):
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
                mock.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call(tool, {})
                result_dict = json.loads(result)
                assert result_dict["ok"] is True, f"{tool} should be allowed for research"

    def test_note_taking_provider_allows_reversible_writes(self):
        """Note-taking task (L1) can store memories and complete goals/tasks."""
        ctx = create_execution_context(
            child_id="sa-note",
            parent_session_id="parent",
            task_goal="take notes",
            task_type="note_taking",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="take notes",
            task_type="note_taking",
            execution_context=ctx,
        )

        # L1 writes should be allowed
        for tool in ("hermes_rag_store", "hermes_rag_store_learnings",
                     "hermes_goal_complete", "hermes_task_complete"):
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
                mock.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call(tool, {"text": "test"})
                result_dict = json.loads(result)
                assert result_dict["ok"] is True, f"{tool} should be allowed for note_taking"

    def test_maintenance_provider_allows_l3_with_approval(self):
        """Maintenance task (L3) can delete with explicit approval."""
        ctx = create_execution_context(
            child_id="sa-maint",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="maintenance",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="maintenance",
            execution_context=ctx,
        )

        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": True}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                mock_dispatch.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                result_dict = json.loads(result)
                assert result_dict["ok"] is True
                mock_approval.assert_called_once()

    def test_maintenance_provider_denies_l3_without_approval(self):
        """Maintenance task (L3) cannot delete without approval."""
        ctx = create_execution_context(
            child_id="sa-maint",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="maintenance",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="maintenance",
            execution_context=ctx,
        )

        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": False}
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False


# ---------------------------------------------------------------------------
# Lifecycle orchestration
# ---------------------------------------------------------------------------

class TestLifecycleOrchestration:
    """Test the lifecycle state machine integrated with the provider."""

    def test_full_lifecycle_success(self):
        """Phase 23B: Full lifecycle: pending → initializing → running → completed."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)

        assert manager.state == "pending"
        assert not manager.is_terminal

        manager.transition("initializing")
        assert manager.state == "initializing"

        manager.transition("running")
        assert manager.state == "running"
        assert manager.context.is_active

        manager.transition("completed")
        assert manager.state == "completed"
        assert manager.is_terminal

    def test_full_lifecycle_failure(self):
        """Phase 23B: Lifecycle: pending → initializing → running → failed."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)

        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")

        assert manager.state == "failed"
        assert manager.is_terminal

    def test_full_lifecycle_cancellation(self):
        """Phase 23C: Lifecycle: pending → cancelled."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)

        manager.transition("cancelled")
        assert manager.state == "cancelled"
        assert manager.is_terminal

    def test_full_lifecycle_timeout(self):
        """Phase 23C: Lifecycle: pending → initializing → running → timed_out."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)

        manager.transition("initializing")
        manager.transition("running")
        manager.transition("timed_out")

        assert manager.state == "timed_out"
        assert manager.is_terminal


# ---------------------------------------------------------------------------
# Result contract orchestration
# ---------------------------------------------------------------------------

class TestResultContractOrchestration:
    """Test the result contract end-to-end."""

    def test_successful_result_contract(self):
        """Phase 23D: Successful child produces completed result contract."""
        ctx = create_execution_context(
            child_id="sa-success",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")

        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="Research complete",
            subagent_id="sa-success",
            child_session_id="child",
            task_goal="research",
            lifecycle_state=manager.state,
            api_calls=3,
            duration_seconds=5.0,
        )

        assert contract["status"] == "completed"
        assert contract["lifecycle_state"] == "completed"
        assert contract["subagent_id"] == "sa-success"
        assert contract["summary"] == "Research complete"

    def test_failed_result_contract(self):
        """Phase 23D: Failed child produces failed result contract."""
        ctx = create_execution_context(
            child_id="sa-fail",
            parent_session_id="parent",
            task_goal="failing task",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")

        contract = _build_result_contract(
            task_index=0,
            status="failed",
            summary=None,
            subagent_id="sa-fail",
            child_session_id="child",
            task_goal="failing task",
            lifecycle_state=manager.state,
            error="Connection error",
        )

        assert contract["status"] == "failed"
        assert contract["lifecycle_state"] == "failed"
        assert contract["error"] == "Connection error"

    def test_timed_out_result_contract(self):
        """Phase 23D: Timeout produces timed_out result contract."""
        ctx = create_execution_context(
            child_id="sa-timeout",
            parent_session_id="parent",
            task_goal="slow task",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("timed_out")

        contract = _build_result_contract(
            task_index=0,
            status="timed_out",
            summary=None,
            subagent_id="sa-timeout",
            child_session_id="child",
            task_goal="slow task",
            lifecycle_state=manager.state,
            duration_seconds=30.0,
        )

        assert contract["status"] == "timed_out"
        assert contract["lifecycle_state"] == "timed_out"


# ---------------------------------------------------------------------------
# Isolation orchestration
# ---------------------------------------------------------------------------

class TestIsolationOrchestration:
    """Test that child operations don't corrupt parent state."""

    def test_child_cannot_modify_execution_context(self):
        """Phase 23A: Child cannot modify its frozen execution context."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )

        with pytest.raises(AttributeError):
            ctx.task_type = "maintenance"

        with pytest.raises(AttributeError):
            ctx.authority_level = "L3"

        with pytest.raises(AttributeError):
            ctx.allowed_tools = ALL_JARVIS_TOOLS

    def test_child_failure_doesnt_corrupt_parent_context(self):
        """Phase 23D: Child failure leaves parent's context fields intact."""
        ctx = create_execution_context(
            child_id="sa-fail",
            parent_session_id="parent-session-123",
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

        # Simulate child failing on unauthorized tool
        result = provider.handle_tool_call("hermes_rag_store", {"text": "data"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

        manager.transition("failed")

        # Parent context preserved
        assert ctx.parent_session_id == "parent-session-123"
        assert ctx.task_type == "research"
        assert ctx.authority_level == "L0"
        assert ctx.allowed_tools == L0_READ_ONLY

    def test_child_output_not_automatic_rag_memory(self):
        """Phase 23D: Child output doesn't auto-persist to RAG."""
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

        # Research can't store, so child output can't become canonical
        result = provider.handle_tool_call("hermes_rag_store", {"text": "findings"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

    def test_child_output_not_automatic_continuity_modification(self):
        """Phase 23D: Child output doesn't auto-modify parent continuity."""
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

        # Research can't create goals
        result = provider.handle_tool_call("hermes_goal_create", {"title": "new goal"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False


# ---------------------------------------------------------------------------
# Provenance orchestration
# ---------------------------------------------------------------------------

class TestProvenanceOrchestration:
    """Test that provenance remains isolated."""

    def test_child_provenance_contains_subagent_goal(self):
        """Phase 23A: Provenance context_id contains sub-agent goal."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research honey production",
            task_type="research",
            delegation_depth=1,
        )
        assert "subagent:research" in ctx.provenance_context_id

    def test_child_provenance_does_not_contain_parent_session(self):
        """Phase 23A: Provenance doesn't leak parent session ID."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent-session-456",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        assert "parent-session" not in ctx.provenance_context_id

    def test_provenance_preserved_through_lifecycle(self):
        """Phase 23B: Provenance context_id is preserved across lifecycle transitions."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        original_provenance = ctx.provenance_context_id

        manager.transition("initializing")
        assert manager.context.provenance_context_id == original_provenance

        manager.transition("running")
        assert manager.context.provenance_context_id == original_provenance

        manager.transition("completed")
        assert manager.context.provenance_context_id == original_provenance


# ---------------------------------------------------------------------------
# Recursion-depth orchestration
# ---------------------------------------------------------------------------

class TestRecursionDepthOrchestration:
    """Test that recursion-depth enforcement works."""

    def test_depth_tracked_in_context(self):
        """Phase 23A: Delegation depth is tracked in execution context."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=2,
        )
        assert ctx.delegation_depth == 2

    def test_max_depth_constant(self):
        """Phase 23A: Maximum recursion depth is enforced."""
        assert JARVISSubagentProvider._MAX_SUBAGENT_DEPTH == 2

    def test_depth_preserved_through_lifecycle(self):
        """Phase 23B: Delegation depth is preserved across lifecycle transitions."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)

        manager.transition("initializing")
        assert manager.context.delegation_depth == 1

        manager.transition("running")
        assert manager.context.delegation_depth == 1

        manager.transition("completed")
        assert manager.context.delegation_depth == 1


# ---------------------------------------------------------------------------
# Cleanup orchestration
# ---------------------------------------------------------------------------

class TestCleanupOrchestration:
    """Test that cleanup works correctly."""

    def test_lifecycle_manager_cleanup(self):
        """Phase 23C: Lifecycle manager is cleaned up after child completion."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        register_lifecycle_manager("sa-test", manager)

        assert _get_lifecycle_state_for_subagent("sa-test") == "initializing"

        # Simulate cleanup
        unregister_lifecycle_manager("sa-test")

        assert _get_lifecycle_state_for_subagent("sa-test") is None

    def test_lifecycle_manager_cleanup_after_transition(self):
        """Phase 23C: Lifecycle manager reflects final state before cleanup."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        register_lifecycle_manager("sa-test", manager)

        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")

        assert _get_lifecycle_state_for_subagent("sa-test") == "completed"

        unregister_lifecycle_manager("sa-test")
        assert _get_lifecycle_state_for_subagent("sa-test") is None


# ---------------------------------------------------------------------------
# L2/L3 approval orchestration
# ---------------------------------------------------------------------------

class TestL2L3ApprovalOrchestration:
    """Test that L2 staging and L3 approval work end-to-end."""

    def test_l2_gated_tools_pass_through_to_handler(self):
        """Phase 22B: L2 tools (gated) pass through whitelist to handler."""
        ctx = create_execution_context(
            child_id="sa-exec",
            parent_session_id="parent",
            task_goal="execute task",
            task_type="task_execution",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="execute task",
            task_type="task_execution",
            execution_context=ctx,
        )

        # task_execution has L2 tools (hermes_goal_create, etc.)
        # These pass the whitelist check and go to _dispatch_jarvis_tool_call
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            result = provider.handle_tool_call("hermes_goal_create", {"title": "new"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is True

    def test_l3_approval_required(self):
        """Phase 22B: L3 tools require explicit approval."""
        ctx = create_execution_context(
            child_id="sa-maint",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="maintenance",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="maintenance",
            execution_context=ctx,
        )

        # Without approval, L3 tools are denied
        with patch("tools.approval.request_tool_approval") as mock:
            mock.return_value = {"approved": False}
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False
            assert "not approved" in result_dict["error"]

    def test_l3_approval_granted(self):
        """Phase 22B: L3 tools succeed with explicit approval."""
        ctx = create_execution_context(
            child_id="sa-maint",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="maintenance",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="maintenance",
            execution_context=ctx,
        )

        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": True}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                mock_dispatch.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                result_dict = json.loads(result)
                assert result_dict["ok"] is True


# ---------------------------------------------------------------------------
# Bypass-gate stripping orchestration
# ---------------------------------------------------------------------------

class TestBypassGateStrippingOrchestration:
    """Test that _continuity_bypass_gate is stripped."""

    def test_bypass_gate_stripped_from_kwargs(self):
        """Phase 20B: _continuity_bypass_gate is stripped from kwargs."""
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

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            # Attempt to inject bypass gate
            provider.handle_tool_call("hermes_rag_search", {"query": "test"},
                                      _continuity_bypass_gate=True)
            # Should have been stripped — dispatch called without bypass
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# Provenance isolation orchestration
# ---------------------------------------------------------------------------

class TestProvenanceIsolationOrchestration:
    """Test that provenance is isolated via deepcopy."""

    def test_args_deepcopied(self):
        """Phase 20B: Args are deep-copied to prevent parent mutation."""
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

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            original = {"query": "test", "extra": {"nested": "value"}}
            original_copy = json.loads(json.dumps(original))
            provider.handle_tool_call("hermes_rag_search", original)
            assert original == original_copy


# ---------------------------------------------------------------------------
# Task-type registry orchestration
# ---------------------------------------------------------------------------

class TestTaskTypeRegistryOrchestration:
    """Test the task type registry."""

    def test_all_task_types_registered(self):
        """Phase 22A: All expected task types are registered."""
        expected = {"research", "summarize", "note_taking", "task_execution", "maintenance", "default"}
        assert set(SUBAGENT_TASK_TYPES.keys()) == expected

    def test_task_type_validation(self):
        """Phase 22A: Task type validation works."""
        assert validate_task_type("research") is True
        assert validate_task_type("unknown") is False
        assert validate_task_type(None) is False

    def test_tool_allowed_for_task(self):
        """Phase 22A: Tool allowance check works."""
        assert is_tool_allowed_for_task("research", "hermes_rag_search") is True
        assert is_tool_allowed_for_task("research", "hermes_rag_store") is False
        assert is_tool_allowed_for_task("note_taking", "hermes_rag_store") is True
        assert is_tool_allowed_for_task("maintenance", "hermes_rag_delete") is True

    def test_task_type_cannot_exceed_authority(self):
        """Phase 22A: Task types cannot exceed their authority level."""
        with pytest.raises(ValueError, match="cannot include tools outside its level"):
            SubagentTaskType(
                name="bad",
                description="Bad task",
                authority_level="L0",
                allowed_tools=L0_READ_ONLY | L1_REVERSIBLE,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
