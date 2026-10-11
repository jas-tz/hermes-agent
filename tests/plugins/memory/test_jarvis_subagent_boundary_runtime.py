"""Phase 32 — Sub-Agent Lifecycle & Authority Transition Certification.

Adversarial certification that lifecycle transitions never weaken authorization:
- All valid lifecycle transitions succeed with correct new state
- All invalid transitions are rejected (terminal states, skipped states)
- Identity fields (child_id, provenance) are immutable across transitions
- Authority level / allowed_tools are immutable across transitions
- Delegation depth is immutable
- L3 approval is per-call (never cached/replayed)
- Task type cannot change after context creation
- Provider authorization uses original context regardless of lifecycle state
- Concurrent lifecycle transitions are safe
- TOCTOU: context snapshot at auth time is preserved through completion
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

import plugins.memory.jarvis.__init__ as jarvis_init
from plugins.memory.jarvis.__init__ import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    L0_READ_ONLY,
    L1_REVERSIBLE,
    L2_GATED,
    L3_APPROVAL,
    ALL_JARVIS_TOOLS,
    SUBAGENT_TASK_TYPES,
    VALID_TRANSITIONS,
    TERMINAL_STATES,
    LIFECYCLE_STATES,
    SubagentExecutionContext,
    SubagentLifecycleManager,
    create_execution_context,
    is_valid_transition,
    get_task_type,
)
from plugins.memory.jarvis.tool_capability_registry import (
    is_tool_permitted,
    is_tool_parent_only,
    requires_approval,
    get_capability,
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
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(task_type="research", depth=1, child_id=None):
    ctx = create_execution_context(
        child_id=child_id or f"sa-{task_type}-{depth}",
        parent_session_id="parent-session-001",
        task_goal=f"Test {task_type} goal",
        task_type=task_type,
        delegation_depth=depth,
    )
    return JARVISSubagentProvider(
        goal=f"Test {task_type} goal",
        context="test context",
        depth=depth,
        task_type=task_type,
        execution_context=ctx,
    ), ctx


# ====================================================================
# 1. Valid Lifecycle Transitions
# ====================================================================

class TestValidLifecycleTransitions:
    """Every valid transition must produce the correct new state."""

    @pytest.mark.parametrize("path", [
        # Standard success path
        ("pending", "initializing"),
        ("initializing", "running"),
        ("running", "completed"),
        # Failure from each active state
        ("initializing", "failed"),
        ("running", "failed"),
        # Cancellation from any non-terminal state
        ("pending", "cancelled"),
        ("initializing", "cancelled"),
        ("running", "cancelled"),
        # Timeout
        ("running", "timed_out"),
    ])
    def test_valid_transition_succeeds(self, path):
        provider, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)
        from_state, to_state = path

        # Advance to from_state first
        if from_state != "pending":
            if from_state == "running":
                manager.transition("initializing")
            if from_state in ("completed", "failed", "cancelled", "timed_out"):
                manager.transition("initializing")
                manager.transition("running")

        if from_state == "pending":
            pass  # already pending
        elif from_state == "initializing":
            manager.transition("initializing")
        elif from_state == "running":
            manager.transition("running")

        new_ctx = manager.transition(to_state)
        assert new_ctx.lifecycle_state == to_state
        assert manager.state == to_state

    def test_full_success_path(self):
        provider, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)

        manager.transition("initializing")
        assert manager.state == "initializing"

        manager.transition("running")
        assert manager.state == "running"

        manager.transition("completed")
        assert manager.state == "completed"
        assert manager.is_terminal

    def test_full_failure_path(self):
        provider, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)

        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")
        assert manager.is_terminal

    def test_pending_to_cancelled_direct(self):
        provider, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)

        manager.transition("cancelled")
        assert manager.is_terminal


# ====================================================================
# 2. Invalid Lifecycle Transitions
# ====================================================================

class TestInvalidLifecycleTransitions:
    """All invalid transitions must be rejected."""

    @pytest.mark.parametrize("from_state,to_state", [
        # Terminal states cannot transition
        ("completed", "running"),
        ("completed", "failed"),
        ("completed", "cancelled"),
        ("failed", "running"),
        ("failed", "completed"),
        ("cancelled", "running"),
        ("timed_out", "running"),
        # Skipped states
        ("pending", "running"),
        ("pending", "completed"),
        ("pending", "failed"),
        ("pending", "timed_out"),
        ("initializing", "completed"),
        ("initializing", "timed_out"),
    ])
    def test_invalid_transition_rejected(self, from_state, to_state):
        provider, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)

        # Advance to from_state
        if from_state != "pending":
            transitions_map = {
                "initializing": ["initializing"],
                "running": ["initializing", "running"],
                "completed": ["initializing", "running", "completed"],
                "failed": ["initializing", "running", "failed"],
                "cancelled": ["cancelled"],
                "timed_out": ["initializing", "running", "timed_out"],
            }
            for t in transitions_map[from_state]:
                manager.transition(t)

        with pytest.raises(ValueError):
            manager.transition(to_state)

    def test_invalid_from_state_not_in_machine(self):
        assert is_valid_transition("nonexistent", "running") is False

    def test_invalid_to_state_not_in_machine(self):
        assert is_valid_transition("pending", "nonexistent") is False

    def test_terminal_state_is_terminal(self):
        for state in TERMINAL_STATES:
            provider, ctx = _make_provider()
            manager = SubagentLifecycleManager(ctx)
            if state == "cancelled":
                manager.transition("cancelled")
            else:
                manager.transition("initializing")
                manager.transition("running")
                manager.transition(state)
            assert manager.is_terminal

    def test_active_states_not_terminal(self):
        for state in ("pending", "initializing", "running"):
            provider, ctx = _make_provider()
            manager = SubagentLifecycleManager(ctx)
            if state == "initializing":
                manager.transition("initializing")
            elif state == "running":
                manager.transition("initializing")
                manager.transition("running")
            assert not manager.is_terminal


# ====================================================================
# 3. Identity Immutability Across Transitions
# ====================================================================

class TestIdentityImmutability:
    """Identity fields must be immutable across lifecycle transitions."""

    @pytest.mark.parametrize("field", [
        "child_id",
        "parent_session_id",
        "task_goal",
        "task_type",
        "authority_level",
        "delegation_depth",
        "allowed_tools",
        "provenance_context_id",
        "created_at",
    ])
    def test_field_immutable_through_all_states(self, field):
        provider, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)
        original_value = getattr(manager.context, field)

        # Advance through full lifecycle
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")

        assert getattr(manager.context, field) == original_value

    def test_child_id_unique_per_context(self):
        _, ctx1 = _make_provider(child_id="sa-unique-1", task_type="research")
        _, ctx2 = _make_provider(child_id="sa-unique-2", task_type="maintenance")
        assert ctx1.child_id != ctx2.child_id
        assert ctx1.provenance_context_id != ctx2.provenance_context_id

    def test_provenance_isolated_from_parent(self):
        _, ctx = _make_provider()
        assert "subagent:" in ctx.provenance_context_id

    def test_created_at_never_changes(self):
        provider, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)
        original = manager.context.created_at

        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")

        assert manager.context.created_at == original


# ====================================================================
# 4. Authority Immutability
# ====================================================================

class TestAuthorityImmutability:
    """Authority level and allowed tools must never change during lifecycle."""

    def test_authority_level_immutable_for_research(self):
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)
        assert manager.context.authority_level == "L0"
        assert manager.context.allowed_tools == L0_READ_ONLY

        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")

        assert manager.context.authority_level == "L0"
        assert manager.context.allowed_tools == L0_READ_ONLY

    def test_authority_level_immutable_for_maintenance(self):
        provider, ctx = _make_provider(task_type="maintenance")
        manager = SubagentLifecycleManager(ctx)
        assert manager.context.authority_level == "L3"

        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")

        assert manager.context.authority_level == "L3"

    def test_allowed_tools_immutable(self):
        for task_type in ["research", "summarize", "note_taking", "task_execution", "maintenance", "default"]:
            provider, ctx = _make_provider(task_type=task_type)
            manager = SubagentLifecycleManager(ctx)
            original_tools = manager.context.allowed_tools

            manager.transition("initializing")
            manager.transition("running")
            manager.transition("completed")

            assert manager.context.allowed_tools == original_tools, \
                f"Tools changed for task type {task_type}"


# ====================================================================
# 5. L3 Approval is Per-Call (No Caching/Replay)
# ====================================================================

class TestL3ApprovalPerCall:
    """Decision 42: delegated L3 calls are refused on every call before
    approval is requested, whatever the approval mock would answer
    (previously: approval was requested per call and an approved call ran)."""

    def test_l3_approval_never_requested(self):
        provider, ctx = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval") as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock_approval.return_value = {"approved": True}
                mock_dispatch.return_value = json.dumps({"ok": True})

                for memory_id in ("x", "y"):
                    result = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": memory_id}))
                    assert result["code"] == "delegated_read_only"

                assert mock_approval.call_count == 0
                mock_dispatch.assert_not_called()

    def test_l3_refused_whether_approval_denied_or_granted(self):
        provider, ctx = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval") as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                for answer in ({"approved": False}, {"approved": True}):
                    mock_approval.return_value = answer
                    mock_dispatch.return_value = json.dumps({"ok": True})
                    result = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": "y"}))
                    assert result["ok"] is False and result["code"] == "delegated_read_only"
                mock_approval.assert_not_called()
                mock_dispatch.assert_not_called()

    def test_l3_refused_even_if_approval_would_raise(self):
        provider, ctx = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval") as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock_approval.side_effect = Exception("Approval failed")
                result = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"}))
                assert result["ok"] is False and result["code"] == "delegated_read_only"
                mock_approval.assert_not_called()
                mock_dispatch.assert_not_called()


# ====================================================================
# 6. Task Type Cannot Change After Context Creation
# ====================================================================

class TestTaskTypeImmutability:
    """Task type cannot change after context creation."""

    def test_task_type_frozen_in_context(self):
        _, ctx = _make_provider(task_type="research")
        with pytest.raises(AttributeError):
            ctx.task_type = "maintenance"

    def test_cannot_create_provider_with_unknown_task_type(self):
        with pytest.raises(ValueError, match="Unknown task type"):
            create_execution_context(
                child_id="sa-test",
                parent_session_id="parent",
                task_goal="test",
                task_type="nonexistent",
                delegation_depth=1,
            )

    def test_provider_uses_original_task_type(self):
        """Provider uses the task type from its original context, not any new value."""
        provider, ctx = _make_provider(task_type="research")
        # Even if we could modify the context (we can't), provider uses its own task_type
        assert provider._task_type == "research"
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        assert json.loads(result)["ok"] is False

    def test_task_type_determines_allowed_tools(self):
        """Task type determines allowed tools at creation time."""
        _, ctx_research = _make_provider(task_type="research")
        _, ctx_maintenance = _make_provider(task_type="maintenance")

        assert "hermes_rag_search" in ctx_research.allowed_tools
        assert "hermes_rag_store" not in ctx_research.allowed_tools

        assert "hermes_rag_search" in ctx_maintenance.allowed_tools
        assert "hermes_rag_store" in ctx_maintenance.allowed_tools


# ====================================================================
# 7. Provider Authorization Uses Correct Current Context
# ====================================================================

class TestProviderAuthorizationContext:
    """Provider authorization always uses the correct sub-agent context."""

    def test_provider_uses_context_at_creation(self):
        """Provider freezes the context at creation time."""
        provider, ctx = _make_provider(task_type="research")
        assert provider._execution_context.task_type == "research"
        assert provider._execution_context.authority_level == "L0"

    def test_provider_enforces_restrictions_in_running_state(self):
        """Provider enforces restrictions even when lifecycle is running."""
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")

        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        assert json.loads(result)["ok"] is False

    def test_provider_enforces_restrictions_in_completed_state(self):
        """Provider enforces restrictions even after completion."""
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")

        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        assert json.loads(result)["ok"] is False

    def test_provider_enforces_restrictions_in_failed_state(self):
        """Provider enforces restrictions even after failure."""
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")

        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        assert json.loads(result)["ok"] is False

    def test_maintenance_provider_allows_reads_only(self):
        """Decision 42: maintenance keeps reads; its writes (formerly allowed
        with approval) are refused before approval."""
        provider, ctx = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval") as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock_approval.return_value = {"approved": True}
                mock_dispatch.return_value = json.dumps({"ok": True})

                assert json.loads(provider.handle_tool_call("hermes_rag_search", {}))["ok"] is True
                for tool in ["hermes_rag_store", "hermes_goal_create", "hermes_rag_delete"]:
                    result = json.loads(provider.handle_tool_call(tool, {}))
                    assert result["code"] == "delegated_read_only", tool
                mock_approval.assert_not_called()
                mock_dispatch.assert_called_once()  # the read only


# ====================================================================
# 8. Concurrent Tool Calls and Lifecycle Transitions
# ====================================================================

class TestConcurrentSafety:
    """Concurrent tool calls and lifecycle transitions must be safe."""

    def test_concurrent_tool_calls_during_running(self):
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")

        results = []
        errors = []

        def call_tool():
            try:
                with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True, "results": []})):
                    result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                    results.append(json.loads(result)["ok"])
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(lambda _: call_tool(), range(10))

        assert len(errors) == 0
        assert all(results)

    def test_concurrent_lifecycle_transitions_are_serialized(self):
        """Lifecycle transitions are serialized through the manager."""
        _, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)

        errors = []

        def transition_lifecycle(state):
            try:
                manager.transition(state)
            except ValueError:
                pass  # Expected for invalid transitions
            except Exception as e:
                errors.append(e)

        # Try concurrent valid transitions
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(transition_lifecycle, ["initializing", "running", "completed"])

        assert len(errors) == 0
        assert manager.is_terminal

    def test_tool_call_during_lifecycle_transition(self):
        """Tool calls during lifecycle transitions remain authorized."""
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)

        # Start transitioning
        manager.transition("initializing")

        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            assert json.loads(result)["ok"] is True


# ====================================================================
# 9. TOCTOU Conditions
# ====================================================================

class TestTOCTOU:
    """Context snapshot at authorization time must be preserved."""

    def test_context_snapshot_at_authorization(self):
        """Authorization uses the context snapshot at the time of the call."""
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)

        # Authorize while running
        manager.transition("initializing")
        manager.transition("running")

        # Complete lifecycle
        manager.transition("completed")

        # Provider should still enforce original restrictions
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        assert json.loads(result)["ok"] is False

    def test_refusal_runs_before_approval_and_dispatch(self):
        """Decision 42: the read-only refusal precedes both approval and
        dispatch (previously: approval ran first, then dispatch on success)."""
        provider, ctx = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval") as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock_approval.return_value = {"approved": False}
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})

                mock_approval.assert_not_called()
                mock_dispatch.assert_not_called()
                assert json.loads(result)["code"] == "delegated_read_only"

    def test_context_not_mutated_by_tool_call(self):
        """Tool calls do not mutate the execution context."""
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")

        original_task_type = provider._execution_context.task_type
        original_authority = provider._execution_context.authority_level

        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider.handle_tool_call("hermes_rag_search", {"query": "test"})

        assert provider._execution_context.task_type == original_task_type
        assert provider._execution_context.authority_level == original_authority


# ====================================================================
# 10. Lifecycle Manager Registry Integration
# ====================================================================

class TestLifecycleRegistry:
    """Lifecycle manager registry integration."""

    def test_register_and_transition(self):
        _, ctx = _make_provider(child_id="sa-reg-test")
        manager = SubagentLifecycleManager(ctx)
        register_lifecycle_manager("sa-reg-test", manager)

        assert transition_lifecycle("sa-reg-test", "initializing") is True
        assert _get_lifecycle_state_for_subagent("sa-reg-test") == "initializing"

    def test_unregister_stops_transitions(self):
        _, ctx = _make_provider(child_id="sa-unreg-test")
        manager = SubagentLifecycleManager(ctx)
        register_lifecycle_manager("sa-unreg-test", manager)
        unregister_lifecycle_manager("sa-unreg-test")

        assert transition_lifecycle("sa-unreg-test", "initializing") is False
        assert _get_lifecycle_state_for_subagent("sa-unreg-test") is None

    def test_transition_unknown_subagent(self):
        assert transition_lifecycle("sa-nonexistent", "running") is False

    def test_result_contract_built_correctly(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="test summary",
            subagent_id="sa-test",
            child_session_id="child-session-123",
            task_goal="research",
            lifecycle_state="completed",
        )
        assert contract["status"] == "completed"
        assert contract["lifecycle_state"] == "completed"
        assert contract["subagent_id"] == "sa-test"


# ====================================================================
# 11. Lifecycle State Machine Completeness
# ====================================================================

class TestLifecycleStateMachine:
    """Verify the lifecycle state machine is complete."""

    def test_all_states_in_machine(self):
        all_states = set(VALID_TRANSITIONS.keys()) | TERMINAL_STATES
        assert all_states == LIFECYCLE_STATES

    def test_valid_transitions_reference_valid_states(self):
        for from_state, to_states in VALID_TRANSITIONS.items():
            assert from_state in LIFECYCLE_STATES
            for to_state in to_states:
                assert to_state in LIFECYCLE_STATES

    def test_terminal_states_have_no_outgoing_transitions(self):
        for state in TERMINAL_STATES:
            assert state not in VALID_TRANSITIONS

    def test_is_valid_transition_matches_machine(self):
        for from_state in LIFECYCLE_STATES:
            for to_state in LIFECYCLE_STATES:
                expected = is_valid_transition(from_state, to_state)
                if from_state in TERMINAL_STATES:
                    assert expected is False
                elif from_state in VALID_TRANSITIONS:
                    assert expected == (to_state in VALID_TRANSITIONS[from_state])
                else:
                    assert expected is False


# ====================================================================
# 12. Cross-Task-Type Security Boundaries
# ====================================================================

class TestCrossTaskTypeSecurity:
    """Cross-task-type security boundaries are enforced."""

    def test_research_cannot_write(self):
        provider, _ = _make_provider(task_type="research")
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        assert json.loads(result)["ok"] is False

    # Decision 42: no task type grants a delegated write any more. The four
    # tests below formerly asserted that note_taking could store,
    # task_execution could create goals, maintenance could delete with
    # approval, and default had full access.

    def test_note_taking_cannot_write(self):
        provider, _ = _make_provider(task_type="note_taking")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True, "memory_id": "test"})) as d:
            result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
            assert json.loads(result)["code"] == "delegated_read_only"
            d.assert_not_called()

    def test_task_execution_cannot_create_goals(self):
        provider, _ = _make_provider(task_type="task_execution")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True, "goal": {"id": "g_new"}})) as d:
            result = provider.handle_tool_call("hermes_goal_create", {"title": "new"})
            assert json.loads(result)["code"] == "delegated_read_only"
            d.assert_not_called()

    def test_maintenance_cannot_delete_even_with_approval(self):
        provider, _ = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": True}) as a:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})) as d:
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                assert json.loads(result)["code"] == "delegated_read_only"
                a.assert_not_called()
                d.assert_not_called()

    def test_default_is_read_only(self):
        provider, _ = _make_provider(task_type="default")
        with patch("tools.approval.request_tool_approval", return_value={"approved": True}):
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                assert json.loads(provider.handle_tool_call("hermes_rag_search", {}))["ok"] is True
                for tool in ["hermes_rag_store", "hermes_goal_create", "hermes_rag_delete"]:
                    result = provider.handle_tool_call(tool, {})
                    assert json.loads(result)["code"] == "delegated_read_only", tool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
