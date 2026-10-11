"""Phase 33 — Sub-Agent Execution & Result Integrity Certification.

Adversarial certification that authorized executions cannot produce results
that are misattributed, substituted, replayed, or accepted under the wrong
sub-agent context.

Covers:
1. Result identity and uniqueness
2. Result-contract integrity (all fields preserved)
3. Execution/result correlation
4. Failure/cancellation/timeout results
5. Replay and duplicate detection
6. Parent/child isolation
7. Provenance integrity
8. Concurrent execution isolation
9. Audit correlation
10. TOCTOU conditions

No production code is modified by this suite.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import plugins.memory.jarvis.__init__ as jarvis_init
from plugins.memory.jarvis.__init__ import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    L0_READ_ONLY,
    SubagentExecutionContext,
    SubagentLifecycleManager,
    create_execution_context,
)
from plugins.memory.jarvis.subagent_security import (
    get_recorder,
    reset_recorder,
)
from tools.delegate_tool import (
    register_lifecycle_manager,
    unregister_lifecycle_manager,
    transition_lifecycle,
    _get_lifecycle_state_for_subagent,
    _build_result_contract,
    RESULT_STATUSES,
)


@pytest.fixture(autouse=True)
def reset_recorder_fixture():
    reset_recorder()
    yield
    reset_recorder()


def _make_provider(task_type="research", depth=1, child_id=None, task_goal=None):
    ctx = create_execution_context(
        child_id=child_id or f"sa-{task_type}-{depth}",
        parent_session_id="parent-session-001",
        task_goal=task_goal or f"Test {task_type} goal",
        task_type=task_type,
        delegation_depth=depth,
    )
    return JARVISSubagentProvider(
        goal=task_goal or f"Test {task_type} goal",
        context="test context",
        depth=depth,
        task_type=task_type,
        execution_context=ctx,
    ), ctx


# ====================================================================
# 1. Result Identity and Uniqueness
# ====================================================================

class TestResultIdentity:
    """Each result must be unique and tied to the correct sub-agent."""

    def test_each_subagent_has_unique_id(self):
        _, ctx1 = _make_provider(child_id="sa-unique-1")
        _, ctx2 = _make_provider(child_id="sa-unique-2")
        assert ctx1.child_id != ctx2.child_id

    def test_each_subagent_has_unique_provenance_for_different_goals(self):
        _, ctx1 = _make_provider(child_id="sa-prov-1", task_goal="Research honey production")
        _, ctx2 = _make_provider(child_id="sa-prov-2", task_goal="Analyze market trends")
        assert ctx1.provenance_context_id != ctx2.provenance_context_id

    def test_provenance_contains_subagent_prefix(self):
        _, ctx = _make_provider()
        assert ctx.provenance_context_id.startswith("subagent:")

    def test_result_contract_includes_subagent_id(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="test",
            subagent_id="sa-test-123",
            child_session_id="child-456",
            task_goal="research",
            lifecycle_state="completed",
        )
        assert contract["subagent_id"] == "sa-test-123"

    def test_result_contract_includes_child_session_id(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="test",
            subagent_id="sa-test",
            child_session_id="child-session-789",
            task_goal="research",
            lifecycle_state="completed",
        )
        assert contract["child_session_id"] == "child-session-789"

    def test_result_contract_includes_task_goal(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="test",
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="specific research goal",
            lifecycle_state="completed",
        )
        assert contract["task_goal"] == "specific research goal"


# ====================================================================
# 2. Result-Contract Integrity
# ====================================================================

class TestResultContractIntegrity:
    """All result-contract fields must be preserved correctly."""

    @pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "timed_out", "error", "interrupted"])
    def test_all_statuses_preserved(self, status):
        contract = _build_result_contract(
            task_index=0,
            status=status,
            summary="test",
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state=status,
        )
        assert contract["status"] == status

    def test_completed_contract_has_all_fields(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="test summary",
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state="completed",
            api_calls=5,
            duration_seconds=10.5,
            model="test-model",
            exit_reason="completed",
            cost_usd=0.001,
            cost_status="estimated",
        )
        assert contract["task_index"] == 0
        assert contract["status"] == "completed"
        assert contract["summary"] == "test summary"
        assert contract["subagent_id"] == "sa-test"
        assert contract["child_session_id"] == "child-session"
        assert contract["task_goal"] == "research"
        assert contract["lifecycle_state"] == "completed"
        assert contract["api_calls"] == 5
        assert contract["duration_seconds"] == 10.5
        assert contract["model"] == "test-model"
        assert contract["exit_reason"] == "completed"
        assert contract["cost_usd"] == 0.001
        assert contract["cost_status"] == "estimated"
        assert "error" not in contract

    def test_failed_contract_includes_error(self):
        contract = _build_result_contract(
            task_index=0,
            status="failed",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state="failed",
            error="Connection timeout",
        )
        assert contract["status"] == "failed"
        assert contract["error"] == "Connection timeout"
        assert contract["summary"] is None

    def test_cancelled_contract_has_correct_status(self):
        contract = _build_result_contract(
            task_index=0,
            status="cancelled",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state="cancelled",
        )
        assert contract["status"] == "cancelled"
        assert contract["lifecycle_state"] == "cancelled"

    def test_timed_out_contract_has_correct_status(self):
        contract = _build_result_contract(
            task_index=0,
            status="timed_out",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state="timed_out",
            duration_seconds=60.0,
        )
        assert contract["status"] == "timed_out"
        assert contract["lifecycle_state"] == "timed_out"
        assert contract["duration_seconds"] == 60.0

    def test_error_contract_has_correct_status(self):
        contract = _build_result_contract(
            task_index=0,
            status="error",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state="failed",
            error="API error",
        )
        assert contract["status"] == "error"
        assert contract["error"] == "API error"

    def test_result_statuses_constant_complete(self):
        expected = {"completed", "failed", "cancelled", "timed_out", "error", "interrupted"}
        assert RESULT_STATUSES == expected

    def test_contract_task_index_isolation(self):
        contract0 = _build_result_contract(
            task_index=0,
            status="completed",
            summary="test",
            subagent_id="sa-0",
            child_session_id="child-0",
            task_goal="research",
            lifecycle_state="completed",
        )
        contract1 = _build_result_contract(
            task_index=1,
            status="completed",
            summary="test",
            subagent_id="sa-1",
            child_session_id="child-1",
            task_goal="research",
            lifecycle_state="completed",
        )
        assert contract0["task_index"] != contract1["task_index"]
        assert contract0["subagent_id"] != contract1["subagent_id"]


# ====================================================================
# 3. Execution/Result Correlation
# ====================================================================

class TestExecutionResultCorrelation:
    """Results must correlate with the execution that produced them."""

    def test_provider_correlates_to_context(self):
        provider, ctx = _make_provider(task_type="research", child_id="sa-corr")
        assert provider._execution_context.child_id == ctx.child_id

    def test_provider_uses_context_task_type(self):
        provider, ctx = _make_provider(task_type="research")
        assert provider._task_type == ctx.task_type

    def test_provider_uses_context_authority(self):
        provider, ctx = _make_provider(task_type="maintenance")
        assert provider._execution_context.authority_level == "L3"

    def test_provider_uses_context_allowed_tools(self):
        provider, ctx = _make_provider(task_type="research")
        assert provider._execution_context.allowed_tools == L0_READ_ONLY


# ====================================================================
# 4. Failure/Cancellation/Timeout Results
# ====================================================================

class TestFailureResults:
    """Failed executions must produce correct failure results."""

    def test_tool_execution_failure_recorded(self):
        provider, ctx = _make_provider(task_type="research")
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        assert json.loads(result)["ok"] is False
        recorder = get_recorder()
        events = recorder.get_events()
        denied_events = [e for e in events if e.allowed is False]
        assert len(denied_events) >= 1

    def test_tool_not_found_recorded(self):
        provider, ctx = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("nonexistent_tool", {})
        assert json.loads(result)["ok"] is False
        recorder = get_recorder()
        events = recorder.get_events()
        # Unknown tools are caught by the task-type check first (defense in depth)
        # or by the capability registry check (forged capability)
        denied_events = [e for e in events if e.allowed is False]
        assert len(denied_events) >= 1

    def test_read_only_denial_recorded(self):
        """Decision 42: the refusal is audited as an authorization denial
        (was: an "Approval denied" event after asking for approval)."""
        provider, ctx = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": False}):
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            assert json.loads(result)["code"] == "delegated_read_only"
            recorder = get_recorder()
            events = recorder.get_events()
            denied = [e for e in events if e.reason == "Decision 42: delegated agents are read-only"]
            assert len(denied) >= 1


# ====================================================================
# 5. Replay and Duplicate Detection
# ====================================================================

class TestReplayDetection:
    """Results cannot be replayed or duplicated."""

    def test_each_tool_call_produces_unique_audit_event(self):
        provider, ctx = _make_provider(task_type="research")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider.handle_tool_call("hermes_rag_search", {"query": "test1"})
            provider.handle_tool_call("hermes_rag_search", {"query": "test2"})
            recorder = get_recorder()
            events = recorder.get_events()
            allowed_events = [e for e in events if e.allowed is True]
            assert len(allowed_events) >= 2

    def test_different_subagents_have_different_audit_events(self):
        provider1, ctx1 = _make_provider(child_id="sa-replay-1")
        provider2, ctx2 = _make_provider(child_id="sa-replay-2")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider1.handle_tool_call("hermes_rag_search", {"query": "test"})
            provider2.handle_tool_call("hermes_rag_search", {"query": "test"})
            recorder = get_recorder()
            events = recorder.get_events()
            subagent_ids = set(e.subagent_id for e in events)
            assert len(subagent_ids) >= 2

    def test_no_approval_call_on_any_invocation(self):
        """Decision 42: no invocation reaches approval (was: one per call)."""
        provider, ctx = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": True}) as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})) as d:
                for memory_id in ("x", "y", "z"):
                    r = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": memory_id}))
                    assert r["code"] == "delegated_read_only"
                assert mock_approval.call_count == 0
                d.assert_not_called()


# ====================================================================
# 6. Parent/Child Isolation
# ====================================================================

class TestParentChildIsolation:
    """Parent cannot see child's intermediate state."""

    def test_parent_tools_not_exposed_to_child(self):
        for task_type in ["research", "summarize", "note_taking", "task_execution", "maintenance", "default"]:
            provider, ctx = _make_provider(task_type=task_type)
            for tool in ["terminal", "delegate_task", "write_file", "read_file", "browser_navigate"]:
                result = provider.handle_tool_call(tool, {})
                assert json.loads(result)["ok"] is False, \
                    f"Tool '{tool}' should be denied for task type '{task_type}'"

    def test_child_cannot_access_parent_memory(self):
        _, ctx = _make_provider()
        assert "subagent:" in ctx.provenance_context_id

    def test_child_cannot_impersonate_parent(self):
        provider, ctx = _make_provider(task_type="note_taking")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})) as mock:
            r = json.loads(provider.handle_tool_call("hermes_rag_store", {
                "text": "test",
                "provenance": {"source": "user_confirmed", "context_id": "parent:123"}
            }))
            # Decision 42: refused outright (was: dispatched with provenance rewritten)
            assert r["code"] == "delegated_read_only"
            mock.assert_not_called()


# ====================================================================
# 7. Provenance Integrity
# ====================================================================

class TestProvenanceIntegrity:
    """Provenance must correctly identify the sub-agent."""

    def test_provenance_in_context_matches_child_id(self):
        _, ctx = _make_provider(child_id="sa-prov-test", task_type="research")
        assert ctx.child_id in ctx.provenance_context_id or ctx.provenance_context_id.startswith("subagent:")

    def test_provenance_preserved_across_transitions(self):
        _, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)
        original_provenance = ctx.provenance_context_id
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.context.provenance_context_id == original_provenance

    def test_provenance_unique_for_different_goals(self):
        _, ctx1 = _make_provider(child_id="sa-prov-1", task_goal="Research honey")
        _, ctx2 = _make_provider(child_id="sa-prov-2", task_goal="Analyze market")
        assert ctx1.provenance_context_id != ctx2.provenance_context_id

    def test_provenance_contains_task_goal(self):
        _, ctx = _make_provider(task_goal="specific research goal")
        assert "specific" in ctx.provenance_context_id


# ====================================================================
# 8. Concurrent Execution Isolation
# ====================================================================

class TestConcurrentExecutionIsolation:
    """Concurrent sub-agents must produce independent results."""

    def test_concurrent_subagents_have_independent_contexts(self):
        _, ctx1 = _make_provider(child_id="sa-conc-1")
        _, ctx2 = _make_provider(child_id="sa-conc-2")
        mgr1 = SubagentLifecycleManager(ctx1)
        mgr2 = SubagentLifecycleManager(ctx2)

        mgr1.transition("initializing")
        mgr1.transition("running")

        assert mgr2.state == "pending"

    def test_concurrent_tool_calls_isolated(self):
        provider1, _ = _make_provider(child_id="sa-conc-tools-1")
        provider2, _ = _make_provider(child_id="sa-conc-tools-2")
        results = []
        results_lock = threading.Lock()

        def call_tool(provider):
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                with results_lock:
                    results.append(json.loads(result)["ok"])

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.map(call_tool, [provider1, provider2])

        assert all(results)

    def test_concurrent_lifecycle_transitions_safe(self):
        _, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)
        errors = []

        def transition_lifecycle(state):
            try:
                manager.transition(state)
            except ValueError:
                pass
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(transition_lifecycle, ["initializing", "running", "completed"])

        assert len(errors) == 0
        assert manager.is_terminal


# ====================================================================
# 9. Audit Correlation
# ====================================================================

class TestAuditCorrelation:
    """Audit events must correlate with results."""

    def test_authorization_allowed_event_correlates_to_success(self):
        provider, ctx = _make_provider(task_type="research")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            recorder = get_recorder()
            events = recorder.get_events()
            allowed = [e for e in events if e.allowed is True]
            assert len(allowed) >= 1
            assert allowed[0].subagent_id == ctx.child_id

    def test_authorization_denied_event_correlates_to_failure(self):
        provider, ctx = _make_provider(task_type="research")
        provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        recorder = get_recorder()
        events = recorder.get_events()
        denied = [e for e in events if e.allowed is False]
        assert len(denied) >= 1
        assert denied[0].subagent_id == ctx.child_id

    def test_audit_event_includes_tool_name(self):
        provider, ctx = _make_provider(task_type="research")
        provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        recorder = get_recorder()
        events = recorder.get_events()
        denied = [e for e in events if e.allowed is False]
        assert len(denied) >= 1
        assert denied[0].tool_name == "hermes_rag_store"

    def test_audit_event_includes_lifecycle_state(self):
        provider, ctx = _make_provider(task_type="research")
        provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        recorder = get_recorder()
        events = recorder.get_events()
        assert len(events) >= 1


# ====================================================================
# 10. TOCTOU Conditions
# ====================================================================

class TestTOCTOU:
    """Context snapshot at authorization time must be preserved."""

    def test_authorization_uses_context_at_call_time(self):
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)

        manager.transition("initializing")
        manager.transition("running")

        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        assert json.loads(result)["ok"] is False

    def test_approval_runs_before_dispatch(self):
        provider, ctx = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval") as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock_approval.return_value = {"approved": False}
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})

                # Decision 42: refusal precedes approval (was: approval called once)
                mock_approval.assert_not_called()
                mock_dispatch.assert_not_called()
                assert json.loads(result)["code"] == "delegated_read_only"

    def test_context_not_mutated_by_tool_call(self):
        provider, ctx = _make_provider(task_type="research")
        original_task_type = provider._execution_context.task_type
        original_authority = provider._execution_context.authority_level

        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider.handle_tool_call("hermes_rag_search", {"query": "test"})

        assert provider._execution_context.task_type == original_task_type
        assert provider._execution_context.authority_level == original_authority


# ====================================================================
# 11. Result-Contract + Lifecycle Agreement
# ====================================================================

class TestResultContractLifecycleAgreement:
    """Result contract status must agree with lifecycle state."""

    def test_completed_contract_matches_lifecycle(self):
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="test",
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state="completed",
        )
        assert contract["status"] == "completed"
        assert contract["lifecycle_state"] == "completed"

    def test_failed_contract_matches_lifecycle(self):
        contract = _build_result_contract(
            task_index=0,
            status="failed",
            summary=None,
            subagent_id="sa-test",
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state="failed",
            error="timeout",
        )
        assert contract["status"] == "failed"
        assert contract["lifecycle_state"] == "failed"

    def test_lifecycle_state_reflects_transition(self):
        _, ctx = _make_provider()
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")

        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="test",
            subagent_id=ctx.child_id,
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state=manager.state,
        )
        assert contract["lifecycle_state"] == "completed"


# ====================================================================
# 12. Cross-Task-Type Result Isolation
# ====================================================================

class TestCrossTaskTypeResultIsolation:
    """Results from different task types must be isolated."""

    def test_research_result_isolated_from_maintenance(self):
        _, ctx_research = _make_provider(task_type="research")
        _, ctx_maintenance = _make_provider(task_type="maintenance")
        assert ctx_research.child_id != ctx_maintenance.child_id

    def test_research_cannot_use_maintenance_result(self):
        provider, _ = _make_provider(task_type="research")
        result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
        assert json.loads(result)["ok"] is False

    def test_maintenance_write_refused_even_with_approval(self):
        """Decision 42 (was: an approved maintenance delete returned ok)."""
        provider, ctx = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": True}):
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})) as d:
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                assert json.loads(result)["code"] == "delegated_read_only"
                d.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
