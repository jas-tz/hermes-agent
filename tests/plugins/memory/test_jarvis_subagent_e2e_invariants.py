"""Phase 34 — End-to-End Security Invariant Certification.

Cross-component adversarial tests that exercise multiple Phase 20-33
systems together. Focus: invariant violations that component-level
tests could miss.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, call

import pytest

import plugins.memory.jarvis.__init__ as jarvis_init
from plugins.memory.jarvis.__init__ import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    L0_READ_ONLY,
    ALL_JARVIS_TOOLS,
    VALID_TRANSITIONS,
    TERMINAL_STATES,
    SubagentExecutionContext,
    SubagentLifecycleManager,
    create_execution_context,
)
from plugins.memory.jarvis.tool_capability_registry import (
    is_tool_permitted,
    is_tool_parent_only,
    requires_approval,
    get_capability,
    get_permitted_tools_for_task_type,
    TOOL_CAPABILITY_REGISTRY,
)
from plugins.memory.jarvis.subagent_security import (
    get_recorder,
    reset_recorder,
    SecurityEvent,
    SecurityEventType,
)
from plugins.memory.jarvis.subagent_security_store import (
    SecurityAuditStore,
    create_audit_store,
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
    goal = task_goal or f"Test {task_type} goal"
    ctx = create_execution_context(
        child_id=child_id or f"sa-{task_type}-{depth}",
        parent_session_id="parent-session-001",
        task_goal=goal,
        task_type=task_type,
        delegation_depth=depth,
    )
    return JARVISSubagentProvider(
        goal=goal,
        context="test context",
        depth=depth,
        task_type=task_type,
        execution_context=ctx,
    ), ctx


# I1: No Bypass Path
class TestNoBypassPath:
    def test_dispatch_only_reachable_via_handle_tool_call(self):
        import inspect
        source = inspect.getsource(jarvis_init)
        assert source.count("_dispatch_jarvis_tool_call(") == 2

    def test_delegate_tool_does_not_dispatch(self):
        import pathlib
        path = pathlib.Path("tools/delegate_tool.py")
        if path.exists():
            text = path.read_text()
            assert "_dispatch_jarvis_tool_call(" not in text


# I2: Capability Registry Authoritative
class TestCapabilityRegistryAuthoritative:
    def test_all_parent_only_tools_denied(self):
        parent_only = {n for n, c in TOOL_CAPABILITY_REGISTRY.items() if c.parent_only}
        for tool_name in parent_only:
            for task_type in ["research", "maintenance", "default"]:
                provider, _ = _make_provider(task_type=task_type)
                result = provider.handle_tool_call(tool_name, {})
                data = json.loads(result)
                assert data["ok"] is False, \
                    f"Parent-only '{tool_name}' allowed for '{task_type}'"

    def test_unknown_tool_denied_for_all_task_types(self):
        for task_type in ["research", "maintenance", "default"]:
            provider, _ = _make_provider(task_type=task_type)
            result = provider.handle_tool_call("forged_tool_xyz", {})
            assert json.loads(result)["ok"] is False


# I3: Task Type Immutable
class TestTaskTypeImmutable:
    def test_research_stays_read_only_through_lifecycle(self):
        provider, ctx = _make_provider(task_type="research")
        manager = SubagentLifecycleManager(ctx)
        for state in ["pending", "initializing", "running", "completed"]:
            if state == "initializing":
                manager.transition("initializing")
            elif state == "running":
                manager.transition("running")
            elif state == "completed":
                manager.transition("completed")
            result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
            assert json.loads(result)["ok"] is False


# I4: L3 Approval Per-Call
class TestL3ApprovalPerCall:
    def test_each_l3_call_refused_without_approval(self):
        """Decision 42: each delegated L3 call is refused before approval
        (previously: each call requested a fresh approval)."""
        provider, _ = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval") as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})) as d:
                mock_approval.return_value = {"approved": True}
                for memory_id in ("a", "b"):
                    r = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": memory_id}))
                    assert r["code"] == "delegated_read_only"
                assert mock_approval.call_count == 0
                d.assert_not_called()


# I5: Provenance Isolation
class TestProvenanceIsolation:
    def test_original_args_not_mutated(self):
        provider, _ = _make_provider(task_type="note_taking")
        original = {"text": "test", "provenance": {"source": "conversation"}}
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider.handle_tool_call("hermes_rag_store", original)
        assert original["provenance"]["source"] == "conversation"

    def test_child_store_with_parent_provenance_refused(self):
        """Decision 42: a child store carrying parent provenance is refused
        outright (previously: dispatched with provenance overwritten)."""
        provider, _ = _make_provider(task_type="note_taking")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})) as mock:
            r = json.loads(provider.handle_tool_call("hermes_rag_store", {
                "text": "test",
                "provenance": {"source": "user_confirmed", "context_id": "parent:123"}
            }))
            assert r["code"] == "delegated_read_only"
            mock.assert_not_called()


# I6: Bypass Gate Stripped
class TestBypassGateStripped:
    def test_bypass_gate_ignored_for_parent_only(self):
        provider, _ = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("delegate_task", {}, _continuity_bypass_gate=True)
        assert json.loads(result)["ok"] is False
        assert "parent-only" in json.loads(result)["error"]


# I7: Audit Failure Isolation
class TestAuditFailureIsolation:
    def test_all_observability_failed_still_allows(self):
        provider, _ = _make_provider(task_type="research")
        with patch("plugins.memory.jarvis.subagent_security.record_authorization_allowed", side_effect=Exception("fail")):
            with patch("plugins.memory.jarvis.subagent_security.record_tool_execution_completed", side_effect=Exception("fail")):
                with patch.object(get_recorder(), "record", side_effect=Exception("fail")):
                    with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                        result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                        assert json.loads(result)["ok"] is True

    def test_all_observability_failed_still_denies(self):
        provider, _ = _make_provider(task_type="research")
        with patch("plugins.memory.jarvis.subagent_security.record_authorization_denied", side_effect=Exception("fail")):
            with patch("plugins.memory.jarvis.subagent_security.record_forged_capability", side_effect=Exception("fail")):
                with patch.object(get_recorder(), "record", side_effect=Exception("fail")):
                    result = provider.handle_tool_call("forged_tool", {})
                    assert json.loads(result)["ok"] is False


# I8: Lifecycle Doesn't Weaken Authorization
class TestLifecycleDoesntWeaken:
    def test_restrictions_same_at_every_state(self):
        for task_type in ["research", "note_taking"]:
            provider, ctx = _make_provider(task_type=task_type)
            manager = SubagentLifecycleManager(ctx)
            for state in ["pending", "initializing", "running"]:
                if state != "pending":
                    manager.transition(state)
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                assert json.loads(result)["ok"] is False


# I11: Cross-Task-Type Boundaries
class TestCrossTaskTypeBoundaries:
    @pytest.mark.parametrize("task_type,denied_tool", [
        ("research", "hermes_rag_store"),
        ("research", "hermes_rag_delete"),
        ("note_taking", "hermes_rag_delete"),
    ])
    def test_task_type_denies_unauthorized(self, task_type, denied_tool):
        provider, _ = _make_provider(task_type=task_type)
        result = provider.handle_tool_call(denied_tool, {})
        assert json.loads(result)["ok"] is False

    # Decision 42: note_taking/hermes_rag_store and maintenance/hermes_rag_delete
    # moved from "allowed" to test_task_type_write_refused below.
    @pytest.mark.parametrize("task_type,allowed_tool", [
        ("research", "hermes_rag_search"),
        ("maintenance", "hermes_rag_search"),
    ])
    def test_task_type_allows_authorized(self, task_type, allowed_tool):
        provider, _ = _make_provider(task_type=task_type)
        with patch("tools.approval.request_tool_approval", return_value={"approved": True}):
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                result = provider.handle_tool_call(allowed_tool, {})
                assert json.loads(result)["ok"] is True

    @pytest.mark.parametrize("task_type,write_tool", [
        ("note_taking", "hermes_rag_store"),
        ("maintenance", "hermes_rag_delete"),
    ])
    def test_task_type_write_refused(self, task_type, write_tool):
        provider, _ = _make_provider(task_type=task_type)
        with patch("tools.approval.request_tool_approval", return_value={"approved": True}) as a:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})) as d:
                result = json.loads(provider.handle_tool_call(write_tool, {}))
                assert result["code"] == "delegated_read_only"
                a.assert_not_called()
                d.assert_not_called()


# I12: Concurrent Sub-Agents Isolated
class TestConcurrentIsolation:
    def test_sibling_contexts_independent(self):
        _, ctx1 = _make_provider(child_id="sa-sib-1")
        _, ctx2 = _make_provider(child_id="sa-sib-2")
        mgr1 = SubagentLifecycleManager(ctx1)
        mgr2 = SubagentLifecycleManager(ctx2)
        mgr1.transition("initializing")
        mgr1.transition("running")
        assert mgr2.state == "pending"

    def test_concurrent_tool_calls_isolated(self):
        provider1, _ = _make_provider(child_id="sa-c1")
        provider2, _ = _make_provider(child_id="sa-c2")
        results = []
        lock = threading.Lock()
        def call(provider):
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                with lock:
                    results.append(json.loads(result)["ok"])
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.map(call, [provider1, provider2])
        assert all(results)


# I13: L3 Denied → Dispatch Never Reached
class TestL3Denied:
    def test_denied_l3_never_reaches_dispatch(self):
        provider, _ = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval") as mock_approval:
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call") as mock_dispatch:
                mock_approval.return_value = {"approved": False}
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                # Decision 42: refused before approval (was: approval asked once)
                mock_approval.assert_not_called()
                mock_dispatch.assert_not_called()
                assert json.loads(result)["code"] == "delegated_read_only"

    def test_denied_l3_no_success_audit(self):
        provider, _ = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": False}):
            provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            recorder = get_recorder()
            events = recorder.get_events()
            allowed = [e for e in events if e.allowed is True]
            assert len(allowed) == 0


# I14: Unknown Tool Denied Regardless of State
class TestUnknownToolDenied:
    def test_unknown_denied_with_bypass_and_approval(self):
        provider, _ = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": True}):
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                result = provider.handle_tool_call("unknown_xyz", {}, _continuity_bypass_gate=True)
                assert json.loads(result)["ok"] is False


# I16: Audit Correlation
class TestAuditCorrelationE2E:
    def test_allowed_event_correlates(self):
        provider, ctx = _make_provider(task_type="research")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            recorder = get_recorder()
            events = recorder.get_events()
            allowed = [e for e in events if e.allowed is True]
            assert len(allowed) >= 1
            assert allowed[0].subagent_id == ctx.child_id
            assert allowed[0].tool_name == "hermes_rag_search"

    def test_denied_event_correlates(self):
        provider, ctx = _make_provider(task_type="research")
        provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        recorder = get_recorder()
        events = recorder.get_events()
        denied = [e for e in events if e.allowed is False]
        assert len(denied) >= 1
        assert denied[0].subagent_id == ctx.child_id


# I17: Durable Storage Integration
class TestDurableStorageIntegration:
    def test_audit_store_append_and_verify(self, tmp_path):
        store = create_audit_store(store_path=str(tmp_path / "audit.jsonl"))
        recorder = get_recorder()
        recorder.configure_audit_store(store)
        provider, ctx = _make_provider(task_type="research")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider.handle_tool_call("hermes_rag_search", {"query": "test"})
        events = store.read_events()
        assert len(events) > 0
        result = store.verify_integrity()
        assert result["valid"] is True

    def test_audit_store_failure_doesnt_block(self, tmp_path):
        store = create_audit_store(store_path=str(tmp_path / "audit.jsonl"))
        recorder = get_recorder()
        recorder.configure_audit_store(store)
        with patch.object(store, "append", side_effect=Exception("Disk full")):
            provider, _ = _make_provider(task_type="research")
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                assert json.loads(result)["ok"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
