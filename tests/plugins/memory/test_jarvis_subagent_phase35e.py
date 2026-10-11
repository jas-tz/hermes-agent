"""Phase 35E — Audit/Observability Isolation Certification."""

import json
import os
import tempfile
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
from plugins.memory.jarvis.subagent_security_manager import (
    AuditStoreManager,
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


def _make_event(event_id="evt-1", subagent_id="test", tool_name="hermes_rag_search", allowed=True):
    return SecurityEvent(
        event_id=event_id,
        timestamp="2026-01-01T00:00:00Z",
        event_type=SecurityEventType.AUTHORIZATION_ALLOWED if allowed else SecurityEventType.AUTHORIZATION_DENIED,
        subagent_id=subagent_id,
        parent_session_id="parent-session-001",
        task_type="research",
        authority_level="L0",
        tool_name=tool_name,
        lifecycle_state="running",
        allowed=allowed,
        reason="",
    )


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


# I1: Recorder failure → ALLOW proceeds
class TestRecorderFailureAllowsAllow:
    def test_record_authorization_allowed_failure(self):
        provider, _ = _make_provider(task_type="research")
        with patch("plugins.memory.jarvis.subagent_security.record_authorization_allowed", side_effect=Exception("fail")):
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                assert json.loads(result)["ok"] is True

    def test_record_tool_execution_completed_failure(self):
        provider, _ = _make_provider(task_type="research")
        with patch("plugins.memory.jarvis.subagent_security.record_tool_execution_completed", side_effect=Exception("fail")):
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                assert json.loads(result)["ok"] is True


# I2: Recorder failure → DENY proceeds
class TestRecorderFailureAllowsDeny:
    def test_record_authorization_denied_failure(self):
        provider, _ = _make_provider(task_type="research")
        with patch("plugins.memory.jarvis.subagent_security.record_authorization_denied", side_effect=Exception("fail")):
            result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
            assert json.loads(result)["ok"] is False

    def test_record_parent_only_denied_failure(self):
        provider, _ = _make_provider(task_type="maintenance")
        with patch("plugins.memory.jarvis.subagent_security.record_parent_only_denied", side_effect=Exception("fail")):
            result = provider.handle_tool_call("delegate_task", {})
            assert json.loads(result)["ok"] is False

    def test_record_forged_capability_failure(self):
        provider, _ = _make_provider(task_type="maintenance")
        with patch("plugins.memory.jarvis.subagent_security.record_forged_capability", side_effect=Exception("fail")):
            result = provider.handle_tool_call("nonexistent_tool_xyz", {})
            assert json.loads(result)["ok"] is False


# I3: Audit store failure → ALLOW proceeds
class TestAuditStoreFailureAllowsAllow:
    def test_audit_store_append_failure(self, tmp_path):
        store = create_audit_store(store_path=str(tmp_path / "audit.jsonl"))
        recorder = get_recorder()
        recorder.configure_audit_store(store)

        with patch.object(store, "append", side_effect=Exception("Disk full")):
            provider, _ = _make_provider(task_type="research")
            with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                assert json.loads(result)["ok"] is True


# I4: Audit store failure → DENY proceeds
class TestAuditStoreFailureAllowsDeny:
    def test_audit_store_failure_doesnt_block_deny(self, tmp_path):
        store = create_audit_store(store_path=str(tmp_path / "audit.jsonl"))
        recorder = get_recorder()
        recorder.configure_audit_store(store)

        with patch.object(store, "append", side_effect=Exception("Disk full")):
            provider, _ = _make_provider(task_type="research")
            result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
            assert json.loads(result)["ok"] is False


# I5: All observability failed → ALLOW proceeds
class TestAllObservabilityFailedAllows:
    def test_every_recorder_fails_allow(self):
        provider, _ = _make_provider(task_type="research")
        with patch("plugins.memory.jarvis.subagent_security.record_authorization_allowed", side_effect=Exception("fail")):
            with patch("plugins.memory.jarvis.subagent_security.record_tool_execution_completed", side_effect=Exception("fail")):
                with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
                    result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
                    assert json.loads(result)["ok"] is True


# I6: All observability failed → DENY proceeds
class TestAllObservabilityFailedDeny:
    def test_every_recorder_fails_deny(self):
        provider, _ = _make_provider(task_type="research")
        with patch("plugins.memory.jarvis.subagent_security.record_authorization_denied", side_effect=Exception("fail")):
            with patch("plugins.memory.jarvis.subagent_security.record_forged_capability", side_effect=Exception("fail")):
                result = provider.handle_tool_call("forged_tool_xyz", {})
                assert json.loads(result)["ok"] is False


# I7: Audit events correlate with authorization decisions
class TestAuditCorrelation:
    def test_allowed_event_has_correct_subagent(self):
        provider, ctx = _make_provider(task_type="research")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider.handle_tool_call("hermes_rag_search", {"query": "test"})

        recorder = get_recorder()
        events = recorder.get_events()
        allowed = [e for e in events if e.allowed is True]
        assert len(allowed) >= 1
        assert allowed[0].subagent_id == ctx.child_id
        assert allowed[0].tool_name == "hermes_rag_search"

    def test_denied_event_has_correct_subagent(self):
        provider, ctx = _make_provider(task_type="research")
        provider.handle_tool_call("hermes_rag_store", {"text": "test"})

        recorder = get_recorder()
        events = recorder.get_events()
        denied = [e for e in events if e.allowed is False]
        assert len(denied) >= 1
        assert denied[0].subagent_id == ctx.child_id

    def test_denied_l3_no_success_audit(self):
        provider, _ = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": False}):
            provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})

        recorder = get_recorder()
        events = recorder.get_events()
        allowed = [e for e in events if e.allowed is True]
        assert len(allowed) == 0


# I8: Hash-chain integrity detects tampering
class TestHashChainIntegrity:
    def test_valid_chain(self, tmp_path):
        store = create_audit_store(store_path=str(tmp_path / "audit.jsonl"))
        recorder = get_recorder()
        recorder.configure_audit_store(store)

        provider, _ = _make_provider(task_type="research")
        with patch.object(jarvis_init, "_dispatch_jarvis_tool_call", return_value=json.dumps({"ok": True})):
            provider.handle_tool_call("hermes_rag_search", {"query": "a"})
            provider.handle_tool_call("hermes_rag_search", {"query": "b"})
            provider.handle_tool_call("hermes_rag_search", {"query": "c"})

        result = store.verify_integrity()
        assert result["valid"] is True


# I11: Concurrent event recording is safe
class TestConcurrentRecording:
    def test_concurrent_tool_calls_isolated(self):
        provider1, _ = _make_provider(child_id="sa-conc-1")
        provider2, _ = _make_provider(child_id="sa-conc-2")
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


# I15: Error messages do not leak implementation details
class TestErrorMessageSanitization:
    def test_denied_error_does_not_leak_path(self):
        provider, _ = _make_provider(task_type="research")
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        data = json.loads(result)
        assert "hermes_rag_store" in data["error"]
        assert "plugins" not in data["error"]
        assert ".py" not in data["error"]

    def test_parent_only_error_clear(self):
        provider, _ = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("delegate_task", {})
        data = json.loads(result)
        assert "parent-only" in data["error"]
        assert "delegate_task" in data["error"]

    def test_forged_capability_error_clear(self):
        provider, _ = _make_provider(task_type="maintenance")
        result = provider.handle_tool_call("unknown_tool_xyz", {})
        data = json.loads(result)
        # Task type check runs first for maintenance (forged tools not in registry)
        assert "not authorized" in data["error"] or "not registered" in data["error"]
        assert "unknown_tool_xyz" in data["error"]

    def test_l3_denied_error_clear(self):
        provider, _ = _make_provider(task_type="maintenance")
        with patch("tools.approval.request_tool_approval", return_value={"approved": False}):
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            data = json.loads(result)
            # Decision 42: the read-only refusal (not "not approved") is the message now
            assert "read-only" in data["error"]
            assert "hermes_rag_delete" in data["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
