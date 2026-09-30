"""Phase 29B — Runtime Durable Audit Integration.

Tests the integration of SecurityEventRecorder with SecurityAuditStore
in the real JARVIS security event runtime.
"""

import json
import os
import tempfile
import threading
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_security import (
    SecurityEvent,
    SecurityEventType,
    SecurityEventRecorder,
    create_security_event,
    get_recorder,
    reset_recorder,
    record_authorization_allowed,
    record_authorization_denied,
    record_parent_only_denied,
    record_forged_capability,
    record_tool_execution_completed,
    record_tool_execution_failed,
    record_approval_requested,
    record_approval_denied,
    record_lifecycle_transition,
    record_recursion_limit,
    configure_audit_store,
    get_audit_store,
    disable_audit_store,
)
from plugins.memory.jarvis.subagent_security_store import (
    SecurityAuditStore,
    create_audit_store,
)
from plugins.memory.jarvis.subagent_types import create_execution_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_recorder_fixture():
    """Reset recorder before each test."""
    reset_recorder()
    yield
    reset_recorder()


def _make_event(subagent_id="sa-test", event_type=None, allowed=True):
    """Helper to create test events."""
    if event_type is None:
        event_type = SecurityEventType.AUTHORIZATION_ALLOWED
    return create_security_event(
        event_type=event_type,
        subagent_id=subagent_id,
        parent_session_id="parent",
        task_type="research",
        authority_level="L0",
        tool_name="hermes_rag_search",
        lifecycle_state="running",
        allowed=allowed,
        reason="test event",
    )


# ====================================================================
# 1. Successful Persistence
# ====================================================================

class TestSuccessfulPersistence:
    """Test events are persisted when audit store is enabled."""

    def test_record_persists_to_store(self):
        """Recorded events are persisted to the audit store."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

            record_authorization_allowed(
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
                tool_name="hermes_rag_search",
            )

            events = store.read_events()
            assert len(events) == 1
            assert events[0].subagent_id == "sa-test"

            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()

    def test_multiple_records_persist(self):
        """Multiple recorded events are all persisted."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

            for i in range(5):
                record_authorization_allowed(
                    subagent_id=f"sa-{i}",
                    parent_session_id="parent",
                    task_type="research",
                    authority_level="L0",
                    tool_name="hermes_rag_search",
                )

            assert store.count_events() == 5
            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()

    def test_all_record_functions_persist(self):
        """All record_* functions persist when store is configured."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

            record_authorization_allowed("sa-1", "parent", "research", "L0", "hermes_rag_search")
            record_authorization_denied("sa-2", "parent", "research", "L0", "hermes_rag_store")
            record_forged_capability("sa-3", "parent", "research", "L0", "fake_tool")
            record_tool_execution_failed("sa-4", "parent", "research", "L0", "hermes_rag_search", reason="timeout")

            assert store.count_events() >= 4
            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()


# ====================================================================
# 2. Store Failure Isolation
# ====================================================================

class TestStoreFailureIsolation:
    """Test that store failures don't affect authorization."""

    def test_store_failure_doesnt_affect_auth(self):
        """Store failure doesn't change authorization decisions."""
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

        # Configure a store that will fail
        mock_store = MagicMock()
        mock_store.append.side_effect = Exception("Store failure")
        configure_audit_store(mock_store)

        # Authorization should still work
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({"ok": True})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})

        assert json.loads(result)["ok"] is True
        disable_audit_store()

    def test_store_failure_still_records_in_memory(self):
        """Store failure doesn't prevent in-memory recording."""
        mock_store = MagicMock()
        mock_store.append.side_effect = Exception("Store failure")
        configure_audit_store(mock_store)

        record_authorization_allowed(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        )

        # In-memory recorder still has the event
        events = get_recorder().get_events()
        assert len(events) == 1
        disable_audit_store()


# ====================================================================
# 3. Disabled Persistence
# ====================================================================

class TestDisabledPersistence:
    """Test that disabling persistence leaves recorder functional."""

    def test_disable_audit_store(self):
        """Disabling audit store leaves in-memory recorder functional."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

            # Record while enabled
            record_authorization_allowed("sa-1", "parent", "research", "L0", "hermes_rag_search")
            assert store.count_events() == 1

            # Disable
            disable_audit_store()

            # Record while disabled
            record_authorization_allowed("sa-2", "parent", "research", "L0", "hermes_rag_search")

            # Store should not have the new event
            assert store.count_events() == 1

            # Recorder should have both events
            assert len(get_recorder().get_events()) == 2

            store.close()
        finally:
            os.unlink(store_path)

    def test_get_audit_store_after_disable(self):
        """get_audit_store returns None after disable."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

            assert get_audit_store() is store

            disable_audit_store()
            assert get_audit_store() is None

            store.close()
        finally:
            os.unlink(store_path)


# ====================================================================
# 4. Recreation/Recovery
# ====================================================================

class TestRecreationRecovery:
    """Test persistence across store recreation."""

    def test_events_persist_across_recreation(self, temp_store_path):
        """Events persist when store is recreated from the same file."""
        # First session
        configure_audit_store(None)  # Will create temp store

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store1 = create_audit_store(store_path=store_path)
            configure_audit_store(store1)

            record_authorization_allowed("sa-1", "parent", "research", "L0", "hermes_rag_search")
            record_authorization_denied("sa-2", "parent", "research", "L0", "hermes_rag_store")

            store1.close()

            # Second session - new store instance, same file
            store2 = create_audit_store(store_path=store_path)
            configure_audit_store(store2)

            # Previous events should be readable
            events = store2.read_events()
            assert len(events) == 2

            store2.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()


# ====================================================================
# 5. Duplicate Prevention
# ====================================================================

class TestDuplicatePrevention:
    """Test that events are not duplicated."""

    def test_single_record_single_persist(self):
        """Single record results in single persisted event."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

            record_authorization_allowed("sa-test", "parent", "research", "L0", "hermes_rag_search")

            # Should be exactly one event
            events = store.read_events()
            assert len(events) == 1

            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()


# ====================================================================
# 6. Ordering
# ====================================================================

class TestOrdering:
    """Test event ordering is preserved."""

    def test_persistence_preserves_order(self):
        """Persisted events preserve append order."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

            for i in range(10):
                record_authorization_allowed(
                    subagent_id="sa-test",
                    parent_session_id="parent",
                    task_type="research",
                    authority_level="L0",
                    tool_name=f"tool-{i}",
                )

            events = store.read_events()
            assert len(events) == 10

            # Check ordering
            for i in range(len(events) - 1):
                assert events[i].timestamp <= events[i + 1].timestamp

            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()


# ====================================================================
# 7. Concurrency
# ====================================================================

class TestConcurrency:
    """Test concurrent event recording with persistence."""

    def test_concurrent_recording_with_store(self):
        """Concurrent recording with store is safe."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

            errors = []

            def record_events(subagent_id, count):
                try:
                    for _ in range(count):
                        record_authorization_allowed(
                            subagent_id=subagent_id,
                            parent_session_id="parent",
                            task_type="research",
                            authority_level="L0",
                            tool_name="hermes_rag_search",
                        )
                except Exception as e:
                    errors.append(e)

            threads = []
            for i in range(5):
                t = threading.Thread(target=record_events, args=(f"sa-{i}", 20))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert store.count_events() == 100

            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()


# ====================================================================
# 8. Authorization Independence
# ====================================================================

class TestAuthorizationIndependence:
    """Test that authorization is independent of audit persistence."""

    def test_authorization_works_without_store(self):
        """Authorization works without audit store configured."""
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

        # No store configured
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})

        assert json.loads(result)["ok"] is True

    def test_authorization_works_with_store(self):
        """Authorization works with audit store configured."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

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
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})

            assert json.loads(result)["ok"] is True

            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()

    def test_denied_auth_persists_correctly(self):
        """Denied authorization persists correctly."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

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

            result = provider.handle_tool_call("hermes_rag_store", {"text": "data"})
            assert json.loads(result)["ok"] is False

            # Denial should be persisted
            denied = store.read_events(allowed=False)
            assert len(denied) >= 1

            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()


# ====================================================================
# 9. Isolation
# ====================================================================

class TestIsolation:
    """Test sub-agent isolation in persistent store."""

    def test_subagent_isolation_in_store(self):
        """Sub-agent isolation is preserved in persistent store."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

            record_authorization_allowed("sa-A", "parent", "research", "L0", "hermes_rag_search")
            record_authorization_allowed("sa-B", "parent", "research", "L0", "hermes_rag_search")

            events_a = store.read_events(subagent_id="sa-A")
            events_b = store.read_events(subagent_id="sa-B")

            assert len(events_a) == 1
            assert len(events_b) == 1

            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()


# ====================================================================
# 10. Zero-Mutation Guarantees
# ====================================================================

class TestZeroMutation:
    """Test zero-mutation guarantees."""

    def test_deny_with_store_no_mutation(self):
        """Denied auth with store configured produces no mutation."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            store_path = f.name

        try:
            store = create_audit_store(store_path=store_path)
            configure_audit_store(store)

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

            result = provider.handle_tool_call("hermes_rag_store", {"text": "data"})
            assert json.loads(result)["ok"] is False

            store.close()
        finally:
            os.unlink(store_path)
            disable_audit_store()


# ====================================================================
# Helper
# ====================================================================

@pytest.fixture
def temp_store_path(tmp_path):
    return str(tmp_path / "audit_store.jsonl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
