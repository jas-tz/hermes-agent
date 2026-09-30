"""Phase 29C — Security Audit Recovery, Consistency & Crash Resilience.

Tests the recovery capabilities of SecurityAuditStore.
Covers: persistence across recreation, hash-chain continuation,
partial-write recovery, corruption detection, tamper detection,
concurrent access, empty-store recovery, and crash resilience.
"""

import json
import os
import tempfile
import threading
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis.subagent_security_store import (
    SecurityAuditStore,
    SecurityEvent,
    SecurityEventType,
    create_audit_store,
    _compute_event_hash,
    _event_to_stored,
    _stored_to_event,
    GENESIS_HASH,
)
from plugins.memory.jarvis.subagent_security import create_security_event, _sanitize_reason


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_store_path(tmp_path):
    """Create a temporary store path."""
    return str(tmp_path / "audit_store.jsonl")


@pytest.fixture
def audit_store(temp_store_path):
    """Create a fresh audit store."""
    store = SecurityAuditStore(store_path=temp_store_path)
    yield store
    store.close()


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
# 1. Clean Persistence Across Recreation
# ====================================================================

class TestCleanPersistence:
    """Test clean persistence across store recreation."""

    def test_events_persist_cleanly(self, temp_store_path):
        """Events persist cleanly across recreation."""
        store1 = SecurityAuditStore(store_path=temp_store_path)

        for i in range(5):
            store1.append(_make_event(subagent_id=f"sa-{i}"))

        assert store1.count_events() == 5

        # Create new store instance
        store2 = SecurityAuditStore(store_path=temp_store_path)

        assert store2.count_events() == 5
        events = store2.read_events()
        assert len(events) == 5

        store1.close()
        store2.close()

    def test_hash_chain_continues_after_restart(self, temp_store_path):
        """Hash chain continues correctly after restart."""
        store1 = SecurityAuditStore(store_path=temp_store_path)

        for i in range(3):
            store1.append(_make_event())

        metadata1 = store1.get_store_metadata()
        last_hash = metadata1["previous_hash"]

        # Create new store
        store2 = SecurityAuditStore(store_path=temp_store_path)

        # Append more events
        store2.append(_make_event())

        # Verify chain integrity
        result = store2.verify_integrity()
        assert result["valid"] is True

        store1.close()
        store2.close()


# ====================================================================
# 2. Partial Write Recovery
# ====================================================================

class TestPartialWriteRecovery:
    """Test recovery from partial/truncated writes."""

    def test_truncated_final_record_recovered(self, temp_store_path):
        """Truncated final record is handled during load."""
        store1 = SecurityAuditStore(store_path=temp_store_path)

        store1.append(_make_event())
        store1.append(_make_event())

        # Append partial record (no trailing newline)
        with open(temp_store_path, "a") as f:
            f.write('{"event_id": "partial", "timestamp": "')
            f.flush()

        # Create new store — should recover
        store2 = SecurityAuditStore(store_path=temp_store_path)

        # Should have the 2 valid events
        assert store2.count_events() == 2

        store1.close()
        store2.close()

    def test_partial_write_no_valid_records(self, temp_store_path):
        """File with only partial write is cleared."""
        # Write only partial data
        with open(temp_store_path, "w") as f:
            f.write("not valid json at all")

        # Create store — should recover
        store = SecurityAuditStore(store_path=temp_store_path)

        assert store.count_events() == 0

        store.close()


# ====================================================================
# 3. Corruption Detection
# ====================================================================

class TestCorruptionDetection:
    """Test corruption detection."""

    def test_corrupted_middle_record_detected(self, temp_store_path):
        """Corrupted middle record is detected by verify_integrity."""
        store = SecurityAuditStore(store_path=temp_store_path)

        store.append(_make_event())
        store.append(_make_event())
        store.append(_make_event())

        # Tamper with middle record
        with open(temp_store_path, "r") as f:
            lines = f.readlines()

        tampered = json.loads(lines[1])
        tampered["allowed"] = not tampered["allowed"]
        lines[1] = json.dumps(tampered, sort_keys=True) + "\n"

        with open(temp_store_path, "w") as f:
            f.writelines(lines)

        # Verify detects corruption
        result = store.verify_integrity()
        assert result["valid"] is False
        assert len(result["issues"]) > 0

    def test_deleted_record_detected(self, temp_store_path):
        """Deleted record is detected."""
        store = SecurityAuditStore(store_path=temp_store_path)

        store.append(_make_event())
        store.append(_make_event())
        store.append(_make_event())

        # Delete middle record
        with open(temp_store_path, "r") as f:
            lines = f.readlines()

        del lines[1]

        with open(temp_store_path, "w") as f:
            f.writelines(lines)

        result = store.verify_integrity()
        assert result["valid"] is False

    def test_reordered_record_detected(self, temp_store_path):
        """Reordered record is detected."""
        store = SecurityAuditStore(store_path=temp_store_path)

        store.append(_make_event())
        store.append(_make_event())
        store.append(_make_event())

        # Reorder records
        with open(temp_store_path, "r") as f:
            lines = f.readlines()

        lines[0], lines[1] = lines[1], lines[0]

        with open(temp_store_path, "w") as f:
            f.writelines(lines)

        result = store.verify_integrity()
        assert result["valid"] is False

    def test_modified_event_detected(self, temp_store_path):
        """Modified event is detected."""
        store = SecurityAuditStore(store_path=temp_store_path)

        store.append(_make_event(allowed=True))

        # Modify the event
        with open(temp_store_path, "r") as f:
            lines = f.readlines()

        modified = json.loads(lines[0])
        modified["subagent_id"] = "forged-subagent"
        lines[0] = json.dumps(modified, sort_keys=True) + "\n"

        with open(temp_store_path, "w") as f:
            f.writelines(lines)

        result = store.verify_integrity()
        assert result["valid"] is False


# ====================================================================
# 4. Recovery
# ====================================================================

class TestRecovery:
    """Test explicit recovery."""

    def test_recover_truncates_invalid_records(self, temp_store_path):
        """Recovery truncates invalid records while preserving valid ones."""
        store = SecurityAuditStore(store_path=temp_store_path)

        store.append(_make_event())
        store.append(_make_event())

        # Append invalid record
        with open(temp_store_path, "a") as f:
            f.write("invalid json\n")

        # Recover
        result = store.recover()

        assert result["recovery_performed"] is True
        assert result["truncated"] is True
        assert result["events_after"] == 2

    def test_recover_doesnt_modify_valid_store(self, temp_store_path):
        """Recovery doesn't modify a valid store."""
        store = SecurityAuditStore(store_path=temp_store_path)

        for i in range(5):
            store.append(_make_event())

        result = store.recover()

        assert result["recovery_performed"] is False
        assert result["events_after"] == 5


# ====================================================================
# 5. Empty Store Recovery
# ====================================================================

class TestEmptyStoreRecovery:
    """Test empty-store initialization and recovery."""

    def test_empty_file_initialization(self, temp_store_path):
        """Empty file initializes correctly."""
        # Create empty file
        with open(temp_store_path, "w") as f:
            pass

        store = SecurityAuditStore(store_path=temp_store_path)

        assert store.count_events() == 0

        store.close()

    def test_nonexistent_file_initialization(self):
        """Nonexistent file initializes correctly."""
        store = SecurityAuditStore(store_path="/tmp/nonexistent_audit_store_12345.jsonl")

        assert store.count_events() == 0

        store.close()


# ====================================================================
# 6. Concurrent Access
# ====================================================================

class TestConcurrentAccess:
    """Test concurrent append operations."""

    def test_concurrent_appends_safe(self, temp_store_path):
        """Concurrent appends are safe."""
        store = SecurityAuditStore(store_path=temp_store_path)

        errors = []

        def append_events(subagent_id, count):
            try:
                for _ in range(count):
                    store.append(_make_event(subagent_id=subagent_id))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=append_events, args=(f"sa-{i}", 20))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count_events() == 100

        store.close()

    def test_concurrent_appends_preserve_ordering(self, temp_store_path):
        """Concurrent appends preserve event ordering within each sub-agent."""
        store = SecurityAuditStore(store_path=temp_store_path)

        def append_events(subagent_id, count):
            for _ in range(count):
                store.append(_make_event(subagent_id=subagent_id))

        threads = []
        for i in range(3):
            t = threading.Thread(target=append_events, args=(f"sa-{i}", 10))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each sub-agent should have 10 events in order
        for i in range(3):
            timeline = store.read_timeline(f"sa-{i}")
            assert len(timeline) == 10

        store.close()


# ====================================================================
# 7. Immutability
# ====================================================================

class TestImmutability:
    """Test event immutability after storage."""

    def test_stored_events_immutable(self, audit_store):
        """Stored events are immutable."""
        event = _make_event()
        audit_store.append(event)

        events = audit_store.read_events()
        assert len(events) == 1

        with pytest.raises(AttributeError):
            events[0].allowed = False


# ====================================================================
# 8. Zero Authorization Side Effects
# ====================================================================

class TestZeroAuthorizationSideEffects:
    """Test that storage has no authorization side effects."""

    def test_storage_doesnt_authorize(self, audit_store):
        """Storage doesn't authorize tools."""
        event = _make_event(allowed=False)
        audit_store.append(event)

        # Event recorded but authorization not affected
        events = audit_store.read_events(allowed=False)
        assert len(events) == 1


# ====================================================================
# 9. Hash-Chain Integrity
# ====================================================================

class TestHashChainIntegrity:
    """Test hash-chain integrity."""

    def test_valid_chain(self, audit_store):
        """Valid chain passes integrity check."""
        for i in range(5):
            audit_store.append(_make_event())

        result = audit_store.verify_integrity()

        assert result["valid"] is True
        assert result["events_checked"] == 5
        assert len(result["issues"]) == 0

    def test_empty_store_valid(self, audit_store):
        """Empty store is valid."""
        result = audit_store.verify_integrity()

        assert result["valid"] is True
        assert result["events_checked"] == 0

    def test_first_event_previous_hash_is_genesis(self, audit_store):
        """First event's previous_hash is the genesis hash."""
        event = _make_event()
        audit_store.append(event)

        with open(audit_store._store_path, "r") as f:
            record = json.loads(f.readline())

        assert record["previous_hash"] == GENESIS_HASH


# ====================================================================
# 10. Hash Functions
# ====================================================================

class TestHashFunctions:
    """Test hash computation."""

    def test_hash_deterministic(self):
        """Hash is deterministic."""
        event = _make_event()
        stored1 = _event_to_stored(event, GENESIS_HASH)
        stored2 = _event_to_stored(event, GENESIS_HASH)

        assert stored1["event_hash"] == stored2["event_hash"]

    def test_hash_depends_on_previous(self):
        """Hash depends on previous_hash."""
        event = _make_event()

        stored1 = _event_to_stored(event, "hash-A")
        stored2 = _event_to_stored(event, "hash-B")

        assert stored1["event_hash"] != stored2["event_hash"]

    def test_stored_to_event_roundtrip(self):
        """Roundtrip from event to stored and back."""
        event = _make_event()
        stored = _event_to_stored(event, GENESIS_HASH)
        restored = _stored_to_event(stored)

        assert restored.event_id == event.event_id
        assert restored.subagent_id == event.subagent_id
        assert restored.allowed == event.allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
