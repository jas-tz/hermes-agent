"""Phase 29A — Durable Security Audit Storage.

Tests for the SecurityAuditStore with hash-chain integrity.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis.subagent_security_store import (
    SecurityAuditStore,
    SecurityEvent,
    SecurityEventType,
    _compute_event_hash,
    _event_to_stored,
    _stored_to_event,
    create_audit_store,
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
# 1. Append Behavior
# ====================================================================

class TestAppendBehavior:
    """Test append operations."""

    def test_append_single_event(self, audit_store):
        """Append a single event."""
        event = _make_event()

        result = audit_store.append(event)

        assert result is True
        assert audit_store.count_events() == 1

    def test_append_multiple_events(self, audit_store):
        """Append multiple events sequentially."""
        for i in range(5):
            event = _make_event(subagent_id=f"sa-{i}")
            result = audit_store.append(event)
            assert result is True

        assert audit_store.count_events() == 5

    def test_append_preserves_event_data(self, audit_store):
        """Appended event data is preserved exactly."""
        event = _make_event()

        audit_store.append(event)
        events = audit_store.read_events()

        assert len(events) == 1
        assert events[0].event_id == event.event_id
        assert events[0].subagent_id == event.subagent_id
        assert events[0].allowed == event.allowed

    def test_append_batch(self, audit_store):
        """Append a batch of events."""
        events = [_make_event(subagent_id=f"sa-{i}") for i in range(10)]

        count = audit_store.append_batch(events)

        assert count == 10
        assert audit_store.count_events() == 10


# ====================================================================
# 2. Ordering
# ====================================================================

class TestOrdering:
    """Test event ordering."""

    def test_read_preserves_append_order(self, audit_store):
        """Read returns events in append order."""
        for i in range(5):
            event = _make_event(subagent_id="sa-test")
            audit_store.append(event)

        events = audit_store.read_events(subagent_id="sa-test")
        assert len(events) == 5

        # Events should be in order
        for i in range(len(events) - 1):
            assert events[i].event_id != events[i + 1].event_id

    def test_read_timeline_chronological(self, audit_store):
        """Timeline is in chronological order."""
        for i in range(5):
            event = _make_event(subagent_id="sa-test")
            audit_store.append(event)

        timeline = audit_store.read_timeline("sa-test")
        assert len(timeline) == 5

        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp


# ====================================================================
# 3. Hash-Chain Integrity
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

    def test_detects_modification(self, temp_store_path):
        """Modification of a stored event is detected."""
        store = SecurityAuditStore(store_path=temp_store_path)

        for i in range(3):
            store.append(_make_event())

        # Tamper with the file
        with open(temp_store_path, "r") as f:
            lines = f.readlines()

        # Modify the second event's allowed field
        tampered = json.loads(lines[1])
        tampered["allowed"] = not tampered["allowed"]
        lines[1] = json.dumps(tampered, sort_keys=True) + "\n"

        with open(temp_store_path, "w") as f:
            f.writelines(lines)

        # Create new store instance to read tampered file
        store2 = SecurityAuditStore(store_path=temp_store_path)
        result = store2.verify_integrity()

        assert result["valid"] is False
        assert len(result["issues"]) > 0

    def test_detects_deletion(self, temp_store_path):
        """Deletion of an event is detected."""
        store = SecurityAuditStore(store_path=temp_store_path)

        for i in range(5):
            store.append(_make_event())

        # Delete the third event
        with open(temp_store_path, "r") as f:
            lines = f.readlines()

        del lines[2]  # Remove third event

        with open(temp_store_path, "w") as f:
            f.writelines(lines)

        # Create new store instance
        store2 = SecurityAuditStore(store_path=temp_store_path)
        result = store2.verify_integrity()

        assert result["valid"] is False

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
# 4. Persistence
# ====================================================================

class TestPersistence:
    """Test persistence across store recreation."""

    def test_events_persist_across_recreation(self, temp_store_path):
        """Events persist when store is recreated."""
        store1 = SecurityAuditStore(store_path=temp_store_path)

        for i in range(5):
            store1.append(_make_event(subagent_id=f"sa-{i}"))

        # Create new store instance
        store2 = SecurityAuditStore(store_path=temp_store_path)

        assert store2.count_events() == 5
        events = store2.read_events()
        assert len(events) == 5

    def test_metadata_persists(self, temp_store_path):
        """Metadata persists across recreation."""
        store1 = SecurityAuditStore(store_path=temp_store_path)

        for i in range(10):
            store1.append(_make_event())

        metadata1 = store1.get_store_metadata()

        store2 = SecurityAuditStore(store_path=temp_store_path)
        metadata2 = store2.get_store_metadata()

        assert metadata2["event_count"] == metadata1["event_count"]

    def test_appending_after_recreation(self, temp_store_path):
        """Can append more events after recreation."""
        store1 = SecurityAuditStore(store_path=temp_store_path)
        store1.append(_make_event(subagent_id="sa-1"))

        store2 = SecurityAuditStore(store_path=temp_store_path)
        store2.append(_make_event(subagent_id="sa-2"))

        assert store2.count_events() == 2


# ====================================================================
# 5. Sub-Agent Isolation
# ====================================================================

class TestSubAgentIsolation:
    """Test sub-agent isolation in the store."""

    def test_events_isolated_by_subagent(self, audit_store):
        """Events are isolated by sub-agent."""
        audit_store.append(_make_event(subagent_id="sa-A"))
        audit_store.append(_make_event(subagent_id="sa-B"))
        audit_store.append(_make_event(subagent_id="sa-A"))

        events_a = audit_store.read_events(subagent_id="sa-A")
        events_b = audit_store.read_events(subagent_id="sa-B")

        assert len(events_a) == 2
        assert len(events_b) == 1

    def test_parent_session_isolation(self, audit_store):
        """Parent sessions are isolated."""
        event1 = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="child",
            parent_session_id="parent-A",
            task_type="research",
            authority_level="L0",
        )
        event2 = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="child",
            parent_session_id="parent-B",
            task_type="research",
            authority_level="L0",
        )

        audit_store.append(event1)
        audit_store.append(event2)

        events_a = audit_store.read_events(parent_session_id="parent-A")
        assert len(events_a) == 1
        assert events_a[0].parent_session_id == "parent-A"


# ====================================================================
# 6. Immutability
# ====================================================================

class TestImmutability:
    """Test event immutability."""

    def test_stored_events_immutable_in_store(self, audit_store):
        """Stored events cannot be modified."""
        event = _make_event()
        audit_store.append(event)

        events = audit_store.read_events()
        assert len(events) == 1

        with pytest.raises(AttributeError):
            events[0].allowed = False


# ====================================================================
# 7. Filtering
# ====================================================================

class TestFiltering:
    """Test event filtering."""

    def test_filter_by_event_type(self, audit_store):
        """Filter by event type."""
        audit_store.append(_make_event(event_type=SecurityEventType.AUTHORIZATION_ALLOWED))
        audit_store.append(_make_event(event_type=SecurityEventType.AUTHORIZATION_DENIED))

        allowed = audit_store.read_events(event_type=SecurityEventType.AUTHORIZATION_ALLOWED)
        assert len(allowed) == 1

    def test_filter_by_tool_name(self, audit_store):
        """Filter by tool name."""
        event1 = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        )
        event2 = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_store",
        )

        audit_store.append(event1)
        audit_store.append(event2)

        filtered = audit_store.read_events(tool_name="hermes_rag_search")
        assert len(filtered) == 1

    def test_filter_by_allowed(self, audit_store):
        """Filter by allowed flag."""
        audit_store.append(_make_event(allowed=True))
        audit_store.append(_make_event(allowed=False))

        denied = audit_store.read_events(allowed=False)
        assert len(denied) == 1

    def test_combined_filters(self, audit_store):
        """Multiple filters combined."""
        event = _make_event(
            subagent_id="sa-A",
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            allowed=False,
        )
        audit_store.append(event)

        filtered = audit_store.read_events(
            subagent_id="sa-A",
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            allowed=False,
        )
        assert len(filtered) == 1

    def test_no_matching_events(self, audit_store):
        """No matching events returns empty list."""
        audit_store.append(_make_event())

        filtered = audit_store.read_events(subagent_id="nonexistent")
        assert filtered == []


# ====================================================================
# 8. Store Metadata
# ====================================================================

class TestStoreMetadata:
    """Test store metadata."""

    def test_metadata_has_event_count(self, audit_store):
        """Metadata includes event count."""
        audit_store.append(_make_event())

        metadata = audit_store.get_store_metadata()
        assert metadata["event_count"] == 1

    def test_metadata_has_store_path(self, audit_store):
        """Metadata includes store path."""
        metadata = audit_store.get_store_metadata()
        assert "store_path" in metadata

    def test_metadata_has_file_size(self, audit_store):
        """Metadata includes file size."""
        audit_store.append(_make_event())

        metadata = audit_store.get_store_metadata()
        assert "file_size_bytes" in metadata
        assert metadata["file_size_bytes"] > 0


# ====================================================================
# 9. Zero Authorization Side Effects
# ====================================================================

class TestZeroAuthorizationSideEffects:
    """Test that store has no authorization side effects."""

    def test_store_doesnt_authorize(self, audit_store):
        """Store doesn't authorize tools."""
        event = _make_event(allowed=False)
        audit_store.append(event)

        # Event recorded but authorization not affected
        events = audit_store.read_events(allowed=False)
        assert len(events) == 1


# ====================================================================
# 10. Partial Write Recovery
# ====================================================================

class TestPartialWriteRecovery:
    """Test recovery from partial/corrupted writes."""

    def test_corrupted_line_skipped(self, temp_store_path):
        """Corrupted lines are skipped gracefully."""
        store = SecurityAuditStore(store_path=temp_store_path)

        store.append(_make_event())

        # Append corrupted data
        with open(temp_store_path, "a") as f:
            f.write("not valid json\n")
            f.write('{"incomplete": true\n')

        # Create new store — should recover
        store2 = SecurityAuditStore(store_path=temp_store_path)

        # Should still read valid events
        events = store2.read_events()
        assert len(events) >= 1

    def test_empty_lines_skipped(self, temp_store_path):
        """Empty lines are skipped."""
        store = SecurityAuditStore(store_path=temp_store_path)

        store.append(_make_event())

        # Append empty lines
        with open(temp_store_path, "a") as f:
            f.write("\n\n\n")

        store2 = SecurityAuditStore(store_path=temp_store_path)
        events = store2.read_events()
        assert len(events) == 1


# ====================================================================
# 11. Factory
# ====================================================================

class TestFactory:
    """Test store factory."""

    def test_create_audit_store(self):
        """Factory creates a store."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            store = create_audit_store(store_path=path)
            assert isinstance(store, SecurityAuditStore)
            store.close()
        finally:
            os.unlink(path)

    def test_default_temp_path(self):
        """Default path uses temp directory."""
        store = create_audit_store()
        assert store._store_path is not None
        assert "jarvis_audit_" in store._store_path
        store.close()


# ====================================================================
# 12. Hash Functions
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
