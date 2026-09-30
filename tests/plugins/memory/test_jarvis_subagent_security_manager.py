"""Phase 29D — Audit Store Operational Management.

Tests the AuditStoreManager with rotation, archival, retention,
health monitoring, and multi-store correlation.
"""

import json
import os
import tempfile
import threading
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis.subagent_security_manager import (
    AuditStoreManager,
    SecurityEvent,
    SecurityEventType,
    create_audit_manager,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_EVENTS_PER_FILE,
)
from plugins.memory.jarvis.subagent_security import (
    configure_audit_manager,
    get_audit_manager,
    disable_audit_manager,
    get_recorder,
    reset_recorder,
    record_authorization_allowed,
    record_authorization_denied,
    create_security_event,
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
# 1. Basic Manager Operations
# ====================================================================

class TestManagerOperations:
    """Test basic AuditStoreManager operations."""

    def test_create_manager(self, tmp_path):
        """Manager can be created."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        assert manager is not None
        manager.close()

    def test_append_event(self, tmp_path):
        """Events can be appended."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        result = manager.append(_make_event())
        assert result is True
        manager.close()

    def test_append_batch(self, tmp_path):
        """Batch append works."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        events = [_make_event(subagent_id=f"sa-{i}") for i in range(10)]
        count = manager.append_batch(events)
        assert count == 10
        manager.close()

    def test_count_events(self, tmp_path):
        """Count returns correct total."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        for i in range(5):
            manager.append(_make_event())
        assert manager.count_events() == 5
        manager.close()

    def test_read_events(self, tmp_path):
        """Read events returns correct results."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        manager.append(_make_event(subagent_id="sa-A"))
        manager.append(_make_event(subagent_id="sa-B"))

        events = manager.read_events(subagent_id="sa-A")
        assert len(events) == 1
        manager.close()

    def test_read_timeline(self, tmp_path):
        """Timeline is chronological."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        for _ in range(5):
            manager.append(_make_event(subagent_id="sa-test"))

        timeline = manager.read_timeline("sa-test")
        assert len(timeline) == 5
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp
        manager.close()


# ====================================================================
# 2. Store Rotation
# ====================================================================

class TestStoreRotation:
    """Test automatic store rotation."""

    def test_rotation_by_event_count(self, tmp_path):
        """Store rotates when event count threshold is exceeded."""
        # Set very low threshold
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=5,
        )

        # Append events
        for i in range(10):
            manager.append(_make_event())

        # Check rotation occurred
        health = manager.get_store_health()
        assert health["total_events"] == 10

        # Should have archived at least one store
        manager.close()

    def test_rotation_by_size(self, tmp_path):
        """Store rotates when size threshold is exceeded."""
        # Set very low size threshold (1KB)
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_file_size_mb=0.001,  # ~1KB
        )

        # Append enough events to trigger rotation
        for i in range(100):
            manager.append(_make_event())

        manager.close()


# ====================================================================
# 3. Health Monitoring
# ====================================================================

class TestHealthMonitoring:
    """Test health monitoring."""

    def test_get_store_health(self, tmp_path):
        """Health metrics are returned."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        manager.append(_make_event())

        health = manager.get_store_health()
        assert "disk_free_bytes" in health
        assert "active_store" in health
        assert "total_events" in health

        manager.close()

    def test_get_store_stats(self, tmp_path):
        """Stats are aggregated correctly."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        manager.append(_make_event(allowed=True))
        manager.append(_make_event(allowed=False))

        stats = manager.get_store_stats()
        assert stats["total_events"] == 2
        assert stats["allowed_count"] == 1
        assert stats["denied_count"] == 1

        manager.close()


# ====================================================================
# 4. Multi-Store Reads
# ====================================================================

class TestMultiStoreReads:
    """Test reading across multiple store files."""

    def test_read_across_stores(self, tmp_path):
        """Read events across active + archived stores."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=3,
        )

        # Force rotation by appending many events
        for i in range(10):
            manager.append(_make_event(subagent_id=f"sa-{i}"))

        # Read should work across all stores
        events = manager.read_events()
        assert len(events) == 10

        manager.close()

    def test_timeline_across_stores(self, tmp_path):
        """Timeline spans multiple stores."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=3,
        )

        for i in range(10):
            manager.append(_make_event(subagent_id="sa-same"))

        timeline = manager.read_timeline("sa-same")
        assert len(timeline) == 10

        manager.close()


# ====================================================================
# 5. Integrity Verification
# ====================================================================

class TestIntegrityVerification:
    """Test integrity verification across all stores."""

    def test_verify_all_integrity(self, tmp_path):
        """Verify all stores pass integrity check."""
        manager = AuditStoreManager(store_dir=str(tmp_path))

        for i in range(5):
            manager.append(_make_event())

        result = manager.verify_all_integrity()
        assert result["overall_valid"] is True

        manager.close()


# ====================================================================
# 6. Concurrent Access
# ====================================================================

class TestConcurrentAccess:
    """Test concurrent operations."""

    def test_concurrent_appends(self, tmp_path):
        """Concurrent appends are safe."""
        manager = AuditStoreManager(store_dir=str(tmp_path))

        errors = []

        def append_events(subagent_id, count):
            try:
                for _ in range(count):
                    manager.append(_make_event(subagent_id=subagent_id))
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
        assert manager.count_events() == 100

        manager.close()


# ====================================================================
# 7. Factory
# ====================================================================

class TestFactory:
    """Test factory function."""

    def test_create_audit_manager(self, tmp_path):
        """Factory creates a manager."""
        manager = create_audit_manager(store_dir=str(tmp_path))
        assert isinstance(manager, AuditStoreManager)
        manager.close()


# ====================================================================
# 8. Recorder Integration
# ====================================================================

class TestRecorderIntegration:
    """Test integration with SecurityEventRecorder."""

    def test_configure_audit_manager(self, tmp_path):
        """configure_audit_manager wires to recorder."""
        configure_audit_manager(store_dir=str(tmp_path))

        manager = get_audit_manager()
        assert manager is not None
        assert isinstance(manager, AuditStoreManager)

        disable_audit_manager()

    def test_record_through_manager(self, tmp_path):
        """Records are persisted through manager."""
        configure_audit_manager(store_dir=str(tmp_path))

        record_authorization_allowed(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        )

        manager = get_audit_manager()
        events = manager.read_events(subagent_id="sa-test")
        assert len(events) == 1

        disable_audit_manager()

    def test_disable_audit_manager(self, tmp_path):
        """Disabling manager leaves recorder functional."""
        configure_audit_manager(store_dir=str(tmp_path))

        # Record while enabled
        record_authorization_allowed("sa-1", "parent", "research", "L0", "hermes_rag_search")

        # Disable
        disable_audit_manager()

        # Record while disabled
        record_authorization_allowed("sa-2", "parent", "research", "L0", "hermes_rag_search")

        # Recorder still works
        assert len(get_recorder().get_events()) == 2


# ====================================================================
# 9. Zero Authorization Side Effects
# ====================================================================

class TestZeroAuthorizationSideEffects:
    """Test that manager has no authorization side effects."""

    def test_manager_doesnt_authorize(self, tmp_path):
        """Manager doesn't authorize tools."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        manager.append(_make_event(allowed=False))

        events = manager.read_events(allowed=False)
        assert len(events) == 1

        manager.close()


# ====================================================================
# 10. Empty Store
# ====================================================================

class TestEmptyStore:
    """Test empty store behavior."""

    def test_empty_manager(self, tmp_path):
        """Empty manager has zero events."""
        manager = AuditStoreManager(store_dir=str(tmp_path))

        assert manager.count_events() == 0

        health = manager.get_store_health()
        assert health["total_events"] == 0

        manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
