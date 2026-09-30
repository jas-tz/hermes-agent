"""Phase 29D.1 — Cross-File Audit Chain Integrity.

Tests cross-file hash-anchor chaining in AuditStoreManager.
Covers anchor propagation, chain verification, missing/reordered stores,
and tamper detection.
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
from plugins.memory.jarvis.subagent_security_store import GENESIS_HASH
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
# 1. Anchor Propagation
# ====================================================================

class TestAnchorPropagation:
    """Test cross-file anchor propagation."""

    def test_first_store_has_genesis_anchor(self, tmp_path):
        """First store uses GENESIS_HASH as anchor."""
        manager = AuditStoreManager(store_dir=str(tmp_path))

        anchor = manager._active_store.get_anchor_hash()
        assert anchor == GENESIS_HASH

        manager.close()

    def test_anchor_propagates_on_rotation(self, tmp_path):
        """Anchor is propagated to new store after rotation."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=3,
        )

        # Append events to fill first store
        for i in range(3):
            manager.append(_make_event(subagent_id=f"sa-{i}"))

        # Store first store's final hash
        first_store_final = manager._active_store.get_final_hash()

        # Append one more to trigger rotation
        manager.append(_make_event(subagent_id="sa-3"))

        # New store should have the first store's final hash as anchor
        new_store_anchor = manager._active_store.get_anchor_hash()
        assert new_store_anchor == first_store_final

        manager.close()

    def test_multiple_rotations_chain(self, tmp_path):
        """Multiple rotations create a chain of anchors."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=2,
        )

        # Capture chain links at each rotation point
        chain_links = []

        # Track previous store's final hash
        prev_final = manager._active_store.get_final_hash()
        prev_anchor = manager._active_store.get_anchor_hash()

        for i in range(6):
            manager.append(_make_event(subagent_id=f"sa-{i}"))

            current_anchor = manager._active_store.get_anchor_hash()
            current_final = manager._active_store.get_final_hash()

            # Check if rotation occurred
            if current_anchor != prev_anchor:
                # Rotation happened - new store's anchor should equal previous store's final
                chain_links.append((prev_final, current_anchor))
                prev_anchor = current_anchor

            prev_final = current_final

        manager.close()

        # Verify chain: each rotation's new anchor should equal previous store's final
        for prev_final_hash, new_anchor in chain_links:
            assert new_anchor == prev_final_hash, (
                f"Chain broken: new store anchor={new_anchor[:16]}... "
                f"!= previous store final={prev_final_hash[:16]}..."
            )


# ====================================================================
# 2. Chain Verification
# ====================================================================

class TestChainVerification:
    """Test cross-file chain verification."""

    def test_verify_all_integrity_single_store(self, tmp_path):
        """Single store passes integrity check."""
        manager = AuditStoreManager(store_dir=str(tmp_path))

        for i in range(5):
            manager.append(_make_event(subagent_id=f"sa-{i}"))

        result = manager.verify_all_integrity()
        assert result["overall_valid"] is True
        assert result["cross_file_chain_valid"] is True

        manager.close()

    def test_verify_all_integrity_with_rotation(self, tmp_path):
        """Rotated stores pass cross-file integrity check."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=3,
        )

        for i in range(10):
            manager.append(_make_event(subagent_id=f"sa-{i}"))

        result = manager.verify_all_integrity()
        if not result["overall_valid"]:
            print(f"Active valid: {result.get('active_store_valid')}")
            print(f"Archived valid: {result.get('archived_stores_valid')}")
            print(f"Cross-file valid: {result.get('cross_file_chain_valid')}")
            print(f"Issues: {result.get('issues', [])}")
        assert result["overall_valid"] is True

        manager.close()

    def test_missing_middle_store_detected(self, tmp_path):
        """Missing middle store breaks chain.

        Note: This test requires cross-file chain verification that detects
        gaps in the store sequence. The current implementation verifies
        anchor matching between consecutive stores in sorted order.
        """
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=2,
        )

        # Create stores - need at least 2 archived files to have a "middle"
        for i in range(10):
            manager.append(_make_event(subagent_id=f"sa-{i}"))

        manager.close()

        # Delete a middle archived file
        archive_dir = os.path.join(str(tmp_path), "archive")
        archived_files = sorted([f for f in os.listdir(archive_dir) if f.endswith(".jsonl")])
        if len(archived_files) > 1:
            middle_file = os.path.join(archive_dir, archived_files[len(archived_files) // 2])
            os.unlink(middle_file)

        # Recreate manager
        manager2 = AuditStoreManager(store_dir=str(tmp_path))

        # Chain should be broken
        result = manager2.verify_all_integrity()
        # The chain is broken because the remaining stores' anchors don't match
        # their new predecessors in the sequence
        has_chain_issues = not result["cross_file_chain_valid"] or any(
            "anchor mismatch" in issue for issue in result.get("issues", [])
        )
        assert has_chain_issues or not result["overall_valid"]

        manager2.close()

    def test_reordered_stores_detected(self, tmp_path):
        """Reordered stores break chain."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=2,
        )

        # Create stores - need at least 2 archived files
        for i in range(10):
            manager.append(_make_event(subagent_id=f"sa-{i}"))

        manager.close()

        # Rename archived files to swap order
        archive_dir = os.path.join(str(tmp_path), "archive")
        archived_files = sorted([f for f in os.listdir(archive_dir) if f.endswith(".jsonl")])
        if len(archived_files) >= 2:
            # Swap first two files by renaming
            first = os.path.join(archive_dir, archived_files[0])
            second = os.path.join(archive_dir, archived_files[1])
            os.rename(first, first + ".tmp")
            os.rename(second, first)
            os.rename(first + ".tmp", second)

        manager2 = AuditStoreManager(store_dir=str(tmp_path))

        result = manager2.verify_all_integrity()
        # Reordering breaks the anchor chain
        has_chain_issues = not result["cross_file_chain_valid"] or any(
            "anchor mismatch" in issue for issue in result.get("issues", [])
        )
        assert has_chain_issues or not result["overall_valid"]

        manager2.close()

    def test_modified_anchor_detected(self, tmp_path):
        """Modified anchor breaks chain."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=2,
        )

        # Create stores
        for i in range(4):
            manager.append(_make_event(subagent_id=f"sa-{i}"))

        manager.close()

        # Modify anchor in active store
        active_path = os.path.join(str(tmp_path), "audit_active.jsonl")
        if os.path.exists(active_path):
            with open(active_path, "r") as f:
                lines = f.readlines()
            if lines:
                record = json.loads(lines[0])
                record["previous_hash"] = "forged_hash_value"
                lines[0] = json.dumps(record, sort_keys=True) + "\n"
                with open(active_path, "w") as f:
                    f.writelines(lines)

        manager2 = AuditStoreManager(store_dir=str(tmp_path))

        result = manager2.verify_all_integrity()
        assert result["cross_file_chain_valid"] is False

        manager2.close()

    def test_modified_previous_final_hash_detected(self, tmp_path):
        """Modifying a previous store's final event hash breaks chain."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=2,
        )

        # Create stores
        for i in range(4):
            manager.append(_make_event(subagent_id=f"sa-{i}"))

        manager.close()

        # Modify last event in first archived store
        archive_dir = os.path.join(str(tmp_path), "archive")
        archived_files = sorted([f for f in os.listdir(archive_dir) if f.endswith(".jsonl")])
        if archived_files:
            first_archive = os.path.join(archive_dir, archived_files[0])
            with open(first_archive, "r") as f:
                lines = f.readlines()
            if lines:
                # Modify last event's hash
                last_record = json.loads(lines[-1])
                last_record["event_hash"] = "forged_hash_value"
                lines[-1] = json.dumps(last_record, sort_keys=True) + "\n"
                with open(first_archive, "w") as f:
                    f.writelines(lines)

        manager2 = AuditStoreManager(store_dir=str(tmp_path))

        result = manager2.verify_all_integrity()
        assert result["cross_file_chain_valid"] is False

        manager2.close()


# ====================================================================
# 3. Empty Rotation Boundary
# ====================================================================

class TestEmptyRotationBoundary:
    """Test behavior at empty rotation boundary."""

    def test_empty_store_then_rotation(self, tmp_path):
        """Empty store followed by rotation works."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=1,
        )

        # First store gets 1 event
        manager.append(_make_event(subagent_id="sa-0"))

        # Second store gets events
        manager.append(_make_event(subagent_id="sa-1"))
        manager.append(_make_event(subagent_id="sa-2"))

        result = manager.verify_all_integrity()
        assert result["overall_valid"] is True

        manager.close()

    def test_no_events_then_events(self, tmp_path):
        """Starting empty then adding events works."""
        manager = AuditStoreManager(store_dir=str(tmp_path))

        result1 = manager.verify_all_integrity()
        assert result1["overall_valid"] is True

        manager.append(_make_event())

        result2 = manager.verify_all_integrity()
        assert result2["overall_valid"] is True

        manager.close()


# ====================================================================
# 4. Restart/Recovery with Anchors
# ====================================================================

class TestRestartRecoveryWithAnchors:
    """Test restart and recovery with cross-file anchors."""

    def test_anchors_persist_across_recreation(self, tmp_path):
        """Anchors survive store recreation."""
        # First session
        manager1 = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=2,
        )

        for i in range(6):
            manager1.append(_make_event(subagent_id=f"sa-{i}"))

        # Get chain info before close
        archive_dir = os.path.join(str(tmp_path), "archive")
        archived_files_1 = sorted([f for f in os.listdir(archive_dir) if f.endswith(".jsonl")])

        manager1.close()

        # Second session
        manager2 = AuditStoreManager(store_dir=str(tmp_path))
        archived_files_2 = sorted([f for f in os.listdir(archive_dir) if f.endswith(".jsonl")])

        # Same files should exist
        assert archived_files_1 == archived_files_2

        # Chain should still be valid
        result = manager2.verify_all_integrity()
        assert result["overall_valid"] is True

        manager2.close()

    def test_append_after_recreation_maintains_chain(self, tmp_path):
        """Appending after recreation maintains chain."""
        manager1 = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=2,
        )

        for i in range(4):
            manager1.append(_make_event(subagent_id=f"sa-{i}"))

        manager1.close()

        # Second session
        manager2 = AuditStoreManager(store_dir=str(tmp_path))

        # Append more events
        for i in range(4, 8):
            manager2.append(_make_event(subagent_id=f"sa-{i}"))

        result = manager2.verify_all_integrity()
        assert result["overall_valid"] is True

        manager2.close()


# ====================================================================
# 5. Concurrent Rotation
# ====================================================================

class TestConcurrentRotation:
    """Test concurrent rotation operations."""

    def test_concurrent_rotation_safe(self, tmp_path):
        """Concurrent appends with rotation are safe."""
        manager = AuditStoreManager(
            store_dir=str(tmp_path),
            max_events_per_file=10,
        )

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

        # Verify chain integrity after concurrent operations
        result = manager.verify_all_integrity()
        assert result["overall_valid"] is True

        manager.close()


# ====================================================================
# 6. Zero Authorization Side Effects
# ====================================================================

class TestZeroAuthorizationSideEffects:
    """Test that chain integrity doesn't affect authorization."""

    def test_chain_verification_doesnt_authorize(self, tmp_path):
        """Chain verification doesn't authorize tools."""
        manager = AuditStoreManager(store_dir=str(tmp_path))
        manager.append(_make_event(allowed=False))

        # Verify doesn't change anything
        result = manager.verify_all_integrity()
        assert result["overall_valid"] is True

        events = manager.read_events(allowed=False)
        assert len(events) == 1

        manager.close()


# ====================================================================
# Helper
# ====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
