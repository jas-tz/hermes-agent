"""JARVIS Security Audit Store Manager — Phase 29D.

Operational management for the durable security audit store:
- Store rotation (size/count thresholds)
- Archival of sealed store files
- Retention management (time/count-based purging)
- Health monitoring and disk-space guards
- Multi-store read correlation

This module is observational only — it must NEVER affect authorization.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from plugins.memory.jarvis.subagent_security import SecurityEvent, SecurityEventType
from plugins.memory.jarvis.subagent_security_store import (
    SecurityAuditStore,
    create_audit_store,
    GENESIS_HASH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation policy defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_FILE_SIZE_MB = 50
DEFAULT_MAX_EVENTS_PER_FILE = 100_000
DEFAULT_KEEP_DAYS = 30
DEFAULT_KEEP_ARCHIVED = 10
DEFAULT_MIN_DISK_FREE_MB = 100


# ---------------------------------------------------------------------------
# Audit Store Manager
# ---------------------------------------------------------------------------

class AuditStoreManager:
    """Manages a series of audit store files with rotation and archival.

    Observational only — never modifies authorization decisions.
    """

    def __init__(
        self,
        store_dir: str,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
        max_events_per_file: int = DEFAULT_MAX_EVENTS_PER_FILE,
        keep_days: int = DEFAULT_KEEP_DAYS,
        keep_archived: int = DEFAULT_KEEP_ARCHIVED,
        min_disk_free_mb: int = DEFAULT_MIN_DISK_FREE_MB,
    ) -> None:
        self._store_dir = store_dir
        self._archive_dir = os.path.join(store_dir, "archive")
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._max_events_per_file = max_events_per_file
        self._keep_days = keep_days
        self._keep_archived = keep_archived
        self._min_disk_free_bytes = min_disk_free_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._active_store: Optional[SecurityAuditStore] = None

        # Create directories
        os.makedirs(self._store_dir, exist_ok=True)
        os.makedirs(self._archive_dir, exist_ok=True)

        # Initialize active store
        self._open_new_store()

    def _open_new_store(self, cross_file_anchor: Optional[str] = None) -> None:
        """Open the active store file.

        Uses a fixed name so the active store can be reloaded across restarts.
        """
        store_path = os.path.join(self._store_dir, "audit_active.jsonl")

        # If active store already exists, load it (preserve chain)
        if os.path.exists(store_path):
            self._active_store = SecurityAuditStore(
                store_path=store_path,
                cross_file_anchor=cross_file_anchor,
            )
            return

        self._active_store = SecurityAuditStore(
            store_path=store_path,
            cross_file_anchor=cross_file_anchor,
        )

    def _should_rotate(self) -> bool:
        """Check if the active store should be rotated."""
        if self._active_store is None:
            return False

        metadata = self._active_store.get_store_metadata()
        file_size = metadata.get("file_size_bytes", 0)
        event_count = metadata.get("event_count", 0)

        if file_size >= self._max_file_size_bytes:
            return True
        if event_count >= self._max_events_per_file:
            return True
        return False

    def _rotate(self) -> None:
        """Rotate the active store to archive."""
        if self._active_store is None:
            return

        # Get the final hash of the current store for cross-file chaining
        previous_final_hash = self._active_store.get_final_hash()

        # Close current store
        self._active_store.close()

        # Move to archive with timestamp
        old_path = self._active_store._store_path
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"audit_{timestamp}.jsonl"
        archive_path = os.path.join(self._archive_dir, filename)

        try:
            shutil.move(old_path, archive_path)
        except Exception as e:
            logger.warning("Failed to archive audit store: %s", e)

        # Open new store with cross-file anchor
        self._open_new_store(cross_file_anchor=previous_final_hash)

    def append(self, event: SecurityEvent) -> bool:
        """Append an event to the active store."""
        with self._lock:
            # Check disk space
            try:
                disk_usage = shutil.disk_usage(self._store_dir)
                if disk_usage.free < self._min_disk_free_bytes:
                    logger.warning("Disk space low, skipping audit append")
                    return False
            except Exception:
                pass

            # Check rotation
            if self._should_rotate():
                self._rotate()

            return self._active_store.append(event)

    def append_batch(self, events: List[SecurityEvent]) -> int:
        """Append a batch of events."""
        count = 0
        for event in events:
            if self.append(event):
                count += 1
        return count

    def read_events(
        self,
        subagent_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        event_type: Optional[SecurityEventType] = None,
        tool_name: Optional[str] = None,
        task_type: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        allowed: Optional[bool] = None,
    ) -> List[SecurityEvent]:
        """Read events across all stores (active + archived)."""
        results = []

        # Read from active store
        if self._active_store is not None:
            results.extend(self._active_store.read_events(
                subagent_id=subagent_id,
                parent_session_id=parent_session_id,
                event_type=event_type,
                tool_name=tool_name,
                task_type=task_type,
                lifecycle_state=lifecycle_state,
                allowed=allowed,
            ))

        # Read from archived stores
        try:
            for filename in sorted(os.listdir(self._archive_dir)):
                if not filename.endswith(".jsonl"):
                    continue
                archive_path = os.path.join(self._archive_dir, filename)
                store = SecurityAuditStore(store_path=archive_path)
                results.extend(store.read_events(
                    subagent_id=subagent_id,
                    parent_session_id=parent_session_id,
                    event_type=event_type,
                    tool_name=tool_name,
                    task_type=task_type,
                    lifecycle_state=lifecycle_state,
                    allowed=allowed,
                ))
                store.close()
        except Exception as e:
            logger.debug("Failed to read archived stores: %s", e)

        return results

    def read_timeline(self, subagent_id: str) -> List[SecurityEvent]:
        """Read a sub-agent's timeline across all stores."""
        events = self.read_events(subagent_id=subagent_id)
        events.sort(key=lambda e: (e.timestamp, e.event_id))
        return events

    def count_events(self) -> int:
        """Count events across all stores."""
        total = 0

        if self._active_store is not None:
            total += self._active_store.count_events()

        try:
            for filename in os.listdir(self._archive_dir):
                if not filename.endswith(".jsonl"):
                    continue
                archive_path = os.path.join(self._archive_dir, filename)
                store = SecurityAuditStore(store_path=archive_path)
                total += store.count_events()
                store.close()
        except Exception:
            pass

        return total

    def get_store_health(self) -> Dict[str, Any]:
        """Get store health metrics."""
        try:
            disk_usage = shutil.disk_usage(self._store_dir)
            disk_free_bytes = disk_usage.free
        except Exception:
            disk_free_bytes = 0

        active_metadata = {}
        if self._active_store is not None:
            active_metadata = self._active_store.get_store_metadata()

        archived_count = 0
        try:
            archived_count = len([
                f for f in os.listdir(self._archive_dir) if f.endswith(".jsonl")
            ])
        except Exception:
            pass

        return {
            "disk_free_bytes": disk_free_bytes,
            "disk_free_mb": disk_free_bytes / (1024 * 1024),
            "active_store": active_metadata,
            "archived_store_count": archived_count,
            "total_events": self.count_events(),
            "disk_space_ok": disk_free_bytes >= self._min_disk_free_bytes,
        }

    def get_store_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across all stores."""
        all_events = self.read_events()

        return {
            "total_events": len(all_events),
            "allowed_count": sum(1 for e in all_events if e.allowed),
            "denied_count": sum(1 for e in all_events if not e.allowed),
            "forged_capability_count": sum(
                1 for e in all_events if e.event_type == SecurityEventType.FORGED_CAPABILITY
            ),
            "approval_requested_count": sum(
                1 for e in all_events if e.event_type == SecurityEventType.APPROVAL_REQUESTED
            ),
            "approval_denied_count": sum(
                1 for e in all_events if e.event_type == SecurityEventType.APPROVAL_DENIED
            ),
            "execution_failure_count": sum(
                1 for e in all_events if e.event_type == SecurityEventType.TOOL_EXECUTION_FAILED
            ),
            "lifecycle_transition_count": sum(
                1 for e in all_events if e.event_type == SecurityEventType.LIFECYCLE_TRANSITION
            ),
            "unique_subagents": len(set(e.subagent_id for e in all_events)),
            "unique_tools": len(set(e.tool_name for e in all_events if e.tool_name)),
        }

    def archive_old_stores(self, keep_n: Optional[int] = None) -> int:
        """Ensure only keep_n archived stores remain."""
        if keep_n is None:
            keep_n = self._keep_archived

        try:
            archived_files = sorted([
                f for f in os.listdir(self._archive_dir) if f.endswith(".jsonl")
            ])
            removed = 0
            while len(archived_files) > keep_n:
                oldest = archived_files.pop(0)
                archive_path = os.path.join(self._archive_dir, oldest)
                try:
                    os.unlink(archive_path)
                    removed += 1
                except Exception as e:
                    logger.warning("Failed to remove archived store: %s", e)
            return removed
        except Exception:
            return 0

    def purge_archived_stores(self, keep_days: Optional[int] = None) -> int:
        """Remove archived stores older than keep_days."""
        if keep_days is None:
            keep_days = self._keep_days

        removed = 0
        cutoff = datetime.now(timezone.utc).timestamp() - (keep_days * 86400)

        try:
            for filename in os.listdir(self._archive_dir):
                if not filename.endswith(".jsonl"):
                    continue
                archive_path = os.path.join(self._archive_dir, filename)
                try:
                    mtime = os.path.getmtime(archive_path)
                    if mtime < cutoff:
                        os.unlink(archive_path)
                        removed += 1
                except Exception as e:
                    logger.warning("Failed to purge archived store: %s", e)
        except Exception:
            pass

        return removed

    def verify_all_integrity(self) -> Dict[str, Any]:
        """Verify integrity of all stores.

        Phase 29D.1: Cross-file anchor chain verification.
        Verifies:
        - Each store's internal hash chain
        - Store ordering (by filename/timestamp)
        - Each store's anchor matches previous store's final hash
        """
        results = {
            "active_store_valid": True,
            "archived_stores_valid": True,
            "cross_file_chain_valid": True,
            "issues": [],
        }

        # Collect all stores in order (archived first, then active)
        all_stores = []

        try:
            for filename in sorted(os.listdir(self._archive_dir)):
                if not filename.endswith(".jsonl"):
                    continue
                archive_path = os.path.join(self._archive_dir, filename)
                store = SecurityAuditStore(store_path=archive_path)
                all_stores.append((filename, store))
        except Exception:
            pass

        if self._active_store is not None:
            active_path = self._active_store._store_path
            active_filename = os.path.basename(active_path)
            # Create a wrapper for the active store to read its metadata
            all_stores.append(("active:" + active_filename, self._active_store))

        # Verify each store's internal integrity
        for filename, store in all_stores:
            result = store.verify_integrity()
            if not result["valid"]:
                results["archived_stores_valid"] = False
                results["issues"].extend([
                    f"{filename}: {issue}" for issue in result.get("issues", [])
                ])

        # Verify cross-file anchor chain
        previous_final_hash = GENESIS_HASH
        for filename, store in all_stores:
            anchor = store.get_anchor_hash()
            if anchor != previous_final_hash:
                results["cross_file_chain_valid"] = False
                results["issues"].append(
                    f"{filename}: cross-file anchor mismatch "
                    f"(expected {previous_final_hash[:16]}..., got {anchor[:16]}...)"
                )
            previous_final_hash = store.get_final_hash()

        results["overall_valid"] = (
            results["active_store_valid"]
            and results["archived_stores_valid"]
            and results["cross_file_chain_valid"]
        )
        return results

    def get_current_file_path(self) -> Optional[str]:
        """Get the path of the active store file."""
        if self._active_store is not None:
            return self._active_store._store_path
        return None

    def close(self) -> None:
        """Graceful shutdown."""
        if self._active_store is not None:
            try:
                self._active_store.close()
            except Exception:
                pass
            self._active_store = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_audit_manager(
    store_dir: str,
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    max_events_per_file: int = DEFAULT_MAX_EVENTS_PER_FILE,
    keep_days: int = DEFAULT_KEEP_DAYS,
    keep_archived: int = DEFAULT_KEEP_ARCHIVED,
    min_disk_free_mb: int = DEFAULT_MIN_DISK_FREE_MB,
) -> AuditStoreManager:
    """Create a new AuditStoreManager."""
    return AuditStoreManager(
        store_dir=store_dir,
        max_file_size_mb=max_file_size_mb,
        max_events_per_file=max_events_per_file,
        keep_days=keep_days,
        keep_archived=keep_archived,
        min_disk_free_mb=min_disk_free_mb,
    )
