"""JARVIS Durable Security Audit Store — Phase 29A.

Append-only persistent security audit store with hash-chain integrity.

This module provides durable storage for SecurityEvent records across
process restarts. It is observational only and must NEVER affect authorization.

Uses a JSONL append-only log with SHA-256 hash chaining for tamper detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from plugins.memory.jarvis.subagent_security import SecurityEvent, SecurityEventType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stored event (extends SecurityEvent with hash-chain fields)
# ---------------------------------------------------------------------------

GENESIS_HASH = "0" * 64


def _compute_event_hash(event_data: Dict[str, Any], previous_hash: str) -> str:
    """Compute deterministic hash for an event."""
    payload = json.dumps(event_data, sort_keys=True) + previous_hash
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_to_stored(event: SecurityEvent, previous_hash: str) -> Dict[str, Any]:
    """Convert SecurityEvent to stored dict with hash fields."""
    event_data = {
        "event_id": event.event_id,
        "timestamp": event.timestamp,
        "event_type": event.event_type.value,
        "subagent_id": event.subagent_id,
        "parent_session_id": event.parent_session_id,
        "task_type": event.task_type,
        "authority_level": event.authority_level,
        "tool_name": event.tool_name,
        "lifecycle_state": event.lifecycle_state,
        "allowed": event.allowed,
        "reason": event.reason,
    }
    event_hash = _compute_event_hash(event_data, previous_hash)
    event_data["previous_hash"] = previous_hash
    event_data["event_hash"] = event_hash
    return event_data


def _stored_to_event(stored: Dict[str, Any]) -> SecurityEvent:
    """Convert stored dict back to SecurityEvent (without hash fields)."""
    return SecurityEvent(
        event_id=stored["event_id"],
        timestamp=stored["timestamp"],
        event_type=SecurityEventType(stored["event_type"]),
        subagent_id=stored["subagent_id"],
        parent_session_id=stored["parent_session_id"],
        task_type=stored["task_type"],
        authority_level=stored["authority_level"],
        tool_name=stored["tool_name"],
        lifecycle_state=stored["lifecycle_state"],
        allowed=stored["allowed"],
        reason=stored["reason"],
    )


# ---------------------------------------------------------------------------
# Durable Audit Store
# ---------------------------------------------------------------------------

class SecurityAuditStore:
    """Append-only durable security audit store with hash-chain integrity.

    Persists SecurityEvent records to a JSONL file.
    Each event contains previous_hash and event_hash for tamper detection.

    Phase 29D.1: Supports cross-file anchor for chain-of-chains integrity.
    The first record's previous_hash is the anchor from the previous store file.

    Observational only — never modifies authorization.
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        cross_file_anchor: Optional[str] = None,
    ) -> None:
        """Initialize the store.

        Args:
            store_path: Path to the audit log file. If None, uses a temp file.
            cross_file_anchor: Hash of the final event from the previous store file.
                If None, uses GENESIS_HASH (first file in a chain).
        """
        if store_path is None:
            self._store_path = os.path.join(
                tempfile.gettempdir(),
                f"jarvis_audit_{os.getpid()}.jsonl"
            )
        else:
            self._store_path = store_path

        anchor = cross_file_anchor if cross_file_anchor is not None else GENESIS_HASH

        self._lock = threading.Lock()
        self._cross_file_anchor = anchor
        self._previous_hash = anchor  # First event uses anchor as previous_hash
        self._event_count = 0

        # Load existing state if file exists
        self._load_state()

    def get_final_hash(self) -> str:
        """Get the hash of the final event in this store.

        For a store with events, returns the last event's hash.
        For an empty store, returns the cross_file_anchor.
        Used by AuditStoreManager to chain rotated stores.
        """
        return self._previous_hash

    def get_anchor_hash(self) -> str:
        """Get the cross-file anchor hash.

        This is the value that should match the previous store's get_final_hash().
        For the first store in a chain, this is GENESIS_HASH.
        """
        return self._cross_file_anchor

    def _load_state(self) -> None:
        """Load existing state from the store file.

        Phase 29C: Enhanced recovery from partial writes and corruption.
        Phase 29D.1: Restore cross_file_anchor from first record's previous_hash.
        """
        try:
            if not os.path.exists(self._store_path):
                return

            with open(self._store_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content:
                return

            # Check for partial final write (no trailing newline)
            if not content.endswith("\n"):
                # Find the last complete line
                last_newline = content.rfind("\n")
                if last_newline == -1:
                    # Entire file is a single partial write - clear it
                    logger.warning("Audit store contains only partial write, clearing")
                    self._clear_store()
                    return
                # Truncate to last complete line
                content = content[:last_newline + 1]
                # Rewrite file without partial write
                try:
                    with open(self._store_path, "w", encoding="utf-8") as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    pass  # If rewrite fails, continue with in-memory content

            # Process complete lines
            first_record = True
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if first_record:
                        # Restore cross-file anchor from first record
                        self._cross_file_anchor = record.get("previous_hash", GENESIS_HASH)
                        first_record = False
                    self._previous_hash = record.get("event_hash", GENESIS_HASH)
                    self._event_count += 1
                except (json.JSONDecodeError, KeyError):
                    # Skip corrupted lines — partial write recovery
                    logger.debug("Skipping corrupted audit record")
                    continue
        except Exception as e:
            logger.debug("Failed to load audit store state: %s", e)

    def _clear_store(self) -> None:
        """Clear the store file."""
        try:
            with open(self._store_path, "w", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def recover(self) -> Dict[str, Any]:
        """Explicitly recover the audit store.

        Verifies hash-chain integrity and truncates to the last valid record
        if corruption is detected.

        Returns:
            Dict with recovery results.
        """
        result = {
            "recovery_performed": False,
            "events_before": 0,
            "events_after": 0,
            "issues_found": [],
            "truncated": False,
        }

        try:
            if not os.path.exists(self._store_path):
                return result

            # Read all records
            with open(self._store_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content:
                return result

            lines = [l.strip() for l in content.split("\n") if l.strip()]
            result["events_before"] = len(lines)

            # Verify chain and find last valid record
            previous_hash = GENESIS_HASH
            last_valid_index = -1            # Track invalid lines for potential recovery
            invalid_lines = []

            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Corrupted line - stop here
                    invalid_lines.append(i)
                    break

                # Verify previous_hash matches chain
                stored_previous_hash = record.get("previous_hash", "")
                if stored_previous_hash != previous_hash:
                    # Chain broken - stop here
                    invalid_lines.append(i)
                    break

                # Verify event_hash
                event_data = {k: v for k, v in record.items()                             if k not in ("previous_hash", "event_hash")}
                expected_hash = _compute_event_hash(event_data, stored_previous_hash)
                stored_hash = record.get("event_hash", "")
                if stored_hash != expected_hash:
                    # Hash mismatch - stop here
                    invalid_lines.append(i)
                    break

                previous_hash = stored_hash
                last_valid_index = i

            # If we found invalid lines, truncate
            if invalid_lines and last_valid_index >= 0:
                # Truncate to last valid record
                valid_lines = lines[:last_valid_index + 1]
                try:
                    with open(self._store_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(valid_lines) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

                    result["recovery_performed"] = True
                    result["truncated"] = True
                    result["events_after"] = len(valid_lines)
                    result["issues_found"] = [
                        f"Truncated {len(invalid_lines)} invalid records starting at line {invalid_lines[0] + 1}"
                    ]

                    # Update internal state
                    self._event_count = len(valid_lines)
                    self._previous_hash = previous_hash
                except Exception as e:
                    result["issues_found"].append(f"Recovery write failed: {e}")
            elif invalid_lines and last_valid_index < 0:
                # All records invalid - clear store
                self._clear_store()
                result["recovery_performed"] = True
                result["truncated"] = True
                result["events_after"] = 0
                result["issues_found"] = ["All records invalid, store cleared"]
                self._event_count = 0
                self._previous_hash = GENESIS_HASH
            else:
                result["events_after"] = result["events_before"]

            return result
        except Exception as e:
            result["issues_found"].append(f"Recovery error: {e}")
            return result

    def _write_record(self, record: Dict[str, Any]) -> bool:
        """Write a single record to the store file atomically.

        Uses write-to-temp + rename for partial-write safety.
        """
        try:
            # Append to file
            with open(self._store_path, "a", encoding="utf-8") as f:
                json_line = json.dumps(record, ensure_ascii=False, sort_keys=True)
                f.write(json_line + "\n")
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as e:
            logger.debug("Failed to write audit record: %s", e)
            return False

    def append(self, event: SecurityEvent) -> bool:
        """Append a single event to the store.

        Args:
            event: The SecurityEvent to store.

        Returns:
            True if successfully appended.
        """
        try:
            with self._lock:
                stored = _event_to_stored(event, self._previous_hash)
                if self._write_record(stored):
                    self._previous_hash = stored["event_hash"]
                    self._event_count += 1
                    return True
                return False
        except Exception as e:
            logger.debug("Failed to append audit event: %s", e)
            return False

    def append_batch(self, events: List[SecurityEvent]) -> int:
        """Append multiple events to the store.

        Args:
            events: List of SecurityEvent to store.

        Returns:
            Number of events successfully appended.
        """
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
        """Read events with optional filtering.

        Read-only. All parameters optional.
        """
        try:
            all_events = self._read_all_events()

            results = all_events

            if subagent_id is not None:
                results = [e for e in results if e.subagent_id == subagent_id]
            if parent_session_id is not None:
                results = [e for e in results if e.parent_session_id == parent_session_id]
            if event_type is not None:
                results = [e for e in results if e.event_type == event_type]
            if tool_name is not None:
                results = [e for e in results if e.tool_name == tool_name]
            if task_type is not None:
                results = [e for e in results if e.task_type == task_type]
            if lifecycle_state is not None:
                results = [e for e in results if e.lifecycle_state == lifecycle_state]
            if allowed is not None:
                results = [e for e in results if e.allowed == allowed]

            return results
        except Exception as e:
            logger.debug("Failed to read audit events: %s", e)
            return []

    def read_timeline(self, subagent_id: str) -> List[SecurityEvent]:
        """Read a sub-agent's chronological timeline."""
        events = self.read_events(subagent_id=subagent_id)
        events.sort(key=lambda e: (e.timestamp, e.event_id))
        return events

    def count_events(self) -> int:
        """Count total events in the store."""
        try:
            with self._lock:
                return self._event_count
        except Exception:
            return 0

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify the hash-chain integrity of the store.

        Detects: modification, deletion, insertion, reordering.
        Starts verification from the cross-file anchor (not GENESIS).
        """
        issues = []
        events_checked = 0
        previous_hash = self._cross_file_anchor  # Start from anchor, not GENESIS

        try:
            if not os.path.exists(self._store_path):
                return {
                    "valid": True,
                    "events_checked": 0,
                    "issues": [],
                }

            with open(self._store_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        issues.append(f"Line {line_num}: corrupted JSON")
                        continue

                    # Verify previous_hash matches
                    stored_previous_hash = record.get("previous_hash", "")
                    if stored_previous_hash != previous_hash:
                        issues.append(
                            f"Line {line_num}: hash chain broken "
                            f"(expected {previous_hash[:16]}..., got {stored_previous_hash[:16]}...)"
                        )

                    # Verify event_hash
                    event_data = {k: v for k, v in record.items()
                                 if k not in ("previous_hash", "event_hash")}
                    expected_hash = _compute_event_hash(event_data, stored_previous_hash)
                    stored_hash = record.get("event_hash", "")
                    if stored_hash != expected_hash:
                        issues.append(f"Line {line_num}: event hash mismatch")

                    previous_hash = stored_hash
                    events_checked += 1

            return {
                "valid": len(issues) == 0,
                "events_checked": events_checked,
                "issues": issues,
            }
        except Exception as e:
            return {
                "valid": False,
                "events_checked": events_checked,
                "issues": [str(e)],
            }

    def get_store_metadata(self) -> Dict[str, Any]:
        """Get store metadata."""
        try:
            file_size = 0
            if os.path.exists(self._store_path):
                file_size = os.path.getsize(self._store_path)

            return {
                "store_path": self._store_path,
                "event_count": self._event_count,
                "file_size_bytes": file_size,
                "previous_hash": self._previous_hash,
                "cross_file_anchor": self._cross_file_anchor,
            }
        except Exception as e:
            logger.debug("Failed to get store metadata: %s", e)
            return {
                "store_path": self._store_path,
                "event_count": 0,
                "file_size_bytes": 0,
                "previous_hash": GENESIS_HASH,
                "cross_file_anchor": GENESIS_HASH,
            }

    def _read_all_events(self) -> List[SecurityEvent]:
        """Read all events from the store."""
        events = []

        try:
            if not os.path.exists(self._store_path):
                return events

            with open(self._store_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        events.append(_stored_to_event(record))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.debug("Failed to read all audit events: %s", e)

        return events

    def close(self) -> None:
        """Close the store (no-op for file-based store)."""
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_audit_store(
    store_path: Optional[str] = None,
    cross_file_anchor: Optional[str] = None,
) -> SecurityAuditStore:
    """Create a new SecurityAuditStore.

    Args:
        store_path: Path to the audit log file.
        cross_file_anchor: Hash of the final event from the previous store file.
    """
    return SecurityAuditStore(store_path=store_path, cross_file_anchor=cross_file_anchor)
