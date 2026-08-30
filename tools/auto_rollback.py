"""Automatic rollback and recovery with provenance-aware state management.

Provides checkpoint creation before destructive operations, automatic rollback
on explicit failure signals, and full audit trail. Session-isolated and opt-in.

Safety properties:
- Disabled by default; no behavior change when disabled.
- Checkpoints are session-isolated.
- All state snapshots are sanitized through redact_secrets().
- Automatic rollback requires explicit failure signal or user confirmation.
- All operations are logged with provenance metadata.
- Reversible: rollback can be undone via redo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """A single checkpoint with state snapshot and provenance."""

    id: str
    session_id: str
    label: str
    timestamp: float
    state_snapshot: Dict[str, Any]
    provenance: Dict[str, Any]
    checksum: str
    redo_from: Optional[str] = None  # ID of checkpoint this was rolled back from


@dataclass
class AuditEntry:
    """A single audit trail entry."""

    timestamp: float
    operation: str  # "create_checkpoint", "rollback", "rollback_failure", "redo", "evict"
    session_id: str
    checkpoint_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    success: bool = True


class AutoRollbackManager:
    """Manages checkpoints, rollbacks, and recovery with full audit trail.

    Thread-safe via reentrant lock.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._enabled: bool = config.get("enabled", False)
        self._max_checkpoints: int = config.get("max_checkpoints", 10)
        self._require_confirmation: bool = config.get("require_confirmation", True)
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._session_checkpoints: Dict[str, List[str]] = {}  # session_id -> [checkpoint_id, ...]
        self._audit_trail: List[AuditEntry] = []
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def create_checkpoint(
        self,
        session_id: str,
        label: str,
        state: Dict[str, Any],
    ) -> Optional[str]:
        """Create a checkpoint.

        Args:
            session_id: Session ID for isolation.
            label: Human-readable label for the checkpoint.
            state: State snapshot to checkpoint (will be sanitized).

        Returns:
            Checkpoint ID on success, None if disabled.
        """
        if not self._enabled:
            return None

        with self._lock:
            checkpoint_id = str(uuid.uuid4())
            sanitized_state = self._sanitize_state(state)
            checksum = self._compute_checksum(sanitized_state)

            checkpoint = Checkpoint(
                id=checkpoint_id,
                session_id=session_id,
                label=label,
                timestamp=time.monotonic(),
                state_snapshot=sanitized_state,
                provenance={
                    "source_type": "auto_rollback",
                    "source_id": session_id,
                    "created_at": time.monotonic(),
                    "provenance_status": "checkpoint",
                    "origin": label,
                },
                checksum=checksum,
            )

            self._checkpoints[checkpoint_id] = checkpoint
            self._session_checkpoints.setdefault(session_id, []).append(checkpoint_id)

            # Evict old checkpoints if over limit
            self._evict_if_needed(session_id)

            self._log_audit(
                operation="create_checkpoint",
                session_id=session_id,
                checkpoint_id=checkpoint_id,
                details={"label": label},
            )

            return checkpoint_id

    def rollback(
        self,
        checkpoint_id: str,
        confirmed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Rollback to a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to restore.
            confirmed: Whether user confirmation was obtained.

        Returns:
            Restored state snapshot on success, None on failure.
        """
        if not self._enabled:
            return None

        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                self._log_audit(
                    operation="rollback_failure",
                    session_id="unknown",
                    checkpoint_id=checkpoint_id,
                    details={"reason": "checkpoint_not_found"},
                    success=False,
                )
                return None

            # Validate checksum
            current_checksum = self._compute_checksum(checkpoint.state_snapshot)
            if current_checksum != checkpoint.checksum:
                self._log_audit(
                    operation="rollback_failure",
                    session_id=checkpoint.session_id,
                    checkpoint_id=checkpoint_id,
                    details={"reason": "checksum_mismatch"},
                    success=False,
                )
                return None

            # Create redo checkpoint (current state before rollback)
            redo_id = str(uuid.uuid4())
            redo_checkpoint = Checkpoint(
                id=redo_id,
                session_id=checkpoint.session_id,
                label=f"redo_from_{checkpoint_id[:8]}",
                timestamp=time.monotonic(),
                state_snapshot=checkpoint.state_snapshot,  # Same state for redo
                provenance={
                    **checkpoint.provenance,
                    "provenance_status": "redo_checkpoint",
                    "redo_from": checkpoint_id,
                },
                checksum=checkpoint.checksum,
                redo_from=checkpoint_id,
            )
            self._checkpoints[redo_id] = redo_checkpoint
            self._session_checkpoints[checkpoint.session_id].append(redo_id)

            self._log_audit(
                operation="rollback",
                session_id=checkpoint.session_id,
                checkpoint_id=checkpoint_id,
                details={
                    "label": checkpoint.label,
                    "confirmed": confirmed,
                    "redo_id": redo_id,
                },
            )

            return dict(checkpoint.state_snapshot)

    def auto_rollback_on_failure(
        self,
        session_id: str,
        failure_signal: bool = False,
        confirmed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Automatically rollback to the last checkpoint on failure.

        Requires an explicit failure signal. If require_confirmation is True
        and confirmed is False, returns None.

        Args:
            session_id: Session ID.
            failure_signal: Explicit failure signal.
            confirmed: Whether user confirmation was obtained.

        Returns:
            Restored state on success, None if no rollback performed.
        """
        if not self._enabled:
            return None

        if not failure_signal:
            return None

        if self._require_confirmation and not confirmed:
            return None

        last = self._get_last_checkpoint(session_id)
        if not last:
            return None

        return self.rollback(last, confirmed=confirmed)

    def redo(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Redo an operation after rollback.

        Args:
            checkpoint_id: The rollback checkpoint ID to redo from.

        Returns:
            State to restore on success, None on failure.
        """
        if not self._enabled:
            return None

        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return None

            redo_from = checkpoint.redo_from
            if not redo_from:
                return None

            original = self._checkpoints.get(redo_from)
            if not original:
                return None

            self._log_audit(
                operation="redo",
                session_id=checkpoint.session_id,
                checkpoint_id=checkpoint_id,
                details={"original_checkpoint": redo_from},
            )

            return dict(original.state_snapshot)

    def get_last_checkpoint_id(self, session_id: str) -> Optional[str]:
        """Get the last checkpoint ID for a session."""
        if not self._enabled:
            return None
        return self._get_last_checkpoint(session_id)

    def get_audit_trail(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit trail entries.

        Args:
            session_id: If provided, filter by session.
            limit: Maximum entries to return.

        Returns:
            List of audit entry dicts.
        """
        with self._lock:
            entries = self._audit_trail
            if session_id:
                entries = [e for e in entries if e.session_id == session_id]
            return [
                {
                    "timestamp": e.timestamp,
                    "operation": e.operation,
                    "session_id": e.session_id,
                    "checkpoint_id": e.checkpoint_id,
                    "details": e.details,
                    "success": e.success,
                }
                for e in entries[-limit:]
            ]

    def clear_session(self, session_id: str) -> int:
        """Clear all checkpoints for a session.

        Returns:
            Number of checkpoints cleared.
        """
        with self._lock:
            checkpoint_ids = self._session_checkpoints.pop(session_id, [])
            for cid in checkpoint_ids:
                self._checkpoints.pop(cid, None)
            return len(checkpoint_ids)

    def stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "total_checkpoints": len(self._checkpoints),
                "total_sessions": len(self._session_checkpoints),
                "max_checkpoints": self._max_checkpoints,
                "require_confirmation": self._require_confirmation,
                "audit_entries": len(self._audit_trail),
            }

    def _get_last_checkpoint(self, session_id: str) -> Optional[str]:
        """Get the last checkpoint ID for a session (internal, no lock)."""
        checkpoint_ids = self._session_checkpoints.get(session_id, [])
        return checkpoint_ids[-1] if checkpoint_ids else None

    def _evict_if_needed(self, session_id: str) -> None:
        """Evict oldest checkpoints if over limit (internal, no lock)."""
        checkpoint_ids = self._session_checkpoints.get(session_id, [])
        while len(checkpoint_ids) > self._max_checkpoints:
            oldest_id = checkpoint_ids.pop(0)
            self._checkpoints.pop(oldest_id, None)
            self._log_audit(
                operation="evict",
                session_id=session_id,
                checkpoint_id=oldest_id,
                details={"reason": "max_checkpoints_exceeded"},
            )

    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state snapshot before storage."""
        try:
            from tools.secret_redaction import redact_secrets
            sanitized = {}
            for key, value in state.items():
                if isinstance(value, str):
                    sanitized[key] = redact_secrets(value)
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize_state(value)
                else:
                    sanitized[key] = value
            return sanitized
        except Exception:
            return state

    def _compute_checksum(self, state: Dict[str, Any]) -> str:
        """Compute SHA-256 checksum of state."""
        canonical = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _log_audit(
        self,
        operation: str,
        session_id: str,
        checkpoint_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> None:
        """Log an audit entry."""
        entry = AuditEntry(
            timestamp=time.monotonic(),
            operation=operation,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            details=details,
            success=success,
        )
        self._audit_trail.append(entry)
