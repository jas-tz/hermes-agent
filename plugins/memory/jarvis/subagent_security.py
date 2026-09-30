"""JARVIS Sub-Agent Security Event Infrastructure — Phase 28A.1.

Formal security event recording for sub-agent authorization decisions.

CRITICAL: This module is observational ONLY. A recorder exception must NEVER:
- turn DENY into ALLOW
- execute a blocked tool
- bypass approval
- alter task-type authority
- alter filesystem scope
- alter terminal restrictions
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security Event Types
# ---------------------------------------------------------------------------

class SecurityEventType(Enum):
    """Types of security-relevant events for sub-agent operations."""
    AUTHORIZATION_ALLOWED = "authorization_allowed"
    AUTHORIZATION_DENIED = "authorization_denied"
    PARENT_ONLY_DENIED = "parent_only_denied"
    TASK_TYPE_ESCALATION = "task_type_escalation"
    FORGED_CAPABILITY = "forged_capability"
    FILESYSTEM_TRAVERSAL = "filesystem_traversal"
    FILESYSTEM_ESCAPE = "filesystem_escape"
    TERMINAL_INJECTION = "terminal_injection"
    DELEGATION_ESCAPE = "delegation_escape"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DENIED = "approval_denied"
    LIFECYCLE_TRANSITION = "lifecycle_transition"
    LIFECYCLE_VIOLATION = "lifecycle_violation"
    RECURSION_LIMIT = "recursion_limit"
    TOOL_EXECUTION_BLOCKED = "tool_execution_blocked"
    TOOL_EXECUTION_COMPLETED = "tool_execution_completed"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"


# ---------------------------------------------------------------------------
# Security Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SecurityEvent:
    """Immutable security event record.

    Captures a single security-relevant decision or outcome.
    Frozen dataclass ensures immutability.
    """
    event_id: str
    timestamp: str
    event_type: SecurityEventType
    subagent_id: str
    parent_session_id: str
    task_type: str
    authority_level: str
    tool_name: str
    lifecycle_state: str
    allowed: bool
    reason: str


def create_security_event(
    event_type: SecurityEventType,
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str = "",
    lifecycle_state: str = "",
    allowed: bool = True,
    reason: str = "",
) -> SecurityEvent:
    """Factory for SecurityEvent with auto-generated ID and timestamp."""
    return SecurityEvent(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type=event_type,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=allowed,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Security Event Recorder
# ---------------------------------------------------------------------------

# Sensitive patterns that must never be stored in events
SENSITIVE_PATTERNS: FrozenSet[str] = frozenset({
    "api_key", "token", "secret", "password", "credential",
    "auth", "bearer", "private_key", "access_key",
})

# Maximum number of events to retain
DEFAULT_MAX_EVENTS = 10_000


class SecurityEventRecorder:
    """Thread-safe security event recorder with bounded retention.

    Observational only — never modifies authorization decisions.

    Phase 29B: Optional durable audit store integration.
    """

    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS) -> None:
        self._events: Deque[SecurityEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._audit_store: Optional[Any] = None

    def configure_audit_store(self, audit_store: Any) -> None:
        """Configure a SecurityAuditStore for durable persistence.

        When configured, all recorded events are also persisted durably.
        Store failures are isolated and do not affect authorization.
        """
        with self._lock:
            self._audit_store = audit_store

    def get_audit_store(self) -> Optional[Any]:
        """Get the configured audit store, if any."""
        return self._audit_store

    def disable_audit_store(self) -> None:
        """Disable durable audit persistence.

        The in-memory recorder remains functional.
        """
        with self._lock:
            self._audit_store = None

    def record(self, event: SecurityEvent) -> None:
        """Record a security event.

        Thread-safe. Silently drops events on error to ensure
        recorder failures don't affect authorization.

        Phase 29B: Also persists to audit store if configured.
        Store failure is isolated and does not affect authorization.
        """
        try:
            with self._lock:
                self._events.append(event)
                audit_store = self._audit_store

            # Persist to audit store outside the main lock to avoid
            # holding the lock during I/O. Store failure is isolated.
            if audit_store is not None:
                try:
                    audit_store.append(event)
                except Exception as store_exc:
                    # Store failure must NOT affect authorization
                    logger.debug("Audit store persistence failed: %s", store_exc)
        except Exception as e:
            # Recorder failure must NOT affect authorization
            logger.debug("Security event recording failed: %s", e)

    def get_events(self) -> List[SecurityEvent]:
        """Get all recorded events.

        Returns a copy — mutations don't affect recorder state.
        """
        try:
            with self._lock:
                return list(self._events)
        except Exception as e:
            logger.debug("Security event retrieval failed: %s", e)
            return []

    def get_events_for_subagent(self, subagent_id: str) -> List[SecurityEvent]:
        """Get events for a specific sub-agent."""
        try:
            with self._lock:
                return [e for e in self._events if e.subagent_id == subagent_id]
        except Exception as e:
            logger.debug("Security event retrieval failed: %s", e)
            return []

    def get_events_by_type(self, event_type: SecurityEventType) -> List[SecurityEvent]:
        """Get events of a specific type."""
        try:
            with self._lock:
                return [e for e in self._events if e.event_type == event_type]
        except Exception as e:
            logger.debug("Security event retrieval failed: %s", e)
            return []

    def query(
        self,
        subagent_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        event_type: Optional[SecurityEventType] = None,
        tool_name: Optional[str] = None,
        task_type: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        allowed: Optional[bool] = None,
    ) -> List[SecurityEvent]:
        """Query events with flexible filtering.

        Read-only. All parameters are optional. Returns a copy.
        Invalid/empty filters return empty list safely.
        """
        try:
            with self._lock:
                results = list(self._events)

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
            logger.debug("Security event query failed: %s", e)
            return []

    def get_timeline(self, subagent_id: str) -> List[SecurityEvent]:
        """Reconstruct a sub-agent's chronological security history.

        Preserves event ordering. Handles equal timestamps deterministically.
        No duplicate events.
        """
        try:
            with self._lock:
                events = [e for e in self._events if e.subagent_id == subagent_id]

            # Sort by timestamp, then by event_id for deterministic ordering
            events.sort(key=lambda e: (e.timestamp, e.event_id))

            # Remove duplicates (by event_id)
            seen_ids = set()
            unique_events = []
            for e in events:
                if e.event_id not in seen_ids:
                    seen_ids.add(e.event_id)
                    unique_events.append(e)

            return unique_events
        except Exception as e:
            logger.debug("Security event timeline failed: %s", e)
            return []

    def get_denied_events(self) -> List[SecurityEvent]:
        """Get all denied authorization events."""
        return self.query(allowed=False)

    def get_forged_capability_events(self) -> List[SecurityEvent]:
        """Get all forged capability events."""
        return self.query(event_type=SecurityEventType.FORGED_CAPABILITY)

    def get_approval_events(self) -> List[SecurityEvent]:
        """Get all approval-related events."""
        try:
            with self._lock:
                return [e for e in self._events
                        if e.event_type in (SecurityEventType.APPROVAL_REQUESTED,
                                           SecurityEventType.APPROVAL_DENIED)]
        except Exception as e:
            logger.debug("Security event retrieval failed: %s", e)
            return []

    def get_execution_failures(self) -> List[SecurityEvent]:
        """Get all tool execution failure events."""
        return self.query(event_type=SecurityEventType.TOOL_EXECUTION_FAILED)

    def get_security_investigation_events(self) -> List[SecurityEvent]:
        """Get all security-relevant events for investigation."""
        try:
            with self._lock:
                security_types = {
                    SecurityEventType.AUTHORIZATION_DENIED,
                    SecurityEventType.PARENT_ONLY_DENIED,
                    SecurityEventType.TASK_TYPE_ESCALATION,
                    SecurityEventType.FORGED_CAPABILITY,
                    SecurityEventType.FILESYSTEM_TRAVERSAL,
                    SecurityEventType.FILESYSTEM_ESCAPE,
                    SecurityEventType.TERMINAL_INJECTION,
                    SecurityEventType.DELEGATION_ESCAPE,
                    SecurityEventType.APPROVAL_DENIED,
                    SecurityEventType.LIFECYCLE_VIOLATION,
                    SecurityEventType.RECURSION_LIMIT,
                    SecurityEventType.TOOL_EXECUTION_BLOCKED,
                    SecurityEventType.TOOL_EXECUTION_FAILED,
                }
                return [e for e in self._events if e.event_type in security_types]
        except Exception as e:
            logger.debug("Security event retrieval failed: %s", e)
            return []

    def export_events(
        self,
        subagent_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        event_type: Optional[SecurityEventType] = None,
        tool_name: Optional[str] = None,
        task_type: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        allowed: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Export events as JSON-safe dicts.

        Read-only. Uses same filters as query().
        Preserves sanitization.
        """
        events = self.query(
            subagent_id=subagent_id,
            parent_session_id=parent_session_id,
            event_type=event_type,
            tool_name=tool_name,
            task_type=task_type,
            lifecycle_state=lifecycle_state,
            allowed=allowed,
        )

        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "event_type": e.event_type.value,
                "subagent_id": e.subagent_id,
                "parent_session_id": e.parent_session_id,
                "task_type": e.task_type,
                "authority_level": e.authority_level,
                "tool_name": e.tool_name,
                "lifecycle_state": e.lifecycle_state,
                "allowed": e.allowed,
                "reason": e.reason,
            }
            for e in events
        ]

    def export_timeline(self, subagent_id: str) -> List[Dict[str, Any]]:
        """Export a sub-agent's timeline as JSON-safe dicts."""
        events = self.get_timeline(subagent_id)

        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "event_type": e.event_type.value,
                "subagent_id": e.subagent_id,
                "parent_session_id": e.parent_session_id,
                "task_type": e.task_type,
                "authority_level": e.authority_level,
                "tool_name": e.tool_name,
                "lifecycle_state": e.lifecycle_state,
                "allowed": e.allowed,
                "reason": e.reason,
            }
            for e in events
        ]

    def generate_security_report(
        self,
        subagent_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        event_type: Optional[SecurityEventType] = None,
        tool_name: Optional[str] = None,
        task_type: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        allowed: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate a JSON-safe security report.

        Read-only. Includes metadata, filter criteria, event counts,
        allowed/denied counts, forged capability events, approval activity,
        execution failures, lifecycle transitions, affected sub-agents,
        affected tools, and chronological timeline.
        """
        events = self.query(
            subagent_id=subagent_id,
            parent_session_id=parent_session_id,
            event_type=event_type,
            tool_name=tool_name,
            task_type=task_type,
            lifecycle_state=lifecycle_state,
            allowed=allowed,
        )

        # Build report
        report = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_events": len(events),
            },
            "filter_criteria": {
                "subagent_id": subagent_id,
                "parent_session_id": parent_session_id,
                "event_type": event_type.value if event_type else None,
                "tool_name": tool_name,
                "task_type": task_type,
                "lifecycle_state": lifecycle_state,
                "allowed": allowed,
            },
            "summary": {
                "total_events": len(events),
                "allowed_count": sum(1 for e in events if e.allowed),
                "denied_count": sum(1 for e in events if not e.allowed),
                "forged_capability_count": sum(
                    1 for e in events if e.event_type == SecurityEventType.FORGED_CAPABILITY
                ),
                "approval_requested_count": sum(
                    1 for e in events if e.event_type == SecurityEventType.APPROVAL_REQUESTED
                ),
                "approval_denied_count": sum(
                    1 for e in events if e.event_type == SecurityEventType.APPROVAL_DENIED
                ),
                "execution_failure_count": sum(
                    1 for e in events if e.event_type == SecurityEventType.TOOL_EXECUTION_FAILED
                ),
                "lifecycle_transition_count": sum(
                    1 for e in events if e.event_type == SecurityEventType.LIFECYCLE_TRANSITION
                ),
            },
            "affected_subagents": sorted(set(e.subagent_id for e in events)),
            "affected_tools": sorted(set(e.tool_name for e in events if e.tool_name)),
            "timeline": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type.value,
                    "subagent_id": e.subagent_id,
                    "parent_session_id": e.parent_session_id,
                    "task_type": e.task_type,
                    "authority_level": e.authority_level,
                    "tool_name": e.tool_name,
                    "lifecycle_state": e.lifecycle_state,
                    "allowed": e.allowed,
                    "reason": e.reason,
                }
                for e in sorted(events, key=lambda e: (e.timestamp, e.event_id))
            ],
        }

        return report

    def clear(self) -> None:
        """Clear all recorded events."""
        try:
            with self._lock:
                self._events.clear()
        except Exception as e:
            logger.debug("Security event clear failed: %s", e)

    @property
    def count(self) -> int:
        """Number of recorded events."""
        try:
            with self._lock:
                return len(self._events)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Global recorder instance
# ---------------------------------------------------------------------------

_recorder = SecurityEventRecorder()


def get_recorder() -> SecurityEventRecorder:
    """Get the global security event recorder."""
    return _recorder


def reset_recorder() -> None:
    """Reset the global recorder (for testing)."""
    global _recorder
    _recorder = SecurityEventRecorder()


# ---------------------------------------------------------------------------
# Phase 29B: Audit store configuration
# ---------------------------------------------------------------------------

def configure_audit_store(audit_store: Any) -> None:
    """Configure the audit store for durable persistence.

    When configured, all recorded events are also persisted durably.
    Store failures are isolated and do not affect authorization.
    """
    _recorder.configure_audit_store(audit_store)


def get_audit_store() -> Optional[Any]:
    """Get the configured audit store, if any."""
    return _recorder.get_audit_store()


def disable_audit_store() -> None:
    """Disable durable audit persistence.

    The in-memory recorder remains functional.
    """
    _recorder.disable_audit_store()


# ---------------------------------------------------------------------------
# Phase 29D: Audit manager configuration
# ---------------------------------------------------------------------------

def configure_audit_manager(store_dir: str, **kwargs) -> None:
    """Configure the audit store manager for production use.

    This creates an AuditStoreManager with rotation, archival, and retention,
    and wires it to the global recorder for durable persistence.

    Args:
        store_dir: Directory for audit store files.
        **kwargs: Forwarded to create_audit_manager() (max_file_size_mb, etc.)
    """
    from plugins.memory.jarvis.subagent_security_manager import create_audit_manager
    manager = create_audit_manager(store_dir, **kwargs)
    _recorder.configure_audit_store(manager)


def get_audit_manager() -> Optional[Any]:
    """Get the configured audit manager, if any."""
    store = _recorder.get_audit_store()
    if store is not None:
        from plugins.memory.jarvis.subagent_security_manager import AuditStoreManager
        if isinstance(store, AuditStoreManager):
            return store
    return None


def disable_audit_manager() -> None:
    """Disable audit manager persistence.

    The in-memory recorder remains functional.
    """
    _recorder.disable_audit_store()


# ---------------------------------------------------------------------------
# Helper: sanitize reason to avoid storing secrets
# ---------------------------------------------------------------------------

def _sanitize_reason(reason: str) -> str:
    """Remove potential secrets from reason strings."""
    if not reason:
        return reason

    # Split on common delimiters and filter out sensitive key-value pairs
    parts = reason.split()
    sanitized = []
    for part in parts:
        lower = part.lower()
        is_sensitive = any(pattern in lower for pattern in SENSITIVE_PATTERNS)
        if is_sensitive:
            sanitized.append("[REDACTED]")
        else:
            sanitized.append(part)

    return " ".join(sanitized)


# ---------------------------------------------------------------------------
# Convenience functions for recording events
# ---------------------------------------------------------------------------

def record_authorization_allowed(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
    reason: str = "",
) -> None:
    """Record an allowed authorization decision."""
    event = create_security_event(
        event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=True,
        reason=_sanitize_reason(reason),
    )
    _recorder.record(event)


def record_authorization_denied(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
    reason: str = "",
) -> None:
    """Record a denied authorization decision."""
    event = create_security_event(
        event_type=SecurityEventType.AUTHORIZATION_DENIED,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=False,
        reason=_sanitize_reason(reason),
    )
    _recorder.record(event)


def record_parent_only_denied(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
) -> None:
    """Record a parent-only tool denial."""
    event = create_security_event(
        event_type=SecurityEventType.PARENT_ONLY_DENIED,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=False,
        reason="Parent-only tool",
    )
    _recorder.record(event)


def record_forged_capability(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
) -> None:
    """Record a forged capability attempt."""
    event = create_security_event(
        event_type=SecurityEventType.FORGED_CAPABILITY,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=False,
        reason="Forged/unknown capability",
    )
    _recorder.record(event)


def record_tool_execution_blocked(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
    reason: str = "",
) -> None:
    """Record a blocked tool execution."""
    event = create_security_event(
        event_type=SecurityEventType.TOOL_EXECUTION_BLOCKED,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=False,
        reason=_sanitize_reason(reason),
    )
    _recorder.record(event)


def record_tool_execution_completed(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
) -> None:
    """Record a successful tool execution."""
    event = create_security_event(
        event_type=SecurityEventType.TOOL_EXECUTION_COMPLETED,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=True,
        reason="Execution completed",
    )
    _recorder.record(event)


def record_tool_execution_failed(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
    reason: str = "",
) -> None:
    """Record a failed tool execution."""
    event = create_security_event(
        event_type=SecurityEventType.TOOL_EXECUTION_FAILED,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=True,  # Authorization was allowed, execution failed
        reason=_sanitize_reason(reason),
    )
    _recorder.record(event)


def record_approval_requested(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
) -> None:
    """Record an approval request."""
    event = create_security_event(
        event_type=SecurityEventType.APPROVAL_REQUESTED,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=False,
        reason="Approval required",
    )
    _recorder.record(event)


def record_approval_denied(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
) -> None:
    """Record an approval denial."""
    event = create_security_event(
        event_type=SecurityEventType.APPROVAL_DENIED,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=False,
        reason="Approval denied",
    )
    _recorder.record(event)


def record_lifecycle_transition(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str,
) -> None:
    """Record a lifecycle transition."""
    event = create_security_event(
        event_type=SecurityEventType.LIFECYCLE_TRANSITION,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=True,
        reason=f"Transitioned to {lifecycle_state}",
    )
    _recorder.record(event)


def record_recursion_limit(
    subagent_id: str,
    parent_session_id: str,
    task_type: str,
    authority_level: str,
    tool_name: str,
    lifecycle_state: str = "",
) -> None:
    """Record a recursion limit violation."""
    event = create_security_event(
        event_type=SecurityEventType.RECURSION_LIMIT,
        subagent_id=subagent_id,
        parent_session_id=parent_session_id,
        task_type=task_type,
        authority_level=authority_level,
        tool_name=tool_name,
        lifecycle_state=lifecycle_state,
        allowed=False,
        reason="Recursion depth limit reached",
    )
    _recorder.record(event)
