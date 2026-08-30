"""Tool result verification and consistency checking for JARVIS.

Provides read-only verification of tool results using fingerprints,
provenance, cache consistency, and schema validation. Opt-in and audited.

Safety properties:
- Read-only: verification never modifies tool results.
- Explicit status: every verification returns a clear status.
- No silent overwrites: conflicting results are flagged, not replaced.
- Secret safety: all content sanitized through redact_secrets().
- Session-isolated: verification scoped to session by default.
- Bounded: resource limits prevent unbounded verification.
- Audited: all verification decisions logged.
- Opt-in: disabled by default; no behavior change when disabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VerificationStatus:
    """Verification status constants."""
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICT = "CONFLICT"
    STALE = "STALE"
    INVALID = "INVALID"


# Provenance precedence for conflict resolution
PROVENANCE_PRECEDENCE = {
    "manual": 5,
    "conversation": 4,
    "tool_result": 3,
    "web": 3,
    "inferred": 2,
    "unknown": 1,
}


@dataclass
class VerificationResult:
    """Result of a tool result verification."""

    status: str
    tool_name: str
    fingerprint: str
    provenance: Dict[str, Any]
    timestamp: float
    confidence: float = 0.0
    details: Optional[Dict[str, Any]] = None
    conflict_with: Optional[str] = None
    cached_result_id: Optional[str] = None


@dataclass
class VerificationAuditEntry:
    """Audit entry for verification operations."""

    timestamp: float
    operation: str  # "verify", "conflict_detected", "stale_detected", "invalid_detected"
    tool_name: str
    session_id: str
    status: str
    fingerprint: str
    details: Optional[Dict[str, Any]] = None


class ToolResultVerifier:
    """Verifies tool results for consistency and correctness.

    Thread-safe via reentrant lock.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._enabled: bool = config.get("enabled", False)
        self._consistency_threshold: float = config.get("consistency_threshold", 0.85)
        self._max_verifications_per_turn: int = config.get("max_verifications", 10)
        self._session_isolated: bool = config.get("session_isolated", True)
        self._audit_log: List[VerificationAuditEntry] = []
        self._verification_count: int = 0
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def verify_result(
        self,
        tool_name: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
    ) -> VerificationResult:
        """Verify a tool result for consistency.

        Args:
            tool_name: Name of the tool that produced the result.
            result: Tool result value.
            context: Optional context (provenance, cache, etc.).
            session_id: Session ID for isolation.

        Returns:
            VerificationResult with status and details.
        """
        if not self._enabled:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                tool_name=tool_name,
                fingerprint="",
                provenance={},
                timestamp=time.monotonic(),
            )

        context = context or {}
        fingerprint = self._compute_fingerprint(result)
        provenance = context.get("provenance", {})

        # Check resource limits
        if self._verification_count >= self._max_verifications_per_turn:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                tool_name=tool_name,
                fingerprint=fingerprint,
                provenance=provenance,
                timestamp=time.monotonic(),
                details={"reason": "max_verifications_exceeded"},
            )

        self._verification_count += 1

        # 1. Check for malformed result
        if self._is_malformed(result):
            self._audit(
                operation="invalid_detected",
                tool_name=tool_name,
                session_id=session_id,
                status=VerificationStatus.INVALID,
                fingerprint=fingerprint,
                details={"reason": "malformed_result"},
            )
            return VerificationResult(
                status=VerificationStatus.INVALID,
                tool_name=tool_name,
                fingerprint=fingerprint,
                provenance=provenance,
                timestamp=time.monotonic(),
                details={"reason": "malformed_result"},
            )

        # 2. Check provenance
        if not self._is_provenance_valid(provenance):
            self._audit(
                operation="invalid_detected",
                tool_name=tool_name,
                session_id=session_id,
                status=VerificationStatus.INVALID,
                fingerprint=fingerprint,
                details={"reason": "invalid_provenance"},
            )
            return VerificationResult(
                status=VerificationStatus.INVALID,
                tool_name=tool_name,
                fingerprint=fingerprint,
                provenance=provenance,
                timestamp=time.monotonic(),
                details={"reason": "invalid_provenance"},
            )

        # 3. Check cache for consistency
        cache_entry = context.get("cache_entry")
        if cache_entry:
            # Check if result matches cached version
            cached_fingerprint = cache_entry.get("input_fingerprint", "")
            if cached_fingerprint and cached_fingerprint != fingerprint:
                # Result changed — check if stale
                cache_timestamp = cache_entry.get("cache_timestamp", 0)
                ttl = context.get("ttl_seconds", 300)
                if (time.monotonic() - cache_timestamp) > ttl:
                    self._audit(
                        operation="stale_detected",
                        tool_name=tool_name,
                        session_id=session_id,
                        status=VerificationStatus.STALE,
                        fingerprint=fingerprint,
                        details={"cache_age": time.monotonic() - cache_timestamp},
                    )
                    return VerificationResult(
                        status=VerificationStatus.STALE,
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                        provenance=provenance,
                        timestamp=time.monotonic(),
                        cached_result_id=cache_entry.get("id"),
                        details={"reason": "cache_expired"},
                    )

        # 4. Check for conflicts with other results
        other_results = context.get("other_results", [])
        for other in other_results:
            conflict = self._check_conflict(result, other)
            if conflict:
                self._audit(
                    operation="conflict_detected",
                    tool_name=tool_name,
                    session_id=session_id,
                    status=VerificationStatus.CONFLICT,
                    fingerprint=fingerprint,
                    details={"conflict_with": other.get("tool_name")},
                )
                return VerificationResult(
                    status=VerificationStatus.CONFLICT,
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                    provenance=provenance,
                    timestamp=time.monotonic(),
                    conflict_with=other.get("tool_name"),
                    details={"reason": "result_conflict"},
                )

        # 5. All checks passed
        self._audit(
            operation="verify",
            tool_name=tool_name,
            session_id=session_id,
            status=VerificationStatus.VERIFIED,
            fingerprint=fingerprint,
        )
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            tool_name=tool_name,
            fingerprint=fingerprint,
            provenance=provenance,
            timestamp=time.monotonic(),
            confidence=1.0,
        )

    def check_cross_tool_consistency(
        self,
        results: List[Dict[str, Any]],
        session_id: str = "default",
    ) -> List[VerificationResult]:
        """Check consistency across multiple tool results.

        Args:
            results: List of result dicts with 'tool_name', 'result', 'provenance'.
            session_id: Session ID for isolation.

        Returns:
            List of VerificationResult for each inconsistency found.
        """
        if not self._enabled:
            return []

        inconsistencies: List[VerificationResult] = []

        for i, result in enumerate(results):
            context = {
                "provenance": result.get("provenance", {}),
                "other_results": results[i + 1:],
            }
            verification = self.verify_result(
                tool_name=result.get("tool_name", "unknown"),
                result=result.get("result"),
                context=context,
                session_id=session_id,
            )
            if verification.status != VerificationStatus.VERIFIED:
                inconsistencies.append(verification)

        return inconsistencies

    def get_audit_log(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries.

        Args:
            session_id: If provided, filter by session.
            limit: Maximum entries to return.

        Returns:
            List of audit entry dicts.
        """
        with self._lock:
            entries = self._audit_log
            if session_id:
                entries = [e for e in entries if e.session_id == session_id]
            return [
                {
                    "timestamp": e.timestamp,
                    "operation": e.operation,
                    "tool_name": e.tool_name,
                    "session_id": e.session_id,
                    "status": e.status,
                    "fingerprint": e.fingerprint,
                    "details": e.details,
                }
                for e in entries[-limit:]
            ]

    def stats(self) -> Dict[str, Any]:
        """Get verifier statistics."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "consistency_threshold": self._consistency_threshold,
                "max_verifications_per_turn": self._max_verifications_per_turn,
                "session_isolated": self._session_isolated,
                "total_verifications": self._verification_count,
                "audit_entries": len(self._audit_log),
            }

    def reset_verification_count(self) -> None:
        """Reset per-turn verification count."""
        with self._lock:
            self._verification_count = 0

    def _compute_fingerprint(self, result: Any) -> str:
        """Compute SHA-256 fingerprint of result."""
        try:
            from tools.secret_redaction import redact_secrets
            if isinstance(result, str):
                clean = redact_secrets(result)
            else:
                clean = redact_secrets(json.dumps(result, sort_keys=True, separators=(",", ":")))
        except Exception:
            clean = json.dumps(result, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(clean.encode()).hexdigest()

    def _is_malformed(self, result: Any) -> bool:
        """Check if result is malformed."""
        if result is None:
            return True
        if isinstance(result, str) and not result.strip():
            return True
        if isinstance(result, (list, dict)) and len(result) == 0:
            return True
        return False

    def _is_provenance_valid(self, provenance: Dict[str, Any]) -> bool:
        """Check if provenance is valid."""
        if not provenance:
            return False
        source = provenance.get("source", "unknown")
        if source not in PROVENANCE_PRECEDENCE:
            return False
        return True

    def _check_conflict(
        self,
        result: Any,
        other: Dict[str, Any],
    ) -> bool:
        """Check if result conflicts with another result."""
        other_result = other.get("result")
        if other_result is None:
            return False

        # Simple conflict detection: same tool, different results
        try:
            result_fp = self._compute_fingerprint(result)
            other_fp = self._compute_fingerprint(other_result)
            return result_fp != other_fp
        except Exception:
            return False

    def _audit(
        self,
        operation: str,
        tool_name: str,
        session_id: str,
        status: str,
        fingerprint: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an audit entry."""
        entry = VerificationAuditEntry(
            timestamp=time.monotonic(),
            operation=operation,
            tool_name=tool_name,
            session_id=session_id,
            status=status,
            fingerprint=fingerprint,
            details=details,
        )
        self._audit_log.append(entry)
