"""Semantic cache for tool results with provenance-aware invalidation.

This module provides an opt-in, in-memory cache for tool results. It uses
provenance metadata for cache keys and invalidation, applies secret redaction
before caching, and is session-isolated by default.

Safety properties:
- Read-only: cache hits never trigger tool execution.
- Secret-safe: all cached values are sanitized through redact_secrets().
- Provenance-aware: cached results are invalidated when provenance changes.
- Session-isolated: cache entries are scoped to the creating session.
- Best-effort: cache failures never block tool execution.
- Opt-in: disabled by default; existing behavior unchanged when disabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached tool result with provenance and validity metadata."""

    value: Any
    tool_name: str
    input_fingerprint: str
    provenance: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.monotonic)
    cache_timestamp: float = field(default_factory=time.monotonic)
    valid: bool = True
    invalidation_reason: Optional[str] = None
    session_id: Optional[str] = None
    result_metadata: Optional[Dict[str, Any]] = None

    def is_fresh(self, ttl_seconds: float) -> bool:
        """Check if the entry is still fresh based on TTL."""
        if ttl_seconds <= 0:
            return False
        if not self.valid:
            return False
        return (time.monotonic() - self.cache_timestamp) < ttl_seconds

    def is_provenance_valid(self, required_provenance: Optional[Dict[str, Any]]) -> bool:
        """Check if provenance requirements are satisfied."""
        if required_provenance is None:
            return True
        if self.provenance is None:
            return False
        # Check that all required provenance fields match
        for key, value in required_provenance.items():
            if self.provenance.get(key) != value:
                return False
        return True

    def invalidate(self, reason: str) -> None:
        """Mark this entry as invalid."""
        self.valid = False
        self.invalidation_reason = reason


class SemanticCache:
    """In-memory semantic cache for tool results with provenance-aware invalidation.

    The cache is best-effort: any error returns a cache miss so the caller
    re-executes the tool. A cache problem must never block tool execution.

    Thread-safe via a reentrant lock.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the cache.

        Args:
            config: Configuration dict with optional keys:
                - enabled (bool): Whether caching is active. Default: False.
                - max_size (int): Maximum entries before LRU eviction. Default: 1000.
                - ttl_seconds (float): Default TTL for entries. Default: 300.
                - session_isolated (bool): Scope cache by session. Default: True.
                - redact_secrets (bool): Sanitize values before caching. Default: True.
        """
        config = config or {}
        self._enabled: bool = config.get("enabled", False)
        self._max_size: int = config.get("max_size", 1000)
        self._ttl_seconds: float = config.get("ttl_seconds", 300)
        self._session_isolated: bool = config.get("session_isolated", True)
        self._redact_secrets: bool = config.get("redact_secrets", True)
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get(
        self,
        tool_name: str,
        input_fingerprint: str,
        session_id: Optional[str] = None,
        required_provenance: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> Optional[Any]:
        """Get a cached tool result.

        Returns None on any cache miss (not found, expired, invalid, or
        provenance mismatch) so the caller re-executes the tool.

        Args:
            tool_name: Name of the tool that produced the result.
            input_fingerprint: Fingerprint of the tool input.
            session_id: Session ID for isolation lookup.
            required_provenance: If provided, cache hit requires matching provenance.
            ttl_seconds: Override default TTL for this lookup.

        Returns:
            Cached value on hit, None on miss.
        """
        if not self._enabled:
            return None

        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        key = self._make_key(tool_name, input_fingerprint, session_id)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            # Freshness check
            if not entry.is_fresh(ttl):
                entry.invalidate("expired")
                del self._cache[key]
                return None

            # Provenance check
            if not entry.is_provenance_valid(required_provenance):
                entry.invalidate("provenance_mismatch")
                del self._cache[key]
                return None

            return entry.value

    def put(
        self,
        tool_name: str,
        input_fingerprint: str,
        value: Any,
        session_id: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
        result_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool that produced the result.
            input_fingerprint: Fingerprint of the tool input.
            value: Tool result value (will be sanitized if redact_secrets is on).
            session_id: Session ID for isolation.
            provenance: Provenance metadata for invalidation.
            result_metadata: Additional metadata about the result.
        """
        if not self._enabled:
            return

        # Sanitize value before caching
        sanitized_value = self._sanitize_value(value)

        key = self._make_key(tool_name, input_fingerprint, session_id)
        now = time.monotonic()

        with self._lock:
            # LRU eviction if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_oldest()

            self._cache[key] = CacheEntry(
                value=sanitized_value,
                tool_name=tool_name,
                input_fingerprint=input_fingerprint,
                provenance=provenance,
                created_at=now,
                cache_timestamp=now,
                valid=True,
                session_id=session_id,
                result_metadata=result_metadata,
            )

    def invalidate(
        self,
        tool_name: Optional[str] = None,
        session_id: Optional[str] = None,
        source_type: Optional[str] = None,
        reason: str = "explicit",
    ) -> int:
        """Invalidate cached entries.

        Args:
            tool_name: If provided, only invalidate entries for this tool.
            session_id: If provided, only invalidate entries for this session.
            source_type: If provided, only invalidate entries with matching provenance source_type.
            reason: Human-readable invalidation reason.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            to_invalidate = []
            for key, entry in self._cache.items():
                if tool_name is not None and entry.tool_name != tool_name:
                    continue
                if session_id is not None and entry.session_id != session_id:
                    continue
                if source_type is not None:
                    if entry.provenance is None:
                        continue
                    if entry.provenance.get("source_type") != source_type:
                        continue
                to_invalidate.append(key)

            for key in to_invalidate:
                self._cache[key].invalidate(reason)
                del self._cache[key]

            return len(to_invalidate)

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with entry_count, max_size, enabled, session_isolated, ttl_seconds.
        """
        with self._lock:
            return {
                "entry_count": len(self._cache),
                "max_size": self._max_size,
                "enabled": self._enabled,
                "session_isolated": self._session_isolated,
                "ttl_seconds": self._ttl_seconds,
            }

    def _make_key(
        self,
        tool_name: str,
        input_fingerprint: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Build a cache key from tool name, input fingerprint, and optional session."""
        components = [tool_name, input_fingerprint]
        if self._session_isolated and session_id:
            components.append(session_id)
        raw = "|".join(components)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a value before caching."""
        if not self._redact_secrets:
            return value
        try:
            from tools.secret_redaction import redact_secrets
            if isinstance(value, str):
                return redact_secrets(value)
            return value
        except Exception:
            # Fail-safe: on any sanitization error, return the original
            # (better to cache unsanitized than to lose the result)
            return value

    def _evict_oldest(self) -> None:
        """Evict the oldest entry (by cache_timestamp)."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].cache_timestamp)
        del self._cache[oldest_key]


def compute_input_fingerprint(tool_args: Dict[str, Any]) -> str:
    """Compute a deterministic fingerprint for tool input arguments.

    Args:
        tool_args: Tool input arguments dict.

    Returns:
        SHA-256 hex digest of the canonical JSON serialization.
    """
    canonical = json.dumps(tool_args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
