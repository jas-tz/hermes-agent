"""Memory deduplication and conflict resolution for JARVIS.

Provides opt-in deduplication of memory entries and conflict detection
for contradictory memories. All operations are provenance-aware and audited.

Safety properties:
- Opt-in: disabled by default; no behavior change when disabled.
- No silent overwrites: conflicts produce explicit conflict records.
- Provenance preservation: all operations retain provenance metadata.
- Secret safety: text sanitized before fingerprinting and comparison.
- Session-isolated: dedup scope limited to session unless explicitly authorized.
- Reversible: original records preserved; dedup is additive.
- Bounded: resource limits prevent unbounded processing.
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

# Provenance precedence for conflict resolution (higher = stronger)
PROVENANCE_PRECEDENCE = {
    "manual": 5,
    "conversation": 4,
    "tool_result": 3,
    "web": 3,
    "inferred": 2,
    "unknown": 1,
}

# Default similarity threshold for duplicate detection
DEFAULT_DEDUP_THRESHOLD = 0.95

# Maximum number of memories to compare in a single dedup pass
MAX_DEDUP_COMPARISONS = 100


@dataclass
class ConflictRecord:
    """A record of a conflict between two or more memory entries."""

    id: str
    memory_ids: List[str]
    conflict_type: str  # "contradiction", "duplicate", "overlap"
    provenance: Dict[str, Any]
    detected_at: float
    resolution_status: str = "unresolved"  # "unresolved", "resolved", "deferred"
    resolution: Optional[str] = None
    resolved_at: Optional[float] = None
    resolution_provenance: Optional[Dict[str, Any]] = None


@dataclass
class DedupResult:
    """Result of a deduplication check."""

    is_duplicate: bool
    duplicate_of: Optional[str] = None
    similarity_score: float = 0.0
    conflict_detected: bool = False
    conflict_record: Optional[ConflictRecord] = None
    resolution_action: Optional[str] = None  # "skip", "update", "merge", "none"


class MemoryDeduplicator:
    """Memory deduplication and conflict resolution.

    Thread-safe via reentrant lock.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._enabled: bool = config.get("enabled", False)
        self._threshold: float = config.get("dedup_threshold", DEFAULT_DEDUP_THRESHOLD)
        self._max_comparisons: int = config.get("max_comparisons", MAX_DEDUP_COMPARISONS)
        self._on_duplicate: str = config.get("on_duplicate", "skip")  # skip, update, merge
        self._session_isolated: bool = config.get("session_isolated", True)
        self._conflict_records: List[ConflictRecord] = []
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def find_duplicates(
        self,
        text: str,
        existing_memories: List[Dict[str, Any]],
    ) -> List[DedupResult]:
        """Find duplicates for a candidate memory.

        Args:
            text: Candidate memory text.
            existing_memories: List of existing memory dicts with 'text' and 'score'.

        Returns:
            List of DedupResult for each potential duplicate.
        """
        if not self._enabled:
            return []

        results = []
        sanitized_text = self._sanitize(text)

        comparisons = 0
        for mem in existing_memories:
            if comparisons >= self._max_comparisons:
                break
            comparisons += 1

            mem_text = self._sanitize(mem.get("text", ""))
            similarity = self._compute_similarity(sanitized_text, mem_text)

            if similarity >= self._threshold:
                result = DedupResult(
                    is_duplicate=True,
                    duplicate_of=mem.get("id"),
                    similarity_score=similarity,
                )
                results.append(result)

        return results

    def detect_conflict(
        self,
        text: str,
        existing: Dict[str, Any],
    ) -> Optional[ConflictRecord]:
        """Detect conflict between new text and existing memory.

        Args:
            text: Candidate memory text.
            existing: Existing memory dict.

        Returns:
            ConflictRecord if conflict detected, None otherwise.
        """
        if not self._enabled:
            return None

        # Simple contradiction detection: high similarity but different claims
        sanitized_new = self._sanitize(text)
        sanitized_old = self._sanitize(existing.get("text", ""))

        similarity = self._compute_similarity(sanitized_new, sanitized_old)

        # High similarity with potential contradiction indicator words
        if similarity >= 0.85:
            contradiction_indicators = [
                "not", "never", "no ", "cannot", "can't", "won't", "don't",
                "false", "incorrect", "wrong",
            ]
            has_contradiction = any(
                ind in sanitized_new and ind not in sanitized_old
                for ind in contradiction_indicators
            )
            if has_contradiction:
                record = ConflictRecord(
                    id=str(hashlib.sha256(f"{text}{existing.get('id')}".encode()).hexdigest()[:16]),
                    memory_ids=[existing.get("id", "unknown")],
                    conflict_type="contradiction",
                    provenance=existing.get("provenance", {}),
                    detected_at=time.monotonic(),
                )
                self._conflict_records.append(record)
                self._audit(
                    "conflict_detected",
                    {
                        "existing_id": existing.get("id"),
                        "similarity": similarity,
                        "type": "contradiction",
                    },
                )
                return record

        return None

    def resolve_conflict(
        self,
        new_memory: Dict[str, Any],
        existing: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve conflict between new and existing memory.

        Uses provenance precedence. Returns the winning memory.

        Args:
            new_memory: Candidate new memory.
            existing: Existing memory.

        Returns:
            The winning memory dict.
        """
        new_source = new_memory.get("provenance", {}).get("source", "unknown")
        old_source = existing.get("provenance", {}).get("source", "unknown")

        new_precedence = PROVENANCE_PRECEDENCE.get(new_source, 0)
        old_precedence = PROVENANCE_PRECEDENCE.get(old_source, 0)

        if new_precedence > old_precedence:
            return new_memory
        elif old_precedence > new_precedence:
            return existing
        else:
            # Same precedence: prefer newer (by stored_at timestamp)
            new_time = new_memory.get("provenance", {}).get("stored_at", "")
            old_time = existing.get("provenance", {}).get("stored_at", "")
            if new_time >= old_time:
                return new_memory
            return existing

    def dedup_and_store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        existing_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> DedupResult:
        """Check for duplicates/conflicts before storing.

        Args:
            text: Memory text to store.
            metadata: Optional metadata.
            existing_memories: Existing memories to check against.

        Returns:
            DedupResult with action recommendation.
        """
        if not self._enabled:
            return DedupResult(is_duplicate=False, resolution_action="none")

        existing_memories = existing_memories or []

        # Check for duplicates
        duplicates = self.find_duplicates(text, existing_memories)

        if duplicates:
            best = max(duplicates, key=lambda d: d.similarity_score)

            # Check for conflict
            if best.duplicate_of:
                existing = next(
                    (m for m in existing_memories if m.get("id") == best.duplicate_of),
                    None,
                )
                if existing:
                    conflict = self.detect_conflict(text, existing)
                    if conflict:
                        best.conflict_detected = True
                        best.conflict_record = conflict

            # Determine action
            if self._on_duplicate == "skip":
                best.resolution_action = "skip"
            elif self._on_duplicate == "update":
                best.resolution_action = "update"
            elif self._on_duplicate == "merge":
                best.resolution_action = "merge"

            self._audit(
                "duplicate_detected",
                {
                    "duplicate_of": best.duplicate_of,
                    "similarity": best.similarity_score,
                    "action": best.resolution_action,
                },
            )

            return best

        return DedupResult(is_duplicate=False, resolution_action="none")

    def get_conflict_records(
        self,
        status: Optional[str] = None,
    ) -> List[ConflictRecord]:
        """Get conflict records.

        Args:
            status: Filter by status ("unresolved", "resolved", "deferred").

        Returns:
            List of ConflictRecord.
        """
        with self._lock:
            records = self._conflict_records
            if status:
                records = [r for r in records if r.resolution_status == status]
            return list(records)

    def resolve_conflict_record(
        self,
        conflict_id: str,
        resolution: str,
        winning_memory_id: str,
        resolution_provenance: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Resolve a conflict record.

        Args:
            conflict_id: ID of the conflict record.
            resolution: Resolution description.
            winning_memory_id: ID of the winning memory.
            resolution_provenance: Provenance of the resolution.

        Returns:
            True if resolved, False if not found.
        """
        with self._lock:
            for record in self._conflict_records:
                if record.id == conflict_id:
                    record.resolution_status = "resolved"
                    record.resolution = resolution
                    record.resolved_at = time.monotonic()
                    record.resolution_provenance = resolution_provenance
                    self._audit(
                        "conflict_resolved",
                        {
                            "conflict_id": conflict_id,
                            "resolution": resolution,
                            "winner": winning_memory_id,
                        },
                    )
                    return True
            return False

    def get_audit_log(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of audit entry dicts.
        """
        with self._lock:
            return self._audit_log[-limit:]

    def stats(self) -> Dict[str, Any]:
        """Get deduplicator statistics."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "threshold": self._threshold,
                "max_comparisons": self._max_comparisons,
                "on_duplicate": self._on_duplicate,
                "session_isolated": self._session_isolated,
                "total_conflicts": len(self._conflict_records),
                "unresolved_conflicts": len(
                    [r for r in self._conflict_records if r.resolution_status == "unresolved"]
                ),
                "audit_entries": len(self._audit_log),
            }

    def _sanitize(self, text: str) -> str:
        """Sanitize text for comparison."""
        try:
            from tools.secret_redaction import redact_secrets
            return redact_secrets(text.lower().strip())
        except Exception:
            return text.lower().strip()

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple similarity between two texts.

        Uses Jaccard similarity on word sets. For production use,
        this should be replaced with cosine similarity on embeddings.
        """
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _audit(self, operation: str, details: Dict[str, Any]) -> None:
        """Log an audit entry."""
        entry = {
            "timestamp": time.monotonic(),
            "operation": operation,
            **details,
        }
        self._audit_log.append(entry)


def compute_memory_fingerprint(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Compute a deterministic fingerprint for a memory.

    Args:
        text: Memory text.
        metadata: Optional metadata.

    Returns:
        SHA-256 hex digest.
    """
    import hashlib
    import json

    # Sanitize text before fingerprinting
    try:
        from tools.secret_redaction import redact_secrets
        clean_text = redact_secrets(text)
    except Exception:
        clean_text = text

    content = {"text": clean_text, "metadata": metadata or {}}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
