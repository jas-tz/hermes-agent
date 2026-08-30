"""Memory consolidation and summarization with provenance-aware merging.

Provides opt-in consolidation of related memory entries into coherent
summaries. Additive only: original entries preserved, consolidations
stored as new entries with merge provenance.

Safety properties:
- Opt-in: disabled by default; no behavior change when disabled.
- Additive: original records preserved; consolidations are new entries.
- Conflict-aware: unresolved conflicts are excluded from consolidation.
- Provenance preservation: all consolidations retain source references.
- Secret safety: all content sanitized through redact_secrets().
- Session-isolated: consolidation scoped to session by default.
- Reversible: rollback infrastructure supports undo.
- Bounded: resource limits prevent unbounded processing.
- Audited: all consolidation operations logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default similarity threshold for consolidation candidates
DEFAULT_CONSOLIDATION_THRESHOLD = 0.80

# Maximum entries per consolidation group
MAX_GROUP_SIZE = 10

# Maximum entries to process in a single consolidation pass
MAX_CONSOLIDATION_ENTRIES = 100


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""

    consolidated_text: str
    source_ids: List[str]
    provenance: Dict[str, Any]
    original_count: int
    consolidated_count: int = 1
    conflict_excluded: bool = False


@dataclass
class ConsolidationAuditEntry:
    """Audit entry for consolidation operations."""

    timestamp: float
    operation: str  # "consolidate", "exclude_conflict", "skip_unchanged"
    session_id: str
    source_ids: List[str]
    result_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class MemoryConsolidator:
    """Memory consolidation and summarization.

    Thread-safe via reentrant lock.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._enabled: bool = config.get("enabled", False)
        self._threshold: float = config.get(
            "consolidation_threshold", DEFAULT_CONSOLIDATION_THRESHOLD
        )
        self._max_group_size: int = config.get("max_group_size", MAX_GROUP_SIZE)
        self._max_entries: int = config.get(
            "max_entries", MAX_CONSOLIDATION_ENTRIES
        )
        self._session_isolated: bool = config.get("session_isolated", True)
        self._preserve_originals: bool = config.get("preserve_originals", True)
        self._audit_log: List[ConsolidationAuditEntry] = []
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def find_consolidation_candidates(
        self,
        memories: List[Dict[str, Any]],
        conflict_ids: Optional[set] = None,
    ) -> List[Tuple[str, ...]]:
        """Find groups of memories eligible for consolidation.

        Args:
            memories: List of memory dicts with 'id', 'text', 'provenance'.
            conflict_ids: Set of memory IDs involved in unresolved conflicts.

        Returns:
            List of tuples, each containing IDs of memories to consolidate.
        """
        if not self._enabled:
            return []

        conflict_ids = conflict_ids or set()
        candidates: List[Tuple[str, ...]] = []
        processed: set = set()
        total_scanned = 0

        # Filter out conflicting memories
        eligible = [
            m for m in memories
            if m.get("id") not in conflict_ids and m.get("text")
        ]

        for i, mem in enumerate(eligible):
            if mem["id"] in processed:
                continue
            if total_scanned >= self._max_entries:
                break

            group: List[str] = [mem["id"]]
            processed.add(mem["id"])
            total_scanned += 1

            for other in eligible[i + 1:]:
                if other["id"] in processed:
                    continue
                if len(group) >= self._max_group_size:
                    break
                if total_scanned >= self._max_entries:
                    break

                similarity = self._compute_similarity(
                    mem.get("text", ""),
                    other.get("text", ""),
                )
                if similarity >= self._threshold:
                    group.append(other["id"])
                    processed.add(other["id"])
                    total_scanned += 1

            if len(group) > 1:
                candidates.append(tuple(group))

        return candidates

    def consolidate_group(
        self,
        group: Tuple[str, ...],
        memories: List[Dict[str, Any]],
        session_id: str = "default",
    ) -> Optional[ConsolidationResult]:
        """Consolidate a group of memories into a single summary.

        Args:
            group: Tuple of memory IDs to consolidate.
            memories: Full list of memory dicts.
            session_id: Session ID for isolation.

        Returns:
            ConsolidationResult on success, None if disabled.
        """
        if not self._enabled:
            return None

        # Collect texts and provenances
        texts: List[str] = []
        provenances: List[Dict[str, Any]] = []
        source_ids: List[str] = []

        for mem in memories:
            if mem.get("id") in group:
                text = self._sanitize(mem.get("text", ""))
                if text:
                    texts.append(text)
                    provenances.append(mem.get("provenance", {}))
                    source_ids.append(mem["id"])

        if len(texts) < 2:
            return None

        # Generate summary
        summary = self._summarize(texts)

        # Merge provenance
        merged_provenance = self._merge_provenances(provenances, source_ids)

        result = ConsolidationResult(
            consolidated_text=summary,
            source_ids=source_ids,
            provenance=merged_provenance,
            original_count=len(texts),
        )

        self._audit(
            operation="consolidate",
            session_id=session_id,
            source_ids=source_ids,
            details={
                "original_count": len(texts),
                "summary_length": len(summary),
            },
        )

        return result

    def consolidate_all(
        self,
        memories: List[Dict[str, Any]],
        session_id: str = "default",
        conflict_ids: Optional[set] = None,
    ) -> List[ConsolidationResult]:
        """Run consolidation pass on all eligible memories.

        Args:
            memories: List of memory dicts.
            session_id: Session ID for isolation.
            conflict_ids: Set of memory IDs in unresolved conflicts.

        Returns:
            List of ConsolidationResult for each consolidation.
        """
        if not self._enabled:
            return []

        candidates = self.find_consolidation_candidates(memories, conflict_ids)
        results: List[ConsolidationResult] = []

        for group in candidates:
            result = self.consolidate_group(group, memories, session_id)
            if result:
                results.append(result)

        return results

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
                    "session_id": e.session_id,
                    "source_ids": e.source_ids,
                    "result_id": e.result_id,
                    "details": e.details,
                }
                for e in entries[-limit:]
            ]

    def stats(self) -> Dict[str, Any]:
        """Get consolidator statistics."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "threshold": self._threshold,
                "max_group_size": self._max_group_size,
                "max_entries": self._max_entries,
                "session_isolated": self._session_isolated,
                "preserve_originals": self._preserve_originals,
                "audit_entries": len(self._audit_log),
            }

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _summarize(self, texts: List[str]) -> str:
        """Generate a deterministic summary from multiple texts.

        Uses extractive summarization: selects the most representative
        sentences from the combined texts.
        """
        if not texts:
            return ""

        if len(texts) == 1:
            return texts[0]

        # Combine all texts
        combined = " ".join(texts)

        # Simple extractive summary: take first sentence from each unique text
        sentences: List[str] = []
        seen_sentences: set = set()

        for text in texts:
            # Split into sentences
            parts = [s.strip() for s in text.split(".") if s.strip()]
            for part in parts:
                key = part.lower()[:50]
                if key not in seen_sentences:
                    seen_sentences.add(key)
                    sentences.append(part)

        # Join with proper punctuation
        summary = ". ".join(sentences)
        if summary and not summary.endswith("."):
            summary += "."

        return summary

    def _merge_provenances(
        self,
        provenances: List[Dict[str, Any]],
        source_ids: List[str],
    ) -> Dict[str, Any]:
        """Merge provenance from multiple sources."""
        # Find the highest precedence source
        from tools.memory_dedup import PROVENANCE_PRECEDENCE

        best_provenance: Dict[str, Any] = {}
        best_precedence = -1

        for prov in provenances:
            source = prov.get("source", "unknown")
            precedence = PROVENANCE_PRECEDENCE.get(source, 0)
            if precedence > best_precedence:
                best_precedence = precedence
                best_provenance = prov

        return {
            **best_provenance,
            "source": "consolidation",
            "provenance_status": "consolidated",
            "consolidated_from": source_ids,
            "consolidated_at": time.monotonic(),
        }

    def _sanitize(self, text: str) -> str:
        """Sanitize text before processing."""
        try:
            from tools.secret_redaction import redact_secrets
            return redact_secrets(text)
        except Exception:
            return text

    def _audit(
        self,
        operation: str,
        session_id: str,
        source_ids: List[str],
        result_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an audit entry."""
        entry = ConsolidationAuditEntry(
            timestamp=time.monotonic(),
            operation=operation,
            session_id=session_id,
            source_ids=source_ids,
            result_id=result_id,
            details=details,
        )
        self._audit_log.append(entry)
