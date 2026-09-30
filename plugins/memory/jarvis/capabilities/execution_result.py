"""Capability Execution Result — Phase 40B.

Stable, bounded representation of capability execution results.
Passed from CapabilityRuntime → child/parent without exposing
large raw tool outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CapabilityExecutionStatus(Enum):
    """Status of capability plan execution."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    SKIPPED_APPROVAL_REQUIRED = "skipped_approval_required"
    NOT_EXECUTED = "not_executed"


@dataclass(frozen=True)
class TaskSummary:
    """Bounded summary of a single task execution."""
    capability_name: str
    success: bool
    error: Optional[str] = None
    # Bounded: only a short preview, never full tool output
    output_preview: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "capability_name": self.capability_name,
            "success": self.success,
        }
        if self.error:
            d["error"] = self.error[:200] if self.error else None
        if self.output_preview:
            d["output_preview"] = self.output_preview[:200]
        return d


@dataclass(frozen=True)
class CapabilityExecutionResult:
    """Immutable, bounded result of capability plan execution."""
    status: CapabilityExecutionStatus
    task_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    tasks: List[TaskSummary] = field(default_factory=list)
    # Bounded summary string (not full output)
    summary: str = ""
    # Original plan/graph for inspection (not serialized to parent)
    _plan: Any = None
    _graph: Any = None

    @property
    def all_succeeded(self) -> bool:
        return self.status == CapabilityExecutionStatus.SUCCESS

    @property
    def has_failures(self) -> bool:
        return self.failure_count > 0

    @property
    def approval_required(self) -> bool:
        return self.status == CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED

    @property
    def executed(self) -> bool:
        """True if a plan was actually run (success/partial/failure), False if skipped or not present."""
        return self.status in (
            CapabilityExecutionStatus.SUCCESS,
            CapabilityExecutionStatus.PARTIAL,
            CapabilityExecutionStatus.FAILURE,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a bounded dict safe for parent contract."""
        return {
            "status": self.status.value,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "summary": self.summary[:500] if self.summary else "",
            "tasks": [t.to_dict() for t in self.tasks[:10]],  # Cap at 10 task summaries
        }

    @classmethod
    def from_plan_execution_result(cls, result) -> "CapabilityExecutionResult":
        """Construct from a CapabilityRuntime PlanExecutionResult."""
        # Determine status
        if result.all_succeeded:
            status = CapabilityExecutionStatus.SUCCESS
        elif result.success_count > 0:
            status = CapabilityExecutionStatus.PARTIAL
        elif result.task_count > 0:
            status = CapabilityExecutionStatus.FAILURE
        else:
            status = CapabilityExecutionStatus.NOT_EXECUTED

        # Build bounded task summaries
        tasks = []
        for stage_result in result.stage_results:
            for tr in stage_result.task_results:
                # Extract bounded output preview
                output_preview = None
                if tr.result.data and isinstance(tr.result.data, dict):
                    # Only include summary fields, not raw tool output
                    if "ok" in tr.result.data:
                        output_preview = f"ok={tr.result.data['ok']}"
                    elif "count" in tr.result.data:
                        output_preview = f"count={tr.result.data['count']}"
                    elif "results" in tr.result.data and isinstance(tr.result.data["results"], list):
                        output_preview = f"{len(tr.result.data['results'])} results"

                tasks.append(TaskSummary(
                    capability_name=tr.capability_name,
                    success=tr.result.success,
                    error=tr.result.error[:200] if tr.result.error else None,
                    output_preview=output_preview,
                ))

        summary = f"Capability plan executed: {result.success_count}/{result.task_count} succeeded"

        return cls(
            status=status,
            task_count=result.task_count,
            success_count=result.success_count,
            failure_count=result.failure_count,
            tasks=tasks,
            summary=summary,
        )

    @classmethod
    def skipped_for_approval(cls, reason: str = "") -> "CapabilityExecutionResult":
        """Create a result indicating execution was skipped due to approval requirement."""
        return cls(
            status=CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED,
            summary=reason or "Skipped: plan contains approval-required capabilities",
        )

    @classmethod
    def failed(cls, reason: str) -> "CapabilityExecutionResult":
        """Create a failure result."""
        return cls(
            status=CapabilityExecutionStatus.FAILURE,
            summary=reason[:500],
        )
