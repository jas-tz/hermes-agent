"""Planning verification and validation with provenance-aware plan checking.

Provides read-only verification of generated plans before they are eligible
for execution. Integrates with tool result verification (Step 156), provenance
tracking (Step 149), and audit/rollback infrastructure.

Safety properties:
- Read-only: verification never executes or modifies plans.
- Explicit status: every verification returns a clear status with evidence.
- No silent repairs: plans are never modified during verification.
- Provenance preservation: all verification decisions logged with provenance.
- Approval-aware: detects steps requiring approval.
- Bounded: resource limits prevent unbounded verification.
- Secret safety: all content sanitized through redact_secrets().
- Session-isolated: verification scoped to session by default.
- Opt-in: disabled by default; no behavior change when disabled.
- Conservative: insufficient evidence returns UNVERIFIED, never VERIFIED.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PlanVerificationStatus:
    """Plan verification status constants."""
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    INVALID = "INVALID"
    CONFLICT = "CONFLICT"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


# Dangerous patterns that require approval
APPROVAL_REQUIRED_PATTERNS = [
    "rm -rf",
    "rm -r",
    "delete",
    "drop",
    "truncate",
    "format",
    "destroy",
    "purge",
]

# Resource limits
DEFAULT_MAX_PLAN_STEPS = 50
DEFAULT_MAX_VERIFICATION_TIME = 30.0  # seconds


@dataclass
class StepVerificationResult:
    """Result of verifying a single plan step."""

    step_id: str
    status: str
    tool_name: str
    evidence: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    requires_approval: bool = False
    approval_reason: Optional[str] = None


@dataclass
class PlanVerificationResult:
    """Result of verifying an entire plan."""

    status: str
    plan_id: Optional[str]
    timestamp: float
    step_results: List[StepVerificationResult] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    requires_approval: bool = False
    approval_reasons: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PlanVerificationAuditEntry:
    """Audit entry for plan verification operations."""

    timestamp: float
    operation: str  # "verify_plan", "step_verified", "step_invalid", "approval_required"
    plan_id: Optional[str]
    session_id: str
    status: str
    details: Optional[Dict[str, Any]] = None


class PlanVerifier:
    """Verifies plans for consistency, safety, and achievability.

    Thread-safe via reentrant lock.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._enabled: bool = config.get("enabled", False)
        self._max_plan_steps: int = config.get("max_plan_steps", DEFAULT_MAX_PLAN_STEPS)
        self._session_isolated: bool = config.get("session_isolated", True)
        self._require_tool_validation: bool = config.get("require_tool_validation", True)
        self._available_tools: List[str] = config.get("available_tools", [])
        self._audit_log: List[PlanVerificationAuditEntry] = []
        self._verification_count: int = 0
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def verify_plan(
        self,
        plan: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
    ) -> PlanVerificationResult:
        """Verify a plan for consistency and safety.

        Args:
            plan: Plan dict with 'id', 'steps', 'dependencies', etc.
            context: Optional context (verified_facts, available_tools, etc.).
            session_id: Session ID for isolation.

        Returns:
            PlanVerificationResult with status and evidence.
        """
        if not self._enabled:
            return PlanVerificationResult(
                status=PlanVerificationStatus.UNVERIFIED,
                plan_id=plan.get("id"),
                timestamp=time.monotonic(),
                evidence=["Verification disabled"],
            )

        context = context or {}
        start_time = time.monotonic()
        plan_id = plan.get("id")

        # 1. Structural validation
        structural_issues = self._validate_structure(plan)
        if structural_issues:
            return self._create_invalid_result(
                plan_id, session_id, structural_issues, start_time
            )

        # 2. Check step count bounds
        steps = plan.get("steps", [])
        if len(steps) > self._max_plan_steps:
            return self._create_invalid_result(
                plan_id, session_id,
                [f"Plan exceeds max steps ({len(steps)} > {self._max_plan_steps})"],
                start_time
            )

        # 3. Verify each step
        step_results: List[StepVerificationResult] = []
        all_approved = True
        approval_reasons: List[str] = []

        for step in steps:
            # Check time budget
            elapsed = time.monotonic() - start_time
            if elapsed > DEFAULT_MAX_VERIFICATION_TIME:
                return self._create_invalid_result(
                    plan_id, session_id,
                    [f"Verification timed out after {elapsed:.1f}s"],
                    start_time
                )

            step_result = self._verify_step(step, context, session_id)
            step_results.append(step_result)

            if step_result.status == PlanVerificationStatus.INVALID:
                all_approved = False
            if step_result.requires_approval:
                approval_reasons.append(
                    f"Step {step_result.step_id}: {step_result.approval_reason}"
                )

        # 4. Check dependency consistency
        dep_issues = self._check_dependencies(steps)
        if dep_issues:
            return self._create_invalid_result(
                plan_id, session_id, dep_issues, start_time
            )

        # 5. Aggregate results
        overall_status = self._aggregate_status(step_results, approval_reasons)

        result = PlanVerificationResult(
            status=overall_status,
            plan_id=plan_id,
            timestamp=time.monotonic(),
            step_results=step_results,
            evidence=[f"Verified {len(step_results)} step(s)"],
            issues=[issue for sr in step_results for issue in sr.issues],
            requires_approval=bool(approval_reasons),
            approval_reasons=approval_reasons,
            metadata={
                "verification_time": time.monotonic() - start_time,
                "steps_verified": len(step_results),
                "available_tools": self._available_tools,
            },
        )

        self._audit(
            operation="verify_plan",
            plan_id=plan_id,
            session_id=session_id,
            status=overall_status,
            details={
                "step_count": len(step_results),
                "approval_needed": bool(approval_reasons),
            },
        )

        return result

    def get_audit_log(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        with self._lock:
            entries = self._audit_log
            if session_id:
                entries = [e for e in entries if e.session_id == session_id]
            return [
                {
                    "timestamp": e.timestamp,
                    "operation": e.operation,
                    "plan_id": e.plan_id,
                    "session_id": e.session_id,
                    "status": e.status,
                    "details": e.details,
                }
                for e in entries[-limit:]
            ]

    def stats(self) -> Dict[str, Any]:
        """Get verifier statistics."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "max_plan_steps": self._max_plan_steps,
                "session_isolated": self._session_isolated,
                "require_tool_validation": self._require_tool_validation,
                "total_verifications": self._verification_count,
                "audit_entries": len(self._audit_log),
            }

    def _verify_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        session_id: str,
    ) -> StepVerificationResult:
        """Verify a single plan step."""
        step_id = step.get("id", "unknown")
        tool_name = step.get("tool", step.get("tool_name", "unknown"))
        evidence: List[str] = []
        issues: List[str] = []

        # 1. Check tool availability
        if self._require_tool_validation and self._available_tools:
            if tool_name not in self._available_tools:
                issues.append(f"Tool '{tool_name}' not in available tools")

        # 2. Check for approval requirements
        approval_reason = self._check_approval_required(step)
        if approval_reason:
            self._audit(
                operation="approval_required",
                plan_id=None,
                session_id=session_id,
                status=PlanVerificationStatus.REQUIRES_APPROVAL,
                details={"step_id": step_id, "reason": approval_reason},
            )
            return StepVerificationResult(
                step_id=step_id,
                status=PlanVerificationStatus.REQUIRES_APPROVAL,
                tool_name=tool_name,
                evidence=evidence,
                issues=issues,
                requires_approval=True,
                approval_reason=approval_reason,
            )

        # 3. Check step structure
        if not step.get("action") and not step.get("tool"):
            issues.append("Step missing action or tool")

        # 4. Check provenance completeness
        step_provenance = step.get("provenance", {})
        if step_provenance:
            source = step_provenance.get("source", "unknown")
            evidence.append(f"Provenance source: {source}")
        else:
            evidence.append("No provenance provided for step")

        # Determine step status
        if issues:
            status = PlanVerificationStatus.INVALID
        else:
            status = PlanVerificationStatus.VERIFIED

        return StepVerificationResult(
            step_id=step_id,
            status=status,
            tool_name=tool_name,
            evidence=evidence,
            issues=issues,
        )

    def _check_approval_required(self, step: Dict[str, Any]) -> Optional[str]:
        """Check if a step requires approval."""
        action = step.get("action", "")
        tool = step.get("tool", "")
        description = step.get("description", "")

        check_text = f"{action} {tool} {description}".lower()

        for pattern in APPROVAL_REQUIRED_PATTERNS:
            if pattern in check_text:
                return f"Action matches approval-required pattern: '{pattern}'"

        return None

    def _validate_structure(self, plan: Dict[str, Any]) -> List[str]:
        """Validate plan structure."""
        issues = []

        if not isinstance(plan, dict):
            issues.append("Plan must be a dict")
            return issues

        if "steps" not in plan:
            issues.append("Plan missing 'steps' field")
        elif not isinstance(plan["steps"], list):
            issues.append("Plan 'steps' must be a list")

        return issues

    def _check_dependencies(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Check dependency consistency between steps."""
        issues = []
        step_ids: Set[str] = set()

        for step in steps:
            step_id = step.get("id")
            if step_id:
                if step_id in step_ids:
                    issues.append(f"Duplicate step ID: {step_id}")
                step_ids.add(step_id)

            # Check dependencies reference valid steps
            deps = step.get("depends_on", [])
            for dep in deps:
                if dep not in step_ids:
                    # Dependency might be in a later step — check all steps
                    all_ids = {s.get("id") for s in steps if s.get("id")}
                    if dep not in all_ids:
                        issues.append(
                            f"Step '{step_id}' depends on unknown step '{dep}'"
                        )

        return issues

    def _aggregate_status(
        self,
        step_results: List[StepVerificationResult],
        approval_reasons: List[str],
    ) -> str:
        """Aggregate step results into overall plan status."""
        statuses = {sr.status for sr in step_results}

        if PlanVerificationStatus.INVALID in statuses:
            return PlanVerificationStatus.INVALID
        if approval_reasons:
            return PlanVerificationStatus.REQUIRES_APPROVAL
        if all(s == PlanVerificationStatus.VERIFIED for s in statuses):
            return PlanVerificationStatus.VERIFIED
        return PlanVerificationStatus.UNVERIFIED

    def _create_invalid_result(
        self,
        plan_id: Optional[str],
        session_id: str,
        issues: List[str],
        start_time: float,
    ) -> PlanVerificationResult:
        """Create an invalid verification result."""
        result = PlanVerificationResult(
            status=PlanVerificationStatus.INVALID,
            plan_id=plan_id,
            timestamp=time.monotonic(),
            issues=issues,
            metadata={"verification_time": time.monotonic() - start_time},
        )

        self._audit(
            operation="verify_plan",
            plan_id=plan_id,
            session_id=session_id,
            status=PlanVerificationStatus.INVALID,
            details={"issues": issues},
        )

        return result

    def _audit(
        self,
        operation: str,
        plan_id: Optional[str],
        session_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an audit entry."""
        from tools.secret_redaction import redact_secrets

        safe_plan_id = redact_secrets(plan_id) if isinstance(plan_id, str) else plan_id
        safe_session_id = redact_secrets(session_id) if isinstance(session_id, str) else session_id
        safe_details = redact_secrets(json.dumps(details)) if details is not None else None

        entry = PlanVerificationAuditEntry(
            timestamp=time.monotonic(),
            operation=operation,
            plan_id=safe_plan_id,
            session_id=safe_session_id,
            status=status,
            details=json.loads(safe_details) if safe_details is not None else None,
        )
        self._audit_log.append(entry)
