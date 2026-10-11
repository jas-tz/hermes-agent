"""Capability Runtime — Phase 38/44/46.

Executes an ExecutionPlan by instantiating each concrete capability from
the registry and invoking its execute() method with a constructed
CapabilityContext. Results from each stage are collected and injected as
dependency_results into subsequent stages.

Phase 44: Adds per-task and plan-level timeout. A timed-out capability is
marked as failed; remaining tasks continue.

Phase 46: Timeout validation and non-blocking thread shutdown.

Soft timeout limitation:
    Python cannot forcibly kill a running thread. When a capability times
    out, the runtime marks it as failed and continues, but the underlying
    thread may continue executing in the background until the capability's
    execute() method returns naturally. This means post-timeout side effects
    (e.g., Qdrant writes, continuity updates) may still occur. The caller
    must be aware that a timed-out task's side effects are not guaranteed
    to be prevented.

Constraints:
- No new terminal/filesystem privileges
- No modification to tool_capability_registry.py or subagent_types.py
- Authorization enforcement stays inside the JARVIS boundary
- Failures in one task do not abort the entire plan (configurable per-task)
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from plugins.memory.jarvis.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityResult,
)
from plugins.memory.jarvis.capabilities.registry import JARVISCapabilityRegistry
from plugins.memory.jarvis.planner.task_graph import (
    ExecutionPlan,
    ExecutionStage,
    Task,
    TaskGraph,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskExecutionResult:
    """Result of executing a single task."""
    task_id: str
    capability_name: str
    result: CapabilityResult
    execution_time_seconds: float


@dataclass(frozen=True)
class StageExecutionResult:
    """Result of executing a single stage."""
    stage_index: int
    task_results: List[TaskExecutionResult]
    all_succeeded: bool
    any_succeeded: bool


@dataclass(frozen=True)
class PlanExecutionResult:
    """Result of executing an entire plan."""
    plan: ExecutionPlan
    stage_results: List[StageExecutionResult]
    all_succeeded: bool
    total_execution_time_seconds: float
    task_count: int
    success_count: int
    failure_count: int


class CapabilityRuntimeError(Exception):
    """Raised when the runtime encounters an unrecoverable error."""
    pass


class CapabilityRuntime:
    """Executes capability plans through the authorized dispatch path."""

    def __init__(self, registry: Optional[JARVISCapabilityRegistry] = None) -> None:
        from plugins.memory.jarvis.capabilities.registry import get_jarvis_registry
        self._registry = registry or get_jarvis_registry()

    def execute_plan(
        self,
        plan: ExecutionPlan,
        task_graph: TaskGraph,
        parent_session_id: str = "parent-001",
        subagent_id: str = "sa-runtime",
        authority_level: str = "L0",
        task_type: str = "default",
        delegation_depth: int = 1,
        base_parameters: Optional[Dict[str, Any]] = None,
        task_timeout_seconds: float = 30.0,
        plan_timeout_seconds: float = 120.0,
    ) -> PlanExecutionResult:
        """Execute a complete plan, stage by stage.

        Each stage is executed sequentially. Tasks within a stage are
        executed sequentially (parallel execution deferred to a future phase).
        Results from prior stages are injected as dependency_results.

        Phase 44: task_timeout_seconds bounds each individual capability
        execution. plan_timeout_seconds bounds the entire plan. A timed-out
        task is marked as failed; remaining tasks continue.
        """
        if not plan.stages:
            return PlanExecutionResult(
                plan=plan,
                stage_results=[],
                all_succeeded=True,
                total_execution_time_seconds=0.0,
                task_count=0,
                success_count=0,
                failure_count=0,
            )

        # Phase 46: Validate timeout values
        if task_timeout_seconds <= 0:
            raise ValueError(
                f"task_timeout_seconds must be > 0, got {task_timeout_seconds}"
            )
        if plan_timeout_seconds < 0:
            raise ValueError(
                f"plan_timeout_seconds must be >= 0, got {plan_timeout_seconds}"
            )

        stage_results: List[StageExecutionResult] = []
        accumulated_results: Dict[str, Any] = {}
        total_start = time.monotonic()
        total_success = 0
        total_failure = 0

        for stage in plan.stages:
            # Check plan timeout before each stage
            elapsed = time.monotonic() - total_start
            if elapsed >= plan_timeout_seconds:
                logger.warning(
                    "Plan timeout (%.1fs) exceeded after %.1fs — skipping remaining stages",
                    plan_timeout_seconds, elapsed,
                )
                break

            stage_result = self._execute_stage(
                stage=stage,
                task_graph=task_graph,
                accumulated_results=accumulated_results,
                parent_session_id=parent_session_id,
                subagent_id=subagent_id,
                authority_level=authority_level,
                task_type=task_type,
                delegation_depth=delegation_depth,
                base_parameters=base_parameters or {},
                task_timeout_seconds=task_timeout_seconds,
            )
            stage_results.append(stage_result)

            # Merge results for downstream injection
            for tr in stage_result.task_results:
                accumulated_results[tr.capability_name] = tr.result.data
                # Also index by task_id for generality
                accumulated_results[tr.task_id] = tr.result.data

            total_success += sum(1 for tr in stage_result.task_results if tr.result.success)
            total_failure += sum(1 for tr in stage_result.task_results if not tr.result.success)

        total_elapsed = time.monotonic() - total_start
        all_succeeded = all(sr.all_succeeded for sr in stage_results) if stage_results else True

        return PlanExecutionResult(
            plan=plan,
            stage_results=stage_results,
            all_succeeded=all_succeeded,
            total_execution_time_seconds=total_elapsed,
            task_count=total_success + total_failure,
            success_count=total_success,
            failure_count=total_failure,
        )

    def _execute_stage(
        self,
        stage: ExecutionStage,
        task_graph: TaskGraph,
        accumulated_results: Dict[str, Any],
        parent_session_id: str,
        subagent_id: str,
        authority_level: str,
        task_type: str,
        delegation_depth: int,
        base_parameters: Dict[str, Any],
        task_timeout_seconds: float = 30.0,
    ) -> StageExecutionResult:
        """Execute all tasks in a single stage."""
        task_results: List[TaskExecutionResult] = []

        for task_id in sorted(stage.task_ids):
            task = task_graph.get_task(task_id)
            if task is None:
                # A plan that names a task its graph lacks is inconsistent;
                # record a failure so it can never be reported as success.
                logger.warning("Task '%s' not found in graph; recording it as failed", task_id)
                task_results.append(TaskExecutionResult(
                    task_id=task_id,
                    capability_name="",
                    result=CapabilityResult.failure_result(
                        error=f"Task '{task_id}' not found in plan graph",
                    ),
                    execution_time_seconds=0.0,
                ))
                continue

            result = self._execute_task(
                task=task,
                accumulated_results=accumulated_results,
                parent_session_id=parent_session_id,
                subagent_id=subagent_id,
                authority_level=authority_level,
                task_type=task_type,
                delegation_depth=delegation_depth,
                base_parameters=base_parameters,
                task_timeout_seconds=task_timeout_seconds,
            )
            task_results.append(result)

        all_succeeded = all(r.result.success for r in task_results)
        any_succeeded = any(r.result.success for r in task_results)

        return StageExecutionResult(
            stage_index=stage.stage_index,
            task_results=task_results,
            all_succeeded=all_succeeded,
            any_succeeded=any_succeeded,
        )

    def _execute_task(
        self,
        task: Task,
        accumulated_results: Dict[str, Any],
        parent_session_id: str,
        subagent_id: str,
        authority_level: str,
        task_type: str,
        delegation_depth: int,
        base_parameters: Dict[str, Any],
        task_timeout_seconds: float = 30.0,
    ) -> TaskExecutionResult:
        """Execute a single task capability.

        Phase 44: Wraps execution in ThreadPoolExecutor to enforce
        task_timeout_seconds. On timeout, marks task as failed and
        continues remaining tasks.
        """
        start = time.monotonic()

        # Look up the capability class from registry
        cap_instance = self._registry.get_capability(task.capability_name)
        if cap_instance is None:
            elapsed = time.monotonic() - start
            return TaskExecutionResult(
                task_id=task.id,
                capability_name=task.capability_name,
                result=CapabilityResult.failure_result(
                    error=f"Capability '{task.capability_name}' not found in registry",
                ),
                execution_time_seconds=elapsed,
            )

        # Build dependency results for this task
        dep_results = {}
        reqs = cap_instance.get_requirements()
        for dep_name in reqs.dependencies:
            if dep_name in accumulated_results:
                dep_results[dep_name] = accumulated_results[dep_name]

        # Build context
        context = CapabilityContext(
            execution_id=f"exec-{task.id}",
            subagent_id=subagent_id,
            parent_session_id=parent_session_id,
            task_goal=base_parameters.get("goal", task.capability_name),
            task_type=task_type,
            authority_level=authority_level,
            delegation_depth=delegation_depth,
            parameters={**base_parameters, **task.parameters},
            dependency_results=dep_results,
        )

        # Validate
        valid, reason = cap_instance.validate(context)
        if not valid:
            elapsed = time.monotonic() - start
            return TaskExecutionResult(
                task_id=task.id,
                capability_name=task.capability_name,
                result=CapabilityResult.failure_result(
                    error=f"Validation failed: {reason}",
                ),
                execution_time_seconds=elapsed,
            )

        # Phase 44/46: Execute with timeout using ThreadPoolExecutor.
        # NOTE: Python cannot forcibly kill a thread. We use a dedicated
        # executor per task and call shutdown(wait=False) so the caller
        # does not block if the underlying thread is still running.
        # The thread may continue executing in the background.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(cap_instance.execute, context)
            result = future.result(timeout=task_timeout_seconds)
        except FutureTimeoutError:
            elapsed = time.monotonic() - start
            logger.warning(
                "Capability '%s' timed out after %.1fs",
                task.capability_name, task_timeout_seconds,
            )
            pool.shutdown(wait=False)
            return TaskExecutionResult(
                task_id=task.id,
                capability_name=task.capability_name,
                result=CapabilityResult.failure_result(
                    error=f"Capability execution timed out after {task_timeout_seconds:.0f} seconds",
                ),
                execution_time_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.warning("Capability '%s' raised exception: %s", task.capability_name, exc)
            pool.shutdown(wait=True)
            return TaskExecutionResult(
                task_id=task.id,
                capability_name=task.capability_name,
                result=CapabilityResult.failure_result(
                    error=f"Execution exception: {exc}",
                ),
                execution_time_seconds=elapsed,
            )

        elapsed = time.monotonic() - start
        pool.shutdown(wait=True)
        return TaskExecutionResult(
            task_id=task.id,
            capability_name=task.capability_name,
            result=result,
            execution_time_seconds=elapsed,
        )


def execute_plan(
    plan: ExecutionPlan,
    task_graph: TaskGraph,
    **kwargs: Any,
) -> PlanExecutionResult:
    """Convenience function using the global registry runtime."""
    runtime = CapabilityRuntime()
    return runtime.execute_plan(plan, task_graph, **kwargs)
