"""Capability Plan Result — Phase 40B.

Clean public contract for planner output that bundles ExecutionPlan
with its source TaskGraph, eliminating the need for callers to
reconstruct the graph from planner internals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from plugins.memory.jarvis.planner.task_graph import (
    ExecutionPlan,
    TaskGraph,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapabilityPlanResult:
    """Immutable plan result containing both ExecutionPlan and TaskGraph."""
    plan: ExecutionPlan
    graph: TaskGraph

    @property
    def is_empty(self) -> bool:
        return len(self.plan.stages) == 0

    @property
    def stage_count(self) -> int:
        return len(self.plan.stages)

    @property
    def task_count(self) -> int:
        return sum(len(s.task_ids) for s in self.plan.stages)


def plan_with_result(
    planner,
    goal: str,
    task_type: str = "default",
    authority_level: str = "L0",
) -> CapabilityPlanResult:
    """Plan a goal and return both ExecutionPlan and TaskGraph.

    This is the public contract for callers that need the graph
    (e.g., CapabilityRuntime). Preserves backward compatibility
    with SubAgentPlanner.plan() which returns only ExecutionPlan.

    The graph returned is the one the plan was linearized from. (It used to be
    rebuilt separately; _build_graph assigns fresh random task ids, so none of
    the plan's task ids resolved in the returned graph.) Raises ValueError if
    any plan task id does not resolve, rather than returning an inconsistent
    pair.
    """
    plan, graph = planner.plan_with_graph(goal=goal, task_type=task_type, authority_level=authority_level)
    if not plan.stages:
        return CapabilityPlanResult(plan=plan, graph=TaskGraph())

    missing = sorted(
        task_id
        for stage in plan.stages
        for task_id in stage.task_ids
        if graph.get_task(task_id) is None
    )
    if missing:
        raise ValueError(f"Plan task ids do not resolve in the plan graph: {missing}")
    return CapabilityPlanResult(plan=plan, graph=graph)
