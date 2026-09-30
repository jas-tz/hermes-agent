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
    """
    plan = planner.plan(goal=goal, task_type=task_type, authority_level=authority_level)
    if not plan.stages:
        return CapabilityPlanResult(plan=plan, graph=TaskGraph())

    capabilities = planner._find_capabilities(task_type, authority_level, goal=goal)
    graph, _ = planner._build_graph(capabilities, task_type, authority_level, goal)
    return CapabilityPlanResult(plan=plan, graph=graph)
