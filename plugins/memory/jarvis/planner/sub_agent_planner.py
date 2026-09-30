"""Sub-Agent Planner — Phase 36B.

Decomposes a user goal into a staged capability execution plan by:
1. Consulting the JARVIS capability registry for available capabilities
2. Selecting those whose requirements are satisfied by the goal's task_type
3. Building a TaskGraph that respects capability dependencies
4. Producing an ExecutionPlan with parallelizable stages

Constraints (Phase 36B):
- No Odoo / external tools
- No new terminal / filesystem privileges for sub-agents
- Authorization enforcement stays inside the JARVIS boundary
- The planner does NOT execute anything — it only produces a plan.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from plugins.memory.jarvis.capabilities.base import (
    Capability,
    CapabilityCategory,
    CapabilityRequirements,
)
from plugins.memory.jarvis.capabilities.registry import (
    JARVISCapabilityRegistry,
    get_jarvis_registry,
)
from plugins.memory.jarvis.planner.task_graph import (
    Task,
    TaskGraph,
    TaskGraphError,
    ExecutionPlan,
    execution_plan_from_graph,
)

logger = logging.getLogger(__name__)


class SubAgentPlannerError(Exception):
    pass


class SubAgentPlanner:
    """Decomposes a user goal into an ExecutionPlan using the capability registry."""

    def __init__(self, registry: Optional[JARVISCapabilityRegistry] = None) -> None:
        self._registry = registry or get_jarvis_registry()

    def plan(
        self,
        goal: str,
        task_type: str = "default",
        authority_level: str = "L0",
    ) -> ExecutionPlan:
        """Build an execution plan for a goal.

        Steps:
        1. Find capabilities executable at (authority_level, task_type)
        2. Build a TaskGraph with dependency edges from capability requirements
        3. Linearize into a staged ExecutionPlan
        """
        if not goal:
            raise SubAgentPlannerError("Goal must be non-empty")
        if authority_level not in ("L0", "L1", "L2", "L3"):
            raise SubAgentPlannerError(f"Invalid authority_level: {authority_level}")

        # 1. Find executable capabilities
        capabilities = self._find_capabilities(task_type, authority_level)
        if not capabilities:
            logger.warning("No capabilities available for task_type=%s level=%s",
                           task_type, authority_level)
            return ExecutionPlan(stages=[])

        # 2. Build task graph
        graph, cap_by_id = self._build_graph(capabilities, task_type, authority_level, goal)

        # 3. Linearize
        try:
            plan = execution_plan_from_graph(graph)
        except TaskGraphError as exc:
            raise SubAgentPlannerError(f"Failed to build plan: {exc}") from exc

        logger.info("Planned %d stages for goal '%s'",
                     len(plan.stages), goal[:60])
        return plan

    def _find_capabilities(
        self, task_type: str, authority_level: str
    ) -> List[Capability]:
        """Find capabilities executable in the given context."""
        all_caps = []
        for cap in self._registry.list_capabilities():
            instance = self._registry.get_capability(cap)
            if instance is None:
                continue
            reqs = instance.get_requirements()
            # Check authority level
            level_order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
            if level_order.get(authority_level, 0) < level_order.get(reqs.min_authority_level, 0):
                continue
            # Check task-type permissibility
            permitted, _ = self._registry.validate_capability_for_execution(
                instance.get_name(), task_type, authority_level
            )
            if not permitted:
                continue
            all_caps.append(instance)
        return all_caps

    def _build_graph(
        self,
        capabilities: List[Capability],
        task_type: str,
        authority_level: str,
        goal: str,
    ) -> tuple[TaskGraph, dict]:
        """Build a TaskGraph from a set of capabilities.

        Dependency edges are derived from capability requirements' `dependencies` field.
        """
        graph = TaskGraph()
        cap_by_id: dict = {}

        # First pass: add all tasks
        for cap in capabilities:
            task_id = f"task_{cap.get_name()}_{uuid.uuid4().hex[:6]}"
            task = Task(
                id=task_id,
                capability_name=cap.get_name(),
                task_type=task_type,
                authority_level=authority_level,
                parameters={"goal": goal},
            )
            graph.add_task(task)
            cap_by_id[cap.get_name()] = task_id

        # Second pass: add dependency edges
        for cap in capabilities:
            reqs = cap.get_requirements()
            if reqs.dependencies:
                target_id = cap_by_id.get(cap.get_name())
                for dep_name in reqs.dependencies:
                    source_id = cap_by_id.get(dep_name)
                    if source_id and target_id:
                        try:
                            graph.add_edge(source_id, target_id)
                        except TaskGraphError:
                            # Skip invalid edges (e.g., authority escalation)
                            logger.warning(
                                "Skipping dependency edge %s → %s for cap %s",
                                dep_name, cap.get_name(), cap.get_name(),
                            )

        return graph, cap_by_id


def create_default_planner() -> SubAgentPlanner:
    """Create a SubAgentPlanner using the global JARVIS registry."""
    return SubAgentPlanner(registry=get_jarvis_registry())
