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
from typing import List, Optional, Tuple

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
        return self.plan_with_graph(goal, task_type, authority_level)[0]

    def plan_with_graph(
        self,
        goal: str,
        task_type: str = "default",
        authority_level: str = "L0",
    ) -> Tuple[ExecutionPlan, TaskGraph]:
        """Build an execution plan and return it with the TaskGraph it was
        linearized from, so every plan task id resolves in that graph.

        _build_graph assigns fresh random task ids on every call, so the graph
        must never be rebuilt separately for a plan. An empty plan comes with
        an empty graph.
        """
        if not goal:
            raise SubAgentPlannerError("Goal must be non-empty")
        if authority_level not in ("L0", "L1", "L2", "L3"):
            raise SubAgentPlannerError(f"Invalid authority_level: {authority_level}")

        # 1. Find executable capabilities
        capabilities = self._find_capabilities(task_type, authority_level, goal)
        if not capabilities:
            logger.info("No capabilities available for task_type=%s level=%s",
                        task_type, authority_level)
            return ExecutionPlan(stages=[]), TaskGraph()

        # 2. Build task graph
        graph, cap_by_id = self._build_graph(capabilities, task_type, authority_level, goal)

        # 3. Linearize
        try:
            plan = execution_plan_from_graph(graph)
        except TaskGraphError as exc:
            raise SubAgentPlannerError(f"Failed to build plan: {exc}") from exc

        logger.info("Planned %d stages for goal '%s'",
                     len(plan.stages), goal[:60])
        return plan, graph

    def _find_capabilities(
        self, task_type: str, authority_level: str, goal: str = ""
    ) -> List[Capability]:
        """Find capabilities executable in the given context.

        If goal is provided, filters to only capabilities relevant to the goal
        (plus their dependencies). Without a goal, returns all authorized capabilities
        (backward-compatible behavior).
        """
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

        # Phase 41: Goal-aware filtering
        if goal and goal.strip():
            relevant = self._filter_by_goal_relevance(all_caps, goal)
            if relevant:
                # Always include dependencies of relevant capabilities
                with_deps = self._include_dependencies(relevant)
                return with_deps
            # No relevant capabilities found — fall back to all for vague goals
            goal_tokens = [t for t in goal.lower().split() if len(t) >= 3]
            if len(goal_tokens) < 3:
                return all_caps  # Vague goal — return all authorized
            logger.info("No capabilities match goal '%s' — deferring to model", goal[:60])
            return []

        return all_caps

    def _filter_by_goal_relevance(self, capabilities: List[Capability], goal: str) -> List[Capability]:
        """Filter capabilities by relevance to the goal.

        Scores each capability by matching goal tokens against:
        - capability name (high weight)
        - capability description (medium weight)
        - capability category (low weight)

        Returns capabilities with score > 0.
        """
        import re
        goal_lower = goal.lower()
        # Tokenize: split on non-alphanumeric, filter to meaningful words (>=3 chars)
        goal_tokens = set(t for t in re.findall(r'[a-z0-9]+', goal_lower) if len(t) >= 3)

        scored = []
        for cap in capabilities:
            score = self._score_capability_relevance(cap, goal_lower, goal_tokens)
            if score > 0:
                scored.append((score, cap))

        # Sort by score descending for deterministic ordering
        scored.sort(key=lambda x: x[0], reverse=True)
        return [cap for _, cap in scored]

    def _score_capability_relevance(self, cap: Capability, goal_lower: str, goal_tokens: set) -> int:
        """Score a capability's relevance to the goal.

        Returns:
            0 if not relevant, higher score = more relevant.
        """
        score = 0
        name = cap.get_name().lower()
        desc = cap.get_description().lower()
        category = cap.get_category().value.lower()

        # Name match (highest weight)
        if name in goal_lower or goal_lower in name:
            score += 10
        else:
            # Partial name token match
            name_tokens = set(name.replace('_', ' ').split())
            matching = goal_tokens & name_tokens
            score += len(matching) * 3

        # Description match (medium weight)
        desc_tokens = set(t for t in desc.split() if len(t) >= 3)
        matching_desc = goal_tokens & desc_tokens
        score += len(matching_desc)

        # Category match (low weight)
        if category in goal_lower:
            score += 1

        return score

    def _include_dependencies(self, capabilities: List[Capability]) -> List[Capability]:
        """Include all dependencies of the given capabilities.

        Walks the dependency graph and adds any required capabilities.
        """
        included = {cap.get_name() for cap in capabilities}
        result = list(capabilities)

        # Walk dependencies
        to_process = list(capabilities)
        while to_process:
            cap = to_process.pop()
            reqs = cap.get_requirements()
            for dep_name in reqs.dependencies:
                if dep_name not in included:
                    dep_cap = self._registry.get_capability(dep_name)
                    if dep_cap is not None:
                        included.add(dep_name)
                        result.append(dep_cap)
                        to_process.append(dep_cap)

        return result

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
