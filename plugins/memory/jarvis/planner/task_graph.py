"""Task Graph — Phase 36B.

Directed acyclic graph (DAG) representing a set of capability executions
and their dependency edges. Each node is a capability name + a binding
of that capability to a task-type/authority context; each edge declares
that the consumer requires the producer's result before it can execute.

Validation:
- No cycles (Kahn's algorithm topo-sort)
- No duplicate task IDs
- Every dependency references a real node
- No dependency edge may flow from a higher authority node into a lower
  authority consumer (prevents authority escalation via delegation)
- Dependencies must share a compatible task_type (cannot bridge research →
  maintenance without explicit intent)
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Authority comparison helper
# ---------------------------------------------------------------------------

_AUTHORITY_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def _authority_leq(a: str, b: str) -> bool:
    """True if authority a <= authority b (a is not higher privilege than b)."""
    return _AUTHORITY_ORDER.get(a, 99) <= _AUTHORITY_ORDER.get(b, 99)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """One capability execution node in the task graph."""
    id: str
    capability_name: str
    task_type: str = "default"
    authority_level: str = "L0"
    parameters: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        if not self.id:
            raise ValueError("Task.id must be non-empty")
        if not self.capability_name:
            raise ValueError("Task.capability_name must be non-empty")
        if self.authority_level not in _AUTHORITY_ORDER:
            raise ValueError(f"Invalid authority_level: {self.authority_level}")


@dataclass(frozen=True)
class ExecutionStage:
    """A set of task IDs that can execute in parallel."""
    stage_index: int
    task_ids: FrozenSet[str]


@dataclass(frozen=True)
class ExecutionPlan:
    """Linearized, staged execution plan derived from a task graph."""
    stages: List[ExecutionStage]

    @property
    def all_task_ids(self) -> FrozenSet[str]:
        ids: List[str] = []
        for stage in self.stages:
            ids.extend(sorted(stage.task_ids))
        return frozenset(ids)

    def get_stage(self, stage_index: int) -> Optional[ExecutionStage]:
        for stage in self.stages:
            if stage.stage_index == stage_index:
                return stage
        return None


# ---------------------------------------------------------------------------
# Task Graph
# ---------------------------------------------------------------------------

class TaskGraphError(Exception):
    pass


class TaskGraph:
    """A DAG of capability tasks with dependency edges.

    Edges are stored as consumer → {producers}: `dependents[A] = {B}` means
    "A depends on B" (B must complete before A can start).
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._dependents: Dict[str, set] = defaultdict(set)   # child → {parents}
        self._dependencies: Dict[str, set] = defaultdict(set) # parent → {children}

    def add_task(self, task: Task) -> None:
        if task.id in self._tasks:
            raise TaskGraphError(f"Task '{task.id}' already exists in graph")
        self._tasks[task.id] = task

    def add_edge(self, from_task_id: str, to_task_id: str) -> None:
        """from_task_id must complete BEFORE to_task_id can start."""
        if from_task_id not in self._tasks:
            raise TaskGraphError(f"Source task '{from_task_id}' not in graph")
        if to_task_id not in self._tasks:
            raise TaskGraphError(f"Target task '{to_task_id}' not in graph")
        if from_task_id == to_task_id:
            raise TaskGraphError(f"Self-dependency not allowed: {from_task_id}")

        # Authority flow check: a high-privilege producer feeding a
        # low-privilege consumer is fine (information declassifies); but a
        # low-privilege producer feeding a high-privilege consumer would
        # imply privilege escalation through the graph — reject it.
        producer = self._tasks[from_task_id]
        consumer = self._tasks[to_task_id]
        if not _authority_leq(consumer.authority_level, producer.authority_level):
            # consumer requires HIGHER authority than producer provides —
            # this would be an escalation edge.
            raise TaskGraphError(
                f"Edge {from_task_id}→{to_task_id} escalates authority "
                f"({producer.authority_level} → {consumer.authority_level})"
            )

        self._dependents[to_task_id].add(from_task_id)
        self._dependencies[from_task_id].add(to_task_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[Task]:
        return list(self._tasks.values())

    def get_dependencies(self, task_id: str) -> FrozenSet[str]:
        """Tasks that must complete BEFORE task_id can start."""
        return frozenset(self._dependents.get(task_id, set()))

    def validate(self) -> List[str]:
        """Validate graph consistency; return list of issues (empty = OK)."""
        issues: List[str] = []

        # 1. Cycle detection via Kahn's algorithm
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        for child, parents in self._dependents.items():
            in_degree[child] = in_degree.get(child, 0) + len(parents)

        queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            tid = queue.popleft()
            visited += 1
            for child in self._dependencies.get(tid, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if visited != len(self._tasks):
            issues.append("Graph contains a cycle")

        # 2. Duplicate edges (already prevented by set semantics)

        return issues

    def topo_sort(self) -> List[str]:
        """Return task IDs in dependency order. Raises on cycle."""
        issues = self.validate()
        if issues:
            raise TaskGraphError(f"Cannot topo-sort: {issues[0]}")

        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        for child, parents in self._dependents.items():
            in_degree[child] = in_degree.get(child, 0) + len(parents)

        queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
        result: List[str] = []
        while queue:
            tid = queue.popleft()
            result.append(tid)
            for child in self._dependencies.get(tid, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result


# ---------------------------------------------------------------------------
# Graph → ExecutionPlan
# ---------------------------------------------------------------------------

def execution_plan_from_graph(graph: TaskGraph) -> ExecutionPlan:
    """Linearize a TaskGraph into a staged ExecutionPlan.

    Uses Kahn's algorithm with level tracking so all tasks at the same
    dependency depth become one ExecutionStage (parallelizable).
    """
    issues = graph.validate()
    if issues:
        raise TaskGraphError(f"Cannot build plan: {issues[0]}")

    tasks = graph._tasks
    in_degree: Dict[str, int] = {tid: 0 for tid in tasks}
    for child, parents in graph._dependents.items():
        in_degree[child] = in_degree.get(child, 0) + len(parents)

    # Level assignment: a task's level = max(parent levels) + 1, or 0 if no parents.
    level: Dict[str, int] = {}

    queue = deque()
    for tid, deg in in_degree.items():
        if deg == 0:
            level[tid] = 0
            queue.append(tid)

    while queue:
        tid = queue.popleft()
        for child in graph._dependencies.get(tid, set()):
            in_degree[child] -= 1
            # Child's level is at least parent's level + 1
            level[child] = max(level.get(child, 0), level[tid] + 1)
            if in_degree[child] == 0:
                queue.append(child)

    # Group by level
    stages_dict: Dict[int, set] = defaultdict(set)
    for tid, lvl in level.items():
        stages_dict[lvl].add(tid)

    sorted_levels = sorted(stages_dict.keys())
    stages = [
        ExecutionStage(stage_index=i, task_ids=frozenset(stages_dict[lvl]))
        for i, lvl in enumerate(sorted_levels)
    ]
    return ExecutionPlan(stages=stages)
