"""JARVIS Sub-Agent Task Types + Execution Context + Lifecycle State Machine.

Phase 22A/23A/23B: Task registry, immutable execution context, and controlled
lifecycle state machine.

The lifecycle state machine enforces:
- Valid transitions between states
- Terminal states cannot transition further
- Child/model cannot modify lifecycle state
- Trusted runtime code (SubagentLifecycleManager) performs all transitions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, FrozenSet, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Authority levels
# ---------------------------------------------------------------------------

L0_READ_ONLY: FrozenSet[str] = frozenset({
    "hermes_rag_search",
    "hermes_rag_retrieve_learnings",
    "hermes_rag_extract_learnings",
    "hermes_goal_list",
    "hermes_task_list",
    "hermes_decision_list",
})

L1_REVERSIBLE: FrozenSet[str] = frozenset({
    "hermes_rag_store",
    "hermes_rag_store_learnings",
    "hermes_goal_complete",
    "hermes_task_complete",
})

L2_GATED: FrozenSet[str] = frozenset({
    "hermes_goal_create",
    "hermes_goal_update",
    "hermes_task_create",
    "hermes_task_update",
    "hermes_decision_record",
})

L3_APPROVAL: FrozenSet[str] = frozenset({
    "hermes_rag_delete",
    "hermes_rag_rollback_learnings",
    "hermes_decision_supersede",
})

ALL_JARVIS_TOOLS = L0_READ_ONLY | L1_REVERSIBLE | L2_GATED | L3_APPROVAL


# ---------------------------------------------------------------------------
# Task type definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubagentTaskType:
    """A specialized sub-agent task type with its permitted tool access."""
    name: str
    description: str
    authority_level: str
    allowed_tools: FrozenSet[str]
    l2_tools: FrozenSet[str] = field(default_factory=frozenset)
    l3_tools: FrozenSet[str] = field(default_factory=frozenset)
    system_prompt_hint: str = ""

    def __post_init__(self):
        if self.authority_level == "L0":
            max_tools = L0_READ_ONLY
        elif self.authority_level == "L1":
            max_tools = L0_READ_ONLY | L1_REVERSIBLE
        elif self.authority_level == "L2":
            max_tools = L0_READ_ONLY | L1_REVERSIBLE | L2_GATED
        elif self.authority_level == "L3":
            max_tools = L0_READ_ONLY | L1_REVERSIBLE | L2_GATED | L3_APPROVAL
        else:
            raise ValueError(f"Invalid authority level: {self.authority_level}")

        if not self.allowed_tools.issubset(max_tools):
            raise ValueError(
                f"Task type '{self.name}' cannot include tools outside its level: "
                f"{self.allowed_tools - max_tools}"
            )
        if not self.l2_tools.issubset(L2_GATED):
            raise ValueError(f"Task type '{self.name}' has non-L2 tools in l2_tools")
        if not self.l3_tools.issubset(L3_APPROVAL):
            raise ValueError(f"Task type '{self.name}' has non-L3 tools in l3_tools")


# ---------------------------------------------------------------------------
# Execution context — Phase 23A (frozen/immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubagentExecutionContext:
    """Immutable execution context for a JARVIS sub-agent.

    All security-critical fields are frozen. The only mutable field is
    lifecycle_state, but it can only be modified through the trusted
    SubagentLifecycleManager which returns a NEW context object.
    """
    child_id: str
    parent_session_id: str
    task_goal: str
    task_type: str
    authority_level: str
    delegation_depth: int
    allowed_tools: FrozenSet[str]
    provenance_context_id: str
    created_at: str
    lifecycle_state: str = "pending"

    @property
    def is_active(self) -> bool:
        return self.lifecycle_state == "running"

    @property
    def is_terminal(self) -> bool:
        return self.lifecycle_state in ("completed", "failed", "cancelled", "timed_out")


# ---------------------------------------------------------------------------
# Lifecycle state machine — Phase 23B
# ---------------------------------------------------------------------------

# All valid lifecycle states
LIFECYCLE_STATES: FrozenSet[str] = frozenset({
    "pending", "initializing", "running", "completed", "failed", "cancelled", "timed_out",
})

# Terminal states cannot transition to any other state
TERMINAL_STATES: FrozenSet[str] = frozenset({
    "completed", "failed", "cancelled", "timed_out",
})

# Valid state transitions: from_state -> set of allowed to_states
VALID_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "pending": frozenset({"initializing", "cancelled"}),
    "initializing": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"completed", "failed", "cancelled", "timed_out"}),
}


def is_valid_transition(from_state: str, to_state: str) -> bool:
    """Return True if a lifecycle transition is valid."""
    if from_state in TERMINAL_STATES:
        return False
    allowed = VALID_TRANSITIONS.get(from_state, frozenset())
    return to_state in allowed


class SubagentLifecycleManager:
    """Trusted runtime lifecycle state machine manager.

    Only this class can transition lifecycle states. The child/model
    cannot access this manager — it only receives the immutable context.
    Transitions produce NEW context objects (copy-on-write).

    Phase 23B: controlled lifecycle state machine.
    """

    def __init__(self, context: SubagentExecutionContext) -> None:
        self._context = context

    @property
    def context(self) -> SubagentExecutionContext:
        """Return the current execution context (read-only reference)."""
        return self._context

    @property
    def state(self) -> str:
        """Return the current lifecycle state."""
        return self._context.lifecycle_state

    @property
    def is_terminal(self) -> bool:
        """Return True if the current state is terminal."""
        return self._context.lifecycle_state in TERMINAL_STATES

    def transition(self, new_state: str) -> SubagentExecutionContext:
        """Transition to a new lifecycle state.

        Validates the transition against the state machine. Returns a NEW
        immutable context object with the updated lifecycle_state.

        Raises ValueError if the transition is invalid.
        """
        current = self._context.lifecycle_state

        # Validate transition
        if not is_valid_transition(current, new_state):
            if current in TERMINAL_STATES:
                raise ValueError(
                    f"Cannot transition from terminal state '{current}' to '{new_state}'"
                )
            raise ValueError(
                f"Invalid lifecycle transition: '{current}' -> '{new_state}'"
            )

        # Create new immutable context with updated state
        new_context = SubagentExecutionContext(
            child_id=self._context.child_id,
            parent_session_id=self._context.parent_session_id,
            task_goal=self._context.task_goal,
            task_type=self._context.task_type,
            authority_level=self._context.authority_level,
            delegation_depth=self._context.delegation_depth,
            allowed_tools=self._context.allowed_tools,
            provenance_context_id=self._context.provenance_context_id,
            created_at=self._context.created_at,
            lifecycle_state=new_state,
        )

        # Update internal reference (trusted runtime only)
        self._context = new_context

        logger.debug(
            "Lifecycle transition: %s -> %s (child=%s, task=%s)",
            current, new_state, self._context.child_id, self._context.task_goal[:40],
        )

        return new_context


# ---------------------------------------------------------------------------
# Task type registry
# ---------------------------------------------------------------------------

SUBAGENT_TASK_TYPES: Dict[str, SubagentTaskType] = {}


def _register_task_type(task_type: SubagentTaskType) -> None:
    if task_type.name in SUBAGENT_TASK_TYPES:
        raise ValueError(f"Task type '{task_type.name}' already registered")
    SUBAGENT_TASK_TYPES[task_type.name] = task_type


# Register all task types
_register_task_type(SubagentTaskType(
    name="research",
    description="Search and synthesize information",
    authority_level="L0",
    allowed_tools=L0_READ_ONLY,
    system_prompt_hint="You are a research sub-agent. Read-only access.",
))

_register_task_type(SubagentTaskType(
    name="summarize",
    description="Summarize findings",
    authority_level="L0",
    allowed_tools=L0_READ_ONLY,
    system_prompt_hint="You are a summarization sub-agent. Read-only access.",
))

_register_task_type(SubagentTaskType(
    name="note_taking",
    description="Record findings as RAG memories",
    authority_level="L1",
    allowed_tools=L0_READ_ONLY | L1_REVERSIBLE,
    system_prompt_hint="You are a note-taking sub-agent. Reversible writes allowed.",
))

_register_task_type(SubagentTaskType(
    name="task_execution",
    description="Execute tasks with gated continuity writes",
    authority_level="L2",
    allowed_tools=L0_READ_ONLY | L1_REVERSIBLE | L2_GATED,
    l2_tools=L2_GATED,
    system_prompt_hint="You are a task execution sub-agent. Gated writes allowed.",
))

_register_task_type(SubagentTaskType(
    name="maintenance",
    description="Perform maintenance including deletions",
    authority_level="L3",
    allowed_tools=L0_READ_ONLY | L1_REVERSIBLE | L2_GATED | L3_APPROVAL,
    l2_tools=L2_GATED,
    l3_tools=L3_APPROVAL,
    system_prompt_hint="You are a maintenance sub-agent. All operations allowed with approval.",
))

_register_task_type(SubagentTaskType(
    name="default",
    description="Default sub-agent with full whitelist (backward-compatible)",
    authority_level="L3",
    allowed_tools=ALL_JARVIS_TOOLS,
    l2_tools=L2_GATED,
    l3_tools=L3_APPROVAL,
    system_prompt_hint="You are a sub-agent with full JARVIS tool access.",
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_task_type(name: str) -> Optional[SubagentTaskType]:
    return SUBAGENT_TASK_TYPES.get(name)


def list_task_types() -> list:
    return list(SUBAGENT_TASK_TYPES.keys())


def resolve_task_type_tools(task_type_name: str | None) -> Optional[SubagentTaskType]:
    if task_type_name is None:
        return None
    return SUBAGENT_TASK_TYPES.get(task_type_name)


def validate_task_type(task_type_name: str | None) -> bool:
    return task_type_name in SUBAGENT_TASK_TYPES if task_type_name else False


def is_tool_allowed_for_task(task_type_name: str | None, tool_name: str) -> bool:
    task_type = SUBAGENT_TASK_TYPES.get(task_type_name) if task_type_name else None
    if task_type is None:
        return False
    all_permitted = task_type.allowed_tools | task_type.l2_tools | task_type.l3_tools
    return tool_name in all_permitted


def create_execution_context(
    child_id: str,
    parent_session_id: str,
    task_goal: str,
    task_type: str,
    delegation_depth: int,
) -> SubagentExecutionContext:
    """Create an immutable execution context from trusted delegation inputs."""
    task_type_def = SUBAGENT_TASK_TYPES.get(task_type)
    if task_type_def is None:
        raise ValueError(f"Unknown task type: {task_type}")

    allowed = task_type_def.allowed_tools | task_type_def.l2_tools | task_type_def.l3_tools
    provenance_id = f"subagent:{task_goal[:40]}"

    return SubagentExecutionContext(
        child_id=child_id,
        parent_session_id=parent_session_id,
        task_goal=task_goal,
        task_type=task_type,
        authority_level=task_type_def.authority_level,
        delegation_depth=delegation_depth,
        allowed_tools=allowed,
        provenance_context_id=provenance_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        lifecycle_state="pending",
    )
