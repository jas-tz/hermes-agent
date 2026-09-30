"""JARVIS Tool Capability Registry — Phase 25A.

A formal capability registry for tools that may eventually be exposed to JARVIS
sub-agents. This registry is SEPARATE from the existing Hermes toolsets.py and
the JARVIS RAG/continuity tools.

Each tool capability defines:
- tool name
- capability/category
- authority level (L0-L3)
- read/write/destructive classification
- allowed task types
- whether approval is required
- whether the tool is parent-only
- whether the tool may be exposed to sub-agents
- provenance requirements
- sandbox/isolation requirements

Policy resolution:
    task_type → capability policy → permitted tool set

IMPORTANT: This registry does NOT connect new production tools. It defines the
policy framework for future tool exposure. Existing JARVIS RAG/continuity
tools continue to use their own registry in subagent_types.py.

Phase 25A: formal capability registry with policy resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Authority levels (compatible with existing JARVIS authority model)
# ---------------------------------------------------------------------------

AUTHORITY_LEVELS = frozenset({"L0", "L1", "L2", "L3"})

# ---------------------------------------------------------------------------
# Tool capability dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCapability:
    """A tool's capability definition for the JARVIS sub-agent registry.

    This is metadata about what a tool does and how it may be accessed.
    It does NOT implement the tool itself — that remains in the Hermes
    tool registry and the JARVIS RAG/continuity handlers.
    """
    name: str
    category: str
    authority_level: str  # "L0", "L1", "L2", "L3"
    classification: str  # "read", "write", "destructive"
    allowed_task_types: FrozenSet[str]
    requires_approval: bool = False
    parent_only: bool = False
    exposed_to_subagents: bool = False
    provenance_required: bool = True
    sandbox_required: bool = False
    description: str = ""

    def __post_init__(self):
        if self.authority_level not in AUTHORITY_LEVELS:
            raise ValueError(f"Invalid authority level: {self.authority_level}")
        if self.classification not in ("read", "write", "destructive"):
            raise ValueError(f"Invalid classification: {self.classification}")
        # Destructive tools must require approval
        if self.classification == "destructive" and not self.requires_approval:
            raise ValueError(f"Destructive tool '{self.name}' must require approval")
        # Parent-only tools cannot be exposed to sub-agents
        if self.parent_only and self.exposed_to_subagents:
            raise ValueError(f"Parent-only tool '{self.name}' cannot be exposed to sub-agents")


# ---------------------------------------------------------------------------
# Tool capability registry
# ---------------------------------------------------------------------------

TOOL_CAPABILITY_REGISTRY: Dict[str, ToolCapability] = {}

# --- RAG tools (existing JARVIS tools, registered for capability tracking) ---

def _register_capability(capability: ToolCapability) -> None:
    """Register a tool capability in the registry."""
    if capability.name in TOOL_CAPABILITY_REGISTRY:
        raise ValueError(f"Tool capability '{capability.name}' already registered")
    TOOL_CAPABILITY_REGISTRY[capability.name] = capability


# RAG read tools (L0)
_register_capability(ToolCapability(
    name="hermes_rag_search",
    category="rag",
    authority_level="L0",
    classification="read",
    allowed_task_types=frozenset({"research", "summarize", "note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=False,
    description="Search RAG semantic memory",
))

_register_capability(ToolCapability(
    name="hermes_rag_retrieve_learnings",
    category="rag",
    authority_level="L0",
    classification="read",
    allowed_task_types=frozenset({"research", "summarize", "note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=False,
    description="Retrieve cross-session learnings",
))

_register_capability(ToolCapability(
    name="hermes_rag_extract_learnings",
    category="rag",
    authority_level="L0",
    classification="read",
    allowed_task_types=frozenset({"research", "summarize", "note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=False,
    description="Extract learnings from session",
))

# RAG write tools (L1)
_register_capability(ToolCapability(
    name="hermes_rag_store",
    category="rag",
    authority_level="L1",
    classification="write",
    allowed_task_types=frozenset({"note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=True,
    description="Store memory with provenance",
))

_register_capability(ToolCapability(
    name="hermes_rag_store_learnings",
    category="rag",
    authority_level="L1",
    classification="write",
    allowed_task_types=frozenset({"note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=True,
    description="Store session learnings",
))

# RAG destructive tools (L3)
_register_capability(ToolCapability(
    name="hermes_rag_delete",
    category="rag",
    authority_level="L3",
    classification="destructive",
    allowed_task_types=frozenset({"maintenance", "default"}),
    requires_approval=True,
    exposed_to_subagents=True,
    provenance_required=True,
    description="Delete RAG memory by ID",
))

_register_capability(ToolCapability(
    name="hermes_rag_rollback_learnings",
    category="rag",
    authority_level="L3",
    classification="destructive",
    allowed_task_types=frozenset({"maintenance", "default"}),
    requires_approval=True,
    exposed_to_subagents=True,
    provenance_required=True,
    description="Bulk rollback of session learnings",
))

# --- Continuity tools (existing JARVIS tools) ---

# Continuity read tools (L0)
_register_capability(ToolCapability(
    name="hermes_goal_list",
    category="continuity",
    authority_level="L0",
    classification="read",
    allowed_task_types=frozenset({"research", "summarize", "note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=False,
    description="List continuity goals",
))

_register_capability(ToolCapability(
    name="hermes_task_list",
    category="continuity",
    authority_level="L0",
    classification="read",
    allowed_task_types=frozenset({"research", "summarize", "note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=False,
    description="List continuity tasks",
))

_register_capability(ToolCapability(
    name="hermes_decision_list",
    category="continuity",
    authority_level="L0",
    classification="read",
    allowed_task_types=frozenset({"research", "summarize", "note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=False,
    description="List continuity decisions",
))

# Continuity reversible tools (L1)
_register_capability(ToolCapability(
    name="hermes_goal_complete",
    category="continuity",
    authority_level="L1",
    classification="write",
    allowed_task_types=frozenset({"note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=True,
    description="Mark goal as complete",
))

_register_capability(ToolCapability(
    name="hermes_task_complete",
    category="continuity",
    authority_level="L1",
    classification="write",
    allowed_task_types=frozenset({"note_taking", "task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=True,
    description="Mark task as complete",
))

# Continuity gated tools (L2)
_register_capability(ToolCapability(
    name="hermes_goal_create",
    category="continuity",
    authority_level="L2",
    classification="write",
    allowed_task_types=frozenset({"task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=True,
    description="Create new goal",
))

_register_capability(ToolCapability(
    name="hermes_goal_update",
    category="continuity",
    authority_level="L2",
    classification="write",
    allowed_task_types=frozenset({"task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=True,
    description="Update existing goal",
))

_register_capability(ToolCapability(
    name="hermes_task_create",
    category="continuity",
    authority_level="L2",
    classification="write",
    allowed_task_types=frozenset({"task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=True,
    description="Create new task",
))

_register_capability(ToolCapability(
    name="hermes_task_update",
    category="continuity",
    authority_level="L2",
    classification="write",
    allowed_task_types=frozenset({"task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=True,
    description="Update existing task",
))

_register_capability(ToolCapability(
    name="hermes_decision_record",
    category="continuity",
    authority_level="L2",
    classification="write",
    allowed_task_types=frozenset({"task_execution", "maintenance", "default"}),
    exposed_to_subagents=True,
    provenance_required=True,
    description="Record a new decision",
))

# Continuity destructive tools (L3)
_register_capability(ToolCapability(
    name="hermes_decision_supersede",
    category="continuity",
    authority_level="L3",
    classification="destructive",
    allowed_task_types=frozenset({"maintenance", "default"}),
    requires_approval=True,
    exposed_to_subagents=True,
    provenance_required=True,
    description="Supersede an existing decision",
))

# --- Parent-only tools (NOT exposed to sub-agents) ---

_register_capability(ToolCapability(
    name="delegate_task",
    category="delegation",
    authority_level="L3",
    classification="write",
    allowed_task_types=frozenset(),
    parent_only=True,
    exposed_to_subagents=False,
    provenance_required=True,
    description="Spawn child agents (parent-only)",
))

_register_capability(ToolCapability(
    name="terminal",
    category="system",
    authority_level="L3",
    classification="write",
    allowed_task_types=frozenset(),
    parent_only=True,
    exposed_to_subagents=False,
    sandbox_required=True,
    description="Execute shell commands (parent-only)",
))

_register_capability(ToolCapability(
    name="write_file",
    category="filesystem",
    authority_level="L3",
    classification="write",
    allowed_task_types=frozenset(),
    parent_only=True,
    exposed_to_subagents=False,
    sandbox_required=True,
    description="Write files (parent-only)",
))

_register_capability(ToolCapability(
    name="read_file",
    category="filesystem",
    authority_level="L3",
    classification="read",
    allowed_task_types=frozenset(),
    parent_only=True,
    exposed_to_subagents=False,
    sandbox_required=True,
    description="Read files (parent-only)",
))

_register_capability(ToolCapability(
    name="browser_navigate",
    category="browser",
    authority_level="L3",
    classification="read",
    allowed_task_types=frozenset(),
    parent_only=True,
    exposed_to_subagents=False,
    sandbox_required=True,
    description="Browser navigation (parent-only)",
))

_register_capability(ToolCapability(
    name="computer_use",
    category="desktop",
    authority_level="L3",
    classification="write",
    allowed_task_types=frozenset(),
    parent_only=True,
    exposed_to_subagents=False,
    sandbox_required=True,
    description="Desktop computer control (parent-only)",
))

_register_capability(ToolCapability(
    name="cronjob",
    category="scheduling",
    authority_level="L3",
    classification="write",
    allowed_task_types=frozenset(),
    parent_only=True,
    exposed_to_subagents=False,
    description="Schedule cron jobs (parent-only)",
))

_register_capability(ToolCapability(
    name="send_message",
    category="messaging",
    authority_level="L3",
    classification="write",
    allowed_task_types=frozenset(),
    parent_only=True,
    exposed_to_subagents=False,
    description="Send messages (parent-only)",
))


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------

def get_capability(tool_name: str) -> Optional[ToolCapability]:
    """Return the capability definition for a tool, or None if not registered."""
    return TOOL_CAPABILITY_REGISTRY.get(tool_name)


def get_capabilities_for_task_type(task_type: str) -> List[ToolCapability]:
    """Return all capabilities allowed for a task type."""
    return [
        cap for cap in TOOL_CAPABILITY_REGISTRY.values()
        if task_type in cap.allowed_task_types and cap.exposed_to_subagents
    ]


def get_permitted_tools_for_task_type(task_type: str) -> FrozenSet[str]:
    """Return the set of tool names permitted for a task type.

    Policy resolution:
        task_type → capability policy → permitted tool set

    Only tools that are:
    1. Explicitly allowlisted for the task type
    2. Marked as exposed_to_subagents
    3. Not parent-only

    are included in the permitted set.
    """
    return frozenset(
        cap.name
        for cap in TOOL_CAPABILITY_REGISTRY.values()
        if task_type in cap.allowed_task_types
        and cap.exposed_to_subagents
        and not cap.parent_only
    )


def is_tool_permitted(tool_name: str, task_type: str) -> bool:
    """Check if a tool is permitted for a task type."""
    cap = TOOL_CAPABILITY_REGISTRY.get(tool_name)
    if cap is None:
        return False
    if cap.parent_only:
        return False
    if not cap.exposed_to_subagents:
        return False
    return task_type in cap.allowed_task_types


def is_tool_parent_only(tool_name: str) -> bool:
    """Check if a tool is parent-only."""
    cap = TOOL_CAPABILITY_REGISTRY.get(tool_name)
    if cap is None:
        return False
    return cap.parent_only


def requires_approval(tool_name: str) -> bool:
    """Check if a tool requires approval."""
    cap = TOOL_CAPABILITY_REGISTRY.get(tool_name)
    if cap is None:
        return False
    return cap.requires_approval


def get_tools_by_authority_level(level: str) -> List[ToolCapability]:
    """Return all capabilities at a given authority level."""
    return [cap for cap in TOOL_CAPABILITY_REGISTRY.values() if cap.authority_level == level]


def get_tools_by_category(category: str) -> List[ToolCapability]:
    """Return all capabilities in a category."""
    return [cap for cap in TOOL_CAPABILITY_REGISTRY.values() if cap.category == category]


def validate_capability_policy() -> List[str]:
    """Validate the capability registry for policy consistency.

    Returns a list of warnings/errors.
    """
    issues = []

    for name, cap in TOOL_CAPABILITY_REGISTRY.items():
        # Destructive tools must require approval
        if cap.classification == "destructive" and not cap.requires_approval:
            issues.append(f"ERROR: Destructive tool '{name}' does not require approval")

        # Parent-only tools cannot be exposed to sub-agents
        if cap.parent_only and cap.exposed_to_subagents:
            issues.append(f"ERROR: Parent-only tool '{name}' is exposed to sub-agents")

        # Exposed tools must have at least one allowed task type
        if cap.exposed_to_subagents and not cap.allowed_task_types:
            issues.append(f"WARNING: Exposed tool '{name}' has no allowed task types")

        # Parent-only tools should not have allowed task types
        if cap.parent_only and cap.allowed_task_types:
            issues.append(f"WARNING: Parent-only tool '{name}' has allowed task types")

    return issues
