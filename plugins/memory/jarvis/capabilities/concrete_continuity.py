"""Concrete JARVIS Continuity capabilities — Phase 37.

Each class wraps an existing JARVIS continuity tool call through the
AUTHORIZED HermesRAGProvider.handle_tool_call path. No new privileges.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from plugins.memory.jarvis.capabilities.base import (
    Capability,
    CapabilityCategory,
    CapabilityContext,
    CapabilityInputType,
    CapabilityOutputType,
    CapabilityRequirements,
    CapabilityResult,
)

logger = logging.getLogger(__name__)


def _dispatch_continuity(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke an authorized continuity tool call.

    Uses the same pattern as apply_continuity_pending: construct a
    throwaway HermesRAGProvider, initialize it, and call handle_tool_call.
    This routes through _handle_continuity_tool_call and the write_approval
    CONTINUITY gate — never bypasses authorization.

    Decision 42: the throwaway provider runs as a primary session, so a
    delegated child's continuity write is refused here, before it is built.
    """
    # Local import to avoid circular import at module load
    from plugins.memory.jarvis import HermesRAGProvider, JARVISSubagentProvider
    if tool_name not in JARVISSubagentProvider._SUBAGENT_READ_TOOLS:
        from agent.delegated_policy import delegated_refusal, is_delegated_read_only
        if is_delegated_read_only():
            return json.loads(delegated_refusal(tool_name))
    provider = HermesRAGProvider()
    # The active Hermes home (profile override, then HERMES_HOME, then the
    # platform default ~/.hermes) -- not a hard-coded ~/.hermes, which ignored
    # profiles and let sandboxed callers reach the real continuity.db.
    from hermes_constants import get_hermes_home
    hermes_home = str(get_hermes_home())
    provider.initialize("capability-dispatch", hermes_home=hermes_home, agent_context="primary")
    raw = provider.handle_tool_call(tool_name, args)
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {"ok": False, "error": f"Malformed response: {raw!r}"}


class GoalListCapability(Capability):
    """Wraps hermes_goal_list — list continuity goals (L0 read)."""

    def get_name(self) -> str:
        return "goal_list"

    def get_description(self) -> str:
        return "List active continuity goals"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_goal_list"}),
            input_type=CapabilityInputType.QUERY,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=True,
            max_execution_time_seconds=30,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {}
        if "status" in context.parameters:
            args["status"] = context.parameters["status"]
        if "project_id" in context.parameters:
            args["project_id"] = context.parameters["project_id"]
        if "limit" in context.parameters:
            args["limit"] = context.parameters["limit"]
        result = _dispatch_continuity("hermes_goal_list", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown goal list error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class GoalCreateCapability(Capability):
    """Wraps hermes_goal_create — create a new goal (L2 gated)."""

    def get_name(self) -> str:
        return "goal_create"

    def get_description(self) -> str:
        return "Create a new continuity goal"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L2",
            required_tools=frozenset({"hermes_goal_create"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,  # write_approval gate handles staging
            parallelizable=True,
            max_execution_time_seconds=60,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        title = context.parameters.get("title", "")
        if not title or not title.strip():
            return False, "title parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {"title": context.parameters.get("title", "")}
        if "description" in context.parameters:
            args["description"] = context.parameters["description"]
        if "priority" in context.parameters:
            args["priority"] = context.parameters["priority"]
        if "project_id" in context.parameters:
            args["project_id"] = context.parameters["project_id"]
        if "provenance" in context.parameters:
            args["provenance"] = context.parameters["provenance"]
        result = _dispatch_continuity("hermes_goal_create", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown goal create error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class TaskListCapability(Capability):
    """Wraps hermes_task_list — list continuity tasks (L0 read)."""

    def get_name(self) -> str:
        return "task_list"

    def get_description(self) -> str:
        return "List open continuity tasks"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_task_list"}),
            input_type=CapabilityInputType.QUERY,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=True,
            max_execution_time_seconds=30,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {}
        if "status" in context.parameters:
            args["status"] = context.parameters["status"]
        if "goal_id" in context.parameters:
            args["goal_id"] = context.parameters["goal_id"]
        if "project_id" in context.parameters:
            args["project_id"] = context.parameters["project_id"]
        if "limit" in context.parameters:
            args["limit"] = context.parameters["limit"]
        result = _dispatch_continuity("hermes_task_list", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown task list error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class TaskCreateCapability(Capability):
    """Wraps hermes_task_create — create a new task (L2 gated)."""

    def get_name(self) -> str:
        return "task_create"

    def get_description(self) -> str:
        return "Create a new continuity task"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L2",
            required_tools=frozenset({"hermes_task_create"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=True,
            max_execution_time_seconds=60,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        title = context.parameters.get("title", "")
        if not title or not title.strip():
            return False, "title parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {"title": context.parameters.get("title", "")}
        if "description" in context.parameters:
            args["description"] = context.parameters["description"]
        if "priority" in context.parameters:
            args["priority"] = context.parameters["priority"]
        if "goal_id" in context.parameters:
            args["goal_id"] = context.parameters["goal_id"]
        if "project_id" in context.parameters:
            args["project_id"] = context.parameters["project_id"]
        if "deadline_at" in context.parameters:
            args["deadline_at"] = context.parameters["deadline_at"]
        if "provenance" in context.parameters:
            args["provenance"] = context.parameters["provenance"]
        result = _dispatch_continuity("hermes_task_create", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown task create error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class DecisionListCapability(Capability):
    """Wraps hermes_decision_list — list decisions (L0 read)."""

    def get_name(self) -> str:
        return "decision_list"

    def get_description(self) -> str:
        return "List current decisions"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_decision_list"}),
            input_type=CapabilityInputType.QUERY,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=True,
            max_execution_time_seconds=30,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {}
        if "goal_id" in context.parameters:
            args["goal_id"] = context.parameters["goal_id"]
        if "project_id" in context.parameters:
            args["project_id"] = context.parameters["project_id"]
        if "limit" in context.parameters:
            args["limit"] = context.parameters["limit"]
        result = _dispatch_continuity("hermes_decision_list", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown decision list error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class DecisionRecordCapability(Capability):
    """Wraps hermes_decision_record — record a new decision (L2 gated)."""

    def get_name(self) -> str:
        return "decision_record"

    def get_description(self) -> str:
        return "Record a new decision"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L2",
            required_tools=frozenset({"hermes_decision_record"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=True,
            max_execution_time_seconds=60,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        decision = context.parameters.get("decision", "")
        if not decision or not decision.strip():
            return False, "decision parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {"decision": context.parameters.get("decision", "")}
        if "rationale" in context.parameters:
            args["rationale"] = context.parameters["rationale"]
        if "goal_id" in context.parameters:
            args["goal_id"] = context.parameters["goal_id"]
        if "project_id" in context.parameters:
            args["project_id"] = context.parameters["project_id"]
        if "provenance" in context.parameters:
            args["provenance"] = context.parameters["provenance"]
        result = _dispatch_continuity("hermes_decision_record", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown decision record error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class GoalCompleteCapability(Capability):
    """Wraps hermes_goal_complete — mark a goal done (L1 reversible)."""

    def get_name(self) -> str:
        return "goal_complete"

    def get_description(self) -> str:
        return "Mark a goal as complete"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.ORCHESTRATION

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L1",
            required_tools=frozenset({"hermes_goal_complete"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=False,
            max_execution_time_seconds=30,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        goal_id = context.parameters.get("goal_id", "")
        if not goal_id or not goal_id.strip():
            return False, "goal_id parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {"goal_id": context.parameters.get("goal_id", "")}
        result = _dispatch_continuity("hermes_goal_complete", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown goal complete error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class TaskCompleteCapability(Capability):
    """Wraps hermes_task_complete — mark a task done (L1 reversible)."""

    def get_name(self) -> str:
        return "task_complete"

    def get_description(self) -> str:
        return "Mark a task as complete"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.ORCHESTRATION

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L1",
            required_tools=frozenset({"hermes_task_complete"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=False,
            max_execution_time_seconds=30,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        task_id = context.parameters.get("task_id", "")
        if not task_id or not task_id.strip():
            return False, "task_id parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {"task_id": context.parameters.get("task_id", "")}
        result = _dispatch_continuity("hermes_task_complete", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown task complete error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


# ---------------------------------------------------------------------------
# Phase 39: Mutable Continuity Capabilities
# ---------------------------------------------------------------------------


class GoalUpdateCapability(Capability):
    """Wraps hermes_goal_update — update an existing goal (L2 gated).

    Per the existing handler, only title/description changes trigger the
    write_approval CONTINUITY gate. Pure status/priority/project_id
    transitions on a record the user already referenced execute directly.
    """

    def get_name(self) -> str:
        return "goal_update"

    def get_description(self) -> str:
        return "Update an existing continuity goal"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L2",
            required_tools=frozenset({"hermes_goal_update"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,  # write_approval gate handles content changes
            parallelizable=True,
            max_execution_time_seconds=60,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        goal_id = context.parameters.get("goal_id", "")
        if not goal_id or not goal_id.strip():
            return False, "goal_id parameter is required"
        # Must have at least one field to update
        updatable = ("title", "description", "status", "priority", "project_id")
        if not any(f in context.parameters for f in updatable):
            return False, "At least one updatable field (title/description/status/priority/project_id) is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args: Dict[str, Any] = {"goal_id": context.parameters["goal_id"]}
        for field in ("title", "description", "status", "priority", "project_id"):
            if field in context.parameters:
                args[field] = context.parameters[field]
        if "provenance" in context.parameters:
            args["provenance"] = context.parameters["provenance"]
        result = _dispatch_continuity("hermes_goal_update", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown goal update error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class TaskUpdateCapability(Capability):
    """Wraps hermes_task_update — update an existing task (L2 gated).

    Per the existing handler, title/description/deadline/goal/project/priority
    changes trigger the write_approval CONTINUITY gate. A pure status
    transition executes directly.
    """

    def get_name(self) -> str:
        return "task_update"

    def get_description(self) -> str:
        return "Update an existing continuity task"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L2",
            required_tools=frozenset({"hermes_task_update"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,  # write_approval gate handles content changes
            parallelizable=True,
            max_execution_time_seconds=60,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        task_id = context.parameters.get("task_id", "")
        if not task_id or not task_id.strip():
            return False, "task_id parameter is required"
        updatable = ("title", "description", "status", "priority",
                     "goal_id", "project_id", "deadline_at")
        if not any(f in context.parameters for f in updatable):
            return False, "At least one updatable field is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args: Dict[str, Any] = {"task_id": context.parameters["task_id"]}
        for field in ("title", "description", "status", "priority",
                      "goal_id", "project_id", "deadline_at"):
            if field in context.parameters:
                args[field] = context.parameters[field]
        if "provenance" in context.parameters:
            args["provenance"] = context.parameters["provenance"]
        result = _dispatch_continuity("hermes_task_update", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown task update error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class DecisionSupersedeCapability(Capability):
    """Wraps hermes_decision_supersede — supersede an existing decision (L3 destructive)."""

    def get_name(self) -> str:
        return "decision_supersede"

    def get_description(self) -> str:
        return "Supersede an existing decision"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L3",
            required_tools=frozenset({"hermes_decision_supersede"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=True,
            parallelizable=False,
            max_execution_time_seconds=60,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        decision_id = context.parameters.get("decision_id", "")
        if not decision_id or not decision_id.strip():
            return False, "decision_id parameter is required"
        decision = context.parameters.get("decision", "")
        if not decision or not decision.strip():
            return False, "decision parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args: Dict[str, Any] = {
            "decision_id": context.parameters["decision_id"],
            "decision": context.parameters["decision"],
        }
        if "rationale" in context.parameters:
            args["rationale"] = context.parameters["rationale"]
        if "goal_id" in context.parameters:
            args["goal_id"] = context.parameters["goal_id"]
        if "project_id" in context.parameters:
            args["project_id"] = context.parameters["project_id"]
        if "provenance" in context.parameters:
            args["provenance"] = context.parameters["provenance"]
        result = _dispatch_continuity("hermes_decision_supersede", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown decision supersede error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )
