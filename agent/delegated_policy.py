"""Delegated-agent read-only policy (JARVIS Decision 42).

A delegated (``delegate_task``) child may read approved information and return
results to its parent, but must not modify persistent state: RAG memory,
MEMORY.md / USER.md, continuity records, skills, capability-plan writes, or
files -- directly or through terminal, process or code-execution tools -- and
it never runs background reviews or session-end learning.

``tools.delegate_tool._build_child_agent`` marks every child with
``_delegated_read_only = True`` immediately after constructing it, before any
optional setup that could fail. Code that holds an agent checks
``is_delegated_read_only(agent)``; code without one (module-level dispatch,
tool handlers) falls back to the delegated-child ContextVar, which is set while
a child runs. Prohibited operations are refused before any approval handling,
so no approval mode can authorize them. The policy applies equally in ordinary
and no-learning sessions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DELEGATED_READ_ONLY_CODE = "delegated_read_only"

# Toolsets a delegated child never receives (removed whole from its tool
# surface): shell and code execution, desktop control and media generation can
# write files or act outside the task in ways no path check can bound. Only
# toolsets made up entirely of denied tools belong here -- disabling a toolset
# subtracts everything it resolves to, so a mixed one (debugging includes
# file/web, browser lists web_search) would also strip tools a child keeps.
DELEGATED_DENIED_TOOLSETS = (
    "terminal",
    "code_execution",
    "computer_use",
    "image_gen",
    "tts",
    "bfl",
    "video_gen",
)

# Internal toolset (toolsets.py) holding the denied tools that share a toolset
# with tools a child keeps (read_file, skills_list, ha_get_state, web_search,
# ...), so they can be removed precisely: file writes, skill management, Home
# Assistant service calls and browser automation.
DELEGATED_DENY_TOOLSET = "delegated_read_only_deny"

# Tools a delegated child never receives and that are refused at dispatch.
# Static so the refusal never depends on a registry lookup; tools of the
# denied toolsets are also resolved at runtime (plugin toolsets such as bfl).
DELEGATED_DENIED_TOOLS = frozenset({
    "terminal",
    "process",
    "execute_code",
    "write_file",
    "patch",
    "skill_manage",
    "computer_use",
    "image_generate",
    "text_to_speech",
    "video_generate",
    "xai_video_edit",
    "xai_video_extend",
    "ha_call_service",
})
_DENIED_TOOL_PREFIXES = ("browser_", "bfl_")


def is_delegated_read_only(agent: Any = None) -> bool:
    """True when *agent* (or, failing that, the running code) is a delegated child.

    Only an explicit ``True`` marker counts, so a mock's auto-attribute or a
    stray truthy value never decides the question; the ContextVar covers code
    paths that have no agent object.
    """
    if agent is not None and getattr(agent, "_delegated_read_only", None) is True:
        return True
    try:
        from agent.delegation_context import is_delegated_child_context
        return is_delegated_child_context()
    except Exception:
        # The ContextVar module is part of core; failing to import it means the
        # restriction cannot be evaluated -- treat the caller as delegated.
        logger.warning("Delegated-child context unavailable; treating caller as delegated")
        return True


def delegated_refusal(operation: str, *, outcome: str = "Nothing was changed.",
                      session_id: Optional[str] = None) -> str:
    """Structured JSON refusal for an operation a delegated agent may not perform.

    Logs the refusal (operation and session only, never the payload) and
    returns ``{"ok": false, "success": false, "code": "delegated_read_only", ...}``
    so both tool-result conventions read it as a failure.
    """
    logger.info(
        "Delegated read-only: %s refused for session %s (Decision 42)",
        operation, session_id or "-",
    )
    return json.dumps({
        "ok": False,
        "success": False,
        "code": DELEGATED_READ_ONLY_CODE,
        "operation": operation,
        "error": (
            "Delegated agents are read-only for persistent state (Decision 42); "
            f"'{operation}' is not permitted. {outcome} "
            "Return the result to the parent agent instead."
        ),
    })


def _is_mcp_toolset(name: Optional[str]) -> bool:
    """True for an MCP toolset (``mcp-<server>``) or an alias of one."""
    if not name:
        return False
    if str(name).startswith("mcp-"):
        return True
    from tools.registry import registry

    target = registry.get_toolset_alias_target(str(name))
    return bool(target and str(target).startswith("mcp-"))


def delegated_tool_denial_reason(tool_name: str) -> Optional[str]:
    """Why a delegated child may not call *tool_name*, or None if it may.

    Raises if the tool's toolset cannot be looked up; callers treat that as a
    denial (fail closed).
    """
    if tool_name in DELEGATED_DENIED_TOOLS or tool_name.startswith(_DENIED_TOOL_PREFIXES):
        return "denied tool"
    from tools.registry import registry

    toolset = registry.get_toolset_for_tool(tool_name)
    if toolset in DELEGATED_DENIED_TOOLSETS:
        return f"denied toolset '{toolset}'"
    if _is_mcp_toolset(toolset):
        return f"MCP toolset '{toolset}'"
    return None


def delegated_tool_refusal(agent: Any, tool_name: str) -> Optional[str]:
    """The refusal message for a delegated agent calling a denied tool, else None.

    Checked before any approval, plugin hook or execution. Non-delegated
    agents are never affected. If the decision cannot be made for a delegated
    agent, the call is refused.
    """
    if not is_delegated_read_only(agent):
        return None
    try:
        reason = delegated_tool_denial_reason(str(tool_name))
    except Exception as exc:
        reason = f"tool policy could not be evaluated ({type(exc).__name__})"
    if reason is None:
        return None
    logger.info("Delegated read-only: tool %s refused (%s) (Decision 42)", tool_name, reason)
    return (
        f"{DELEGATED_READ_ONLY_CODE}: delegated agents may not call '{tool_name}' "
        f"({reason}; Decision 42). Nothing was executed. Return your findings to the parent agent."
    )
