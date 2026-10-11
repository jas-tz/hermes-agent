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

# Tools a delegated child never receives and that are refused at dispatch:
# each can write persistent state (or arbitrary files) in a way no path check
# can bound.
DELEGATED_DENIED_TOOLS = frozenset({
    "terminal",
    "process",
    "execute_code",
    "write_file",
    "patch",
    "skill_manage",
})


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
