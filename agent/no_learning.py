"""No-learning control for persistent memory (JARVIS Memory Safety, Priority 2).

A session is excluded from persistent learning -- automatic session-end
learning, the explicit RAG store tools, background memory review, built-in
memory tool writes (MEMORY.md / USER.md), continuity writes (goals, tasks,
decisions) and RAG delete/rollback -- when:

* its session ID starts with ``NO_LEARNING_PREFIX`` ("nolearn_"). Clients that
  choose their own session IDs (the desktop app, API callers) use this for test
  or opted-out conversations. No Hermes ID generator produces this prefix.
* the process runs with ``HERMES_NO_LEARNING`` set to a true value. This covers
  CLI runs and test harnesses whose session IDs Hermes generates.

Unset, empty or false values leave learning on. An unrecognised value turns
learning OFF -- someone set the variable deliberately -- and logs a warning
once per distinct value. Callers log the skip themselves, once per decision.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

NO_LEARNING_PREFIX = "nolearn_"
NO_LEARNING_ENV = "HERMES_NO_LEARNING"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
_warned_values: set = set()


def env_disables_learning() -> bool:
    """True when ``HERMES_NO_LEARNING`` disables learning for this process."""
    raw = os.environ.get(NO_LEARNING_ENV)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _FALSE_VALUES:
        return False
    if value in _TRUE_VALUES:
        return True
    if raw not in _warned_values:
        _warned_values.add(raw)
        logger.warning(
            "%s has unrecognised value %r; treating persistent learning as disabled",
            NO_LEARNING_ENV, raw,
        )
    return True


def learning_disabled_reason(session_id: Optional[str] = None) -> Optional[str]:
    """Why persistent learning is disabled for *session_id*, or None if it is allowed."""
    if isinstance(session_id, str) and session_id.startswith(NO_LEARNING_PREFIX):
        return f"session id starts with '{NO_LEARNING_PREFIX}'"
    if env_disables_learning():
        return f"{NO_LEARNING_ENV} is set"
    return None


def agent_learning_disabled_reason(agent: Any) -> Optional[str]:
    """Why persistent learning is disabled for *agent*, or None if it is allowed.

    Honours a decision inherited from a parent agent (``_no_learning_reason``,
    set on delegated children), then the agent's own session id and the env
    switch. The background-review fork shares its parent's session id, so it
    resolves to the same decision.
    """
    inherited = getattr(agent, "_no_learning_reason", None)
    if inherited:
        return inherited
    return learning_disabled_reason(getattr(agent, "session_id", None))


def learning_disabled_refusal(operation: str, reason: str, *,
                              session_id: Optional[str] = None,
                              outcome: str = "Nothing was changed.") -> str:
    """Structured JSON refusal for a persistent write blocked by no-learning.

    Logs the refusal (operation, session, reason -- never the payload) and
    returns ``{"ok": false, "success": false, "code": "learning_disabled", ...}``
    so both tool-result conventions read it as a failure.
    """
    logger.info(
        "No-learning: %s refused for session %s: persistent learning disabled (%s)",
        operation, session_id or "-", reason,
    )
    return json.dumps({
        "ok": False,
        "success": False,
        "code": "learning_disabled",
        "operation": operation,
        "error": f"Persistent learning is disabled for this session ({reason}). {outcome}",
    })
