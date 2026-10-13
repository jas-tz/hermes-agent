"""Read-only JARVIS dashboard API, mounted at ``/api/plugins/jarvis/``.

Discovered as a bundled backend-only dashboard plugin (``manifest.json`` sets
``tab.hidden`` and declares no UI bundle). Routes inherit the web server's
session-token auth like every other ``/api/`` route.

Every route here is a GET and strictly read-only:

* ``GET /status``     — whether JARVIS memory is the configured provider, a
  whitelisted summary of ``rag_memory.health_check()``, and continuity
  record counts. Never returns memory content.
* ``GET /continuity`` — active/blocked goals, open/blocked tasks and current
  decisions from ``continuity.db``, opened with SQLite ``mode=ro``.
* ``GET /search``     — ``rag_memory.search_memory()`` results.
* ``GET /learnings``  — ``rag_memory.retrieve_cross_session_learnings()``
  results.
* ``GET /pending``    — ``write_approval.list_pending()`` summaries for the
  ``memory`` and ``continuity`` approval queues (never the staged payload).

Search and learnings run only when ``/status`` would report the RAG backend
``available``, so the local store is never opened. Results carry an explicit
field allowlist: score, redacted (and capped) text, the record ``type``, and
allowlisted provenance scalars with path- or URL-like values dropped.

Deliberately not used: ``continuity_db.connect()`` (it creates the schema,
runs migrations and switches the journal mode, i.e. writes), the JARVIS
provider module (it imports approval machinery), every rag_memory write or
delete function, and every write_approval staging/discard/gate function.
When the provider is not ``jarvis`` the RAG store, continuity.db and the
pending queues are never touched. When availability cannot be established
safely the state is ``unavailable`` — there is no fallback store.

Errors are returned as generic messages; queries, record contents, secrets
and exception details are never logged or echoed.
"""

from __future__ import annotations

import importlib
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

JARVIS_PROVIDER = "jarvis"

CONTINUITY_VIEWS = ("list", "goals", "tasks", "decisions")

# Upper bound on records returned per kind; counts always reflect the full set.
CONTINUITY_MAX_ITEMS = 50

# Same "visible" sets as the provider's continuity snapshot
# (plugins/memory/jarvis/__init__.py:_fetch_active_continuity_candidates).
_VISIBLE_GOAL_STATUSES = ("active", "blocked")
_VISIBLE_TASK_STATUSES = ("pending", "in_progress", "blocked")

_GOAL_FIELDS = ("id", "title", "description", "status", "priority", "project_id",
                "created_at", "updated_at", "completed_at")
_TASK_FIELDS = ("id", "title", "description", "status", "priority", "goal_id",
                "project_id", "deadline_at", "created_at", "updated_at", "completed_at")
_DECISION_FIELDS = ("id", "decision", "rationale", "goal_id", "project_id",
                    "superseded_by", "created_at")

# Same default and clamping semantics as the RAG tools / continuity list
# limits; hard cap of 20 results per request.
SEARCH_DEFAULT_LIMIT = 5
SEARCH_MAX_LIMIT = 20
RESULT_TEXT_MAX_CHARS = 2000

# Only these provenance keys are ever returned for a search/learning result.
_RESULT_PROVENANCE_FIELDS = ("source", "source_type", "source_id", "provenance_status",
                             "origin", "transfer_status", "created_at", "stored_at")
_PROVENANCE_VALUE_MAX_CHARS = 200

PENDING_SUBSYSTEMS = ("memory", "continuity")
PENDING_MAX_ITEMS = 50
PENDING_SUMMARY_MAX_CHARS = 500

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def _jarvis_enabled() -> bool:
    try:
        from hermes_cli.config import load_config

        memory = load_config().get("memory") or {}
    except Exception:
        return False
    if not isinstance(memory, dict):
        return False
    return str(memory.get("provider") or "").strip() == JARVIS_PROVIDER


def _disabled() -> Dict[str, Any]:
    return {"enabled": False, "reason": "provider_not_jarvis"}


# ---------------------------------------------------------------------------
# Output sanitising
# ---------------------------------------------------------------------------

def _redact(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    from tools.secret_redaction import redact_secrets

    return redact_secrets(value)


def _safe_provenance(provenance: Any) -> Optional[Dict[str, Any]]:
    """Scalar provenance fields only, strings redacted; nested values dropped."""
    if not isinstance(provenance, dict):
        return None
    safe: Dict[str, Any] = {}
    for key, value in provenance.items():
        if not isinstance(key, str):
            continue
        if value is None or isinstance(value, (bool, int, float)):
            safe[key] = value
        elif isinstance(value, str):
            safe[key] = _redact(value)
    return safe


def _record(obj: Any, fields: tuple) -> Dict[str, Any]:
    data = obj.to_dict()
    out = {field: _redact(data.get(field)) for field in fields}
    out["provenance"] = _safe_provenance(data.get("provenance"))
    return out


def _capped_text(value: Any, max_chars: int):
    """Redact first, then cap; returns ``(text, truncated)``."""
    if not isinstance(value, str):
        return "", False
    text = _redact(value)
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def _identifier(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and _IDENTIFIER_RE.match(value) else None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _looks_like_path_or_url(value: str) -> bool:
    stripped = value.strip()
    return (
        stripped.startswith(("/", "~", "\\", "file:"))
        or "://" in stripped
        or bool(_WINDOWS_PATH_RE.match(stripped))
    )


def _result_provenance(provenance: Any) -> Optional[Dict[str, Any]]:
    """Allowlisted provenance scalars only; path/URL-like strings dropped."""
    if not isinstance(provenance, dict):
        return None
    safe: Dict[str, Any] = {}
    for key in _RESULT_PROVENANCE_FIELDS:
        value = provenance.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            safe[key] = value
        elif isinstance(value, str) and not _looks_like_path_or_url(value):
            safe[key] = _redact(value)[:_PROVENANCE_VALUE_MAX_CHARS]
    return safe


def _search_result(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    text, truncated = _capped_text(row.get("text"), RESULT_TEXT_MAX_CHARS)
    return {
        "score": _number(row.get("score")),
        "text": text,
        "text_truncated": truncated,
        "type": _identifier(payload.get("type")),
        "provenance": _result_provenance(payload.get("provenance")),
    }


def _pending_item(record: Dict[str, Any]) -> Dict[str, Any]:
    """Proposal summary only — the staged ``payload`` is never returned."""
    summary, truncated = _capped_text(record.get("summary"), PENDING_SUMMARY_MAX_CHARS)
    return {
        "id": _identifier(record.get("id")),
        "action": _identifier(record.get("action")),
        "origin": _identifier(record.get("origin")),
        "created_at": _number(record.get("created_at")),
        "summary": summary,
        "summary_truncated": truncated,
    }


# ---------------------------------------------------------------------------
# RAG health
# ---------------------------------------------------------------------------

def _rag_unavailable(reason: str, **extra: Any) -> Dict[str, Any]:
    return {"state": "unavailable", "reason": reason, "mode": None,
            "collection_exists": None, "server_required": None, **extra}


def _load_rag_memory(hermes_home: Path):
    """Import rag_memory from ``$HERMES_HOME/rag`` the way the provider does.

    Returns None when the engine is not installed there.
    """
    rag_dir = hermes_home / "rag"
    if not (rag_dir / "rag_memory.py").is_file():
        return None
    rag_dir_str = str(rag_dir)
    if rag_dir_str not in sys.path:
        sys.path.insert(0, rag_dir_str)
    return importlib.import_module("rag_memory")


def _rag_status(hermes_home: Path) -> Dict[str, Any]:
    """Whitelisted summary of rag_memory.health_check().

    The health check's ``target`` (URL/path) and ``error`` text are never
    passed through. Local mode is reported unavailable: confirming it would
    mean opening the store, which health_check() deliberately does not do.
    """
    try:
        rag_memory = _load_rag_memory(hermes_home)
    except Exception:
        logger.debug("jarvis dashboard: rag engine import failed")
        return _rag_unavailable("engine_unavailable")
    if rag_memory is None:
        return _rag_unavailable("engine_not_installed")

    try:
        health = rag_memory.health_check()
    except Exception:
        logger.debug("jarvis dashboard: rag health check failed")
        return _rag_unavailable("health_check_failed")
    if not isinstance(health, dict):
        return _rag_unavailable("health_check_failed")

    mode = health.get("mode")
    server_required = health.get("server_required")
    server_required = server_required if isinstance(server_required, bool) else None
    if mode == "server":
        reachable = health.get("reachable") is True
        collection_exists = health.get("collection_exists")
        collection_exists = collection_exists if isinstance(collection_exists, bool) else None
        if not reachable:
            reason = "server_unreachable"
        elif collection_exists is not True:
            reason = "collection_missing"
        elif health.get("ok") is True:
            return {"state": "available", "reason": None, "mode": "server",
                    "collection_exists": True, "server_required": server_required}
        else:
            reason = "health_check_failed"
        return {"state": "unavailable", "reason": reason, "mode": "server",
                "collection_exists": collection_exists, "server_required": server_required}
    if mode == "local":
        return {"state": "unavailable", "reason": "local_mode_unverified", "mode": "local",
                "collection_exists": None, "server_required": server_required}
    return _rag_unavailable("config_error", server_required=server_required)


# ---------------------------------------------------------------------------
# Continuity (read-only)
# ---------------------------------------------------------------------------

def _read_continuity(db_path: Path):
    """Visible goals/tasks/decisions, or empty lists when the DB is absent.

    Opens the database read-only, so a missing schema or column raises
    instead of being created; callers map that to ``unavailable``.
    """
    if not db_path.is_file():
        return [], [], []
    from hermes_cli.continuity_db import list_decisions, list_goals, list_tasks

    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        goals = [g for g in list_goals(conn) if g.status in _VISIBLE_GOAL_STATUSES]
        tasks = [t for t in list_tasks(conn) if t.status in _VISIBLE_TASK_STATUSES]
        decisions = list_decisions(conn, include_superseded=False)
    finally:
        conn.close()
    return goals, tasks, decisions


def _continuity_counts(hermes_home: Path) -> Dict[str, Any]:
    try:
        goals, tasks, decisions = _read_continuity(hermes_home / "continuity.db")
    except Exception:
        logger.debug("jarvis dashboard: continuity read failed")
        return {"state": "unavailable", "goals": None, "tasks": None, "decisions": None}
    return {"state": "available", "goals": len(goals), "tasks": len(tasks),
            "decisions": len(decisions)}


def _query_max_chars() -> int:
    # Shared memory-retrieval query cap (plugins/memory/query_rewrite.py).
    from plugins.memory.query_rewrite import _MAX_QUERY_CHARS

    return _MAX_QUERY_CHARS


def _validated_query(q: str) -> str:
    """Reject, never truncate, like the provider's ``_validate_length``.

    The query itself is never echoed back or logged.
    """
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    max_chars = _query_max_chars()
    if len(query) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Query exceeds maximum length of {max_chars} characters",
        )
    return query


def _clamped_limit(raw_limit: Optional[str]) -> int:
    """Missing/invalid/non-positive → default; above the cap → the cap."""
    try:
        limit = int(raw_limit) if raw_limit not in (None, "") else SEARCH_DEFAULT_LIMIT
    except (TypeError, ValueError):
        limit = SEARCH_DEFAULT_LIMIT
    if limit <= 0:
        limit = SEARCH_DEFAULT_LIMIT
    return min(limit, SEARCH_MAX_LIMIT)


def _rag_query(q: str, raw_limit: Optional[str], run, unavailable_detail: str) -> Dict[str, Any]:
    query = _validated_query(q)
    limit = _clamped_limit(raw_limit)
    if not _jarvis_enabled():
        return _disabled()
    hermes_home = _hermes_home()
    if _rag_status(hermes_home)["state"] != "available":
        raise HTTPException(status_code=503, detail=unavailable_detail)
    try:
        rows = run(_load_rag_memory(hermes_home), query, limit)
    except Exception:
        logger.debug("jarvis dashboard: rag read failed")
        raise HTTPException(status_code=503, detail=unavailable_detail)
    if not isinstance(rows, list):
        raise HTTPException(status_code=503, detail=unavailable_detail)
    results = [_search_result(row) for row in rows[:limit] if isinstance(row, dict)]
    return {"enabled": True, "count": len(results), "results": results}


def _section(records: List[Any], fields: tuple) -> Dict[str, Any]:
    return {
        "total": len(records),
        "items": [_record(r, fields) for r in records[:CONTINUITY_MAX_ITEMS]],
        "truncated": len(records) > CONTINUITY_MAX_ITEMS,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/status")
def get_status() -> Dict[str, Any]:
    if not _jarvis_enabled():
        return _disabled()
    hermes_home = _hermes_home()
    return {
        "enabled": True,
        "rag": _rag_status(hermes_home),
        "continuity": _continuity_counts(hermes_home),
    }


@router.get("/continuity")
def get_continuity(view: str = "list") -> Dict[str, Any]:
    if view not in CONTINUITY_VIEWS:
        raise HTTPException(status_code=400, detail="Invalid view")
    if not _jarvis_enabled():
        return _disabled()
    try:
        goals, tasks, decisions = _read_continuity(_hermes_home() / "continuity.db")
    except Exception:
        logger.debug("jarvis dashboard: continuity read failed")
        raise HTTPException(status_code=503, detail="Continuity data is unavailable")

    result: Dict[str, Any] = {"enabled": True, "view": view}
    if view in ("list", "goals"):
        result["goals"] = _section(goals, _GOAL_FIELDS)
    if view in ("list", "tasks"):
        result["tasks"] = _section(tasks, _TASK_FIELDS)
    if view in ("list", "decisions"):
        result["decisions"] = _section(decisions, _DECISION_FIELDS)
    return result


@router.get("/search")
def get_search(q: str = "", limit: Optional[str] = None) -> Dict[str, Any]:
    return _rag_query(
        q, limit,
        lambda rag_memory, query, n: rag_memory.search_memory(query, limit=n),
        "Memory search is unavailable",
    )


@router.get("/learnings")
def get_learnings(q: str = "", limit: Optional[str] = None) -> Dict[str, Any]:
    return _rag_query(
        q, limit,
        lambda rag_memory, query, n: rag_memory.retrieve_cross_session_learnings(query, limit=n),
        "Learning history is unavailable",
    )


@router.get("/pending")
def get_pending(subsystem: str = "") -> Dict[str, Any]:
    if subsystem not in PENDING_SUBSYSTEMS:
        raise HTTPException(status_code=400, detail="Invalid subsystem")
    if not _jarvis_enabled():
        return _disabled()
    try:
        from tools import write_approval

        queue = write_approval.MEMORY if subsystem == "memory" else write_approval.CONTINUITY
        records = write_approval.list_pending(queue)
    except Exception:
        logger.debug("jarvis dashboard: pending read failed")
        raise HTTPException(status_code=503, detail="Pending proposals are unavailable")
    items = [_pending_item(r) for r in records if isinstance(r, dict)]
    return {
        "enabled": True,
        "subsystem": subsystem,
        "total": len(items),
        "items": items[:PENDING_MAX_ITEMS],
        "truncated": len(items) > PENDING_MAX_ITEMS,
    }
