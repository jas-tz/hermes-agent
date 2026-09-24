"""Per-profile structured continuity store: Goals, Tasks, Decisions.

Inert storage foundation for the Jarvis continuity layer (Phase 9A). This
module is intentionally NOT wired into the Jarvis RAG provider, the Hermes
memory-provider ABC, or any tool schema yet — it is a standalone, directly
testable SQLite store, following the exact pattern already established by
``hermes_cli/projects_db.py`` (per-profile file, WAL, additive migrations,
``write_txn`` for writes) and reusing its shared primitives from
``hermes_cli/sqlite_util.py`` rather than inventing new ones.

Scope: **per-profile**, stored at ``$HERMES_HOME/continuity.db`` (resolved via
``get_hermes_home()``), mirroring projects.db / sessions / config / cron.

Three tables, deliberately kept separate from Jarvis's semantic Qdrant store:

- **goals** — durable, cross-session objectives ("what is being worked
  toward"). NOT the same thing as ``hermes_cli/goals.py``'s ``/goal`` Ralph
  loop, which is a live, single-session, turn-budgeted execution-control
  mechanism keyed by ``session_id`` — that module is untouched by this one.
- **tasks** — actionable work items, optionally linked to a goal.
- **decisions** — records of what was decided and why. Decisions are NOT
  workflow items: they have no status machine. A decision is corrected by
  *superseding* it (a new decision row, with the old row's
  ``superseded_by`` pointed at it) — the old decision is preserved, never
  overwritten or deleted, matching the "do not treat decisions as
  workflow items" requirement.

Cross-database relationship note: ``project_id`` logically refers to a row
in ``hermes_cli/projects_db.py``'s ``projects`` table, which lives in a
*separate* SQLite file (``projects.db``). SQLite cannot enforce a real
foreign key across two different database files/connections, so — exactly
mirroring ``hermes_cli/kanban_db.py``'s own ``tasks.project_id`` column
(same cross-file situation, same precedent) — ``project_id`` is stored as
a plain ``TEXT`` column with no SQL-level ``REFERENCES`` clause. Callers
are responsible for passing a real project id; this module does not
validate it against ``projects.db``. ``goal_id`` and ``superseded_by``,
by contrast, point at rows *within this same database* (``goals`` /
``decisions``), so those ARE real, enforced foreign keys
(``PRAGMA foreign_keys=ON``, default ``NO ACTION`` — deliberately NOT
``ON DELETE CASCADE``, so a referenced goal/decision can never be silently
cascade-deleted out from under a task/decision that points at it; this
module does not expose a hard-delete for goals/tasks/decisions at all in
this phase — see the module docstring's "no delete" note below).

No hard-delete API is exposed for goals, tasks, or decisions in this
phase: lifecycle is soft (status transitions / archive-restore / decision
supersession) per the "do not hard-delete normal completed history as
part of ordinary lifecycle operations" requirement. Nothing in this module
imports, opens, or otherwise depends on Jarvis's Qdrant store
(``~/.hermes/rag/rag_memory.py``) — the two are intentionally
unconnected at this phase.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.sqlite_util import add_column_if_missing as _add_column_if_missing, write_txn
from hermes_constants import get_hermes_home
from tools.secret_redaction import redact_secrets

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def continuity_db_path() -> Path:
    """The per-profile continuity DB path (``$HERMES_HOME/continuity.db``).

    Profile-aware: ``get_hermes_home()`` already points at the active
    profile's home. Tests pass an explicit ``db_path`` to :func:`connect`
    so production ``~/.hermes/continuity.db`` is never opened or created
    by the test suite.
    """
    return get_hermes_home() / "continuity.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS goals (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT,
    status        TEXT NOT NULL DEFAULT 'active',
    priority      INTEGER NOT NULL DEFAULT 0,
    project_id    TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    completed_at  INTEGER,
    provenance    TEXT
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_project_id ON goals(project_id);

CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    priority      INTEGER NOT NULL DEFAULT 0,
    goal_id       TEXT REFERENCES goals(id),
    project_id    TEXT,
    deadline_at   INTEGER,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    completed_at  INTEGER,
    provenance    TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_goal_id ON tasks(goal_id);
CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id);

CREATE TABLE IF NOT EXISTS decisions (
    id             TEXT PRIMARY KEY,
    decision       TEXT NOT NULL,
    rationale      TEXT,
    goal_id        TEXT REFERENCES goals(id),
    project_id     TEXT,
    superseded_by  TEXT REFERENCES decisions(id),
    created_at     INTEGER NOT NULL,
    provenance     TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_goal_id ON decisions(goal_id);
CREATE INDEX IF NOT EXISTS idx_decisions_project_id ON decisions(project_id);
"""

VALID_GOAL_STATUSES = {"active", "paused", "blocked", "done", "abandoned"}
VALID_TASK_STATUSES = {"pending", "in_progress", "blocked", "done", "cancelled"}

# Columns considered "optional" for additive-migration purposes: present in
# SCHEMA_SQL for a fresh database, and re-applied idempotently to a legacy
# database on every open via _add_column_if_missing (mirrors
# projects_db.py's _OPTIONAL_PROJECT_COLUMNS / _migrate_add_optional_columns
# exactly). provenance is the column most likely to have been absent from an
# earlier revision of a per-profile store in this codebase's own history
# (see projects_db.py, whose board_slug/primary_path/icon/color all arrived
# after v1), so it anchors this table's migration path from day one.
_OPTIONAL_COLUMNS = {
    "goals": ("provenance",),
    "tasks": ("provenance",),
    "decisions": ("provenance",),
}


# ---------------------------------------------------------------------------
# id + timestamp helpers
# ---------------------------------------------------------------------------


def _new_goal_id() -> str:
    return "g_" + secrets.token_hex(4)


def _new_task_id() -> str:
    return "t_" + secrets.token_hex(4)


def _new_decision_id() -> str:
    return "d_" + secrets.token_hex(4)


def _now() -> int:
    """Server-generated timestamp (epoch seconds). Never caller-supplied."""
    return int(time.time())


# ---------------------------------------------------------------------------
# Provenance discipline (mirrors rag_memory.py::_sanitize_provenance's
# contract without importing rag_memory — this module has zero dependency
# on Jarvis RAG / Qdrant, by design).
# ---------------------------------------------------------------------------

# Same whitelist rag_memory.py uses for Jarvis RAG provenance, kept
# consistent across the continuity layer and the semantic layer. The two
# frozensets are defined independently in each file (no cross-import) —
# see rag_memory.py's own PROVENANCE_SOURCES for its copy.
#
# Phase 10A additions (continuity-intelligence provenance, see the Phase 10
# audit):
#   user_confirmed  — a durable record explicitly approved by the user.
#                      This is the expected source for a Goal/Task/Decision
#                      created after the user confirmed a proposal.
#   model_proposed   — TRANSIENT ONLY. Marks an in-flight proposal the model
#                      has not yet gotten user confirmation for. A record
#                      with this source must never be written to
#                      continuity.db as a durable row — no code path in
#                      this module or its callers does so today (the
#                      proposal mechanism itself is not implemented yet).
#                      The sanitizer accepts it as syntactically valid
#                      provenance so a future proposal-tracking layer has
#                      somewhere to put it; nothing currently persists it.
#   imported         — reserved for a future external/import workflow.
#                      Unused by any current write path.
#   inferred         — pre-existing value. Valid for RAG/session-learning
#                      use as before; must NOT be used as the provenance
#                      for a newly created Goal or Decision in later
#                      phases (a Goal/Decision should always trace back to
#                      an explicit statement, not an inference).
PROVENANCE_SOURCES = frozenset({
    "conversation",
    "tool_result",
    "web",
    "manual",
    "inferred",
    "unknown",
    "user_confirmed",
    "model_proposed",
    "imported",
})

# Matches rag_memory.py's MAX_PROVENANCE_SIZE — a deliberately small cap so
# one oversized/hostile provenance blob can't bloat a row without bound.
MAX_PROVENANCE_SIZE = 1024


def _sanitize_provenance(provenance: Optional[Dict[str, Any]]) -> str:
    """Sanitize caller-supplied provenance into a safe JSON string.

    - Validates ``source`` against a fixed whitelist (unknown/missing ->
      "unknown").
    - Redacts secrets from every string value (top-level and one level of
      nested dict), so a credential accidentally passed as provenance
      ("session token", "api key used", etc.) never reaches disk.
    - Silently drops values of unsupported types (lists, nested objects
      beyond one level) rather than raising — a caller-supplied metadata
      blob is untrusted shape, not a hard contract.
    - Enforces a size cap: an oversized blob is truncated down to just
      ``source`` + a "truncated" status marker rather than rejected
      outright, so a call never fails purely because provenance was too
      chatty.
    - Never trusts a caller-supplied timestamp; the row's own
      ``created_at``/``updated_at`` columns are the source of truth for
      when a record was written, not anything inside ``provenance``.

    Returns a JSON string (never raises on bad input).
    """
    if not provenance or not isinstance(provenance, dict):
        return json.dumps({"source": "unknown", "provenance_status": "unavailable"})

    sanitized: Dict[str, Any] = {}

    source = provenance.get("source", "unknown")
    if source not in PROVENANCE_SOURCES:
        source = "unknown"
    sanitized["source"] = source

    for key, value in provenance.items():
        if key == "source":
            continue
        if isinstance(value, str):
            sanitized[key] = redact_secrets(value)
        elif isinstance(value, (int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = {
                k: (redact_secrets(v) if isinstance(v, str) else v)
                for k, v in value.items()
            }
        # Other types (lists, etc.) are silently dropped.

    sanitized["provenance_status"] = "tracked" if source != "unknown" else "unavailable"

    encoded = json.dumps(sanitized, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_PROVENANCE_SIZE:
        sanitized = {"source": source, "provenance_status": "truncated"}
        encoded = json.dumps(sanitized, ensure_ascii=False)

    return encoded


def _parse_provenance(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a stored provenance JSON string back to a dict.

    Never raises: a corrupt/legacy value reads back as None rather than
    breaking row retrieval.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_INITIALIZED_PATHS: set[str] = set()


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the per-profile continuity DB.

    WAL with DELETE fallback for network filesystems (shared helper from
    ``hermes_state``, same one ``projects_db.py`` uses). Schema init is
    idempotent (``CREATE TABLE IF NOT EXISTS`` + additive migrations) and
    cached per-path per-process.
    """
    path = db_path if db_path is not None else continuity_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="continuity.db")
        conn.execute("PRAGMA foreign_keys=ON")
        if resolved not in _INITIALIZED_PATHS:
            conn.executescript(SCHEMA_SQL)
            _migrate_add_optional_columns(conn)
            _INITIALIZED_PATHS.add(resolved)
    except Exception:
        conn.close()
        raise
    return conn


@contextlib.contextmanager
def connect_closing(db_path: Optional[Path] = None):
    """Open a continuity DB connection and guarantee it is closed on exit.

    sqlite3's connection context manager only commits/rollbacks; it does
    NOT close the file descriptor. Mirrors
    ``projects_db.connect_closing`` / ``kanban_db.connect_closing``.
    """
    conn = connect(db_path=db_path)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _migrate_add_optional_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after v1 to legacy DBs (safe on every open).

    A fresh database created from SCHEMA_SQL already has every column in
    ``_OPTIONAL_COLUMNS``, so this is a no-op there; it only does real
    work when opening a database created by an earlier schema revision
    that predates one of these columns.
    """
    for table, columns in _OPTIONAL_COLUMNS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col in columns:
            if col not in existing:
                _add_column_if_missing(conn, table, col, f"{col} TEXT")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    id: str
    title: str
    created_at: int
    updated_at: int
    description: Optional[str] = None
    status: str = "active"
    priority: int = 0
    project_id: Optional[str] = None
    completed_at: Optional[int] = None
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "provenance": self.provenance,
        }


@dataclass
class Task:
    id: str
    title: str
    created_at: int
    updated_at: int
    description: Optional[str] = None
    status: str = "pending"
    priority: int = 0
    goal_id: Optional[str] = None
    project_id: Optional[str] = None
    deadline_at: Optional[int] = None
    completed_at: Optional[int] = None
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "goal_id": self.goal_id,
            "project_id": self.project_id,
            "deadline_at": self.deadline_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "provenance": self.provenance,
        }


@dataclass
class Decision:
    id: str
    decision: str
    created_at: int
    rationale: Optional[str] = None
    goal_id: Optional[str] = None
    project_id: Optional[str] = None
    superseded_by: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "decision": self.decision,
            "rationale": self.rationale,
            "goal_id": self.goal_id,
            "project_id": self.project_id,
            "superseded_by": self.superseded_by,
            "created_at": self.created_at,
            "provenance": self.provenance,
        }


def _goal_from_row(row: sqlite3.Row) -> Goal:
    return Goal(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        description=row["description"],
        status=row["status"],
        priority=row["priority"],
        project_id=row["project_id"],
        completed_at=row["completed_at"],
        provenance=_parse_provenance(row["provenance"]),
    )


def _task_from_row(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        description=row["description"],
        status=row["status"],
        priority=row["priority"],
        goal_id=row["goal_id"],
        project_id=row["project_id"],
        deadline_at=row["deadline_at"],
        completed_at=row["completed_at"],
        provenance=_parse_provenance(row["provenance"]),
    )


def _decision_from_row(row: sqlite3.Row) -> Decision:
    return Decision(
        id=row["id"],
        decision=row["decision"],
        created_at=row["created_at"],
        rationale=row["rationale"],
        goal_id=row["goal_id"],
        project_id=row["project_id"],
        superseded_by=row["superseded_by"],
        provenance=_parse_provenance(row["provenance"]),
    )


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


def create_goal(
    conn: sqlite3.Connection,
    *,
    title: str,
    description: Optional[str] = None,
    status: str = "active",
    priority: int = 0,
    project_id: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a goal and return its id."""
    title = str(title or "").strip()
    if not title:
        raise ValueError("goal title must not be empty")
    if status not in VALID_GOAL_STATUSES:
        raise ValueError(
            f"invalid goal status {status!r}: must be one of {sorted(VALID_GOAL_STATUSES)}"
        )

    gid = _new_goal_id()
    now = _now()
    completed_at = now if status == "done" else None

    with write_txn(conn):
        conn.execute(
            "INSERT INTO goals "
            "(id, title, description, status, priority, project_id, "
            " created_at, updated_at, completed_at, provenance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                gid,
                title,
                description,
                status,
                priority,
                project_id,
                now,
                now,
                completed_at,
                _sanitize_provenance(provenance),
            ),
        )
    return gid


def get_goal(conn: sqlite3.Connection, goal_id: str) -> Optional[Goal]:
    row = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
    return _goal_from_row(row) if row is not None else None


def list_goals(
    conn: sqlite3.Connection,
    *,
    status: Optional[str] = None,
    project_id: Optional[str] = None,
) -> List[Goal]:
    sql = "SELECT * FROM goals"
    clauses: List[str] = []
    params: List[object] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY priority DESC, created_at ASC"
    rows = conn.execute(sql, params).fetchall()
    return [_goal_from_row(r) for r in rows]


def update_goal(
    conn: sqlite3.Connection,
    goal_id: str,
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[int] = None,
    project_id: Optional[str] = None,
) -> bool:
    """Patch top-level goal fields. Only provided (non-None) fields change.

    A status transition into "done" stamps ``completed_at``; a transition
    away from "done" clears it back to NULL — completion time is always
    derived from the status change, never caller-supplied.
    """
    sets: List[str] = []
    params: List[object] = []
    if title is not None:
        t = str(title).strip()
        if not t:
            raise ValueError("goal title must not be empty")
        sets.append("title = ?")
        params.append(t)
    if description is not None:
        sets.append("description = ?")
        params.append(description)
    if status is not None:
        if status not in VALID_GOAL_STATUSES:
            raise ValueError(
                f"invalid goal status {status!r}: must be one of {sorted(VALID_GOAL_STATUSES)}"
            )
        sets.append("status = ?")
        params.append(status)
        sets.append("completed_at = ?")
        params.append(_now() if status == "done" else None)
    if priority is not None:
        sets.append("priority = ?")
        params.append(priority)
    if project_id is not None:
        sets.append("project_id = ?")
        params.append(project_id or None)
    if not sets:
        return False
    sets.append("updated_at = ?")
    params.append(_now())
    params.append(goal_id)
    with write_txn(conn):
        cur = conn.execute(f"UPDATE goals SET {', '.join(sets)} WHERE id = ?", params)
    return cur.rowcount > 0


def archive_goal(conn: sqlite3.Connection, goal_id: str) -> bool:
    """Soft-archive a goal (status -> "abandoned"). Never deletes the row."""
    return update_goal(conn, goal_id, status="abandoned")


def restore_goal(conn: sqlite3.Connection, goal_id: str) -> bool:
    """Restore an archived/completed goal back to "active"."""
    return update_goal(conn, goal_id, status="active")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

# Sentinel distinguishing "caller did not mention deadline_at" (leave
# unchanged) from "caller explicitly passed None" (clear it) in update_task.
# create_task has no such ambiguity: omitting deadline_at there already
# means NULL, which is the "never infer a deadline" default.
_UNSET = object()


def create_task(
    conn: sqlite3.Connection,
    *,
    title: str,
    description: Optional[str] = None,
    status: str = "pending",
    priority: int = 0,
    goal_id: Optional[str] = None,
    project_id: Optional[str] = None,
    deadline_at: Optional[int] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a task and return its id.

    ``deadline_at`` defaults to ``None`` (NULL) and is stored exactly as
    given — it is never inferred or defaulted to "now" or any computed
    value. Pass an explicit epoch-seconds int only when a deadline is
    actually known.
    """
    title = str(title or "").strip()
    if not title:
        raise ValueError("task title must not be empty")
    if status not in VALID_TASK_STATUSES:
        raise ValueError(
            f"invalid task status {status!r}: must be one of {sorted(VALID_TASK_STATUSES)}"
        )

    tid = _new_task_id()
    now = _now()
    completed_at = now if status == "done" else None

    with write_txn(conn):
        conn.execute(
            "INSERT INTO tasks "
            "(id, title, description, status, priority, goal_id, project_id, "
            " deadline_at, created_at, updated_at, completed_at, provenance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tid,
                title,
                description,
                status,
                priority,
                goal_id,
                project_id,
                deadline_at,
                now,
                now,
                completed_at,
                _sanitize_provenance(provenance),
            ),
        )
    return tid


def get_task(conn: sqlite3.Connection, task_id: str) -> Optional[Task]:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _task_from_row(row) if row is not None else None


def list_tasks(
    conn: sqlite3.Connection,
    *,
    status: Optional[str] = None,
    goal_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> List[Task]:
    sql = "SELECT * FROM tasks"
    clauses: List[str] = []
    params: List[object] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if goal_id is not None:
        clauses.append("goal_id = ?")
        params.append(goal_id)
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY priority DESC, created_at ASC"
    rows = conn.execute(sql, params).fetchall()
    return [_task_from_row(r) for r in rows]


def update_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[int] = None,
    goal_id: Optional[str] = None,
    project_id: Optional[str] = None,
    deadline_at: Any = _UNSET,
) -> bool:
    """Patch top-level task fields. Only provided fields change.

    ``deadline_at`` uses a sentinel default: omit it entirely to leave the
    stored deadline untouched, pass ``None`` to explicitly clear it, or
    pass an epoch-seconds int to set it. This is the only way to
    distinguish "don't touch the deadline" from "clear the deadline" —
    both are legitimate callers, and neither may ever *infer* a value.

    A status transition into "done" stamps ``completed_at``; a transition
    away from "done" clears it back to NULL.
    """
    sets: List[str] = []
    params: List[object] = []
    if title is not None:
        t = str(title).strip()
        if not t:
            raise ValueError("task title must not be empty")
        sets.append("title = ?")
        params.append(t)
    if description is not None:
        sets.append("description = ?")
        params.append(description)
    if status is not None:
        if status not in VALID_TASK_STATUSES:
            raise ValueError(
                f"invalid task status {status!r}: must be one of {sorted(VALID_TASK_STATUSES)}"
            )
        sets.append("status = ?")
        params.append(status)
        sets.append("completed_at = ?")
        params.append(_now() if status == "done" else None)
    if priority is not None:
        sets.append("priority = ?")
        params.append(priority)
    if goal_id is not None:
        sets.append("goal_id = ?")
        params.append(goal_id or None)
    if project_id is not None:
        sets.append("project_id = ?")
        params.append(project_id or None)
    if deadline_at is not _UNSET:
        sets.append("deadline_at = ?")
        params.append(deadline_at)
    if not sets:
        return False
    sets.append("updated_at = ?")
    params.append(_now())
    params.append(task_id)
    with write_txn(conn):
        cur = conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", params)
    return cur.rowcount > 0


def complete_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Mark a task done (status="done", completed_at stamped). Soft — no delete."""
    return update_task(conn, task_id, status="done")


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def create_decision(
    conn: sqlite3.Connection,
    *,
    decision: str,
    rationale: Optional[str] = None,
    goal_id: Optional[str] = None,
    project_id: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> str:
    """Record a decision and return its id.

    Decisions have no status field — they are a record of what was
    decided, not a workflow item. Use :func:`supersede_decision` to
    correct one; the original row is preserved.
    """
    decision = str(decision or "").strip()
    if not decision:
        raise ValueError("decision text must not be empty")

    did = _new_decision_id()
    now = _now()

    with write_txn(conn):
        conn.execute(
            "INSERT INTO decisions "
            "(id, decision, rationale, goal_id, project_id, superseded_by, "
            " created_at, provenance) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                did,
                decision,
                rationale,
                goal_id,
                project_id,
                now,
                _sanitize_provenance(provenance),
            ),
        )
    return did


def get_decision(conn: sqlite3.Connection, decision_id: str) -> Optional[Decision]:
    row = conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
    return _decision_from_row(row) if row is not None else None


def list_decisions(
    conn: sqlite3.Connection,
    *,
    goal_id: Optional[str] = None,
    project_id: Optional[str] = None,
    include_superseded: bool = False,
) -> List[Decision]:
    sql = "SELECT * FROM decisions"
    clauses: List[str] = []
    params: List[object] = []
    if not include_superseded:
        clauses.append("superseded_by IS NULL")
    if goal_id is not None:
        clauses.append("goal_id = ?")
        params.append(goal_id)
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC"
    rows = conn.execute(sql, params).fetchall()
    return [_decision_from_row(r) for r in rows]


def supersede_decision(
    conn: sqlite3.Connection,
    old_decision_id: str,
    *,
    decision: str,
    rationale: Optional[str] = None,
    goal_id: Optional[str] = None,
    project_id: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> str:
    """Correct a decision by recording a new one that supersedes it.

    The old row is preserved unchanged except for ``superseded_by``, which
    is pointed at the new row's id — history is never overwritten or
    deleted. ``goal_id``/``project_id`` inherit from the old decision when
    not explicitly given, so a correction stays linked to the same
    goal/project by default.

    Raises ``ValueError`` if ``old_decision_id`` does not exist.
    """
    decision = str(decision or "").strip()
    if not decision:
        raise ValueError("decision text must not be empty")

    with write_txn(conn):
        old_row = conn.execute(
            "SELECT goal_id, project_id FROM decisions WHERE id = ?",
            (old_decision_id,),
        ).fetchone()
        if old_row is None:
            raise ValueError(f"no such decision: {old_decision_id}")

        new_id = _new_decision_id()
        now = _now()
        conn.execute(
            "INSERT INTO decisions "
            "(id, decision, rationale, goal_id, project_id, superseded_by, "
            " created_at, provenance) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                new_id,
                decision,
                rationale,
                goal_id if goal_id is not None else old_row["goal_id"],
                project_id if project_id is not None else old_row["project_id"],
                now,
                _sanitize_provenance(provenance),
            ),
        )
        conn.execute(
            "UPDATE decisions SET superseded_by = ? WHERE id = ?",
            (new_id, old_decision_id),
        )
    return new_id
