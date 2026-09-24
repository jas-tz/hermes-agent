"""Tests for the per-profile continuity store (hermes_cli/continuity_db).

Phase 9A: this module is a standalone SQLite store, not yet wired into the
Jarvis RAG provider or any Hermes tool. Every test below uses an explicit
tmp_path database via ``connect(db_path=...)`` — never
``continuity_db_path()`` / the default ``$HERMES_HOME`` resolution — so
production ``~/.hermes/continuity.db`` is never opened, created, or
modified by this suite. Nothing here imports, opens, or otherwise touches
Jarvis's Qdrant store (rag_memory.py) or runs Hermes.
"""

from __future__ import annotations

import inspect
import sqlite3

import pytest

from hermes_cli import continuity_db as cdb


@pytest.fixture
def conn(tmp_path):
    c = cdb.connect(db_path=tmp_path / "continuity.db")
    try:
        yield c
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Initialization / schema
# ---------------------------------------------------------------------------


def test_database_initializes_and_creates_all_three_tables(conn):
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {"goals", "tasks", "decisions"} <= tables


def test_wal_mode_is_enabled(conn):
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_foreign_keys_pragma_is_on(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_connect_is_idempotent_on_reopen(tmp_path):
    path = tmp_path / "continuity.db"
    a = cdb.connect(db_path=path)
    try:
        gid = cdb.create_goal(a, title="Persisted goal")
    finally:
        a.close()

    b = cdb.connect(db_path=path)
    try:
        assert cdb.get_goal(b, gid) is not None
    finally:
        b.close()


# ---------------------------------------------------------------------------
# Phase 13D: "blocked" is a Python-level allowed-status addition only — no
# SQL CHECK constraint exists on the status columns, so no schema migration
# is required or expected. These tests prove that directly.
# ---------------------------------------------------------------------------


def test_status_columns_have_no_check_constraint(conn):
    """blocked was accepted purely by extending VALID_GOAL_STATUSES /
    VALID_TASK_STATUSES in Python — confirm there is genuinely no SQL-level
    constraint that a migration would have needed to touch."""
    goals_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='goals'"
    ).fetchone()[0]
    tasks_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='tasks'"
    ).fetchone()[0]
    assert "CHECK" not in goals_sql.upper()
    assert "CHECK" not in tasks_sql.upper()


def test_creating_a_blocked_record_does_not_alter_table_schema(conn):
    """Writing a "blocked" row must not trigger any migration path — the
    goals/tasks table DDL before and after must be byte-identical."""
    before_goals = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='goals'"
    ).fetchone()[0]
    before_tasks = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='tasks'"
    ).fetchone()[0]

    cdb.create_goal(conn, title="Blocked goal", status="blocked")
    cdb.create_task(conn, title="Blocked task", status="blocked")

    after_goals = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='goals'"
    ).fetchone()[0]
    after_tasks = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='tasks'"
    ).fetchone()[0]
    assert before_goals == after_goals
    assert before_tasks == after_tasks


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


def test_create_get_list_goal(conn):
    gid = cdb.create_goal(conn, title="Expand the honey business", priority=5)
    goal = cdb.get_goal(conn, gid)

    assert goal is not None
    assert goal.title == "Expand the honey business"
    assert goal.status == "active"
    assert goal.priority == 5
    assert goal.completed_at is None

    listed = cdb.list_goals(conn)
    assert [g.id for g in listed] == [gid]


def test_list_goals_filters_by_status_and_project(conn):
    gid_a = cdb.create_goal(conn, title="A", project_id="proj-1")
    gid_b = cdb.create_goal(conn, title="B", project_id="proj-2")
    cdb.update_goal(conn, gid_b, status="paused")

    assert [g.id for g in cdb.list_goals(conn, status="active")] == [gid_a]
    assert [g.id for g in cdb.list_goals(conn, project_id="proj-2")] == [gid_b]


def test_update_goal_only_changes_provided_fields(conn):
    gid = cdb.create_goal(conn, title="Original", description="orig-desc", priority=1)

    changed = cdb.update_goal(conn, gid, priority=9)
    assert changed is True

    goal = cdb.get_goal(conn, gid)
    assert goal.title == "Original"
    assert goal.description == "orig-desc"
    assert goal.priority == 9


def test_goal_status_transition_stamps_and_clears_completed_at(conn):
    gid = cdb.create_goal(conn, title="Ship the thing")
    goal = cdb.get_goal(conn, gid)
    assert goal.completed_at is None

    cdb.update_goal(conn, gid, status="done")
    goal = cdb.get_goal(conn, gid)
    assert goal.status == "done"
    assert goal.completed_at is not None

    cdb.update_goal(conn, gid, status="active")
    goal = cdb.get_goal(conn, gid)
    assert goal.status == "active"
    assert goal.completed_at is None


def test_archive_and_restore_goal(conn):
    gid = cdb.create_goal(conn, title="Seasonal goal")

    assert cdb.archive_goal(conn, gid) is True
    assert cdb.get_goal(conn, gid).status == "abandoned"
    # Archiving is soft — the row still exists and is retrievable.
    assert cdb.get_goal(conn, gid) is not None

    assert cdb.restore_goal(conn, gid) is True
    assert cdb.get_goal(conn, gid).status == "active"


def test_create_goal_rejects_empty_title(conn):
    with pytest.raises(ValueError):
        cdb.create_goal(conn, title="   ")


def test_create_goal_rejects_invalid_status(conn):
    with pytest.raises(ValueError):
        cdb.create_goal(conn, title="X", status="not-a-real-status")


def test_update_goal_rejects_invalid_status(conn):
    gid = cdb.create_goal(conn, title="X")
    with pytest.raises(ValueError):
        cdb.update_goal(conn, gid, status="bogus")


# ---------------------------------------------------------------------------
# Phase 13D: blocked status (Goals)
# ---------------------------------------------------------------------------


def test_create_goal_with_blocked_status(conn):
    gid = cdb.create_goal(conn, title="Waiting on legal sign-off", status="blocked")
    goal = cdb.get_goal(conn, gid)
    assert goal.status == "blocked"


def test_update_active_goal_to_blocked(conn):
    gid = cdb.create_goal(conn, title="Ship the thing")
    assert cdb.get_goal(conn, gid).status == "active"

    changed = cdb.update_goal(conn, gid, status="blocked")
    assert changed is True
    assert cdb.get_goal(conn, gid).status == "blocked"


def test_blocked_goal_completed_at_not_stamped(conn):
    """blocked is non-terminal — only "done" stamps completed_at, exactly
    like every other non-done status (paused, active)."""
    gid = cdb.create_goal(conn, title="Ship the thing")
    cdb.update_goal(conn, gid, status="blocked")
    assert cdb.get_goal(conn, gid).completed_at is None


def test_blocked_goal_can_transition_to_done(conn):
    gid = cdb.create_goal(conn, title="Ship the thing", status="blocked")
    cdb.update_goal(conn, gid, status="done")
    goal = cdb.get_goal(conn, gid)
    assert goal.status == "done"
    assert goal.completed_at is not None


def test_list_goals_filters_by_blocked_status(conn):
    gid_active = cdb.create_goal(conn, title="Active one")
    gid_blocked = cdb.create_goal(conn, title="Blocked one", status="blocked")

    assert [g.id for g in cdb.list_goals(conn, status="blocked")] == [gid_blocked]
    assert [g.id for g in cdb.list_goals(conn, status="active")] == [gid_active]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def test_create_get_list_task(conn):
    tid = cdb.create_task(conn, title="Prepare TBS submission documents", priority=3)
    task = cdb.get_task(conn, tid)

    assert task is not None
    assert task.title == "Prepare TBS submission documents"
    assert task.status == "pending"
    assert task.priority == 3

    assert [t.id for t in cdb.list_tasks(conn)] == [tid]


def test_task_linked_to_goal(conn):
    gid = cdb.create_goal(conn, title="Regulatory compliance")
    tid = cdb.create_task(conn, title="Prepare TBS submission", goal_id=gid)

    task = cdb.get_task(conn, tid)
    assert task.goal_id == gid
    assert [t.id for t in cdb.list_tasks(conn, goal_id=gid)] == [tid]


def test_task_deadline_remains_null_when_omitted(conn):
    tid = cdb.create_task(conn, title="No deadline given")
    task = cdb.get_task(conn, tid)
    assert task.deadline_at is None


def test_task_deadline_is_stored_exactly_when_explicitly_given(conn):
    tid = cdb.create_task(conn, title="Has a real deadline", deadline_at=1234567890)
    task = cdb.get_task(conn, tid)
    assert task.deadline_at == 1234567890


def test_update_task_deadline_sentinel_semantics(conn):
    tid = cdb.create_task(conn, title="Deadline lifecycle")

    # Omitting deadline_at entirely leaves it untouched (still NULL).
    cdb.update_task(conn, tid, priority=2)
    assert cdb.get_task(conn, tid).deadline_at is None

    # Explicitly setting it works.
    cdb.update_task(conn, tid, deadline_at=1700000000)
    assert cdb.get_task(conn, tid).deadline_at == 1700000000

    # Explicitly passing None clears it back to NULL.
    cdb.update_task(conn, tid, deadline_at=None)
    assert cdb.get_task(conn, tid).deadline_at is None


def test_update_task_and_complete_task(conn):
    tid = cdb.create_task(conn, title="Work item", status="pending")

    cdb.update_task(conn, tid, status="in_progress")
    assert cdb.get_task(conn, tid).status == "in_progress"
    assert cdb.get_task(conn, tid).completed_at is None

    assert cdb.complete_task(conn, tid) is True
    task = cdb.get_task(conn, tid)
    assert task.status == "done"
    assert task.completed_at is not None


def test_create_task_rejects_empty_title(conn):
    with pytest.raises(ValueError):
        cdb.create_task(conn, title="")


def test_create_task_rejects_invalid_status(conn):
    with pytest.raises(ValueError):
        cdb.create_task(conn, title="X", status="not-a-status")


# ---------------------------------------------------------------------------
# Phase 13D: blocked status (Tasks)
# ---------------------------------------------------------------------------


def test_create_task_with_blocked_status(conn):
    tid = cdb.create_task(conn, title="Waiting on API access", status="blocked")
    task = cdb.get_task(conn, tid)
    assert task.status == "blocked"


def test_update_pending_task_to_blocked(conn):
    tid = cdb.create_task(conn, title="Work item", status="pending")
    changed = cdb.update_task(conn, tid, status="blocked")
    assert changed is True
    assert cdb.get_task(conn, tid).status == "blocked"


def test_update_in_progress_task_to_blocked(conn):
    tid = cdb.create_task(conn, title="Work item", status="in_progress")
    cdb.update_task(conn, tid, status="blocked")
    assert cdb.get_task(conn, tid).status == "blocked"


def test_blocked_task_completed_at_not_stamped(conn):
    tid = cdb.create_task(conn, title="Work item", status="pending")
    cdb.update_task(conn, tid, status="blocked")
    assert cdb.get_task(conn, tid).completed_at is None


def test_blocked_task_can_be_completed(conn):
    tid = cdb.create_task(conn, title="Work item", status="blocked")
    assert cdb.complete_task(conn, tid) is True
    task = cdb.get_task(conn, tid)
    assert task.status == "done"
    assert task.completed_at is not None


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def test_create_list_get_decision(conn):
    did = cdb.create_decision(
        conn,
        decision="Use generic company names in FSMS documents unless explicitly requested.",
        rationale="Client requested anonymized templates by default.",
    )
    decision = cdb.get_decision(conn, did)

    assert decision is not None
    assert decision.decision.startswith("Use generic company names")
    assert decision.rationale == "Client requested anonymized templates by default."
    assert decision.superseded_by is None

    assert [d.id for d in cdb.list_decisions(conn)] == [did]


def test_decision_linked_to_goal_and_project(conn):
    gid = cdb.create_goal(conn, title="FSMS rollout")
    did = cdb.create_decision(
        conn, decision="Adopt template X", goal_id=gid, project_id="proj-fsms"
    )
    decision = cdb.get_decision(conn, did)
    assert decision.goal_id == gid
    assert decision.project_id == "proj-fsms"
    assert [d.id for d in cdb.list_decisions(conn, goal_id=gid)] == [did]
    assert [d.id for d in cdb.list_decisions(conn, project_id="proj-fsms")] == [did]


def test_decision_supersession_preserves_old_decision(conn):
    old_id = cdb.create_decision(conn, decision="Use company real names in documents.")

    new_id = cdb.supersede_decision(
        conn, old_id, decision="Use generic company names unless explicitly requested.",
        rationale="Client changed policy.",
    )

    old = cdb.get_decision(conn, old_id)
    new = cdb.get_decision(conn, new_id)

    assert old is not None  # the old decision is preserved, never deleted
    assert old.superseded_by == new_id
    assert new.decision.startswith("Use generic company names")

    # Default listing hides superseded decisions; the new one is current.
    current_ids = [d.id for d in cdb.list_decisions(conn)]
    assert new_id in current_ids
    assert old_id not in current_ids

    # But history remains available on request.
    all_ids = [d.id for d in cdb.list_decisions(conn, include_superseded=True)]
    assert old_id in all_ids and new_id in all_ids


def test_supersede_decision_inherits_goal_and_project_by_default(conn):
    gid = cdb.create_goal(conn, title="G")
    old_id = cdb.create_decision(conn, decision="X", goal_id=gid, project_id="proj-1")
    new_id = cdb.supersede_decision(conn, old_id, decision="Y")
    new = cdb.get_decision(conn, new_id)
    assert new.goal_id == gid
    assert new.project_id == "proj-1"


def test_supersede_nonexistent_decision_raises(conn):
    with pytest.raises(ValueError):
        cdb.supersede_decision(conn, "d_doesnotexist", decision="Y")


def test_create_decision_rejects_empty_text(conn):
    with pytest.raises(ValueError):
        cdb.create_decision(conn, decision="   ")


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def test_timestamps_are_populated(conn):
    gid = cdb.create_goal(conn, title="X")
    goal = cdb.get_goal(conn, gid)
    assert isinstance(goal.created_at, int) and goal.created_at > 0
    assert isinstance(goal.updated_at, int) and goal.updated_at > 0
    assert goal.updated_at >= goal.created_at

    cdb.update_goal(conn, gid, priority=1)
    updated = cdb.get_goal(conn, gid)
    assert updated.updated_at >= updated.created_at


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_round_trips_with_valid_source(conn):
    gid = cdb.create_goal(
        conn, title="X", provenance={"source": "conversation", "session_id": "s-123"}
    )
    goal = cdb.get_goal(conn, gid)
    assert goal.provenance["source"] == "conversation"
    assert goal.provenance["session_id"] == "s-123"
    assert goal.provenance["provenance_status"] == "tracked"


def test_provenance_unknown_source_normalized(conn):
    gid = cdb.create_goal(conn, title="X", provenance={"source": "totally-not-a-real-source"})
    goal = cdb.get_goal(conn, gid)
    assert goal.provenance["source"] == "unknown"
    assert goal.provenance["provenance_status"] == "unavailable"


def test_provenance_missing_defaults_safely(conn):
    gid = cdb.create_goal(conn, title="X")
    goal = cdb.get_goal(conn, gid)
    assert goal.provenance == {"source": "unknown", "provenance_status": "unavailable"}


def test_provenance_redacts_secret_looking_values(conn):
    secret = "sk-abcdef1234567890abcdef1234567890"
    tid = cdb.create_task(
        conn, title="X", provenance={"source": "tool_result", "note": f"api_key={secret}"}
    )
    task = cdb.get_task(conn, tid)
    assert secret not in str(task.provenance)


def test_provenance_oversized_is_truncated_not_rejected(conn):
    huge = "x" * 5000
    did = cdb.create_decision(
        conn, decision="X", provenance={"source": "manual", "blob": huge}
    )
    decision = cdb.get_decision(conn, did)
    assert decision.provenance["provenance_status"] == "truncated"
    assert "blob" not in decision.provenance


def test_provenance_drops_unsupported_list_values(conn):
    gid = cdb.create_goal(
        conn, title="X", provenance={"source": "manual", "tags": ["a", "b"]}
    )
    goal = cdb.get_goal(conn, gid)
    assert "tags" not in goal.provenance


# ---------------------------------------------------------------------------
# Provenance — Phase 10A whitelist extension (user_confirmed, model_proposed,
# imported). These prove the sanitizer accepts the three new values exactly
# like any other whitelisted source — no new logic, no special-casing, no
# proposal/confirmation mechanism implemented here (that is explicitly out
# of scope for Phase 10A).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("new_source", ["user_confirmed", "model_proposed", "imported"])
def test_phase10a_new_provenance_sources_are_accepted(conn, new_source):
    gid = cdb.create_goal(conn, title="X", provenance={"source": new_source})
    goal = cdb.get_goal(conn, gid)
    assert goal.provenance["source"] == new_source
    assert goal.provenance["provenance_status"] == "tracked"


def test_phase10a_new_sources_work_across_all_three_record_types(conn):
    gid = cdb.create_goal(conn, title="G", provenance={"source": "user_confirmed"})
    tid = cdb.create_task(conn, title="T", provenance={"source": "model_proposed"})
    did = cdb.create_decision(conn, decision="D", provenance={"source": "imported"})

    assert cdb.get_goal(conn, gid).provenance["source"] == "user_confirmed"
    assert cdb.get_task(conn, tid).provenance["source"] == "model_proposed"
    assert cdb.get_decision(conn, did).provenance["source"] == "imported"


def test_phase10a_all_expected_sources_present_in_whitelist():
    """Locks in the full vocabulary this phase adds, without asserting an
    exact set size (a change-detector anti-pattern) — just that every
    expected value is present, old and new."""
    expected = {
        "conversation", "tool_result", "web", "manual", "inferred", "unknown",
        "user_confirmed", "model_proposed", "imported",
    }
    assert expected <= cdb.PROVENANCE_SOURCES


def test_phase10a_invalid_source_still_falls_back_to_unknown(conn):
    """Existing fallback behavior is unchanged by the whitelist extension."""
    gid = cdb.create_goal(conn, title="X", provenance={"source": "definitely-invalid"})
    goal = cdb.get_goal(conn, gid)
    assert goal.provenance["source"] == "unknown"
    assert goal.provenance["provenance_status"] == "unavailable"


def test_phase10a_model_proposed_is_not_written_by_any_existing_call_path():
    """model_proposed must remain representable-but-unused: no create_*/
    update_*/complete_task/archive_goal/restore_goal/supersede_decision
    function BODY defaults to or auto-populates it. This is a static guard
    against Phase 10A accidentally starting to implement the proposal
    mechanism (explicitly out of scope) — checked per-function (not by
    string-surgery on the whole module) so it doesn't care that the
    PROVENANCE_SOURCES comment block legitimately explains the term."""
    write_functions = [
        cdb.create_goal, cdb.update_goal, cdb.archive_goal, cdb.restore_goal,
        cdb.create_task, cdb.update_task, cdb.complete_task,
        cdb.create_decision, cdb.supersede_decision,
    ]
    for fn in write_functions:
        assert "model_proposed" not in inspect.getsource(fn), (
            f"{fn.__name__} must not reference model_proposed directly"
        )


def test_phase10a_redaction_and_size_limits_unchanged_for_new_sources(conn):
    """Existing redaction/size-cap behavior must be identical regardless of
    which (old or new) source value is used."""
    secret = "sk-abcdef1234567890abcdef1234567890"
    gid = cdb.create_goal(
        conn, title="X",
        provenance={"source": "user_confirmed", "note": f"api_key={secret}"},
    )
    assert secret not in str(cdb.get_goal(conn, gid).provenance)

    huge = "y" * 5000
    tid = cdb.create_task(
        conn, title="X", provenance={"source": "model_proposed", "blob": huge},
    )
    task = cdb.get_task(conn, tid)
    assert task.provenance["provenance_status"] == "truncated"
    assert "blob" not in task.provenance


# ---------------------------------------------------------------------------
# Foreign keys
# ---------------------------------------------------------------------------


def test_task_with_nonexistent_goal_id_is_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        cdb.create_task(conn, title="X", goal_id="g_doesnotexist")


def test_decision_with_nonexistent_goal_id_is_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        cdb.create_decision(conn, decision="X", goal_id="g_doesnotexist")


def test_task_with_real_goal_id_succeeds(conn):
    gid = cdb.create_goal(conn, title="Real goal")
    tid = cdb.create_task(conn, title="X", goal_id=gid)
    assert cdb.get_task(conn, tid).goal_id == gid


def test_deleting_a_goal_with_dependent_task_is_blocked_not_cascaded(conn):
    """No cascading deletion: a goal with a dependent task cannot be silently
    removed, and this module exposes no hard-delete for goals anyway."""
    gid = cdb.create_goal(conn, title="Has a dependent task")
    cdb.create_task(conn, title="Depends on it", goal_id=gid)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM goals WHERE id = ?", (gid,))

    # The goal and its task both survive the blocked delete attempt.
    assert cdb.get_goal(conn, gid) is not None


# ---------------------------------------------------------------------------
# Additive migration
# ---------------------------------------------------------------------------


def test_legacy_database_missing_provenance_column_is_migrated(tmp_path):
    """Simulate a pre-provenance-column continuity.db and verify connect()
    adds the missing column in place, matching projects_db.py's established
    additive-migration convention (schema init never drops/recreates a
    table that already exists — only ALTER TABLE ADD COLUMN is used)."""
    path = tmp_path / "legacy_continuity.db"

    legacy = sqlite3.connect(str(path))
    try:
        legacy.execute(
            "CREATE TABLE goals ("
            " id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT,"
            " status TEXT NOT NULL DEFAULT 'active', priority INTEGER NOT NULL DEFAULT 0,"
            " project_id TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,"
            " completed_at INTEGER"
            ")"
        )
        legacy.commit()
    finally:
        legacy.close()

    cols_before = {
        row[1]
        for row in sqlite3.connect(str(path)).execute("PRAGMA table_info(goals)")
    }
    assert "provenance" not in cols_before

    conn = cdb.connect(db_path=path)
    try:
        cols_after = {row["name"] for row in conn.execute("PRAGMA table_info(goals)")}
        assert "provenance" in cols_after

        # The migrated table is fully usable.
        gid = cdb.create_goal(conn, title="Works after migration")
        assert cdb.get_goal(conn, gid).title == "Works after migration"
    finally:
        conn.close()


def test_reopening_same_db_is_idempotent_and_does_not_duplicate_columns(tmp_path):
    path = tmp_path / "continuity.db"
    a = cdb.connect(db_path=path)
    a.close()
    # Second open must not raise (e.g. "duplicate column name") and must
    # still work normally.
    b = cdb.connect(db_path=path)
    try:
        gid = cdb.create_goal(b, title="Second open works")
        assert cdb.get_goal(b, gid) is not None
    finally:
        b.close()


# ---------------------------------------------------------------------------
# Per-profile isolation
# ---------------------------------------------------------------------------


def test_per_profile_isolation(tmp_path):
    a = cdb.connect(db_path=tmp_path / "a" / "continuity.db")
    b = cdb.connect(db_path=tmp_path / "b" / "continuity.db")
    try:
        cdb.create_goal(a, title="Only in A")
        assert len(cdb.list_goals(a)) == 1
        assert cdb.list_goals(b) == []
    finally:
        a.close()
        b.close()
