"""Tests for the Phase 9B Jarvis continuity tools (goals/tasks/decisions).

These tools dispatch to hermes_cli.continuity_db directly and have zero
dependency on rag_memory.py / Qdrant (Phase 9B scope). Unlike Jarvis's RAG
tools, continuity_db needs no elaborate Qdrant-isolation fixture: the
provider resolves its SQLite path from ``hermes_home`` (passed to
``initialize()``), so pointing that at a pytest ``tmp_path`` is sufficient
to guarantee production ``~/.hermes/continuity.db`` is never opened,
created, or modified by this suite. No test here imports, opens, or
otherwise touches Jarvis's Qdrant store.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def provider(tmp_path):
    from plugins.memory.jarvis import HermesRAGProvider

    p = HermesRAGProvider()
    p.initialize("session-continuity", hermes_home=str(tmp_path), agent_context="primary")
    return p


def _call(provider, tool_name, args):
    return json.loads(provider.handle_tool_call(tool_name, args))


# ---------------------------------------------------------------------------
# Tool registration / schema presence
# ---------------------------------------------------------------------------

_EXPECTED_CONTINUITY_TOOLS = {
    "hermes_goal_create",
    "hermes_goal_update",
    "hermes_goal_list",
    "hermes_goal_complete",
    "hermes_task_create",
    "hermes_task_update",
    "hermes_task_list",
    "hermes_task_complete",
    "hermes_decision_record",
    "hermes_decision_list",
    "hermes_decision_supersede",
}


class TestContinuityToolRegistration:
    def test_all_eleven_continuity_tools_are_registered(self, provider):
        names = {t["name"] for t in provider.get_tool_schemas()}
        assert _EXPECTED_CONTINUITY_TOOLS <= names

    def test_existing_rag_tools_still_present_alongside_continuity_tools(self, provider):
        names = {t["name"] for t in provider.get_tool_schemas()}
        assert "hermes_rag_search" in names
        assert "hermes_rag_store" in names
        # 7 original + 11 new = 18, nothing lost, nothing duplicated.
        assert len(provider.get_tool_schemas()) == 18

    def test_continuity_tool_schemas_have_required_fields(self, provider):
        for tool in provider.get_tool_schemas():
            if tool["name"] in _EXPECTED_CONTINUITY_TOOLS:
                assert "description" in tool and tool["description"]
                assert "parameters" in tool
                assert "properties" in tool["parameters"]

    def test_goal_create_and_task_create_require_title(self, provider):
        schemas = {t["name"]: t for t in provider.get_tool_schemas()}
        assert schemas["hermes_goal_create"]["parameters"]["required"] == ["title"]
        assert schemas["hermes_task_create"]["parameters"]["required"] == ["title"]
        assert schemas["hermes_decision_record"]["parameters"]["required"] == ["decision"]


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


class TestGoalTools:
    def test_create_list_update_complete_round_trip(self, provider):
        created = _call(provider, "hermes_goal_create", {
            "title": "Expand the honey business", "priority": 5,
        })
        assert created["ok"] is True
        gid = created["goal"]["id"]
        assert created["goal"]["status"] == "active"

        listed = _call(provider, "hermes_goal_list", {})
        assert listed["ok"] is True
        assert gid in [g["id"] for g in listed["goals"]]

        updated = _call(provider, "hermes_goal_update", {"goal_id": gid, "priority": 9})
        assert updated["ok"] is True
        assert updated["goal"]["priority"] == 9
        assert updated["goal"]["title"] == "Expand the honey business"  # untouched

        completed = _call(provider, "hermes_goal_complete", {"goal_id": gid})
        assert completed["ok"] is True
        assert completed["goal"]["status"] == "done"
        assert completed["goal"]["completed_at"] is not None

    def test_goal_list_filters_by_status(self, provider):
        _call(provider, "hermes_goal_create", {"title": "A"})
        b = _call(provider, "hermes_goal_create", {"title": "B"})["goal"]["id"]
        _call(provider, "hermes_goal_update", {"goal_id": b, "status": "paused"})

        active_only = _call(provider, "hermes_goal_list", {"status": "active"})
        assert [g["id"] for g in active_only["goals"]] != [b]
        assert b not in [g["id"] for g in active_only["goals"]]

    def test_goal_create_missing_title_is_rejected(self, provider):
        result = _call(provider, "hermes_goal_create", {})
        assert result["ok"] is False
        assert "error" in result

    def test_goal_update_missing_goal_id_is_rejected(self, provider):
        result = _call(provider, "hermes_goal_update", {"title": "x"})
        assert result["ok"] is False
        assert "goal_id is required" in result["error"]

    def test_goal_update_nonexistent_id_is_rejected(self, provider):
        result = _call(provider, "hermes_goal_update", {"goal_id": "g_nope", "priority": 1})
        assert result["ok"] is False

    def test_goal_complete_missing_goal_id_is_rejected(self, provider):
        result = _call(provider, "hermes_goal_complete", {})
        assert result["ok"] is False
        assert "goal_id is required" in result["error"]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TestTaskTools:
    def test_create_list_update_complete_round_trip(self, provider):
        goal = _call(provider, "hermes_goal_create", {"title": "Regulatory compliance"})
        gid = goal["goal"]["id"]

        created = _call(provider, "hermes_task_create", {
            "title": "Prepare TBS submission documents", "goal_id": gid,
        })
        assert created["ok"] is True
        tid = created["task"]["id"]
        assert created["task"]["status"] == "pending"
        assert created["task"]["goal_id"] == gid
        assert created["task"]["deadline_at"] is None

        listed = _call(provider, "hermes_task_list", {"goal_id": gid})
        assert tid in [t["id"] for t in listed["tasks"]]

        updated = _call(provider, "hermes_task_update", {"task_id": tid, "status": "in_progress"})
        assert updated["ok"] is True
        assert updated["task"]["status"] == "in_progress"
        assert updated["task"]["completed_at"] is None

        completed = _call(provider, "hermes_task_complete", {"task_id": tid})
        assert completed["ok"] is True
        assert completed["task"]["status"] == "done"
        assert completed["task"]["completed_at"] is not None

    def test_task_create_with_explicit_deadline(self, provider):
        created = _call(provider, "hermes_task_create", {
            "title": "Has a deadline", "deadline_at": "2026-09-30",
        })
        assert created["ok"] is True
        assert created["task"]["deadline_at"] is not None

    def test_task_create_never_infers_deadline_when_omitted(self, provider):
        created = _call(provider, "hermes_task_create", {"title": "No deadline given"})
        assert created["ok"] is True
        assert created["task"]["deadline_at"] is None

    def test_task_create_invalid_deadline_is_rejected_not_guessed(self, provider):
        result = _call(provider, "hermes_task_create", {
            "title": "Bad deadline", "deadline_at": "not-a-date",
        })
        assert result["ok"] is False
        assert "deadline_at" in result["error"]

    def test_task_update_deadline_sentinel_semantics(self, provider):
        tid = _call(provider, "hermes_task_create", {"title": "X"})["task"]["id"]

        # Omitting deadline_at entirely leaves it untouched.
        r1 = _call(provider, "hermes_task_update", {"task_id": tid, "priority": 2})
        assert r1["task"]["deadline_at"] is None

        # Explicitly setting it works.
        r2 = _call(provider, "hermes_task_update", {"task_id": tid, "deadline_at": "2026-12-01"})
        assert r2["task"]["deadline_at"] is not None

        # Explicitly clearing it (empty string) works.
        r3 = _call(provider, "hermes_task_update", {"task_id": tid, "deadline_at": ""})
        assert r3["task"]["deadline_at"] is None

    def test_task_create_missing_title_is_rejected(self, provider):
        result = _call(provider, "hermes_task_create", {})
        assert result["ok"] is False

    def test_task_update_missing_task_id_is_rejected(self, provider):
        result = _call(provider, "hermes_task_update", {"priority": 1})
        assert result["ok"] is False
        assert "task_id is required" in result["error"]

    def test_task_complete_missing_task_id_is_rejected(self, provider):
        result = _call(provider, "hermes_task_complete", {})
        assert result["ok"] is False
        assert "task_id is required" in result["error"]

    def test_task_with_nonexistent_goal_id_is_rejected(self, provider):
        result = _call(provider, "hermes_task_create", {"title": "X", "goal_id": "g_nope"})
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


class TestDecisionTools:
    def test_record_list_supersede_round_trip(self, provider):
        recorded = _call(provider, "hermes_decision_record", {
            "decision": "Use generic company names in FSMS documents unless explicitly requested.",
            "rationale": "Client requested anonymized templates by default.",
        })
        assert recorded["ok"] is True
        did = recorded["decision"]["id"]
        assert recorded["decision"]["superseded_by"] is None

        listed = _call(provider, "hermes_decision_list", {})
        assert did in [d["id"] for d in listed["decisions"]]

        superseded = _call(provider, "hermes_decision_supersede", {
            "decision_id": did,
            "decision": "Use real company names when the client explicitly opts in.",
        })
        assert superseded["ok"] is True
        new_id = superseded["decision"]["id"]
        assert new_id != did

        # Old decision hidden from default listing, visible with the flag.
        default_listing = _call(provider, "hermes_decision_list", {})
        assert did not in [d["id"] for d in default_listing["decisions"]]
        assert new_id in [d["id"] for d in default_listing["decisions"]]

        full_listing = _call(provider, "hermes_decision_list", {"include_superseded": True})
        assert did in [d["id"] for d in full_listing["decisions"]]

    def test_decision_linked_to_goal(self, provider):
        gid = _call(provider, "hermes_goal_create", {"title": "FSMS rollout"})["goal"]["id"]
        did = _call(provider, "hermes_decision_record", {
            "decision": "Adopt template X", "goal_id": gid,
        })["decision"]["id"]
        listed = _call(provider, "hermes_decision_list", {"goal_id": gid})
        assert did in [d["id"] for d in listed["decisions"]]

    def test_decision_record_missing_text_is_rejected(self, provider):
        result = _call(provider, "hermes_decision_record", {})
        assert result["ok"] is False

    def test_decision_supersede_missing_decision_id_is_rejected(self, provider):
        result = _call(provider, "hermes_decision_supersede", {"decision": "Y"})
        assert result["ok"] is False
        assert "decision_id is required" in result["error"]

    def test_decision_supersede_nonexistent_id_is_rejected(self, provider):
        result = _call(provider, "hermes_decision_supersede", {
            "decision_id": "d_nope", "decision": "Y",
        })
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Input length validation (provider/tool boundary — never silently truncates)
# ---------------------------------------------------------------------------


class TestLengthValidation:
    def test_goal_title_over_limit_is_rejected_with_clear_error(self, provider):
        from plugins.memory.jarvis import CONTINUITY_TITLE_MAX_LENGTH

        too_long = "x" * (CONTINUITY_TITLE_MAX_LENGTH + 1)
        result = _call(provider, "hermes_goal_create", {"title": too_long})
        assert result["ok"] is False
        assert "title" in result["error"] and "maximum length" in result["error"]

    def test_goal_description_over_limit_is_rejected(self, provider):
        from plugins.memory.jarvis import CONTINUITY_TEXT_MAX_LENGTH

        too_long = "x" * (CONTINUITY_TEXT_MAX_LENGTH + 1)
        result = _call(provider, "hermes_goal_create", {"title": "ok", "description": too_long})
        assert result["ok"] is False
        assert "description" in result["error"]

    def test_over_limit_input_is_rejected_not_truncated_and_not_stored(self, provider):
        from plugins.memory.jarvis import CONTINUITY_TITLE_MAX_LENGTH

        too_long = "y" * (CONTINUITY_TITLE_MAX_LENGTH + 50)
        with patch("hermes_cli.continuity_db.create_goal") as mock_create:
            result = _call(provider, "hermes_goal_create", {"title": too_long})
        assert result["ok"] is False
        mock_create.assert_not_called()  # rejected before ever reaching storage

    def test_decision_text_over_limit_is_rejected(self, provider):
        from plugins.memory.jarvis import CONTINUITY_DECISION_MAX_LENGTH

        too_long = "z" * (CONTINUITY_DECISION_MAX_LENGTH + 1)
        result = _call(provider, "hermes_decision_record", {"decision": too_long})
        assert result["ok"] is False

    def test_decision_rationale_over_limit_is_rejected(self, provider):
        from plugins.memory.jarvis import CONTINUITY_TEXT_MAX_LENGTH

        too_long = "w" * (CONTINUITY_TEXT_MAX_LENGTH + 1)
        result = _call(provider, "hermes_decision_record", {"decision": "ok", "rationale": too_long})
        assert result["ok"] is False

    def test_task_title_over_limit_is_rejected(self, provider):
        from plugins.memory.jarvis import CONTINUITY_TITLE_MAX_LENGTH

        too_long = "v" * (CONTINUITY_TITLE_MAX_LENGTH + 1)
        result = _call(provider, "hermes_task_create", {"title": too_long})
        assert result["ok"] is False

    def test_title_at_exactly_the_limit_is_accepted(self, provider):
        from plugins.memory.jarvis import CONTINUITY_TITLE_MAX_LENGTH

        exactly = "a" * CONTINUITY_TITLE_MAX_LENGTH
        result = _call(provider, "hermes_goal_create", {"title": exactly})
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Bounded list output
# ---------------------------------------------------------------------------


class TestBoundedListOutput:
    def test_list_output_is_capped_at_hard_limit_with_visible_total(self, provider):
        from plugins.memory.jarvis import CONTINUITY_LIST_MAX_LIMIT

        n = CONTINUITY_LIST_MAX_LIMIT + 10
        for i in range(n):
            _call(provider, "hermes_goal_create", {"title": f"Goal {i}"})

        result = _call(provider, "hermes_goal_list", {"limit": 10_000})  # model tries to bypass the cap
        assert result["ok"] is True
        assert len(result["goals"]) == CONTINUITY_LIST_MAX_LIMIT
        assert result["count"] == CONTINUITY_LIST_MAX_LIMIT
        assert result["total_matching"] == n

    def test_default_limit_is_used_when_omitted(self, provider):
        from plugins.memory.jarvis import CONTINUITY_LIST_DEFAULT_LIMIT

        n = CONTINUITY_LIST_DEFAULT_LIMIT + 5
        for i in range(n):
            _call(provider, "hermes_task_create", {"title": f"Task {i}"})

        result = _call(provider, "hermes_task_list", {})
        assert len(result["tasks"]) == CONTINUITY_LIST_DEFAULT_LIMIT
        assert result["total_matching"] == n

    def test_negative_or_zero_limit_falls_back_to_default(self, provider):
        from plugins.memory.jarvis import CONTINUITY_LIST_DEFAULT_LIMIT

        _call(provider, "hermes_goal_create", {"title": "X"})
        result = _call(provider, "hermes_goal_list", {"limit": -5})
        assert result["ok"] is True
        # Doesn't crash, and behaves like the default (1 result <= default cap).
        assert len(result["goals"]) <= CONTINUITY_LIST_DEFAULT_LIMIT


# ---------------------------------------------------------------------------
# Provider dispatch actually reaches continuity_db (not a no-op / mocked)
# ---------------------------------------------------------------------------


class TestDispatchReachesContinuityDb:
    def test_created_goal_is_readable_directly_via_continuity_db(self, provider, tmp_path):
        from hermes_cli import continuity_db as cdb

        created = _call(provider, "hermes_goal_create", {"title": "Direct-read check"})
        gid = created["goal"]["id"]

        # Bypass the provider entirely and read the same file directly.
        conn = cdb.connect(db_path=tmp_path / "continuity.db")
        try:
            goal = cdb.get_goal(conn, gid)
        finally:
            conn.close()

        assert goal is not None
        assert goal.title == "Direct-read check"

    def test_provider_uses_hermes_home_scoped_database_path(self, provider, tmp_path):
        _call(provider, "hermes_goal_create", {"title": "X"})
        assert (tmp_path / "continuity.db").exists()


# ---------------------------------------------------------------------------
# No Qdrant / rag_memory involvement
# ---------------------------------------------------------------------------


class TestNoRagMemoryInvolvement:
    def test_continuity_tools_work_even_when_rag_memory_is_unimportable(self, provider):
        """Structural proof of independence: continuity tools must not need
        rag_memory to be importable at all. sys.modules['rag_memory'] = None
        makes any `import rag_memory` raise ImportError immediately."""
        with patch.dict(sys.modules, {"rag_memory": None}):
            created = _call(provider, "hermes_goal_create", {"title": "Independent of RAG"})
            assert created["ok"] is True

            task = _call(provider, "hermes_task_create", {"title": "Also independent"})
            assert task["ok"] is True

            decision = _call(provider, "hermes_decision_record", {"decision": "Also independent"})
            assert decision["ok"] is True

    def test_rag_memory_is_never_imported_as_a_side_effect_of_continuity_calls(self, provider):
        """Confirms handle_tool_call's early continuity-dispatch branch
        returns before the `from rag_memory import (...)` line is ever
        reached — checked by capturing sys.modules membership, not by
        patching rag_memory (which would itself require importing it,
        risking exactly the production-Qdrant-touching issue this suite
        must avoid)."""
        was_imported_before = "rag_memory" in sys.modules
        _call(provider, "hermes_goal_create", {"title": "X"})
        _call(provider, "hermes_task_create", {"title": "Y"})
        _call(provider, "hermes_decision_record", {"decision": "Z"})
        # Calling continuity tools must never cause a NEW rag_memory import.
        if not was_imported_before:
            assert "rag_memory" not in sys.modules
