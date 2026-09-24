"""Tests for Phase 10E: staged/background Jarvis continuity proposals via
the existing tools/write_approval.py infrastructure.

Every test that enables the gate uses a fixture that sets BOTH the
provider's ``hermes_home`` kwarg AND the ``HERMES_HOME`` environment
variable (which write_approval.py's pending store resolves through
``hermes_constants.get_hermes_home()``) so no test ever touches production
``~/.hermes``. One test deliberately DIVERGES the two to prove Phase 10E's
path-reconciliation requirement (payload-embedded hermes_home, not the
live env var, decides where an approved write lands).

No test here touches production ~/.hermes/continuity.db or Qdrant.
"""

from __future__ import annotations

import json
import os

import pytest


def _set_continuity_approval(enabled: bool) -> None:
    """Mirrors tests/tools/test_write_approval.py's own _set_approval
    helper exactly, for the new "continuity" subsystem."""
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault("continuity", {})["write_approval"] = enabled
    cfg.save_config(c)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolates BOTH continuity.db resolution (provider's hermes_home
    kwarg) AND write_approval's pending-store resolution (HERMES_HOME env
    var) under the same tmp_path — the common, non-divergent case."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def provider():
    from plugins.memory.jarvis import HermesRAGProvider
    return HermesRAGProvider()


def _call(provider, tool_name, args):
    return json.loads(provider.handle_tool_call(tool_name, args))


class TestContinuitySubsystemRecognized:
    def test_continuity_constant_and_subsystem_membership(self, hermes_home):
        from tools import write_approval as wa
        assert wa.CONTINUITY == "continuity"
        assert wa.CONTINUITY in wa._SUBSYSTEMS

    def test_write_approval_enabled_defaults_false(self, hermes_home):
        from tools import write_approval as wa
        assert wa.write_approval_enabled(wa.CONTINUITY) is False

    def test_evaluate_gate_returns_allow_by_default(self, hermes_home):
        from tools import write_approval as wa
        decision = wa.evaluate_gate(wa.CONTINUITY)
        assert decision.allow is True


class TestApprovalDisabledDirectWrite:
    def test_gate_off_goal_create_writes_directly(self, provider, hermes_home):
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        result = _call(provider, "hermes_goal_create", {"title": "Expand the honey business"})
        assert result["ok"] is True
        assert "staged" not in result
        assert result["goal"]["title"] == "Expand the honey business"

    def test_gate_off_task_and_decision_write_directly(self, provider, hermes_home):
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        t = _call(provider, "hermes_task_create", {"title": "Prepare TBS submission"})
        d = _call(provider, "hermes_decision_record", {"decision": "Use generic company names"})
        assert t["ok"] is True and "staged" not in t
        assert d["ok"] is True and "staged" not in d


class TestApprovalEnabledStaging:
    def test_gate_on_goal_create_stages_no_row_created(self, provider, hermes_home):
        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        result = _call(provider, "hermes_goal_create", {"title": "Expand the honey business"})

        assert result["ok"] is True
        assert result["staged"] is True
        assert "pending_id" in result

        listed = _call(provider, "hermes_goal_list", {})
        assert listed["goals"] == []  # no durable row created

    def test_gate_on_task_and_decision_stage_no_rows_created(self, provider, hermes_home):
        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        t = _call(provider, "hermes_task_create", {"title": "Prepare TBS submission"})
        d = _call(provider, "hermes_decision_record", {"decision": "Use generic company names"})

        assert t["ok"] is True and t["staged"] is True
        assert d["ok"] is True and d["staged"] is True
        assert _call(provider, "hermes_task_list", {})["tasks"] == []
        assert _call(provider, "hermes_decision_list", {})["decisions"] == []

    def test_gate_on_invalid_title_rejected_before_staging(self, provider, hermes_home):
        """Validation runs BEFORE gating — an invalid write is rejected
        outright, never staged at all."""
        from plugins.memory.jarvis import CONTINUITY_TITLE_MAX_LENGTH
        from tools import write_approval as wa

        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        too_long = "x" * (CONTINUITY_TITLE_MAX_LENGTH + 1)
        result = _call(provider, "hermes_goal_create", {"title": too_long})

        assert result["ok"] is False
        assert "staged" not in result
        assert wa.list_pending(wa.CONTINUITY) == []


class TestPendingPayloadRoundTrip:
    def test_staged_payload_survives_stage_list_get(self, provider, hermes_home):
        from tools import write_approval as wa

        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        staged = _call(provider, "hermes_goal_create", {"title": "Expand the honey business"})
        pid = staged["pending_id"]

        listed = wa.list_pending(wa.CONTINUITY)
        assert len(listed) == 1
        assert listed[0]["id"] == pid

        rec = wa.get_pending(wa.CONTINUITY, pid)
        assert rec is not None
        assert rec["payload"]["tool"] == "hermes_goal_create"
        assert rec["payload"]["args"]["title"] == "Expand the honey business"
        assert rec["payload"]["hermes_home"] == str(hermes_home)


class TestApprovalAppliesThroughToolLayer:
    def test_approved_goal_creates_expected_record(self, provider, hermes_home):
        from tools import write_approval as wa
        from plugins.memory.jarvis import apply_continuity_pending

        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        staged = _call(provider, "hermes_goal_create", {"title": "Expand the honey business", "priority": 3})
        pid = staged["pending_id"]
        rec = wa.get_pending(wa.CONTINUITY, pid)

        outcome = apply_continuity_pending(rec["payload"])
        assert outcome["success"] is True

        listed = _call(provider, "hermes_goal_list", {})
        assert len(listed["goals"]) == 1
        assert listed["goals"][0]["title"] == "Expand the honey business"
        assert listed["goals"][0]["priority"] == 3

    def test_approved_task_and_decision_create_expected_records(self, provider, hermes_home):
        from tools import write_approval as wa
        from plugins.memory.jarvis import apply_continuity_pending

        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        staged_t = _call(provider, "hermes_task_create", {"title": "Prepare TBS submission"})
        staged_d = _call(provider, "hermes_decision_record", {"decision": "Use generic company names"})

        for staged in (staged_t, staged_d):
            rec = wa.get_pending(wa.CONTINUITY, staged["pending_id"])
            assert apply_continuity_pending(rec["payload"])["success"] is True

        assert len(_call(provider, "hermes_task_list", {})["tasks"]) == 1
        assert len(_call(provider, "hermes_decision_list", {})["decisions"]) == 1

    def test_invalid_pending_payload_still_rejected_at_apply_time(self, hermes_home):
        """Defense in depth: even a hand-crafted / corrupted pending record
        with an invalid title cannot bypass provider validation by going
        through apply_continuity_pending — the exact same tool-layer
        validation runs again at apply time, not just at stage time."""
        from plugins.memory.jarvis import apply_continuity_pending, CONTINUITY_TITLE_MAX_LENGTH

        too_long = "x" * (CONTINUITY_TITLE_MAX_LENGTH + 1)
        payload = {
            "tool": "hermes_goal_create",
            "args": {"title": too_long},
            "hermes_home": str(hermes_home),
        }
        outcome = apply_continuity_pending(payload)
        assert outcome["success"] is False
        assert "error" in outcome

    def test_apply_rejects_non_gated_tool_names(self, hermes_home):
        """apply_continuity_pending only accepts the gated write tools —
        not list/complete tools, and not an arbitrary/unknown name."""
        from plugins.memory.jarvis import apply_continuity_pending

        for bad_tool in ("hermes_goal_list", "hermes_task_complete", "not_a_real_tool"):
            outcome = apply_continuity_pending({
                "tool": bad_tool, "args": {}, "hermes_home": str(hermes_home),
            })
            assert outcome["success"] is False


class TestProvenanceEnforcement:
    def test_model_proposed_in_staged_payload_never_becomes_durable(self, provider, hermes_home):
        from tools import write_approval as wa
        from plugins.memory.jarvis import apply_continuity_pending

        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        staged = _call(provider, "hermes_goal_create", {
            "title": "Expand the honey business",
            "provenance": {"source": "model_proposed"},
        })
        pid = staged["pending_id"]
        rec = wa.get_pending(wa.CONTINUITY, pid)

        # The pending payload itself may legitimately carry model_proposed —
        # that's fine, it's not a durable row yet.
        assert rec["payload"]["args"]["provenance"]["source"] == "model_proposed"

        outcome = apply_continuity_pending(rec["payload"])
        assert outcome["success"] is True

        listed = _call(provider, "hermes_goal_list", {})
        assert listed["goals"][0]["provenance"]["source"] == "user_confirmed"
        # Prove it nowhere near "model_proposed" in the durable row.
        assert "model_proposed" not in json.dumps(listed["goals"][0])

    def test_apply_overrides_provenance_even_when_staged_payload_has_none(self, hermes_home):
        from plugins.memory.jarvis import apply_continuity_pending

        payload = {
            "tool": "hermes_decision_record",
            "args": {"decision": "Keep structured continuity in SQLite"},
            "hermes_home": str(hermes_home),
        }
        outcome = apply_continuity_pending(payload)
        assert outcome["success"] is True

        from hermes_cli.continuity_db import connect, list_decisions
        conn = connect(db_path=hermes_home / "continuity.db")
        try:
            decisions = list_decisions(conn)
        finally:
            conn.close()
        assert decisions[0].provenance["source"] == "user_confirmed"


class TestStatusOnlyOperationsRemainDirect:
    def test_goal_and_task_complete_never_gated(self, provider, hermes_home):
        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        # Create directly first (gate off) so we have something to complete —
        # re-toggle the gate off just for setup, then back on for the real check.
        _set_continuity_approval(False)
        gid = _call(provider, "hermes_goal_create", {"title": "Seasonal goal"})["goal"]["id"]
        tid = _call(provider, "hermes_task_create", {"title": "Seasonal task"})["task"]["id"]

        _set_continuity_approval(True)
        completed_goal = _call(provider, "hermes_goal_complete", {"goal_id": gid})
        completed_task = _call(provider, "hermes_task_complete", {"task_id": tid})

        assert completed_goal["ok"] is True and "staged" not in completed_goal
        assert completed_goal["goal"]["status"] == "done"
        assert completed_task["ok"] is True and "staged" not in completed_task
        assert completed_task["task"]["status"] == "done"

    def test_goal_update_status_only_never_gated(self, provider, hermes_home):
        _set_continuity_approval(False)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        gid = _call(provider, "hermes_goal_create", {"title": "Seasonal goal"})["goal"]["id"]

        _set_continuity_approval(True)
        result = _call(provider, "hermes_goal_update", {"goal_id": gid, "status": "paused"})
        assert result["ok"] is True
        assert "staged" not in result
        assert result["goal"]["status"] == "paused"

    def test_goal_update_title_change_IS_gated(self, provider, hermes_home):
        """Contrast case: changing title (a content change) on the same
        goal must be staged even though status-only updates are not."""
        _set_continuity_approval(False)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        gid = _call(provider, "hermes_goal_create", {"title": "Seasonal goal"})["goal"]["id"]

        _set_continuity_approval(True)
        result = _call(provider, "hermes_goal_update", {"goal_id": gid, "title": "Renamed goal"})
        assert result["ok"] is True
        assert result["staged"] is True


class TestDeduplicationInteraction:
    def test_staged_write_does_not_enter_dedup_tracking(self, provider, hermes_home):
        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        _call(provider, "hermes_goal_create", {"title": "Expand the honey business"})
        assert provider._session_continuity_fragments == set()

    def test_bypassed_write_enters_dedup_tracking(self, provider, hermes_home):
        """Proves the mechanism apply_continuity_pending relies on: a call
        with _continuity_bypass_gate=True performs the real write AND
        tracks it for dedup, on whichever provider instance makes the
        call."""
        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        raw = provider.handle_tool_call(
            "hermes_goal_create", {"title": "Expand the honey business"},
            _continuity_bypass_gate=True,
        )
        result = json.loads(raw)
        assert result["ok"] is True
        assert "staged" not in result
        assert "Expand the honey business" in provider._session_continuity_fragments


class TestOneshotUnaffected:
    def test_oneshot_gate_off_still_writes_directly(self, hermes_home):
        """Preserves Phase 9D's validated behavior: with the gate at its
        default (off), a oneshot session can still use continuity tools
        directly — Phase 10E only changes behavior when the user opts into
        continuity.write_approval."""
        from plugins.memory.jarvis import HermesRAGProvider

        provider = HermesRAGProvider()
        provider.initialize("session-oneshot", hermes_home=str(hermes_home), agent_context="oneshot")
        result = json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Oneshot goal"}))
        assert result["ok"] is True
        assert "staged" not in result

    def test_oneshot_gate_on_stages_rather_than_writes(self, hermes_home):
        """With the gate explicitly enabled, oneshot (no interactive
        channel) stages instead of writing directly — never a silent
        direct write, and never blocked outright either."""
        from plugins.memory.jarvis import HermesRAGProvider

        _set_continuity_approval(True)
        provider = HermesRAGProvider()
        provider.initialize("session-oneshot", hermes_home=str(hermes_home), agent_context="oneshot")
        result = json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Oneshot goal"}))
        assert result["ok"] is True
        assert result["staged"] is True

    def test_oneshot_on_session_end_still_does_not_learn(self, hermes_home):
        """Uses the same sys.modules-forced-unimportable technique Phases
        9B/9C established, rather than patch("rag_memory.X") — this file
        never inserts ~/.hermes/rag onto sys.path, so a bare patch() target
        would require rag_memory to already be importable, which this test
        must not depend on (and must not risk touching production Qdrant
        by triggering a real import)."""
        import sys
        from unittest.mock import patch
        from plugins.memory.jarvis import HermesRAGProvider

        provider = HermesRAGProvider()
        provider.initialize("session-oneshot", hermes_home=str(hermes_home), agent_context="oneshot")
        assert provider._session_end_learning_enabled is False
        with patch.dict(sys.modules, {"rag_memory": None}):
            # Must not raise — and since the gate check returns before any
            # `from rag_memory import ...` line, forcing rag_memory
            # unimportable would only surface as an exception if it were
            # actually attempted.
            provider.on_session_end([{"role": "assistant", "content": "Some long enough answer text here."}])


class TestProfileIsolation:
    def test_pending_store_and_continuity_db_share_the_same_profile(self, provider, hermes_home):
        from tools import write_approval as wa

        _set_continuity_approval(True)
        provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
        _call(provider, "hermes_goal_create", {"title": "Expand the honey business"})

        assert (hermes_home / "pending" / "continuity").is_dir()
        # _handle_continuity_tool_call opens (and schema-initializes) the
        # connection BEFORE the per-tool gate check runs — shared by all 11
        # branches — so continuity.db's empty schema may now exist as a
        # side effect of the write ATTEMPT. That's harmless: no goal/task/
        # decision ROW exists, which is what "must not create a continuity
        # DB row until approved" (requirement 5) actually specifies.
        from hermes_cli.continuity_db import connect, list_goals

        conn = connect(db_path=hermes_home / "continuity.db")
        try:
            assert list_goals(conn) == []
        finally:
            conn.close()

    def test_approval_applies_against_the_profile_that_staged_it_even_when_env_diverges(
        self, tmp_path, monkeypatch
    ):
        """The strongest proof of requirement 7: stage under profile A
        (HERMES_HOME env var pointed at profile B at stage time — a
        deliberately divergent setup), then approve; the write MUST land
        in profile A's continuity.db (from the payload), never profile B's
        (from whatever HERMES_HOME happens to resolve to at apply time)."""
        from plugins.memory.jarvis import HermesRAGProvider, apply_continuity_pending
        from tools import write_approval as wa

        profile_a = tmp_path / "profile_a"
        profile_b = tmp_path / "profile_b"
        profile_a.mkdir()
        profile_b.mkdir()

        # Stage while HERMES_HOME (env) points at profile B, but the
        # PROVIDER's own hermes_home is profile A — exactly the divergence
        # requirement 7 warns against.
        monkeypatch.setenv("HERMES_HOME", str(profile_b))
        _set_continuity_approval(True)  # writes into profile_b's config.yaml

        provider = HermesRAGProvider()
        provider.initialize("session-1", hermes_home=str(profile_a), agent_context="primary")
        staged = json.loads(provider.handle_tool_call(
            "hermes_goal_create", {"title": "Profile A goal"}
        ))
        assert staged["staged"] is True
        pid = staged["pending_id"]

        # The pending JSON file itself lives under profile B (env-resolved),
        # matching write_approval.py's own get_hermes_home() behavior.
        rec = wa.get_pending(wa.CONTINUITY, pid)
        assert rec["payload"]["hermes_home"] == str(profile_a)

        # Now approve — still with HERMES_HOME env pointed at profile B.
        outcome = apply_continuity_pending(rec["payload"])
        assert outcome["success"] is True

        from hermes_cli.continuity_db import connect, list_goals
        conn_a = connect(db_path=profile_a / "continuity.db")
        try:
            goals_a = list_goals(conn_a)
        finally:
            conn_a.close()
        assert len(goals_a) == 1
        assert goals_a[0].title == "Profile A goal"

        # Profile B's continuity.db must never have been created/written.
        assert not (profile_b / "continuity.db").exists()
