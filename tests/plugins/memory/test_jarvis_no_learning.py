"""No-learning sessions (JARVIS Memory Safety, Priority 2).

Sessions whose id starts with ``nolearn_``, and processes running with
HERMES_NO_LEARNING set, must create zero persistent memories: no automatic
session-end learning (every trigger path), no explicit RAG stores, and no
learning by delegated sub-agents of such a session. Ordinary sessions keep
learning exactly as before.

Storage is a disposable tmp_path Qdrant client (production JARVIS_QDRANT_* /
JARVIS_EMBED_* are removed) or a mock; production memory is never touched.
"""

import json
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.expanduser("~/.hermes/rag"))

import agent.no_learning as nl  # noqa: E402

PRODUCTION_ENV = ("JARVIS_QDRANT_URL", "JARVIS_QDRANT_PATH", "JARVIS_QDRANT_TIMEOUT",
                  "JARVIS_EMBED_SOCKET", "JARVIS_EMBED_TIMEOUT", "JARVIS_EMBED_FALLBACK")
ORDINARY_IDS = [  # every id shape present in production state.db
    "20261011_111046_2f8b485a", "api-3750560e0646c3a2", "desktop_1791706201_2f8b485a",
    "api_1791668514_121f220c", "cron_20261011_010203_abcdef12", "eval-behavioral_memory_001",
]
NOLEARN_ID = "nolearn_desktop_1791706201_2f8b485a"
LEARNABLE = [
    {"role": "user", "content": "How should answers be formatted?"},
    {"role": "assistant", "content": "The user prefers concise answers without emoji."},
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in PRODUCTION_ENV + (nl.NO_LEARNING_ENV,):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(nl, "_warned_values", set())


@pytest.fixture
def rag(monkeypatch, tmp_path):
    import rag_memory as rm
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection(collection_name=rm.COLLECTION_NAME,
                             vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    monkeypatch.setattr(rm, "QDRANT_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(rm, "client", client)
    yield rm
    client.close()


def _count(rm):
    return rm.client.count(collection_name=rm.COLLECTION_NAME, exact=True).count


def _provider(session_id, tmp_path):
    from plugins.memory.jarvis import HermesRAGProvider
    p = HermesRAGProvider()
    p.initialize(session_id, hermes_home=str(tmp_path / "hermes_home"), agent_context="primary")
    return p


def _skip_logs(caplog, needle="persistent learning disabled"):
    return [r for r in caplog.records if needle in r.getMessage()]


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


class TestHelper:
    def test_prefix_disables(self):
        assert nl.learning_disabled_reason(NOLEARN_ID) == "session id starts with 'nolearn_'"

    @pytest.mark.parametrize("session_id", ORDINARY_IDS + [None, "", "NOLEARN_x", "x_nolearn_y", " nolearn_x"])
    def test_ordinary_and_lookalike_ids_keep_learning(self, session_id):
        assert nl.learning_disabled_reason(session_id) is None

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
    def test_env_true_values_disable(self, monkeypatch, value):
        monkeypatch.setenv(nl.NO_LEARNING_ENV, value)
        assert nl.learning_disabled_reason(ORDINARY_IDS[0]) == "HERMES_NO_LEARNING is set"

    @pytest.mark.parametrize("value", ["", "0", "false", "No", " off "])
    def test_env_false_values_keep_learning(self, monkeypatch, value):
        monkeypatch.setenv(nl.NO_LEARNING_ENV, value)
        assert nl.learning_disabled_reason(ORDINARY_IDS[0]) is None

    def test_env_unset_keeps_learning(self):
        assert nl.env_disables_learning() is False

    def test_malformed_env_disables_and_warns_once_per_value(self, monkeypatch, caplog):
        monkeypatch.setenv(nl.NO_LEARNING_ENV, "maybe")
        with caplog.at_level(logging.WARNING, logger="agent.no_learning"):
            assert nl.env_disables_learning() is True
            assert nl.env_disables_learning() is True
        warnings = [r for r in caplog.records if "unrecognised value" in r.getMessage()]
        assert len(warnings) == 1 and "'maybe'" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# Automatic session-end learning (real hook, temp Qdrant)
# ---------------------------------------------------------------------------


class TestSessionEndLearning:
    def test_control_ordinary_session_learns(self, rag, tmp_path):
        _provider(ORDINARY_IDS[2], tmp_path).on_session_end(LEARNABLE)
        assert _count(rag) == 1

    def test_prefix_session_stores_nothing_and_logs_once(self, rag, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="plugins.memory.jarvis"):
            _provider(NOLEARN_ID, tmp_path).on_session_end(LEARNABLE)
        assert _count(rag) == 0
        logs = _skip_logs(caplog)
        assert len(logs) == 1 and NOLEARN_ID in logs[0].getMessage()

    def test_env_switch_stores_nothing_and_logs_once(self, rag, tmp_path, caplog, monkeypatch):
        monkeypatch.setenv(nl.NO_LEARNING_ENV, "1")
        with caplog.at_level(logging.INFO, logger="plugins.memory.jarvis"):
            _provider(ORDINARY_IDS[0], tmp_path).on_session_end(LEARNABLE)
        assert _count(rag) == 0
        assert len(_skip_logs(caplog)) == 1

    def test_env_false_value_still_learns(self, rag, tmp_path, monkeypatch):
        monkeypatch.setenv(nl.NO_LEARNING_ENV, "0")
        _provider(ORDINARY_IDS[0], tmp_path).on_session_end(LEARNABLE)
        assert _count(rag) == 1

    def test_malformed_env_stores_nothing(self, rag, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv(nl.NO_LEARNING_ENV, "please")
        with caplog.at_level(logging.INFO):
            _provider(ORDINARY_IDS[0], tmp_path).on_session_end(LEARNABLE)
        assert _count(rag) == 0
        assert any("unrecognised value" in r.getMessage() for r in caplog.records)
        assert len(_skip_logs(caplog)) == 1

    def test_every_trigger_path_is_covered(self, rag, tmp_path):
        """CLI exit, /new and compression all reach the hook via MemoryManager.on_session_end;
        API finalize via commit_memory_session, which calls the same method."""
        from agent.memory_manager import MemoryManager
        mm = MemoryManager()
        mm.add_provider(_provider(NOLEARN_ID, tmp_path))
        mm.on_session_end(LEARNABLE)                       # shutdown_memory_provider / commit_memory_session
        mm.commit_session_boundary_async(LEARNABLE, new_session_id="nolearn_next")  # /new
        if hasattr(mm, "flush_pending"):
            mm.flush_pending(timeout=10)
        assert _count(rag) == 0


# ---------------------------------------------------------------------------
# Explicit stores
# ---------------------------------------------------------------------------


class TestExplicitStores:
    def test_store_refused_in_prefix_session(self, rag, tmp_path, caplog):
        p = _provider(NOLEARN_ID, tmp_path)
        with caplog.at_level(logging.INFO, logger="plugins.memory.jarvis"):
            r = json.loads(p.handle_tool_call("hermes_rag_store", {"text": "The user prefers tea.",
                                                                   "provenance": {"source": "conversation"}}))
        assert r["ok"] is False and r["code"] == "learning_disabled" and "Nothing was stored" in r["error"]
        assert _count(rag) == 0
        assert len(_skip_logs(caplog)) == 1

    def test_store_learnings_refused_for_current_or_target_nolearn_session(self, rag, tmp_path):
        p = _provider(NOLEARN_ID, tmp_path)
        r1 = json.loads(p.handle_tool_call("hermes_rag_store_learnings", {"session_id": ORDINARY_IDS[2], "messages": LEARNABLE}))
        q = _provider(ORDINARY_IDS[2], tmp_path)
        r2 = json.loads(q.handle_tool_call("hermes_rag_store_learnings", {"session_id": NOLEARN_ID, "messages": LEARNABLE}))
        assert r1["code"] == r2["code"] == "learning_disabled"
        assert _count(rag) == 0

    def test_store_refused_under_env_switch(self, rag, tmp_path, monkeypatch):
        monkeypatch.setenv(nl.NO_LEARNING_ENV, "true")
        r = json.loads(_provider(ORDINARY_IDS[2], tmp_path).handle_tool_call("hermes_rag_store", {"text": "x y z"}))
        assert r["code"] == "learning_disabled"
        assert _count(rag) == 0

    def test_ordinary_session_store_and_priority1_validation_unchanged(self, rag, tmp_path):
        p = _provider(ORDINARY_IDS[2], tmp_path)
        bad = json.loads(p.handle_tool_call("hermes_rag_store", {"text": "t", "provenance": {"source": "user_input"}}))
        assert bad["code"] == "invalid_provenance_source"
        ok = json.loads(p.handle_tool_call("hermes_rag_store", {"text": "The user prefers tea.", "provenance": {"source": "conversation"}}))
        assert ok["ok"] is True
        [point] = rag.client.scroll(collection_name=rag.COLLECTION_NAME, limit=5, with_payload=True)[0]
        assert point.payload["provenance"]["session_id"] == ORDINARY_IDS[2]

    def test_no_learning_is_checked_before_provenance(self, rag, tmp_path):
        r = json.loads(_provider(NOLEARN_ID, tmp_path).handle_tool_call(
            "hermes_rag_store", {"text": "t", "provenance": {"source": "user_input"}}))
        assert r["code"] == "learning_disabled"

    def test_module_dispatcher_enforces_env_switch(self, monkeypatch):
        from plugins.memory.jarvis import _dispatch_jarvis_tool_call
        with patch("rag_memory.store_memory_with_provenance", return_value="id") as store:
            assert json.loads(_dispatch_jarvis_tool_call("hermes_rag_store", {"text": "t"}))["ok"] is True
            monkeypatch.setenv(nl.NO_LEARNING_ENV, "1")
            assert json.loads(_dispatch_jarvis_tool_call("hermes_rag_store", {"text": "t"}))["code"] == "learning_disabled"
        assert store.call_count == 1


# ---------------------------------------------------------------------------
# Delegated sub-agents
# ---------------------------------------------------------------------------


def _attach(parent_session_id, tmp_path, parent_extra=None):
    from agent.memory_manager import MemoryManager
    from tools.delegate_tool import _attach_jarvis_subagent_provider
    parent_mm = MemoryManager()
    parent_mm.add_provider(_provider(parent_session_id, tmp_path))
    parent = SimpleNamespace(session_id=parent_session_id, _memory_manager=parent_mm, **(parent_extra or {}))
    child = SimpleNamespace(session_id="20261011_120000_c0ffee00", _memory_manager=None, tools=[], valid_tool_names=set())
    _attach_jarvis_subagent_provider(parent, child, "summarise notes", depth=1)
    providers = [p for p in child._memory_manager._providers if p.__class__.__name__ == "JARVISSubagentProvider"]
    assert len(providers) == 1
    return child, providers[0]


class TestDelegatedSubagents:
    def test_subagent_of_nolearn_session_inherits_and_learns_nothing(self, tmp_path, caplog):
        child, sub = _attach(NOLEARN_ID, tmp_path)
        assert child._no_learning_reason == sub._no_learning_reason == "session id starts with 'nolearn_'"
        with patch("plugins.memory.jarvis._on_session_end_learning") as learn, \
             patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as dispatch, \
             caplog.at_level(logging.INFO, logger="plugins.memory.jarvis"):
            sub.on_session_end(LEARNABLE)
            r = json.loads(sub.handle_tool_call("hermes_rag_store", {"text": "t"}))
        learn.assert_not_called()
        dispatch.assert_not_called()
        # Decision 42: the delegated read-only refusal now applies to every
        # sub-agent store, so it is the reported code (was learning_disabled).
        assert r["code"] == "delegated_read_only"
        assert len(_skip_logs(caplog, "session-end learning skipped")) == 1

    def test_nested_delegation_inherits_from_child(self, tmp_path):
        _, sub = _attach(ORDINARY_IDS[0], tmp_path, {"_no_learning_reason": "session id starts with 'nolearn_'"})
        assert sub._no_learning_reason == "session id starts with 'nolearn_'"

    def test_subagent_of_ordinary_session_learns_nothing(self, tmp_path):
        """Decision 42 inverts the former expectation: a sub-agent of an
        ordinary session no longer learns at session end either."""
        child, sub = _attach(ORDINARY_IDS[0], tmp_path)
        assert child._no_learning_reason is None and sub._no_learning_reason is None
        with patch("plugins.memory.jarvis._on_session_end_learning") as learn:
            sub.on_session_end(LEARNABLE)
        learn.assert_not_called()

    def test_subagent_learning_function_enforces_env_switch(self, monkeypatch, caplog):
        from plugins.memory.jarvis import _on_session_end_learning
        monkeypatch.setenv(nl.NO_LEARNING_ENV, "1")
        with patch("rag_memory.store_session_learnings") as store, \
             caplog.at_level(logging.INFO, logger="plugins.memory.jarvis"):
            _on_session_end_learning(LEARNABLE)
        store.assert_not_called()
        assert len(_skip_logs(caplog)) == 1
