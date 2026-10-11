"""Compression rotation keeps a no-learning session protected.

Every no-learning guard (agent/no_learning.py) keys on the session id. When
_compress_context rotates a session into a continuation child, the child id is
minted fresh; before this fix it was always ``<YYYYmmdd_HHMMSS>_<6 hex>``, so a
long ``nolearn_`` conversation silently became learnable after its first
compression. The child of a ``nolearn_`` parent must keep the prefix (and so
every guard), while ordinary rotation stays byte-for-byte the same format.

All storage is disposable: a tmp SessionDB, a tmp memory dir, a tmp
continuity.db. HERMES_NO_LEARNING is removed so only the prefix is under test.
"""

import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent.no_learning as nl
import tools.memory_tool as mt
from agent.conversation_compression import (
    _compression_child_session_id,
    recover_rotated_compression_session,
)

ORDINARY_CHILD = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")
NOLEARN_CHILD = re.compile(r"^nolearn_\d{8}_\d{6}_[0-9a-f]{6}$")
NOLEARN_PARENT = "nolearn_desktop_1791713225_5fd242e4"
ORDINARY_PARENTS = ["desktop_1791706201_2f8b485a", "20261011_111046_2f8b485a",
                    "api-3750560e0646c3a2", "desktop_nolearn_123", "NOLEARN_x"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(nl.NO_LEARNING_ENV, raising=False)
    monkeypatch.setattr(nl, "_warned_values", set())


# ---------------------------------------------------------------------------
# The id helper
# ---------------------------------------------------------------------------


class TestChildIdHelper:
    def test_nolearn_parent_keeps_prefix(self):
        assert NOLEARN_CHILD.match(_compression_child_session_id(NOLEARN_PARENT))

    def test_repeated_compression_does_not_stack_prefixes(self):
        first = _compression_child_session_id(NOLEARN_PARENT)
        second = _compression_child_session_id(first)
        assert NOLEARN_CHILD.match(second)
        assert second.count(nl.NO_LEARNING_PREFIX) == 1

    @pytest.mark.parametrize("parent", ORDINARY_PARENTS + [None, ""])
    def test_ordinary_format_unchanged(self, parent):
        assert ORDINARY_CHILD.match(_compression_child_session_id(parent))

    def test_ids_are_unique(self):
        ids = {_compression_child_session_id(NOLEARN_PARENT) for _ in range(200)}
        assert len(ids) == 200


# ---------------------------------------------------------------------------
# A real rotation through _compress_context
# ---------------------------------------------------------------------------


def _agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = False  # force the rotation path
    return agent


def _rotate(agent):
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail question"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    agent.context_compressor = compressor
    parent = agent.session_id
    agent._compress_context([{"role": "user", "content": f"m{i}"} for i in range(10)], "sys", approx_tokens=10_000)
    assert agent.session_id != parent, "rotation did not happen"
    return parent, agent.session_id


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    close = getattr(session_db, "close", None)
    if callable(close):
        close()


class TestRotation:
    def test_nolearn_session_child_keeps_prefix(self, db):
        agent = _agent(db, NOLEARN_PARENT)
        parent, child = _rotate(agent)
        assert parent == NOLEARN_PARENT
        assert NOLEARN_CHILD.match(child)
        row = db.get_session(child)
        assert row is not None and row["parent_session_id"] == NOLEARN_PARENT

    def test_second_rotation_still_protected(self, db):
        agent = _agent(db, NOLEARN_PARENT)
        _, first = _rotate(agent)
        _, second = _rotate(agent)
        assert NOLEARN_CHILD.match(first) and NOLEARN_CHILD.match(second)

    @pytest.mark.parametrize("parent", ["desktop_1791706201_2f8b485a", "20261011_111046_2f8b485a"])
    def test_ordinary_rotation_unchanged(self, db, parent):
        agent = _agent(db, parent)
        _, child = _rotate(agent)
        assert ORDINARY_CHILD.match(child)
        assert nl.agent_learning_disabled_reason(agent) is None

    def test_next_turn_adopts_the_protected_child(self, db):
        # A later request still names the original nolearn_ id; turn start
        # adopts the live compression child (turn_context.py) -- which must
        # still carry the prefix.
        first = _agent(db, NOLEARN_PARENT)
        _, child = _rotate(first)
        later = _agent(db, NOLEARN_PARENT)
        recover_rotated_compression_session(later)
        assert later.session_id == child
        assert nl.agent_learning_disabled_reason(later)


# ---------------------------------------------------------------------------
# The rotated child is still covered by the existing guards
# ---------------------------------------------------------------------------


class TestChildStaysProtected:
    def test_memory_tool_refused_after_rotation(self, db, tmp_path, monkeypatch):
        mem_dir = tmp_path / "memories"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("Existing entry.", encoding="utf-8")
        monkeypatch.setattr(mt, "get_memory_dir", lambda: mem_dir)
        agent = _agent(db, NOLEARN_PARENT)
        _rotate(agent)
        store = mt.MemoryStore()
        store.load_from_disk()
        agent._memory_store = store
        before = (mem_dir / "MEMORY.md").read_bytes()
        result = json.loads(agent._invoke_tool("memory", {"action": "add", "content": "Learned."}, "task-1"))
        assert result["code"] == "learning_disabled"
        assert (mem_dir / "MEMORY.md").read_bytes() == before

    def test_background_memory_review_skipped_after_rotation(self, db, monkeypatch):
        import agent.background_review as br
        spawned = []
        monkeypatch.setattr(br, "spawn_background_review_thread",
                            lambda *a, **k: spawned.append(k) or ((lambda: None), ""))
        agent = _agent(db, NOLEARN_PARENT)
        _rotate(agent)
        agent._spawn_background_review([], review_memory=True, review_skills=False)
        assert spawned == []

    def test_provider_guards_follow_the_session_switch(self, db, tmp_path):
        # The memory manager forwards the rotation to the provider via
        # on_session_switch; the provider's guards then see the child id.
        from plugins.memory.jarvis import HermesRAGProvider
        agent = _agent(db, NOLEARN_PARENT)
        parent, child = _rotate(agent)
        provider = HermesRAGProvider()
        provider.initialize(parent, hermes_home=str(tmp_path / "hermes_home"), agent_context="primary")
        provider.on_session_switch(child, parent_session_id=parent, reset=False)
        assert provider._session_id == child
        # Continuity guards need no rag_memory import; the RAG store/delete
        # refusals for the same session id are covered with an isolated
        # Qdrant in tests/plugins/memory/test_jarvis_no_learning*.py.
        for tool_name, args in (
            ("hermes_goal_create", {"title": "Blocked"}),
            ("hermes_task_create", {"title": "Blocked"}),
            ("hermes_decision_record", {"decision": "Blocked"}),
        ):
            data = json.loads(provider.handle_tool_call(tool_name, args))
            assert data["code"] == "learning_disabled", (tool_name, data)

    def test_ordinary_child_provider_not_blocked(self, db, tmp_path):
        from plugins.memory.jarvis import HermesRAGProvider
        agent = _agent(db, "desktop_1791706201_2f8b485a")
        parent, child = _rotate(agent)
        provider = HermesRAGProvider()
        provider.initialize(parent, hermes_home=str(tmp_path / "hermes_home"), agent_context="primary")
        provider.on_session_switch(child, parent_session_id=parent, reset=False)
        data = json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Allowed"}))
        assert data["ok"] is True
