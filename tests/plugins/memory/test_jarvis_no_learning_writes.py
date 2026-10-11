"""No-learning sessions block every persistent memory write path (follow-up to Priority 2).

The 2026-10-11 write-path audit found writes the original no-learning control
did not cover. For a session whose id starts with ``nolearn_`` -- or any session
in a process running with HERMES_NO_LEARNING -- these must now be refused with a
structured ``learning_disabled`` result, while read-only operations and ordinary
sessions behave exactly as before:

1. background memory review (every spawn path funnels through
   AIAgent._spawn_background_review)
2. built-in memory tool writes to MEMORY.md / USER.md, single and batch, on both
   agent-loop dispatch paths (invoke_tool and the sequential executor)
3. continuity writes (create / update / complete / record / supersede)
4. hermes_rag_delete
5. hermes_rag_rollback_learnings, refused before any approval is requested

Storage is always disposable: memories in tmp_path (get_memory_dir patched),
continuity.db under a tmp hermes_home, Qdrant as a tmp_path local client with
the production JARVIS_QDRANT_* / JARVIS_EMBED_* variables removed.
"""

import json
import os
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.expanduser("~/.hermes/rag"))

import agent.no_learning as nl  # noqa: E402
import tools.memory_tool as mt  # noqa: E402
from tools.memory_tool import ENTRY_DELIMITER, MemoryStore  # noqa: E402

PRODUCTION_ENV = ("JARVIS_QDRANT_URL", "JARVIS_QDRANT_PATH", "JARVIS_QDRANT_TIMEOUT",
                  "JARVIS_EMBED_SOCKET", "JARVIS_EMBED_TIMEOUT", "JARVIS_EMBED_FALLBACK")
NOLEARN_ID = "nolearn_desktop_1791713225_5fd242e4"
ORDINARY_ID = "desktop_1791706201_2f8b485a"
EXISTING = ["User prefers concise answers.", "Project codename is BLUE-ORBIT."]

# (session id, env value) -> protected?
PROTECTED_CASES = [
    pytest.param(NOLEARN_ID, None, id="nolearn-prefix"),
    pytest.param(ORDINARY_ID, "1", id="env-switch"),
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in PRODUCTION_ENV + (nl.NO_LEARNING_ENV,):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(nl, "_warned_values", set())


def _protect(monkeypatch, env_value):
    if env_value is not None:
        monkeypatch.setenv(nl.NO_LEARNING_ENV, env_value)


def _refusal(raw):
    data = json.loads(raw)
    assert data["code"] == "learning_disabled", data
    assert data["ok"] is False and data["success"] is False
    assert "Persistent learning is disabled" in data["error"]
    return data


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


class TestAgentHelper:
    def test_inherited_decision_wins(self):
        agent = SimpleNamespace(session_id=ORDINARY_ID, _no_learning_reason="parent says no")
        assert nl.agent_learning_disabled_reason(agent) == "parent says no"

    def test_prefix_env_and_ordinary(self, monkeypatch):
        assert "nolearn_" in nl.agent_learning_disabled_reason(SimpleNamespace(session_id=NOLEARN_ID))
        assert nl.agent_learning_disabled_reason(SimpleNamespace(session_id=ORDINARY_ID)) is None
        assert nl.agent_learning_disabled_reason(SimpleNamespace()) is None
        monkeypatch.setenv(nl.NO_LEARNING_ENV, "yes")
        assert nl.NO_LEARNING_ENV in nl.agent_learning_disabled_reason(SimpleNamespace(session_id=ORDINARY_ID))

    def test_refusal_shape_and_log(self, caplog):
        caplog.set_level("INFO", logger="agent.no_learning")
        data = _refusal(nl.learning_disabled_refusal("hermes_x", "why", session_id=NOLEARN_ID, outcome="Done."))
        assert data["operation"] == "hermes_x"
        assert data["error"].endswith("(why). Done.")
        assert any("hermes_x refused for session " + NOLEARN_ID in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 1. Background memory review
# ---------------------------------------------------------------------------


class _Spawn:
    """Captures spawn_background_review_thread calls and the started thread."""

    def __init__(self, monkeypatch):
        import agent.background_review as br
        import run_agent
        self.calls = []
        self.threads = []
        monkeypatch.setattr(br, "spawn_background_review_thread", self._spawn)
        monkeypatch.setattr(run_agent.threading, "Thread", self._thread)

    def _spawn(self, agent, snapshot, *, review_memory, review_skills, focus=None):
        self.calls.append({"review_memory": review_memory, "review_skills": review_skills, "focus": focus})
        return (lambda: None), "prompt"

    def _thread(self, *, target, daemon=None, name=None):
        thread = MagicMock()
        self.threads.append(thread)
        return thread


def _bare_agent(session_id):
    from run_agent import AIAgent
    agent = object.__new__(AIAgent)
    agent.session_id = session_id
    return agent


class TestBackgroundReview:
    @pytest.mark.parametrize("session_id,env", PROTECTED_CASES)
    def test_memory_review_never_spawns(self, monkeypatch, caplog, session_id, env):
        _protect(monkeypatch, env)
        spawn = _Spawn(monkeypatch)
        caplog.set_level("INFO", logger="run_agent")
        _bare_agent(session_id)._spawn_background_review([], review_memory=True, review_skills=False)
        assert spawn.calls == [] and spawn.threads == []
        assert any("Background memory review skipped" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("session_id,env", PROTECTED_CASES)
    def test_refine_style_memory_and_skills_keeps_only_skills(self, monkeypatch, session_id, env):
        _protect(monkeypatch, env)
        spawn = _Spawn(monkeypatch)
        _bare_agent(session_id)._spawn_background_review([], review_memory=True, review_skills=True, focus="x")
        assert spawn.calls == [{"review_memory": False, "review_skills": True, "focus": "x"}]
        assert len(spawn.threads) == 1

    def test_inherited_decision_blocks_review(self, monkeypatch):
        spawn = _Spawn(monkeypatch)
        agent = _bare_agent(ORDINARY_ID)
        agent._no_learning_reason = "sub-agent of a no-learning session"
        agent._spawn_background_review([], review_memory=True)
        assert spawn.calls == []

    def test_ordinary_session_unchanged(self, monkeypatch):
        spawn = _Spawn(monkeypatch)
        _bare_agent(ORDINARY_ID)._spawn_background_review([], review_memory=True, review_skills=False)
        assert spawn.calls == [{"review_memory": True, "review_skills": False, "focus": None}]
        spawn.threads[0].start.assert_called_once()

    def test_skills_only_review_unaffected_when_protected(self, monkeypatch):
        spawn = _Spawn(monkeypatch)
        _bare_agent(NOLEARN_ID)._spawn_background_review([], review_memory=False, review_skills=True)
        assert spawn.calls == [{"review_memory": False, "review_skills": True, "focus": None}]

    def test_finalize_turn_reaches_the_guarded_method(self):
        # turn_finalizer / codex_runtime / both /refine commands call this
        # method rather than spawn_background_review_thread directly.
        import inspect
        import agent.codex_runtime as cr
        import agent.turn_finalizer as tf
        for module in (tf, cr):
            source = inspect.getsource(module)
            assert "agent._spawn_background_review(" in source
            assert "spawn_background_review_thread(" not in source

    def test_refine_commands_reach_the_guarded_method(self):
        # The gateway and CLI /refine handlers must spawn the review through
        # AIAgent._spawn_background_review (where the no-learning guard lives),
        # never the thread helpers underneath it.
        import inspect
        from gateway.slash_commands import GatewaySlashCommandsMixin
        from hermes_cli.cli_commands_mixin import CLICommandsMixin
        for handler in (GatewaySlashCommandsMixin._handle_refine_command,
                        CLICommandsMixin._handle_refine_command):
            source = inspect.getsource(handler)
            assert "agent._spawn_background_review(" in source, handler.__qualname__
            assert "spawn_background_review_thread(" not in source, handler.__qualname__
            assert "_run_review_in_thread(" not in source, handler.__qualname__


# ---------------------------------------------------------------------------
# 2. Built-in memory tool (MEMORY.md / USER.md)
# ---------------------------------------------------------------------------


@pytest.fixture
def mem_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    for name in ("MEMORY.md", "USER.md"):
        (tmp_path / name).write_text(ENTRY_DELIMITER.join(EXISTING), encoding="utf-8")
    return tmp_path


def _snapshot(mem_dir):
    return {name: (mem_dir / name).read_bytes() for name in ("MEMORY.md", "USER.md")}


def _store():
    store = MemoryStore()
    store.load_from_disk()
    return store


MEMORY_CALLS = [
    pytest.param({"action": "add", "target": "memory", "content": "New fact."}, id="add-memory"),
    pytest.param({"action": "add", "target": "user", "content": "New user fact."}, id="add-user"),
    pytest.param({"action": "replace", "target": "memory", "old_text": "BLUE-ORBIT", "content": "Codename is RED."}, id="replace"),
    pytest.param({"action": "remove", "target": "user", "old_text": "concise"}, id="remove"),
    pytest.param({"target": "memory", "operations": [
        {"action": "remove", "old_text": "concise"},
        {"action": "add", "content": "Batch fact."},
    ]}, id="batch"),
    pytest.param({"target": "user", "operations": [
        {"action": "remove", "old_text": "concise"},
        {"action": "add", "content": "Batch user fact."},
    ]}, id="batch-user"),
]


class TestMemoryToolFunction:
    @pytest.mark.parametrize("call", MEMORY_CALLS)
    def test_refused_and_files_untouched(self, mem_dir, call):
        before = _snapshot(mem_dir)
        data = _refusal(mt.memory_tool(store=_store(), no_learning_reason="session id starts with 'nolearn_'", **call))
        assert data["operation"] == ("memory (batch)" if "operations" in call else "memory")
        assert "MEMORY.md and USER.md were not changed" in data["error"]
        assert _snapshot(mem_dir) == before

    def test_refusal_is_not_mirrored_to_providers(self):
        from agent.memory_manager import MemoryManager
        raw = mt.memory_tool(action="add", content="x", store=MagicMock(), no_learning_reason="r")
        assert MemoryManager._memory_tool_result_succeeded(raw) is False

    def test_refusal_bypasses_the_write_gate(self, mem_dir, monkeypatch):
        gate = MagicMock(return_value=None)
        monkeypatch.setattr(mt, "_apply_write_gate", gate)
        monkeypatch.setattr(mt, "_apply_batch_write_gate", gate)
        mt.memory_tool(action="add", content="x", store=_store(), no_learning_reason="r")
        mt.memory_tool(operations=[{"action": "add", "content": "x"}], store=_store(), no_learning_reason="r")
        gate.assert_not_called()  # nothing is staged for later approval either

    def test_no_reason_behaves_as_before(self, mem_dir):
        result = json.loads(mt.memory_tool(action="add", content="New fact.", store=_store()))
        assert result["success"] is True
        assert "New fact." in (mem_dir / "MEMORY.md").read_text(encoding="utf-8")


def _memory_tool_def():
    return [{"type": "function", "function": {
        "name": "memory", "description": "memory",
        "parameters": {"type": "object", "properties": {}},
    }}]


@pytest.fixture
def loop_agent(mem_dir):
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=_memory_tool_def()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._memory_store = _store()
    agent._memory_manager = MagicMock()
    return agent


def _run_sequential(agent, args):
    call = SimpleNamespace(id=f"call_{uuid.uuid4().hex[:8]}", type="function",
                           function=SimpleNamespace(name="memory", arguments=json.dumps(args)))
    messages = []
    agent._execute_tool_calls_sequential(SimpleNamespace(content="", tool_calls=[call]), messages, "task-1")
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    return tool_messages[0]["content"]


DISPATCHERS = [
    pytest.param(lambda agent, args: agent._invoke_tool("memory", args, "task-1"), id="invoke_tool"),
    pytest.param(_run_sequential, id="sequential"),
]


class TestMemoryToolAgentLoop:
    @pytest.mark.parametrize("dispatch", DISPATCHERS)
    @pytest.mark.parametrize("session_id,env", PROTECTED_CASES)
    @pytest.mark.parametrize("args", [
        {"action": "add", "target": "user", "content": "New user fact."},
        {"target": "memory", "operations": [{"action": "add", "content": "Batch fact."}]},
        {"target": "user", "operations": [{"action": "add", "content": "Batch user fact."},
                                          {"action": "remove", "old_text": "concise"}]},
    ], ids=["single", "batch", "batch-user"])
    def test_protected_session_refused(self, loop_agent, mem_dir, monkeypatch, dispatch, session_id, env, args):
        _protect(monkeypatch, env)
        loop_agent.session_id = session_id
        before = _snapshot(mem_dir)
        _refusal(dispatch(loop_agent, args))
        assert _snapshot(mem_dir) == before
        loop_agent._memory_manager.on_memory_write.assert_not_called()

    @pytest.mark.parametrize("dispatch", DISPATCHERS)
    def test_inherited_decision_refused(self, loop_agent, mem_dir, dispatch):
        loop_agent.session_id = ORDINARY_ID
        loop_agent._no_learning_reason = "sub-agent of a no-learning session"
        before = _snapshot(mem_dir)
        _refusal(dispatch(loop_agent, {"action": "add", "content": "x"}))
        assert _snapshot(mem_dir) == before

    @pytest.mark.parametrize("dispatch", DISPATCHERS)
    def test_ordinary_session_writes(self, loop_agent, mem_dir, dispatch):
        loop_agent.session_id = ORDINARY_ID
        result = json.loads(dispatch(loop_agent, {"action": "add", "content": "Ordinary fact."}))
        assert result["success"] is True
        assert "Ordinary fact." in (mem_dir / "MEMORY.md").read_text(encoding="utf-8")


def _tool_call(name, args=None):
    return SimpleNamespace(id=f"call_{uuid.uuid4().hex[:8]}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))


class TestNoConcurrentDispatch:
    """The memory tool and the JARVIS provider tools are never batched onto the
    concurrent executor: they are barriers, so they always run on the guarded
    sequential path (agent/tool_dispatch_helpers.py)."""

    def _provider_tool_names(self):
        from plugins.memory.jarvis import HermesRAGProvider
        names = {schema["name"] for schema in HermesRAGProvider().get_tool_schemas()}
        assert names and all(name.startswith("hermes_") for name in names)
        return names

    def test_memory_and_hermes_tools_are_not_parallel_safe(self):
        from agent.tool_dispatch_helpers import _PARALLEL_SAFE_TOOLS
        assert "memory" not in _PARALLEL_SAFE_TOOLS
        assert not {name for name in _PARALLEL_SAFE_TOOLS if name.startswith("hermes_")}
        assert not self._provider_tool_names() & _PARALLEL_SAFE_TOOLS

    def test_planner_keeps_memory_and_hermes_calls_off_parallel_segments(self):
        from agent.tool_dispatch_helpers import _plan_tool_batch_segments
        guarded = {"memory"} | self._provider_tool_names()
        calls = [
            _tool_call("web_search", {"query": "a"}),
            _tool_call("web_search", {"query": "b"}),          # control: a parallel pair
            _tool_call("memory", {"action": "add", "content": "x"}),
            _tool_call("web_search", {"query": "c"}),
            _tool_call("hermes_rag_delete", {"memory_id": "x"}),
            _tool_call("hermes_goal_create", {"title": "x"}),
            _tool_call("hermes_rag_store", {"text": "x"}),
            _tool_call("web_search", {"query": "d"}),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert ("parallel", [calls[0], calls[1]]) in [(kind, list(seg)) for kind, seg in segments]
        for kind, seg in segments:
            if kind == "parallel":
                assert not {c.function.name for c in seg} & guarded, segments
        placed = [c for _, seg in segments for c in seg]
        assert placed == calls  # order preserved, nothing dropped


# ---------------------------------------------------------------------------
# 3. Continuity writes
# ---------------------------------------------------------------------------


def _provider(session_id, tmp_path):
    from plugins.memory.jarvis import HermesRAGProvider
    provider = HermesRAGProvider()
    provider.initialize(session_id, hermes_home=str(tmp_path / "hermes_home"), agent_context="primary")
    return provider


def _call(provider, tool_name, args, **kwargs):
    return json.loads(provider.handle_tool_call(tool_name, args, **kwargs))


def _seed(tmp_path):
    """One goal, task and decision, written through an ordinary session."""
    ordinary = _provider(ORDINARY_ID, tmp_path)
    goal = _call(ordinary, "hermes_goal_create", {"title": "Seed goal"})["goal"]["id"]
    task = _call(ordinary, "hermes_task_create", {"title": "Seed task"})["task"]["id"]
    decision = _call(ordinary, "hermes_decision_record", {"decision": "Seed decision"})["decision"]["id"]
    return {"goal": goal, "task": task, "decision": decision}


def _continuity_rows(tmp_path):
    import sqlite3
    db = tmp_path / "hermes_home" / "continuity.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {t: conn.execute(f"select * from {t} order by id").fetchall() for t in ("goals", "tasks", "decisions")}
    finally:
        conn.close()


def _write_calls(ids):
    return {
        "hermes_goal_create": {"title": "Blocked goal"},
        "hermes_goal_update": {"goal_id": ids["goal"], "title": "Changed"},
        "hermes_goal_complete": {"goal_id": ids["goal"]},
        "hermes_task_create": {"title": "Blocked task"},
        "hermes_task_update": {"task_id": ids["task"], "status": "in_progress"},
        "hermes_task_complete": {"task_id": ids["task"]},
        "hermes_decision_record": {"decision": "Blocked decision"},
        "hermes_decision_supersede": {"decision_id": ids["decision"], "decision": "Replacement"},
    }


class TestContinuity:
    def test_write_set_is_every_non_list_tool(self):
        from plugins.memory.jarvis import _CONTINUITY_READ_ONLY_TOOL_NAMES, _CONTINUITY_TOOL_NAMES
        writes = _CONTINUITY_TOOL_NAMES - _CONTINUITY_READ_ONLY_TOOL_NAMES
        assert writes == set(_write_calls({"goal": "", "task": "", "decision": ""}))
        assert all(name.endswith("_list") for name in _CONTINUITY_READ_ONLY_TOOL_NAMES)

    @pytest.mark.parametrize("session_id,env", PROTECTED_CASES)
    def test_all_writes_refused_and_db_unchanged(self, tmp_path, monkeypatch, session_id, env):
        ids = _seed(tmp_path)
        before = _continuity_rows(tmp_path)
        _protect(monkeypatch, env)
        protected = _provider(session_id, tmp_path)
        for tool_name, args in _write_calls(ids).items():
            data = json.loads(protected.handle_tool_call(tool_name, args))
            assert data["code"] == "learning_disabled", (tool_name, data)
            assert data["operation"] == tool_name
            assert "No continuity record was created or changed" in data["error"]
        assert _continuity_rows(tmp_path) == before

    @pytest.mark.parametrize("session_id,env", PROTECTED_CASES)
    def test_list_tools_still_work(self, tmp_path, monkeypatch, session_id, env):
        ids = _seed(tmp_path)
        _protect(monkeypatch, env)
        protected = _provider(session_id, tmp_path)
        assert [g["id"] for g in _call(protected, "hermes_goal_list", {})["goals"]] == [ids["goal"]]
        assert [t["id"] for t in _call(protected, "hermes_task_list", {})["tasks"]] == [ids["task"]]
        assert [d["id"] for d in _call(protected, "hermes_decision_list", {})["decisions"]] == [ids["decision"]]

    def test_refused_before_the_approval_gate_stages_anything(self, tmp_path):
        with patch("tools.write_approval.evaluate_gate") as gate, \
             patch("tools.write_approval.stage_write") as stage:
            data = _call(_provider(NOLEARN_ID, tmp_path), "hermes_goal_create", {"title": "x"})
        assert data["code"] == "learning_disabled"
        gate.assert_not_called()
        stage.assert_not_called()

    def test_approved_replay_is_exempt(self, tmp_path):
        # apply_continuity_pending replays a user-approved write with the
        # system-only bypass flag; the no-learning guard does not apply to it.
        data = _call(_provider(NOLEARN_ID, tmp_path), "hermes_goal_create", {"title": "Approved"},
                     _continuity_bypass_gate=True)
        assert data["ok"] is True

    def test_ordinary_session_writes(self, tmp_path):
        ids = _seed(tmp_path)
        ordinary = _provider(ORDINARY_ID, tmp_path)
        for tool_name, args in _write_calls(ids).items():
            assert _call(ordinary, tool_name, args)["ok"] is True, tool_name


# ---------------------------------------------------------------------------
# 4-5. Qdrant delete and rollback
# ---------------------------------------------------------------------------


@pytest.fixture
def rag(monkeypatch, tmp_path):
    import rag_memory as rm
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection(collection_name=rm.COLLECTION_NAME,
                             vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    point_id = str(uuid.uuid4())
    client.upsert(collection_name=rm.COLLECTION_NAME, points=[PointStruct(
        id=point_id, vector=[0.1, 0.2, 0.3, 0.4],
        payload={"text": "seed", "type": "memory", "provenance": {"origin": "cross_session_transfer"}},
    )])
    monkeypatch.setattr(rm, "QDRANT_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(rm, "client", client)
    rm.seed_id = point_id
    yield rm
    client.close()


def _ids(rm):
    points, _ = rm.client.scroll(collection_name=rm.COLLECTION_NAME, limit=100, with_payload=False)
    return sorted(str(p.id) for p in points)


class TestRagDeleteAndRollback:
    @pytest.mark.parametrize("session_id,env", PROTECTED_CASES)
    def test_delete_refused(self, rag, tmp_path, monkeypatch, session_id, env):
        _protect(monkeypatch, env)
        before = _ids(rag)
        data = _refusal(_provider(session_id, tmp_path).handle_tool_call("hermes_rag_delete", {"memory_id": rag.seed_id}))
        assert data["operation"] == "hermes_rag_delete"
        assert "No memory was deleted" in data["error"]
        assert _ids(rag) == before

    def test_delete_still_validates_input_first(self, rag, tmp_path):
        data = _call(_provider(NOLEARN_ID, tmp_path), "hermes_rag_delete", {})
        assert data == {"ok": False, "error": "memory_id is required"}

    @pytest.mark.parametrize("session_id,env", PROTECTED_CASES)
    def test_rollback_refused_before_approval(self, rag, tmp_path, monkeypatch, session_id, env):
        _protect(monkeypatch, env)
        before = _ids(rag)
        with patch("plugins.memory.jarvis.request_tool_approval") as approval, \
             patch.object(rag, "rollback_cross_session_learnings") as rollback:
            data = _refusal(_provider(session_id, tmp_path).handle_tool_call("hermes_rag_rollback_learnings", {}))
        assert "no approval was requested" in data["error"]
        approval.assert_not_called()
        rollback.assert_not_called()
        assert _ids(rag) == before

    def test_search_still_works_when_protected(self, rag, tmp_path, monkeypatch):
        monkeypatch.setattr(rag, "search_memory", lambda query, limit=5: [{"score": 1.0, "text": "seed"}])
        data = _call(_provider(NOLEARN_ID, tmp_path), "hermes_rag_search", {"query": "seed"})
        assert data == {"ok": True, "results": [{"score": 1.0, "text": "seed"}]}

    def test_ordinary_delete_unchanged(self, rag, tmp_path):
        data = _call(_provider(ORDINARY_ID, tmp_path), "hermes_rag_delete", {"memory_id": rag.seed_id})
        assert data.get("ok") is True, data
        assert _ids(rag) == []

    def test_ordinary_rollback_still_requires_approval(self, rag, tmp_path, monkeypatch):
        # The rollback branch checks _get_client() before approval; keep it on
        # the tmp client so this test never resolves the configured store.
        monkeypatch.setattr(rag, "_get_client", lambda: rag.client)
        with patch("plugins.memory.jarvis.request_tool_approval",
                   return_value={"approved": False}) as approval:
            data = _call(_provider(ORDINARY_ID, tmp_path), "hermes_rag_rollback_learnings", {})
        approval.assert_called_once()
        assert data["status"] == "approval_required"
        assert _ids(rag) == [rag.seed_id]
