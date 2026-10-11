"""Decision 42 — delegated agents are read-only for persistent state.

A delegated (delegate_task) child may read approved information and return
results to its parent. It must not modify persistent memory (RAG, MEMORY.md /
USER.md, continuity, skills, capability plans), directly or through
terminal / process / code-execution / file-write tools, and it must not run
background reviews or session-end learning. Prohibited operations are refused
before any approval handling, so yolo mode, permanent approvals and subagent
auto-approval can never authorize them. The restriction is applied fail-closed
in ``_build_child_agent`` and holds in ordinary and no-learning sessions.

Every test here runs against mocks or the per-test temporary HERMES_HOME the
suite conftest provides; ``rag_memory`` is replaced by a fake module, so no
Qdrant (local or server) is touched.
"""

import json
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _build_child_agent

NOLEARN_ID = "nolearn_desktop_1791706201_2f8b485a"
ORDINARY_ID = "20261011_111046_2f8b485a"

# Tools a delegated child must never be able to call (Decision 42 policy 2).
WRITE_CAPABLE_TOOLS = ("terminal", "process", "execute_code", "write_file", "patch", "skill_manage")
READ_TOOLS = ("read_file", "search_files")

JARVIS_READ_TOOLS = frozenset({
    "hermes_rag_search",
    "hermes_rag_retrieve_learnings",
    "hermes_rag_extract_learnings",  # computes candidates only; stores nothing
    "hermes_goal_list",
    "hermes_task_list",
    "hermes_decision_list",
})
JARVIS_WRITE_TOOLS = (
    "hermes_rag_store",
    "hermes_rag_store_learnings",
    "hermes_rag_delete",
    "hermes_rag_rollback_learnings",
    "hermes_goal_create",
    "hermes_goal_update",
    "hermes_goal_complete",
    "hermes_task_create",
    "hermes_task_update",
    "hermes_task_complete",
    "hermes_decision_record",
    "hermes_decision_supersede",
)
RAG_WRITE_FUNCS = (
    "add_memory",
    "store_memory_with_provenance",
    "store_session_learnings",
    "delete_memory",
    "rollback_cross_session_learnings",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("HERMES_NO_LEARNING", "JARVIS_QDRANT_URL", "JARVIS_QDRANT_PATH", "JARVIS_EMBED_SOCKET"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_rag(monkeypatch):
    """A stand-in ``rag_memory`` module; every function is a MagicMock."""
    mod = types.ModuleType("rag_memory")
    for name in RAG_WRITE_FUNCS + ("search_memory", "extract_session_learnings",
                                   "retrieve_cross_session_learnings"):
        setattr(mod, name, MagicMock(name=name, return_value=[]))
    mod.PROVENANCE_SOURCES = frozenset({"conversation", "tool_result", "web", "user_confirmed"})
    monkeypatch.setitem(sys.modules, "rag_memory", mod)
    return mod


def _assert_no_rag_writes(mod):
    for name in RAG_WRITE_FUNCS:
        getattr(mod, name).assert_not_called()


class HermesRAGProvider:  # name-matched stand-in: the builder checks the class name
    pass


def _parent(session_id=ORDINARY_ID, *, jarvis=True, enabled=("hermes-cli",), **extra):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.acp_command = None
    parent.acp_args = []
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.session_id = session_id
    parent._no_learning_reason = None
    parent.enabled_toolsets = list(enabled)
    parent.disabled_toolsets = []
    mm = MagicMock()
    mm._providers = [HermesRAGProvider()] if jarvis else []
    parent._memory_manager = mm
    for key, value in extra.items():
        setattr(parent, key, value)
    return parent


def _build(parent, role="leaf", **kwargs):
    """Run the real builder with AIAgent mocked; return (child, AIAgent kwargs)."""
    with patch("run_agent.AIAgent") as MockAgent:
        child = SimpleNamespace(session_id="20261011_120000_c0ffee00", _memory_manager=None,
                                tools=[], valid_tool_names=set())
        MockAgent.return_value = child
        built = _build_child_agent(
            task_index=0, goal="Summarise the notes", context=None, toolsets=None,
            model=None, max_iterations=10, task_count=1, parent_agent=parent,
            role=role, **kwargs,
        )
        return built, MockAgent.call_args.kwargs


def _child_tool_names(agent_kwargs):
    import model_tools
    defs = model_tools.get_tool_definitions(
        enabled_toolsets=agent_kwargs["enabled_toolsets"],
        disabled_toolsets=agent_kwargs["disabled_toolsets"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    return {d["function"]["name"] for d in defs}


# ---------------------------------------------------------------------------
# Construction: fail-closed marker, no-learning inheritance, background review
# ---------------------------------------------------------------------------


class TestChildConstruction:
    @pytest.mark.parametrize("jarvis", [True, False])
    def test_child_is_marked_read_only(self, jarvis):
        child, _ = _build(_parent(jarvis=jarvis))
        assert getattr(child, "_delegated_read_only", None) is True

    def test_marker_and_no_learning_survive_attach_failure(self):
        """An exception while attaching the JARVIS provider must never yield an
        unrestricted child: either the build fails, or the child is marked."""
        parent = _parent(NOLEARN_ID)
        with patch("tools.delegate_tool._attach_jarvis_subagent_provider",
                   side_effect=RuntimeError("boom")):
            try:
                child, _ = _build(parent)
            except RuntimeError:
                return  # failed closed: no child was produced
        assert child._delegated_read_only is True
        assert child._no_learning_reason == "session id starts with 'nolearn_'"

    def test_no_learning_inherited_without_jarvis_provider(self):
        child, _ = _build(_parent(NOLEARN_ID, jarvis=False))
        assert child._no_learning_reason == "session id starts with 'nolearn_'"

    def test_no_learning_inherited_from_delegating_child(self):
        parent = _parent(ORDINARY_ID, jarvis=False, _no_learning_reason="inherited reason")
        child, _ = _build(parent)
        assert child._no_learning_reason == "inherited reason"

    def test_child_background_review_disabled(self):
        _, kwargs = _build(_parent())
        assert kwargs.get("skip_background_review") is True


# ---------------------------------------------------------------------------
# Tool surface: no write-capable tools, reads preserved, MCP/ACP denied
# ---------------------------------------------------------------------------


class TestChildToolSurface:
    @pytest.mark.parametrize("role", ["leaf", "orchestrator"])
    def test_write_capable_tools_removed(self, role):
        with patch("tools.delegate_tool._get_orchestrator_enabled", return_value=True), \
             patch("tools.delegate_tool._get_max_spawn_depth", return_value=2):
            _, kwargs = _build(_parent(), role=role)
        names = _child_tool_names(kwargs)
        assert not names & set(WRITE_CAPABLE_TOOLS), names & set(WRITE_CAPABLE_TOOLS)

    def test_read_tools_preserved(self):
        _, kwargs = _build(_parent())
        names = _child_tool_names(kwargs)
        assert set(READ_TOOLS) <= names

    def test_mcp_toolsets_not_inherited(self):
        _, kwargs = _build(_parent(enabled=("hermes-cli", "mcp-notes")))
        assert not [t for t in kwargs["enabled_toolsets"] if str(t).startswith("mcp-")]

    def test_acp_child_is_refused(self):
        parent = _parent(acp_command="copilot")
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with pytest.raises(Exception, match="(?i)acp"):
                _build(parent)


# ---------------------------------------------------------------------------
# Dispatch: write-capable tools refused before execution or approval
# ---------------------------------------------------------------------------


def _run_middleware(function_name, *, delegated=True):
    from agent.tool_executor import _run_agent_tool_execution_middleware

    agent = MagicMock()
    agent._delegated_read_only = delegated
    agent.session_id = ORDINARY_ID
    agent._current_turn_id = ""
    agent._current_api_request_id = ""
    agent._tool_guardrails.before_call.return_value = SimpleNamespace(allows_execution=True)
    execute = MagicMock(return_value=json.dumps({"ok": True}))
    managed = _run_agent_tool_execution_middleware(
        agent, function_name=function_name, function_args={}, effective_task_id="t",
        tool_call_id="c1", execute=execute,
    )
    return managed, execute


class TestDispatchRefusal:
    @pytest.mark.parametrize("tool", WRITE_CAPABLE_TOOLS)
    def test_write_capable_tool_refused_for_delegated_agent(self, tool):
        managed, execute = _run_middleware(tool)
        execute.assert_not_called()
        assert managed.blocked
        assert "delegated_read_only" in managed.result

    def test_read_tool_dispatched_for_delegated_agent(self):
        managed, execute = _run_middleware("read_file")
        execute.assert_called_once()
        assert not managed.blocked

    def test_write_tool_dispatched_for_ordinary_agent(self):
        managed, execute = _run_middleware("write_file", delegated=False)
        execute.assert_called_once()


# ---------------------------------------------------------------------------
# JARVIS sub-agent provider: read allowlist, refuse-before-approval
# ---------------------------------------------------------------------------


def _sub_provider(no_learning_reason=None):
    from plugins.memory.jarvis import JARVISSubagentProvider
    return JARVISSubagentProvider(goal="g", depth=1, no_learning_reason=no_learning_reason)


class TestJarvisSubagentProvider:
    def test_schemas_are_read_only(self):
        names = {s["name"] for s in _sub_provider().get_tool_schemas()}
        assert names <= JARVIS_READ_TOOLS

    @pytest.mark.parametrize("no_learning", [None, "session id starts with 'nolearn_'"])
    @pytest.mark.parametrize("tool", JARVIS_WRITE_TOOLS)
    def test_write_refused_before_approval(self, tool, no_learning, fake_rag):
        """Even with every approval path saying yes, the write is refused and
        neither approval nor dispatch is reached."""
        approve = MagicMock(return_value={"approved": True, "message": None})
        with patch("tools.approval.request_tool_approval", approve), \
             patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as dispatch:
            result = json.loads(_sub_provider(no_learning).handle_tool_call(
                tool, {"text": "t", "memory_id": "m", "title": "x", "session_id": ORDINARY_ID}))
        assert result.get("ok") is False and result.get("code") == "delegated_read_only"
        approve.assert_not_called()
        dispatch.assert_not_called()
        _assert_no_rag_writes(fake_rag)

    def test_read_still_dispatched(self, fake_rag):
        result = json.loads(_sub_provider().handle_tool_call("hermes_rag_search", {"query": "q"}))
        assert result.get("ok") is True
        fake_rag.search_memory.assert_called_once()

    @pytest.mark.parametrize("tool", sorted(JARVIS_READ_TOOLS))
    def test_reads_are_not_refused_in_delegated_context(self, tool, fake_rag):
        """Allowlisted reads pass both the provider and the dispatch guard and
        keep their existing results (continuity lists are not routed by the
        sub-agent dispatcher today and still answer 'unknown tool')."""
        from agent.delegation_context import delegated_child_context
        with delegated_child_context("child-session"):
            result = json.loads(_sub_provider().handle_tool_call(tool, {"query": "q", "messages": []}))
        assert result.get("code") != "delegated_read_only", result
        _assert_no_rag_writes(fake_rag)

    @pytest.mark.parametrize("no_learning", [None, "session id starts with 'nolearn_'"])
    def test_session_end_never_learns(self, no_learning):
        with patch("plugins.memory.jarvis._on_session_end_learning") as learn:
            _sub_provider(no_learning).on_session_end([{"role": "user", "content": "remember x"}])
        learn.assert_not_called()


class TestModuleDispatchGuards:
    @pytest.mark.parametrize("tool", ["hermes_rag_store", "hermes_rag_store_learnings",
                                      "hermes_rag_delete", "hermes_rag_rollback_learnings"])
    def test_rag_dispatch_refuses_writes_in_delegated_context(self, tool, fake_rag):
        from agent.delegation_context import delegated_child_context
        from plugins.memory.jarvis import _dispatch_jarvis_tool_call
        with delegated_child_context("child-session"):
            result = json.loads(_dispatch_jarvis_tool_call(
                tool, {"text": "t", "memory_id": "m", "session_id": ORDINARY_ID}))
        assert result.get("ok") is False and result.get("code") == "delegated_read_only"
        _assert_no_rag_writes(fake_rag)

    def test_continuity_capability_dispatch_refuses_in_delegated_context(self):
        from agent.delegation_context import delegated_child_context
        from plugins.memory.jarvis.capabilities import concrete_continuity
        with patch("plugins.memory.jarvis.HermesRAGProvider") as Provider, \
             delegated_child_context("child-session"):
            result = concrete_continuity._dispatch_continuity("hermes_goal_create", {"title": "x"})
        Provider.assert_not_called()
        assert result.get("ok") is False and result.get("code") == "delegated_read_only"


class TestContinuityDispatchHome:
    """_dispatch_continuity opens the active Hermes home's continuity.db, not
    a hard-coded ~/.hermes (which let sandboxed tests reach production)."""

    def _initialized_home(self):
        from plugins.memory.jarvis.capabilities import concrete_continuity
        with patch("plugins.memory.jarvis.HermesRAGProvider") as Provider:
            Provider.return_value.handle_tool_call.return_value = json.dumps({"ok": True})
            concrete_continuity._dispatch_continuity("hermes_goal_list", {})
        return Provider.return_value.initialize.call_args.kwargs["hermes_home"]

    def test_uses_configured_hermes_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert self._initialized_home() == str(tmp_path)

    def test_profile_override_wins(self, monkeypatch, tmp_path):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "env-home"))
        token = set_hermes_home_override(tmp_path / "profile-home")
        try:
            assert self._initialized_home() == str(tmp_path / "profile-home")
        finally:
            reset_hermes_home_override(token)

    def test_reads_the_configured_continuity_db(self, monkeypatch, tmp_path):
        """End to end with the real provider: a goal written to a temporary
        home's continuity.db is what the list returns."""
        from hermes_cli import continuity_db
        from plugins.memory.jarvis.capabilities import concrete_continuity
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with continuity_db.connect_closing(tmp_path / "continuity.db") as conn:
            continuity_db.create_goal(conn, title="Decision 42 isolation probe")
            conn.commit()
        result = concrete_continuity._dispatch_continuity("hermes_goal_list", {})
        assert "Decision 42 isolation probe" in json.dumps(result)


# ---------------------------------------------------------------------------
# Background review and skills
# ---------------------------------------------------------------------------


class TestBackgroundAndSkills:
    def test_delegated_agent_never_spawns_background_review(self):
        import run_agent
        agent = SimpleNamespace(_delegated_read_only=True, _no_learning_reason=None,
                                session_id=ORDINARY_ID)
        with patch("agent.background_review.spawn_background_review_thread") as spawn, \
             patch.object(run_agent.threading, "Thread") as Thread:
            run_agent.AIAgent._spawn_background_review(agent, [], review_memory=True, review_skills=True)
        spawn.assert_not_called()
        Thread.assert_not_called()

    def test_skill_manage_create_refused_in_delegated_context(self):
        from agent.delegation_context import delegated_child_context
        from hermes_constants import get_hermes_home
        from tools.skill_manager_tool import skill_manage
        skills_dir = get_hermes_home() / "skills"
        before = sorted(p.as_posix() for p in skills_dir.rglob("*")) if skills_dir.exists() else []
        content = "---\nname: d42-probe\ndescription: probe\n---\nbody\n"
        with delegated_child_context("child-session"):
            result = json.loads(skill_manage(action="create", name="d42-probe", content=content))
        after = sorted(p.as_posix() for p in skills_dir.rglob("*")) if skills_dir.exists() else []
        assert result.get("success") is not True
        assert "delegated_read_only" in json.dumps(result)
        assert before == after


# ---------------------------------------------------------------------------
# Capability plans
# ---------------------------------------------------------------------------


@pytest.fixture
def capability_registry(monkeypatch):
    """The default capability set installed as the process-wide registry
    (production's is empty), restored afterwards."""
    from plugins.memory.jarvis.capabilities import register_default_capabilities
    from plugins.memory.jarvis.capabilities import registry as registry_module

    reg = registry_module.JARVISCapabilityRegistry()
    register_default_capabilities(reg)
    monkeypatch.setattr(registry_module, "_jarvis_registry", reg)
    return reg


@pytest.fixture
def jarvis_dispatch(monkeypatch, fake_rag):
    """Record every JARVIS tool dispatched by capabilities; nothing persists.

    RAG capabilities call plugins.memory.jarvis._dispatch_jarvis_tool_call and
    continuity capabilities build a HermesRAGProvider -- both are replaced, so
    a dispatched write is observable but never reaches storage.
    """
    import plugins.memory.jarvis as jarvis

    calls = []

    def _dispatch(tool_name, args, **kwargs):
        calls.append((tool_name, dict(args)))
        return json.dumps({"ok": True, "memory_id": "m-1", "results": []})

    provider = MagicMock(name="HermesRAGProvider")
    provider.return_value.handle_tool_call.side_effect = (
        lambda tool_name, args: calls.append((tool_name, dict(args))) or json.dumps({"ok": True})
    )
    monkeypatch.setattr(jarvis, "_dispatch_jarvis_tool_call", _dispatch)
    monkeypatch.setattr(jarvis, "HermesRAGProvider", provider)
    return calls


def _plan(*tasks):
    """A consistent plan/graph pair. Each task: (capability_name, parameters)."""
    from plugins.memory.jarvis.planner.task_graph import Task, TaskGraph, execution_plan_from_graph

    graph = TaskGraph()
    for i, (capability, params) in enumerate(tasks):
        graph.add_task(Task(id=f"t{i}_{capability}", capability_name=capability,
                            task_type="default", authority_level="L2", parameters=params))
    return execution_plan_from_graph(graph), graph


# A valid write: rag_store with real text passes RagStoreCapability.validate.
STORE_PLAN = (("rag_store", {"text": "the user prefers dark mode"}),)
GOAL_PLAN = (("goal_create", {"title": "Ship Decision 42"}),)
READ_PLAN = (("rag_search", {"query": "dark mode"}),)


def _delegate_with_plan(tasks, parent=None, batch=False):
    """Run the public delegate_task() path with a capability plan attached
    the way _attach_capability_plan does, and the model loop mocked."""
    from tools.delegate_tool import delegate_task

    def _attach(parent_agent, child, goal, task_type, depth=1):
        child._capability_plan, child._capability_graph = _plan(*tasks)
        child._capability_task_type = "default"
        child._capability_authority_level = "L2"

    def _child():
        child = MagicMock()
        child.session_id = "20261011_120000_c0ffee00"
        child.model = "test-model"
        child._capability_plan = None
        child._capability_graph = None
        child.run_conversation.return_value = {
            "final_response": "done", "completed": True, "api_calls": 1, "messages": []}
        return child

    with patch("run_agent.AIAgent", side_effect=lambda *a, **k: _child()), \
         patch("tools.delegate_tool._attach_capability_plan", side_effect=_attach):
        if batch:
            raw = delegate_task(tasks=[{"goal": "record the user's theme preference"},
                                       {"goal": "record the user's font preference"}],
                                parent_agent=parent or _parent())
        else:
            raw = delegate_task(goal="remember the user's preference", parent_agent=parent or _parent())
    return json.loads(raw)


class TestCapabilityPlans:
    """Decision 42: capability plans run by delegated children are read-only.

    These go through the real callers -- delegate_task() (single task and
    batch) -> _run_single_child() -> _execute_capability_plan_if_present() ->
    CapabilityRuntime -> concrete capability -- with valid parameters, so a
    refusal can only come from the policy guard.
    """

    @pytest.mark.parametrize("tasks", [STORE_PLAN, GOAL_PLAN], ids=["rag_store", "goal_create"])
    @pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
    def test_delegated_write_plan_refused_by_guard(self, capability_registry, jarvis_dispatch,
                                                   fake_rag, tasks, batch):
        result = _delegate_with_plan(tasks, batch=batch)
        assert jarvis_dispatch == [], jarvis_dispatch
        _assert_no_rag_writes(fake_rag)
        for entry in result["results"]:
            summary = entry["capability_execution"]["summary"]
            assert entry["capability_execution"]["status"] == "failure"
            assert "Decision 42" in summary and "is not read-only" in summary, summary

    def test_control_same_plan_writes_when_guard_bypassed(self, capability_registry, jarvis_dispatch):
        """Proves the plans above are valid: with the guard stubbed out, the
        write reaches downstream dispatch. If the guard is removed, the
        refusal test above fails."""
        with patch("tools.delegate_tool._capability_plan_read_only_violation", return_value=None):
            result = _delegate_with_plan(STORE_PLAN)
        assert [name for name, _ in jarvis_dispatch] == ["hermes_rag_store"]
        assert result["results"][0]["capability_execution"]["status"] == "success"

    def test_delegated_read_plan_still_runs(self, capability_registry, jarvis_dispatch):
        result = _delegate_with_plan(READ_PLAN)
        assert [name for name, _ in jarvis_dispatch] == ["hermes_rag_search"]
        assert result["results"][0]["capability_execution"]["status"] == "success"

    def test_mixed_plan_refused_whole_before_any_task_runs(self, capability_registry, jarvis_dispatch):
        result = _delegate_with_plan(READ_PLAN + STORE_PLAN)
        assert jarvis_dispatch == []
        assert result["results"][0]["capability_execution"]["status"] == "failure"

    def test_parent_capability_plan_unaffected(self, capability_registry, jarvis_dispatch):
        """A non-delegated caller using the runtime directly still writes."""
        from plugins.memory.jarvis.capabilities.runtime import CapabilityRuntime

        plan, graph = _plan(*STORE_PLAN)
        result = CapabilityRuntime(registry=capability_registry).execute_plan(
            plan=plan, task_graph=graph, authority_level="L2", base_parameters={"goal": "g"})
        assert result.all_succeeded
        assert [name for name, _ in jarvis_dispatch] == ["hermes_rag_store"]

    def test_runtime_counts_missing_task_as_failure(self, capability_registry, jarvis_dispatch):
        """A plan naming a task its graph lacks fails; it is never 0/0 success."""
        from plugins.memory.jarvis.capabilities.runtime import CapabilityRuntime
        from plugins.memory.jarvis.planner.task_graph import ExecutionPlan, ExecutionStage, TaskGraph

        plan = ExecutionPlan(stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"ghost"}))])
        result = CapabilityRuntime(registry=capability_registry).execute_plan(plan=plan, task_graph=TaskGraph())
        assert not result.all_succeeded
        assert (result.task_count, result.failure_count) == (1, 1)
        assert jarvis_dispatch == []

    def test_empty_delegated_plan_is_not_a_success(self, capability_registry, jarvis_dispatch):
        """An attached plan with no tasks is NOT_EXECUTED, so the child's model
        loop still runs (a 0/0 'success' used to skip it)."""
        from plugins.memory.jarvis.planner.task_graph import ExecutionPlan, TaskGraph
        from tools.delegate_tool import _run_single_child

        child = MagicMock()
        child.session_id = "20261011_120000_c0ffee00"
        child.run_conversation.return_value = {
            "final_response": "done", "completed": True, "api_calls": 1, "messages": []}
        child._capability_plan, child._capability_graph = ExecutionPlan(stages=[]), TaskGraph()
        entry = _run_single_child(0, "look something up", child, _parent())
        child.run_conversation.assert_called_once()
        assert entry.get("exit_reason") != "capability_plan_completed"
        assert "capability_execution" not in entry
        assert jarvis_dispatch == []

    @pytest.mark.parametrize("marker", ["absent", False, "mock"])
    def test_guard_independent_of_delegation_marker(self, capability_registry, jarvis_dispatch, marker):
        """_run_single_child refuses the write even when the child's
        delegation state is missing, false or not a real boolean."""
        from tools.delegate_tool import _run_single_child

        child = MagicMock()
        child.session_id = "20261011_120000_c0ffee00"
        child.run_conversation.return_value = {
            "final_response": "done", "completed": True, "api_calls": 1, "messages": []}
        if marker == "absent":
            del child._delegated_read_only
        elif marker is False:
            child._delegated_read_only = False
        child._capability_plan, child._capability_graph = _plan(*STORE_PLAN)
        child._capability_task_type = "default"
        child._capability_authority_level = "L2"
        entry = _run_single_child(0, "remember this", child, _parent())
        assert jarvis_dispatch == []
        assert "Decision 42" in entry["capability_execution"]["summary"]

    @pytest.mark.parametrize("failure", ["registry_unavailable", "unresolvable_task", "unknown_capability"])
    def test_guard_fails_closed_when_plan_cannot_be_checked(self, capability_registry, jarvis_dispatch, failure):
        from tools.delegate_tool import _run_single_child

        child = MagicMock()
        child.session_id = "20261011_120000_c0ffee00"
        child.run_conversation.return_value = {
            "final_response": "done", "completed": True, "api_calls": 1, "messages": []}
        plan, graph = _plan(*READ_PLAN)
        if failure == "unresolvable_task":
            graph = MagicMock(get_task=MagicMock(return_value=None))
        elif failure == "unknown_capability":
            plan, graph = _plan(("not_a_capability", {}))
        child._capability_plan, child._capability_graph = plan, graph
        child._capability_task_type = "default"
        child._capability_authority_level = "L2"
        if failure == "registry_unavailable":
            with patch("plugins.memory.jarvis.capabilities.registry.get_jarvis_registry",
                       side_effect=RuntimeError("registry down")):
                entry = _run_single_child(0, "look something up", child, _parent())
        else:
            entry = _run_single_child(0, "look something up", child, _parent())
        assert jarvis_dispatch == []
        assert "Decision 42" in entry["capability_execution"]["summary"]
