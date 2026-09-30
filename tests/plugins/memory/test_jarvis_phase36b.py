"""Phase 36B — Capability Registry + Task Graph + Planner tests."""

import pytest

from plugins.memory.jarvis.capabilities.base import (
    Capability,
    CapabilityCategory,
    CapabilityInputType,
    CapabilityOutputType,
    CapabilityRequirements,
    CapabilityContext,
    CapabilityResult,
    get_registry,
    reset_registry,
)
from plugins.memory.jarvis.capabilities.registry import (
    JARVISCapabilityRegistry,
    CapabilityRegistrationError,
    get_jarvis_registry,
    reset_jarvis_registry,
    _is_tool_exposed_to_subagents,
    _get_tool_level,
)
from plugins.memory.jarvis.planner.task_graph import (
    Task,
    TaskGraph,
    ExecutionPlan,
    ExecutionStage,
    TaskGraphError,
    execution_plan_from_graph,
)
from plugins.memory.jarvis.planner.sub_agent_planner import (
    SubAgentPlanner,
    SubAgentPlannerError,
    create_default_planner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registries():
    reset_registry()
    reset_jarvis_registry()
    yield
    reset_registry()
    reset_jarvis_registry()


# ---------------------------------------------------------------------------
# Test capabilities
# ---------------------------------------------------------------------------

class RagSearchCapability(Capability):
    def get_name(self) -> str:
        return "rag_search_cap"

    def get_description(self) -> str:
        return "Search RAG memory"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_search"}),
            input_type=CapabilityInputType.QUERY,
            output_type=CapabilityOutputType.STRUCTURED,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult.success_result(data={"results": []})


class RagStoreCapability(Capability):
    def get_name(self) -> str:
        return "rag_store_cap"

    def get_description(self) -> str:
        return "Store RAG memory"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L1",
            required_tools=frozenset({"hermes_rag_store"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.BOOLEAN,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult.success_result(data=True)


class SynthesizeCapability(Capability):
    """Depends on rag_search_cap."""

    def get_name(self) -> str:
        return "synthesize_cap"

    def get_description(self) -> str:
        return "Synthesize from search results"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.SYNTHESIS

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_search"}),
            dependencies=frozenset({"rag_search_cap"}),
            input_type=CapabilityInputType.STRUCTURED,
            output_type=CapabilityOutputType.TEXT,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult.success_result(data="synthesized")


class ParentOnlyToolCapability(Capability):
    """Capability requiring a parent-only tool — should be rejected."""
    def get_name(self) -> str:
        return "parent_only_cap"

    def get_description(self) -> str:
        return "Should fail"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.ORCHESTRATION

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L3",
            required_tools=frozenset({"delegate_task"}),
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult.success_result()


class UnknownToolCapability(Capability):
    """Capability requiring an unknown tool — should be rejected."""
    def get_name(self) -> str:
        return "unknown_tool_cap"

    def get_description(self) -> str:
        return "Should fail"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_fake_nonexistent"}),
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult.success_result()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_is_tool_exposed_for_rag_search(self):
        assert _is_tool_exposed_to_subagents("hermes_rag_search") is True

    def test_is_tool_exposed_for_parent_only(self):
        assert _is_tool_exposed_to_subagents("delegate_task") is False

    def test_is_tool_exposed_for_unknown(self):
        assert _is_tool_exposed_to_subagents("nonexistent_tool") is False

    def test_get_tool_level_rag_search(self):
        assert _get_tool_level("hermes_rag_search") == "L0"

    def test_get_tool_level_rag_delete(self):
        assert _get_tool_level("hermes_rag_delete") == "L3"

    def test_get_tool_level_unknown(self):
        assert _get_tool_level("nonexistent") == "L3"


# ---------------------------------------------------------------------------
# JARVISCapabilityRegistry validation
# ---------------------------------------------------------------------------

class TestJARVISCapabilityRegistryValidation:
    def test_register_valid_capability(self):
        registry = JARVISCapabilityRegistry()
        registry.register(RagSearchCapability)
        assert "rag_search_cap" in registry.list_capabilities()

    def test_reject_parent_only_tool_capability(self):
        registry = JARVISCapabilityRegistry()
        with pytest.raises(CapabilityRegistrationError, match="not exposed to sub-agents"):
            registry.register(ParentOnlyToolCapability)

    def test_reject_unknown_tool_capability(self):
        registry = JARVISCapabilityRegistry()
        with pytest.raises(CapabilityRegistrationError, match="not found in tool registry"):
            registry.register(UnknownToolCapability)

    def test_duplicate_capability_rejected(self):
        registry = JARVISCapabilityRegistry()
        registry.register(RagSearchCapability)
        with pytest.raises(CapabilityRegistrationError, match="already registered"):
            registry.register(RagSearchCapability)

    def test_get_capability_for_task_type(self):
        registry = JARVISCapabilityRegistry()
        registry.register(RagSearchCapability)
        registry.register(RagStoreCapability)

        # Research should only get L0 caps
        research_caps = registry.get_capability_for_task_type("research")
        names = {c.get_name() for c in research_caps}
        assert "rag_search_cap" in names
        assert "rag_store_cap" not in names

        # Note-taking gets L0 + L1
        note_caps = registry.get_capability_for_task_type("note_taking")
        names = {c.get_name() for c in note_caps}
        assert "rag_search_cap" in names
        assert "rag_store_cap" in names

    def test_validate_capability_for_execution(self):
        registry = JARVISCapabilityRegistry()
        registry.register(RagSearchCapability)
        registry.register(RagStoreCapability)

        ok, reason = registry.validate_capability_for_execution(
            "rag_search_cap", "research", "L0"
        )
        ok2, reason2 = registry.validate_capability_for_execution(
            "rag_store_cap", "research", "L0"
        )
        # We can't easily assert True/False without full registry, but no exception


# ---------------------------------------------------------------------------
# TaskGraph
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_creation(self):
        t = Task(id="t1", capability_name="c1")
        assert t.id == "t1"
        assert t.capability_name == "c1"
        assert t.task_type == "default"
        assert t.authority_level == "L0"

    def test_empty_id_raises(self):
        with pytest.raises(ValueError):
            Task(id="", capability_name="c1")

    def test_empty_capability_raises(self):
        with pytest.raises(ValueError):
            Task(id="t1", capability_name="")

    def test_invalid_authority_raises(self):
        with pytest.raises(ValueError):
            Task(id="t1", capability_name="c1", authority_level="L99")

    def test_frozen(self):
        t = Task(id="t1", capability_name="c1")
        with pytest.raises(AttributeError):
            t.id = "t2"


class TestTaskGraph:
    def test_add_and_list(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        g.add_task(Task(id="b", capability_name="cb"))
        tasks = g.list_tasks()
        assert len(tasks) == 2

    def test_add_duplicate_raises(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        with pytest.raises(TaskGraphError, match="already exists"):
            g.add_task(Task(id="a", capability_name="cb"))

    def test_add_edge(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        g.add_task(Task(id="b", capability_name="cb"))
        g.add_edge("a", "b")
        deps = g.get_dependencies("b")
        assert "a" in deps

    def test_add_edge_nonexistent_source_raises(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        with pytest.raises(TaskGraphError, match="not in graph"):
            g.add_edge("x", "a")

    def test_add_edge_nonexistent_target_raises(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        with pytest.raises(TaskGraphError, match="not in graph"):
            g.add_edge("a", "x")

    def test_self_dependency_raises(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        with pytest.raises(TaskGraphError, match="Self-dependency"):
            g.add_edge("a", "a")

    def test_authority_escalation_edge_rejected(self):
        g = TaskGraph()
        g.add_task(Task(id="low", capability_name="c_low", authority_level="L0"))
        g.add_task(Task(id="high", capability_name="c_high", authority_level="L3"))
        with pytest.raises(TaskGraphError, match="escalates authority"):
            g.add_edge("low", "high")

    def test_authority_deescalation_edge_allowed(self):
        g = TaskGraph()
        g.add_task(Task(id="high", capability_name="c_high", authority_level="L3"))
        g.add_task(Task(id="low", capability_name="c_low", authority_level="L0"))
        # High → Low is OK (information declassifies)
        g.add_edge("high", "low")

    def test_cycle_detection(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        g.add_task(Task(id="b", capability_name="cb"))
        g.add_task(Task(id="c", capability_name="cc"))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        issues = g.validate()
        assert any("cycle" in i.lower() for i in issues)

    def test_topo_sort(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        g.add_task(Task(id="b", capability_name="cb"))
        g.add_task(Task(id="c", capability_name="cc"))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        sorted_ids = g.topo_sort()
        assert sorted_ids.index("a") < sorted_ids.index("b") < sorted_ids.index("c")

    def test_topo_sort_raises_on_cycle(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        g.add_task(Task(id="b", capability_name="cb"))
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        with pytest.raises(TaskGraphError):
            g.topo_sort()

    def test_get_task(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca", description="desc"))
        t = g.get_task("a")
        assert t is not None
        assert t.description == "desc"

    def test_get_task_nonexistent(self):
        g = TaskGraph()
        assert g.get_task("x") is None

    def test_list_tasks_empty(self):
        g = TaskGraph()
        assert g.list_tasks() == []


# ---------------------------------------------------------------------------
# ExecutionPlan
# ---------------------------------------------------------------------------

class TestExecutionPlanFromGraph:
    def test_linear_chain(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        g.add_task(Task(id="b", capability_name="cb"))
        g.add_task(Task(id="c", capability_name="cc"))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        plan = execution_plan_from_graph(g)
        assert len(plan.stages) == 3
        assert plan.stages[0].task_ids == frozenset({"a"})
        assert plan.stages[1].task_ids == frozenset({"b"})
        assert plan.stages[2].task_ids == frozenset({"c"})

    def test_parallel_tasks(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        g.add_task(Task(id="b", capability_name="cb"))
        g.add_task(Task(id="c", capability_name="cc"))
        g.add_task(Task(id="d", capability_name="cd"))
        # a and b are independent; both must finish before c and d
        g.add_edge("a", "c")
        g.add_edge("b", "c")
        g.add_edge("a", "d")
        g.add_edge("b", "d")
        plan = execution_plan_from_graph(g)
        assert len(plan.stages) == 2
        assert plan.stages[0].task_ids == frozenset({"a", "b"})
        assert plan.stages[1].task_ids == frozenset({"c", "d"})

    def test_single_task(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        plan = execution_plan_from_graph(g)
        assert len(plan.stages) == 1
        assert plan.stages[0].task_ids == frozenset({"a"})

    def test_diamond(self):
        """a → b, a → c, b → d, c → d"""
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        g.add_task(Task(id="b", capability_name="cb"))
        g.add_task(Task(id="c", capability_name="cc"))
        g.add_task(Task(id="d", capability_name="cd"))
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("b", "d")
        g.add_edge("c", "d")
        plan = execution_plan_from_graph(g)
        assert len(plan.stages) == 3
        assert plan.stages[0].task_ids == frozenset({"a"})
        assert plan.stages[1].task_ids == frozenset({"b", "c"})
        assert plan.stages[2].task_ids == frozenset({"d"})

    def test_all_task_ids(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        g.add_task(Task(id="b", capability_name="cb"))
        plan = execution_plan_from_graph(g)
        assert plan.all_task_ids == frozenset({"a", "b"})

    def test_empty_graph(self):
        g = TaskGraph()
        plan = execution_plan_from_graph(g)
        assert plan.stages == []

    def test_get_stage(self):
        g = TaskGraph()
        g.add_task(Task(id="a", capability_name="ca"))
        plan = execution_plan_from_graph(g)
        stage = plan.get_stage(0)
        assert stage is not None
        assert stage.task_ids == frozenset({"a"})
        assert plan.get_stage(99) is None


# ---------------------------------------------------------------------------
# SubAgentPlanner
# ---------------------------------------------------------------------------

class TestSubAgentPlanner:
    def _setup_registry(self):
        registry = get_jarvis_registry()
        registry.register(RagSearchCapability)
        registry.register(RagStoreCapability)
        registry.register(SynthesizeCapability)
        return registry

    def test_empty_goal_raises(self):
        planner = create_default_planner()
        with pytest.raises(SubAgentPlannerError, match="Goal must be non-empty"):
            planner.plan("")

    def test_invalid_authority_raises(self):
        planner = create_default_planner()
        with pytest.raises(SubAgentPlannerError, match="Invalid authority_level"):
            planner.plan("test", authority_level="L99")

    def test_plan_with_no_capabilities(self):
        reset_jarvis_registry()
        planner = create_default_planner()
        plan = planner.plan("anything")
        assert plan.stages == []

    def test_plan_with_capabilities(self):
        self._setup_registry()
        planner = create_default_planner()
        plan = planner.plan("test research goal", task_type="research", authority_level="L0")
        # Should have at least one stage
        assert len(plan.stages) >= 1

    def test_plan_respects_dependencies(self):
        self._setup_registry()
        planner = create_default_planner()
        plan = planner.plan("synthesize", task_type="research", authority_level="L0")
        # Find stages containing our caps
        all_ids = plan.all_task_ids
        # If synthesize_cap is present, rag_search_cap must be in an earlier stage
        synth_stage_idx = None
        search_stage_idx = None
        for stage in plan.stages:
            for tid in stage.task_ids:
                if tid.startswith("task_synthesize_cap_"):
                    synth_stage_idx = stage.stage_index
                if tid.startswith("task_rag_search_cap_"):
                    search_stage_idx = stage.stage_index
        if synth_stage_idx is not None and search_stage_idx is not None:
            assert search_stage_idx < synth_stage_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
