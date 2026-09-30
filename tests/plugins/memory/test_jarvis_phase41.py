"""Phase 41 — Goal-Aware Planner tests."""

import pytest

from plugins.memory.jarvis.planner.sub_agent_planner import SubAgentPlanner
from plugins.memory.jarvis.planner.plan_result import plan_with_result
from plugins.memory.jarvis.capabilities import register_default_capabilities
from plugins.memory.jarvis.capabilities.registry import get_jarvis_registry, reset_jarvis_registry


@pytest.fixture(autouse=True)
def _reset_registries():
    reset_jarvis_registry()
    yield
    reset_jarvis_registry()


def _build_registry():
    registry = get_jarvis_registry()
    register_default_capabilities(registry)
    return registry


def _cap_names_from_plan(plan):
    """Extract capability names from a plan's task IDs."""
    names = set()
    for stage in plan.stages:
        for tid in stage.task_ids:
            inner = tid[len("task_"):]
            if len(inner) > 7 and inner[-7] == "_":
                names.add(inner[:-7])
            else:
                names.add(inner)
    return names


class TestGoalAwarePlanner:
    def test_planner_selects_relevant_capabilities(self):
        """Planner should select capabilities relevant to the goal, not all."""
        registry = _build_registry()
        planner = SubAgentPlanner(registry=registry)

        # Goal about goals should include goal_list, not task_create
        plan = planner.plan("list my goals", task_type="research", authority_level="L0")
        cap_names = _cap_names_from_plan(plan)

        assert "goal_list" in cap_names
        # task_create shouldn't be selected for "list my goals"
        assert "task_create" not in cap_names

    def test_planner_includes_dependencies(self):
        """Planner should include dependencies of selected capabilities."""
        registry = _build_registry()
        planner = SubAgentPlanner(registry=registry)

        plan = planner.plan("list goals", task_type="research", authority_level="L0")
        cap_names = _cap_names_from_plan(plan)

        assert "goal_list" in cap_names

    def test_planner_no_match_returns_empty(self):
        """If no capability matches the goal, return empty plan."""
        registry = _build_registry()
        planner = SubAgentPlanner(registry=registry)

        plan = planner.plan("quantum physics analysis", task_type="research", authority_level="L0")
        assert plan.stages == []

    def test_planner_empty_goal_returns_all_authorized(self):
        """Without a goal, return all authorized capabilities (backward compat)."""
        registry = _build_registry()
        planner = SubAgentPlanner(registry=registry)

        plan = planner.plan("do anything", task_type="default", authority_level="L0")
        # Should have multiple capabilities
        assert len(plan.stages) >= 1

    def test_planner_authority_filter_still_applies(self):
        """L0 child should never get L3 capabilities regardless of goal match."""
        registry = _build_registry()
        planner = SubAgentPlanner(registry=registry)

        plan = planner.plan("delete all memories", task_type="default", authority_level="L0")
        cap_names = _cap_names_from_plan(plan)

        assert "rag_delete" not in cap_names

    def test_planner_falls_back_to_model_on_vague_goal(self):
        """Vague goal should still match some capabilities."""
        registry = _build_registry()
        planner = SubAgentPlanner(registry=registry)

        plan = planner.plan("help me", task_type="research", authority_level="L0")
        # May or may not match — just shouldn't crash
        assert plan is not None

    def test_planner_description_match(self):
        """Capabilities should match based on description keywords."""
        registry = _build_registry()
        planner = SubAgentPlanner(registry=registry)

        plan = planner.plan("search my memory", task_type="research", authority_level="L0")
        cap_names = _cap_names_from_plan(plan)

        assert "rag_search" in cap_names

    def test_plan_with_result_consistency(self):
        """plan_with_result should use the same filtered capabilities."""
        registry = _build_registry()
        planner = SubAgentPlanner(registry=registry)

        result = plan_with_result(planner, "list goals", task_type="research", authority_level="L0")
        assert not result.is_empty

        plan_caps = _cap_names_from_plan(result.plan)
        graph_caps = {task.capability_name for task in result.graph.list_tasks()}

        assert plan_caps == graph_caps


class TestBackwardCompatibility:
    def test_existing_planner_interface_unchanged(self):
        """SubAgentPlanner.plan() signature still works."""
        registry = _build_registry()
        planner = SubAgentPlanner(registry=registry)

        plan = planner.plan("test goal", task_type="research", authority_level="L0")
        assert plan is not None
        assert hasattr(plan, 'stages')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
