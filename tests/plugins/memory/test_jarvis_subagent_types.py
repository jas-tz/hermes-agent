"""Tests for JARVIS sub-agent task types — Phase 22A."""

import pytest
from plugins.memory.jarvis.subagent_types import (
    SubagentTaskType,
    SUBAGENT_TASK_TYPES,
    L0_READ_ONLY,
    L1_REVERSIBLE,
    L2_GATED,
    L3_APPROVAL,
    get_task_type,
    list_task_types,
    resolve_task_type_tools,
    validate_task_type,
    is_tool_allowed_for_task,
    _register_task_type,
)


class TestTaskTypeRegistry:
    """Test the task type registry."""

    def test_research_task_exists(self):
        task = get_task_type("research")
        assert task is not None
        assert task.authority_level == "L0"

    def test_summarize_task_exists(self):
        task = get_task_type("summarize")
        assert task is not None
        assert task.authority_level == "L0"

    def test_note_taking_task_exists(self):
        task = get_task_type("note_taking")
        assert task is not None
        assert task.authority_level == "L1"

    def test_task_execution_task_exists(self):
        task = get_task_type("task_execution")
        assert task is not None
        assert task.authority_level == "L2"

    def test_maintenance_task_exists(self):
        task = get_task_type("maintenance")
        assert task is not None
        assert task.authority_level == "L3"

    def test_all_task_types_registered(self):
        expected = {"research", "summarize", "note_taking", "task_execution", "maintenance", "default"}
        assert set(list_task_types()) == expected


class TestTaskTypeToolMapping:
    """Test that each task type exposes only its permitted tools."""

    def test_research_only_has_l0_tools(self):
        task = get_task_type("research")
        assert task.allowed_tools == L0_READ_ONLY
        # Should not include any write tools
        assert "hermes_rag_store" not in task.allowed_tools
        assert "hermes_goal_create" not in task.allowed_tools

    def test_note_taking_has_l0_and_l1(self):
        task = get_task_type("note_taking")
        assert task.allowed_tools == L0_READ_ONLY | L1_REVERSIBLE
        assert "hermes_rag_store" in task.allowed_tools
        assert "hermes_goal_create" not in task.allowed_tools

    def test_task_execution_has_l0_l1_l2(self):
        task = get_task_type("task_execution")
        assert task.allowed_tools == L0_READ_ONLY | L1_REVERSIBLE | L2_GATED
        assert "hermes_goal_create" in task.allowed_tools
        assert "hermes_rag_delete" not in task.allowed_tools

    def test_maintenance_has_all_tools(self):
        task = get_task_type("maintenance")
        assert task.allowed_tools == L0_READ_ONLY | L1_REVERSIBLE | L2_GATED | L3_APPROVAL
        assert "hermes_rag_delete" in task.allowed_tools


class TestTaskTypeAuthorityValidation:
    """Test that task type authority levels are validated."""

    def test_l0_cannot_include_write_tools(self):
        with pytest.raises(ValueError, match="cannot include tools outside its level"):
            SubagentTaskType(
                name="bad_research",
                description="Bad research",
                authority_level="L0",
                allowed_tools=L0_READ_ONLY | {"hermes_rag_store"},
            )

    def test_l1_cannot_include_l2_tools(self):
        with pytest.raises(ValueError, match="cannot include tools outside its level"):
            SubagentTaskType(
                name="bad_note",
                description="Bad note",
                authority_level="L1",
                allowed_tools=L0_READ_ONLY | L1_REVERSIBLE | {"hermes_goal_create"},
            )

    def test_invalid_authority_level(self):
        with pytest.raises(ValueError, match="Invalid authority level"):
            SubagentTaskType(
                name="bad_task",
                description="Bad task",
                authority_level="L5",
                allowed_tools=L0_READ_ONLY,
            )

    def test_duplicate_registration_fails(self):
        with pytest.raises(ValueError, match="already registered"):
            _register_task_type(SubagentTaskType(
                name="research",
                description="Duplicate",
                authority_level="L0",
                allowed_tools=L0_READ_ONLY,
            ))


class TestTaskTypeResolution:
    """Test task type resolution functions."""

    def test_validate_task_type_valid(self):
        assert validate_task_type("research") is True
        assert validate_task_type("maintenance") is True

    def test_validate_task_type_invalid(self):
        assert validate_task_type("unknown") is False
        assert validate_task_type("") is False

    def test_resolve_task_type_tools(self):
        task = resolve_task_type_tools("research")
        assert task is not None
        assert task.name == "research"

    def test_resolve_unknown_task_type(self):
        assert resolve_task_type_tools("unknown") is None


class TestIsToolAllowed:
    """Test tool allowance check."""

    def test_research_can_search(self):
        assert is_tool_allowed_for_task("research", "hermes_rag_search") is True

    def test_research_cannot_store(self):
        assert is_tool_allowed_for_task("research", "hermes_rag_store") is False

    def test_note_taking_can_store(self):
        assert is_tool_allowed_for_task("note_taking", "hermes_rag_store") is True

    def test_maintenance_can_delete(self):
        assert is_tool_allowed_for_task("maintenance", "hermes_rag_delete") is True

    def test_unknown_task_type_denies_all(self):
        assert is_tool_allowed_for_task("unknown", "hermes_rag_search") is False


class TestTaskTypeSecurityProperties:
    """Test security properties of task types."""

    def test_task_type_cannot_expand_whitelist(self):
        """Task types can only restrict the default whitelist, never expand it."""
        default_whitelist = L0_READ_ONLY | L1_REVERSIBLE | L2_GATED | L3_APPROVAL
        for name, task_type in SUBAGENT_TASK_TYPES.items():
            task_tools = task_type.allowed_tools | task_type.l2_tools | task_type.l3_tools
            assert task_tools.issubset(default_whitelist), (
                f"Task type '{name}' exceeds default whitelist: "
                f"{task_tools - default_whitelist}"
            )

    def test_research_is_read_only(self):
        """Research task type cannot modify any data."""
        task = get_task_type("research")
        write_tools = {"hermes_rag_store", "hermes_rag_store_learnings", "hermes_goal_create",
                       "hermes_task_create", "hermes_rag_delete", "hermes_rag_rollback_learnings"}
        for tool in write_tools:
            assert tool not in task.allowed_tools, f"Research should not have {tool}"

    def test_maintenance_has_explicit_l2_l3_tools(self):
        """Maintenance task type has explicit L2 and L3 tool sets."""
        task = get_task_type("maintenance")
        assert task.l2_tools == L2_GATED
        assert task.l3_tools == L3_APPROVAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
