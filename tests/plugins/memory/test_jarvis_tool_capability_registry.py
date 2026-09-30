"""Tests for Phase 25A — JARVIS Tool Capability Registry."""

import pytest

from plugins.memory.jarvis.tool_capability_registry import (
    ToolCapability,
    TOOL_CAPABILITY_REGISTRY,
    AUTHORITY_LEVELS,
    get_capability,
    get_capabilities_for_task_type,
    get_permitted_tools_for_task_type,
    is_tool_permitted,
    is_tool_parent_only,
    requires_approval,
    get_tools_by_authority_level,
    get_tools_by_category,
    validate_capability_policy,
    _register_capability,
)


class TestRegistryStructure:
    """Test the registry structure and constants."""

    def test_authority_levels_set(self):
        assert AUTHORITY_LEVELS == {"L0", "L1", "L2", "L3"}

    def test_rag_tools_registered(self):
        """RAG tools are registered in the registry."""
        assert "hermes_rag_search" in TOOL_CAPABILITY_REGISTRY
        assert "hermes_rag_store" in TOOL_CAPABILITY_REGISTRY
        assert "hermes_rag_delete" in TOOL_CAPABILITY_REGISTRY
        assert "hermes_rag_retrieve_learnings" in TOOL_CAPABILITY_REGISTRY
        assert "hermes_rag_extract_learnings" in TOOL_CAPABILITY_REGISTRY
        assert "hermes_rag_store_learnings" in TOOL_CAPABILITY_REGISTRY
        assert "hermes_rag_rollback_learnings" in TOOL_CAPABILITY_REGISTRY

    def test_continuity_tools_registered(self):
        """Continuity tools are registered in the registry."""
        for tool in ("hermes_goal_list", "hermes_task_list", "hermes_decision_list",
                     "hermes_goal_complete", "hermes_task_complete",
                     "hermes_goal_create", "hermes_task_create",
                     "hermes_goal_update", "hermes_task_update",
                     "hermes_decision_record", "hermes_decision_supersede"):
            assert tool in TOOL_CAPABILITY_REGISTRY

    def test_parent_only_tools_registered(self):
        """Parent-only tools are registered but marked parent-only."""
        for tool in ("delegate_task", "terminal", "write_file", "read_file",
                     "browser_navigate", "computer_use", "cronjob", "send_message"):
            assert tool in TOOL_CAPABILITY_REGISTRY
            assert TOOL_CAPABILITY_REGISTRY[tool].parent_only is True


class TestPolicyResolution:
    """Test task_type → capability policy → permitted tool set."""

    def test_research_gets_l0_tools(self):
        """Research task type gets only L0 read-only tools."""
        tools = get_permitted_tools_for_task_type("research")

        # Should include read tools
        assert "hermes_rag_search" in tools
        assert "hermes_rag_retrieve_learnings" in tools
        assert "hermes_rag_extract_learnings" in tools
        assert "hermes_goal_list" in tools
        assert "hermes_task_list" in tools
        assert "hermes_decision_list" in tools

        # Should NOT include write/destructive tools
        assert "hermes_rag_store" not in tools
        assert "hermes_goal_create" not in tools
        assert "hermes_rag_delete" not in tools
        assert "hermes_decision_supersede" not in tools

        # Should NOT include parent-only tools
        assert "delegate_task" not in tools
        assert "terminal" not in tools
        assert "write_file" not in tools

    def test_summarize_gets_l0_tools(self):
        """Summarize task type gets only L0 read-only tools."""
        tools = get_permitted_tools_for_task_type("summarize")
        assert "hermes_rag_search" in tools
        assert "hermes_rag_store" not in tools

    def test_note_taking_gets_l0_l1_tools(self):
        """Note-taking task type gets L0 + L1 tools."""
        tools = get_permitted_tools_for_task_type("note_taking")

        # L0
        assert "hermes_rag_search" in tools
        assert "hermes_goal_list" in tools

        # L1
        assert "hermes_rag_store" in tools
        assert "hermes_rag_store_learnings" in tools
        assert "hermes_goal_complete" in tools
        assert "hermes_task_complete" in tools

        # Not L2/L3
        assert "hermes_goal_create" not in tools
        assert "hermes_rag_delete" not in tools

    def test_task_execution_gets_l0_l1_l2_tools(self):
        """Task execution task type gets L0 + L1 + L2 tools."""
        tools = get_permitted_tools_for_task_type("task_execution")

        # L0
        assert "hermes_rag_search" in tools

        # L1
        assert "hermes_rag_store" in tools

        # L2
        assert "hermes_goal_create" in tools
        assert "hermes_task_create" in tools
        assert "hermes_decision_record" in tools

        # Not L3
        assert "hermes_rag_delete" not in tools
        assert "hermes_decision_supersede" not in tools

    def test_maintenance_gets_all_tools(self):
        """Maintenance task type gets all tools (L0-L3)."""
        tools = get_permitted_tools_for_task_type("maintenance")

        # L0
        assert "hermes_rag_search" in tools

        # L1
        assert "hermes_rag_store" in tools

        # L2
        assert "hermes_goal_create" in tools

        # L3
        assert "hermes_rag_delete" in tools
        assert "hermes_decision_supersede" in tools
        assert "hermes_rag_rollback_learnings" in tools

        # Not parent-only
        assert "delegate_task" not in tools
        assert "terminal" not in tools

    def test_default_gets_all_tools(self):
        """Default task type gets all tools (backward-compatible)."""
        tools = get_permitted_tools_for_task_type("default")
        assert "hermes_rag_search" in tools
        assert "hermes_rag_store" in tools
        assert "hermes_rag_delete" in tools
        assert "delegate_task" not in tools


class TestIsToolPermitted:
    """Test the is_tool_permitted function."""

    def test_permitted_tool_returns_true(self):
        assert is_tool_permitted("hermes_rag_search", "research") is True

    def test_unauthorized_tool_returns_false(self):
        assert is_tool_permitted("hermes_rag_store", "research") is False

    def test_parent_only_tool_returns_false(self):
        assert is_tool_permitted("delegate_task", "maintenance") is False

    def test_unknown_tool_returns_false(self):
        assert is_tool_permitted("unknown_tool", "research") is False


class TestIsToolParentOnly:
    """Test the is_tool_parent_only function."""

    def test_delegate_task_is_parent_only(self):
        assert is_tool_parent_only("delegate_task") is True

    def test_terminal_is_parent_only(self):
        assert is_tool_parent_only("terminal") is True

    def test_rag_search_is_not_parent_only(self):
        assert is_tool_parent_only("hermes_rag_search") is False

    def test_unknown_tool_is_not_parent_only(self):
        assert is_tool_parent_only("unknown_tool") is False


class TestRequiresApproval:
    """Test the requires_approval function."""

    def test_destructive_tool_requires_approval(self):
        assert requires_approval("hermes_rag_delete") is True

    def test_read_tool_does_not_require_approval(self):
        assert requires_approval("hermes_rag_search") is False

    def test_write_tool_does_not_require_approval(self):
        assert requires_approval("hermes_rag_store") is False

    def test_unknown_tool_does_not_require_approval(self):
        assert requires_approval("unknown_tool") is False


class TestGetCapabilitiesByLevel:
    """Test filtering by authority level."""

    def test_l0_tools_are_read_only(self):
        l0_tools = get_tools_by_authority_level("L0")
        for cap in l0_tools:
            assert cap.classification == "read"

    def test_l1_tools_include_writes(self):
        """L1 tools include reversible writes."""
        l1_tools = get_tools_by_authority_level("L1")
        classifications = {cap.classification for cap in l1_tools}
        assert "write" in classifications or "read" in classifications

    def test_l3_destructive_tools_require_approval(self):
        l3_tools = get_tools_by_authority_level("L3")
        for cap in l3_tools:
            if cap.classification == "destructive":
                assert cap.requires_approval is True


class TestGetCapabilitiesByCategory:
    """Test filtering by category."""

    def test_rag_category(self):
        rag_tools = get_tools_by_category("rag")
        assert len(rag_tools) > 0
        for cap in rag_tools:
            assert cap.category == "rag"

    def test_continuity_category(self):
        continuity_tools = get_tools_by_category("continuity")
        assert len(continuity_tools) > 0
        for cap in continuity_tools:
            assert cap.category == "continuity"


class TestToolCapabilityValidation:
    """Test ToolCapability validation."""

    def test_invalid_authority_level_raises(self):
        with pytest.raises(ValueError, match="Invalid authority level"):
            ToolCapability(
                name="bad",
                category="test",
                authority_level="L5",
                classification="read",
                allowed_task_types=frozenset(),
            )

    def test_invalid_classification_raises(self):
        with pytest.raises(ValueError, match="Invalid classification"):
            ToolCapability(
                name="bad",
                category="test",
                authority_level="L0",
                classification="invalid",
                allowed_task_types=frozenset(),
            )

    def test_destructive_without_approval_raises(self):
        with pytest.raises(ValueError, match="must require approval"):
            ToolCapability(
                name="bad",
                category="test",
                authority_level="L3",
                classification="destructive",
                allowed_task_types=frozenset(),
                requires_approval=False,
            )

    def test_parent_only_exposed_raises(self):
        with pytest.raises(ValueError, match="cannot be exposed"):
            ToolCapability(
                name="bad",
                category="test",
                authority_level="L3",
                classification="read",
                allowed_task_types=frozenset({"research"}),
                parent_only=True,
                exposed_to_subagents=True,
            )


class TestValidateCapabilityPolicy:
    """Test the policy validation function."""

    def test_no_errors_in_current_registry(self):
        """The current registry should have no policy errors."""
        issues = validate_capability_policy()
        errors = [i for i in issues if i.startswith("ERROR")]
        assert len(errors) == 0, f"Policy errors found: {errors}"


class TestDuplicateRegistration:
    """Test that duplicate registration is prevented."""

    def test_duplicate_raises_error(self):
        with pytest.raises(ValueError, match="already registered"):
            _register_capability(ToolCapability(
                name="hermes_rag_search",  # Already registered
                category="test",
                authority_level="L0",
                classification="read",
                allowed_task_types=frozenset(),
            ))


class TestForgedToolNames:
    """Test that forged tool names are handled correctly."""

    def test_forged_tool_not_permitted(self):
        """A forged tool name is not permitted."""
        assert is_tool_permitted("hermes_rag_fake", "research") is False

    def test_forged_tool_not_parent_only(self):
        """A forged tool name is not parent-only."""
        assert is_tool_parent_only("hermes_rag_fake") is False

    def test_forged_tool_no_approval(self):
        """A forged tool name doesn't require approval."""
        assert requires_approval("hermes_rag_fake") is False

    def test_forged_tool_no_capability(self):
        """A forged tool name has no capability."""
        assert get_capability("hermes_rag_fake") is None


class TestParentOnlyIsolation:
    """Test that parent-only tools are isolated from sub-agents."""

    def test_parent_only_tools_not_in_permitted_set(self):
        """Parent-only tools are never in the permitted set."""
        for task_type in ("research", "summarize", "note_taking", "task_execution", "maintenance", "default"):
            tools = get_permitted_tools_for_task_type(task_type)
            assert "delegate_task" not in tools
            assert "terminal" not in tools
            assert "write_file" not in tools
            assert "read_file" not in tools
            assert "browser_navigate" not in tools
            assert "computer_use" not in tools
            assert "cronjob" not in tools
            assert "send_message" not in tools

    def test_parent_only_tools_have_no_allowed_task_types(self):
        """Parent-only tools should not have allowed task types."""
        for tool in ("delegate_task", "terminal", "write_file", "read_file"):
            cap = get_capability(tool)
            assert cap is not None
            assert len(cap.allowed_task_types) == 0


class TestProvenanceRequirements:
    """Test provenance requirements."""

    def test_write_tools_require_provenance(self):
        """Write tools require provenance."""
        cap = get_capability("hermes_rag_store")
        assert cap.provenance_required is True

    def test_destructive_tools_require_provenance(self):
        """Destructive tools require provenance."""
        cap = get_capability("hermes_rag_delete")
        assert cap.provenance_required is True

    def test_read_tools_do_not_require_provenance(self):
        """Read tools don't require provenance."""
        cap = get_capability("hermes_rag_search")
        assert cap.provenance_required is False


class TestSandboxRequirements:
    """Test sandbox/isolation requirements."""

    def test_system_tools_require_sandbox(self):
        """System tools require sandbox."""
        cap = get_capability("terminal")
        assert cap.sandbox_required is True

    def test_rag_tools_do_not_require_sandbox(self):
        """RAG tools don't require sandbox."""
        cap = get_capability("hermes_rag_search")
        assert cap.sandbox_required is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
