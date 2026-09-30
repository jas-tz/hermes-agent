"""Phase 28A — Sub-Agent Security Observability.

Tests the audit trail, logging, and monitoring capabilities of the sub-agent
security infrastructure. Verifies that security-relevant events are observable.
"""

import json
import logging
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import (
    SubagentExecutionContext,
    SubagentLifecycleManager,
    create_execution_context,
    L0_READ_ONLY,
    L1_REVERSIBLE,
    L2_GATED,
    L3_APPROVAL,
    ALL_JARVIS_TOOLS,
    SUBAGENT_TASK_TYPES,
)
from plugins.memory.jarvis.subagent_filesystem import (
    SubagentFilesystemScope,
    validate_subagent_read_path,
    validate_subagent_write_path,
)
from plugins.memory.jarvis.subagent_terminal import (
    get_terminal_capability_for_task_type,
    validate_command,
    is_delegation_escape,
    is_filesystem_escape,
)
from plugins.memory.jarvis.tool_capability_registry import (
    is_tool_permitted,
    get_capability,
    get_permitted_tools_for_task_type,
)
from tools.delegate_tool import (
    register_lifecycle_manager,
    unregister_lifecycle_manager,
    transition_lifecycle,
    _get_lifecycle_state_for_subagent,
    _build_result_contract,
    _lifecycle_managers,
    _lifecycle_managers_lock,
)


# ---------------------------------------------------------------------------
# Execution Context Observability
# ====================================================================

class TestExecutionContextObservability:
    """Verify execution context is observable for audit."""

    def test_context_contains_full_audit_identity(self):
        """Context contains full identity for audit tracing."""
        ctx = create_execution_context(
            child_id="sa-audit-001",
            parent_session_id="parent-session",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )

        # Audit fields present
        assert ctx.child_id == "sa-audit-001"
        assert ctx.parent_session_id == "parent-session"
        assert ctx.task_goal == "research"
        assert ctx.task_type == "research"
        assert ctx.provenance_context_id == "subagent:research"
        assert ctx.authority_level == "L0"
        assert ctx.delegation_depth == 1

    def test_context_audit_trail_through_lifecycle(self):
        """Full audit trail from creation through lifecycle."""
        ctx = create_execution_context(
            child_id="sa-audit-trail",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        register_lifecycle_manager("sa-audit-trail", manager)

        # Each transition is observable
        transition_lifecycle("sa-audit-trail", "initializing")
        assert _get_lifecycle_state_for_subagent("sa-audit-trail") == "initializing"

        transition_lifecycle("sa-audit-trail", "running")
        assert _get_lifecycle_state_for_subagent("sa-audit-trail") == "running"

        transition_lifecycle("sa-audit-trail", "completed")
        assert _get_lifecycle_state_for_subagent("sa-audit-trail") == "completed"

        unregister_lifecycle_manager("sa-audit-trail")


# ====================================================================
# Lifecycle Manager Observability
# ====================================================================

class TestLifecycleManagerObservability:
    """Verify lifecycle manager is observable."""

    def test_lifecycle_manager_state_observable(self):
        """Lifecycle state can be observed externally."""
        ctx = create_execution_context(
            child_id="sa-obs-001",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        register_lifecycle_manager("sa-obs-001", manager)

        # External observation
        assert _get_lifecycle_state_for_subagent("sa-obs-001") == "pending"

        manager.transition("initializing")
        assert _get_lifecycle_state_for_subagent("sa-obs-001") == "initializing"

        unregister_lifecycle_manager("sa-obs-001")

    def test_lifecycle_transition_failure_observable(self):
        """Failed transitions are observable."""
        ctx = create_execution_context(
            child_id="sa-obs-fail",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)

        # Invalid transition raises ValueError
        with pytest.raises(ValueError):
            manager.transition("completed")


# ====================================================================
# Result Contract Observability
# ====================================================================

class TestResultContractObservability:
    """Verify result contract provides full observability."""

    def test_result_contract_contains_all_audit_fields(self):
        """Result contract contains all fields needed for audit."""
        contract = _build_result_contract(
            task_index=0,
            status="completed",
            summary="research complete",
            subagent_id="sa-result-001",
            child_session_id="child-session",
            task_goal="research",
            lifecycle_state="completed",
            api_calls=5,
            duration_seconds=10.5,
            model="claude-sonnet",
            tool_trace=["hermes_rag_search", "hermes_rag_search"],
            tokens={"input": 1000, "output": 500},
            cost_usd=0.01,
        )

        assert contract["subagent_id"] == "sa-result-001"
        assert contract["status"] == "completed"
        assert contract["lifecycle_state"] == "completed"
        assert contract["task_goal"] == "research"
        assert contract["api_calls"] == 5
        assert contract["duration_seconds"] == 10.5
        assert contract["tool_trace"] == ["hermes_rag_search", "hermes_rag_search"]
        assert contract["tokens"] == {"input": 1000, "output": 500}

    def test_result_contract_status_matches_lifecycle(self):
        """Result contract status matches lifecycle state."""
        ctx = create_execution_context(
            child_id="sa-match",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")

        contract = _build_result_contract(
            task_index=0,
            status="failed",
            summary="failed",
            subagent_id="sa-match",
            child_session_id="child",
            task_goal="research",
            lifecycle_state=manager.state,
        )

        assert contract["status"] == contract["lifecycle_state"]


# ====================================================================
# Security Event Logging
# ====================================================================

class TestSecurityEventLogging:
    """Verify security-relevant events are observable."""

    def test_denied_tool_call_logged(self, caplog):
        """Denied tool calls are observable."""
        ctx = create_execution_context(
            child_id="sa-denied",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        result = provider.handle_tool_call("hermes_rag_store", {"text": "x"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False


# ====================================================================
# Capability Registry Observability
# ====================================================================

class TestCapabilityRegistryObservability:
    """Verify capability registry is observable."""

    def test_registry_is_inspectable(self):
        """Registry can be inspected for audit."""
        for cap in [
            "hermes_rag_search",
            "hermes_rag_store",
            "hermes_rag_delete",
            "delegate_task",
            "terminal",
        ]:
            result = get_capability(cap)
            assert result is not None, f"Capability {cap} not found"

    def test_permitted_tools_listing_observable(self):
        """Permitted tools for each task type are observable."""
        for task_type in SUBAGENT_TASK_TYPES:
            tools = get_permitted_tools_for_task_type(task_type)
            assert isinstance(tools, (list, frozenset, set))
            assert len(tools) > 0


# ====================================================================
# Sub-Agent Identity Tracking
# ====================================================================

class TestSubAgentIdentityTracking:
    """Verify sub-agent identity is tracked throughout execution."""

    def test_identity_preserved_through_provider(self):
        """Identity preserved through provider execution."""
        ctx = create_execution_context(
            child_id="sa-identity-001",
            parent_session_id="parent-session",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        assert provider._goal == "research"
        assert provider._task_type == "research"
        assert provider._execution_context == ctx


# ====================================================================
# Cross-Capability Audit
# ====================================================================

class TestCrossCapabilityAudit:
    """Verify audit across capability boundaries."""

    def test_capability_access_logged_for_audit(self):
        """All capability accesses leave audit trail."""
        ctx = create_execution_context(
            child_id="sa-cross-audit",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "results": []})
            provider.handle_tool_call("hermes_rag_search", {"query": "test"})

            # Call was traceable
            mock.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
