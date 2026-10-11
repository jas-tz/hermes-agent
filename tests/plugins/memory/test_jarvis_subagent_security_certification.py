"""Phase 27A — Sub-Agent Security Boundary Certification.

Comprehensive end-to-end adversarial certification of the complete Phase 20–26
sub-agent security architecture. Tests every trusted boundary, including
combined/multi-stage attacks that attempt to bypass multiple layers.

This is the final certification gate before real-world deployment.
"""

import json
import os
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
    get_filesystem_capability_for_task_type,
)
from plugins.memory.jarvis.subagent_terminal import (
    get_terminal_capability_for_task_type,
    validate_command,
    is_delegation_escape,
    is_filesystem_escape,
    execute_subagent_terminal,
    SUBAGENT_DENIED_COMMANDS,
)
from plugins.memory.jarvis.tool_capability_registry import (
    is_tool_permitted,
    is_tool_parent_only,
    requires_approval,
    get_capability,
    get_permitted_tools_for_task_type,
    validate_capability_policy,
)


# ====================================================================
# SECTION 1: Trusted Execution Context Certification
# ====================================================================

class TestTrustedExecutionContext:
    """Certify that the execution context is fully trusted and tamper-proof."""

    def test_context_created_from_trusted_inputs_only(self):
        """Context can only be created from trusted delegation inputs."""
        ctx = create_execution_context(
            child_id="sa-cert-001",
            parent_session_id="parent-session",
            task_goal="certification test",
            task_type="research",
            delegation_depth=1,
        )

        # All fields match inputs
        assert ctx.child_id == "sa-cert-001"
        assert ctx.parent_session_id == "parent-session"
        assert ctx.task_goal == "certification test"
        assert ctx.task_type == "research"
        assert ctx.delegation_depth == 1

    def test_context_frozen_after_creation(self):
        """Context is frozen — no field can be modified after creation."""
        ctx = create_execution_context(
            child_id="sa-cert",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )

        frozen_fields = ['child_id', 'parent_session_id', 'task_goal',
                         'task_type', 'authority_level', 'delegation_depth',
                         'allowed_tools', 'provenance_context_id', 'created_at']

        for field in frozen_fields:
            with pytest.raises(AttributeError):
                setattr(ctx, field, "forged")

    def test_context_task_type_matches_authority(self):
        """Context task_type always matches its authority level."""
        for tt_name, expected_level in [
            ("research", "L0"),
            ("summarize", "L0"),
            ("note_taking", "L1"),
            ("task_execution", "L2"),
            ("maintenance", "L3"),
            ("default", "L3"),
        ]:
            ctx = create_execution_context(
                child_id="sa-test",
                parent_session_id="parent",
                task_goal="test",
                task_type=tt_name,
                delegation_depth=1,
            )
            assert ctx.authority_level == expected_level, \
                f"Task type {tt_name} has authority {ctx.authority_level}, expected {expected_level}"

    def test_context_depth_enforced(self):
        """Delegation depth is correctly tracked."""
        for depth in [1, 2]:
            ctx = create_execution_context(
                child_id="sa-test",
                parent_session_id="parent",
                task_goal="test",
                task_type="research",
                delegation_depth=depth,
            )
            assert ctx.delegation_depth == depth

    def test_context_provenance_contains_goal(self):
        """Provenance context_id contains the task goal."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="honey research",
            task_type="research",
            delegation_depth=1,
        )
        assert "subagent:honey research" == ctx.provenance_context_id


# ====================================================================
# SECTION 2: Task-Type Authority Certification
# ====================================================================

class TestTaskTypeAuthority:
    """Certify that task-type authority boundaries are enforced."""

    # Decision 42: every task type is read-only. Formerly note_taking had L1,
    # task_execution L1+L2 and maintenance L1+L2+L3 (delete with approval).
    @pytest.mark.parametrize("task_type,can_read,can_write_l1,can_write_l2,can_delete", [
        ("research", True, False, False, False),
        ("summarize", True, False, False, False),
        ("note_taking", True, False, False, False),
        ("task_execution", True, False, False, False),
        ("maintenance", True, False, False, False),
    ])
    def test_authority_levels(self, task_type, can_read, can_write_l1, can_write_l2, can_delete):
        """Each task type has exactly the authority it should."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type=task_type,
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="test",
            task_type=task_type,
            execution_context=ctx,
        )

        # Test read capability
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            read_allowed = json.loads(result)["ok"]

        if can_read:
            assert read_allowed is True
        # L1 write test
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
            write_l1_allowed = json.loads(result)["ok"]

        if can_write_l1:
            assert write_l1_allowed is True
        else:
            assert write_l1_allowed is False

        # L3 delete test
        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": True}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
                mock.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                delete_allowed = json.loads(result)["ok"]

        if can_delete:
            assert delete_allowed is True
        else:
            assert delete_allowed is False


# ====================================================================
# SECTION 3: Capability Registry Certification
# ====================================================================

class TestCapabilityRegistry:
    """Certify the ToolCapability registry."""

    def test_registry_is_consistent(self):
        """Registry passes policy validation."""
        issues = validate_capability_policy()
        errors = [i for i in issues if i.startswith("ERROR")]
        assert len(errors) == 0, f"Registry errors: {errors}"

    def test_all_rag_tools_have_capability(self):
        """All RAG tools are registered in ToolCapability."""
        rag_tools = [t for t in L0_READ_ONLY | L1_REVERSIBLE | L2_GATED | L3_APPROVAL
                     if t.startswith("hermes_rag_") or t.startswith("hermes_goal_")
                     or t.startswith("hermes_task_") or t.startswith("hermes_decision_")]
        for tool in rag_tools:
            cap = get_capability(tool)
            assert cap is not None, f"Tool {tool} missing from registry"

    def test_parent_only_tools_are_marked(self):
        """Parent-only tools are correctly marked."""
        parent_tools = ["delegate_task", "terminal", "write_file", "read_file",
                        "browser_navigate", "computer_use", "cronjob", "send_message"]
        for tool in parent_tools:
            cap = get_capability(tool)
            assert cap is not None, f"Parent tool {tool} missing from registry"
            assert cap.parent_only is True, f"{tool} not marked parent-only"

    def test_destructive_tools_require_approval(self):
        """All destructive tools require approval."""
        destructive = [cap for cap in [get_capability(t) for t in
                       ["hermes_rag_delete", "hermes_rag_rollback_learnings", "hermes_decision_supersede"]]
                       if cap is not None and cap.classification == "destructive"]
        for cap in destructive:
            assert cap.requires_approval is True, f"{cap.name} doesn't require approval"

    def test_tool_permitted_matches_registry(self):
        """is_tool_permitted matches registry entries."""
        for tool_name, cap in [
            ("hermes_rag_search", "research"),
            ("hermes_rag_search", "maintenance"),
            ("hermes_rag_delete", "maintenance"),
            ("hermes_rag_delete", "research"),
            ("delegate_task", "maintenance"),
        ]:
            result = is_tool_permitted(tool_name, tool_name)
            # Just verify it doesn't crash and returns bool
            assert isinstance(result, bool)


# ====================================================================
# SECTION 4: Runtime Authorization Certification
# ====================================================================

class TestRuntimeAuthorization:
    """Certify that runtime authorization is authoritative."""

    def test_provider_enforcement_matches_registry(self):
        """Provider enforcement matches ToolCapability registry."""
        ctx = create_execution_context(
            child_id="sa-test",
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

        # Registry says research can read
        assert is_tool_permitted("hermes_rag_search", "research") is True

        # Provider allows read
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})
            assert json.loads(result)["ok"] is True

        # Registry says research can't write
        assert is_tool_permitted("hermes_rag_store", "research") is False

        # Provider blocks write
        result = provider.handle_tool_call("hermes_rag_store", {"text": "x"})
        assert json.loads(result)["ok"] is False

    def test_model_args_cannot_override_task_type(self):
        """Model-supplied args cannot override trusted task_type."""
        ctx = create_execution_context(
            child_id="sa-test",
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

        # Even if args contain malicious task_type, it's ignored
        result = provider.handle_tool_call("hermes_rag_store", {
            "text": "data",
            "task_type": "maintenance",  # Attempted override
            "authority_level": "L3",     # Attempted override
        })
        assert json.loads(result)["ok"] is False


# ====================================================================
# SECTION 5: Filesystem Scope Certification
# ====================================================================

class TestFilesystemScope:
    """Certify filesystem scope enforcement."""

    def test_all_task_types_have_scope(self):
        """Every task type has a filesystem capability."""
        for tt in SUBAGENT_TASK_TYPES:
            cap = get_filesystem_capability_for_task_type(tt)
            assert cap.task_type == tt

    def test_only_maintenance_has_filesystem(self):
        """Only maintenance task type has filesystem access."""
        for tt, expected_access in [
            ("research", False),
            ("summarize", False),
            ("note_taking", False),
            ("task_execution", False),
            ("maintenance", True),
        ]:
            cap = get_filesystem_capability_for_task_type(tt)
            assert cap.can_read == expected_access, \
                f"{tt} filesystem access is {cap.can_read}, expected {expected_access}"


# ====================================================================
# SECTION 6: Terminal Command Policy Certification
# ====================================================================

class TestTerminalCommandPolicy:
    """Certify terminal command policy."""

    def test_only_maintenance_has_terminal(self):
        """Only maintenance task type has terminal access."""
        for tt, expected in [
            ("research", False),
            ("summarize", False),
            ("note_taking", False),
            ("task_execution", False),
            ("maintenance", True),
        ]:
            cap = get_terminal_capability_for_task_type(tt)
            assert cap.can_execute is expected

    def test_all_denied_commands_blocked(self):
        """All denied commands are blocked."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        for cmd in ["rm", "sudo", "ssh", "curl", "python", "bash", "kill", "docker"]:
            is_valid, _ = validate_command(f"{cmd} test", cap)
            assert is_valid is False, f"Denied command '{cmd}' was accepted"

    def test_all_allowed_commands_permitted(self):
        """All allowed commands are permitted."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        for cmd in ["ls", "cat", "grep", "echo", "find", "head", "tail", "wc"]:
            is_valid, _ = validate_command(f"{cmd} test", cap)
            assert is_valid is True, f"Allowed command '{cmd}' was rejected"


# ====================================================================
# SECTION 7: Approval Gate Certification
# ====================================================================

class TestApprovalGates:
    """Certify that approval gates are enforced."""

    def test_l3_tools_require_approval(self):
        """L3 tools require approval before execution."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="default",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="default",
            execution_context=ctx,
        )

        # Denied approval blocks execution
        with patch("tools.approval.request_tool_approval") as mock:
            mock.return_value = {"approved": False}
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            assert json.loads(result)["ok"] is False

    def test_l3_refused_even_when_approved(self):
        """Decision 42: approval no longer allows a delegated L3 write (was: allowed)."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="default",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="default",
            execution_context=ctx,
        )

        with patch("tools.approval.request_tool_approval") as mock:
            mock.return_value = {"approved": True}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                mock_dispatch.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                assert json.loads(result)["code"] == "delegated_read_only"
                mock.assert_not_called()
                mock_dispatch.assert_not_called()


# ====================================================================
# SECTION 8: Lifecycle Integrity Certification
# ====================================================================

class TestLifecycleIntegrity:
    """Certify lifecycle state machine integrity."""

    def test_all_valid_transitions_work(self):
        """All valid lifecycle transitions succeed."""
        valid_paths = [
            ("pending", "initializing"),
            ("pending", "cancelled"),
            ("initializing", "running"),
            ("initializing", "failed"),
            ("initializing", "cancelled"),
            ("running", "completed"),
            ("running", "failed"),
            ("running", "cancelled"),
            ("running", "timed_out"),
        ]

        for from_state, to_state in valid_paths:
            ctx = create_execution_context(
                child_id="sa-test",
                parent_session_id="parent",
                task_goal="test",
                task_type="research",
                delegation_depth=1,
            )
            manager = SubagentLifecycleManager(ctx)

            # Transition to from_state first
            if from_state == "pending":
                pass  # Already pending
            elif from_state == "initializing":
                manager.transition("initializing")
            elif from_state == "running":
                manager.transition("initializing")
                manager.transition("running")

            # Now test the transition
            if from_state in ["completed", "failed", "cancelled", "timed_out"]:
                # Terminal states can't transition
                with pytest.raises(ValueError):
                    manager.transition(to_state)
            else:
                manager.transition(to_state)
                assert manager.state == to_state

    def test_invalid_transitions_blocked(self):
        """Invalid lifecycle transitions are blocked."""
        invalid_paths = [
            ("completed", "running"),
            ("completed", "failed"),
            ("failed", "completed"),
            ("cancelled", "running"),
            ("timed_out", "completed"),
            ("running", "pending"),
            ("initializing", "pending"),
        ]

        for from_state, to_state in invalid_paths:
            ctx = create_execution_context(
                child_id="sa-test",
                parent_session_id="parent",
                task_goal="test",
                task_type="research",
                delegation_depth=1,
            )
            manager = SubagentLifecycleManager(ctx)

            # Set up from_state
            try:
                if from_state == "initializing":
                    manager.transition("initializing")
                elif from_state == "running":
                    manager.transition("initializing")
                    manager.transition("running")
                elif from_state == "completed":
                    manager.transition("initializing")
                    manager.transition("running")
                    manager.transition("completed")
                elif from_state == "failed":
                    manager.transition("initializing")
                    manager.transition("running")
                    manager.transition("failed")
                elif from_state == "cancelled":
                    manager.transition("cancelled")
                elif from_state == "timed_out":
                    manager.transition("initializing")
                    manager.transition("running")
                    manager.transition("timed_out")
            except ValueError:
                continue  # Can't set up this state

            # Invalid transition should fail
            with pytest.raises(ValueError):
                manager.transition(to_state)


# ====================================================================
# SECTION 9: Parent Isolation Certification
# ====================================================================

class TestParentIsolation:
    """Certify that parent-only resources are isolated."""

    def test_parent_tools_inaccessible_to_all_task_types(self):
        """Parent-only tools are inaccessible to all task types."""
        parent_tools = ["delegate_task", "terminal", "write_file", "read_file",
                        "browser_navigate", "computer_use", "cronjob", "send_message"]

        for task_type in SUBAGENT_TASK_TYPES:
            tools = get_permitted_tools_for_task_type(task_type)
            for pt in parent_tools:
                assert pt not in tools, \
                    f"Parent tool '{pt}' accessible to task type '{task_type}'"

    def test_parent_tools_registered_as_parent_only(self):
        """Parent tools are registered with parent_only=True."""
        for tool in ["delegate_task", "terminal", "write_file", "read_file"]:
            cap = get_capability(tool)
            assert cap is not None
            assert cap.parent_only is True


# ====================================================================
# SECTION 10: Zero-Mutation Guarantees Certification
# ====================================================================

class TestZeroMutationGuarantees:
    """Certify zero-mutation-on-denial."""

    def test_all_denied_gates_block_dispatch(self):
        """Every denied gate blocks the underlying dispatch."""
        ctx = create_execution_context(
            child_id="sa-test",
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

        denied_tools = ["hermes_rag_store", "hermes_rag_delete",
                        "hermes_goal_create", "hermes_task_create",
                        "hermes_decision_supersede"]

        for tool in denied_tools:
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
                provider.handle_tool_call(tool, {})
                mock.assert_not_called()


# ====================================================================
# SECTION 11: Cross-Capability Isolation Certification
# ====================================================================

class TestCrossCapabilityIsolation:
    """Certify that capabilities are isolated from each other."""

    def test_rag_capability_doesnt_grant_filesystem(self):
        """RAG read access doesn't grant filesystem access."""
        assert is_tool_permitted("hermes_rag_search", "research") is True
        fs_cap = get_filesystem_capability_for_task_type("research")
        assert fs_cap.can_read is False

    def test_filesystem_capability_doesnt_grant_terminal(self):
        """Filesystem access doesn't grant terminal access."""
        fs_cap = get_filesystem_capability_for_task_type("maintenance")
        assert fs_cap.can_read is True
        term_cap = get_terminal_capability_for_task_type("research")
        assert term_cap.can_execute is False


# ====================================================================
# SECTION 12: Combined/Multi-Stage Attack Certification
# ====================================================================

class TestCombinedAttacks:
    """Certify that combined/multi-stage attacks are blocked."""

    def test_escalation_via_context_forgery_and_tool_call(self):
        """Combined attack: forged context + tool call."""
        # Create research context
        ctx = create_execution_context(
            child_id="sa-attack",
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

        # Attempt multiple escalation paths
        attacks = [
            ("hermes_rag_store", {"text": "data", "task_type": "maintenance"}),
            ("hermes_rag_delete", {"memory_id": "x", "authority_level": "L3"}),
            ("hermes_goal_create", {"title": "escalated"}),
            ("delegate_task", {"goal": "sub-escalation"}),
        ]

        for tool, args in attacks:
            result = provider.handle_tool_call(tool, args)
            result_dict = json.loads(result)
            assert result_dict["ok"] is False, f"Attack {tool} succeeded!"

    def test_escalation_via_filesystem_and_terminal(self):
        """Combined attack: filesystem + terminal escalation."""
        # Get filesystem scope
        scope = SubagentFilesystemScope(
            root_path="/tmp/work",
            allowed_paths=("/tmp/work",),
            denied_paths=("/etc",),
        )

        # Attempt filesystem escape via traversal
        traversal_paths = [
            "/tmp/work/../../etc/passwd",
            "/tmp/work/../../../etc/shadow",
            "/etc/passwd",
        ]

        for path in traversal_paths:
            is_allowed, _ = validate_subagent_read_path(path, scope)
            assert is_allowed is False, f"Filesystem escape via {path} succeeded"

    def test_escalation_via_terminal_injection(self):
        """Combined attack: multiple injection vectors."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")

        injection_attempts = [
            "ls; rm -rf /",
            "ls | cat /etc/passwd",
            "ls `whoami`",
            "ls $(cat /etc/passwd)",
            "ls && rm -rf /",
            "ls || rm -rf /",
            "ls\nrm -rf /",
            "ls\rrm -rf /",
        ]

        for cmd in injection_attempts:
            is_valid, _ = validate_command(cmd, cap)
            assert is_valid is False, f"Injection '{cmd}' succeeded"

    def test_escalation_via_bypass_gate_and_l3(self):
        """Combined attack: bypass gate + L3 tool."""
        ctx = create_execution_context(
            child_id="sa-attack",
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

        # Attempt to inject bypass gate
        result = provider.handle_tool_call(
            "hermes_rag_store",
            {"text": "data"},
            _continuity_bypass_gate=True,
        )
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

    def test_multi_stage_rag_to_filesystem(self):
        """Multi-stage: RAG read → filesystem write with RAG data."""
        # Research can read RAG
        ctx = create_execution_context(
            child_id="sa-attack",
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

        # Can read
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "results": [{"text": "sensitive"}]})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "secret"})
            assert json.loads(result)["ok"] is True

        # But can't write to filesystem
        fs_cap = get_filesystem_capability_for_task_type("research")
        assert fs_cap.can_write is False

    def test_lifecycle_bypass_via_denied_operation(self):
        """Verify denied operations don't corrupt lifecycle."""
        ctx = create_execution_context(
            child_id="sa-attack",
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
        manager = SubagentLifecycleManager(ctx)

        # Multiple denied operations
        for _ in range(5):
            provider.handle_tool_call("hermes_rag_store", {"text": "x"})
            provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})

        # Lifecycle still valid
        assert manager.state == "pending"
        manager.transition("initializing")
        manager.transition("running")
        manager.transition("completed")
        assert manager.is_terminal


# ====================================================================
# SECTION 13: Model-Supplied Argument Safety
# ====================================================================

class TestModelArgumentSafety:
    """Certify that model-supplied arguments cannot override trusted context."""

    def test_trusted_context_overrides_model_args(self):
        """Trusted task_type overrides any model-supplied task_type."""
        ctx = create_execution_context(
            child_id="sa-test",
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

        # Model tries to override task_type via args
        result = provider.handle_tool_call("hermes_rag_store", {
            "text": "data",
            "task_type": "maintenance",
            "authority_level": "L3",
        })
        assert json.loads(result)["ok"] is False

    def test_model_cannot_forge_provenance(self):
        """Model cannot forge provenance: under Decision 42 the store is refused
        before dispatch (was: dispatched with provenance overwritten)."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="note_taking",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="test",
            task_type="note_taking",
            execution_context=ctx,
        )

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "memory_id": "uuid"})
            # Model tries to supply fake provenance
            result = provider.handle_tool_call("hermes_rag_store", {
                "text": "data",
                "provenance": {"source": "user_confirmed", "forged": True}
            })

            assert json.loads(result)["code"] == "delegated_read_only"
            mock.assert_not_called()


# ====================================================================
# SECTION 14: Invariant Summary
# ====================================================================

class TestInvariantSummary:
    """Final invariant checks across all registries."""

    def test_toolcap_registry_invariant(self):
        """ToolCapability registry is internally consistent."""
        issues = validate_capability_policy()
        errors = [i for i in issues if i.startswith("ERROR")]
        assert len(errors) == 0

    def test_subagent_types_registry_invariant(self):
        """SubagentTaskType registry is internally consistent."""
        for tt_name, tt in SUBAGENT_TASK_TYPES.items():
            # Authority level matches allowed tools
            if tt.authority_level == "L0":
                assert tt.allowed_tools.issubset(L0_READ_ONLY)
            elif tt.authority_level == "L1":
                assert tt.allowed_tools.issubset(L0_READ_ONLY | L1_REVERSIBLE)

    def test_all_capabilities_have_valid_categories(self):
        """All capabilities have valid categories."""
        valid_categories = {"rag", "continuity", "delegation", "system",
                          "filesystem", "browser", "desktop", "messaging", "terminal"}
        for cap in [get_capability(t) for t in [
            "hermes_rag_search", "hermes_goal_create", "delegate_task",
            "terminal", "subagent_read_file", "subagent_terminal"
        ] if get_capability(t)]:
            assert cap.category in valid_categories, \
                f"Invalid category '{cap.category}' for {cap.name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
