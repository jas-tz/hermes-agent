"""Phase 24B — Real Sub-Agent Tool Execution.

Tests real multi-step child-agent execution for every registered task type.
Uses mocked backends (Qdrant, approval) while exercising the real provider,
lifecycle, and execution context code.

Covers: research, summarize, note_taking, task_execution, maintenance.
"""

import json
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


# ---------------------------------------------------------------------------
# Research task execution
# ---------------------------------------------------------------------------

class TestResearchTaskExecution:
    """Test research task type: L0 read-only."""

    @pytest.fixture
    def research_ctx(self):
        return create_execution_context(
            child_id="sa-research",
            parent_session_id="parent",
            task_goal="research honey production methods",
            task_type="research",
            delegation_depth=1,
        )

    @pytest.fixture
    def research_provider(self, research_ctx):
        return JARVISSubagentProvider(
            goal="research honey production methods",
            task_type="research",
            execution_context=research_ctx,
        )

    def test_research_can_search_rag(self, research_provider):
        """Research child can perform RAG search."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "results": [
                {"score": 0.9, "text": "Honey production basics"},
                {"score": 0.85, "text": "Beekeeping techniques"},
            ]})
            result = research_provider.handle_tool_call("hermes_rag_search", {"query": "honey"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is True
            assert len(result_dict["results"]) == 2

    def test_research_can_retrieve_learnings(self, research_provider):
        """Research child can retrieve cross-session learnings."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "results": [{"text": "learned"}]})
            result = research_provider.handle_tool_call("hermes_rag_retrieve_learnings", {"query": "test"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is True

    def test_research_can_list_goals(self, research_provider):
        """Research child can list continuity goals."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "goals": []})
            result = research_provider.handle_tool_call("hermes_goal_list", {})
            result_dict = json.loads(result)
            assert result_dict["ok"] is True

    def test_research_cannot_store(self, research_provider):
        """Research child cannot store RAG memories (L1)."""
        result = research_provider.handle_tool_call("hermes_rag_store", {"text": "findings"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False
        assert "not authorized" in result_dict["error"]

    def test_research_cannot_create_goal(self, research_provider):
        """Research child cannot create goals (L2)."""
        result = research_provider.handle_tool_call("hermes_goal_create", {"title": "new"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

    def test_research_cannot_delete(self, research_provider):
        """Research child cannot delete memories (L3)."""
        result = research_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
        result_dict = json.loads(result)
        assert result_dict["ok"] is False

    def test_research_multiple_sequential_reads(self, research_provider):
        """Research child can perform multiple sequential reads."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "results": []})

            # First read
            result1 = research_provider.handle_tool_call("hermes_rag_search", {"query": "query1"})
            assert json.loads(result1)["ok"] is True

            # Second read
            result2 = research_provider.handle_tool_call("hermes_rag_search", {"query": "query2"})
            assert json.loads(result2)["ok"] is True

            # Third read (different tool)
            result3 = research_provider.handle_tool_call("hermes_goal_list", {})
            assert json.loads(result3)["ok"] is True

            # All three calls should have been dispatched
            assert mock.call_count == 3


# ---------------------------------------------------------------------------
# Summarize task execution
# ---------------------------------------------------------------------------

class TestSummarizeTaskExecution:
    """Test summarize task type: L0 read-only with synthesis."""

    @pytest.fixture
    def summarize_ctx(self):
        return create_execution_context(
            child_id="sa-summarize",
            parent_session_id="parent",
            task_goal="summarize honey research",
            task_type="summarize",
            delegation_depth=1,
        )

    @pytest.fixture
    def summarize_provider(self, summarize_ctx):
        return JARVISSubagentProvider(
            goal="summarize honey research",
            task_type="summarize",
            execution_context=summarize_ctx,
        )

    def test_summarize_can_read(self, summarize_provider):
        """Summarize child can read RAG data."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "results": [
                {"text": "finding 1"}, {"text": "finding 2"},
            ]})
            result = summarize_provider.handle_tool_call("hermes_rag_search", {"query": "honey"})
            assert json.loads(result)["ok"] is True

    def test_summarize_can_extract_learnings(self, summarize_provider):
        """Summarize child can extract learnings."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "learnings": []})
            result = summarize_provider.handle_tool_call("hermes_rag_extract_learnings", {})
            assert json.loads(result)["ok"] is True

    def test_summarize_cannot_write(self, summarize_provider):
        """Summarize child remains read-only."""
        result = summarize_provider.handle_tool_call("hermes_rag_store", {"text": "summary"})
        assert json.loads(result)["ok"] is False


# ---------------------------------------------------------------------------
# Note-taking task execution
# ---------------------------------------------------------------------------

class TestNoteTakingTaskExecution:
    """Test note-taking task type: L1 reversible writes."""

    @pytest.fixture
    def note_ctx(self):
        return create_execution_context(
            child_id="sa-note",
            parent_session_id="parent",
            task_goal="record findings about honey",
            task_type="note_taking",
            delegation_depth=1,
        )

    @pytest.fixture
    def note_provider(self, note_ctx):
        return JARVISSubagentProvider(
            goal="record findings about honey",
            task_type="note_taking",
            execution_context=note_ctx,
        )

    def test_note_taking_can_read(self, note_provider):
        """Note-taking child can read."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "results": []})
            result = note_provider.handle_tool_call("hermes_rag_search", {"query": "honey"})
            assert json.loads(result)["ok"] is True

    def test_note_taking_cannot_store(self, note_provider):
        """Decision 42: note-taking child can no longer store (formerly L1-allowed)."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "memory_id": "note-uuid-123"})
            result = note_provider.handle_tool_call("hermes_rag_store", {"text": "important finding"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False and result_dict["code"] == "delegated_read_only"
            mock.assert_not_called()

    def test_note_taking_cannot_complete_goal(self, note_provider):
        """Decision 42: completing a goal is a continuity write (formerly L1-allowed)."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "goal": {"id": "g_test", "status": "done"}})
            result_dict = json.loads(note_provider.handle_tool_call("hermes_goal_complete", {"goal_id": "g_test"}))
            assert result_dict["ok"] is False and result_dict["code"] == "delegated_read_only"
            mock.assert_not_called()

    def test_note_taking_cannot_create_goal(self, note_provider):
        """Note-taking child cannot create goals (L2)."""
        result = note_provider.handle_tool_call("hermes_goal_create", {"title": "new"})
        assert json.loads(result)["ok"] is False

    def test_note_taking_cannot_delete(self, note_provider):
        """Note-taking child cannot delete (L3)."""
        result = note_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
        assert json.loads(result)["ok"] is False

    def test_note_taking_store_with_provenance_refused(self, note_provider):
        """Decision 42: no sub-agent store reaches dispatch, so there is no
        provenance to preserve (previously: dispatched with sub-agent context)."""
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "memory_id": "uuid"})
            result_dict = json.loads(note_provider.handle_tool_call(
                "hermes_rag_store",
                {"text": "finding", "provenance": {"source": "conversation"}}
            ))
            assert result_dict["ok"] is False and result_dict["code"] == "delegated_read_only"
            mock.assert_not_called()


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

class TestTaskExecutionTaskExecution:
    """Test task_execution task type: L2 gated writes."""

    @pytest.fixture
    def exec_ctx(self):
        return create_execution_context(
            child_id="sa-exec",
            parent_session_id="parent",
            task_goal="execute honey production task",
            task_type="task_execution",
            delegation_depth=1,
        )

    @pytest.fixture
    def exec_provider(self, exec_ctx):
        return JARVISSubagentProvider(
            goal="execute honey production task",
            task_type="task_execution",
            execution_context=exec_ctx,
        )

    # Decision 42: the three L2 continuity writes below were formerly allowed.
    @pytest.mark.parametrize("tool,args", [
        ("hermes_goal_create", {"title": "new goal"}),
        ("hermes_task_create", {"title": "new task"}),
        ("hermes_decision_record", {"decision": "test"}),
    ])
    def test_task_execution_cannot_write_continuity(self, exec_provider, tool, args):
        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            result_dict = json.loads(exec_provider.handle_tool_call(tool, args))
            assert result_dict["ok"] is False and result_dict["code"] == "delegated_read_only"
            mock.assert_not_called()

    def test_task_execution_cannot_delete(self, exec_provider):
        """Task execution child cannot delete (L3)."""
        result = exec_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
        assert json.loads(result)["ok"] is False

    def test_task_execution_cannot_supersede(self, exec_provider):
        """Task execution child cannot supersede decisions (L3)."""
        result = exec_provider.handle_tool_call("hermes_decision_supersede", {"decision_id": "x", "decision": "y"})
        assert json.loads(result)["ok"] is False


# ---------------------------------------------------------------------------
# Maintenance task execution
# ---------------------------------------------------------------------------

class TestMaintenanceTaskExecution:
    """Test maintenance task type: L3 approval-required operations."""

    @pytest.fixture
    def maint_ctx(self):
        return create_execution_context(
            child_id="sa-maint",
            parent_session_id="parent",
            task_goal="maintain honey database",
            task_type="maintenance",
            delegation_depth=1,
        )

    @pytest.fixture
    def maint_provider(self, maint_ctx):
        return JARVISSubagentProvider(
            goal="maintain honey database",
            task_type="maintenance",
            execution_context=maint_ctx,
        )

    def test_maintenance_cannot_delete_even_with_approval(self, maint_provider):
        """Decision 42: refused before approval (formerly: deleted when approved)."""
        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": True}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                mock_dispatch.return_value = json.dumps({"ok": True})
                result = maint_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                result_dict = json.loads(result)
                assert result_dict["ok"] is False and result_dict["code"] == "delegated_read_only"
                mock_approval.assert_not_called()
                mock_dispatch.assert_not_called()

    def test_maintenance_cannot_delete_without_approval(self, maint_provider):
        """Maintenance child cannot delete; Decision 42 refuses before approval
        (formerly: "not approved")."""
        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": False}
            result = maint_provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
            result_dict = json.loads(result)
            assert result_dict["ok"] is False and result_dict["code"] == "delegated_read_only"
            mock_approval.assert_not_called()

    def test_maintenance_cannot_supersede_even_with_approval(self, maint_provider):
        """Decision 42: refused before approval (formerly: superseded when approved)."""
        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": True}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                mock_dispatch.return_value = json.dumps({"ok": True})
                result_dict = json.loads(maint_provider.handle_tool_call(
                    "hermes_decision_supersede",
                    {"decision_id": "d_old", "decision": "new decision"}
                ))
                assert result_dict["ok"] is False and result_dict["code"] == "delegated_read_only"
                mock_approval.assert_not_called()
                mock_dispatch.assert_not_called()

    def test_maintenance_denied_approval_zero_mutation(self, maint_provider):
        """Denied approval produces zero mutation."""
        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": False}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                result = maint_provider.handle_tool_call("hermes_rag_rollback_learnings", {})
                result_dict = json.loads(result)
                assert result_dict["ok"] is False
                # Dispatch should NOT have been called — approval came first
                mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# Multi-step execution
# ---------------------------------------------------------------------------

class TestMultiStepExecution:
    """Test multi-step tool calls within one child."""

    def test_research_multi_step_read_only(self):
        """Research child can perform multiple sequential reads."""
        ctx = create_execution_context(
            child_id="sa-multi",
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

            # Step 1: Search RAG
            provider.handle_tool_call("hermes_rag_search", {"query": "step1"})

            # Step 2: List goals
            provider.handle_tool_call("hermes_goal_list", {})

            # Step 3: Extract learnings
            provider.handle_tool_call("hermes_rag_extract_learnings", {})

            # Step 4: List tasks
            provider.handle_tool_call("hermes_task_list", {})

            # Step 5: List decisions
            provider.handle_tool_call("hermes_decision_list", {})

            assert mock.call_count == 5

    def test_note_taking_read_then_write(self):
        """Note-taking child can read; the write is refused (Decision 42;
        formerly both were dispatched)."""
        ctx = create_execution_context(
            child_id="sa-note-multi",
            parent_session_id="parent",
            task_goal="take notes",
            task_type="note_taking",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="take notes",
            task_type="note_taking",
            execution_context=ctx,
        )

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "results": []})

            # Step 1: Read
            provider.handle_tool_call("hermes_rag_search", {"query": "honey"})

            # Step 2: Store is refused before dispatch
            mock.return_value = json.dumps({"ok": True, "memory_id": "note-1"})
            r = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": "finding"}))
            assert r["code"] == "delegated_read_only"

            assert mock.call_count == 1

    def test_task_execution_read_write_update(self):
        """Task execution child can read; create and update are refused
        (Decision 42; formerly all three were dispatched)."""
        ctx = create_execution_context(
            child_id="sa-exec-multi",
            parent_session_id="parent",
            task_goal="execute",
            task_type="task_execution",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="execute",
            task_type="task_execution",
            execution_context=ctx,
        )

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            # Step 1: Read
            mock.return_value = json.dumps({"ok": True, "goals": []})
            provider.handle_tool_call("hermes_goal_list", {})

            # Steps 2-3: Create and update are refused before dispatch
            mock.return_value = json.dumps({"ok": True, "goal": {"id": "g_new"}})
            r = json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "new"}))
            assert r["code"] == "delegated_read_only"
            r = json.loads(provider.handle_tool_call("hermes_goal_update", {"goal_id": "g_new", "title": "updated"}))
            assert r["code"] == "delegated_read_only"

            assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Tool result returned to child
# ---------------------------------------------------------------------------

class TestToolResultReturn:
    """Test that tool results are returned correctly to the child."""

    def test_tool_result_returned_as_json_string(self):
        """Tool results are returned as JSON strings."""
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

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True, "data": "result"})
            result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})

            # Result should be a JSON string
            assert isinstance(result, str)
            parsed = json.loads(result)
            assert parsed["ok"] is True
            assert parsed["data"] == "result"

    def test_tool_error_returned_as_json(self):
        """Tool errors are returned as JSON."""
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

        # Unauthorized tool returns error JSON
        result = provider.handle_tool_call("hermes_rag_store", {"text": "test"})
        parsed = json.loads(result)
        assert parsed["ok"] is False
        assert "error" in parsed


# ---------------------------------------------------------------------------
# Forged tool call prevention
# ---------------------------------------------------------------------------

class TestForgedToolPrevention:
    """Test that unauthorized tools cannot be reached by forged calls."""

    def test_research_forged_store_rejected(self):
        """Research child cannot forge hermes_rag_store."""
        ctx = create_execution_context(
            child_id="sa-research",
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

        result = provider.handle_tool_call("hermes_rag_store", {"text": "forged"})
        assert json.loads(result)["ok"] is False

    def test_research_forged_delete_rejected(self):
        """Research child cannot forge hermes_rag_delete."""
        ctx = create_execution_context(
            child_id="sa-research",
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

        result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
        assert json.loads(result)["ok"] is False

    def test_unknown_tool_rejected(self):
        """Unknown JARVIS tools are rejected."""
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

        result = provider.handle_tool_call("hermes_rag_nonexistent", {})
        assert json.loads(result)["ok"] is False

    def test_parent_only_tool_terminal_rejected(self):
        """Parent-only tools like 'delegate_task' are rejected."""
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

        result = provider.handle_tool_call("delegate_task", {"goal": "sub"})
        assert json.loads(result)["ok"] is False


# ---------------------------------------------------------------------------
# Bypass gate stripping
# ---------------------------------------------------------------------------

class TestBypassGateStripping:
    """Test that _continuity_bypass_gate is stripped."""

    def test_bypass_gate_stripped(self):
        """_continuity_bypass_gate is stripped from kwargs."""
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

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            provider.handle_tool_call("hermes_rag_search", {"query": "test"},
                                      _continuity_bypass_gate=True)
            # Should have been called without bypass
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# Deep copy isolation
# ---------------------------------------------------------------------------

class TestDeepCopyIsolation:
    """Test that args are deep-copied to prevent parent mutation."""

    def test_args_not_mutated(self):
        """Original args dict is not mutated by the provider."""
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

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            mock.return_value = json.dumps({"ok": True})
            original = {"query": "test", "nested": {"key": "value"}}
            original_copy = json.loads(json.dumps(original))
            provider.handle_tool_call("hermes_rag_search", original)
            assert original == original_copy


# ---------------------------------------------------------------------------
# Provenance isolation
# ---------------------------------------------------------------------------

class TestProvenanceIsolation:
    """Test that provenance remains isolated."""

    def test_child_provenance_in_context_id(self):
        """Child provenance context_id contains sub-agent goal."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research honey",
            task_type="research",
            delegation_depth=1,
        )
        assert "subagent:research" in ctx.provenance_context_id

    def test_child_provenance_no_parent_leak(self):
        """Child provenance doesn't leak parent session ID."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent-secret-123",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        assert "parent-secret" not in ctx.provenance_context_id


# ---------------------------------------------------------------------------
# Recursion limit
# ---------------------------------------------------------------------------

class TestRecursionLimit:
    """Test that recursion-depth enforcement works."""

    def test_max_depth_constant(self):
        """Maximum recursion depth is 2."""
        assert JARVISSubagentProvider._MAX_SUBAGENT_DEPTH == 2

    def test_depth_tracked_in_context(self):
        """Delegation depth is tracked in execution context."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=2,
        )
        assert ctx.delegation_depth == 2


# ---------------------------------------------------------------------------
# Context immutability
# ---------------------------------------------------------------------------

class TestContextImmutability:
    """Test that execution context is immutable."""

    def test_context_frozen(self):
        """Execution context is frozen."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )

        with pytest.raises(AttributeError):
            ctx.task_type = "maintenance"


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------

class TestLifecycleTransitions:
    """Test lifecycle transitions during execution."""

    def test_full_lifecycle_success(self):
        """Full lifecycle: pending → initializing → running → completed."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)

        assert manager.state == "pending"
        manager.transition("initializing")
        assert manager.state == "initializing"
        manager.transition("running")
        assert manager.state == "running"
        manager.transition("completed")
        assert manager.state == "completed"
        assert manager.is_terminal

    def test_full_lifecycle_failure(self):
        """Full lifecycle: pending → initializing → running → failed."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)

        manager.transition("initializing")
        manager.transition("running")
        manager.transition("failed")
        assert manager.is_terminal


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """Test that cleanup occurs after execution."""

    def test_lifecycle_manager_cleanup(self):
        """Lifecycle manager can be cleaned up."""
        from tools.delegate_tool import register_lifecycle_manager, unregister_lifecycle_manager, _get_lifecycle_state_for_subagent

        ctx = create_execution_context(
            child_id="sa-cleanup",
            parent_session_id="parent",
            task_goal="test",
            task_type="research",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)
        manager.transition("initializing")
        register_lifecycle_manager("sa-cleanup", manager)

        assert _get_lifecycle_state_for_subagent("sa-cleanup") == "initializing"

        unregister_lifecycle_manager("sa-cleanup")
        assert _get_lifecycle_state_for_subagent("sa-cleanup") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
