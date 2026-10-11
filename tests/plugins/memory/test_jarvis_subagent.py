"""Tests for JARVISSubagentProvider — Phase 20B/20C authority & isolation."""

import json
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis import JARVISSubagentProvider


@pytest.fixture
def provider():
    """Create a standalone JARVISSubagentProvider (no parent needed)."""
    return JARVISSubagentProvider("test goal", "test context")


class TestJARVISSubagentToolRestrictions:
    """Phase 20B: explicit tool whitelist."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l0_tools_allowed(self, mock_dispatch, provider):
        """L0 read-only tools are allowed."""
        mock_dispatch.return_value = json.dumps({"ok": True})
        for tool in ["hermes_rag_search", "hermes_goal_list", "hermes_task_list",
                      "hermes_decision_list", "hermes_rag_retrieve_learnings",
                      "hermes_rag_extract_learnings"]:
            result = provider.handle_tool_call(tool, {})
            assert json.loads(result)["ok"] is True, f"Tool {tool} should be allowed"

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l1_tools_refused(self, mock_dispatch, provider):
        """Decision 42: L1 writes (formerly allowed) are refused before dispatch."""
        mock_dispatch.return_value = json.dumps({"ok": True})
        for tool in ["hermes_rag_store", "hermes_rag_store_learnings",
                      "hermes_goal_complete", "hermes_task_complete"]:
            result = json.loads(provider.handle_tool_call(tool, {}))
            assert result["ok"] is False and result["code"] == "delegated_read_only", tool
        mock_dispatch.assert_not_called()

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l2_tools_refused(self, mock_dispatch, provider):
        """Decision 42: L2 writes no longer fall through to the parent's gate."""
        mock_dispatch.return_value = json.dumps({"ok": True})
        for tool in ["hermes_goal_create", "hermes_task_create",
                      "hermes_decision_record"]:
            result = json.loads(provider.handle_tool_call(tool, {}))
            assert result["ok"] is False and result["code"] == "delegated_read_only", tool
        mock_dispatch.assert_not_called()

    def test_unknown_tool_blocked(self, provider):
        """Unknown tools are blocked."""
        result = provider.handle_tool_call("unknown_tool", {})
        assert json.loads(result)["ok"] is False
        assert "not registered" in json.loads(result)["error"]


class TestJARVISSubagentL3Approval:
    """Decision 42: L3 tools are refused before approval is ever requested
    (previously: executed when approved, blocked when not)."""

    @patch("tools.approval.request_tool_approval")
    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_l3_refused_even_if_approval_would_pass(self, mock_dispatch, mock_approval, provider):
        mock_approval.return_value = {"approved": True}
        mock_dispatch.return_value = json.dumps({"ok": True})
        result = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"
        mock_approval.assert_not_called()
        mock_dispatch.assert_not_called()

    @patch("tools.approval.request_tool_approval")
    def test_l3_delete_refused_without_approval_request(self, mock_approval, provider):
        mock_approval.return_value = {"approved": False}
        result = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": "test"}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"
        mock_approval.assert_not_called()

    @patch("tools.approval.request_tool_approval")
    def test_l3_rollback_refused_without_approval_request(self, mock_approval, provider):
        mock_approval.return_value = {"approved": False}
        result = json.loads(provider.handle_tool_call("hermes_rag_rollback_learnings", {}))
        assert result["ok"] is False and result["code"] == "delegated_read_only"
        mock_approval.assert_not_called()


class TestJARVISSubagentBypassGateProtection:
    """Phase 20B: sub-agent cannot inherit parent's _continuity_bypass_gate."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_bypass_gate_stripped(self, mock_dispatch, provider):
        """_continuity_bypass_gate is stripped from kwargs."""
        mock_dispatch.return_value = json.dumps({"ok": True})
        provider.handle_tool_call("hermes_rag_search", {},
                                   _continuity_bypass_gate=True)
        mock_dispatch.assert_called_once()


class TestJARVISSubagentProvenanceIsolation:
    """Phase 20B: provenance isolation via deep copy."""

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_args_not_mutated(self, mock_dispatch, provider):
        """Args dict is not mutated."""
        mock_dispatch.return_value = json.dumps({"ok": True})
        original_args = {"text": "test", "provenance": {"source": "conversation"}}
        original_copy = json.loads(json.dumps(original_args))
        provider.handle_tool_call("hermes_rag_store", original_args)
        assert original_args == original_copy, "Args were mutated!"

    @patch("plugins.memory.jarvis._dispatch_jarvis_tool_call")
    def test_store_with_provenance_never_dispatched(self, mock_dispatch, provider):
        """Decision 42: a delegated store is refused, so no provenance-stamped
        write reaches dispatch (previously: dispatched with sub-agent provenance)."""
        mock_dispatch.return_value = json.dumps({"ok": True})
        result = json.loads(provider.handle_tool_call(
            "hermes_rag_store", {"text": "test", "provenance": {"source": "conversation"}}))
        assert result["code"] == "delegated_read_only"
        mock_dispatch.assert_not_called()


class TestJARVISSubagentMethods:
    """Phase 20B: other methods work correctly."""

    @patch("plugins.memory.jarvis._prefetch_rag")
    def test_prefetch(self, mock_prefetch, provider):
        """Prefetch returns context."""
        mock_prefetch.return_value = "prefetched"
        result = provider.prefetch("test query")
        assert result == "prefetched"
        mock_prefetch.assert_called_once_with("test query")

    @patch("plugins.memory.jarvis._on_session_end_learning")
    def test_on_session_end(self, mock_end, provider):
        """Decision 42: on_session_end no longer delegates to session-end learning."""
        provider.on_session_end([{"role": "user", "content": "test"}])
        mock_end.assert_not_called()

    def test_get_continuity_block(self, provider):
        """get_continuity_block returns a string."""
        result = provider.get_continuity_block()
        assert isinstance(result, str)


class TestJARVISSubagentDelegationWiring:
    """Phase 20C: test delegation wiring helpers."""

    def test_get_jarvis_continuity_block_with_jarvis_parent(self):
        """Returns continuity block when parent has JARVIS provider."""
        from tools.delegate_tool import _get_jarvis_continuity_block
        parent = MagicMock()
        provider_mock = MagicMock()
        provider_mock.__class__.__name__ = "HermesRAGProvider"
        provider_mock.get_continuity_block.return_value = "test block"
        parent._memory_manager._providers = [provider_mock]
        result = _get_jarvis_continuity_block(parent)
        assert result == "test block"

    def test_get_jarvis_continuity_block_without_jarvis(self):
        """Returns None when parent has no JARVIS provider."""
        from tools.delegate_tool import _get_jarvis_continuity_block
        parent = MagicMock()
        parent._memory_manager._providers = []
        result = _get_jarvis_continuity_block(parent)
        assert result is None

    def test_get_jarvis_continuity_block_no_memory_manager(self):
        """Returns None when parent has no memory manager."""
        from tools.delegate_tool import _get_jarvis_continuity_block
        parent = MagicMock(spec=[])
        result = _get_jarvis_continuity_block(parent)
        assert result is None

    def test_attach_jarvis_subagent_provider_no_parent(self):
        """Does nothing when parent is None."""
        from tools.delegate_tool import _attach_jarvis_subagent_provider
        _attach_jarvis_subagent_provider(None, MagicMock(), "goal")

    def test_attach_jarvis_subagent_provider_no_child(self):
        """Does nothing when child is None."""
        from tools.delegate_tool import _attach_jarvis_subagent_provider
        _attach_jarvis_subagent_provider(MagicMock(), None, "goal")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
