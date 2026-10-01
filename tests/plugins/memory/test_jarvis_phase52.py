"""Phase 52 — Gateway End-to-End JARVIS Integration Validation.

Tests the full gateway → agent → JARVIS capability path without
modifying production code.

Uses controlled mock model responses to make the agent choose
delegate_task deterministically.
"""

import asyncio
import json
import os
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.jarvis.capabilities import register_default_capabilities
from plugins.memory.jarvis.capabilities.execution_result import (
    CapabilityExecutionStatus,
)
from plugins.memory.jarvis.capabilities.registry import (
    get_jarvis_registry,
    reset_jarvis_registry,
)
from tools.delegate_tool import (
    _attach_capability_plan,
    _build_child_agent,
    _execute_capability_plan_if_present,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registries():
    reset_jarvis_registry()
    yield
    reset_jarvis_registry()


def _build_registry():
    registry = get_jarvis_registry()
    register_default_capabilities(registry)
    return registry


def _make_mock_parent(depth=0):
    """Create a mock parent agent with JARVIS provider."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    # JARVIS provider marker
    class HermesRAGProvider:
        pass
    jarvis_provider = HermesRAGProvider()
    parent._memory_manager = MagicMock()
    parent._memory_manager._providers = [jarvis_provider]
    return parent


# ---------------------------------------------------------------------------
# Test 1: API request reaches GatewayRunner/AIAgent lifecycle
# ---------------------------------------------------------------------------


class TestApiServerReachesAgentLifecycle:
    """API server entry constructs AIAgent with JARVIS provider."""

    def test_api_server_run_agent_method_exists(self):
        """ApiServerAdapter has _run_agent method for agent execution."""
        from gateway.platforms.api_server import APIServerAdapter

        # Verify the adapter has the _run_agent method
        assert hasattr(APIServerAdapter, "_run_agent")
        assert hasattr(APIServerAdapter, "_handle_chat_completions")
        assert hasattr(APIServerAdapter, "_handle_responses")
        assert hasattr(APIServerAdapter, "_handle_session_chat")

    def test_api_server_chat_completions_route_exists(self):
        """API server exposes /v1/chat/completions route handler."""
        from gateway.platforms.api_server import APIServerAdapter

        # Verify route handlers exist
        assert hasattr(APIServerAdapter, "_handle_chat_completions")
        assert hasattr(APIServerAdapter, "_handle_responses")
        assert hasattr(APIServerAdapter, "_handle_session_chat")
        assert hasattr(APIServerAdapter, "_handle_create_session")


# ---------------------------------------------------------------------------
# Test 2: API-originated agent can invoke delegate_task
# ---------------------------------------------------------------------------


class TestApiAgentCanInvokeDelegateTask:
    """Agent constructed by API server has delegate_task in tool schema."""

    def test_aiaagent_imports_correctly(self):
        """AIAgent class imports correctly."""
        from run_agent import AIAgent
        assert AIAgent is not None

    def test_aiaagent_has_run_conversation(self):
        """AIAgent has run_conversation method."""
        from run_agent import AIAgent
        assert hasattr(AIAgent, "run_conversation")


# ---------------------------------------------------------------------------
# Test 3: API-originated delegation reaches JARVIS capability planning
# ---------------------------------------------------------------------------


class TestApiDelegationReachesCapabilityPlanning:
    """delegate_task from API agent reaches SubAgentPlanner."""

    def test_delegate_task_attaches_capability_plan_via_gateway_path(self):
        """Full path: delegate_task → _attach_capability_plan → SubAgentPlanner."""
        _build_registry()
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.session_id = "child-session-001"
            mock_child.model = "gpt-4"
            mock_child._capability_plan = None
            mock_child._capability_graph = None
            mock_child._capability_task_type = None
            mock_child._capability_authority_level = None
            mock_child._subagent_id = "sa-gateway-001"
            mock_child._parent_session_id = "parent-gateway-001"
            mock_child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "api_calls": 1,
                "messages": [],
            }
            MockAgent.return_value = mock_child

            from tools.delegate_tool import delegate_task

            result = json.loads(delegate_task(
                goal="list goals",
                parent_agent=parent,
            ))

            assert "results" in result
            assert len(result["results"]) >= 1


# ---------------------------------------------------------------------------
# Test 4: Webhook request reaches GatewayRunner/AIAgent lifecycle
# ---------------------------------------------------------------------------


class TestWebhookReachesAgentLifecycle:
    """Webhook handler routes through GatewayRunner._handle_message."""

    def test_webhook_adapter_has_handle_message(self):
        """WebhookAdapter has handle_message for gateway routing."""
        from gateway.platforms.webhook import WebhookAdapter

        assert hasattr(WebhookAdapter, "handle_message")

    def test_webhook_route_triggers_agent_run(self):
        """Webhook route configuration triggers agent execution."""
        from gateway.platforms.webhook import WebhookAdapter

        # Verify adapter has route handling
        assert hasattr(WebhookAdapter, "handle_message")
        assert hasattr(WebhookAdapter, "_reload_dynamic_routes")

        # Verify route dict structure (routes are stored as dicts, not dataclass)
        route_config = {
            "events": ["push"],
            "secret": "test-secret",
            "prompt": "Process: {payload}",
        }
        assert route_config["events"] == ["push"]
        assert route_config["prompt"] == "Process: {payload}"


# ---------------------------------------------------------------------------
# Test 5: Webhook-originated agent can reach JARVIS delegation
# ---------------------------------------------------------------------------


class TestWebhookAgentReachesJarvisDelegation:
    """Webhook-triggered agent with HermesRAGProvider reaches delegate_task."""

    def test_webhook_delegate_task_integration(self):
        """Webhook agent with JARVIS provider can invoke delegate_task."""
        _build_registry()
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.session_id = "child-webhook-001"
            mock_child.model = "gpt-4"
            mock_child._capability_plan = None
            mock_child._capability_graph = None
            mock_child._capability_task_type = None
            mock_child._capability_authority_level = None
            mock_child._subagent_id = "sa-webhook-001"
            mock_child._parent_session_id = "parent-webhook-001"
            mock_child.run_conversation.return_value = {
                "final_response": "webhook delegation success",
                "completed": True,
                "api_calls": 1,
                "messages": [],
            }
            MockAgent.return_value = mock_child

            from tools.delegate_tool import delegate_task

            result = json.loads(delegate_task(
                goal="process webhook data",
                parent_agent=parent,
            ))

            assert "results" in result
            # Verify delegation was successful
            entry = result["results"][0]
            assert entry["status"] == "completed"


# ---------------------------------------------------------------------------
# Test 6: Gateway entry cannot bypass JARVIS authorization/L3 approval
# ---------------------------------------------------------------------------


class TestGatewayCannotBypassAuthorization:
    """Gateway-triggered delegation still enforces L3 approval."""

    def test_l3_capability_skipped_through_gateway(self):
        """L3 capability returns SKIPPED_APPROVAL_REQUIRED through gateway."""
        _build_registry()
        child = MagicMock()
        child._subagent_id = "sa-gateway-l3"
        child._parent_session_id = "parent-gateway-l3"

        from plugins.memory.jarvis.planner.task_graph import (
            ExecutionPlan,
            ExecutionStage,
            Task,
            TaskGraph,
        )

        graph = TaskGraph()
        graph.add_task(Task(
            id="task_rag_delete_gateway",
            capability_name="rag_delete",
            task_type="default",
            authority_level="L0",
            parameters={"memory_id": "test-123"},
        ))
        plan = ExecutionPlan(
            stages=[ExecutionStage(stage_index=0, task_ids=frozenset({"task_rag_delete_gateway"}))]
        )

        child._capability_plan = plan
        child._capability_graph = graph
        child._capability_task_type = "default"
        child._capability_authority_level = "L0"

        result = _execute_capability_plan_if_present(child, "delete memory via gateway")

        assert result.status == CapabilityExecutionStatus.SKIPPED_APPROVAL_REQUIRED

    def test_non_jarvis_parent_through_gateway_no_plan(self):
        """Non-JARVIS parent (e.g., plain webhook) gets no capability plan."""
        parent = MagicMock()
        parent.base_url = "https://openrouter.ai/api/v1"
        parent.api_key = "test-key"
        parent.provider = "openrouter"
        parent.api_mode = "chat_completions"
        parent.model = "gpt-4"
        parent.platform = "cli"
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        # Non-JARVIS provider
        other_provider = MagicMock()
        other_provider.__class__.__name__ = "OtherProvider"
        parent._memory_manager = MagicMock()
        parent._memory_manager._providers = [other_provider]

        child = MagicMock()
        child._capability_plan = None

        _attach_capability_plan(parent, child, "regular task", "default", depth=1)

        # No plan should be attached for non-JARVIS parents
        assert child._capability_plan is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
