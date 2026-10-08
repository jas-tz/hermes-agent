"""D13 C: automatic session-end learning is skipped for sessions holding
untrusted/private JARVIS automation results (automation repo,
docs/ACTION-FRAMEWORK-DECISIONS.md D13). The classification comes from the
jarvis_automation plugin's registry-backed result_classification(); anything
undeterminable skips learning."""

import json
import os
import sys
import types
from unittest.mock import patch

import pytest

from agent.tool_dispatch_helpers import make_tool_result_message
from plugins.memory import jarvis as mem

INTERNAL = {"result_trust": "internal", "private_data": False}
DECLARATIONS = {
    "system.status": INTERNAL,
    "workspace.gmail.get": {"result_trust": "untrusted_external", "private_data": True},
}


@pytest.fixture
def automation_plugin(monkeypatch):
    module = types.ModuleType(mem._AUTOMATION_PLUGIN_MODULE)
    module.result_classification = lambda action, version=None: dict(
        DECLARATIONS.get(action, {"result_trust": "untrusted_external", "private_data": True}))
    monkeypatch.setitem(sys.modules, mem._AUTOMATION_PLUGIN_MODULE, module)
    return module


def session(action, framed=True):
    outcome = {"contract_version": "1.1", "action": action, "action_version": "1.0.0",
               "success": True, "result": {"text": "x" * 40}}
    content = json.dumps(outcome)
    msg = (make_tool_result_message("n8n_route_action", content, "c1") if framed
           else {"role": "tool", "name": "n8n_route_action", "tool_call_id": "c1", "content": content})
    return [{"role": "user", "content": "check it"}, msg, {"role": "assistant", "content": "done"}]


def run(messages):
    provider = mem.HermesRAGProvider()
    provider.initialize("s-d13", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary")
    with patch("rag_memory.extract_session_learnings", return_value=[]) as extract:
        provider.on_session_end(messages)
    return extract


@pytest.mark.parametrize("framed", [True, False])
def test_untrusted_private_result_skips_learning(automation_plugin, framed):
    assert mem._session_has_restricted_automation_result(session("workspace.gmail.get", framed))
    run(session("workspace.gmail.get", framed)).assert_not_called()


def test_internal_result_keeps_learning(automation_plugin):
    assert not mem._session_has_restricted_automation_result(session("system.status"))
    run(session("system.status")).assert_called_once()


def test_unknown_action_or_missing_plugin_fails_closed(automation_plugin, monkeypatch):
    run(session("odoo.product.lookup")).assert_not_called()
    monkeypatch.delitem(sys.modules, mem._AUTOMATION_PLUGIN_MODULE)
    run(session("system.status")).assert_not_called()


def test_sessions_without_automation_results_are_unaffected(monkeypatch):
    monkeypatch.delitem(sys.modules, mem._AUTOMATION_PLUGIN_MODULE, raising=False)
    run([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]).assert_called_once()


def test_subagent_session_end_path_also_skips(automation_plugin):
    with patch("rag_memory.extract_session_learnings") as extract:
        mem._on_session_end_learning(session("workspace.gmail.get"))
    extract.assert_not_called()
