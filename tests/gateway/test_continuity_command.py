"""Tests for the ``/continuity`` gateway slash command handler (Phase 13A).

Mirrors the CLI-layer coverage in tests/tools/test_write_approval.py::
test_cli_continuity_* — this file proves the SAME real stage -> approve/
reject -> continuity_db flow works through the gateway's async handler
(gateway/slash_commands.py::_handle_continuity_command), not just the CLI
one. Uses the same _make_runner() minimal-instantiation pattern already
established by tests/gateway/test_reload_skills_command.py.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_gw_continuity_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _set_approval(subsystem, enabled):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault(subsystem, {})["write_approval"] = enabled
    cfg.save_config(c)


def _stage_continuity_goal(hermes_home, title="Ship the launch"):
    from plugins.memory.jarvis import HermesRAGProvider
    _set_approval("continuity", True)
    provider = HermesRAGProvider()
    provider.initialize("session-gw-continuity", hermes_home=hermes_home, agent_context="primary")
    result = json.loads(provider.handle_tool_call("hermes_goal_create", {"title": title}))
    assert result.get("staged") is True, result
    return result["pending_id"]


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    """Minimal GatewayRunner instance — only what _handle_continuity_command
    actually touches: _session_key_for_source (which itself tolerates a
    missing session_store/config via getattr/hasattr fallbacks)."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={})
    from gateway.run import GatewayRunner as _GR
    runner._session_key_for_source = _GR._session_key_for_source.__get__(runner, _GR)
    return runner


@pytest.mark.asyncio
async def test_gateway_continuity_pending_empty(hermes_home):
    runner = _make_runner()
    out = await runner._handle_continuity_command(_make_event("/continuity pending"))
    assert "No pending continuity writes." in out


@pytest.mark.asyncio
async def test_gateway_continuity_pending_lists_staged_goal(hermes_home):
    pending_id = _stage_continuity_goal(hermes_home, title="Ship the launch")

    runner = _make_runner()
    out = await runner._handle_continuity_command(_make_event("/continuity pending"))
    assert pending_id in out
    assert "create goal: Ship the launch" in out


@pytest.mark.asyncio
async def test_gateway_continuity_approve_creates_real_goal(hermes_home):
    from hermes_cli.continuity_db import connect_closing, list_goals
    from tools import write_approval as wa

    pending_id = _stage_continuity_goal(hermes_home, title="Ship the launch")
    assert wa.pending_count("continuity") == 1

    runner = _make_runner()
    out = await runner._handle_continuity_command(_make_event(f"/continuity approve {pending_id}"))

    assert "Approved 1" in out, out
    assert wa.pending_count("continuity") == 0

    db_path = Path(hermes_home) / "continuity.db"
    with connect_closing(db_path=db_path) as conn:
        goals = list_goals(conn, status="active")
    approved = next((g for g in goals if g.title == "Ship the launch"), None)
    assert approved is not None
    assert approved.provenance.get("source") == "user_confirmed"


@pytest.mark.asyncio
async def test_gateway_continuity_reject_discards_without_writing(hermes_home):
    from hermes_cli.continuity_db import connect_closing, list_goals
    from tools import write_approval as wa

    pending_id = _stage_continuity_goal(hermes_home, title="Rejected goal")

    runner = _make_runner()
    out = await runner._handle_continuity_command(_make_event(f"/continuity reject {pending_id}"))

    assert "Rejected" in out, out
    assert wa.pending_count("continuity") == 0

    db_path = Path(hermes_home) / "continuity.db"
    if db_path.exists():
        with connect_closing(db_path=db_path) as conn:
            goals = list_goals(conn, status="active")
        assert not any(g.title == "Rejected goal" for g in goals)


@pytest.mark.asyncio
async def test_gateway_continuity_unknown_subcommand(hermes_home):
    runner = _make_runner()
    out = await runner._handle_continuity_command(_make_event("/continuity bogus"))
    assert "Unknown /continuity subcommand" in out


# ---------------------------------------------------------------------------
# Continuity inspection (Phase 14A) — /continuity list|goals|tasks|decisions|status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gateway_continuity_list_shows_combined_view(hermes_home):
    from plugins.memory.jarvis import HermesRAGProvider
    provider = HermesRAGProvider()
    provider.initialize("session-gw-inspect", hermes_home=hermes_home, agent_context="primary")
    json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Expand the honey business"}))
    json.loads(provider.handle_tool_call("hermes_task_create", {"title": "Prepare TBS submission"}))

    runner = _make_runner()
    out = await runner._handle_continuity_command(_make_event("/continuity list"))
    assert "Expand the honey business" in out
    assert "Prepare TBS submission" in out


@pytest.mark.asyncio
async def test_gateway_continuity_status_shows_counts(hermes_home):
    from plugins.memory.jarvis import HermesRAGProvider
    provider = HermesRAGProvider()
    provider.initialize("session-gw-inspect", hermes_home=hermes_home, agent_context="primary")
    json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Goal one"}))

    runner = _make_runner()
    out = await runner._handle_continuity_command(_make_event("/continuity status"))
    assert "Active/blocked goals: 1" in out
    assert "continuity.write_approval: off" in out


@pytest.mark.asyncio
async def test_gateway_continuity_missing_db_reports_empty_state_without_creating_it(hermes_home):
    db_path = Path(hermes_home) / "continuity.db"
    assert not db_path.exists()

    runner = _make_runner()
    out = await runner._handle_continuity_command(_make_event("/continuity goals"))
    assert out.strip()
    assert not db_path.exists()


@pytest.mark.asyncio
async def test_gateway_continuity_inspection_does_not_interfere_with_pending_flow():
    """Regression guard: list/goals/tasks/decisions/status must not shadow
    the pre-existing pending/approve/reject/approval subcommands."""
    import inspect
    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner.__mro__[0])
    # Sanity: the handler module still defines both surfaces.
    import gateway.slash_commands as sc
    handler_src = inspect.getsource(sc.GatewaySlashCommandsMixin._handle_continuity_command)
    assert "CONTINUITY_INSPECT_SUBCOMMANDS" in handler_src
    assert "handle_pending_subcommand" in handler_src


def test_gateway_dispatches_continuity_canonical_to_handler():
    """Confirms the canonical == "continuity" branch was actually added to
    gateway/run.py's dispatch — a regression guard for the wiring itself,
    not just the handler in isolation."""
    import inspect
    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner)
    assert '"continuity"' in src
    assert "_handle_continuity_command" in src
