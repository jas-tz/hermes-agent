"""Tests for the memory/skill write-approval gate (tools/write_approval.py)
and the shared slash-command handlers (hermes_cli/write_approval_commands.py).

Covers the boolean write_approval gate (off by default = write freely; on =
require approval) for both subsystems, the foreground-vs-background staging
split, pending store CRUD, and the list/approve/reject/diff/approval
subcommand dispatch.
"""

import json
import os
import tempfile
import shutil

import pytest


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_wa_test_")
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


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def test_default_gate_is_off(hermes_home):
    from tools import write_approval as wa
    # Default: gate off → writes flow freely.
    assert wa.write_approval_enabled("memory") is False
    assert wa.write_approval_enabled("skills") is False


def test_invalid_subsystem_is_off(hermes_home):
    from tools import write_approval as wa
    assert wa.write_approval_enabled("bogus") is False


def test_normalize_enabled_coerces_values():
    from tools import write_approval as wa
    # Real bools pass through.
    assert wa._normalize_enabled(True) is True
    assert wa._normalize_enabled(False) is False
    # Truthy strings → True (incl. legacy 'approve').
    assert wa._normalize_enabled("on") is True
    assert wa._normalize_enabled("approve") is True
    assert wa._normalize_enabled("true") is True
    # Everything else → False (gate off is the safe default).
    assert wa._normalize_enabled("off") is False
    assert wa._normalize_enabled("garbage") is False
    assert wa._normalize_enabled(None) is False


# ---------------------------------------------------------------------------
# Memory gate
# ---------------------------------------------------------------------------

def test_memory_gate_off_allows_write(hermes_home):
    # Default (gate off) → write straight through, no staging.
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "user", "save me", store=store))
    assert r["success"] is True
    assert r["entry_count"] == 1
    assert wa.pending_count("memory") == 0


def test_cli_memory_approve_without_live_agent_uses_fresh_store(hermes_home, capsys):
    """#46783: ``/memory approve`` from a context with no live agent (e.g. the
    Desktop GUI) passed ``memory_store=None`` into the shared handler, which
    returned "memory store unavailable" and applied nothing. The CLI handler must
    fall back to a freshly loaded on-disk store, like the gateway path does."""
    import json
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    _set_approval("memory", True)
    staging = MemoryStore(); staging.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "remember the launch date", store=staging))
    assert r.get("pending_id"), r
    assert wa.pending_count("memory") == 1

    # Bare CLI handler with no live agent → store resolves to None pre-fix.
    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler.agent = None
    handler._handle_memory_command("/memory approve all")

    out = capsys.readouterr().out
    assert "memory store unavailable" not in out, out
    assert "Approved 1" in out, out
    assert wa.pending_count("memory") == 0
    # The approved write landed in a freshly loaded on-disk store (MEMORY.md).
    reloaded = MemoryStore(); reloaded.load_from_disk()
    assert any("remember the launch date" in e for e in reloaded.memory_entries)


def test_load_on_disk_store_honors_configured_char_limits(hermes_home, monkeypatch):
    """load_on_disk_store() must read memory.memory_char_limit /
    user_char_limit from config so approvals applied without a live agent
    enforce the SAME caps as the live agent (agent_init.py). Falls back to
    defaults when config can't be loaded.
    """
    from tools.memory_tool import load_on_disk_store

    # Config override path: helper picks up the configured limits.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"memory_char_limit": 999, "user_char_limit": 444}},
    )
    store = load_on_disk_store()
    assert store.memory_char_limit == 999
    assert store.user_char_limit == 444

    # Failure path: config raises → defaults, never blows up.
    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    fallback = load_on_disk_store()
    assert fallback.memory_char_limit == 2200
    assert fallback.user_char_limit == 1375


# ---------------------------------------------------------------------------
# Continuity command (Phase 13A) — /continuity pending|approve|reject|approval
# ---------------------------------------------------------------------------

def _stage_continuity_goal(hermes_home, title="Ship the launch"):
    """Stage a real CONTINUITY proposal via the actual Jarvis provider gate
    path (not hand-built), so these tests exercise the real staging shape
    _evaluate_and_maybe_stage_continuity_write() produces."""
    from plugins.memory.jarvis import HermesRAGProvider
    _set_approval("continuity", True)
    provider = HermesRAGProvider()
    provider.initialize("session-cli-continuity", hermes_home=hermes_home, agent_context="primary")
    result = json.loads(provider.handle_tool_call("hermes_goal_create", {"title": title}))
    assert result.get("staged") is True, result
    return result["pending_id"]


def test_cli_continuity_pending_empty(hermes_home, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler.agent = None
    handler._handle_continuity_command("/continuity pending")
    out = capsys.readouterr().out
    assert "No pending continuity writes." in out


def test_cli_continuity_pending_lists_staged_goal(hermes_home, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    pending_id = _stage_continuity_goal(hermes_home, title="Ship the launch")

    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler.agent = None
    handler._handle_continuity_command("/continuity pending")
    out = capsys.readouterr().out
    assert pending_id in out
    assert "create goal: Ship the launch" in out


def test_cli_continuity_approve_creates_real_goal(hermes_home, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from hermes_cli.continuity_db import connect_closing, list_goals
    from tools import write_approval as wa
    from pathlib import Path

    pending_id = _stage_continuity_goal(hermes_home, title="Ship the launch")
    assert wa.pending_count("continuity") == 1

    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler.agent = None
    handler._handle_continuity_command(f"/continuity approve {pending_id}")

    out = capsys.readouterr().out
    assert "Approved 1" in out, out
    assert wa.pending_count("continuity") == 0

    db_path = Path(hermes_home) / "continuity.db"
    with connect_closing(db_path=db_path) as conn:
        goals = list_goals(conn, status="active")
    approved = next((g for g in goals if g.title == "Ship the launch"), None)
    assert approved is not None
    # apply_continuity_pending() overrides provenance to user_confirmed
    # regardless of what the staging call passed (Phase 10E requirement).
    assert approved.provenance.get("source") == "user_confirmed"


def test_cli_continuity_reject_discards_without_writing(hermes_home, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from hermes_cli.continuity_db import connect_closing, list_goals
    from tools import write_approval as wa
    from pathlib import Path

    pending_id = _stage_continuity_goal(hermes_home, title="Rejected goal")

    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler.agent = None
    handler._handle_continuity_command(f"/continuity reject {pending_id}")

    out = capsys.readouterr().out
    assert "Rejected" in out, out
    assert wa.pending_count("continuity") == 0

    db_path = Path(hermes_home) / "continuity.db"
    if db_path.exists():
        with connect_closing(db_path=db_path) as conn:
            goals = list_goals(conn, status="active")
        assert not any(g.title == "Rejected goal" for g in goals)


def test_cli_continuity_unknown_subcommand(hermes_home, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler.agent = None
    handler._handle_continuity_command("/continuity bogus")
    out = capsys.readouterr().out
    assert "Unknown /continuity subcommand" in out


def test_handle_continuity_approval_on(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    captured = {}
    out = handle_pending_subcommand(
        wa.CONTINUITY, ["approval", "on"],
        set_mode_fn=lambda enabled: captured.update(enabled=enabled),
    )
    assert captured["enabled"] is True
    assert "on" in out


# ---------------------------------------------------------------------------
# Continuity inspection (Phase 14A) — /continuity list|goals|tasks|decisions|status
# ---------------------------------------------------------------------------

def _make_continuity_provider(hermes_home):
    from plugins.memory.jarvis import HermesRAGProvider
    provider = HermesRAGProvider()
    provider.initialize("session-inspect", hermes_home=hermes_home, agent_context="primary")
    return provider


def _run_continuity(hermes_home, *args, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler.agent = None
    handler._handle_continuity_command("/continuity " + " ".join(args))
    return capsys.readouterr().out


def test_continuity_list_shows_combined_view(hermes_home, capsys):
    import json
    provider = _make_continuity_provider(hermes_home)
    json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Expand the honey business"}))
    json.loads(provider.handle_tool_call("hermes_task_create", {"title": "Prepare TBS submission"}))
    json.loads(provider.handle_tool_call("hermes_decision_record", {"decision": "Use generic company names"}))

    out = _run_continuity(hermes_home, "list", capsys=capsys)
    assert "Jarvis Continuity (stored state" in out
    assert "Expand the honey business" in out
    assert "Prepare TBS submission" in out
    assert "Use generic company names" in out


def test_continuity_goals_shows_only_goals(hermes_home, capsys):
    import json
    provider = _make_continuity_provider(hermes_home)
    json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Expand the honey business"}))
    json.loads(provider.handle_tool_call("hermes_task_create", {"title": "Prepare TBS submission"}))

    out = _run_continuity(hermes_home, "goals", capsys=capsys)
    assert "Expand the honey business" in out
    assert "Prepare TBS submission" not in out
    assert "## Open tasks" not in out


def test_continuity_tasks_shows_only_tasks(hermes_home, capsys):
    import json
    provider = _make_continuity_provider(hermes_home)
    json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Expand the honey business"}))
    json.loads(provider.handle_tool_call("hermes_task_create", {"title": "Prepare TBS submission"}))

    out = _run_continuity(hermes_home, "tasks", capsys=capsys)
    assert "Prepare TBS submission" in out
    assert "Expand the honey business" not in out
    assert "## Active goals" not in out


def test_continuity_decisions_shows_only_decisions(hermes_home, capsys):
    import json
    provider = _make_continuity_provider(hermes_home)
    json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Expand the honey business"}))
    json.loads(provider.handle_tool_call("hermes_decision_record", {"decision": "Use generic company names"}))

    out = _run_continuity(hermes_home, "decisions", capsys=capsys)
    assert "Use generic company names" in out
    assert "Expand the honey business" not in out
    assert "## Active goals" not in out


def test_continuity_status_shows_counts_and_approval_mode(hermes_home, capsys):
    import json
    provider = _make_continuity_provider(hermes_home)
    json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Goal one"}))
    json.loads(provider.handle_tool_call("hermes_task_create", {"title": "Task one"}))
    json.loads(provider.handle_tool_call("hermes_task_create", {"title": "Task two"}))

    out = _run_continuity(hermes_home, "status", capsys=capsys)
    assert "Active/blocked goals: 1" in out
    assert "Open/blocked tasks: 2" in out
    assert "Current decisions: 0" in out
    assert "continuity.write_approval: off" in out
    assert "Pending proposals: 0" in out


def test_continuity_status_reflects_approval_mode_on(hermes_home, capsys):
    _set_approval("continuity", True)
    out = _run_continuity(hermes_home, "status", capsys=capsys)
    assert "continuity.write_approval: on" in out


def test_continuity_missing_db_reports_empty_state_without_creating_it(hermes_home, capsys):
    from pathlib import Path
    db_path = Path(hermes_home) / "continuity.db"
    assert not db_path.exists()

    for sub in ("list", "goals", "tasks", "decisions"):
        out = _run_continuity(hermes_home, sub, capsys=capsys)
        assert out.strip()  # some deterministic "no records" message, not a crash
        assert not db_path.exists()

    out = _run_continuity(hermes_home, "status", capsys=capsys)
    assert "Active/blocked goals: 0" in out
    assert "Open/blocked tasks: 0" in out
    assert "Current decisions: 0" in out
    assert not db_path.exists()


def test_continuity_blocked_goal_and_task_appear_in_inspection(hermes_home, capsys):
    import json
    provider = _make_continuity_provider(hermes_home)
    gid = json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Blocked goal here"}))["goal"]["id"]
    json.loads(provider.handle_tool_call("hermes_goal_update", {"goal_id": gid, "status": "blocked"}))
    tid = json.loads(provider.handle_tool_call("hermes_task_create", {"title": "Blocked task here"}))["task"]["id"]
    json.loads(provider.handle_tool_call("hermes_task_update", {"task_id": tid, "status": "blocked"}))

    goals_out = _run_continuity(hermes_home, "goals", capsys=capsys)
    assert "Blocked goal here" in goals_out

    tasks_out = _run_continuity(hermes_home, "tasks", capsys=capsys)
    assert "Blocked task here" in tasks_out
    line = next(l for l in tasks_out.splitlines() if "Blocked task here" in l)
    assert line.startswith("- [blocked]")

    status_out = _run_continuity(hermes_home, "status", capsys=capsys)
    assert "Active/blocked goals: 1" in status_out
    assert "Open/blocked tasks: 1" in status_out


def test_continuity_superseded_decision_excluded_from_inspection(hermes_home, capsys):
    import json
    provider = _make_continuity_provider(hermes_home)
    old_id = json.loads(provider.handle_tool_call(
        "hermes_decision_record", {"decision": "Old decision text"}
    ))["decision"]["id"]
    json.loads(provider.handle_tool_call(
        "hermes_decision_supersede",
        {"decision_id": old_id, "decision": "New decision text", "rationale": "superseding the old one"},
    ))

    out = _run_continuity(hermes_home, "decisions", capsys=capsys)
    assert "New decision text" in out
    assert "Old decision text" not in out


def test_continuity_inspection_does_not_interfere_with_approve_reject(hermes_home, capsys):
    """Regression: the new list/goals/tasks/decisions/status subcommands
    must not shadow or break the existing pending/approve/reject/approval
    write-approval subcommands they sit alongside."""
    import json
    from tools import write_approval as wa

    _set_approval("continuity", True)
    provider = _make_continuity_provider(hermes_home)
    result = json.loads(provider.handle_tool_call("hermes_goal_create", {"title": "Staged goal"}))
    assert result.get("staged") is True
    pending_id = result["pending_id"]

    pending_out = _run_continuity(hermes_home, "pending", capsys=capsys)
    assert pending_id in pending_out

    approve_out = _run_continuity(hermes_home, "approve", pending_id, capsys=capsys)
    assert "Approved 1" in approve_out
    assert wa.pending_count("continuity") == 0


# ---------------------------------------------------------------------------
# Skill gate
# ---------------------------------------------------------------------------

_SKILL = (
    "---\nname: test-skill\ndescription: A test skill\nversion: 1.0.0\n---\n"
    "# Test\nbody\n"
)


# ---------------------------------------------------------------------------
# Pending store CRUD
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared command handler
# ---------------------------------------------------------------------------


def test_handle_approve_all(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_tool import MemoryStore
    from tools import write_approval as wa
    store = MemoryStore(); store.load_from_disk()
    wa.stage_write("memory", {"action": "add", "target": "user", "content": "a"},
                   summary="a", origin="foreground")
    wa.stage_write("memory", {"action": "add", "target": "user", "content": "b"},
                   summary="b", origin="foreground")
    out = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store)
    assert "Approved 2" in out
    assert wa.pending_count("memory") == 0
    assert len(store.user_entries) == 2


def test_handle_approval_on(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    captured = {}
    out = handle_pending_subcommand(
        wa.MEMORY, ["approval", "on"],
        set_mode_fn=lambda enabled: captured.update(enabled=enabled),
    )
    assert captured["enabled"] is True
    assert "on" in out


def test_handle_approval_off(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    captured = {}
    out = handle_pending_subcommand(
        wa.SKILLS, ["approval", "off"],
        set_mode_fn=lambda enabled: captured.update(enabled=enabled),
    )
    assert captured["enabled"] is False
    assert "off" in out


# ---------------------------------------------------------------------------
# Inline (interactive CLI) approval path — regression for the bug where the
# per-thread approval callback was never passed to prompt_dangerous_approval,
# so every gated foreground memory write was silently denied.
# ---------------------------------------------------------------------------

@pytest.fixture
def approval_callback_cleanup():
    yield
    from tools.terminal_tool import set_approval_callback
    set_approval_callback(None)


def test_memory_inline_approve_writes(hermes_home, approval_callback_cleanup):
    from tools.memory_tool import memory_tool, MemoryStore
    from tools.terminal_tool import set_approval_callback
    from tools import write_approval as wa
    _set_approval("memory", True)

    calls = []
    def approve_cb(command, description, **kw):
        calls.append((command, description))
        return "once"
    set_approval_callback(approve_cb)

    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "approved fact", store=store))
    assert r["success"] is True
    assert r.get("staged") is None  # real write, not staged
    assert store.memory_entries == ["approved fact"]
    assert wa.pending_count("memory") == 0
    # The registered callback must actually be invoked (not the input() path).
    assert len(calls) == 1
    assert "approved fact" in calls[0][0]


def test_memory_inline_deny_blocks(hermes_home, approval_callback_cleanup):
    from tools.memory_tool import memory_tool, MemoryStore
    from tools.terminal_tool import set_approval_callback
    from tools import write_approval as wa
    _set_approval("memory", True)
    set_approval_callback(lambda command, description, **kw: "deny")

    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "denied fact", store=store))
    assert r["success"] is False
    assert "denied" in r["error"].lower()
    assert store.memory_entries == []
    assert wa.pending_count("memory") == 0  # denied, not staged


def test_memory_invalid_params_rejected_before_staging(hermes_home):
    # Param validation must run BEFORE the gate so a broken write is rejected
    # immediately instead of staged and failing at approve time.
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    _set_approval("memory", True)
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", None, store=store))
    assert r["success"] is False
    assert wa.pending_count("memory") == 0


class TestSkillGist:
    """skill_gist builds a heuristic one-line summary for a pending skill write.

    Pure, no model call — every branch is verifiable from the function source.
    """

    def test_create_with_frontmatter_description(self):
        from tools import write_approval as wa
        content = "---\ndescription: My cool skill\n---\nprint('hi')\n"
        assert (
            wa.skill_gist("create", "demo", content=content)
            == f"create 'demo' — My cool skill ({len(content)} chars)"
        )

    def test_edit_without_description_uses_size_only(self):
        from tools import write_approval as wa
        content = "no frontmatter here"
        assert (
            wa.skill_gist("edit", "demo", content=content)
            == f"rewrite 'demo' ({len(content)} chars)"
        )


    def test_file_actions_and_unknown_fallback(self):
        from tools import write_approval as wa
        assert wa.skill_gist("write_file", "demo", file_path="a.py") == "write a.py in 'demo'"
        assert wa.skill_gist("remove_file", "demo", file_path="a.py") == "remove a.py from 'demo'"
        assert wa.skill_gist("delete", "demo") == "delete skill 'demo'"
        assert wa.skill_gist("unknown", "demo") == "unknown 'demo'"
