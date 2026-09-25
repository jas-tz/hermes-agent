"""Integration tests: agent file tools honour write_denied_paths.

``security.filesystem_scope.write_denied_paths`` must stop ``write_file`` and
every ``patch`` form from writing to a denied location (the JARVIS vault) or
anything under it, before approval staging or any filesystem mutation, while
leaving reads and writes elsewhere untouched. The "vault" here is always a
pytest ``tmp_path`` directory; no real vault path is configured or accessed.
"""
import contextvars
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

import tools.file_tools as ft
import tools.filesystem_scope as fs
import tools.terminal_tool as terminal_tool


_FORBIDDEN_FRAGMENTS = ("jarvis-knowledge", "jarvis~1")
_real_vault_events: list = []
_recording = {"on": False}


def _audit_hook(event, args):
    # Filesystem events only ("compile" etc. carry source text, not paths).
    if not _recording["on"] or not (event == "open" or event.startswith(("os.", "shutil."))):
        return
    for arg in args:
        if isinstance(arg, (str, bytes, os.PathLike)):
            text = os.fsdecode(arg).casefold()
            if any(fragment in text for fragment in _FORBIDDEN_FRAGMENTS):
                _real_vault_events.append((event, text))


sys.addaudithook(_audit_hook)  # audit hooks cannot be removed; gated by _recording


@pytest.fixture(autouse=True)
def _no_real_vault_access():
    """Fail any test that opens, lists or mutates a real-vault path."""
    _real_vault_events.clear()
    _recording["on"] = True
    try:
        yield
    finally:
        _recording["on"] = False
    assert not _real_vault_events, f"real-vault path touched: {_real_vault_events}"


NOTE = "line one\nline two\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A stand-in vault, a sibling sharing its prefix, and an outside dir.

    The boundary config is injected through the loader so no config.yaml is
    involved; the task's working directory is ``tmp_path``.
    """
    vault = tmp_path / "vault"
    (vault / "notes").mkdir(parents=True)
    (vault / "notes" / "a.md").write_text(NOTE)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "o.md").write_text(NOTE)
    (tmp_path / "vault2").mkdir()
    config = {"security": {"filesystem_scope": {"write_denied_paths": [str(vault)]}}}
    monkeypatch.setattr(fs, "_load_write_boundary_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.chdir(outside)  # process cwd deliberately NOT the task cwd
    task_id = f"boundary-{tmp_path.name}"
    terminal_tool.record_session_cwd(task_id, str(tmp_path))
    return tmp_path, vault, outside, task_id


def _snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def _refused(result: str) -> bool:
    data = json.loads(result)
    return "error" in data and "write-denied" in data["error"] and "Nothing was written" in data["error"]


def _ok(result: str) -> bool:
    data = json.loads(result)
    return not data.get("error")


def _v4a(*body):
    return "*** Begin Patch\n" + "\n".join(body) + "\n*** End Patch\n"


# ── write_file ───────────────────────────────────────────────────────────


class TestWriteFile:
    def test_allowed_sibling_write_succeeds(self, env):
        root, vault, outside, task = env
        target = root / "vault2" / "x.md"
        assert _ok(ft.write_file_tool(str(target), "hello\n", task_id=task))
        assert target.read_text() == "hello\n"

    @pytest.mark.parametrize("rel", ["notes/a.md", "new.md", "deep/new.md"])
    def test_vault_write_refused_and_nothing_written(self, env, rel):
        root, vault, outside, task = env
        before = _snapshot(root)
        assert _refused(ft.write_file_tool(str(vault / rel), "X\n", task_id=task))
        assert _snapshot(root) == before

    def test_exact_vault_path_refused(self, env):
        root, vault, outside, task = env
        assert _refused(ft.write_file_tool(str(vault), "X\n", task_id=task))
        assert vault.is_dir()

    def test_case_variant_refused(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        assert _refused(ft.write_file_tool(str(root / "VAULT" / "notes" / "a.md"), "X\n", task_id=task))
        assert _snapshot(root) == before

    def test_dotdot_traversal_refused(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        assert _refused(ft.write_file_tool(str(outside / ".." / "vault" / "new.md"), "X\n", task_id=task))
        assert _snapshot(root) == before

    def test_symlink_into_vault_refused(self, env):
        root, vault, outside, task = env
        (outside / "link").symlink_to(vault, target_is_directory=True)
        before = _snapshot(vault)
        assert _refused(ft.write_file_tool(str(outside / "link" / "notes" / "a.md"), "X\n", task_id=task))
        assert _refused(ft.write_file_tool(str(outside / "link" / "new.md"), "X\n", task_id=task))
        assert _snapshot(vault) == before

    def test_relative_path_resolves_against_task_cwd_not_process_cwd(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        # Task cwd is tmp_path; process cwd is outside/. "vault/new.md" must
        # be judged as tmp_path/vault/new.md.
        assert _refused(ft.write_file_tool("vault/new.md", "X\n", task_id=task))
        assert _refused(ft.write_file_tool("./vault/notes/a.md", "X\n", task_id=task))
        assert _snapshot(root) == before
        assert _ok(ft.write_file_tool("vault2/rel.md", "ok\n", task_id=task))
        assert (root / "vault2" / "rel.md").read_text() == "ok\n"

    def test_relative_path_when_task_cwd_is_inside_vault(self, env):
        root, vault, outside, task = env
        terminal_tool.record_session_cwd(task, str(vault / "notes"))
        before = _snapshot(vault)
        assert _refused(ft.write_file_tool("a.md", "X\n", task_id=task))
        assert _refused(ft.write_file_tool("new.md", "X\n", task_id=task))
        assert _snapshot(vault) == before

    def test_refusal_happens_before_staging_and_file_ops(self, env, monkeypatch):
        root, vault, outside, task = env
        from tools import write_approval as wa

        def must_not_run(*a, **k):
            raise AssertionError("reached a later stage")

        monkeypatch.setattr(wa, "evaluate_gate", must_not_run)
        monkeypatch.setattr(wa, "stage_write", must_not_run)
        monkeypatch.setattr(ft, "_get_file_ops", must_not_run)
        assert _refused(ft.write_file_tool(str(vault / "new.md"), "X\n", task_id=task))
        assert not (vault / "new.md").exists()

    def test_approved_pending_replay_is_still_refused(self, env):
        root, vault, outside, task = env
        result = ft.apply_file_pending({"path": str(vault / "notes" / "a.md"), "content": "X\n"})
        assert _refused(result["result"])
        assert (vault / "notes" / "a.md").read_text() == NOTE


class TestFailClosed:
    def test_config_load_error_refuses_every_write(self, env, monkeypatch):
        root, vault, outside, task = env

        def boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(fs, "_load_write_boundary_config", boom)
        target = root / "vault2" / "x.md"
        result = json.loads(ft.write_file_tool(str(target), "X\n", task_id=task))
        assert "boundary check failed" in result["error"]
        assert not target.exists()
        result = json.loads(ft.patch_tool(mode="replace", path=str(outside / "o.md"),
                                          old_string="line one", new_string="CHANGED",
                                          task_id=task))
        assert "boundary check failed" in result["error"]
        assert (outside / "o.md").read_text() == NOTE

    def test_checker_exception_refuses(self, env, monkeypatch):
        root, vault, outside, task = env

        def boom(*a, **k):
            raise RuntimeError("checker crashed")

        monkeypatch.setattr(fs, "is_write_allowed", boom)
        target = root / "vault2" / "x.md"
        result = json.loads(ft.write_file_tool(str(target), "X\n", task_id=task))
        assert "boundary check failed" in result["error"]
        assert not target.exists()

    def test_checker_internal_error_refuses(self, env, monkeypatch):
        root, vault, outside, task = env

        def boom(path):
            raise OSError("resolution exploded")

        monkeypatch.setattr(fs, "_path_forms", boom)
        target = root / "vault2" / "x.md"
        assert "Write boundary check failed" in json.loads(
            ft.write_file_tool(str(target), "X\n", task_id=task))["error"]
        assert not target.exists()

    def test_malformed_config_refuses(self, env, monkeypatch):
        root, vault, outside, task = env
        bad = {"security": {"filesystem_scope": {"write_denied_paths": ["relative/vault"]}}}
        monkeypatch.setattr(fs, "_load_write_boundary_config", lambda: bad)
        target = root / "vault2" / "x.md"
        assert "error" in json.loads(ft.write_file_tool(str(target), "X\n", task_id=task))
        assert not target.exists()

    def test_unconfigured_boundary_changes_nothing(self, env, monkeypatch):
        root, vault, outside, task = env
        monkeypatch.setattr(fs, "_load_write_boundary_config", lambda: {})
        assert _ok(ft.write_file_tool(str(vault / "notes" / "a.md"), "free\n", task_id=task))
        assert (vault / "notes" / "a.md").read_text() == "free\n"


# ── patch ────────────────────────────────────────────────────────────────


class TestPatch:
    def test_replace_outside_vault_still_works(self, env):
        root, vault, outside, task = env
        assert _ok(ft.patch_tool(mode="replace", path=str(outside / "o.md"),
                                 old_string="line one", new_string="LINE ONE", task_id=task))
        assert (outside / "o.md").read_text() == "LINE ONE\nline two\n"

    def test_v4a_outside_vault_still_works(self, env):
        root, vault, outside, task = env
        patch = _v4a(f"*** Update File: {outside / 'o.md'}", "@@", "-line one", "+LINE ONE", " line two")
        assert _ok(ft.patch_tool(mode="patch", patch=patch, task_id=task))
        assert (outside / "o.md").read_text() == "LINE ONE\nline two\n"

    def test_replace_mode_refused(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        assert _refused(ft.patch_tool(mode="replace", path=str(vault / "notes" / "a.md"),
                                      old_string="line one", new_string="X", task_id=task))
        assert _snapshot(root) == before

    def test_replace_mode_relative_path_refused(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        assert _refused(ft.patch_tool(mode="replace", path="vault/notes/a.md",
                                      old_string="line one", new_string="X", task_id=task))
        assert _snapshot(root) == before

    @pytest.mark.parametrize("header", [
        "*** Update File: {vault}/notes/a.md",
        "***Update File: {vault}/notes/a.md",       # no space, accepted by the parser
        "*** Update File: {root}/VAULT/notes/a.md",  # case variant
        "*** Update File: vault/notes/a.md",         # relative to the task cwd
    ])
    def test_v4a_update_refused(self, env, header):
        root, vault, outside, task = env
        before = _snapshot(root)
        patch = _v4a(header.format(vault=vault, root=root), "@@", "-line one", "+X", " line two")
        assert _refused(ft.patch_tool(mode="patch", patch=patch, task_id=task))
        assert _snapshot(root) == before

    def test_v4a_add_refused(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        patch = _v4a(f"*** Add File: {vault / 'new.md'}", "+created")
        assert _refused(ft.patch_tool(mode="patch", patch=patch, task_id=task))
        assert _snapshot(root) == before

    def test_v4a_delete_refused(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        patch = _v4a(f"*** Delete File: {vault / 'notes' / 'a.md'}")
        assert _refused(ft.patch_tool(mode="patch", patch=patch, task_id=task))
        assert _snapshot(root) == before

    def test_v4a_move_source_in_vault_refused(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        patch = _v4a(f"*** Move File: {vault / 'notes' / 'a.md'} -> {outside / 'moved.md'}")
        assert _refused(ft.patch_tool(mode="patch", patch=patch, task_id=task))
        assert _snapshot(root) == before

    def test_v4a_move_destination_in_vault_refused(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        patch = _v4a(f"*** Move File: {outside / 'o.md'} -> {vault / 'moved.md'}")
        assert _refused(ft.patch_tool(mode="patch", patch=patch, task_id=task))
        assert _snapshot(root) == before

    def test_crlf_patch_refused(self, env):
        root, vault, outside, task = env
        before = _snapshot(root)
        patch = _v4a(f"*** Add File: {vault / 'crlf.md'}", "+x").replace("\n", "\r\n")
        assert _refused(ft.patch_tool(mode="patch", patch=patch, task_id=task))
        assert _snapshot(root) == before

    @pytest.mark.parametrize("vault_first", [True, False])
    def test_multi_file_patch_is_all_or_nothing(self, env, vault_first):
        root, vault, outside, task = env
        before = _snapshot(root)
        allowed_ops = [
            f"*** Add File: {outside / 'new-outside.md'}", "+should not exist",
            f"*** Update File: {outside / 'o.md'}", "@@", "-line one", "+CHANGED", " line two",
        ]
        denied_ops = [f"*** Update File: {vault / 'notes' / 'a.md'}", "@@", "-line one", "+X", " line two"]
        body = denied_ops + allowed_ops if vault_first else allowed_ops + denied_ops
        assert _refused(ft.patch_tool(mode="patch", patch=_v4a(*body), task_id=task))
        assert _snapshot(root) == before

    def test_refusal_happens_before_file_ops(self, env, monkeypatch):
        root, vault, outside, task = env

        def must_not_run(*a, **k):
            raise AssertionError("reached file operations")

        monkeypatch.setattr(ft, "_get_file_ops", must_not_run)
        patch = _v4a(f"*** Add File: {vault / 'new.md'}", "+x")
        assert _refused(ft.patch_tool(mode="patch", patch=patch, task_id=task))

    def test_parser_paths_are_checked(self, env, monkeypatch):
        """Paths the parser will apply are checked even if the header
        extraction in patch_tool missed them."""
        root, vault, outside, task = env
        from tools.patch_parser import OperationType, PatchOperation

        def fake_parse(text):
            return [PatchOperation(operation=OperationType.ADD, file_path=str(vault / "p.md"))], None

        monkeypatch.setattr("tools.patch_parser.parse_v4a_patch", fake_parse)
        assert _refused(ft.patch_tool(mode="patch", patch="*** Begin Patch\n*** End Patch\n", task_id=task))
        assert not (vault / "p.md").exists()


# ── reads and delegation ─────────────────────────────────────────────────


class TestReadsAndDelegation:
    def test_reads_in_vault_remain_allowed(self, env):
        root, vault, outside, task = env
        result = json.loads(ft.read_file_tool(str(vault / "notes" / "a.md"), task_id=task))
        assert "error" not in result
        assert "line one" in result["content"]

    def test_search_in_vault_remains_allowed(self, env):
        root, vault, outside, task = env
        result = json.loads(ft.search_tool("line two", path=str(vault), task_id=task))
        assert "error" not in result

    def test_delegated_child_inherits_boundary(self, env):
        """A delegate_task child runs tools in a copied context with the
        delegated-child flag set and its own task id; the boundary still
        applies because it lives in the tool handlers themselves."""
        root, vault, outside, task = env
        from agent.delegation_context import delegated_child_context, is_delegated_child_context
        from tools.registry import registry

        child_task = f"{task}-child"
        terminal_tool.record_session_cwd(child_task, str(root))

        def run_child():
            with delegated_child_context("child-session"):
                assert is_delegated_child_context()
                write = registry.dispatch(
                    "write_file", {"path": str(vault / "child.md"), "content": "X\n"}, task_id=child_task)
                patch = registry.dispatch(
                    "patch", {"mode": "patch", "patch": _v4a(f"*** Add File: {vault / 'child2.md'}", "+x")},
                    task_id=child_task)
                allowed = registry.dispatch(
                    "write_file", {"path": str(root / "vault2" / "child.md"), "content": "ok\n"},
                    task_id=child_task)
                return write, patch, allowed

        before = _snapshot(vault)
        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            write, patch, allowed = pool.submit(ctx.run, run_child).result()
        assert _refused(write)
        assert _refused(patch)
        assert _ok(allowed)
        assert _snapshot(vault) == before
        assert (root / "vault2" / "child.md").read_text() == "ok\n"
