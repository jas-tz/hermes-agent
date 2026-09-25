"""Unit tests for the write-only denied-path boundary in tools.filesystem_scope.

``security.filesystem_scope.write_denied_paths`` blocks agent writes to a
location and all of its descendants (used to keep agents out of the JARVIS
vault). Every test here uses a pytest ``tmp_path`` stand-in for the vault and
passes the config explicitly; no real vault path is ever configured or read.
"""
import os
import sys

import pytest

import tools.filesystem_scope as fs


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


def _cfg(*denied, **scope):
    return {"security": {"filesystem_scope": {"write_denied_paths": list(denied), **scope}}}


@pytest.fixture
def layout(tmp_path):
    vault = tmp_path / "vault"
    (vault / "notes").mkdir(parents=True)
    (vault / "notes" / "a.md").write_text("A\n")
    (tmp_path / "vault2").mkdir()
    (tmp_path / "outside").mkdir()
    return tmp_path, vault


def test_unconfigured_allows_everything(layout):
    root, vault = layout
    assert fs.is_write_allowed(str(vault / "notes" / "a.md"), config={}) == (True, "")
    assert fs.is_write_allowed(str(vault), config=_cfg()) == (True, "")


def test_sibling_sharing_a_prefix_is_allowed(layout):
    root, vault = layout
    cfg = _cfg(str(vault))
    assert fs.is_write_allowed(str(root / "vault2" / "x.md"), config=cfg)[0] is True
    assert fs.is_write_allowed(str(root / "vault-archive.md"), config=cfg)[0] is True
    assert fs.is_write_allowed(str(root / "outside" / "x.md"), config=cfg)[0] is True


def test_exact_denied_path(layout):
    root, vault = layout
    allowed, reason = fs.is_write_allowed(str(vault), config=_cfg(str(vault)))
    assert allowed is False
    assert "write-denied" in reason


def test_trailing_slash_and_dot_forms_denied(layout):
    root, vault = layout
    cfg = _cfg(str(vault) + "/")
    assert fs.is_write_allowed(str(vault) + "/", config=cfg)[0] is False
    assert fs.is_write_allowed(str(vault) + "/./notes/a.md", config=cfg)[0] is False


@pytest.mark.parametrize("rel", ["notes/a.md", "new.md", "deep/new/dir/file.md"])
def test_descendants_denied_existing_or_not(layout, rel):
    root, vault = layout
    assert fs.is_write_allowed(str(vault / rel), config=_cfg(str(vault)))[0] is False


@pytest.mark.parametrize("variant", ["VAULT", "Vault", "vAuLt"])
def test_case_variant_denied(layout, variant):
    root, vault = layout
    target = str(root / variant / "notes" / "a.md")
    assert fs.is_write_allowed(target, config=_cfg(str(vault)))[0] is False


def test_case_variant_in_config_entry_denied(layout):
    root, vault = layout
    assert fs.is_write_allowed(str(vault / "x.md"), config=_cfg(str(root / "VAULT")))[0] is False


def test_dotdot_traversal_into_vault_denied(layout):
    root, vault = layout
    target = str(root / "outside" / ".." / "vault" / "notes" / "a.md")
    assert fs.is_write_allowed(target, config=_cfg(str(vault)))[0] is False


def test_dotdot_traversal_out_of_vault_allowed(layout):
    root, vault = layout
    target = str(vault / ".." / "outside" / "x.md")
    assert fs.is_write_allowed(target, config=_cfg(str(vault)))[0] is True


def test_symlink_into_vault_denied(layout):
    root, vault = layout
    link = root / "outside" / "link"
    link.symlink_to(vault, target_is_directory=True)
    cfg = _cfg(str(vault))
    assert fs.is_write_allowed(str(link / "notes" / "a.md"), config=cfg)[0] is False
    assert fs.is_write_allowed(str(link / "new.md"), config=cfg)[0] is False


def test_file_symlink_into_vault_denied(layout):
    root, vault = layout
    link = root / "outside" / "a-link.md"
    link.symlink_to(vault / "notes" / "a.md")
    assert fs.is_write_allowed(str(link), config=_cfg(str(vault)))[0] is False


def test_dangling_symlink_into_vault_denied(layout):
    root, vault = layout
    link = root / "outside" / "future.md"
    link.symlink_to(vault / "not-yet.md")
    assert fs.is_write_allowed(str(link), config=_cfg(str(vault)))[0] is False


def test_denied_entry_that_is_itself_a_symlink(layout):
    root, vault = layout
    alias = root / "vault-alias"
    alias.symlink_to(vault, target_is_directory=True)
    assert fs.is_write_allowed(str(vault / "x.md"), config=_cfg(str(alias)))[0] is False


def test_symlinks_denied_by_path_resolution_alone(layout, monkeypatch):
    """The realpath comparison covers symlinks without the identity check."""
    root, vault = layout
    monkeypatch.setattr(fs, "_existing_identities", lambda path: set())
    (root / "outside" / "link").symlink_to(vault, target_is_directory=True)
    (root / "outside" / "future.md").symlink_to(vault / "not-yet.md")
    cfg = _cfg(str(vault))
    assert fs.is_write_allowed(str(root / "outside" / "link" / "new.md"), config=cfg)[0] is False
    assert fs.is_write_allowed(str(root / "outside" / "future.md"), config=cfg)[0] is False


def test_identity_check_catches_alias_string_forms_miss(layout, monkeypatch):
    """Simulates a DrvFs/8.3 alias: string forms never match, but the target's
    existing ancestor is the same directory (st_dev, st_ino) as the entry."""
    root, vault = layout
    monkeypatch.setattr(fs, "_path_forms", lambda path: {("never", path)})
    target = str(vault / "notes" / "new.md")
    assert fs.is_write_allowed(target, config=_cfg(str(vault)))[0] is False
    assert fs.is_write_allowed(str(root / "vault2" / "x.md"), config=_cfg(str(vault)))[0] is True


def test_relative_path_resolved_against_base_dir(layout):
    root, vault = layout
    cfg = _cfg(str(vault))
    assert fs.is_write_allowed("vault/notes/a.md", base_dir=str(root), config=cfg)[0] is False
    assert fs.is_write_allowed("notes/a.md", base_dir=str(vault), config=cfg)[0] is False
    assert fs.is_write_allowed("vault2/x.md", base_dir=str(root), config=cfg)[0] is True


def test_relative_path_defaults_to_process_cwd(layout, monkeypatch):
    root, vault = layout
    monkeypatch.chdir(vault)
    assert fs.is_write_allowed("new.md", config=_cfg(str(vault)))[0] is False


def test_relative_base_dir_fails_closed(layout):
    root, vault = layout
    allowed, reason = fs.is_write_allowed("x.md", base_dir="relative/base", config=_cfg(str(vault)))
    assert allowed is False
    assert "Write boundary check failed" in reason


def test_tilde_entry_is_expanded(layout, monkeypatch):
    root, vault = layout
    monkeypatch.setenv("HOME", str(root))
    assert fs.is_write_allowed(str(vault / "x.md"), config=_cfg("~/vault"))[0] is False


@pytest.mark.parametrize("bad", [
    {"security": {"filesystem_scope": {"write_denied_paths": ["relative/vault"]}}},
    {"security": {"filesystem_scope": {"write_denied_paths": [42]}}},
    {"security": {"filesystem_scope": {"write_denied_paths": {"a": 1}}}},
    {"security": {"filesystem_scope": "not-a-mapping"}},
    {"security": "not-a-mapping"},
])
def test_malformed_config_fails_closed(layout, bad):
    root, vault = layout
    allowed, reason = fs.is_write_allowed(str(root / "outside" / "x.md"), config=bad)
    assert allowed is False
    assert "Write boundary check failed" in reason


def test_string_entry_and_blank_entries(layout):
    root, vault = layout
    assert fs.get_write_denied_paths(_cfg("", "  ")) == []
    cfg = {"security": {"filesystem_scope": {"write_denied_paths": str(vault)}}}
    assert fs.is_write_allowed(str(vault / "x.md"), config=cfg)[0] is False


def test_config_load_failure_fails_closed(layout, monkeypatch):
    root, vault = layout

    def boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(fs, "_load_write_boundary_config", boom)
    allowed, reason = fs.is_write_allowed(str(root / "outside" / "x.md"))
    assert allowed is False
    assert "RuntimeError" in reason


def test_internal_error_fails_closed(layout, monkeypatch):
    root, vault = layout

    def boom(path):
        raise OSError("resolution exploded")

    monkeypatch.setattr(fs, "_path_forms", boom)
    allowed, reason = fs.is_write_allowed(str(root / "outside" / "x.md"), config=_cfg(str(vault)))
    assert allowed is False
    assert "Write boundary check failed" in reason


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_empty_target_fails_closed(layout, empty):
    root, vault = layout
    assert fs.is_write_allowed(empty, config=_cfg(str(vault)))[0] is False


def test_nonexistent_denied_entry_still_denies_by_path(tmp_path):
    ghost = tmp_path / "ghost-vault"
    assert fs.is_write_allowed(str(ghost / "x.md"), config=_cfg(str(ghost)))[0] is False
    assert fs.is_write_allowed(str(tmp_path / "other.md"), config=_cfg(str(ghost)))[0] is True


def test_real_config_loading(layout, monkeypatch):
    """The key is read from $HERMES_HOME/config.yaml (the test sandbox)."""
    root, vault = layout
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "security:\n  filesystem_scope:\n    write_denied_paths:\n      - " + str(vault) + "\n"
    )
    assert fs.get_write_denied_paths() == [str(vault)]
    assert fs.is_write_allowed(str(vault / "x.md"))[0] is False
    assert fs.is_write_allowed(str(root / "outside" / "x.md"))[0] is True


class TestReadScopeUnchanged:
    """write_denied_paths must not alter is_path_allowed (used for reads)."""

    def test_write_denied_path_is_still_readable_when_scope_disabled(self, layout):
        root, vault = layout
        assert fs.is_path_allowed(str(vault / "notes" / "a.md"), _cfg(str(vault))) == (True, "")

    def test_write_denied_path_is_still_readable_when_scope_enabled(self, layout):
        root, vault = layout
        cfg = _cfg(str(vault), enabled=True)
        assert fs.is_path_allowed(str(vault / "notes" / "a.md"), cfg) == (True, "")

    def test_existing_denied_paths_semantics_unchanged(self, layout):
        root, vault = layout
        cfg = {"security": {"filesystem_scope": {"enabled": True, "denied_paths": [str(vault)]}}}
        assert fs.is_path_allowed(str(vault / "notes" / "a.md"), cfg)[0] is False
        assert fs.get_filesystem_scope_config(cfg) == {
            "enabled": True,
            "allowed_paths": [],
            "denied_paths": [str(vault)],
            "allow_symlinks": True,
            "allow_absolute_paths": True,
        }
