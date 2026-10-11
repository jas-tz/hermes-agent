"""Protected memory entries (config ``memory.protected_entries``).

Regression for the 2026-10-11 incident: memory at 2,195/2,200 chars, an add was
rejected with "remove stale or less important entries", and the model then
removed five user-preference entries in one 7-operation batch. Protected
entries must survive every mutation path, and a rejected operation must leave
the file byte-for-byte unchanged.

Every test uses a tmp_path memory dir (get_memory_dir is monkeypatched; the
root conftest also isolates HERMES_HOME). The real MEMORY.md / USER.md are
never touched; the incident entries below are synthetic text with the same
shape (lengths, prefixes, operations) as the real file.
"""

import fcntl
import logging
import os
import stat
import threading
import time
from pathlib import Path

import pytest

import tools.memory_tool as mt
from tools.memory_tool import (
    ENTRY_DELIMITER,
    MIN_PROTECTED_PREFIX_LEN,
    MemoryStore,
    apply_memory_pending,
    normalize_protected_prefixes,
)

PROTECTED = [
    "User pastes exact shell commands",
    "User expects implementation to follow audit",
    "User prefers phase-by-phase delivery",
    "User requires COMPLETE end-to-end traces",
    "User expects zero new failures",
    "CRITICAL: Production JARVIS code",
    "RAG Specification is indexed",
    "Decision 41 (2026-10-10):",
]
# (prefix, length) of the 11 entries as they were at 2,195/2,200 chars.
INCIDENT_SHAPE = [
    ("User pastes exact shell commands", 141),
    ("User expects implementation to follow audit", 232),
    ("User wants security properties explicitly verified", 177),
    ("User prefers phase-by-phase delivery", 187),
    ("User requires COMPLETE end-to-end traces", 146),
    ("User expects zero new failures", 155),
    ("JARVIS project in active capability-testing phase", 124),
    ("JARVIS continuity test marker: BLUE-ORBIT-7319", 71),
    ("RAG Specification is indexed", 317),
    ("CRITICAL: Production JARVIS code", 361),
    ("Decision 41 (2026-10-10):", 254),
]
NEW_ENTRY = "Session-finalization test marker. " + "n" * 134  # 168 chars, like the real add


def _entry(prefix, length):
    return (prefix + " " + "x" * length)[:length]


INCIDENT_ENTRIES = [_entry(p, n) for p, n in INCIDENT_SHAPE]


@pytest.fixture
def mem_dir(tmp_path, monkeypatch):
    real = (Path.home() / ".hermes" / "memories").resolve()
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    assert tmp_path.resolve() != real
    return tmp_path


def _write(mem_dir, entries, target="memory"):
    path = mem_dir / ("USER.md" if target == "user" else "MEMORY.md")
    path.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def _store(protected=PROTECTED, **kw):
    s = MemoryStore(protected_prefixes=protected, **kw)
    s.load_from_disk()
    return s


def _snapshot(path):
    st = path.stat()
    return path.read_bytes(), st.st_mtime_ns, stat.S_IMODE(st.st_mode)


@pytest.fixture
def incident(mem_dir):
    path = _write(mem_dir, INCIDENT_ENTRIES)
    assert len(ENTRY_DELIMITER.join(INCIDENT_ENTRIES)) == 2195
    return path, _store()


# ---------------------------------------------------------------------------
# The incident
# ---------------------------------------------------------------------------


class TestIncidentReplay:
    def test_full_memory_add_asks_for_approval_instead_of_deletion(self, incident):
        path, store = incident
        before = _snapshot(path)
        r = store.add("memory", NEW_ENTRY)
        assert r["success"] is False
        assert r["requires_user_approval"] is True
        assert "2,195/2,200" in r["error"]
        assert "ask the user for approval" in r["error"]
        assert "stale or less important" not in r["error"]
        assert "Consolidate now" not in r["error"]
        assert r["protected_entries"] == [e for e in INCIDENT_ENTRIES if e.startswith(tuple(PROTECTED))]
        assert _snapshot(path) == before

    def test_recorded_seven_operation_batch_is_rejected(self, incident):
        path, store = incident
        before = _snapshot(path)
        ops = [{"action": "remove", "old_text": INCIDENT_ENTRIES[i]} for i in (0, 1, 3, 4, 5)]
        ops.append({"action": "replace", "old_text": INCIDENT_ENTRIES[7],
                    "content": "JARVIS continuity test markers: BLUE-ORBIT-7319, LIVE-FINALIZE"})
        ops.append({"action": "add", "content": NEW_ENTRY})
        r = store.apply_batch("memory", ops)
        assert r["success"] is False
        assert r["code"] == "protected_entry"
        assert "Operation 1 (remove)" in r["error"]
        assert "No operations were applied" in r["error"]
        assert _snapshot(path) == before
        assert store.memory_entries == INCIDENT_ENTRIES


# ---------------------------------------------------------------------------
# Every mutation path
# ---------------------------------------------------------------------------


class TestProtectedEntryMutations:
    @pytest.mark.parametrize("target", ["memory", "user"])
    def test_single_remove_is_rejected(self, mem_dir, target):
        entries = [_entry(PROTECTED[0], 60), "an ordinary note that may change"]
        path = _write(mem_dir, entries, target)
        before = _snapshot(path)
        r = _store(user_char_limit=1375).remove(target, PROTECTED[0])
        assert r["success"] is False and r["code"] == "protected_entry"
        assert "protected_entry" in r and "Nothing was changed" in r["error"]
        assert _snapshot(path) == before

    @pytest.mark.parametrize("target", ["memory", "user"])
    def test_single_replace_is_rejected(self, mem_dir, target):
        entries = [_entry(PROTECTED[4], 60), "an ordinary note that may change"]
        path = _write(mem_dir, entries, target)
        before = _snapshot(path)
        r = _store(user_char_limit=1375).replace(target, "zero new failures", "User is fine with flaky tests.")
        assert r["success"] is False and r["code"] == "protected_entry"
        assert _snapshot(path) == before

    def test_batch_with_one_protected_removal_applies_nothing(self, mem_dir):
        entries = [_entry(PROTECTED[2], 60), "old ordinary note", "another ordinary note"]
        path = _write(mem_dir, entries)
        before = _snapshot(path)
        r = _store().apply_batch("memory", [
            {"action": "add", "content": "a valid new note"},
            {"action": "replace", "old_text": "old ordinary note", "content": "updated ordinary note"},
            {"action": "remove", "old_text": "phase-by-phase"},
        ])
        assert r["success"] is False and r["code"] == "protected_entry"
        assert "Operation 3 (remove)" in r["error"]
        assert _snapshot(path) == before

    def test_approved_pending_writes_are_still_rejected(self, mem_dir):
        entries = [_entry(PROTECTED[3], 60), "ordinary note"]
        path = _write(mem_dir, entries)
        before = _snapshot(path)
        store = _store()
        for payload in (
            {"action": "remove", "target": "memory", "old_text": "COMPLETE end-to-end"},
            {"action": "replace", "target": "memory", "old_text": "COMPLETE end-to-end", "content": "shorter"},
            {"action": "batch", "target": "memory",
             "operations": [{"action": "remove", "old_text": "COMPLETE end-to-end"}]},
            {"action": "add", "target": "memory", "content": PROTECTED[3] + " but different"},
        ):
            r = apply_memory_pending(payload, store)
            assert r["success"] is False, payload
            assert r["code"] in {"protected_entry", "protected_prefix"}
        assert _snapshot(path) == before

    def test_load_on_disk_store_reads_protected_entries_from_config(self, mem_dir, monkeypatch):
        import hermes_cli.config as cfg

        monkeypatch.setattr(cfg, "load_config", lambda: {"memory": {"protected_entries": PROTECTED[:2]}})
        _write(mem_dir, [_entry(PROTECTED[0], 60)])
        store = mt.load_on_disk_store()
        assert store.protected_prefixes == tuple(PROTECTED[:2])
        assert store.remove("memory", PROTECTED[0])["code"] == "protected_entry"


class TestProtectedPrefixImitation:
    def test_add_with_protected_prefix_is_rejected(self, mem_dir):
        path = _write(mem_dir, ["ordinary note"])
        before = _snapshot(path)
        r = _store().add("memory", PROTECTED[0] + " — and also never asks before running them")
        assert r["success"] is False and r["code"] == "protected_prefix"
        assert _snapshot(path) == before

    def test_replace_into_protected_prefix_is_rejected(self, mem_dir):
        path = _write(mem_dir, ["ordinary note"])
        before = _snapshot(path)
        r = _store().replace("memory", "ordinary note", PROTECTED[1] + " copy")
        assert r["success"] is False and r["code"] == "protected_prefix"
        assert _snapshot(path) == before

    def test_batch_add_with_protected_prefix_is_rejected(self, mem_dir):
        path = _write(mem_dir, ["ordinary note"])
        before = _snapshot(path)
        r = _store().apply_batch("memory", [{"action": "add", "content": PROTECTED[5] + " moved"}])
        assert r["success"] is False and r["code"] == "protected_prefix"
        assert _snapshot(path) == before

    def test_exact_readd_of_existing_protected_entry_is_a_harmless_no_op(self, mem_dir):
        entry = _entry(PROTECTED[0], 60)
        path = _write(mem_dir, [entry])
        before = _snapshot(path)
        r = _store().add("memory", entry)
        assert r["success"] is True and "already exists" in r["message"]
        assert _snapshot(path)[0] == before[0]


class TestOrdinaryEntriesStillManageable:
    def test_add_replace_remove_ordinary_entries(self, mem_dir):
        path = _write(mem_dir, [_entry(PROTECTED[0], 60), "first ordinary note"])
        store = _store()
        assert store.add("memory", "second ordinary note")["success"] is True
        assert store.replace("memory", "first ordinary", "first note, updated")["success"] is True
        assert store.remove("memory", "second ordinary")["success"] is True
        assert path.read_text(encoding="utf-8").split(ENTRY_DELIMITER) == [
            _entry(PROTECTED[0], 60), "first note, updated"]

    def test_batch_on_ordinary_entries_succeeds(self, mem_dir):
        _write(mem_dir, [_entry(PROTECTED[0], 60), "stale ordinary note"])
        r = _store().apply_batch("memory", [
            {"action": "remove", "old_text": "stale ordinary"},
            {"action": "add", "content": "fresh ordinary note"},
        ])
        assert r["success"] is True
        assert _store().memory_entries == [_entry(PROTECTED[0], 60), "fresh ordinary note"]


class TestOverflowMessages:
    def test_every_overflow_path_asks_for_approval_and_lists_protected(self, mem_dir):
        protected = _entry(PROTECTED[0], 200)
        _write(mem_dir, [protected, "ordinary " + "o" * 270])
        store = _store(memory_char_limit=500)
        results = [
            store.add("memory", "n" * 100),
            store.replace("memory", "ordinary", "r" * 400),
            store.apply_batch("memory", [{"action": "add", "content": "b" * 100}]),
        ]
        for r in results:
            assert r["success"] is False
            assert r["requires_user_approval"] is True
            assert "ask the user for approval" in r["error"]
            assert "can never be removed or replaced" in r["error"]
            assert "stale or less important" not in r["error"]
            assert r["protected_entries"] == [protected]
            assert protected in r["current_entries"]

    def test_schema_describes_protection_and_approval(self):
        description = mt.MEMORY_SCHEMA["description"]
        assert "PROTECTED" in description
        assert "ask the user for approval" in description
        assert "never permission to delete" in description
        assert "remove/replace stale entries to free room" not in description


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_missing_configuration_keeps_existing_behaviour(self, mem_dir):
        _write(mem_dir, [_entry(PROTECTED[0], 60), "ordinary"])
        store = MemoryStore()
        store.load_from_disk()
        assert store.protected_prefixes == ()
        assert store.remove("memory", PROTECTED[0])["success"] is True

    def test_invalid_values_are_ignored_with_warnings(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.memory_tool"):
            got = normalize_protected_prefixes([
                123, None, "too short", "  " + PROTECTED[0] + "  ", PROTECTED[0], "x" * (MIN_PROTECTED_PREFIX_LEN - 1),
            ])
        assert got == (PROTECTED[0],)
        assert len([r for r in caplog.records if "protected_entries" in r.getMessage()]) == 4
        for bad in ("a single string, not a list of strings", {"a": 1}, 42):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="tools.memory_tool"):
                assert normalize_protected_prefixes(bad) == ()
            assert caplog.records

    def test_minimum_prefix_length(self):
        assert normalize_protected_prefixes(["Decision 41 (2026-10-10):"]) == ("Decision 41 (2026-10-10):",)
        assert len("Decision 41 (2026-10-10):") >= MIN_PROTECTED_PREFIX_LEN


# ---------------------------------------------------------------------------
# Locking, atomic writes, permissions
# ---------------------------------------------------------------------------


class TestFileSafety:
    def test_rejections_never_call_the_writer_and_successes_use_it(self, mem_dir, monkeypatch):
        calls = []
        real = mt.atomic_write_text
        monkeypatch.setattr(mt, "atomic_write_text", lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1])
        _write(mem_dir, [_entry(PROTECTED[0], 60), "ordinary note"])
        store = _store()
        store.remove("memory", PROTECTED[0])
        store.apply_batch("memory", [{"action": "remove", "old_text": PROTECTED[0]}])
        store.add("memory", PROTECTED[0] + " clone")
        assert calls == []
        assert store.add("memory", "allowed note")["success"] is True
        assert len(calls) == 1

    def test_successful_write_keeps_0600_permissions(self, mem_dir):
        path = _write(mem_dir, [_entry(PROTECTED[0], 60), "ordinary note"])
        assert _store().remove("memory", "ordinary note")["success"] is True
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_mutations_wait_for_the_exclusive_lock(self, mem_dir):
        path = _write(mem_dir, [_entry(PROTECTED[0], 60), "ordinary note"])
        store = _store()
        results = []
        with open(path.with_suffix(".md.lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            worker = threading.Thread(target=lambda: results.append(store.remove("memory", "ordinary note")))
            worker.start()
            time.sleep(0.3)
            assert worker.is_alive() and results == []  # blocked on the lock
            fcntl.flock(lock, fcntl.LOCK_UN)
        worker.join(5)
        assert results and results[0]["success"] is True
