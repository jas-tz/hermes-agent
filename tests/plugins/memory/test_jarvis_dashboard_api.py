"""Tests for the read-only JARVIS dashboard plugin API (Stages 1-2).

``plugins/memory/jarvis/dashboard/plugin_api.py`` is loaded straight from its
file and mounted on a bare FastAPI app, mirroring the kanban plugin API tests.
Every test runs against a temporary HERMES_HOME with a stubbed ``rag_memory``
module, so no test can reach Qdrant, the embedding service, or any production
memory. Forbidden write operations (rag_memory writes and write_approval
staging/discard/gating) raise ``_ForbiddenCall`` — a BaseException, so a
route's ``except Exception`` cannot swallow it and the test fails loudly.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import continuity_db as cdb

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DASHBOARD_DIR = _REPO_ROOT / "plugins" / "memory" / "jarvis" / "dashboard"
_PLUGIN_FILE = _DASHBOARD_DIR / "plugin_api.py"

_RAG_FORBIDDEN_FUNCTIONS = (
    "add_memory",
    "store_memory_with_provenance",
    "delete_memory",
    "store_session_learnings",
    "rollback_cross_session_learnings",
    "delete_obsidian_document",
    "extract_session_learnings",
)

_WRITE_APPROVAL_FORBIDDEN_FUNCTIONS = ("stage_write", "discard_pending", "evaluate_gate")

# Looks like a credential to tools.secret_redaction.
_FAKE_SECRET = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"


class _ForbiddenCall(BaseException):
    """Raised by any forbidden write; not catchable by ``except Exception``."""


def _forbidden(where: str):
    def _fail(*_a, **_k):
        raise _ForbiddenCall(f"dashboard API must never call {where}")
    return _fail


@pytest.fixture(autouse=True)
def forbid_write_approval_mutations(monkeypatch):
    from tools import write_approval

    for name in _WRITE_APPROVAL_FORBIDDEN_FUNCTIONS:
        monkeypatch.setattr(write_approval, name, _forbidden(f"write_approval.{name}"))


def _load_plugin_module():
    spec = importlib.util.spec_from_file_location("hermes_jarvis_dashboard_plugin_test", _PLUGIN_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin():
    return _load_plugin_module()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("JARVIS_QDRANT_URL", "JARVIS_QDRANT_PATH", "JARVIS_QDRANT_TIMEOUT", "JARVIS_EMBED_SOCKET"):
        monkeypatch.delenv(var, raising=False)
    # _load_rag_memory may prepend $HERMES_HOME/rag; restore sys.path afterwards.
    monkeypatch.setattr(sys, "path", list(sys.path))
    return home


def _write_config(home: Path, provider: str | None) -> None:
    memory = {} if provider is None else {"provider": provider}
    (home / "config.yaml").write_text(json.dumps({"memory": memory}), encoding="utf-8")


class _RagStub(types.ModuleType):
    """Stand-in for rag_memory: reads are scripted and recorded, writes are forbidden."""

    def __init__(self, health=None, raise_health: bool = False, search_rows=None,
                 learning_rows=None, read_error: Exception | None = None):
        super().__init__("rag_memory")
        self.health = health
        self.raise_health = raise_health
        self.health_calls = 0
        self.search_rows = search_rows if search_rows is not None else []
        self.learning_rows = learning_rows if learning_rows is not None else []
        self.read_error = read_error
        self.search_calls: list = []
        self.learning_calls: list = []
        for name in _RAG_FORBIDDEN_FUNCTIONS:
            setattr(self, name, _forbidden(f"rag_memory.{name}"))

    def health_check(self):
        self.health_calls += 1
        if self.raise_health:
            raise RuntimeError(f"boom {_FAKE_SECRET} http://127.0.0.1:6333")
        return self.health

    def search_memory(self, query, limit=5, filter=None):
        self.search_calls.append((query, limit, filter))
        if self.read_error is not None:
            raise self.read_error
        return self.search_rows

    def retrieve_cross_session_learnings(self, query, limit=5):
        self.learning_calls.append((query, limit))
        if self.read_error is not None:
            raise self.read_error
        return self.learning_rows


@pytest.fixture
def install_rag(hermes_home, monkeypatch):
    """Place a rag_memory.py marker in $HERMES_HOME/rag and register a stub module."""

    def _install(stub: _RagStub) -> _RagStub:
        rag_dir = hermes_home / "rag"
        rag_dir.mkdir(exist_ok=True)
        (rag_dir / "rag_memory.py").write_text(
            "raise RuntimeError('the real file must never be imported in tests')\n",
            encoding="utf-8",
        )
        monkeypatch.setitem(sys.modules, "rag_memory", stub)
        return stub

    return _install


@pytest.fixture
def client(plugin):
    app = FastAPI()
    app.include_router(plugin.router, prefix="/api/plugins/jarvis")
    return TestClient(app)


def _server_health(**overrides):
    health = {
        "ok": True, "mode": "server", "target": "http://127.0.0.1:6333",
        "reachable": True, "collection": "jarvis_memory", "collection_exists": True,
        "error": None, "server_required": True,
    }
    health.update(overrides)
    return health


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_continuity(home: Path) -> Path:
    db_path = home / "continuity.db"
    with cdb.connect_closing(db_path=db_path) as conn:
        active = cdb.create_goal(conn, title=f"Ship desktop {_FAKE_SECRET}",
                                 provenance={"source": "user_confirmed", "note": _FAKE_SECRET,
                                             "nested": {"secret": _FAKE_SECRET}})
        cdb.create_goal(conn, title="Blocked goal", status="blocked")
        cdb.create_goal(conn, title="Finished goal", status="done")
        cdb.create_task(conn, title="Open task", goal_id=active)
        cdb.create_task(conn, title="Running task", status="in_progress")
        cdb.create_task(conn, title="Done task", status="done")
        old = cdb.create_decision(conn, decision="Use plan A")
        cdb.supersede_decision(conn, old, decision="Use plan B", rationale=f"key {_FAKE_SECRET}")
    return db_path


# ---------------------------------------------------------------------------
# Discovery and mounting through the real web server
# ---------------------------------------------------------------------------

def test_manifest_is_discovered_as_backend_only_bundled_plugin():
    from hermes_cli import web_server

    plugins = {p["name"]: p for p in web_server._discover_dashboard_plugins()}
    jarvis = plugins["jarvis"]
    assert jarvis["source"] == "bundled"
    assert jarvis["has_api"] is True
    assert jarvis["_api_file"] == "plugin_api.py"
    assert jarvis["tab"].get("hidden") is True
    assert Path(jarvis["_dir"]).resolve() == _DASHBOARD_DIR.resolve()


def test_routes_are_mounted_on_web_server_and_require_session_token():
    from hermes_cli import web_server

    paths = {getattr(r, "path", "") for r in web_server.app.routes}
    assert "/api/plugins/jarvis/status" in paths
    assert "/api/plugins/jarvis/continuity" in paths

    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    try:
        anonymous = TestClient(web_server.app)
        assert anonymous.get("/api/plugins/jarvis/status").status_code == 401
        assert anonymous.get("/api/plugins/jarvis/continuity").status_code == 401
    finally:
        if previous is None:
            delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous


# ---------------------------------------------------------------------------
# Read-only guarantees
# ---------------------------------------------------------------------------

def test_router_exposes_only_get_routes(plugin):
    methods = set()
    for route in plugin.router.routes:
        methods |= set(getattr(route, "methods", set()))
    assert methods and methods <= {"GET", "HEAD"}


def test_plugin_source_imports_no_write_paths():
    tree = ast.parse(_PLUGIN_FILE.read_text(encoding="utf-8"))
    imported = set()  # fully qualified: "pkg.mod" or "pkg.mod.name"
    continuity_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            imported |= {f"{module}.{a.name}" for a in node.names}
            if module == "hermes_cli.continuity_db":
                continuity_names |= {a.name for a in node.names}
    forbidden = {"tools.approval", "plugins.memory.jarvis", "hermes_cli.write_approval_commands",
                 "plugins.memory.jarvis.__init__"}
    assert not imported & forbidden
    assert continuity_names == {"list_goals", "list_tasks", "list_decisions"}

    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & set(_RAG_FORBIDDEN_FUNCTIONS)
    assert not attrs & set(_WRITE_APPROVAL_FORBIDDEN_FUNCTIONS)
    assert not attrs & {"connect_closing", "executescript", "get_pending", "pending_count"}
    write_approval_attrs = {
        n.attr for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "write_approval"
    }
    assert write_approval_attrs == {"list_pending", "MEMORY", "CONTINUITY"}


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", [None, "", "holographic", "JARVIS "])
def test_status_disabled_when_provider_is_not_jarvis(client, hermes_home, install_rag, provider):
    _write_config(hermes_home, provider)
    stub = install_rag(_RagStub(health=_server_health()))
    _seed_continuity(hermes_home)

    res = client.get("/api/plugins/jarvis/status")

    assert res.status_code == 200
    assert res.json() == {"enabled": False, "reason": "provider_not_jarvis"}
    assert stub.health_calls == 0
    assert stub.search_calls == [] and stub.learning_calls == []


def test_status_available_whitelists_health_fields(client, hermes_home, install_rag):
    _write_config(hermes_home, "jarvis")
    install_rag(_RagStub(health=_server_health(target=f"http://user:{_FAKE_SECRET}@127.0.0.1:6333")))

    res = client.get("/api/plugins/jarvis/status")

    assert res.status_code == 200
    body = res.json()
    assert body["enabled"] is True
    assert body["rag"] == {"state": "available", "reason": None, "mode": "server",
                           "collection_exists": True, "server_required": True}
    assert body["continuity"] == {"state": "available", "goals": 0, "tasks": 0, "decisions": 0}
    assert _FAKE_SECRET not in res.text and "6333" not in res.text


@pytest.mark.parametrize(
    ("health", "reason", "mode"),
    [
        (_server_health(ok=False, reachable=False, collection_exists=None,
                        error=f"ConnectError {_FAKE_SECRET}"), "server_unreachable", "server"),
        (_server_health(ok=False, collection_exists=False,
                        error="collection 'jarvis_memory' not found"), "collection_missing", "server"),
        ({"ok": True, "mode": "local", "target": "/home/x/.hermes/rag/qdrant_data", "reachable": None,
          "collection_exists": None, "error": None, "server_required": False},
         "local_mode_unverified", "local"),
        ({"ok": False, "mode": None, "target": None, "reachable": None, "collection_exists": None,
          "error": f"JARVIS_QDRANT_URL is not set {_FAKE_SECRET}", "server_required": True},
         "config_error", None),
    ],
)
def test_status_reports_unavailable_without_leaking_details(client, hermes_home, install_rag,
                                                            health, reason, mode):
    _write_config(hermes_home, "jarvis")
    install_rag(_RagStub(health=health))

    body = client.get("/api/plugins/jarvis/status").json()

    assert body["rag"]["state"] == "unavailable"
    assert body["rag"]["reason"] == reason
    assert body["rag"]["mode"] == mode
    text = json.dumps(body)
    assert _FAKE_SECRET not in text and "qdrant_data" not in text and "not found" not in text


def test_status_health_check_exception_is_generic(client, hermes_home, install_rag, caplog):
    _write_config(hermes_home, "jarvis")
    install_rag(_RagStub(raise_health=True))

    with caplog.at_level("DEBUG"):
        res = client.get("/api/plugins/jarvis/status")

    assert res.status_code == 200
    assert res.json()["rag"]["reason"] == "health_check_failed"
    assert "boom" not in res.text and _FAKE_SECRET not in res.text
    assert _FAKE_SECRET not in caplog.text and "boom" not in caplog.text


def test_status_engine_not_installed_does_not_import(client, hermes_home, monkeypatch):
    _write_config(hermes_home, "jarvis")
    stub = _RagStub(health=_server_health())
    monkeypatch.setitem(sys.modules, "rag_memory", stub)

    body = client.get("/api/plugins/jarvis/status").json()

    assert body["rag"]["state"] == "unavailable"
    assert body["rag"]["reason"] == "engine_not_installed"
    assert stub.health_calls == 0


def test_status_counts_visible_continuity_records(client, hermes_home, install_rag):
    _write_config(hermes_home, "jarvis")
    install_rag(_RagStub(health=_server_health()))
    _seed_continuity(hermes_home)

    body = client.get("/api/plugins/jarvis/status").json()

    assert body["continuity"] == {"state": "available", "goals": 2, "tasks": 2, "decisions": 1}


# ---------------------------------------------------------------------------
# GET /continuity
# ---------------------------------------------------------------------------

def test_continuity_disabled_does_not_touch_db(client, hermes_home):
    _write_config(hermes_home, "holographic")
    db_path = _seed_continuity(hermes_home)
    before = _sha(db_path)

    res = client.get("/api/plugins/jarvis/continuity")

    assert res.status_code == 200
    assert res.json() == {"enabled": False, "reason": "provider_not_jarvis"}
    assert _sha(db_path) == before


def test_continuity_missing_db_is_empty_and_not_created(client, hermes_home):
    _write_config(hermes_home, "jarvis")

    res = client.get("/api/plugins/jarvis/continuity")

    assert res.status_code == 200
    body = res.json()
    for kind in ("goals", "tasks", "decisions"):
        assert body[kind] == {"total": 0, "items": [], "truncated": False}
    assert not (hermes_home / "continuity.db").exists()


def test_continuity_list_filters_redacts_and_leaves_db_unchanged(client, hermes_home):
    _write_config(hermes_home, "jarvis")
    db_path = _seed_continuity(hermes_home)
    before = _sha(db_path)

    res = client.get("/api/plugins/jarvis/continuity")

    assert res.status_code == 200
    body = res.json()
    assert body["view"] == "list"
    assert sorted(g["title"].split()[0] for g in body["goals"]["items"]) == ["Blocked", "Ship"]
    assert sorted(t["title"] for t in body["tasks"]["items"]) == ["Open task", "Running task"]
    assert [d["decision"] for d in body["decisions"]["items"]] == ["Use plan B"]
    assert body["goals"]["total"] == 2 and body["goals"]["truncated"] is False

    goal = next(g for g in body["goals"]["items"] if g["title"].startswith("Ship"))
    assert set(goal) == set(_load_plugin_module()._GOAL_FIELDS) | {"provenance"}
    assert goal["provenance"]["source"] == "user_confirmed"
    assert "nested" not in goal["provenance"]
    assert _FAKE_SECRET not in res.text
    assert "[REDACTED]" in goal["title"]

    assert _sha(db_path) == before


@pytest.mark.parametrize("view", ["goals", "tasks", "decisions"])
def test_continuity_single_view_returns_only_that_kind(client, hermes_home, view):
    _write_config(hermes_home, "jarvis")
    _seed_continuity(hermes_home)

    body = client.get(f"/api/plugins/jarvis/continuity?view={view}").json()

    assert body["view"] == view
    assert set(body) == {"enabled", "view", view}


def test_continuity_invalid_view_is_rejected(client, hermes_home):
    _write_config(hermes_home, "jarvis")

    res = client.get("/api/plugins/jarvis/continuity?view=status")

    assert res.status_code == 400
    assert res.json() == {"detail": "Invalid view"}


def test_continuity_caps_items_but_reports_total(client, hermes_home, plugin, monkeypatch):
    _write_config(hermes_home, "jarvis")
    monkeypatch.setattr(plugin, "CONTINUITY_MAX_ITEMS", 2)
    db_path = hermes_home / "continuity.db"
    with cdb.connect_closing(db_path=db_path) as conn:
        for i in range(3):
            cdb.create_goal(conn, title=f"Goal {i}")

    body = client.get("/api/plugins/jarvis/continuity?view=goals").json()

    assert body["goals"]["total"] == 3
    assert len(body["goals"]["items"]) == 2
    assert body["goals"]["truncated"] is True


def test_continuity_unreadable_schema_is_generic_503_and_never_initialised(client, hermes_home):
    _write_config(hermes_home, "jarvis")
    db_path = hermes_home / "continuity.db"
    sqlite3.connect(db_path).close()  # valid, empty SQLite file: no tables
    before = _sha(db_path)

    res = client.get("/api/plugins/jarvis/continuity")

    assert res.status_code == 503
    assert res.json() == {"detail": "Continuity data is unavailable"}
    assert _sha(db_path) == before
    conn = sqlite3.connect(db_path)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    assert tables == []


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------

_QUERY = "distinctive-query-marker about deadlines"


def _row(text="Remember the deploy window", score=0.91, provenance=None, **payload_extra):
    payload = {"text": text, "type": "memory",
               "provenance": provenance if provenance is not None else {
                   "source": "user_confirmed", "provenance_status": "tracked",
                   "stored_at": "2026-10-01T00:00:00+00:00"}}
    payload.update(payload_extra)
    return {"score": score, "text": text, "metadata": payload}


def _write_pending(home: Path, subsystem: str, records: list) -> Path:
    pending_dir = home / "pending" / subsystem
    pending_dir.mkdir(parents=True, exist_ok=True)
    # File names are positional so hostile record ids stay inside the record.
    for index, record in enumerate(records):
        (pending_dir / f"record{index}.json").write_text(json.dumps(record), encoding="utf-8")
    return pending_dir


def _pending_record(pid, created_at, summary="Add: prefers dark mode", **extra):
    record = {"id": pid, "subsystem": "memory", "action": "add", "summary": summary,
              "origin": "foreground", "created_at": created_at,
              "payload": {"action": "add", "content": f"full content {_FAKE_SECRET}"}}
    record.update(extra)
    return record


# ---------------------------------------------------------------------------
# GET /search and GET /learnings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["search", "learnings"])
def test_rag_routes_disabled_when_provider_is_not_jarvis(client, hermes_home, install_rag, route):
    _write_config(hermes_home, "holographic")
    stub = install_rag(_RagStub(health=_server_health(), search_rows=[_row()], learning_rows=[_row()]))

    res = client.get(f"/api/plugins/jarvis/{route}", params={"q": _QUERY})

    assert res.status_code == 200
    assert res.json() == {"enabled": False, "reason": "provider_not_jarvis"}
    assert stub.health_calls == 0 and stub.search_calls == [] and stub.learning_calls == []


def test_search_returns_only_permitted_fields(client, hermes_home, install_rag):
    _write_config(hermes_home, "jarvis")
    row = _row(
        text=f"Deploy uses token {_FAKE_SECRET}",
        provenance={
            "source": "user_confirmed",
            "source_type": "conversation",
            "source_id": "/home/someone/.hermes/sessions/abc.json",
            "origin": "https://internal.example/hook",
            "provenance_status": "tracked",
            "stored_at": "2026-10-01T00:00:00+00:00",
            "note": f"secret {_FAKE_SECRET}",
            "nested": {"key": _FAKE_SECRET},
        },
        document_path="Vault/Private/notes.md",
        chunk_index=3,
        custom_field=f"leak {_FAKE_SECRET}",
    )
    stub = install_rag(_RagStub(health=_server_health(), search_rows=[row]))

    res = client.get("/api/plugins/jarvis/search", params={"q": _QUERY})

    assert res.status_code == 200
    body = res.json()
    assert body["enabled"] is True and body["count"] == 1
    (result,) = body["results"]
    assert set(result) == {"score", "text", "text_truncated", "type", "provenance"}
    assert result["score"] == 0.91
    assert result["type"] == "memory"
    assert "[REDACTED]" in result["text"]
    assert result["provenance"] == {
        "source": "user_confirmed",
        "source_type": "conversation",
        "provenance_status": "tracked",
        "stored_at": "2026-10-01T00:00:00+00:00",
    }
    for leaked in (_FAKE_SECRET, "Vault/Private", "chunk_index", "custom_field", "/home/someone",
                   "internal.example", "nested", "note"):
        assert leaked not in res.text
    assert stub.search_calls == [(_QUERY, 5, None)]
    assert stub.learning_calls == []


@pytest.mark.parametrize(
    ("raw_limit", "expected"),
    [(None, 5), ("", 5), ("0", 5), ("-3", 5), ("abc", 5), ("7", 7), ("20", 20), ("500", 20)],
)
def test_search_limit_is_clamped_to_twenty(client, hermes_home, install_rag, raw_limit, expected):
    _write_config(hermes_home, "jarvis")
    stub = install_rag(_RagStub(health=_server_health(), search_rows=[_row() for _ in range(30)]))
    params = {"q": _QUERY} if raw_limit is None else {"q": _QUERY, "limit": raw_limit}

    body = client.get("/api/plugins/jarvis/search", params=params).json()

    assert stub.search_calls[-1][1] == expected
    assert body["count"] == expected


@pytest.mark.parametrize("route", ["search", "learnings"])
@pytest.mark.parametrize(
    ("query", "detail"),
    [("", "Query is required"), ("   ", "Query is required"),
     ("x" * 321, "Query exceeds maximum length of 320 characters")],
)
def test_rag_query_validation_rejects_without_echo(client, hermes_home, install_rag, route, query, detail):
    _write_config(hermes_home, "jarvis")
    stub = install_rag(_RagStub(health=_server_health()))

    res = client.get(f"/api/plugins/jarvis/{route}", params={"q": query})

    assert res.status_code == 400
    assert res.json() == {"detail": detail}
    assert stub.health_calls == 0 and stub.search_calls == [] and stub.learning_calls == []


def test_query_length_limit_matches_shared_memory_query_cap(plugin, client, hermes_home, install_rag):
    from plugins.memory.query_rewrite import _MAX_QUERY_CHARS

    _write_config(hermes_home, "jarvis")
    stub = install_rag(_RagStub(health=_server_health()))

    assert plugin._query_max_chars() == _MAX_QUERY_CHARS == 320
    assert client.get("/api/plugins/jarvis/search", params={"q": "y" * 320}).status_code == 200
    assert stub.search_calls[0][0] == "y" * 320


@pytest.mark.parametrize("route", ["search", "learnings"])
@pytest.mark.parametrize(
    "health",
    [
        {"ok": True, "mode": "local", "target": "/tmp/qdrant", "reachable": None,
         "collection_exists": None, "error": None, "server_required": False},
        _server_health(ok=False, reachable=False, collection_exists=None),
        {"ok": False, "mode": None, "target": None, "reachable": None,
         "collection_exists": None, "error": "JARVIS_QDRANT_URL is not set", "server_required": True},
    ],
)
def test_rag_routes_unavailable_never_query_the_store(client, hermes_home, install_rag, route, health):
    _write_config(hermes_home, "jarvis")
    stub = install_rag(_RagStub(health=health, search_rows=[_row()], learning_rows=[_row()]))

    res = client.get(f"/api/plugins/jarvis/{route}", params={"q": _QUERY})

    assert res.status_code == 503
    assert res.json()["detail"] in ("Memory search is unavailable", "Learning history is unavailable")
    assert stub.search_calls == [] and stub.learning_calls == []


@pytest.mark.parametrize(("route", "detail"), [("search", "Memory search is unavailable"),
                                               ("learnings", "Learning history is unavailable")])
def test_rag_read_errors_are_generic_and_not_logged(client, hermes_home, install_rag, caplog, route, detail):
    _write_config(hermes_home, "jarvis")
    install_rag(_RagStub(health=_server_health(),
                         read_error=RuntimeError(f"embed failed {_FAKE_SECRET} /run/jarvis-embed.sock")))

    with caplog.at_level("DEBUG"):
        res = client.get(f"/api/plugins/jarvis/{route}", params={"q": _QUERY})

    assert res.status_code == 503
    assert res.json() == {"detail": detail}
    for leaked in (_FAKE_SECRET, "embed failed", "jarvis-embed.sock", _QUERY):
        assert leaked not in res.text
        assert leaked not in caplog.text


def test_search_non_list_result_is_unavailable(client, hermes_home, install_rag):
    _write_config(hermes_home, "jarvis")
    install_rag(_RagStub(health=_server_health(), search_rows={"unexpected": "shape"}))

    res = client.get("/api/plugins/jarvis/search", params={"q": _QUERY})

    assert res.status_code == 503


def test_search_text_is_capped(plugin, client, hermes_home, install_rag):
    _write_config(hermes_home, "jarvis")
    long_text = "a" * (plugin.RESULT_TEXT_MAX_CHARS + 50)
    install_rag(_RagStub(health=_server_health(), search_rows=[_row(text=long_text)]))

    (result,) = client.get("/api/plugins/jarvis/search", params={"q": _QUERY}).json()["results"]

    assert len(result["text"]) == plugin.RESULT_TEXT_MAX_CHARS
    assert result["text_truncated"] is True


def test_search_does_not_log_query_or_content(client, hermes_home, install_rag, caplog):
    _write_config(hermes_home, "jarvis")
    install_rag(_RagStub(health=_server_health(), search_rows=[_row(text="private recollection")]))

    with caplog.at_level("DEBUG"):
        client.get("/api/plugins/jarvis/search", params={"q": _QUERY})

    assert _QUERY not in caplog.text and "private recollection" not in caplog.text


def test_learnings_use_cross_session_reader_with_provenance(client, hermes_home, install_rag):
    _write_config(hermes_home, "jarvis")
    learning = _row(
        text="User prefers phase-by-phase delivery",
        score=0.88,
        provenance={
            "source": "inferred", "source_type": "session_summary", "source_id": "20261001_abcdef",
            "created_at": "2026-10-01T10:00:00+00:00", "provenance_status": "tracked",
            "origin": "cross_session_transfer", "transfer_timestamp": "2026-10-01T10:00:00+00:00",
            "transfer_status": "completed",
        },
    )
    stub = install_rag(_RagStub(health=_server_health(), learning_rows=[learning]))

    res = client.get("/api/plugins/jarvis/learnings", params={"q": _QUERY, "limit": "50"})

    assert res.status_code == 200
    (result,) = res.json()["results"]
    assert result["provenance"] == {
        "source": "inferred", "source_type": "session_summary", "source_id": "20261001_abcdef",
        "provenance_status": "tracked", "origin": "cross_session_transfer",
        "transfer_status": "completed", "created_at": "2026-10-01T10:00:00+00:00",
    }
    assert stub.learning_calls == [(_QUERY, 20)]
    assert stub.search_calls == []


# ---------------------------------------------------------------------------
# GET /pending
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subsystem", [None, "", "skills", "files", "MEMORY"])
def test_pending_rejects_unknown_subsystem(client, hermes_home, subsystem):
    _write_config(hermes_home, "jarvis")
    params = {} if subsystem is None else {"subsystem": subsystem}

    res = client.get("/api/plugins/jarvis/pending", params=params)

    assert res.status_code == 400
    assert res.json() == {"detail": "Invalid subsystem"}


def test_pending_disabled_does_not_read_queue(client, hermes_home, monkeypatch):
    from tools import write_approval

    _write_config(hermes_home, "holographic")
    _write_pending(hermes_home, "memory", [_pending_record("a1", 1.0)])
    calls = []
    monkeypatch.setattr(write_approval, "list_pending", lambda s: calls.append(s) or [])

    res = client.get("/api/plugins/jarvis/pending", params={"subsystem": "memory"})

    assert res.json() == {"enabled": False, "reason": "provider_not_jarvis"}
    assert calls == []


@pytest.mark.parametrize("subsystem", ["memory", "continuity"])
def test_pending_returns_redacted_summaries_without_payload(client, hermes_home, subsystem):
    _write_config(hermes_home, "jarvis")
    other = "continuity" if subsystem == "memory" else "memory"
    _write_pending(hermes_home, subsystem, [
        _pending_record("b2", 20.0, summary=f"Add key {_FAKE_SECRET}"),
        _pending_record("a1", 10.0, origin="background_review"),
        _pending_record("../../etc", 30.0, action="rm -rf /", origin=None),
    ])
    _write_pending(hermes_home, other, [_pending_record("zz", 5.0, summary="other queue")])

    res = client.get("/api/plugins/jarvis/pending", params={"subsystem": subsystem})

    assert res.status_code == 200
    body = res.json()
    assert body["enabled"] is True and body["subsystem"] == subsystem
    assert body["total"] == 3 and body["truncated"] is False
    first, second, third = body["items"]
    assert first == {"id": "a1", "action": "add", "origin": "background_review", "created_at": 10.0,
                     "summary": "Add: prefers dark mode", "summary_truncated": False}
    assert second["id"] == "b2" and "[REDACTED]" in second["summary"]
    assert third["id"] is None and third["action"] is None and third["origin"] is None
    for leaked in (_FAKE_SECRET, "payload", "full content", "other queue", "rm -rf"):
        assert leaked not in res.text


def test_pending_missing_queue_is_empty_and_not_created(client, hermes_home):
    _write_config(hermes_home, "jarvis")

    body = client.get("/api/plugins/jarvis/pending", params={"subsystem": "continuity"}).json()

    assert body["total"] == 0 and body["items"] == []
    assert not (hermes_home / "pending").exists()


def test_pending_caps_items_and_summary(plugin, client, hermes_home, monkeypatch):
    _write_config(hermes_home, "jarvis")
    monkeypatch.setattr(plugin, "PENDING_MAX_ITEMS", 2)
    long_summary = "s" * (plugin.PENDING_SUMMARY_MAX_CHARS + 10)
    _write_pending(hermes_home, "memory",
                   [_pending_record(f"p{i}", float(i), summary=long_summary) for i in range(3)])

    body = client.get("/api/plugins/jarvis/pending", params={"subsystem": "memory"}).json()

    assert body["total"] == 3 and len(body["items"]) == 2 and body["truncated"] is True
    assert len(body["items"][0]["summary"]) == plugin.PENDING_SUMMARY_MAX_CHARS
    assert body["items"][0]["summary_truncated"] is True


def test_pending_read_failure_is_generic(client, hermes_home, monkeypatch, caplog):
    from tools import write_approval

    _write_config(hermes_home, "jarvis")

    def _boom(_subsystem):
        raise OSError(f"permission denied {hermes_home}/pending {_FAKE_SECRET}")

    monkeypatch.setattr(write_approval, "list_pending", _boom)
    with caplog.at_level("DEBUG"):
        res = client.get("/api/plugins/jarvis/pending", params={"subsystem": "memory"})

    assert res.status_code == 503
    assert res.json() == {"detail": "Pending proposals are unavailable"}
    assert _FAKE_SECRET not in caplog.text and "permission denied" not in caplog.text


def test_pending_files_are_left_untouched(client, hermes_home):
    _write_config(hermes_home, "jarvis")
    pending_dir = _write_pending(hermes_home, "memory", [_pending_record("a1", 1.0)])
    before = {p.name: _sha(p) for p in pending_dir.iterdir()}

    client.get("/api/plugins/jarvis/pending", params={"subsystem": "memory"})

    assert {p.name: _sha(p) for p in pending_dir.iterdir()} == before


# ---------------------------------------------------------------------------
# The guards are live: a forbidden write inside a route fails the request
# even though the route catches ``Exception``.
# ---------------------------------------------------------------------------

def test_forbidden_rag_write_inside_search_fails_the_test(client, hermes_home, install_rag):
    _write_config(hermes_home, "jarvis")
    stub = install_rag(_RagStub(health=_server_health()))
    stub.search_memory = lambda query, limit=5, filter=None: stub.add_memory(query)

    with pytest.raises(_ForbiddenCall, match="rag_memory.add_memory"):
        client.get("/api/plugins/jarvis/search", params={"q": _QUERY})


def test_forbidden_rollback_inside_learnings_fails_the_test(client, hermes_home, install_rag):
    _write_config(hermes_home, "jarvis")
    stub = install_rag(_RagStub(health=_server_health()))
    stub.retrieve_cross_session_learnings = lambda query, limit=5: stub.rollback_cross_session_learnings()

    with pytest.raises(_ForbiddenCall, match="rollback_cross_session_learnings"):
        client.get("/api/plugins/jarvis/learnings", params={"q": _QUERY})


def test_forbidden_discard_inside_pending_fails_the_test(client, hermes_home, monkeypatch):
    from tools import write_approval

    _write_config(hermes_home, "jarvis")
    monkeypatch.setattr(write_approval, "list_pending",
                        lambda subsystem: write_approval.discard_pending(subsystem, "a1"))

    with pytest.raises(_ForbiddenCall, match="write_approval.discard_pending"):
        client.get("/api/plugins/jarvis/pending", params={"subsystem": "memory"})
