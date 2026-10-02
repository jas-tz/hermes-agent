"""HermesRAGProvider wiring for the shared embedding service.

A fake ``rag_memory`` module records whether rag_memory's no-local-fallback
mode was active for each call, so these tests never load a model, open a
socket, or touch Qdrant.
"""

import contextlib
import logging
import sys
import threading
import types

import pytest

import plugins.memory.jarvis as jarvis
from plugins.memory.jarvis import HermesRAGProvider, prewarm_embedding_model_async

QUERY = "what did we decide about the qdrant snapshot backup schedule"


class FakeUnavailable(RuntimeError):
    pass


def _fake_rag_memory(*, with_context_manager=True, embed_raises=False):
    state = threading.local()
    mod = types.ModuleType("rag_memory")
    mod.EMBED_SOCKET_ENV = "JARVIS_EMBED_SOCKET"
    mod.calls = []

    def _no_fallback():
        return getattr(state, "no_fallback", False)

    if with_context_manager:
        @contextlib.contextmanager
        def local_fallback_disabled():
            previous = _no_fallback()
            state.no_fallback = True
            try:
                yield
            finally:
                state.no_fallback = previous

        mod.local_fallback_disabled = local_fallback_disabled

    def search_memory(text, limit=5, filter=None):
        mod.calls.append(("search", text, _no_fallback()))
        return [{"score": 0.9, "text": "stored note", "metadata": {}}]

    def embed_text(text, prefix="passage:"):
        mod.calls.append(("embed", text, _no_fallback()))
        if embed_raises:
            raise FakeUnavailable("embedding service unreachable")
        return [0.0] * 384

    def _get_model():
        mod.calls.append(("load_model", None, _no_fallback()))

    mod.search_memory = search_memory
    mod.embed_text = embed_text
    mod._get_model = _get_model
    return mod


@pytest.fixture
def provider(tmp_path):
    p = HermesRAGProvider()
    p._hermes_home = str(tmp_path)  # continuity lookup finds no db: RAG only
    return p


@pytest.fixture
def fresh_prewarm(monkeypatch, tmp_path):
    monkeypatch.setattr(jarvis, "_embedding_prewarm_started", False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    (tmp_path / "rag").mkdir()
    (tmp_path / "rag" / "rag_memory.py").write_text("")
    return str(tmp_path)


def _run_prewarm(home):
    thread = prewarm_embedding_model_async(home)
    assert thread is not None
    thread.join(10)
    assert not thread.is_alive()


def test_prefetch_search_runs_without_local_fallback(monkeypatch, provider):
    fake = _fake_rag_memory()
    monkeypatch.setitem(sys.modules, "rag_memory", fake)
    out = provider.prefetch(QUERY)
    assert fake.calls == [("search", QUERY, True)]
    assert "jarvis_rag" in out and "stored note" in out


def test_prefetch_works_with_rag_memory_lacking_context_manager(monkeypatch, provider):
    fake = _fake_rag_memory(with_context_manager=False)
    monkeypatch.setitem(sys.modules, "rag_memory", fake)
    out = provider.prefetch(QUERY)
    assert fake.calls == [("search", QUERY, False)]
    assert "stored note" in out


def test_prewarm_with_service_checks_service_and_never_loads(monkeypatch, fresh_prewarm, caplog):
    fake = _fake_rag_memory()
    monkeypatch.setitem(sys.modules, "rag_memory", fake)
    monkeypatch.setenv("JARVIS_EMBED_SOCKET", "/tmp/jarvis-embed-test.sock")
    with caplog.at_level(logging.INFO, logger="plugins.memory.jarvis"):
        _run_prewarm(fresh_prewarm)
    assert fake.calls == [("embed", "warmup", True)]
    assert any("shared embedding service ready" in r.getMessage() for r in caplog.records)


def test_prewarm_with_service_down_warns_and_never_loads(monkeypatch, fresh_prewarm, caplog):
    fake = _fake_rag_memory(embed_raises=True)
    monkeypatch.setitem(sys.modules, "rag_memory", fake)
    monkeypatch.setenv("JARVIS_EMBED_SOCKET", "/tmp/jarvis-embed-test.sock")
    with caplog.at_level(logging.WARNING, logger="plugins.memory.jarvis"):
        _run_prewarm(fresh_prewarm)
    assert fake.calls == [("embed", "warmup", True)]
    assert any("not available" in r.getMessage() for r in caplog.records)


def test_prewarm_without_service_keeps_local_behavior(monkeypatch, fresh_prewarm):
    fake = _fake_rag_memory()
    monkeypatch.setitem(sys.modules, "rag_memory", fake)
    monkeypatch.delenv("JARVIS_EMBED_SOCKET", raising=False)
    _run_prewarm(fresh_prewarm)
    assert fake.calls == [("embed", "warmup", False)]


def test_prewarm_runs_once_per_process(monkeypatch, fresh_prewarm):
    fake = _fake_rag_memory()
    monkeypatch.setitem(sys.modules, "rag_memory", fake)
    monkeypatch.setenv("JARVIS_EMBED_SOCKET", "/tmp/jarvis-embed-test.sock")
    _run_prewarm(fresh_prewarm)
    assert prewarm_embedding_model_async(fresh_prewarm) is None
    assert len(fake.calls) == 1
