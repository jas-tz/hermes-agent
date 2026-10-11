"""Session finalize -> HermesRAGProvider.on_session_end -> real rag_memory storage.

What POST /api/sessions/{id}/finalize hands to the provider (the stored
conversation cut by ``_finalizable_transcript``) is run through the real hook,
quality filter, provenance and idempotent storage, against a disposable
temporary Qdrant store. The production store and the live embedding service
are never used: JARVIS_QDRANT_* / JARVIS_EMBED_* are removed for this module
and ``rag_memory.client`` is swapped for a tmp_path client per test.
"""

import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.expanduser("~/.hermes/rag"))

from gateway.platforms.api_server import _finalizable_transcript  # noqa: E402

PRODUCTION_ENV = ("JARVIS_QDRANT_URL", "JARVIS_QDRANT_PATH", "JARVIS_QDRANT_TIMEOUT",
                  "JARVIS_EMBED_SOCKET", "JARVIS_EMBED_TIMEOUT", "JARVIS_EMBED_FALLBACK")
SID = "desktop_1791700000_abcdef01"
KEEP = "The user decided to keep Qdrant in server mode."
CANCELLED = "The user always wants the cancelled partial reply stored."
FAILED = "The user prefers the truncated reply to be remembered."


@pytest.fixture
def rag(monkeypatch, tmp_path):
    for var in PRODUCTION_ENV:
        monkeypatch.delenv(var, raising=False)
    import rag_memory as rm
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection(
        collection_name=rm.COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    monkeypatch.setattr(rm, "QDRANT_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(rm, "client", client)
    yield rm
    client.close()


@pytest.fixture
def provider(tmp_path):
    from plugins.memory.jarvis import HermesRAGProvider

    p = HermesRAGProvider()
    # api_server agents initialise their provider as a primary context.
    p.initialize(SID, hermes_home=str(tmp_path / "hermes_home"), agent_context="primary")
    return p


def _points(rm):
    points, _ = rm.client.scroll(collection_name=rm.COLLECTION_NAME, limit=100, with_payload=True)
    return points


def _conversation(*extra):
    return [
        {"role": "user", "content": "Where should Qdrant run?"},
        {"role": "assistant", "content": KEEP, "finish_reason": "stop"},
        {"role": "user", "content": "And the other thing?"},
        {"role": "assistant", "content": CANCELLED, "finish_reason": None},
        *extra,
        {"role": "user", "content": "One more"},
        {"role": "assistant", "content": FAILED, "finish_reason": "length"},
    ]


def test_finalized_conversation_stores_only_completed_learnings(rag, provider):
    provider.on_session_end(_finalizable_transcript(_conversation()))

    points = _points(rag)
    assert [p.payload["text"] for p in points] == [KEEP]
    provenance = points[0].payload["provenance"]
    assert provenance["source_id"] == SID
    assert provenance["origin"] == "cross_session_transfer"
    assert provenance["source_type"] == "session_summary"


def test_repeated_finalize_does_not_duplicate(rag, provider):
    transcript = _finalizable_transcript(_conversation())
    provider.on_session_end(transcript)
    provider.on_session_end(transcript)
    assert [p.payload["text"] for p in _points(rag)] == [KEEP]


def test_untrusted_automation_result_blocks_the_whole_session(rag, provider):
    # No jarvis_automation plugin module is loaded here, so the D13 check
    # cannot classify the n8n_route_action result and fails closed.
    call = [{"id": "call_n8n", "type": "function",
             "function": {"name": "n8n_route_action", "arguments": "{}"}}]
    automation = (
        {"role": "assistant", "content": "", "tool_calls": call, "finish_reason": "tool_calls"},
        {"role": "tool", "tool_call_id": "call_n8n", "tool_name": "n8n_route_action",
         "content": json.dumps({"ok": True, "action": "gmail.search", "result": "..."})},
        {"role": "assistant", "content": "The user prefers this summary of the inbox.", "finish_reason": "stop"},
    )
    transcript = _finalizable_transcript(_conversation(*automation))
    assert any(m.get("tool_name") == "n8n_route_action" for m in transcript)

    provider.on_session_end(transcript)
    assert _points(rag) == []


def test_partial_failure_is_logged_and_completed_by_the_next_finalize(rag, provider, monkeypatch, caplog):
    second = "The main goal for Q4 is launching the export channel to Kenya."
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": f"{KEEP}\n{second}", "finish_reason": "stop"},
    ]
    real_store = rag.store_memory_with_provenance
    calls = {"n": 0}

    def flaky_store(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("Qdrant unavailable")
        return real_store(**kwargs)

    monkeypatch.setattr(rag, "store_memory_with_provenance", flaky_store)
    with caplog.at_level(logging.INFO, logger="plugins.memory.jarvis"):
        provider.on_session_end(_finalizable_transcript(messages))
    failure = [r for r in caplog.records if "session-end learning failed" in r.getMessage()]
    assert failure and failure[0].levelno == logging.WARNING
    assert "after 1 of 2 candidate(s)" in failure[0].getMessage() and SID in failure[0].getMessage()
    assert [p.payload["text"] for p in _points(rag)] == [KEEP]

    monkeypatch.setattr(rag, "store_memory_with_provenance", real_store)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="plugins.memory.jarvis"):
        provider.on_session_end(_finalizable_transcript(messages))
    summary = [r for r in caplog.records if "session-end learning for session" in r.getMessage()]
    assert summary and summary[0].levelno == logging.INFO
    assert "2 candidate(s), 2 stored or already present" in summary[0].getMessage()
    assert sorted(p.payload["text"] for p in _points(rag)) == sorted([KEEP, second])
