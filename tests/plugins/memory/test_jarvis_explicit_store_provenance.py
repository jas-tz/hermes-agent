"""Provenance handling for the explicit ``hermes_rag_store`` tool.

Regression: a model call with provenance.source="user_input" was stored with
source "unknown" / provenance_status "unavailable" and reported as success.
Unsupported sources are now refused (nothing stored), defaults and other
normalizations are reported in the result, and the current session is recorded.

Storage is either mocked or a disposable tmp_path Qdrant client; JARVIS_QDRANT_*
and JARVIS_EMBED_* are removed so production Qdrant and the live embedding
service are never used.
"""

import json
import os
import sys
import uuid
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.expanduser("~/.hermes/rag"))

PRODUCTION_ENV = ("JARVIS_QDRANT_URL", "JARVIS_QDRANT_PATH", "JARVIS_QDRANT_TIMEOUT",
                  "JARVIS_EMBED_SOCKET", "JARVIS_EMBED_TIMEOUT", "JARVIS_EMBED_FALLBACK")
SID = "desktop_1791700000_abcdef01"


@pytest.fixture(autouse=True)
def _no_production_env(monkeypatch):
    for var in PRODUCTION_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def provider():
    from plugins.memory.jarvis import HermesRAGProvider
    return HermesRAGProvider()


@pytest.fixture
def session_provider(tmp_path):
    from plugins.memory.jarvis import HermesRAGProvider
    p = HermesRAGProvider()
    p.initialize(SID, hermes_home=str(tmp_path / "hermes_home"), agent_context="primary")
    return p


def _store(p, args):
    with patch("rag_memory.store_memory_with_provenance", return_value="id-1") as mock_store:
        result = json.loads(p.handle_tool_call("hermes_rag_store", args))
    return result, mock_store


# ---------------------------------------------------------------------------
# Validation (storage mocked)
# ---------------------------------------------------------------------------


class TestSourceValidation:
    @pytest.mark.parametrize("source", ["conversation", "tool_result", "user_confirmed", "inferred", "unknown"])
    def test_supported_source_is_stored_unchanged(self, provider, source):
        result, mock_store = _store(provider, {"text": "t", "provenance": {"source": source, "context_id": "c1"}})
        assert result == {"ok": True, "memory_id": "id-1"}
        mock_store.assert_called_once_with("t", metadata=None, provenance={"source": source, "context_id": "c1"})

    def test_user_input_is_rejected_not_mapped(self, provider):
        result, mock_store = _store(provider, {"text": "t", "provenance": {"source": "user_input", "context": "x"}})
        assert result["ok"] is False
        assert result["code"] == "invalid_provenance_source"
        assert "'user_input'" in result["error"] and "Nothing was stored" in result["error"]
        assert "conversation" in result["accepted_sources"]
        mock_store.assert_not_called()

    @pytest.mark.parametrize("source", ["User_Confirmed", "", "  conversation", 7, ["web"]])
    def test_other_unsupported_sources_are_rejected(self, provider, source):
        result, mock_store = _store(provider, {"text": "t", "provenance": {"source": source}})
        assert result["ok"] is False and result["code"] == "invalid_provenance_source"
        mock_store.assert_not_called()

    def test_non_object_provenance_is_rejected(self, provider):
        result, mock_store = _store(provider, {"text": "t", "provenance": "user"})
        assert result["ok"] is False and result["code"] == "invalid_provenance"
        mock_store.assert_not_called()

    def test_missing_provenance_defaults_and_is_reported(self, provider):
        result, mock_store = _store(provider, {"text": "t"})
        assert result["ok"] is True
        assert result["provenance_notes"] == ["No provenance supplied; recorded as source 'model_proposed'."]
        mock_store.assert_called_once_with("t", metadata=None, provenance={"source": "model_proposed"})

    def test_missing_source_field_defaults_and_is_reported(self, provider):
        result, mock_store = _store(provider, {"text": "t", "provenance": {"context_id": "c1"}})
        assert result["ok"] is True
        assert any("source missing" in n for n in result["provenance_notes"])
        assert mock_store.call_args.kwargs["provenance"] == {"context_id": "c1", "source": "model_proposed"}

    def test_caller_args_are_not_mutated(self, session_provider):
        args = {"text": "t", "provenance": {"source": "conversation"}}
        snapshot = json.loads(json.dumps(args))
        _store(session_provider, args)
        assert args == snapshot

    def test_empty_text_still_rejected_first(self, provider):
        result, mock_store = _store(provider, {"text": "", "provenance": {"source": "user_input"}})
        assert result == {"ok": False, "error": "text is required"}
        mock_store.assert_not_called()


class TestSessionTraceability:
    def test_current_session_is_attached(self, session_provider):
        result, mock_store = _store(session_provider, {"text": "t", "provenance": {"source": "conversation"}})
        provenance = mock_store.call_args.kwargs["provenance"]
        assert provenance == {"source": "conversation", "session_id": SID, "source_id": SID}
        assert any("source_id set to the current session" in n for n in result["provenance_notes"])

    def test_existing_source_id_is_kept(self, session_provider):
        _, mock_store = _store(session_provider, {"text": "t", "provenance": {"source": "web", "source_id": "https://example.org"}})
        provenance = mock_store.call_args.kwargs["provenance"]
        assert provenance["source_id"] == "https://example.org" and provenance["session_id"] == SID

    def test_conflicting_session_id_is_replaced_and_reported(self, session_provider):
        result, mock_store = _store(session_provider, {"text": "t", "provenance": {"source": "conversation", "session_id": "other"}})
        assert mock_store.call_args.kwargs["provenance"]["session_id"] == SID
        assert any("'other' replaced with the current session" in n for n in result["provenance_notes"])

    def test_missing_provenance_with_session(self, session_provider):
        _, mock_store = _store(session_provider, {"text": "t"})
        assert mock_store.call_args.kwargs["provenance"] == {"source": "model_proposed", "session_id": SID, "source_id": SID}

    def test_no_session_means_no_session_fields(self, provider):
        _, mock_store = _store(provider, {"text": "t", "provenance": {"source": "conversation"}})
        assert "session_id" not in mock_store.call_args.kwargs["provenance"]


class TestOtherEntryPoints:
    def test_module_dispatcher_also_rejects(self):
        from plugins.memory.jarvis import _dispatch_jarvis_tool_call
        with patch("rag_memory.store_memory_with_provenance") as mock_store:
            result = json.loads(_dispatch_jarvis_tool_call("hermes_rag_store", {"text": "t", "provenance": {"source": "user_input"}}))
        assert result["ok"] is False and result["code"] == "invalid_provenance_source"
        mock_store.assert_not_called()

    def test_subagent_writes_still_forced_to_model_proposed(self):
        from plugins.memory.jarvis import JARVISSubagentProvider
        sub = JARVISSubagentProvider(goal="g", depth=1) if "goal" in JARVISSubagentProvider.__init__.__code__.co_varnames else None
        if sub is None:
            pytest.skip("subagent constructor signature differs")
        with patch("rag_memory.store_memory_with_provenance", return_value="id-2") as mock_store:
            result = json.loads(sub.handle_tool_call("hermes_rag_store", {"text": "t", "provenance": {"source": "user_input"}}))
        assert result["ok"] is True
        assert mock_store.call_args.kwargs["provenance"]["source"] == "model_proposed"

    def test_schema_lists_exactly_the_accepted_sources(self, provider):
        import rag_memory
        schema = next(s for s in provider.get_tool_schemas() if s["name"] == "hermes_rag_store")
        prov = schema["parameters"]["properties"]["provenance"]
        assert set(prov["properties"]["source"]["enum"]) == set(rag_memory.PROVENANCE_SOURCES)
        for source in rag_memory.PROVENANCE_SOURCES:
            assert source in prov["description"]
        assert "rejected" in prov["description"]


# ---------------------------------------------------------------------------
# Real storage on a disposable Qdrant store
# ---------------------------------------------------------------------------


@pytest.fixture
def rag(monkeypatch, tmp_path):
    import rag_memory as rm
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection(collection_name=rm.COLLECTION_NAME,
                             vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    monkeypatch.setattr(rm, "QDRANT_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(rm, "client", client)
    yield rm
    client.close()


def _points(rm):
    points, _ = rm.client.scroll(collection_name=rm.COLLECTION_NAME, limit=50, with_payload=True)
    return points


class TestRealStorage:
    def test_rejected_provenance_writes_nothing(self, rag, session_provider):
        for prov in ({"source": "user_input"}, {"source": 3}, "x"):
            result = json.loads(session_provider.handle_tool_call("hermes_rag_store", {"text": "The user prefers tea.", "provenance": prov}))
            assert result["ok"] is False
        assert _points(rag) == []

    def test_accepted_store_payload_is_compatible(self, rag, session_provider):
        result = json.loads(session_provider.handle_tool_call(
            "hermes_rag_store",
            {"text": "The user prefers tea.", "metadata": {"topic": "drinks"}, "provenance": {"source": "conversation"}},
        ))
        assert result["ok"] is True
        [point] = _points(rag)
        assert point.id == result["memory_id"]
        assert uuid.UUID(point.id).version == 4  # explicit stores keep random IDs
        payload = point.payload
        assert payload["text"] == "The user prefers tea." and payload["type"] == "memory"
        assert payload["topic"] == "drinks"
        prov = payload["provenance"]
        assert prov["source"] == "conversation" and prov["provenance_status"] == "tracked"
        assert prov["session_id"] == SID and prov["source_id"] == SID and "stored_at" in prov
