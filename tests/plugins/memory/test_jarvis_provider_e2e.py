"""End-to-end tests for HermesRAGProvider's cross-session learning chain.

Unlike ``test_jarvis_provider.py`` (fast, ``unittest.mock``-based unit tests
that never touch Qdrant), these tests exercise the REAL ``rag_memory.py``
functions — real embeddings, real Qdrant reads/writes, real eligibility and
redaction logic — through the real ``HermesRAGProvider`` calls
(``on_session_end`` -> ``store_session_learnings`` -> ``retrieve_cross_session_
learnings``). They prove the actual persistence/retrieval behavior, not just
that the right function names get called.

Isolation: the ``isolated_rag_qdrant`` fixture below replicates the same
technique already proven safe by ~/.hermes/rag/tests/conftest.py::
_isolate_qdrant — monkeypatch the module-level ``rag_memory.QDRANT_PATH`` /
``rag_memory.client`` globals to a fresh ``tmp_path`` directory + collection.
Every function in rag_memory.py reads ``client`` from the module namespace at
call time (not a captured local), so this redirects ALL Qdrant I/O —
including calls made transitively through HermesRAGProvider — without
touching any production code. ``monkeypatch`` auto-restores both attributes
after each test. The module-scoped ``_isolated_rag_memory_import`` fixture
below performs the first import of rag_memory with the QdrantClient
constructor redirected to a temp directory (the same technique as
``test_jarvis_provider.py``), so the real ~/.hermes/rag/qdrant_data store is
never opened by importing rag_memory either.

The ~/.hermes/rag/tests/conftest.py fixture itself is intentionally NOT
imported here: that directory is not a git repository and is outside this
repo's change control, so hermes-agent's test suite must not depend on it.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.expanduser("~/.hermes/rag"))


@pytest.fixture(scope="module", autouse=True)
def _isolated_rag_memory_import(tmp_path_factory):
    """Import rag_memory without ever opening the production Qdrant store.

    rag_memory builds ``client = QdrantClient(path=QDRANT_PATH)`` at import
    time, so the constructor is redirected to a temp directory for the
    moment the module body runs. If another test file already imported it
    (test_jarvis_provider.py does so the same way), it is reused as-is.

    JARVIS_QDRANT_* settings are cleared for this module so a shell
    configured for the real Qdrant Server can never route these tests to it;
    JARVIS_EMBED_* likewise, so they never use the live embedding service.
    """
    qdrant_env_vars = ("JARVIS_QDRANT_URL", "JARVIS_QDRANT_PATH", "JARVIS_QDRANT_TIMEOUT",
                       "JARVIS_EMBED_SOCKET", "JARVIS_EMBED_TIMEOUT", "JARVIS_EMBED_FALLBACK")
    saved_env = {v: os.environ.pop(v) for v in qdrant_env_vars if v in os.environ}
    if "rag_memory" not in sys.modules:
        from unittest.mock import patch

        import qdrant_client as _qc

        tmp_dir = str(tmp_path_factory.mktemp("jarvis_e2e_import_qdrant"))
        real_qdrant_client_cls = _qc.QdrantClient

        def _temp_only_client(*_args, **_kwargs):
            return real_qdrant_client_cls(path=tmp_dir)

        with patch.object(_qc, "QdrantClient", _temp_only_client):
            import rag_memory as rm

        rm.QDRANT_PATH = tmp_dir
    yield
    os.environ.update(saved_env)


@pytest.fixture
def isolated_rag_qdrant(monkeypatch, tmp_path):
    """Redirect rag_memory's Qdrant client to a disposable temp store.

    Yields the real ``rag_memory`` module (with ``QDRANT_PATH``/``client``
    monkeypatched) so tests can also assert against it directly if needed.
    """
    import rag_memory as rm
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance

    tmp_qdrant_dir = tmp_path / "qdrant_data"
    test_client = QdrantClient(path=str(tmp_qdrant_dir))
    test_client.create_collection(
        collection_name=rm.COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    monkeypatch.setattr(rm, "QDRANT_PATH", tmp_qdrant_dir)
    monkeypatch.setattr(rm, "client", test_client)

    yield rm


@pytest.fixture
def provider():
    from plugins.memory.jarvis import HermesRAGProvider
    return HermesRAGProvider()


class TestCrossSessionLearningE2E:
    """session A -> on_session_end() -> real Qdrant persistence -> session B -> real retrieval."""

    def test_eligible_assistant_learning_persists_and_is_recovered_in_a_later_session(
        self, isolated_rag_qdrant, provider
    ):
        eligible_text = (
            "Jarvis stores long-term memories in a local Qdrant collection called jarvis_memory."
        )
        session_a_messages = [
            {"role": "user", "content": "Where does Jarvis store long-term memory?"},
            {"role": "assistant", "content": eligible_text},
        ]

        # --- Session A: real on_session_end -> real store_session_learnings -> real Qdrant write.
        provider.initialize(
            "session-a", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary"
        )
        provider.on_session_end(session_a_messages)

        # --- Session B: a fresh provider instance/session id, real retrieval.
        from plugins.memory.jarvis import HermesRAGProvider

        provider_b = HermesRAGProvider()
        provider_b.initialize(
            "session-b", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary"
        )

        raw = provider_b.handle_tool_call(
            "hermes_rag_retrieve_learnings",
            {"query": "Where does Jarvis keep its memory?", "limit": 5},
        )
        result = json.loads(raw)

        assert result["ok"] is True
        assert result["count"] >= 1
        texts = [r["text"] for r in result["results"]]
        assert any(eligible_text == t for t in texts)

        matching = next(r for r in result["results"] if r["text"] == eligible_text)
        provenance = matching["metadata"]["provenance"]
        assert provenance["source_type"] == "session_summary"
        assert provenance["source_id"] == "session-a"

        # Cross-check directly against the real rag_memory function too (not
        # just the provider's tool wrapper), confirming the same persisted data.
        direct = isolated_rag_qdrant.retrieve_cross_session_learnings(
            "Where does Jarvis keep its memory?", limit=5
        )
        assert any(r["text"] == eligible_text for r in direct)

    def test_secret_indicator_content_is_never_persisted_as_a_learning(
        self, isolated_rag_qdrant, provider
    ):
        # Contains the "sk-" secret indicator that _is_eligible_for_transfer()
        # explicitly filters out — real eligibility enforcement, not a mock.
        secret_text = "sk-1234567890abcdef1234567890abcdef"
        messages = [
            {"role": "user", "content": "What's the API key?"},
            {"role": "assistant", "content": secret_text},
        ]

        provider.initialize(
            "session-secret", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary"
        )
        provider.on_session_end(messages)

        raw = provider.handle_tool_call(
            "hermes_rag_retrieve_learnings", {"query": "API key", "limit": 5}
        )
        result = json.loads(raw)
        assert result["ok"] is True
        assert result["count"] == 0

        # The temp collection must contain nothing at all — not just nothing
        # matching the learnings filter — proving the secret was rejected at
        # eligibility time and never reached storage in the first place.
        direct = isolated_rag_qdrant.retrieve_cross_session_learnings("API key", limit=5)
        assert direct == []

    def test_trivially_short_content_is_never_persisted_as_a_learning(
        self, isolated_rag_qdrant, provider
    ):
        # Below the 10-character MIN eligibility threshold in
        # _is_eligible_for_transfer() — real length-based rejection.
        messages = [
            {"role": "user", "content": "ok?"},
            {"role": "assistant", "content": "yes"},
        ]

        provider.initialize(
            "session-short", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary"
        )
        provider.on_session_end(messages)

        raw = provider.handle_tool_call(
            "hermes_rag_retrieve_learnings", {"query": "yes", "limit": 5}
        )
        result = json.loads(raw)
        assert result["ok"] is True
        assert result["count"] == 0

    def test_realistic_reply_persists_only_the_durable_statement(
        self, isolated_rag_qdrant, provider
    ):
        # A typical multi-line assistant reply: headings, bullets, code,
        # filler, and narration around a single durable fact. Before the
        # eligibility allow step, every line here except the ``` fences was
        # persisted as a cross-session learning.
        durable = "The user wants to build a world-class honey processing company in Tanzania."
        reply = "\n".join([
            "## Summary of changes",
            "Here's what I found in the repository:",
            "- `plugins/memory/jarvis/__init__.py` has 2061 lines",
            "```python",
            "def on_session_end(self, messages):",
            "```",
            "Sure! Let me know if you want me to proceed with the refactor.",
            "The weather in Dar es Salaam today is 31°C and sunny.",
            durable,
            "I've updated the file and ran the linter; no errors.",
        ])
        messages = [
            {"role": "user", "content": "Summarize what you changed and remember my business goal."},
            {"role": "assistant", "content": reply},
        ]

        provider.initialize(
            "session-realistic", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary"
        )
        provider.on_session_end(messages)

        points, _ = isolated_rag_qdrant.client.scroll(
            collection_name=isolated_rag_qdrant.COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        assert [p.payload["text"] for p in points] == [durable]

        provenance = points[0].payload["provenance"]
        assert provenance["source"] == "inferred"
        assert provenance["provenance_status"] == "tracked"
        assert provenance["source_type"] == "session_summary"
        assert provenance["source_id"] == "session-realistic"

        # Still reachable through the normal cross-session retrieval path.
        raw = provider.handle_tool_call(
            "hermes_rag_retrieve_learnings", {"query": "honey business in Tanzania", "limit": 5}
        )
        result = json.loads(raw)
        assert result["ok"] is True
        assert [r["text"] for r in result["results"]] == [durable]


class TestPhase10AProvenanceSourcesRagMemory:
    """Real rag_memory._sanitize_provenance() exercise for the Phase 10A
    whitelist extension (user_confirmed, model_proposed, imported).

    ~/.hermes/rag/tests/test_rag_memory.py (a separate, unversioned venv)
    cannot import tools.secret_redaction at all — confirmed pre-existing
    and unrelated to this change, so its own Phase 10A test there is
    limited to a pure whitelist-membership check. This class exercises the
    real sanitizer end-to-end (redaction, size cap, fallback) using the
    same isolated temp Qdrant this file already relies on, in an
    environment where tools.secret_redaction resolves correctly.
    """

    @pytest.mark.parametrize("new_source", ["user_confirmed", "model_proposed", "imported"])
    def test_new_source_accepted_by_sanitizer(self, isolated_rag_qdrant, new_source):
        isolated_rag_qdrant.store_memory_with_provenance(
            f"Phase 10A provenance test {new_source}", provenance={"source": new_source}
        )
        results = isolated_rag_qdrant.search_memory(f"Phase 10A provenance test {new_source}", limit=1)
        assert results
        assert results[0]["metadata"]["provenance"]["source"] == new_source
        assert results[0]["metadata"]["provenance"]["provenance_status"] == "tracked"

    def test_invalid_source_still_falls_back_to_unknown(self, isolated_rag_qdrant):
        isolated_rag_qdrant.store_memory_with_provenance(
            "Invalid source fallback test", provenance={"source": "not-a-real-source"}
        )
        results = isolated_rag_qdrant.search_memory("Invalid source fallback test", limit=1)
        assert results[0]["metadata"]["provenance"]["source"] == "unknown"
        assert results[0]["metadata"]["provenance"]["provenance_status"] == "unavailable"

    def test_redaction_unchanged_for_new_sources(self, isolated_rag_qdrant):
        secret = "sk-abcdef1234567890abcdef1234567890"
        isolated_rag_qdrant.store_memory_with_provenance(
            "Redaction test for new sources",
            provenance={"source": "user_confirmed", "note": f"api_key={secret}"},
        )
        results = isolated_rag_qdrant.search_memory("Redaction test for new sources", limit=1)
        assert secret not in json.dumps(results[0]["metadata"])

    def test_size_limit_unchanged_for_new_sources(self, isolated_rag_qdrant):
        huge = "z" * 5000
        isolated_rag_qdrant.store_memory_with_provenance(
            "Oversized provenance test for new sources",
            provenance={"source": "model_proposed", "blob": huge},
        )
        results = isolated_rag_qdrant.search_memory("Oversized provenance test for new sources", limit=1)
        assert results[0]["metadata"]["provenance"]["provenance_status"] == "truncated"
        assert "blob" not in results[0]["metadata"]["provenance"]

    def test_expected_sources_present_in_whitelist(self, isolated_rag_qdrant):
        expected = {
            "conversation", "tool_result", "web", "manual", "inferred", "unknown",
            "user_confirmed", "model_proposed", "imported",
        }
        assert expected <= isolated_rag_qdrant.PROVENANCE_SOURCES
