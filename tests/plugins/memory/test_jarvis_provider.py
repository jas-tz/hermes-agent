"""Tests for the HermesRAGProvider memory plugin."""
import json
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# Ensure the hermes-agent root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Ensure the rag module is on sys.path (for rag_memory)
rag_path = os.path.expanduser("~/.hermes/rag")
sys.path.insert(0, rag_path)

_PRODUCTION_RAG_DIR = os.path.expanduser("~/.hermes/rag")
_PRODUCTION_QDRANT_PATH = os.path.join(_PRODUCTION_RAG_DIR, "qdrant_data")
_QDRANT_ENV_VARS = ("JARVIS_QDRANT_URL", "JARVIS_QDRANT_PATH", "JARVIS_QDRANT_TIMEOUT")


@pytest.fixture(scope="module", autouse=True)
def _isolated_rag_memory_module(tmp_path_factory):
    """Ensure this file NEVER opens ~/.hermes/rag/qdrant_data.

    rag_memory.py builds ``client = QdrantClient(path=str(QDRANT_PATH))`` at
    MODULE IMPORT TIME, with QDRANT_PATH derived from the module's own file
    location — there is no rag_memory attribute to monkeypatch beforehand,
    because the module doesn't exist in sys.modules until that line has
    already run. Simply resolving a ``patch("rag_memory.search_memory")``
    target (used throughout this file) is enough to trigger that first
    import — so every test here was opening, and exclusively locking, the
    PRODUCTION Qdrant store before this fixture existed, even though every
    actual call into rag_memory is mocked afterward (see the Phase 5
    validation report, item 6).

    Fix: redirect the ``QdrantClient`` constructor to a disposable temp
    directory for the single moment rag_memory.py's module body executes,
    then restore it immediately. What individual tests mock is unchanged —
    this only changes what ``rag_memory.client`` ends up pointing at.

    rag_memory now creates its client lazily from JARVIS_QDRANT_* settings,
    so those are cleared for this module: a shell configured for the real
    Qdrant Server must never route these tests to it.
    """
    saved_env = {v: os.environ.pop(v) for v in _QDRANT_ENV_VARS if v in os.environ}
    tmp_dir = str(tmp_path_factory.mktemp("jarvis_unit_test_qdrant"))

    if "rag_memory" not in sys.modules:
        import qdrant_client as _qc

        real_qdrant_client_cls = _qc.QdrantClient

        def _temp_only_client(*_args, **_kwargs):
            return real_qdrant_client_cls(path=tmp_dir)

        with patch.object(_qc, "QdrantClient", _temp_only_client):
            import rag_memory as rm  # module-level `client = QdrantClient(...)` now targets tmp_dir

        # QDRANT_PATH itself is a separate module constant computed BEFORE
        # `client = QdrantClient(...)` runs — redirecting the constructor
        # above already stops rag_memory.client from ever touching
        # production (the part that actually matters for I/O), but
        # QDRANT_PATH's string value is untouched by that and would still
        # read as the production path for anything introspecting it
        # (e.g. hermes_cli/doctor.py). Keep it consistent with reality.
        rm.QDRANT_PATH = tmp_dir
    else:
        # rag_memory was already imported by something else in this process
        # before this fixture ran (unexpected in normal collection order for
        # this file alone) — redirect it post-hoc. Note: the production
        # directory may already have been briefly opened in this fallback.
        import rag_memory as rm
        from qdrant_client import QdrantClient as _RealQdrantClient

        rm.QDRANT_PATH = tmp_dir
        rm.client = _RealQdrantClient(path=tmp_dir)

    from qdrant_client.models import VectorParams, Distance

    rm.client.create_collection(
        collection_name=rm.COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    yield rm

    os.environ.update(saved_env)


class TestHermesRAGProviderTestIsolation:
    """Verify (Phase 6, item 10) this file's rag_memory never targets production."""

    def test_rag_memory_points_at_a_temp_directory_not_production(self, _isolated_rag_memory_module):
        qdrant_path = str(_isolated_rag_memory_module.QDRANT_PATH)
        assert qdrant_path != _PRODUCTION_QDRANT_PATH
        assert not qdrant_path.startswith(_PRODUCTION_RAG_DIR + os.sep)


class TestHermesRAGProviderInterface:
    """Verify the provider implements the MemoryProvider ABC."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    @pytest.fixture
    def abc(self):
        from agent.memory_provider import MemoryProvider
        return MemoryProvider

    def test_inherits_from_memory_provider(self, provider, abc):
        assert isinstance(provider, abc)

    def test_name_is_jarvis(self, provider):
        assert provider.name == "jarvis"

    def test_implements_is_available(self, provider):
        assert callable(provider.is_available)

    def test_implements_initialize(self, provider):
        assert callable(provider.initialize)

    def test_implements_get_tool_schemas(self, provider):
        assert callable(provider.get_tool_schemas)

    def test_implements_handle_tool_call(self, provider):
        assert callable(provider.handle_tool_call)

    def test_implements_prefetch(self, provider):
        assert callable(provider.prefetch)

    def test_implements_sync_turn(self, provider):
        assert callable(provider.sync_turn)

    def test_implements_shutdown(self, provider):
        assert callable(provider.shutdown)

    def test_implements_get_config_schema(self, provider):
        assert callable(provider.get_config_schema)

    def test_implements_save_config(self, provider):
        assert callable(provider.save_config)


class TestHermesRAGProviderInitializeSysPath:
    """Verify initialize() makes the rag module importable (see audit item 3.1).

    The Hermes entry point runs as a script file, so sys.path[0] is fixed to
    that script's directory and never tracks os.getcwd() — a chdir() alone
    does not make ``import rag_memory`` resolve. initialize() must instead
    add the rag directory to sys.path directly. Uses a temp hermes_home
    (not the real ~/.hermes/rag, which this test file's own module-level
    sys.path.insert already exposes) so the assertion proves initialize()
    did the work, not the test module's own setup.
    """

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def test_initialize_adds_rag_dir_to_sys_path(self, provider, tmp_path):
        rag_dir = tmp_path / "rag"
        rag_dir.mkdir()
        rag_dir_str = str(rag_dir)
        assert rag_dir_str not in sys.path

        try:
            provider.initialize("session-1", hermes_home=str(tmp_path))
            assert rag_dir_str in sys.path
        finally:
            if rag_dir_str in sys.path:
                sys.path.remove(rag_dir_str)

    def test_initialize_is_idempotent_on_sys_path(self, provider, tmp_path):
        rag_dir = tmp_path / "rag"
        rag_dir.mkdir()
        rag_dir_str = str(rag_dir)

        try:
            provider.initialize("session-1", hermes_home=str(tmp_path))
            provider.initialize("session-1", hermes_home=str(tmp_path))
            assert sys.path.count(rag_dir_str) == 1
        finally:
            while rag_dir_str in sys.path:
                sys.path.remove(rag_dir_str)

    def test_initialize_does_not_chdir(self, provider, tmp_path):
        rag_dir = tmp_path / "rag"
        rag_dir.mkdir()
        cwd_before = os.getcwd()

        try:
            provider.initialize("session-1", hermes_home=str(tmp_path))
            assert os.getcwd() == cwd_before
        finally:
            rag_dir_str = str(rag_dir)
            if rag_dir_str in sys.path:
                sys.path.remove(rag_dir_str)


class TestHermesRAGProviderOnSessionEnd:
    """Verify automatic cross-session learning at real session boundaries.

    Since Phase 10D, on_session_end() calls rag_memory's
    extract_session_learnings() + store_memory_with_provenance() directly
    (the exact two calls store_session_learnings() itself makes internally)
    rather than going through that convenience wrapper — this lets a
    dedup check sit between "extracted" and "stored" per candidate (see
    TestHermesRAGProviderSessionEndDeduplication). Eligibility filtering,
    redaction, and the per-session cap are all still rag_memory.py's own,
    unmodified. Must only fire for interactive ("primary") agent contexts,
    never for cron/flush/subagent/oneshot.
    """

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    SAMPLE_MESSAGES = [
        {"role": "user", "content": "What database does Jarvis use?"},
        {"role": "assistant", "content": "Jarvis uses Qdrant in local mode for semantic memory."},
    ]

    SAMPLE_LEARNING = {
        "text": "Jarvis uses Qdrant in local mode for semantic memory.",
        "metadata": {"provenance": {"source_type": "session_summary"}},
    }

    def test_primary_context_reaches_store_memory_with_provenance(self, provider):
        provider.initialize("session-primary", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary")
        with patch("rag_memory.extract_session_learnings", return_value=[self.SAMPLE_LEARNING]) as mock_extract, \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.SAMPLE_MESSAGES)
            mock_extract.assert_called_once_with(self.SAMPLE_MESSAGES, "session-primary")
            mock_store.assert_called_once_with(
                text=self.SAMPLE_LEARNING["text"],
                metadata=self.SAMPLE_LEARNING["metadata"],
                provenance=self.SAMPLE_LEARNING["metadata"]["provenance"],
            )

    def test_default_agent_context_reaches_store_memory_with_provenance(self, provider):
        # No agent_context kwarg at all (matches every current real call site,
        # which always passes "primary" — default must still behave as primary).
        provider.initialize("session-default", hermes_home=os.path.expanduser("~/.hermes"))
        with patch("rag_memory.extract_session_learnings", return_value=[self.SAMPLE_LEARNING]), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.SAMPLE_MESSAGES)
            mock_store.assert_called_once()

    @pytest.mark.parametrize("blocked_context", ["cron", "flush", "subagent", "oneshot"])
    def test_non_interactive_context_skips_learning(self, provider, blocked_context):
        provider.initialize(
            "session-blocked",
            hermes_home=os.path.expanduser("~/.hermes"),
            agent_context=blocked_context,
        )
        with patch("rag_memory.extract_session_learnings") as mock_extract:
            provider.on_session_end(self.SAMPLE_MESSAGES)
            mock_extract.assert_not_called()

    def test_empty_messages_handled_safely(self, provider):
        provider.initialize("session-empty", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary")
        with patch("rag_memory.extract_session_learnings") as mock_extract:
            provider.on_session_end([])
            mock_extract.assert_not_called()

    def test_none_messages_handled_safely(self, provider):
        provider.initialize("session-none", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary")
        with patch("rag_memory.extract_session_learnings") as mock_extract:
            provider.on_session_end(None)
            mock_extract.assert_not_called()

    def test_uninitialized_provider_handled_safely(self, provider):
        # No initialize() call at all — _session_id is None.
        with patch("rag_memory.extract_session_learnings") as mock_extract:
            provider.on_session_end(self.SAMPLE_MESSAGES)
            mock_extract.assert_not_called()

    def test_extract_session_learnings_exception_does_not_raise(self, provider):
        provider.initialize("session-err", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary")
        with patch("rag_memory.extract_session_learnings", side_effect=RuntimeError("Qdrant down")):
            provider.on_session_end(self.SAMPLE_MESSAGES)  # must not raise

    def test_store_memory_with_provenance_exception_does_not_raise(self, provider):
        provider.initialize("session-err2", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary")
        with patch("rag_memory.extract_session_learnings", return_value=[self.SAMPLE_LEARNING]), \
             patch("rag_memory.store_memory_with_provenance", side_effect=RuntimeError("Qdrant down")):
            provider.on_session_end(self.SAMPLE_MESSAGES)  # must not raise

    def test_no_candidates_calls_store_zero_times(self, provider):
        provider.initialize("session-nocand", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary")
        with patch("rag_memory.extract_session_learnings", return_value=[]), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.SAMPLE_MESSAGES)
            mock_store.assert_not_called()


class TestHermesRAGProviderSessionEndDeduplication:
    """Phase 10D: automatic RAG session learning must not redundantly store
    text that substantially overlaps a continuity Goal/Task/Decision this
    provider already wrote successfully during the same session.

    Uses hermes_home=str(tmp_path) (not ~/.hermes) because these tests
    exercise real continuity tool writes to populate the in-session
    deduplication state — production ~/.hermes/continuity.db must never be
    touched. rag_memory's extract_session_learnings/store_memory_with_
    provenance are mocked, consistent with the rest of this file's style
    and this class's focus (the dedup DECISION, not rag_memory's own
    eligibility logic, which the E2E suite already covers unmocked).
    """

    @pytest.fixture
    def provider(self, tmp_path):
        from plugins.memory.jarvis import HermesRAGProvider
        p = HermesRAGProvider()
        p.initialize("session-dedup", hermes_home=str(tmp_path), agent_context="primary")
        return p

    def _call(self, provider, tool_name, args):
        return json.loads(provider.handle_tool_call(tool_name, args))

    MESSAGES = [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "placeholder"}]

    def test_continuity_record_suppresses_equivalent_rag_learning(self, provider):
        self._call(provider, "hermes_goal_create", {"title": "Build the Jarvis continuity system"})
        candidate = {
            "text": "I've created a goal to build the Jarvis continuity system.",
            "metadata": {"provenance": {"source_type": "session_summary"}},
        }
        with patch("rag_memory.extract_session_learnings", return_value=[candidate]), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.MESSAGES)
        mock_store.assert_not_called()

    def test_unrelated_rag_learning_is_still_stored(self, provider):
        self._call(provider, "hermes_goal_create", {"title": "Build the Jarvis continuity system"})
        candidate = {
            "text": "The user's favorite tea is Earl Grey with a splash of milk.",
            "metadata": {"provenance": {"source_type": "session_summary"}},
        }
        with patch("rag_memory.extract_session_learnings", return_value=[candidate]), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.MESSAGES)
        mock_store.assert_called_once_with(
            text=candidate["text"],
            metadata=candidate["metadata"],
            provenance=candidate["metadata"]["provenance"],
        )

    def test_single_common_word_overlap_does_not_suppress(self, provider):
        """Sharing one generic word ("business") must never be enough."""
        self._call(provider, "hermes_goal_create", {"title": "Expand the honey business this year"})
        candidate = {
            "text": "The quarterly business review meeting is scheduled for Friday afternoon.",
            "metadata": {"provenance": {"source_type": "session_summary"}},
        }
        with patch("rag_memory.extract_session_learnings", return_value=[candidate]), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.MESSAGES)
        mock_store.assert_called_once()

    def test_short_generic_continuity_text_does_not_suppress_unrelated_learning(self, provider):
        self._call(provider, "hermes_decision_record", {"decision": "Use the plan"})
        candidate = {
            "text": "The client prefers weekly status updates delivered on Mondays.",
            "metadata": {"provenance": {"source_type": "session_summary"}},
        }
        with patch("rag_memory.extract_session_learnings", return_value=[candidate]), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.MESSAGES)
        mock_store.assert_called_once()

    def test_failed_continuity_write_does_not_affect_deduplication(self, provider):
        # Missing required "title" -> create_goal raises inside continuity_db,
        # handle_tool_call catches it and returns an error; nothing should
        # have been tracked.
        result = self._call(provider, "hermes_goal_create", {})
        assert result["ok"] is False

        candidate = {
            "text": "I've created a goal to build the Jarvis continuity system.",
            "metadata": {"provenance": {"source_type": "session_summary"}},
        }
        with patch("rag_memory.extract_session_learnings", return_value=[candidate]), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.MESSAGES)
        mock_store.assert_called_once()  # not suppressed — the failed write was never tracked

    def test_multiple_continuity_records_are_all_tracked(self, provider):
        self._call(provider, "hermes_goal_create", {"title": "Build the Jarvis continuity system"})
        self._call(provider, "hermes_task_create", {"title": "Validate structured continuity across sessions"})
        self._call(provider, "hermes_decision_record", {
            "decision": "Keep structured continuity in SQLite and semantic memory in Qdrant",
        })

        candidates = [
            {"text": "I built the Jarvis continuity system as requested.",
             "metadata": {"provenance": {"source_type": "session_summary"}}},
            {"text": "I validated structured continuity across sessions successfully.",
             "metadata": {"provenance": {"source_type": "session_summary"}}},
            {"text": "The team enjoys pizza on Fridays.",
             "metadata": {"provenance": {"source_type": "session_summary"}}},
        ]
        with patch("rag_memory.extract_session_learnings", return_value=candidates), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.MESSAGES)

        # Only the unrelated pizza candidate should have been stored.
        assert mock_store.call_count == 1
        assert mock_store.call_args.kwargs["text"] == "The team enjoys pizza on Fridays."

    def test_deduplication_only_applies_to_current_provider_session(self, tmp_path):
        """A fresh provider instance (new session) starts with no tracked
        fragments, even against the same continuity.db."""
        from plugins.memory.jarvis import HermesRAGProvider

        provider_a = HermesRAGProvider()
        provider_a.initialize("session-a", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider_a, "hermes_goal_create", {"title": "Build the Jarvis continuity system"})

        provider_b = HermesRAGProvider()
        provider_b.initialize("session-b", hermes_home=str(tmp_path), agent_context="primary")

        candidate = {
            "text": "I've created a goal to build the Jarvis continuity system.",
            "metadata": {"provenance": {"source_type": "session_summary"}},
        }
        with patch("rag_memory.extract_session_learnings", return_value=[candidate]), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider_b.on_session_end(self.MESSAGES)
        # provider_b never wrote the continuity record itself this session,
        # so it has nothing to deduplicate against.
        mock_store.assert_called_once()

    @pytest.mark.parametrize("blocked_context", ["cron", "flush", "subagent", "oneshot"])
    def test_gating_unchanged_even_with_tracked_fragments(self, tmp_path, blocked_context):
        """Existing session-end context gating (Phase 8) must remain
        unaffected by Phase 10D — a blocked context still extracts nothing
        at all, regardless of any tracked continuity fragments."""
        from plugins.memory.jarvis import HermesRAGProvider

        provider = HermesRAGProvider()
        provider.initialize("session-primary-setup", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Build the Jarvis continuity system"})

        # Re-initialize the SAME instance under a blocked context (mirrors
        # on_session_switch-style rebinding without inventing new API).
        provider.initialize(
            "session-primary-setup", hermes_home=str(tmp_path), agent_context=blocked_context
        )
        with patch("rag_memory.extract_session_learnings") as mock_extract:
            provider.on_session_end(self.MESSAGES)
        mock_extract.assert_not_called()

    def test_rag_path_intact_when_no_continuity_fragments_tracked(self, provider):
        """With zero tracked fragments (no continuity tool ever called this
        session), every eligible candidate is stored exactly as before
        Phase 10D."""
        candidate = {
            "text": "Jarvis uses Qdrant in local mode for semantic memory.",
            "metadata": {"provenance": {"source_type": "session_summary"}},
        }
        with patch("rag_memory.extract_session_learnings", return_value=[candidate]), \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_end(self.MESSAGES)
        mock_store.assert_called_once_with(
            text=candidate["text"],
            metadata=candidate["metadata"],
            provenance=candidate["metadata"]["provenance"],
        )


class TestHermesRAGProviderServerRequiredGuard:
    """Phase 2B: once the Qdrant cutover marker exists, a provider in a
    process without JARVIS_QDRANT_URL must report errors (or stay quiet on
    paths that already were), and must never open a local store.

    The marker and "production" store are simulated under tmp_path; the
    real ~/.hermes/rag marker location and qdrant_data are never used.
    """

    @pytest.fixture
    def guarded(self, _isolated_rag_memory_module, monkeypatch, tmp_path):
        import numpy as np

        rm = _isolated_rag_memory_module
        prod = tmp_path / "rag" / "qdrant_data"
        marker = tmp_path / "rag" / "qdrant_server_required"
        marker.parent.mkdir()
        marker.touch()
        monkeypatch.setattr(rm, "PRODUCTION_QDRANT_PATH", prod)
        monkeypatch.setattr(rm, "QDRANT_PATH", prod)
        monkeypatch.setattr(rm, "SERVER_REQUIRED_MARKER", marker)
        # Force the first-use path; monkeypatch restores this module's
        # shared temp client afterwards. A fresh lazy stand-in is installed
        # too: if another test file imported rag_memory first, this file's
        # module fixture rebinds rm.client to a concrete temp client, which
        # would bypass lazy creation (and so the guard) entirely.
        monkeypatch.setattr(rm, "_client_instance", None)
        monkeypatch.setattr(rm, "client", rm._LazyClient())

        constructed = []

        def refuse_construction(**kwargs):
            constructed.append(kwargs)
            raise AssertionError(f"QdrantClient constructed despite the guard: {kwargs}")

        monkeypatch.setattr(rm, "QdrantClient", refuse_construction)

        class FakeModel:
            def encode(self, text, normalize_embeddings=True):
                return np.full(384, 1 / np.sqrt(384), dtype=np.float32)

        monkeypatch.setattr(rm, "_get_model", lambda: FakeModel())

        from plugins.memory.jarvis import HermesRAGProvider

        provider = HermesRAGProvider()
        provider.initialize("session-guard", hermes_home=str(tmp_path), agent_context="primary")
        yield provider, constructed, prod

    @pytest.mark.parametrize(
        "tool_name, args",
        [
            ("hermes_rag_search", {"query": "What database does Jarvis use?"}),
            ("hermes_rag_store", {"text": "The user prefers concise answers."}),
            ("hermes_rag_delete", {"memory_id": "3827934c-dbb8-48dc-9641-b16f42f573ef"}),
            ("hermes_rag_retrieve_learnings", {"query": "pilot batch"}),
            ("hermes_rag_rollback_learnings", {}),
        ],
    )
    def test_rag_tools_report_guard_error(self, guarded, tool_name, args):
        provider, constructed, prod = guarded

        result = json.loads(provider.handle_tool_call(tool_name, args))

        assert result["ok"] is False
        assert "JARVIS_QDRANT_URL is not set" in result["error"]
        assert constructed == []
        assert not prod.exists()

    def test_prefetch_and_session_end_never_open_a_local_store(self, guarded):
        provider, constructed, prod = guarded

        context = provider.prefetch("What database does Jarvis use for long-term memory?")
        provider.on_session_end([
            {"role": "user", "content": "Remember this."},
            {"role": "assistant", "content": "The user prefers concise answers without emoji."},
        ])

        assert "jarvis_rag" not in context
        assert constructed == []
        assert not prod.exists()


class TestHermesRAGProviderSyncTurnNoop:
    """Verify sync_turn() remains an intentional no-op (not per-turn persistence)."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def test_sync_turn_does_not_persist_anything(self, provider):
        provider.initialize("session-turn", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary")
        with patch("rag_memory.add_memory") as mock_add, \
             patch("rag_memory.store_memory_with_provenance") as mock_store_prov, \
             patch("rag_memory.store_session_learnings") as mock_store_learn:
            result = provider.sync_turn(
                "What database does Jarvis use?",
                "Jarvis uses Qdrant in local mode.",
                session_id="session-turn",
                messages=[
                    {"role": "user", "content": "What database does Jarvis use?"},
                    {"role": "assistant", "content": "Jarvis uses Qdrant in local mode."},
                ],
            )
            assert result is None
            mock_add.assert_not_called()
            mock_store_prov.assert_not_called()
            mock_store_learn.assert_not_called()


class TestHermesRAGProviderTools:
    """Verify the provider exposes the expected tool schemas."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def test_has_rag_search_tool(self, provider):
        tools = provider.get_tool_schemas()
        names = [t["name"] for t in tools]
        assert "hermes_rag_search" in names

    def test_has_rag_store_tool(self, provider):
        tools = provider.get_tool_schemas()
        names = [t["name"] for t in tools]
        assert "hermes_rag_store" in names

    def test_has_rag_delete_tool(self, provider):
        tools = provider.get_tool_schemas()
        names = [t["name"] for t in tools]
        assert "hermes_rag_delete" in names

    def test_tool_schemas_have_required_fields(self, provider):
        tools = provider.get_tool_schemas()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool


class TestHermesRAGProviderDelegation:
    """Verify tool calls delegate correctly to rag_memory.py."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def test_store_delegates_to_add_memory(self, provider):
        with patch("rag_memory.add_memory") as mock_add:
            mock_add.return_value = "test-uuid-123"
            result = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": "test memory"}))
            assert result["ok"] is True
            assert result["memory_id"] == "test-uuid-123"
            mock_add.assert_called_once_with("test memory", metadata=None)

    def test_store_with_metadata_delegates(self, provider):
        with patch("rag_memory.add_memory") as mock_add:
            mock_add.return_value = "test-uuid-456"
            result = json.loads(provider.handle_tool_call(
                "hermes_rag_store", {"text": "test", "metadata": {"source": "test"}}
            ))
            assert result["ok"] is True
            mock_add.assert_called_once_with("test", metadata={"source": "test"})

    def test_store_empty_text_returns_error(self, provider):
        result = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": ""}))
        assert result["ok"] is False
        assert "text is required" in result["error"]

    def test_search_delegates_to_search_memory(self, provider):
        with patch("rag_memory.search_memory") as mock_search:
            mock_search.return_value = [{"score": 0.9, "text": "result", "metadata": {}}]
            result = json.loads(provider.handle_tool_call("hermes_rag_search", {"query": "test query"}))
            assert result["ok"] is True
            assert "results" in result
            mock_search.assert_called_once_with("test query", limit=5)

    def test_search_with_limit_delegates(self, provider):
        with patch("rag_memory.search_memory") as mock_search:
            mock_search.return_value = []
            result = json.loads(provider.handle_tool_call("hermes_rag_search", {"query": "test", "limit": 3}))
            assert result["ok"] is True
            mock_search.assert_called_once_with("test", limit=3)

    def test_delete_delegates_to_delete_memory(self, provider):
        with patch("rag_memory.delete_memory") as mock_delete:
            mock_delete.return_value = {"ok": True, "status": "completed"}
            result = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": "uuid-123"}))
            assert result["ok"] is True
            mock_delete.assert_called_once_with("uuid-123")

    def test_delete_missing_id_returns_error(self, provider):
        result = json.loads(provider.handle_tool_call("hermes_rag_delete", {}))
        assert result["ok"] is False
        assert "memory_id is required" in result["error"]


class TestHermesRAGProviderErrors:
    """Verify errors are propagated clearly."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def test_unknown_tool_returns_error(self, provider):
        result = json.loads(provider.handle_tool_call("unknown_tool", {}))
        assert result["ok"] is False
        # Lowercase, matching every other error string this provider returns
        # ("text is required", "memory_id is required", "session_id is
        # required", "query is required").
        assert "unknown tool" in result["error"]

    def test_exception_in_delegation_returns_error_json(self, provider):
        with patch("rag_memory.search_memory", side_effect=RuntimeError("Qdrant down")):
            result = json.loads(provider.handle_tool_call("hermes_rag_search", {"query": "test"}))
            assert result["ok"] is False
            assert "Qdrant down" in result["error"]

    def test_exception_in_store_returns_error_json(self, provider):
        with patch("rag_memory.add_memory", side_effect=Exception("full disk")):
            result = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": "data"}))
            assert result["ok"] is False
            assert "full disk" in result["error"]


class TestHermesRAGProviderRegistration:
    """Verify the provider can register through the plugin discovery mechanism."""

    def test_register_function_exists(self):
        from plugins.memory.jarvis import register
        assert callable(register)

    def test_register_adds_provider(self):
        from plugins.memory.jarvis import register
        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_memory_provider.assert_called_once()
        provider = mock_ctx.register_memory_provider.call_args[0][0]
        assert provider.name == "jarvis"


class TestHermesRAGProviderDiscovery:
    """Verify the provider is discoverable via the plugin system."""

    def test_discoverable_via_discover_memory_providers(self):
        from plugins.memory import discover_memory_providers
        providers = discover_memory_providers()
        names = [name for name, _, _ in providers]
        assert "jarvis" in names

    def test_loadable_via_load_memory_provider(self):
        from plugins.memory import load_memory_provider
        provider = load_memory_provider("jarvis")
        assert provider is not None
        assert provider.name == "jarvis"

    def test_is_available_returns_bool(self):
        from plugins.memory.jarvis import HermesRAGProvider
        provider = HermesRAGProvider()
        assert isinstance(provider.is_available(), bool)


class TestHermesRAGProviderTurnAwareContinuityPrefetch:
    """Phase 10C: turn-aware, read-only continuity lookup inside prefetch().

    Every test uses hermes_home=str(tmp_path) so continuity.db is isolated
    from production (~/.hermes/continuity.db is never touched). Qdrant is
    independently isolated by this file's module-scoped, autouse
    _isolated_rag_memory_module fixture, which patches rag_memory.client /
    QDRANT_PATH regardless of hermes_home — needed here because, unlike
    the continuity-only test files, prefetch() also calls
    rag_memory.search_memory.
    """

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def _call(self, provider, tool_name, args):
        return json.loads(provider.handle_tool_call(tool_name, args))

    def test_specific_keyword_match_retrieves_relevant_goal(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Expand the honey business"})

        with patch("rag_memory.search_memory", return_value=[]):
            result = provider.prefetch("What's the status of my honey business goal?")

        assert result
        assert "Jarvis Continuity" in result
        assert "Expand the honey business" in result

    def test_generic_intent_marker_retrieves_all_active_records(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Regulatory compliance"})
        self._call(provider, "hermes_task_create", {"title": "Prepare TBS submission"})
        self._call(provider, "hermes_decision_record", {"decision": "Use generic company names"})

        with patch("rag_memory.search_memory", return_value=[]):
            result = provider.prefetch("What are my current goals, tasks, and decisions?")

        assert "Regulatory compliance" in result
        assert "Prepare TBS submission" in result
        assert "Use generic company names" in result

    def test_new_record_created_mid_session_is_visible_to_later_prefetch(self, provider, tmp_path):
        """The turn-aware lookup reflects CURRENT database state — a record
        created after the provider was initialized (and after an earlier
        prefetch already ran) must still be retrievable by a later,
        relevant prefetch in the same provider/session, unlike
        system_prompt_block()'s once-per-session cached snapshot."""
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")

        with patch("rag_memory.search_memory", return_value=[]):
            before = provider.prefetch("What is my honeybee project goal?")
        assert before == ""  # nothing exists yet

        self._call(provider, "hermes_goal_create", {"title": "Honeybee project rollout"})

        with patch("rag_memory.search_memory", return_value=[]):
            after = provider.prefetch("What is my honeybee project goal?")
        assert "Honeybee project rollout" in after

    def test_irrelevant_self_contained_query_does_not_inject_continuity(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Expand the honey business"})

        with patch("rag_memory.search_memory", return_value=[]):
            result = provider.prefetch("What is 2 plus 2?")

        assert result == ""

    def test_missing_continuity_db_does_not_create_it_via_prefetch(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")

        with patch("rag_memory.search_memory", return_value=[]):
            result = provider.prefetch("What are my current goals and tasks?")

        assert result == ""
        assert not (tmp_path / "continuity.db").exists()

    def test_prefetch_performs_no_writes(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Seed goal for read-only check"})
        db_path = tmp_path / "continuity.db"
        before = db_path.stat()

        with patch("rag_memory.search_memory", return_value=[]):
            for query in ["What's my seed goal?", "current status", "unrelated question"]:
                provider.prefetch(query)

        after = db_path.stat()
        assert before.st_mtime == after.st_mtime
        assert before.st_size == after.st_size

    def test_prefetch_never_calls_any_continuity_write_function(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Seed goal"})

        with patch("rag_memory.search_memory", return_value=[]), \
             patch("hermes_cli.continuity_db.create_goal") as m_cg, \
             patch("hermes_cli.continuity_db.update_goal") as m_ug, \
             patch("hermes_cli.continuity_db.create_task") as m_ct, \
             patch("hermes_cli.continuity_db.update_task") as m_ut, \
             patch("hermes_cli.continuity_db.complete_task") as m_cpt, \
             patch("hermes_cli.continuity_db.create_decision") as m_cd, \
             patch("hermes_cli.continuity_db.supersede_decision") as m_sd:
            provider.prefetch("What is my current goal?")
        for mock in (m_cg, m_ug, m_ct, m_ut, m_cpt, m_cd, m_sd):
            mock.assert_not_called()

    def test_existing_rag_prefetch_behavior_unchanged_when_no_continuity_match(self, provider, tmp_path):
        """When continuity has nothing relevant, prefetch()'s output must be
        byte-identical to the pre-Phase-10C RAG-only shape."""
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        fake_results = [{"score": 0.9, "text": "Jarvis uses Qdrant.", "metadata": {}}]

        with patch("rag_memory.search_memory", return_value=fake_results) as mock_search:
            result = provider.prefetch("Where does Jarvis store memory?")

        mock_search.assert_called_once_with("Where does Jarvis store memory?", limit=5)
        expected = json.dumps(
            {
                "source": "jarvis_rag",
                "query": "Where does Jarvis store memory?",
                "results": fake_results,
            },
            ensure_ascii=False,
        )
        assert result == expected

    def test_both_rag_and_continuity_present_when_both_relevant(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Expand the honey business"})
        fake_rag_results = [{"score": 0.9, "text": "We discussed honey pricing last month.", "metadata": {}}]

        with patch("rag_memory.search_memory", return_value=fake_rag_results):
            result = provider.prefetch("What did we discuss about the honey business, and what's my goal?")

        assert '"source": "jarvis_rag"' in result
        assert "We discussed honey pricing last month." in result
        assert "Jarvis Continuity" in result
        assert "Expand the honey business" in result

    def test_continuity_retrieval_bounded_by_existing_goal_limit(self, provider, tmp_path):
        from plugins.memory.jarvis import CONTINUITY_CONTEXT_MAX_GOALS

        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        for i in range(CONTINUITY_CONTEXT_MAX_GOALS + 4):
            self._call(provider, "hermes_goal_create", {"title": f"Honey business goal {i}"})

        with patch("rag_memory.search_memory", return_value=[]):
            result = provider.prefetch("What is my honey business goal?")

        goal_lines = [l for l in result.splitlines() if l.startswith("- [priority")]
        assert len(goal_lines) == CONTINUITY_CONTEXT_MAX_GOALS

    def test_continuity_db_read_failure_is_nonfatal(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Seed goal"})

        with patch("rag_memory.search_memory", return_value=[]), \
             patch("hermes_cli.continuity_db.list_goals", side_effect=RuntimeError("db exploded")):
            result = provider.prefetch("What is my current goal?")  # must not raise

        assert result == ""

    def test_continuity_db_read_failure_does_not_break_rag_prefetch(self, provider, tmp_path):
        """A broken continuity read must not take down the independent,
        already-working RAG prefetch path."""
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Seed goal"})
        # Score is well above Phase 18A's RAG_PREFETCH_MIN_SCORE (0.84) —
        # this test is about continuity/RAG fault isolation, not score
        # filtering, so the mock score must clear the floor to isolate
        # what's actually being tested here.
        fake_results = [{"score": 0.95, "text": "Some RAG memory.", "metadata": {}}]

        with patch("rag_memory.search_memory", return_value=fake_results), \
             patch("hermes_cli.continuity_db.list_goals", side_effect=RuntimeError("db exploded")):
            result = provider.prefetch("What is my current goal?")

        assert '"source": "jarvis_rag"' in result
        assert "Some RAG memory." in result


class TestHermesRAGProviderPrefetchTrivialPromptGating:
    """Phase 11A: skip the RAG/Qdrant search for trivial prompts.

    Reuses agent.memory_provider.is_trivial_prompt() — the same trivial-
    prompt definition other callers already use — so a bare greeting,
    acknowledgement, or slash command never triggers a Qdrant round-trip.
    The continuity branch is deliberately independent of this gate and
    must behave exactly as it did before Phase 11A regardless of whether
    the query is trivial.
    """

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def _call(self, provider, tool_name, args):
        return json.loads(provider.handle_tool_call(tool_name, args))

    def test_trivial_query_skips_rag_search(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")

        with patch("rag_memory.search_memory") as mock_search:
            result = provider.prefetch("ok")

        mock_search.assert_not_called()
        assert result == ""

    def test_substantive_query_still_calls_rag_search(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        fake_results = [{"score": 0.9, "text": "Jarvis uses Qdrant.", "metadata": {}}]

        with patch("rag_memory.search_memory", return_value=fake_results) as mock_search:
            result = provider.prefetch("Where does Jarvis store memory?")

        mock_search.assert_called_once_with("Where does Jarvis store memory?", limit=5)
        expected = json.dumps(
            {
                "source": "jarvis_rag",
                "query": "Where does Jarvis store memory?",
                "results": fake_results,
            },
            ensure_ascii=False,
        )
        assert result == expected

    def test_trivial_query_still_surfaces_relevant_continuity(self, provider, tmp_path):
        """A trivial-per-is_trivial_prompt query that ALSO carries a
        continuity intent marker (e.g. a bare "status" check) must still
        get the continuity block — the RAG gate must not suppress it."""
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Expand the honey business"})

        with patch("rag_memory.search_memory") as mock_search:
            # "status" is a real word, not trivial per is_trivial_prompt,
            # and is also a _CONTINUITY_INTENT_MARKERS entry — exercises
            # both branches together without conflating triviality with
            # continuity relevance.
            result = provider.prefetch("status")

        mock_search.assert_called_once_with("status", limit=5)
        assert "Jarvis Continuity" in result
        assert "Expand the honey business" in result

    def test_substantive_query_preserves_existing_continuity_behavior(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Expand the honey business"})

        with patch("rag_memory.search_memory", return_value=[]):
            result = provider.prefetch("What's the status of my honey business goal?")

        assert "Jarvis Continuity" in result
        assert "Expand the honey business" in result

    def test_rag_failure_remains_nonfatal_and_continuity_still_works(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Expand the honey business"})

        with patch("rag_memory.search_memory", side_effect=RuntimeError("qdrant exploded")):
            result = provider.prefetch("What's the status of my honey business goal?")

        assert '"source": "jarvis_rag"' not in result
        assert "Jarvis Continuity" in result
        assert "Expand the honey business" in result

    def test_empty_query_still_short_circuits_before_either_branch(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")

        with patch("rag_memory.search_memory") as mock_search:
            result = provider.prefetch("")

        mock_search.assert_not_called()
        assert result == ""


class TestHermesRAGProviderPrefetchRelevanceFloor:
    """Phase 18A: automatic per-turn RAG injection is filtered by a
    calibrated relevance floor (RAG_PREFETCH_MIN_SCORE) before reaching
    the model. search_memory()'s own call (still limit=5) is untouched —
    only what gets attached to prefetch()'s output is narrowed. The
    explicit hermes_rag_search tool never goes through this filter.
    """

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def _call(self, provider, tool_name, args):
        return json.loads(provider.handle_tool_call(tool_name, args))

    # -- A: threshold behavior on the filter helper itself -----------------------

    def test_filter_helper_retains_result_at_threshold(self):
        from plugins.memory.jarvis import _filter_rag_results_by_score, RAG_PREFETCH_MIN_SCORE

        results = [{"score": RAG_PREFETCH_MIN_SCORE, "text": "at threshold", "metadata": {}}]
        assert _filter_rag_results_by_score(results) == results

    def test_filter_helper_excludes_result_below_threshold(self):
        from plugins.memory.jarvis import _filter_rag_results_by_score, RAG_PREFETCH_MIN_SCORE

        results = [{"score": RAG_PREFETCH_MIN_SCORE - 0.01, "text": "just below", "metadata": {}}]
        assert _filter_rag_results_by_score(results) == []

    def test_filter_helper_retains_result_above_threshold(self):
        from plugins.memory.jarvis import _filter_rag_results_by_score

        results = [{"score": 0.95, "text": "well above", "metadata": {}}]
        assert _filter_rag_results_by_score(results) == results

    # -- B: automatic RAG behavior through prefetch() -----------------------------

    def test_strong_result_remains_in_automatic_prefetch(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        fake_results = [{"score": 0.95, "text": "Strong match.", "metadata": {}}]

        with patch("rag_memory.search_memory", return_value=fake_results):
            result = provider.prefetch("some substantive query")

        assert '"source": "jarvis_rag"' in result
        assert "Strong match." in result

    def test_weak_result_is_excluded_from_automatic_prefetch(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        fake_results = [{"score": 0.5, "text": "Weak match.", "metadata": {}}]

        with patch("rag_memory.search_memory", return_value=fake_results):
            result = provider.prefetch("some substantive query")

        assert "Weak match." not in result
        assert '"source": "jarvis_rag"' not in result

    def test_mixed_strong_and_weak_results_retain_only_strong(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        fake_results = [
            {"score": 0.92, "text": "Strong one.", "metadata": {}},
            {"score": 0.85, "text": "Strong two.", "metadata": {}},
            {"score": 0.6, "text": "Weak one.", "metadata": {}},
            {"score": 0.4, "text": "Weak two.", "metadata": {}},
        ]

        with patch("rag_memory.search_memory", return_value=fake_results):
            result = provider.prefetch("some substantive query")

        assert "Strong one." in result
        assert "Strong two." in result
        assert "Weak one." not in result
        assert "Weak two." not in result

    def test_all_weak_results_produce_no_rag_block(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        fake_results = [
            {"score": 0.6, "text": "Weak one.", "metadata": {}},
            {"score": 0.5, "text": "Weak two.", "metadata": {}},
        ]

        with patch("rag_memory.search_memory", return_value=fake_results):
            result = provider.prefetch("some substantive query")

        assert result == ""

    def test_filtered_result_order_is_preserved(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        fake_results = [
            {"score": 0.99, "text": "First.", "metadata": {}},
            {"score": 0.3, "text": "Excluded middle.", "metadata": {}},
            {"score": 0.85, "text": "Second.", "metadata": {}},
        ]

        with patch("rag_memory.search_memory", return_value=fake_results):
            result = provider.prefetch("some substantive query")

        parsed = json.loads(result)
        texts = [r["text"] for r in parsed["results"]]
        assert texts == ["First.", "Second."]

    # -- C: continuity independence ------------------------------------------------

    def test_continuity_still_appears_when_rag_produces_no_qualifying_results(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Expand the honey business"})
        fake_results = [{"score": 0.3, "text": "Weak unrelated match.", "metadata": {}}]

        with patch("rag_memory.search_memory", return_value=fake_results):
            result = provider.prefetch("What's the status of my honey business goal?")

        assert '"source": "jarvis_rag"' not in result
        assert "Jarvis Continuity" in result
        assert "Expand the honey business" in result

    def test_rag_filtering_does_not_change_continuity_selection(self, provider, tmp_path):
        """Continuity's own selection logic (Phases 10C/16A/16B/17A) must
        be byte-for-byte identical regardless of what RAG returns."""
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")
        self._call(provider, "hermes_goal_create", {"title": "Expand the honey business"})

        with patch("rag_memory.search_memory", return_value=[]):
            without_rag = provider.prefetch("What's the status of my honey business goal?")
        with patch("rag_memory.search_memory", return_value=[{"score": 0.3, "text": "noise", "metadata": {}}]):
            with_weak_rag = provider.prefetch("What's the status of my honey business goal?")

        assert without_rag == with_weak_rag

    # -- D: existing behavior unchanged --------------------------------------------

    def test_trivial_prompt_still_skips_automatic_rag(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")

        with patch("rag_memory.search_memory") as mock_search:
            result = provider.prefetch("ok")

        mock_search.assert_not_called()
        assert result == ""

    def test_search_memory_still_called_with_limit_five(self, provider, tmp_path):
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="primary")

        with patch("rag_memory.search_memory", return_value=[]) as mock_search:
            provider.prefetch("some substantive query")

        mock_search.assert_called_once_with("some substantive query", limit=5)

    # -- E: explicit RAG search unaffected -----------------------------------------

    def test_explicit_rag_search_tool_does_not_apply_the_threshold(self, provider):
        weak_results = [{"score": 0.1, "text": "Very weak match.", "metadata": {}}]

        with patch("rag_memory.search_memory", return_value=weak_results) as mock_search:
            result = json.loads(provider.handle_tool_call("hermes_rag_search", {"query": "test query"}))

        mock_search.assert_called_once_with("test query", limit=5)
        assert result["ok"] is True
        assert result["results"] == weak_results  # unfiltered, unchanged

    # -- F: production safety (no live Qdrant/rag_memory import needed here) ------

    def test_filter_helper_never_imports_rag_memory(self):
        """_filter_rag_results_by_score is a pure list comprehension over
        already-returned dicts — confirm it touches no I/O."""
        import sys
        from plugins.memory.jarvis import _filter_rag_results_by_score

        had_rag_memory_before = "rag_memory" in sys.modules
        _filter_rag_results_by_score([{"score": 0.9, "text": "x", "metadata": {}}])
        if not had_rag_memory_before:
            assert "rag_memory" not in sys.modules


class TestHermesRAGProviderOnSessionSwitch:
    """Phase 13B: on_session_switch() must refresh session-scoped state
    (_session_id, _session_continuity_fragments) without touching
    profile-scoped state (_hermes_home, _session_end_learning_enabled) and
    without any continuity.db or Qdrant access."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    # -- A: session ID updates -------------------------------------------------

    def test_session_id_updates_to_new_session(self, provider, tmp_path):
        provider.initialize("session-A", hermes_home=str(tmp_path), agent_context="primary")
        assert provider._session_id == "session-A"

        provider.on_session_switch("session-B")
        assert provider._session_id == "session-B"

    def test_empty_new_session_id_is_a_noop(self, provider, tmp_path):
        provider.initialize("session-A", hermes_home=str(tmp_path), agent_context="primary")
        provider.on_session_switch("")
        assert provider._session_id == "session-A"

    # -- B: dedup state clears --------------------------------------------------

    def test_session_continuity_fragments_clear_on_switch(self, provider, tmp_path):
        provider.initialize("session-A", hermes_home=str(tmp_path), agent_context="primary")
        provider._track_continuity_fragments("Expand the honey business", "Some description")
        assert provider._session_continuity_fragments

        provider.on_session_switch("session-B")
        assert provider._session_continuity_fragments == set()

    def test_dedup_no_longer_fires_on_old_session_text_after_switch(self, provider, tmp_path):
        """Behavioral proof, not just an empty-set check: a text fragment
        that WOULD have deduped against session-A's tracked text must NOT
        dedupe anymore once session-B has started."""
        provider.initialize("session-A", hermes_home=str(tmp_path), agent_context="primary")
        provider._track_continuity_fragments("Expand the honey business into new markets")
        assert provider._is_duplicate_of_session_continuity("Expand the honey business into new markets")

        provider.on_session_switch("session-B")
        assert not provider._is_duplicate_of_session_continuity("Expand the honey business into new markets")

    # -- C: hermes_home preserved ------------------------------------------------

    def test_hermes_home_preserved_across_switch(self, provider, tmp_path):
        provider.initialize("session-A", hermes_home=str(tmp_path), agent_context="primary")
        before = provider._hermes_home
        provider.on_session_switch("session-B")
        assert provider._hermes_home == before == str(tmp_path)

    # -- D: learning policy preserved ---------------------------------------------

    def test_primary_learning_stays_enabled_after_switch(self, provider, tmp_path):
        provider.initialize("session-A", hermes_home=str(tmp_path), agent_context="primary")
        assert provider._session_end_learning_enabled is True
        provider.on_session_switch("session-B")
        assert provider._session_end_learning_enabled is True

    @pytest.mark.parametrize("blocked_context", ["oneshot", "cron", "flush", "subagent"])
    def test_noninteractive_learning_stays_disabled_after_switch(self, provider, tmp_path, blocked_context):
        provider.initialize(
            "session-A", hermes_home=str(tmp_path), agent_context=blocked_context
        )
        assert provider._session_end_learning_enabled is False
        provider.on_session_switch("session-B")
        assert provider._session_end_learning_enabled is False

    # -- E: no persistence side effects -------------------------------------------

    def test_session_switch_creates_no_continuity_db(self, provider, tmp_path):
        provider.initialize("session-A", hermes_home=str(tmp_path), agent_context="primary")
        provider.on_session_switch("session-B", parent_session_id="session-A", reset=True)
        assert not (tmp_path / "continuity.db").exists()

    def test_session_switch_never_imports_rag_memory(self, provider, tmp_path):
        """No Qdrant access: on_session_switch must never call into
        rag_memory's search/store functions."""
        provider.initialize("session-A", hermes_home=str(tmp_path), agent_context="primary")

        with patch("rag_memory.search_memory") as mock_search, \
             patch("rag_memory.store_memory_with_provenance") as mock_store:
            provider.on_session_switch("session-B", parent_session_id="session-A", reset=True)
        mock_search.assert_not_called()
        mock_store.assert_not_called()

    # -- F: manager-level wiring regression ---------------------------------------

    def test_memory_manager_forwards_session_switch_to_jarvis(self, tmp_path):
        """Regression guard for the actual call path: MemoryManager.
        on_session_switch() must reach HermesRAGProvider.on_session_switch()
        with the real /reset-style kwargs (reset=True)."""
        from agent.memory_manager import MemoryManager
        from plugins.memory.jarvis import HermesRAGProvider

        provider = HermesRAGProvider()
        provider.initialize("session-A", hermes_home=str(tmp_path), agent_context="primary")
        provider._track_continuity_fragments("Some tracked fragment text here")

        manager = MemoryManager()
        manager.add_provider(provider)
        manager.on_session_switch("session-B", parent_session_id="session-A", reset=True)

        assert provider._session_id == "session-B"
        assert provider._session_continuity_fragments == set()
