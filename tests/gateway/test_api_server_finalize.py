"""POST /api/sessions/{session_id}/finalize — opt-in end-of-session learning.

The endpoint hands a stored API conversation's completed turns to the memory
provider's session-end hook through a normally-built api_server agent. These
tests use a real SessionDB in the per-test tmp dir and a fake agent; they never
build a real AIAgent, call a model, or touch Qdrant.
"""

import asyncio
import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _finalizable_transcript,
    cors_middleware,
    security_headers_middleware,
)
from hermes_state import SessionDB

KEY = "sk-finalize-test-key-0123456789"
AUTH = {"Authorization": f"Bearer {KEY}"}
SID = "desktop_1791700000_abcdef01"


class FakeAgent:
    """Stands in for AIAgent: records the hand-off, refuses any model call."""

    def __init__(self):
        self.committed = []

    def commit_memory_session(self, messages):
        self.committed.append(list(messages))

    def run_conversation(self, *args, **kwargs):  # pragma: no cover - must not happen
        raise AssertionError("finalize must not run a model turn")


@pytest.fixture
def adapter(tmp_path):
    a = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY}))
    a._session_db = SessionDB(tmp_path / "state.db")
    return a


@pytest.fixture
def fake_agents(adapter):
    created = []

    def _create_agent(**kwargs):
        agent = FakeAgent()
        created.append((kwargs, agent))
        return agent

    with patch.object(adapter, "_create_agent", side_effect=_create_agent):
        yield created


def _app(adapter):
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/sessions/{session_id}/finalize", adapter._handle_finalize_session)
    return app


def _add_session(db, session_id, messages):
    db.create_session(session_id, "api_server")
    for message in messages:
        db.append_message(session_id, **message)


async def _finalize(adapter, session_id=SID, headers=AUTH, body=None):
    async with TestClient(TestServer(_app(adapter))) as client:
        kwargs = {"headers": headers}
        if body is not None:
            kwargs["json"] = body
        resp = await client.post(f"/api/sessions/{session_id}/finalize", **kwargs)
        data = await resp.json()
        # Let the background hand-off finish before inspecting its effects.
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))
        return resp.status, data


TOOL_CALL = [{"id": "call_1", "type": "function", "function": {"name": "memory", "arguments": "{}"}}]
FULL_CONVERSATION = [
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "The user prefers short answers.", "finish_reason": "stop"},
    {"role": "user", "content": "Second question (cancelled)"},
    {"role": "assistant", "content": "The user always wants partial noise", "finish_reason": None},
    {"role": "user", "content": "Third question"},
    {"role": "assistant", "content": "", "tool_calls": TOOL_CALL, "finish_reason": "tool_calls"},
    {"role": "tool", "content": "{\"ok\": true}", "tool_call_id": "call_1", "tool_name": "memory"},
    {"role": "assistant", "content": "The user decided to keep Qdrant in server mode.", "finish_reason": "stop"},
    {"role": "user", "content": "Fourth question, still unanswered"},
    {"role": "assistant", "content": "Truncated reply", "finish_reason": "length"},
]


# ---------------------------------------------------------------------------
# Transcript selection
# ---------------------------------------------------------------------------


class TestFinalizableTranscript:
    def test_keeps_completed_turns_and_tool_messages_only(self):
        kept = _finalizable_transcript(FULL_CONVERSATION)
        assert [m["content"] for m in kept] == [
            "First question",
            "The user prefers short answers.",
            "Second question (cancelled)",
            "Third question",
            "",
            "{\"ok\": true}",
            "The user decided to keep Qdrant in server mode.",
        ]

    def test_cancelled_failed_and_trailing_turns_are_excluded(self):
        kept = _finalizable_transcript(FULL_CONVERSATION)
        contents = [m["content"] for m in kept]
        assert "The user always wants partial noise" not in contents  # cancelled reply
        assert "Truncated reply" not in contents  # failed / incomplete reply
        assert "Fourth question, still unanswered" not in contents  # no completed answer
        assert kept[-1]["finish_reason"] == "stop"

    @pytest.mark.parametrize("messages", [
        [],
        [{"role": "user", "content": "hi"}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "partial", "finish_reason": None}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "cut", "finish_reason": "length"}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "finish_reason": "tool_calls"}],
    ], ids=["empty", "user-only", "cancelled", "failed", "tool-call-only"])
    def test_no_completed_reply_means_nothing_to_finalize(self, messages):
        assert _finalizable_transcript(messages) == []


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


class TestFinalizeEndpoint:
    @pytest.mark.asyncio
    async def test_requires_api_key(self, adapter, fake_agents):
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)
        status, _ = await _finalize(adapter, headers={})
        assert status == 401
        status, _ = await _finalize(adapter, headers={"Authorization": "Bearer wrong-key-000000000000"})
        assert status == 401
        assert fake_agents == []

    @pytest.mark.asyncio
    async def test_unknown_session_is_404(self, adapter, fake_agents):
        status, data = await _finalize(adapter, session_id="desktop_missing")
        assert status == 404
        assert data["error"]["code"] == "session_not_found"
        assert fake_agents == []

    @pytest.mark.asyncio
    async def test_completed_conversation_is_handed_to_the_session_end_hook(self, adapter, fake_agents):
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)

        status, data = await _finalize(adapter, body={"reason": "new_conversation"})

        assert status == 202
        assert data == {
            "object": "hermes.session.finalize", "session_id": SID,
            "finalized": True, "status": "accepted", "messages": 7,
        }
        assert len(fake_agents) == 1
        kwargs, agent = fake_agents[0]
        assert kwargs["session_id"] == SID  # provenance source_id
        assert len(agent.committed) == 1
        committed = agent.committed[0]
        assert committed == _finalizable_transcript(
            adapter._session_db.get_messages_as_conversation(SID)
        )
        assert all(m.get("finish_reason") != "length" for m in committed)
        assert any(m.get("role") == "tool" for m in committed)  # visible to the D13 check

    @pytest.mark.asyncio
    async def test_session_key_header_is_passed_through(self, adapter, fake_agents):
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)
        status, _ = await _finalize(adapter, headers={**AUTH, "X-Hermes-Session-Key": "jarvis-desktop"})
        assert status == 202
        assert fake_agents[0][0]["gateway_session_key"] == "jarvis-desktop"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("messages", [
        [{"role": "user", "content": "hi"}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "partial", "finish_reason": None}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "cut", "finish_reason": "length"}],
    ], ids=["no-reply", "cancelled", "failed"])
    async def test_incomplete_conversation_is_not_finalized(self, adapter, fake_agents, messages):
        _add_session(adapter._session_db, SID, messages)
        status, data = await _finalize(adapter)
        assert status == 200
        assert data["finalized"] is False and data["reason"] == "no_completed_turn"
        assert fake_agents == []

    @pytest.mark.asyncio
    async def test_refused_while_a_turn_is_in_progress(self, adapter, fake_agents):
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)
        adapter._session_turn_started(SID)
        status, data = await _finalize(adapter)
        assert status == 409
        assert data["error"]["code"] == "session_turn_in_progress"
        assert fake_agents == []

        adapter._session_turn_finished(SID)
        status, _ = await _finalize(adapter)
        assert status == 202

    @pytest.mark.asyncio
    async def test_repeated_finalize_is_accepted_each_time(self, adapter, fake_agents):
        # Duplicate prevention lives in storage (stable learning IDs); the
        # endpoint itself is safe to call again, e.g. after a partial failure.
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)
        assert (await _finalize(adapter))[0] == 202
        assert (await _finalize(adapter))[0] == 202
        assert [len(agent.committed) for _, agent in fake_agents] == [1, 1]
        assert fake_agents[0][1].committed == fake_agents[1][1].committed

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [{"reason": "Not Valid!"}, {"reason": 5}, {"reason": "x" * 41}])
    async def test_invalid_reason_is_rejected(self, adapter, fake_agents, body):
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)
        status, data = await _finalize(adapter, body=body)
        assert status == 400
        assert data["error"]["code"] == "invalid_reason"
        assert fake_agents == []

    @pytest.mark.asyncio
    async def test_hand_off_failure_is_logged_as_warning(self, adapter, caplog):
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)
        with patch.object(adapter, "_create_agent", side_effect=RuntimeError("provider unavailable")):
            with caplog.at_level(logging.INFO, logger="gateway.platforms.api_server"):
                status, _ = await _finalize(adapter)
        assert status == 202
        failures = [r for r in caplog.records if "session finalize failed" in r.getMessage()]
        assert failures and failures[0].levelno == logging.WARNING
        assert "provider unavailable" in failures[0].getMessage()

    @pytest.mark.asyncio
    async def test_successful_hand_off_is_logged(self, adapter, fake_agents, caplog):
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)
        with caplog.at_level(logging.INFO, logger="gateway.platforms.api_server"):
            await _finalize(adapter, body={"reason": "window_close"})
        ran = [r for r in caplog.records if "ran the session-end hook" in r.getMessage()]
        assert ran and ran[0].levelno == logging.INFO
        assert SID in ran[0].getMessage() and "reason=window_close" in ran[0].getMessage()


# ---------------------------------------------------------------------------
# In-progress tracking on the real turn paths
# ---------------------------------------------------------------------------


class TestSessionTurnTracking:
    @pytest.mark.asyncio
    async def test_run_agent_marks_the_session_busy_only_during_the_turn(self, adapter):
        seen = {}
        agent = MagicMock()
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

        def _run_conversation(**kwargs):
            seen["during"] = dict(adapter._session_turns_in_progress)
            return {"final_response": "ok"}

        agent.run_conversation.side_effect = _run_conversation
        with patch.object(adapter, "_create_agent", return_value=agent):
            await adapter._run_agent(user_message="hi", conversation_history=[], session_id=SID)

        assert seen["during"] == {SID: 1}
        assert adapter._session_turns_in_progress == {}

    @pytest.mark.asyncio
    async def test_failed_turn_still_clears_the_busy_mark(self, adapter):
        with patch.object(adapter, "_create_agent", side_effect=RuntimeError("boom")):
            try:
                await adapter._run_agent(user_message="hi", conversation_history=[], session_id=SID)
            except RuntimeError:
                pass
        assert adapter._session_turns_in_progress == {}

    def test_counter_handles_overlapping_turns(self, adapter):
        adapter._session_turn_started(SID)
        adapter._session_turn_started(SID)
        adapter._session_turn_finished(SID)
        assert adapter._session_turns_in_progress == {SID: 1}
        adapter._session_turn_finished(SID)
        adapter._session_turn_finished(SID)  # extra finish never goes negative
        assert adapter._session_turns_in_progress == {}
        adapter._session_turn_started(None)  # sessionless requests are ignored
        assert adapter._session_turns_in_progress == {}

    def test_route_is_registered(self, adapter):
        routes = {(m, p) for m, p, _ in adapter._http_route_table()}
        assert ("POST", "/api/sessions/{session_id}/finalize") in routes


# ---------------------------------------------------------------------------
# No-learning sessions (agent/no_learning.py)
# ---------------------------------------------------------------------------


class TestNoLearningFinalize:
    NOLEARN = "nolearn_desktop_1791700000_abcdef01"

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        import agent.no_learning as nl
        monkeypatch.delenv(nl.NO_LEARNING_ENV, raising=False)
        monkeypatch.setattr(nl, "_warned_values", set())

    @pytest.mark.asyncio
    async def test_nolearn_session_is_not_finalized_and_no_agent_is_created(self, adapter, fake_agents, caplog):
        _add_session(adapter._session_db, self.NOLEARN, FULL_CONVERSATION)
        with caplog.at_level(logging.INFO, logger="gateway.platforms.api_server"):
            status, data = await _finalize(adapter, session_id=self.NOLEARN, body={"reason": "new_conversation"})
        assert status == 200
        assert data == {"object": "hermes.session.finalize", "session_id": self.NOLEARN,
                        "finalized": False, "reason": "learning_disabled"}
        assert fake_agents == []
        skips = [r for r in caplog.records if "persistent learning disabled" in r.getMessage()]
        assert len(skips) == 1 and self.NOLEARN in skips[0].getMessage()

    @pytest.mark.asyncio
    async def test_env_switch_disables_finalize_for_ordinary_sessions(self, adapter, fake_agents, monkeypatch):
        monkeypatch.setenv("HERMES_NO_LEARNING", "1")
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)
        status, data = await _finalize(adapter)
        assert status == 200 and data["reason"] == "learning_disabled"
        assert fake_agents == []

    @pytest.mark.asyncio
    async def test_unknown_nolearn_session_is_still_404(self, adapter, fake_agents):
        status, data = await _finalize(adapter, session_id="nolearn_missing")
        assert status == 404 and data["error"]["code"] == "session_not_found"

    @pytest.mark.asyncio
    async def test_ordinary_session_unchanged(self, adapter, fake_agents):
        _add_session(adapter._session_db, SID, FULL_CONVERSATION)
        status, data = await _finalize(adapter)
        assert status == 202 and data["finalized"] is True
        assert len(fake_agents) == 1
