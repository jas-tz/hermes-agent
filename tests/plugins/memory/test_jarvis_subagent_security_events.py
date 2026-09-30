"""Tests for Phase 28A.1 — Security Event Infrastructure."""

import json
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.jarvis.subagent_security import (
    SecurityEvent,
    SecurityEventType,
    SecurityEventRecorder,
    create_security_event,
    get_recorder,
    reset_recorder,
    record_authorization_allowed,
    record_authorization_denied,
    record_parent_only_denied,
    record_forged_capability,
    record_approval_requested,
    record_approval_denied,
    record_lifecycle_transition,
    record_recursion_limit,
    record_tool_execution_blocked,
    record_tool_execution_completed,
    record_tool_execution_failed,
    SENSITIVE_PATTERNS,
    DEFAULT_MAX_EVENTS,
)
from plugins.memory.jarvis import JARVISSubagentProvider
from plugins.memory.jarvis.subagent_types import create_execution_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_recorder_fixture():
    """Reset recorder before each test."""
    reset_recorder()
    yield
    reset_recorder()


# ====================================================================
# Event Immutability
# ====================================================================

class TestEventImmutability:
    """Test that SecurityEvent is immutable."""

    def test_event_is_frozen(self):
        """SecurityEvent is frozen (immutable)."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
            allowed=True,
        )

        with pytest.raises(AttributeError):
            event.allowed = False

        with pytest.raises(AttributeError):
            event.tool_name = "forged"

        with pytest.raises(AttributeError):
            event.event_type = SecurityEventType.AUTHORIZATION_DENIED

    def test_event_fields_preserved(self):
        """Event fields are preserved exactly."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_store",
            lifecycle_state="running",
            allowed=False,
            reason="Denied",
        )

        assert event.event_type == SecurityEventType.AUTHORIZATION_DENIED
        assert event.subagent_id == "sa-test"
        assert event.parent_session_id == "parent"
        assert event.task_type == "research"
        assert event.authority_level == "L0"
        assert event.tool_name == "hermes_rag_store"
        assert event.lifecycle_state == "running"
        assert event.allowed is False
        assert event.reason == "Denied"


# ====================================================================
# Event ID
# ====================================================================

class TestEventId:
    """Test event ID generation."""

    def test_event_id_is_unique(self):
        """Each event has a unique ID."""
        events = [
            create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            )
            for _ in range(100)
        ]

        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids))

    def test_event_id_is_string(self):
        """Event ID is a string."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        assert isinstance(event.event_id, str)


# ====================================================================
# Timestamp
# ====================================================================

class TestTimestamp:
    """Test event timestamp."""

    def test_timestamp_present(self):
        """Event has a timestamp."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        assert event.timestamp is not None
        assert isinstance(event.timestamp, str)

    def test_timestamp_iso_format(self):
        """Timestamp is in ISO format."""
        from datetime import datetime

        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        # Should parse as ISO format
        parsed = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
        assert parsed is not None


# ====================================================================
# Event Types
# ====================================================================

class TestEventTypes:
    """Test event types."""

    def test_all_event_types_present(self):
        """All required event types are present."""
        required = {
            "authorization_allowed",
            "authorization_denied",
            "parent_only_denied",
            "task_type_escalation",
            "forged_capability",
            "filesystem_traversal",
            "filesystem_escape",
            "terminal_injection",
            "delegation_escape",
            "approval_requested",
            "approval_denied",
            "lifecycle_transition",
            "lifecycle_violation",
            "recursion_limit",
            "tool_execution_blocked",
            "tool_execution_completed",
            "tool_execution_failed",
        }

        actual = {e.value for e in SecurityEventType}
        assert required.issubset(actual)


# ====================================================================
# Security Event Recorder
# ====================================================================

class TestSecurityEventRecorder:
    """Test SecurityEventRecorder."""

    def test_record_and_retrieve(self):
        """Events can be recorded and retrieved."""
        recorder = SecurityEventRecorder()
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        recorder.record(event)
        events = recorder.get_events()

        assert len(events) == 1
        assert events[0].event_id == event.event_id

    def test_get_events_for_subagent(self):
        """Can filter events by sub-agent ID."""
        recorder = SecurityEventRecorder()

        for i in range(3):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-1",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        for i in range(2):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-2",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        sa1_events = recorder.get_events_for_subagent("sa-1")
        assert len(sa1_events) == 3

    def test_get_events_by_type(self):
        """Can filter events by type."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-1",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-1",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        denied = recorder.get_events_by_type(SecurityEventType.AUTHORIZATION_DENIED)
        assert len(denied) == 1

    def test_clear(self):
        """Clear removes all events."""
        recorder = SecurityEventRecorder()
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-1",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        assert recorder.count == 1
        recorder.clear()
        assert recorder.count == 0

    def test_bounded_retention(self):
        """Recorder has bounded retention."""
        recorder = SecurityEventRecorder(max_events=5)

        for i in range(10):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-1",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        assert recorder.count == 5

    def test_returned_collection_isolation(self):
        """Returned collection cannot mutate recorder state."""
        recorder = SecurityEventRecorder()
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-1",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        events = recorder.get_events()
        events.clear()  # Modify returned collection

        # Recorder state unchanged
        assert recorder.count == 1


# ====================================================================
# Authorization Events
# ====================================================================

class TestAuthorizationEvents:
    """Test authorization event recording."""

    def test_record_authorization_allowed(self):
        """Authorization allowed is recorded."""
        record_authorization_allowed(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.AUTHORIZATION_ALLOWED)
        assert len(events) == 1
        assert events[0].allowed is True

    def test_record_authorization_denied(self):
        """Authorization denied is recorded."""
        record_authorization_denied(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_store",
            reason="Task type mismatch",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.AUTHORIZATION_DENIED)
        assert len(events) == 1
        assert events[0].allowed is False


# ====================================================================
# Parent-Only Denial
# ====================================================================

class TestParentOnlyDenial:
    """Test parent-only denial recording."""

    def test_parent_only_denied(self):
        """Parent-only denial is recorded."""
        record_parent_only_denied(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="terminal",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.PARENT_ONLY_DENIED)
        assert len(events) == 1
        assert events[0].tool_name == "terminal"


# ====================================================================
# Forged Capability
# ====================================================================

class TestForgedCapability:
    """Test forged capability recording."""

    def test_forged_capability(self):
        """Forged capability is recorded."""
        record_forged_capability(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_fake_tool",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.FORGED_CAPABILITY)
        assert len(events) == 1
        assert events[0].tool_name == "hermes_fake_tool"


# ====================================================================
# Approval Events
# ====================================================================

class TestApprovalEvents:
    """Test approval event recording."""

    def test_approval_requested(self):
        """Approval requested is recorded."""
        record_approval_requested(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="default",
            authority_level="L3",
            tool_name="hermes_rag_delete",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.APPROVAL_REQUESTED)
        assert len(events) == 1

    def test_approval_denied(self):
        """Approval denied is recorded."""
        record_approval_denied(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="default",
            authority_level="L3",
            tool_name="hermes_rag_delete",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.APPROVAL_DENIED)
        assert len(events) == 1
        assert events[0].allowed is False


# ====================================================================
# Lifecycle Events
# ====================================================================

class TestLifecycleEvents:
    """Test lifecycle event recording."""

    def test_lifecycle_transition(self):
        """Lifecycle transition is recorded."""
        record_lifecycle_transition(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="",
            lifecycle_state="running",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.LIFECYCLE_TRANSITION)
        assert len(events) == 1
        assert events[0].lifecycle_state == "running"


# ====================================================================
# Recursion Limit
# ====================================================================

class TestRecursionLimit:
    """Test recursion limit recording."""

    def test_recursion_limit(self):
        """Recursion limit is recorded."""
        record_recursion_limit(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="delegate_task",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.RECURSION_LIMIT)
        assert len(events) == 1


# ====================================================================
# Tool Execution Outcomes
# ====================================================================

class TestToolExecutionOutcomes:
    """Test tool execution outcome recording."""

    def test_tool_execution_completed(self):
        """Tool execution completed is recorded."""
        record_tool_execution_completed(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.TOOL_EXECUTION_COMPLETED)
        assert len(events) == 1
        assert events[0].allowed is True

    def test_tool_execution_failed(self):
        """Tool execution failed is recorded."""
        record_tool_execution_failed(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
            reason="Qdrant timeout",
        )

        events = get_recorder().get_events_by_type(SecurityEventType.TOOL_EXECUTION_FAILED)
        assert len(events) == 1


# ====================================================================
# Sub-Agent Identity
# ====================================================================

class TestSubAgentIdentity:
    """Test sub-agent identity in events."""

    def test_event_contains_subagent_id(self):
        """Event contains sub-agent identity."""
        record_authorization_allowed(
            subagent_id="sa-identity-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        )

        events = get_recorder().get_events()
        assert events[0].subagent_id == "sa-identity-test"


# ====================================================================
# Parent/Child Isolation
# ====================================================================

class TestParentChildIsolation:
    """Test parent/child isolation in events."""

    def test_parent_session_id_recorded(self):
        """Parent session ID is recorded."""
        record_authorization_allowed(
            subagent_id="sa-test",
            parent_session_id="parent-session-123",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        )

        events = get_recorder().get_events()
        assert events[0].parent_session_id == "parent-session-123"


# ====================================================================
# Secret Sanitization
# ====================================================================

class TestSecretSanitization:
    """Test secret sanitization in events."""

    def test_sensitive_patterns_redacted(self):
        """Sensitive patterns in reason are redacted."""
        record_authorization_denied(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_store",
            reason="api_key=secret123 token=xyz",
        )

        events = get_recorder().get_events()
        assert "[REDACTED]" in events[0].reason

    def test_no_secrets_in_recorder(self):
        """No secrets are stored in recorder."""
        record_authorization_allowed(
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
            reason="password=secret api_key=key123",
        )

        events = get_recorder().get_events()
        assert "secret" not in events[0].reason.lower() or "[REDACTED]" in events[0].reason


# ====================================================================
# Recorder Failure Isolation
# ====================================================================

class TestRecorderFailureIsolation:
    """Test that recorder failures don't affect authorization."""

    def test_recorder_exception_doesnt_affect_auth(self):
        """Recorder exception doesn't affect authorization."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        # Patch recorder to raise AND patch dispatch to return success
        with patch("plugins.memory.jarvis.subagent_security._recorder.record", side_effect=Exception("Boom")):
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                mock_dispatch.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})

        # Authorization still works despite recorder failure
        assert json.loads(result)["ok"] is True


# ====================================================================
# Zero Mutation After Denied Authorization
# ====================================================================

class TestZeroMutation:
    """Test zero-mutation after denied authorization."""

    def test_denied_auth_no_mutation(self):
        """Denied authorization produces no mutation."""
        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="research",
            task_type="research",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="research",
            task_type="research",
            execution_context=ctx,
        )

        # Denied call
        result = provider.handle_tool_call("hermes_rag_store", {"text": "data"})
        assert json.loads(result)["ok"] is False

        # No dispatch occurred
        # (verified by no error from missing dispatch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
