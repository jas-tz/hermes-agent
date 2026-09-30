"""Phase 28C — Security Audit Integrity & Tamper Resistance.

Tests the tamper resistance and integrity of the security event infrastructure
in plugins/memory/jarvis/subagent_security.py.

Covers: immutability, identity integrity, sub-agent isolation, authority
integrity, event tampering, correlation integrity, concurrency, recorder
failure isolation, sanitization, and zero-mutation guarantees.
"""

import json
import threading
import time
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
    record_tool_execution_blocked,
    record_tool_execution_completed,
    record_tool_execution_failed,
    record_approval_requested,
    record_approval_denied,
    record_lifecycle_transition,
    record_recursion_limit,
    _sanitize_reason,
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
# 1. SecurityEvent Immutability
# ====================================================================

class TestSecurityEventImmutability:
    """Test that SecurityEvent is fully immutable."""

    def test_event_frozen_all_fields(self):
        """All fields are frozen."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
            lifecycle_state="running",
            allowed=True,
            reason="allowed",
        )

        for field in ['event_id', 'timestamp', 'event_type', 'subagent_id',
                      'parent_session_id', 'task_type', 'authority_level',
                      'tool_name', 'lifecycle_state', 'allowed', 'reason']:
            with pytest.raises(AttributeError):
                setattr(event, field, "forged")

    def test_event_equality_by_value(self):
        """Events with same values are equal."""
        event1 = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
            allowed=True,
        )
        event2 = SecurityEvent(
            event_id=event1.event_id,
            timestamp=event1.timestamp,
            event_type=event1.event_type,
            subagent_id=event1.subagent_id,
            parent_session_id=event1.parent_session_id,
            task_type=event1.task_type,
            authority_level=event1.authority_level,
            tool_name=event1.tool_name,
            lifecycle_state=event1.lifecycle_state,
            allowed=event1.allowed,
            reason=event1.reason,
        )

        assert event1 == event2

    def test_event_hashable(self):
        """Events are hashable (can be used in sets/dicts)."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        # Should be hashable
        hash(event)

        # Should work in a set
        s = {event}
        assert len(s) == 1


# ====================================================================
# 2. Event Identity Integrity
# ====================================================================

class TestEventIdentityIntegrity:
    """Test event identity cannot be forged."""

    def test_event_id_unique_per_event(self):
        """Each event has a unique ID."""
        events = [
            create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            )
            for _ in range(1000)
        ]

        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids))

    def test_event_id_is_uuid_format(self):
        """Event ID is a valid UUID string."""
        import uuid

        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        # Should parse as UUID
        uuid.UUID(event.event_id)

    def test_event_id_preserved_through_recording(self):
        """Event ID is preserved when recorded and retrieved."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        recorder = SecurityEventRecorder()
        recorder.record(event)

        retrieved = recorder.get_events()[0]
        assert retrieved.event_id == event.event_id


# ====================================================================
# 3. Sub-Agent Isolation
# ====================================================================

class TestSubAgentIsolation:
    """Test sub-agent isolation in event records."""

    def test_events_isolated_by_subagent(self):
        """Events for subagent A cannot be accessed through subagent B."""
        recorder = SecurityEventRecorder()

        for i in range(5):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-A",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        for i in range(3):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-B",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        sa_a_events = recorder.get_events_for_subagent("sa-A")
        sa_b_events = recorder.get_events_for_subagent("sa-B")

        assert len(sa_a_events) == 5
        assert len(sa_b_events) == 3

        # No overlap
        for e in sa_a_events:
            assert e.subagent_id == "sa-A"
        for e in sa_b_events:
            assert e.subagent_id == "sa-B"

    def test_subagent_cannot_see_other_events(self):
        """Filtering by subagent preserves isolation."""
        record_authorization_allowed(
            subagent_id="sa-private",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        )

        record_authorization_denied(
            subagent_id="sa-other",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_store",
        )

        # Each subagent sees only its own events
        private_events = get_recorder().get_events_for_subagent("sa-private")
        other_events = get_recorder().get_events_for_subagent("sa-other")

        assert len(private_events) == 1
        assert len(other_events) == 1
        assert private_events[0].subagent_id == "sa-private"
        assert other_events[0].subagent_id == "sa-other"

    def test_parent_child_isolation(self):
        """Parent/child event collections remain isolated."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="child-1",
            parent_session_id="parent-A",
            task_type="research",
            authority_level="L0",
        ))

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="child-2",
            parent_session_id="parent-B",
            task_type="research",
            authority_level="L0",
        ))

        parent_a_events = [e for e in recorder.get_events() if e.parent_session_id == "parent-A"]
        parent_b_events = [e for e in recorder.get_events() if e.parent_session_id == "parent-B"]

        assert len(parent_a_events) == 1
        assert len(parent_b_events) == 1


# ====================================================================
# 4. Authority Integrity
# ====================================================================

class TestAuthorityIntegrity:
    """Test that authority cannot be forged in events."""

    def test_authority_level_preserved(self):
        """Authority level is preserved exactly."""
        for level in ["L0", "L1", "L2", "L3"]:
            event = create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level=level,
            )
            assert event.authority_level == level

    def test_allowed_flag_preserved(self):
        """allowed flag is preserved exactly."""
        allowed_event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            allowed=True,
        )
        assert allowed_event.allowed is True

        denied_event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            allowed=False,
        )
        assert denied_event.allowed is False

    def test_subagent_identity_cannot_be_impersonated(self):
        """Sub-agent cannot impersonate another sub-agent."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-impersonate",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        # Identity is fixed
        assert event.subagent_id == "sa-impersonate"

        # Cannot be changed (frozen)
        with pytest.raises(AttributeError):
            event.subagent_id = "sa-victim"

    def test_parent_session_cannot_be_impersonated(self):
        """Parent session ID cannot be impersonated."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent-real",
            task_type="research",
            authority_level="L0",
        )

        assert event.parent_session_id == "parent-real"

        with pytest.raises(AttributeError):
            event.parent_session_id = "parent-fake"


# ====================================================================
# 5. Event Tampering
# ====================================================================

class TestEventTampering:
    """Test that recorded events cannot be tampered with."""

    def test_recorded_events_immutable(self):
        """Recorded events cannot be edited."""
        recorder = SecurityEventRecorder()
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        recorder.record(event)
        retrieved = recorder.get_events()[0]

        with pytest.raises(AttributeError):
            retrieved.allowed = False

    def test_injection_does_not_affect_existing_events(self):
        """Arbitrary event injection cannot alter existing events."""
        recorder = SecurityEventRecorder()

        original = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-original",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )
        recorder.record(original)

        # Inject another event
        injected = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-injected",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )
        recorder.record(injected)

        # Original unchanged
        events = recorder.get_events()
        assert len(events) == 2
        sa_original = [e for e in events if e.subagent_id == "sa-original"]
        assert len(sa_original) == 1
        assert sa_original[0].event_id == original.event_id

    def test_clear_behavior_explicitly_controlled(self):
        """Clear removes all events and cannot be undone."""
        recorder = SecurityEventRecorder()

        for _ in range(5):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        assert recorder.count == 5
        recorder.clear()
        assert recorder.count == 0

    def test_retention_limit_enforced(self):
        """Retention limits cannot be bypassed."""
        recorder = SecurityEventRecorder(max_events=10)

        for _ in range(20):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        assert recorder.count == 10


# ====================================================================
# 6. Correlation Integrity
# ====================================================================

class TestCorrelationIntegrity:
    """Test that correlation fields remain consistent."""

    def test_subagent_id_consistent(self):
        """subagent_id remains consistent across all events."""
        recorder = SecurityEventRecorder()

        for event_type in [SecurityEventType.AUTHORIZATION_ALLOWED, SecurityEventType.AUTHORIZATION_DENIED]:
            recorder.record(create_security_event(
                event_type=event_type,
                subagent_id="sa-correlation",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        events = recorder.get_events_for_subagent("sa-correlation")
        assert len(events) == 2
        for e in events:
            assert e.subagent_id == "sa-correlation"

    def test_parent_session_id_consistent(self):
        """parent_session_id remains consistent."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent-consistent",
            task_type="research",
            authority_level="L0",
        )

        assert event.parent_session_id == "parent-consistent"

    def test_task_type_consistent(self):
        """task_type remains consistent."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        assert event.task_type == "research"

    def test_authority_level_consistent(self):
        """authority_level remains consistent."""
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        assert event.authority_level == "L0"

    def test_lifecycle_state_consistent(self):
        """lifecycle_state remains consistent."""
        event = create_security_event(
            event_type=SecurityEventType.LIFECYCLE_TRANSITION,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            lifecycle_state="running",
        )

        assert event.lifecycle_state == "running"


# ====================================================================
# 7. Concurrency
# ====================================================================

class TestConcurrency:
    """Test concurrent event recording safety."""

    def test_concurrent_recording_safe(self):
        """Concurrent event recording is safe."""
        recorder = SecurityEventRecorder()
        errors = []

        def record_events(subagent_id, count):
            try:
                for _ in range(count):
                    recorder.record(create_security_event(
                        event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                        subagent_id=subagent_id,
                        parent_session_id="parent",
                        task_type="research",
                        authority_level="L0",
                    ))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=record_events, args=(f"sa-{i}", 100))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert recorder.count == 1000

    def test_no_duplicate_events(self):
        """No duplicate events are created under concurrent recording."""
        recorder = SecurityEventRecorder()

        def record_event():
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        threads = [threading.Thread(target=record_event) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = recorder.get_events()
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids))

    def test_bounded_retention_under_concurrency(self):
        """Bounded retention remains enforced under concurrent recording."""
        recorder = SecurityEventRecorder(max_events=50)

        def record_events():
            for _ in range(20):
                recorder.record(create_security_event(
                    event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                    subagent_id="sa-test",
                    parent_session_id="parent",
                    task_type="research",
                    authority_level="L0",
                ))

        threads = [threading.Thread(target=record_events) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert recorder.count <= 50


# ====================================================================
# 8. Recorder Failure Isolation
# ====================================================================

class TestRecorderFailureIsolation:
    """Test that recorder failures don't affect authorization."""

    def test_recorder_exception_doesnt_block_auth(self):
        """Recorder exception doesn't block authorization."""
        ctx = create_exorder_context(
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

        with patch("plugins.memory.jarvis.subagent_security._recorder.record", side_effect=Exception("Boom")):
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                mock_dispatch.return_value = json.dumps({"ok": True})
                result = provider.handle_tool_call("hermes_rag_search", {"query": "test"})

        assert json.loads(result)["ok"] is True

    def test_recorder_failure_no_tool_execution(self):
        """Recorder failure doesn't cause tool execution."""
        ctx = create_exorder_context(
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

        with patch("plugins.memory.jarvis.subagent_security._recorder.record", side_effect=Exception("Boom")):
            result = provider.handle_tool_call("hermes_rag_store", {"text": "data"})

        # Denied despite recorder failure
        assert json.loads(result)["ok"] is False

    def test_recorder_failure_no_mutation(self):
        """Recorder failure doesn't create mutation."""
        ctx = create_exorder_context(
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

        # Denied call with recorder failure
        with patch("plugins.memory.jarvis.subagent_security._recorder.record", side_effect=Exception("Boom")):
            result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})

        assert json.loads(result)["ok"] is False


# ====================================================================
# 9. Security Sanitization
# ====================================================================

class TestSecuritySanitization:
    """Test sensitive value sanitization."""

    def test_api_key_redacted(self):
        """API keys are redacted."""
        result = _sanitize_reason("api_key=secret123")
        assert "[REDACTED]" in result
        assert "secret123" not in result

    def test_token_redacted(self):
        """Tokens are redacted."""
        result = _sanitize_reason("token=abc123")
        assert "[REDACTED]" in result

    def test_password_redacted(self):
        """Passwords are redacted."""
        result = _sanitize_reason("password=mysecret")
        assert "[REDACTED]" in result

    def test_sanitization_preserves_structure(self):
        """Sanitization doesn't corrupt event structure."""
        result = _sanitize_reason("allowed for hermes_rag_search with api_key=secret")
        assert "allowed" in result
        assert "hermes_rag_search" in result
        assert "[REDACTED]" in result

    def test_empty_reason_unchanged(self):
        """Empty reason is unchanged."""
        result = _sanitize_reason("")
        assert result == ""


# ====================================================================
# 10. Zero-Mutation Guarantees
# ====================================================================

class TestZeroMutationGuarantees:
    """Test that denied operations produce zero mutation."""

    def test_denied_auth_no_tool_execution(self):
        """Denied authorization produces no tool execution."""
        ctx = create_exorder_context(
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

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            result = provider.handle_tool_call("hermes_rag_store", {"text": "data"})
            mock.assert_not_called()

        assert json.loads(result)["ok"] is False

    def test_approval_denied_no_execution(self):
        """Approval denial produces no tool execution."""
        ctx = create_exorder_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="maintain",
            task_type="default",
            delegation_depth=1,
        )
        provider = JARVISSubagentProvider(
            goal="maintain",
            task_type="default",
            execution_context=ctx,
        )

        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": False}
            with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock_dispatch:
                result = provider.handle_tool_call("hermes_rag_delete", {"memory_id": "x"})
                mock_dispatch.assert_not_called()

        assert json.loads(result)["ok"] is False

    def test_forged_capability_no_mutation(self):
        """Forged capability produces no mutation."""
        ctx = create_exorder_context(
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

        with patch("plugins.memory.jarvis._dispatch_jarvis_tool_call") as mock:
            result = provider.handle_tool_call("hermes_fake_tool", {})
            mock.assert_not_called()

        assert json.loads(result)["ok"] is False


# ====================================================================
# Helper
# ====================================================================

def create_exorder_context(child_id, parent_session_id, task_goal, task_type, delegation_depth):
    """Helper to create execution context."""
    return create_execution_context(
        child_id=child_id,
        parent_session_id=parent_session_id,
        task_goal=task_goal,
        task_type=task_type,
        delegation_depth=delegation_depth,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
