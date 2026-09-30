"""Phase 28D — Security Audit Query & Investigation.

Tests the query and investigation API of SecurityEventRecorder.
Covers filtering, timeline reconstruction, correlation, investigation,
isolation, immutability, retention, and read-only guarantees.
"""

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
    record_tool_execution_blocked,
    record_tool_execution_completed,
    record_tool_execution_failed,
    record_approval_requested,
    record_approval_denied,
    record_lifecycle_transition,
    record_recursion_limit,
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
# 1. Event Filtering
# ====================================================================

class TestEventFiltering:
    """Test event filtering."""

    def test_filter_by_subagent_id(self):
        """Filter by subagent_id returns only matching events."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-A",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-B",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        result = recorder.query(subagent_id="sa-A")
        assert len(result) == 1
        assert result[0].subagent_id == "sa-A"

    def test_filter_by_parent_session_id(self):
        """Filter by parent_session_id."""
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

        result = recorder.query(parent_session_id="parent-A")
        assert len(result) == 1
        assert result[0].parent_session_id == "parent-A"

    def test_filter_by_event_type(self):
        """Filter by event_type."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        result = recorder.query(event_type=SecurityEventType.AUTHORIZATION_DENIED)
        assert len(result) == 1
        assert result[0].event_type == SecurityEventType.AUTHORIZATION_DENIED

    def test_filter_by_tool_name(self):
        """Filter by tool_name."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_store",
        ))

        result = recorder.query(tool_name="hermes_rag_search")
        assert len(result) == 1
        assert result[0].tool_name == "hermes_rag_search"

    def test_filter_by_task_type(self):
        """Filter by task_type."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="maintenance",
            authority_level="L3",
        ))

        result = recorder.query(task_type="maintenance")
        assert len(result) == 1
        assert result[0].task_type == "maintenance"

    def test_filter_by_lifecycle_state(self):
        """Filter by lifecycle_state."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.LIFECYCLE_TRANSITION,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            lifecycle_state="running",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.LIFECYCLE_TRANSITION,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            lifecycle_state="completed",
        ))

        result = recorder.query(lifecycle_state="running")
        assert len(result) == 1
        assert result[0].lifecycle_state == "running"

    def test_filter_by_allowed(self):
        """Filter by allowed/denied."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            allowed=True,
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            allowed=False,
        ))

        denied = recorder.query(allowed=False)
        assert len(denied) == 1
        assert denied[0].allowed is False

    def test_combined_filters(self):
        """Multiple filters combined."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-A",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-A",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_store",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-B",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        ))

        result = recorder.query(subagent_id="sa-A", event_type=SecurityEventType.AUTHORIZATION_ALLOWED)
        assert len(result) == 1
        assert result[0].subagent_id == "sa-A"
        assert result[0].event_type == SecurityEventType.AUTHORIZATION_ALLOWED


# ====================================================================
# 2. Timeline Reconstruction
# ====================================================================

class TestTimelineReconstruction:
    """Test timeline reconstruction."""

    def test_timeline_chronological_order(self):
        """Timeline is in chronological order."""
        recorder = SecurityEventRecorder()

        for i in range(5):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        timeline = recorder.get_timeline("sa-test")
        assert len(timeline) == 5

        # Check chronological order
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp

    def test_timeline_no_duplicates(self):
        """Timeline has no duplicate events."""
        recorder = SecurityEventRecorder()

        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        )

        # Record same event multiple times
        recorder.record(event)
        recorder.record(event)
        recorder.record(event)

        timeline = recorder.get_timeline("sa-test")
        assert len(timeline) == 1

    def test_timeline_deterministic_ordering(self):
        """Equal timestamps are handled deterministically."""
        recorder = SecurityEventRecorder()

        for _ in range(10):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        timeline1 = recorder.get_timeline("sa-test")
        timeline2 = recorder.get_timeline("sa-test")

        ids1 = [e.event_id for e in timeline1]
        ids2 = [e.event_id for e in timeline2]
        assert ids1 == ids2

    def test_timeline_isolated_by_subagent(self):
        """Timeline is isolated by sub-agent."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-A",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-B",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        timeline_a = recorder.get_timeline("sa-A")
        assert len(timeline_a) == 1
        assert timeline_a[0].subagent_id == "sa-A"


# ====================================================================
# 3. Execution Correlation
# ====================================================================

class TestExecutionCorrelation:
    """Test execution correlation."""

    def test_reconstruct_child_history(self):
        """Reconstruct one child's complete history."""
        recorder = SecurityEventRecorder()

        events = [
            SecurityEventType.AUTHORIZATION_ALLOWED,
            SecurityEventType.TOOL_EXECUTION_COMPLETED,
            SecurityEventType.LIFECYCLE_TRANSITION,
            SecurityEventType.TOOL_EXECUTION_FAILED,
        ]

        for event_type in events:
            recorder.record(create_security_event(
                event_type=event_type,
                subagent_id="sa-child",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        timeline = recorder.get_timeline("sa-child")
        assert len(timeline) == 4

    def test_concurrent_children_separate(self):
        """Concurrent children remain separate."""
        recorder = SecurityEventRecorder()

        for i in range(5):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-1",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        for i in range(3):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-2",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        assert len(recorder.get_timeline("sa-1")) == 5
        assert len(recorder.get_timeline("sa-2")) == 3

    def test_parent_child_identity_preserved(self):
        """Parent/child identity preserved."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="child-1",
            parent_session_id="parent-1",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="child-2",
            parent_session_id="parent-2",
            task_type="research",
            authority_level="L0",
        ))

        child1_events = recorder.query(subagent_id="child-1")
        assert len(child1_events) == 1
        assert child1_events[0].parent_session_id == "parent-1"


# ====================================================================
# 4. Security Investigation
# ====================================================================

class TestSecurityInvestigation:
    """Test security investigation queries."""

    def test_get_denied_events(self):
        """Retrieve denied authorization events."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            allowed=True,
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            allowed=False,
        ))

        denied = recorder.get_denied_events()
        assert len(denied) == 1
        assert denied[0].allowed is False

    def test_get_forged_capability_events(self):
        """Retrieve forged capability events."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.FORGED_CAPABILITY,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="fake_tool",
        ))

        forged = recorder.get_forged_capability_events()
        assert len(forged) == 1
        assert forged[0].event_type == SecurityEventType.FORGED_CAPABILITY

    def test_get_approval_events(self):
        """Retrieve approval events."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.APPROVAL_REQUESTED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="default",
            authority_level="L3",
            tool_name="hermes_rag_delete",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.APPROVAL_DENIED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="default",
            authority_level="L3",
            tool_name="hermes_rag_delete",
        ))

        approval = recorder.get_approval_events()
        assert len(approval) == 2

    def test_get_execution_failures(self):
        """Retrieve execution failures."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.TOOL_EXECUTION_FAILED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
            reason="Qdrant timeout",
        ))

        failures = recorder.get_execution_failures()
        assert len(failures) == 1
        assert failures[0].event_type == SecurityEventType.TOOL_EXECUTION_FAILED

    def test_get_security_investigation_events(self):
        """Retrieve all security investigation events."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.FORGED_CAPABILITY,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        investigation = recorder.get_security_investigation_events()
        assert len(investigation) == 2


# ====================================================================
# 5. Isolation
# ====================================================================

class TestQueryIsolation:
    """Test query isolation."""

    def test_child_cannot_access_other_child_events(self):
        """Child cannot access another child's events."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-private",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        result = recorder.query(subagent_id="sa-other")
        assert len(result) == 0

    def test_query_respects_subagent_boundaries(self):
        """Query respects sub-agent boundaries."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-A",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        # Query for different subagent returns empty
        result = recorder.query(subagent_id="sa-B")
        assert len(result) == 0

        # Query for correct subagent returns events
        result = recorder.query(subagent_id="sa-A")
        assert len(result) == 1


# ====================================================================
# 6. Immutability
# ====================================================================

class TestQueryImmutability:
    """Test query result immutability."""

    def test_returned_events_immutable(self):
        """Returned events cannot mutate recorder state."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        events = recorder.query()
        events.clear()  # Modify returned collection

        # Recorder state unchanged
        assert recorder.count == 1

    def test_returned_collection_isolation(self):
        """Returned collections cannot mutate recorder state."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        events = recorder.get_timeline("sa-test")
        original_count = len(events)
        events.pop()  # Remove from returned list

        # Recorder still has original event
        assert recorder.count == 1
        assert len(recorder.get_timeline("sa-test")) == original_count


# ====================================================================
# 7. Retention
# ====================================================================

class TestQueryRetention:
    """Test query respects retention."""

    def test_expired_events_not_in_timeline(self):
        """Expired events don't appear in timeline."""
        recorder = SecurityEventRecorder(max_events=3)

        for _ in range(5):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        timeline = recorder.get_timeline("sa-test")
        assert len(timeline) <= 3


# ====================================================================
# 8. Invalid/Empty Queries
# ====================================================================

class TestInvalidEmptyQueries:
    """Test invalid and empty query handling."""

    def test_empty_query_returns_all(self):
        """Empty query returns all events."""
        recorder = SecurityEventRecorder()

        for _ in range(3):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        result = recorder.query()
        assert len(result) == 3

    def test_nonexistent_subagent_returns_empty(self):
        """Nonexistent subagent returns empty list."""
        recorder = SecurityEventRecorder()

        result = recorder.query(subagent_id="nonexistent")
        assert result == []

    def test_invalid_filter_values_safe(self):
        """Invalid filter values handled safely."""
        recorder = SecurityEventRecorder()

        # Should not crash
        result = recorder.query(subagent_id="")
        assert isinstance(result, list)

        result = recorder.query(tool_name="nonexistent_tool")
        assert isinstance(result, list)

    def test_no_exception_leakage(self):
        """No information leakage through exceptions."""
        recorder = SecurityEventRecorder()

        # Should never raise
        try:
            result = recorder.query(subagent_id="test")
            assert isinstance(result, list)
        except Exception:
            pytest.fail("Query raised exception")


# ====================================================================
# 9. Read-Only Guarantee
# ====================================================================

class TestReadOnlyGuarantee:
    """Test that query layer is read-only."""

    def test_query_doesnt_modify_recorder(self):
        """Query doesn't modify recorder state."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        initial_count = recorder.count

        # Multiple queries
        recorder.query()
        recorder.query(subagent_id="sa-test")
        recorder.get_timeline("sa-test")
        recorder.get_denied_events()

        assert recorder.count == initial_count

    def test_timeline_doesnt_modify_recorder(self):
        """Timeline doesn't modify recorder state."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        recorder.get_timeline("sa-test")
        recorder.get_timeline("sa-test")
        recorder.get_timeline("sa-test")

        assert recorder.count == 1


# ====================================================================
# 10. Zero-Mutation Guarantees
# ====================================================================

class TestZeroMutationQuery:
    """Test zero-mutation with queries."""

    def test_query_after_denied_auth(self):
        """Query after denied auth shows the denial."""
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

        # Query shows the denial
        events = get_recorder().query(subagent_id="sa-test", allowed=False)
        assert len(events) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
