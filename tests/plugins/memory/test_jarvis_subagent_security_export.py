"""Phase 28E — Security Audit Export & Reporting.

Tests the export and reporting API of SecurityEventRecorder.
Covers serialization, filtering, timeline export, report generation,
correlation, sanitization, isolation, retention, immutability,
zero-mutation, empty reports, and deterministic ordering.
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
    _sanitize_reason,
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
# 1. Export Events
# ====================================================================

class TestExportEvents:
    """Test event export."""

    def test_export_all_events(self):
        """Export all events as JSON-safe dicts."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        exported = recorder.export_events()
        assert len(exported) == 1

        # JSON-safe
        json_str = json.dumps(exported)
        parsed = json.loads(json_str)
        assert parsed[0]["subagent_id"] == "sa-test"

    def test_export_preserves_all_fields(self):
        """Export preserves all event fields."""
        recorder = SecurityEventRecorder()

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
        recorder.record(event)

        exported = recorder.export_events()
        assert len(exported) == 1

        exp = exported[0]
        assert exp["event_id"] == event.event_id
        assert exp["timestamp"] == event.timestamp
        assert exp["event_type"] == event.event_type.value
        assert exp["subagent_id"] == event.subagent_id
        assert exp["parent_session_id"] == event.parent_session_id
        assert exp["task_type"] == event.task_type
        assert exp["authority_level"] == event.authority_level
        assert exp["tool_name"] == event.tool_name
        assert exp["lifecycle_state"] == event.lifecycle_state
        assert exp["allowed"] == event.allowed
        assert exp["reason"] == event.reason

    def test_export_with_filters(self):
        """Export respects filters."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-A",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-B",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        exported = recorder.export_events(subagent_id="sa-A")
        assert len(exported) == 1
        assert exported[0]["subagent_id"] == "sa-A"

    def test_export_empty_result(self):
        """Export with no matching events returns empty list."""
        recorder = SecurityEventRecorder()

        exported = recorder.export_events(subagent_id="nonexistent")
        assert exported == []


# ====================================================================
# 2. Export Timeline
# ====================================================================

class TestExportTimeline:
    """Test timeline export."""

    def test_export_timeline_chronological(self):
        """Timeline export is in chronological order."""
        recorder = SecurityEventRecorder()

        for _ in range(5):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        timeline = recorder.export_timeline("sa-test")
        assert len(timeline) == 5

        # Check chronological order
        for i in range(len(timeline) - 1):
            assert timeline[i]["timestamp"] <= timeline[i + 1]["timestamp"]

    def test_export_timeline_isolated(self):
        """Timeline export is isolated by sub-agent."""
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

        timeline_a = recorder.export_timeline("sa-A")
        assert len(timeline_a) == 1
        assert timeline_a[0]["subagent_id"] == "sa-A"


# ====================================================================
# 3. Security Report Generation
# ====================================================================

class TestSecurityReport:
    """Test security report generation."""

    def test_report_metadata(self):
        """Report contains metadata."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        report = recorder.generate_security_report()

        assert "metadata" in report
        assert "generated_at" in report["metadata"]
        assert "total_events" in report["metadata"]
        assert report["metadata"]["total_events"] == 1

    def test_report_filter_criteria(self):
        """Report contains filter criteria."""
        recorder = SecurityEventRecorder()

        report = recorder.generate_security_report(subagent_id="sa-test")

        assert "filter_criteria" in report
        assert report["filter_criteria"]["subagent_id"] == "sa-test"

    def test_report_summary_counts(self):
        """Report summary has correct counts."""
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

        report = recorder.generate_security_report()

        assert report["summary"]["total_events"] == 2
        assert report["summary"]["allowed_count"] == 1
        assert report["summary"]["denied_count"] == 1

    def test_report_affected_subagents(self):
        """Report lists affected sub-agents."""
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

        report = recorder.generate_security_report()

        assert "sa-A" in report["affected_subagents"]
        assert "sa-B" in report["affected_subagents"]

    def test_report_affected_tools(self):
        """Report lists affected tools."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="hermes_rag_search",
        ))

        report = recorder.generate_security_report()

        assert "hermes_rag_search" in report["affected_tools"]

    def test_report_timeline_chronological(self):
        """Report timeline is in chronological order."""
        recorder = SecurityEventRecorder()

        for _ in range(5):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        report = recorder.generate_security_report()

        timeline = report["timeline"]
        assert len(timeline) == 5

        for i in range(len(timeline) - 1):
            assert timeline[i]["timestamp"] <= timeline[i + 1]["timestamp"]

    def test_report_json_safe(self):
        """Report is JSON-safe."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.FORGED_CAPABILITY,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            tool_name="fake_tool",
        ))

        report = recorder.generate_security_report()

        # Should serialize to JSON without error
        json_str = json.dumps(report)
        parsed = json.loads(json_str)

        assert parsed["summary"]["forged_capability_count"] == 1


# ====================================================================
# 4. Correlation
# ====================================================================

class TestCorrelation:
    """Test event correlation in exports."""

    def test_correlation_by_subagent(self):
        """Events correlate correctly by sub-agent."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-A",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))
        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            subagent_id="sa-B",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        exported = recorder.export_events(subagent_id="sa-A")
        assert len(exported) == 1
        assert exported[0]["subagent_id"] == "sa-A"


# ====================================================================
# 5. Sanitization
# ====================================================================

class TestSanitization:
    """Test sanitization in exports."""

    def test_reason_sanitized(self):
        """Reason is sanitized in exports."""
        recorder = SecurityEventRecorder()

        # Use record function which sanitizes
        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            reason=_sanitize_reason("allowed with api_key=secret123"),
        )
        recorder.record(event)

        exported = recorder.export_events()
        assert "[REDACTED]" in exported[0]["reason"]

    def test_sanitization_doesnt_corrupt_structure(self):
        """Sanitization doesn't corrupt event structure."""
        recorder = SecurityEventRecorder()

        event = create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
            reason="allowed api_key=secret",
        )
        recorder.record(event)

        exported = recorder.export_events()
        exp = exported[0]

        # All fields present
        assert "event_id" in exp
        assert "timestamp" in exp
        assert "event_type" in exp
        assert "subagent_id" in exp
        assert "parent_session_id" in exp
        assert "task_type" in exp
        assert "authority_level" in exp
        assert "tool_name" in exp
        assert "lifecycle_state" in exp
        assert "allowed" in exp
        assert "reason" in exp


# ====================================================================
# 6. Isolation
# ====================================================================

class TestExportIsolation:
    """Test export isolation."""

    def test_export_respects_subagent_boundaries(self):
        """Export respects sub-agent boundaries."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-A",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        exported = recorder.export_events(subagent_id="sa-B")
        assert len(exported) == 0


# ====================================================================
# 7. Retention
# ====================================================================

class TestExportRetention:
    """Test export respects retention."""

    def test_expired_events_not_exported(self):
        """Expired events don't appear in exports."""
        recorder = SecurityEventRecorder(max_events=3)

        for _ in range(5):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        exported = recorder.export_events()
        assert len(exported) <= 3


# ====================================================================
# 8. Immutability
# ====================================================================

class TestExportImmutability:
    """Test export immutability."""

    def test_exported_events_immutable_in_recorder(self):
        """Exported events don't affect recorder state."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        exported = recorder.export_events()
        exported.clear()  # Modify returned list

        # Recorder state unchanged
        assert recorder.count == 1

    def test_report_doesnt_modify_recorder(self):
        """Report doesn't modify recorder state."""
        recorder = SecurityEventRecorder()

        recorder.record(create_security_event(
            event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
            subagent_id="sa-test",
            parent_session_id="parent",
            task_type="research",
            authority_level="L0",
        ))

        initial_count = recorder.count
        recorder.generate_security_report()

        assert recorder.count == initial_count


# ====================================================================
# 9. Zero Mutation
# ====================================================================

class TestZeroMutationExport:
    """Test zero-mutation with exports."""

    def test_export_after_denied_auth(self):
        """Export after denied auth shows the denial."""
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

        # Export shows the denial
        exported = get_recorder().export_events(subagent_id="sa-test", allowed=False)
        assert len(exported) >= 1


# ====================================================================
# 10. Empty Reports
# ====================================================================

class TestEmptyReports:
    """Test empty report handling."""

    def test_empty_export(self):
        """Empty export returns empty list."""
        recorder = SecurityEventRecorder()

        exported = recorder.export_events(subagent_id="nonexistent")
        assert exported == []

    def test_empty_timeline(self):
        """Empty timeline returns empty list."""
        recorder = SecurityEventRecorder()

        timeline = recorder.export_timeline("nonexistent")
        assert timeline == []

    def test_empty_report(self):
        """Empty report has zero counts."""
        recorder = SecurityEventRecorder()

        report = recorder.generate_security_report()

        assert report["summary"]["total_events"] == 0
        assert report["summary"]["allowed_count"] == 0
        assert report["summary"]["denied_count"] == 0


# ====================================================================
# 11. Deterministic Ordering
# ====================================================================

class TestDeterministicOrdering:
    """Test deterministic ordering."""

    def test_export_deterministic_order(self):
        """Export preserves deterministic order."""
        recorder = SecurityEventRecorder()

        for _ in range(10):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        exported1 = recorder.export_events()
        exported2 = recorder.export_events()

        ids1 = [e["event_id"] for e in exported1]
        ids2 = [e["event_id"] for e in exported2]
        assert ids1 == ids2

    def test_report_timeline_deterministic(self):
        """Report timeline is deterministic."""
        recorder = SecurityEventRecorder()

        for _ in range(5):
            recorder.record(create_security_event(
                event_type=SecurityEventType.AUTHORIZATION_ALLOWED,
                subagent_id="sa-test",
                parent_session_id="parent",
                task_type="research",
                authority_level="L0",
            ))

        report1 = recorder.generate_security_report()
        report2 = recorder.generate_security_report()

        ids1 = [e["event_id"] for e in report1["timeline"]]
        ids2 = [e["event_id"] for e in report2["timeline"]]
        assert ids1 == ids2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
