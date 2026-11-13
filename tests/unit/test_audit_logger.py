"""Tests for audit_logger.py — audit log creation, formatting, storage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.test_execution.audit_logger import (
    AuditContext,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditQuery,
    AuditSeverity,
    AuditStorage,
    ComplianceStandard,
    setup_audit_logging,
)

# ---------------------------------------------------------------------------
# AuditContext
# ---------------------------------------------------------------------------


class TestAuditContext:
    def test_to_dict_excludes_none(self):
        ctx = AuditContext(user_id="u1", container_id="c1")
        d = ctx.to_dict()
        assert d == {"user_id": "u1", "container_id": "c1"}
        assert "session_id" not in d

    def test_to_dict_empty(self):
        ctx = AuditContext()
        assert ctx.to_dict() == {}

    def test_to_dict_all_fields(self):
        ctx = AuditContext(
            user_id="u1",
            session_id="s1",
            request_id="r1",
            container_id="c1",
            language="python",
            test_type="unit",
            execution_id="e1",
            source_ip="1.2.3.4",
            user_agent="test-agent",
            client_info={"key": "val"},
        )
        d = ctx.to_dict()
        assert d == {
            "user_id": "u1",
            "session_id": "s1",
            "request_id": "r1",
            "container_id": "c1",
            "language": "python",
            "test_type": "unit",
            "execution_id": "e1",
            "source_ip": "1.2.3.4",
            "user_agent": "test-agent",
            "client_info": {"key": "val"},
        }


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------


class TestAuditEvent:
    def _make_event(self, **overrides) -> AuditEvent:
        defaults = dict(
            event_id="evt-1",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            event_type=AuditEventType.TEST_EXECUTION_STARTED,
            severity=AuditSeverity.INFORMATIONAL,
            description="test event",
            context=AuditContext(),
        )
        defaults.update(overrides)
        return AuditEvent(**defaults)

    def test_integrity_hash_computed(self):
        event = self._make_event()
        # Manually compute the expected hash to verify correctness
        data = {
            "event_id": "evt-1",
            "timestamp": datetime(2025, 1, 1, tzinfo=UTC).isoformat(),
            "event_type": "test_execution_started",
            "severity": "informational",
            "description": "test event",
            "context": {},
            "resource_affected": None,
            "action_taken": None,
            "outcome": None,
            "additional_data": {},
        }
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        expected_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
        assert event.integrity_hash == expected_hash
        assert len(event.integrity_hash) == 64

    def test_verify_integrity_passes(self):
        event = self._make_event()
        assert event.verify_integrity() is True

    def test_verify_integrity_fails_on_tampering(self):
        event = self._make_event()
        event.description = "tampered"
        assert event.verify_integrity() is False

    def test_to_dict_roundtrip_keys(self):
        event = self._make_event(
            compliance_tags=[ComplianceStandard.SOC2],
            resource_affected="res-1",
            action_taken="logged",
            outcome="success",
            additional_data={"key": "val"},
        )
        d = event.to_dict()
        assert d["event_id"] == "evt-1"
        assert d["timestamp"] == "2025-01-01T00:00:00+00:00"
        assert d["event_type"] == "test_execution_started"
        assert d["severity"] == "informational"
        assert d["description"] == "test event"
        assert d["context"] == {}
        assert d["compliance_tags"] == ["soc2"]
        assert d["resource_affected"] == "res-1"
        assert d["action_taken"] == "logged"
        assert d["outcome"] == "success"
        assert d["additional_data"] == {"key": "val"}
        assert d["integrity_hash"] is not None

    def test_custom_integrity_hash_preserved(self):
        event = self._make_event(integrity_hash="custom-hash")
        assert event.integrity_hash == "custom-hash"
        # Custom hash should NOT match the calculated one, so verify_integrity fails
        assert event.verify_integrity() is False


# ---------------------------------------------------------------------------
# AuditQuery
# ---------------------------------------------------------------------------


class TestAuditQuery:
    def test_to_dict_defaults(self):
        q = AuditQuery()
        d = q.to_dict()
        assert d == {
            "start_time": None,
            "end_time": None,
            "event_types": [],
            "severities": [],
            "user_ids": [],
            "container_ids": [],
            "search_text": None,
            "compliance_standards": [],
            "limit": 100,
            "offset": 0,
        }

    def test_to_dict_populated(self):
        q = AuditQuery(
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            event_types=[AuditEventType.SECURITY_VIOLATION],
            severities=[AuditSeverity.HIGH],
            compliance_standards=[ComplianceStandard.GDPR],
            search_text="inject",
            limit=50,
            offset=10,
        )
        d = q.to_dict()
        assert d["start_time"] == "2025-01-01T00:00:00+00:00"
        assert d["end_time"] is None
        assert d["event_types"] == ["security_violation"]
        assert d["severities"] == ["high"]
        assert d["compliance_standards"] == ["gdpr"]
        assert d["search_text"] == "inject"
        assert d["limit"] == 50
        assert d["offset"] == 10
        assert d["user_ids"] == []
        assert d["container_ids"] == []


# ---------------------------------------------------------------------------
# AuditStorage
# ---------------------------------------------------------------------------


class TestAuditStorage:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> AuditStorage:
        return AuditStorage(tmp_path / "audit")

    def _make_event(self, ts=None, **overrides) -> AuditEvent:
        defaults = dict(
            event_id="evt-1",
            timestamp=ts or datetime.now(UTC),
            event_type=AuditEventType.CONTAINER_CREATED,
            severity=AuditSeverity.INFORMATIONAL,
            description="container created",
            context=AuditContext(container_id="c1"),
        )
        defaults.update(overrides)
        return AuditEvent(**defaults)

    def test_store_and_query(self, storage: AuditStorage):
        event = self._make_event()
        storage.store_event(event)
        results = storage.query_events(AuditQuery(limit=10))
        assert len(results) == 1
        assert results[0].event_id == "evt-1"
        assert results[0].event_type == AuditEventType.CONTAINER_CREATED
        assert results[0].severity == AuditSeverity.INFORMATIONAL
        assert results[0].description == "container created"
        assert results[0].context.container_id == "c1"

    def test_query_filter_by_event_type(self, storage: AuditStorage):
        storage.store_event(self._make_event(event_type=AuditEventType.CONTAINER_CREATED))
        storage.store_event(
            self._make_event(
                event_id="evt-2",
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.HIGH,
                description="violation",
            )
        )
        results = storage.query_events(AuditQuery(event_types=[AuditEventType.SECURITY_VIOLATION]))
        assert len(results) == 1
        assert results[0].event_type == AuditEventType.SECURITY_VIOLATION
        assert results[0].event_id == "evt-2"
        assert results[0].description == "violation"

    def test_query_filter_by_severity(self, storage: AuditStorage):
        storage.store_event(self._make_event(severity=AuditSeverity.LOW))
        storage.store_event(
            self._make_event(event_id="evt-2", severity=AuditSeverity.CRITICAL, description="crit")
        )
        results = storage.query_events(AuditQuery(severities=[AuditSeverity.CRITICAL]))
        assert len(results) == 1
        assert results[0].event_id == "evt-2"
        assert results[0].severity == AuditSeverity.CRITICAL
        assert results[0].description == "crit"

    def test_query_filter_by_user_id(self, storage: AuditStorage):
        storage.store_event(self._make_event(context=AuditContext(user_id="alice")))
        storage.store_event(self._make_event(event_id="evt-2", context=AuditContext(user_id="bob")))
        results = storage.query_events(AuditQuery(user_ids=["alice"]))
        assert len(results) == 1
        assert results[0].context.user_id == "alice"
        assert results[0].event_id == "evt-1"

    def test_query_filter_by_search_text(self, storage: AuditStorage):
        storage.store_event(self._make_event(description="connection refused"))
        storage.store_event(self._make_event(event_id="evt-2", description="all good"))
        results = storage.query_events(AuditQuery(search_text="refused"))
        assert len(results) == 1
        assert results[0].description == "connection refused"
        assert results[0].event_id == "evt-1"

    def test_query_pagination(self, storage: AuditStorage):
        for i in range(5):
            storage.store_event(self._make_event(event_id=f"evt-{i}"))
        page1 = storage.query_events(AuditQuery(limit=2, offset=0))
        assert len(page1) == 2
        page2 = storage.query_events(AuditQuery(limit=2, offset=2))
        assert len(page2) == 2
        page3 = storage.query_events(AuditQuery(limit=2, offset=4))
        assert len(page3) == 1
        # All 5 events should be accounted for
        all_ids = {e.event_id for e in page1 + page2 + page3}
        assert all_ids == {"evt-0", "evt-1", "evt-2", "evt-3", "evt-4"}

    def test_encryption_roundtrip(self, tmp_path: Path):
        enc_storage = AuditStorage(tmp_path / "enc", encryption_key="my-secret-key")
        event = self._make_event()
        enc_storage.store_event(event)
        results = enc_storage.query_events(AuditQuery(limit=10))
        assert len(results) == 1
        assert results[0].event_id == "evt-1"

    def test_decrypt_data_integrity_failure(self, tmp_path: Path):
        enc_storage = AuditStorage(tmp_path / "enc2", encryption_key="key")
        with pytest.raises(ValueError, match="integrity"):
            enc_storage._decrypt_data("badhash:some-data\n")

    def test_get_log_files_no_time_filter(self, storage: AuditStorage):
        event = self._make_event()
        storage.store_event(event)
        files = storage._get_log_files_for_query(AuditQuery())
        assert len(files) == 1
        assert files[0].name.startswith("audit_")
        assert files[0].name.endswith(".jsonl")

    def test_query_compliance_filter(self, storage: AuditStorage):
        event = self._make_event(
            compliance_tags=[ComplianceStandard.GDPR],
        )
        storage.store_event(event)
        storage.store_event(
            self._make_event(
                event_id="evt-2",
                compliance_tags=[ComplianceStandard.SOC2],
            )
        )
        results = storage.query_events(AuditQuery(compliance_standards=[ComplianceStandard.GDPR]))
        assert len(results) == 1
        assert results[0].event_id == "evt-1"
        assert results[0].compliance_tags == [ComplianceStandard.GDPR]

    def test_query_container_id_filter(self, storage: AuditStorage):
        storage.store_event(self._make_event(context=AuditContext(container_id="c1")))
        storage.store_event(
            self._make_event(event_id="evt-2", context=AuditContext(container_id="c2"))
        )
        results = storage.query_events(AuditQuery(container_ids=["c1"]))
        assert len(results) == 1
        assert results[0].context.container_id == "c1"
        assert results[0].event_id == "evt-1"

    def test_store_creates_directory(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        AuditStorage(nested)
        assert nested.exists()
        assert nested.is_dir()

    def test_store_writes_jsonl_file(self, storage: AuditStorage):
        event = self._make_event()
        storage.store_event(event)
        log_files = list(storage.storage_directory.glob("audit_*.jsonl"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        parsed = json.loads(content.strip())
        assert parsed["event_id"] == "evt-1"
        assert parsed["event_type"] == "container_created"

    def test_query_time_range_filter(self, storage: AuditStorage):
        ts_old = datetime(2025, 1, 1, tzinfo=UTC)
        ts_new = datetime(2025, 1, 2, tzinfo=UTC)
        storage.store_event(self._make_event(ts=ts_old, event_id="old"))
        storage.store_event(self._make_event(ts=ts_new, event_id="new"))
        # Query only events on or after 2025-01-02
        results = storage.query_events(AuditQuery(start_time=ts_new))
        assert len(results) == 1
        assert results[0].event_id == "new"


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------


class TestAuditLogger:
    @pytest.fixture
    def audit_logger(self, tmp_path: Path) -> AuditLogger:
        return AuditLogger(tmp_path / "audit")

    def test_log_event_returns_valid_uuid(self, audit_logger: AuditLogger):
        event_id = audit_logger.log_event(
            AuditEventType.SYSTEM_STARTUP,
            "System started",
        )
        # uuid7 returns a UUID string with hyphens — 36 chars total
        assert len(event_id) == 36
        assert event_id.count("-") == 4

    def test_log_event_stores_event_with_correct_fields(self, audit_logger: AuditLogger):
        audit_logger.log_event(
            AuditEventType.SYSTEM_STARTUP,
            "System started",
        )
        events = audit_logger.query_events(AuditQuery(limit=10))
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.SYSTEM_STARTUP
        assert events[0].description == "System started"
        assert events[0].severity == AuditSeverity.INFORMATIONAL
        # SYSTEM_STARTUP should get compliance tags from the mapping
        assert ComplianceStandard.SOC2 in events[0].compliance_tags
        assert ComplianceStandard.ISO27001 in events[0].compliance_tags

    def test_log_container_event(self, audit_logger: AuditLogger):
        event_id = audit_logger.log_container_event(
            AuditEventType.CONTAINER_CREATED,
            container_id="c1",
            language="python",
            description="Container created for python",
        )
        assert len(event_id) == 36
        events = audit_logger.query_events(AuditQuery(limit=10))
        assert len(events) == 1
        assert events[0].context.container_id == "c1"
        assert events[0].context.language == "python"
        assert events[0].description == "Container created for python"
        assert events[0].resource_affected == "c1"
        assert events[0].event_type == AuditEventType.CONTAINER_CREATED

    def test_log_test_execution_event(self, audit_logger: AuditLogger):
        event_id = audit_logger.log_test_execution_event(
            AuditEventType.TEST_EXECUTION_STARTED,
            execution_id="exec-1",
            container_id="c1",
            language="go",
            test_type="unit",
            description="Test started",
        )
        assert len(event_id) == 36
        events = audit_logger.query_events(AuditQuery(limit=10))
        assert len(events) == 1
        assert events[0].resource_affected == "execution:exec-1"
        assert events[0].context.execution_id == "exec-1"
        assert events[0].context.container_id == "c1"
        assert events[0].context.language == "go"
        assert events[0].context.test_type == "unit"
        assert events[0].description == "Test started"

    def test_log_security_event(self, audit_logger: AuditLogger):
        event_id = audit_logger.log_security_event(
            AuditEventType.CODE_INJECTION_ATTEMPT,
            description="Injection detected",
            severity=AuditSeverity.CRITICAL,
        )
        assert len(event_id) == 36
        events = audit_logger.query_events(AuditQuery(limit=10))
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.CODE_INJECTION_ATTEMPT
        assert events[0].severity == AuditSeverity.CRITICAL
        assert events[0].description == "Injection detected"
        # CODE_INJECTION_ATTEMPT maps to SOC2, ISO27001, NIST
        assert ComplianceStandard.SOC2 in events[0].compliance_tags
        assert ComplianceStandard.ISO27001 in events[0].compliance_tags
        assert ComplianceStandard.NIST in events[0].compliance_tags

    def test_audit_context_manager(self, audit_logger: AuditLogger):
        with audit_logger.audit_context(user_id="alice", session_id="s1") as ctx:
            assert ctx.user_id == "alice"
            assert ctx.session_id == "s1"
            audit_logger.log_event(
                AuditEventType.DATA_ACCESS,
                "Data accessed",
            )
        # After exiting context, user_id should not persist
        ctx2 = audit_logger._get_current_context()
        assert ctx2.user_id is None
        assert ctx2.session_id is None
        # Verify the logged event captured the context
        events = audit_logger.query_events(AuditQuery(limit=10))
        assert len(events) == 1
        assert events[0].context.user_id == "alice"
        assert events[0].context.session_id == "s1"

    def test_nested_audit_context(self, audit_logger: AuditLogger):
        with audit_logger.audit_context(user_id="alice"):
            with audit_logger.audit_context(container_id="c1") as inner:
                assert inner.user_id == "alice"
                assert inner.container_id == "c1"
            # After inner exits, outer context should still have user_id
            outer = audit_logger._get_current_context()
            assert outer.user_id == "alice"
            assert outer.container_id is None
        # After both exit, context should be empty
        empty = audit_logger._get_current_context()
        assert empty.user_id is None
        assert empty.container_id is None

    def test_compliance_mapping_loaded(self, audit_logger: AuditLogger):
        mapping = audit_logger.compliance_mapping
        # Every AuditEventType should be in the mapping
        for event_type in AuditEventType:
            assert event_type in mapping, f"{event_type} missing from compliance_mapping"
        # Spot-check specific mappings
        assert mapping[AuditEventType.CODE_INJECTION_ATTEMPT] == [
            ComplianceStandard.SOC2,
            ComplianceStandard.ISO27001,
            ComplianceStandard.NIST,
        ]
        assert mapping[AuditEventType.DATA_ACCESS] == [
            ComplianceStandard.SOC2,
            ComplianceStandard.GDPR,
            ComplianceStandard.HIPAA,
        ]

    def test_generate_compliance_report(self, audit_logger: AuditLogger):
        now = datetime.now(UTC)
        audit_logger.log_event(
            AuditEventType.CONTAINER_CREATED,
            "Container created",
        )
        audit_logger.log_event(
            AuditEventType.SECURITY_VIOLATION,
            "Violation",
            severity=AuditSeverity.HIGH,
        )
        report = audit_logger.generate_compliance_report(
            ComplianceStandard.SOC2,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )
        assert report["compliance_standard"] == "soc2"
        assert report["total_events"] == 2
        assert report["security_events"] == 1  # only the HIGH violation
        assert report["severity_breakdown"]["informational"] == 1
        assert report["severity_breakdown"]["high"] == 1
        assert report["severity_breakdown"]["low"] == 0
        assert report["event_breakdown"]["container_created"] == 1
        assert report["event_breakdown"]["security_violation"] == 1

    def test_verify_log_integrity(self, audit_logger: AuditLogger):
        audit_logger.log_event(
            AuditEventType.SYSTEM_STARTUP,
            "System started",
        )
        audit_logger.log_event(
            AuditEventType.SYSTEM_SHUTDOWN,
            "System stopped",
        )
        result = audit_logger.verify_log_integrity(hours=1)
        assert result["total_events"] == 2
        assert result["valid_events"] == 2
        assert result["invalid_events"] == 0
        assert result["integrity_percentage"] == 100.0
        assert result["time_period_hours"] == 1

    def test_log_event_severity_levels(self, audit_logger: AuditLogger):
        logged_ids = []
        for sev in AuditSeverity:
            event_id = audit_logger.log_event(
                AuditEventType.SYSTEM_STARTUP,
                f"test {sev.value}",
                severity=sev,
            )
            logged_ids.append(event_id)
        # Verify all IDs are unique valid UUIDs
        assert len(set(logged_ids)) == len(AuditSeverity)
        # Verify events are queryable and have correct severities
        events = audit_logger.query_events(AuditQuery(limit=100))
        assert len(events) == len(AuditSeverity)
        stored_severities = {e.severity for e in events}
        assert stored_severities == set(AuditSeverity)


class TestSetupAuditLogging:
    def test_setup_returns_configured_logger(self, tmp_path: Path):
        logger = setup_audit_logging(tmp_path / "audit")
        # Verify it's a working logger by logging and querying
        event_id = logger.log_event(AuditEventType.SYSTEM_STARTUP, "boot")
        assert len(event_id) == 36
        events = logger.query_events(AuditQuery(limit=10))
        assert len(events) == 1
        assert events[0].description == "boot"
        # Verify storage directory was created
        assert (tmp_path / "audit").is_dir()
