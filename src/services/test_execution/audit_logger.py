"""
Audit logging system for security and compliance monitoring.

This module provides comprehensive audit logging for all test execution activities,
including security events, compliance tracking, and forensic analysis capabilities.
"""

import hashlib
import hmac
import json
import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid7

logger = logging.getLogger(__name__)


class AuditEventType(StrEnum):
    """Types of audit events."""

    CONTAINER_CREATED = "container_created"
    CONTAINER_STARTED = "container_started"
    CONTAINER_STOPPED = "container_stopped"
    CONTAINER_DESTROYED = "container_destroyed"
    TEST_EXECUTION_STARTED = "test_execution_started"
    TEST_EXECUTION_COMPLETED = "test_execution_completed"
    TEST_EXECUTION_FAILED = "test_execution_failed"
    CODE_INJECTION_ATTEMPT = "code_injection_attempt"
    SECURITY_VIOLATION = "security_violation"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    CONFIGURATION_CHANGED = "configuration_changed"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_DENIED = "authorization_denied"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_STARTUP = "system_startup"


class AuditSeverity(StrEnum):
    """Severity levels for audit events."""

    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStandard(StrEnum):
    """Compliance standards for audit logging."""

    SOC2 = "soc2"
    ISO27001 = "iso27001"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    NIST = "nist"


@dataclass
class AuditContext:
    """Context information for audit events."""

    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    container_id: str | None = None
    language: str | None = None
    test_type: str | None = None
    execution_id: str | None = None
    source_ip: str | None = None
    user_agent: str | None = None
    client_info: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class AuditEvent:
    """Individual audit event with comprehensive metadata."""

    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    description: str
    context: AuditContext
    resource_affected: str | None = None
    action_taken: str | None = None
    outcome: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)
    compliance_tags: list[ComplianceStandard] = field(default_factory=list)
    integrity_hash: str | None = None

    def __post_init__(self) -> None:
        """Calculate integrity hash after initialization."""
        if not self.integrity_hash:
            self.integrity_hash = self._calculate_integrity_hash()

    def _calculate_integrity_hash(self) -> str:
        """Calculate integrity hash for tamper detection."""
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "context": self.context.to_dict(),
            "resource_affected": self.resource_affected,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "additional_data": self.additional_data,
        }
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event integrity using stored hash."""
        calculated_hash = self._calculate_integrity_hash()
        return calculated_hash == self.integrity_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "context": self.context.to_dict(),
            "resource_affected": self.resource_affected,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "additional_data": self.additional_data,
            "compliance_tags": [tag.value for tag in self.compliance_tags],
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class AuditQuery:
    """Query parameters for audit log search."""

    start_time: datetime | None = None
    end_time: datetime | None = None
    event_types: list[AuditEventType] = field(default_factory=list)
    severities: list[AuditSeverity] = field(default_factory=list)
    user_ids: list[str] = field(default_factory=list)
    container_ids: list[str] = field(default_factory=list)
    search_text: str | None = None
    compliance_standards: list[ComplianceStandard] = field(default_factory=list)
    limit: int = 100
    offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "event_types": [event_type.value for event_type in self.event_types],
            "severities": [severity.value for severity in self.severities],
            "user_ids": self.user_ids,
            "container_ids": self.container_ids,
            "search_text": self.search_text,
            "compliance_standards": [std.value for std in self.compliance_standards],
            "limit": self.limit,
            "offset": self.offset,
        }


class AuditStorage:
    """Secure storage for audit events."""

    def __init__(self, storage_directory: Path, encryption_key: str | None = None):
        """Initialize the AuditStorage service.

        This initializes the AuditStorage with required dependencies
        and configuration for service operations.

        Args:
            storage_directory (Path): Storage directory
            encryption_key (Optional[str]): Encryption key
        """
        self.storage_directory = Path(storage_directory)
        self.encryption_key = encryption_key
        self.lock = threading.Lock()
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        self.current_log_file = None
        self.current_log_date = None

    def store_event(self, event: AuditEvent) -> None:
        """Store audit event securely."""
        with self.lock:
            log_file = self._get_log_file(event.timestamp)
            event_data = event.to_dict()
            log_line = json.dumps(event_data, separators=(",", ":"))
            if self.encryption_key:
                log_line = self._encrypt_data(log_line)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
                f.flush()

    def _get_log_file(self, timestamp: datetime) -> Path:
        """Get log file path for timestamp."""
        date_str = timestamp.strftime("%Y-%m-%d")
        return self.storage_directory / f"audit_{date_str}.jsonl"

    def _encrypt_data(self, data: str) -> str:
        """Encrypt data using provided key."""
        if self.encryption_key:
            key_bytes = self.encryption_key.encode("utf-8")
            data_bytes = data.encode("utf-8")
            mac = hmac.new(key_bytes, data_bytes, hashlib.sha256).hexdigest()
            return f"{mac}:{data}"
        return data

    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using provided key."""
        if self.encryption_key and ":" in encrypted_data:
            mac, data = encrypted_data.split(":", 1)
            key_bytes = self.encryption_key.encode("utf-8")
            data_bytes = data.encode("utf-8")
            expected_mac = hmac.new(key_bytes, data_bytes, hashlib.sha256).hexdigest()
            if hmac.compare_digest(mac, expected_mac):
                return data
            else:
                raise ValueError("Data integrity check failed")
        return encrypted_data

    def query_events(self, query: AuditQuery) -> list[AuditEvent]:
        """Query audit events based on criteria."""
        events = []
        log_files = self._get_log_files_for_query(query)
        for log_file in log_files:
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            if self.encryption_key:
                                line = self._decrypt_data(line.strip())
                            event_data = json.loads(line.strip())
                            event = self._dict_to_event(event_data)
                            if self._event_matches_query(event, query):
                                events.append(event)
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Failed to parse audit log line: {e}")
                            continue
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"Error reading audit log file {log_file}: {e}")
        events.sort(key=lambda x: x.timestamp, reverse=True)
        start_idx = query.offset
        end_idx = start_idx + query.limit
        return events[start_idx:end_idx]

    def _get_log_files_for_query(self, query: AuditQuery) -> list[Path]:
        """Get log files relevant to query time range."""
        all_files = list(self.storage_directory.glob("audit_*.jsonl"))
        if not query.start_time and (not query.end_time):
            return sorted(all_files)
        filtered_files = []
        for file_path in all_files:
            try:
                date_str = file_path.stem.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if query.start_time and file_date < query.start_time.date():
                    continue
                if query.end_time and file_date > query.end_time.date():
                    continue
                filtered_files.append(file_path)
            except (ValueError, IndexError):  # fmt: skip
                filtered_files.append(file_path)
        return sorted(filtered_files)

    def _dict_to_event(self, event_data: dict[str, Any]) -> AuditEvent:
        """Convert dictionary to AuditEvent object."""
        context = AuditContext(**event_data.get("context", {}))
        return AuditEvent(
            event_id=event_data["event_id"],
            timestamp=datetime.fromisoformat(event_data["timestamp"]),
            event_type=AuditEventType(event_data["event_type"]),
            severity=AuditSeverity(event_data["severity"]),
            description=event_data["description"],
            context=context,
            resource_affected=event_data.get("resource_affected"),
            action_taken=event_data.get("action_taken"),
            outcome=event_data.get("outcome"),
            additional_data=event_data.get("additional_data", {}),
            compliance_tags=[
                ComplianceStandard(tag) for tag in event_data.get("compliance_tags", [])
            ],
            integrity_hash=event_data.get("integrity_hash"),
        )

    def _event_matches_query(self, event: AuditEvent, query: AuditQuery) -> bool:
        """Check if event matches query criteria."""
        if query.start_time and event.timestamp < query.start_time:
            return False
        if query.end_time and event.timestamp > query.end_time:
            return False
        if query.event_types and event.event_type not in query.event_types:
            return False
        if query.severities and event.severity not in query.severities:
            return False
        if query.user_ids and event.context.user_id not in query.user_ids:
            return False
        if query.container_ids and event.context.container_id not in query.container_ids:
            return False
        if query.search_text:
            searchable_text = f"{event.description} {json.dumps(event.additional_data)}"
            if query.search_text.lower() not in searchable_text.lower():
                return False
        if query.compliance_standards:
            if not any(std in event.compliance_tags for std in query.compliance_standards):
                return False
        return True


class AuditLogger:
    """Main audit logging service."""

    def __init__(self, storage_directory: Path, encryption_key: str | None = None):
        """Initialize the AuditLogger service.

        This initializes the AuditLogger with required dependencies
        and configuration for service operations.

        Args:
            storage_directory (Path): Storage directory
            encryption_key (Optional[str]): Encryption key
        """
        self.storage = AuditStorage(storage_directory, encryption_key)
        self.context_stack = threading.local()
        self.compliance_mapping = self._load_compliance_mapping()

    def _load_compliance_mapping(
        self,
    ) -> dict[AuditEventType, list[ComplianceStandard]]:
        """Load compliance standard mappings for event types."""
        return {
            AuditEventType.CONTAINER_CREATED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.CONTAINER_STARTED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.CONTAINER_STOPPED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.CONTAINER_DESTROYED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.TEST_EXECUTION_STARTED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.NIST,
            ],
            AuditEventType.TEST_EXECUTION_COMPLETED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.NIST,
            ],
            AuditEventType.TEST_EXECUTION_FAILED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.NIST,
            ],
            AuditEventType.CODE_INJECTION_ATTEMPT: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
                ComplianceStandard.NIST,
            ],
            AuditEventType.SECURITY_VIOLATION: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
                ComplianceStandard.NIST,
            ],
            AuditEventType.RESOURCE_LIMIT_EXCEEDED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.UNAUTHORIZED_ACCESS: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
                ComplianceStandard.NIST,
            ],
            AuditEventType.CONFIGURATION_CHANGED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.DATA_ACCESS: [
                ComplianceStandard.SOC2,
                ComplianceStandard.GDPR,
                ComplianceStandard.HIPAA,
            ],
            AuditEventType.DATA_MODIFICATION: [
                ComplianceStandard.SOC2,
                ComplianceStandard.GDPR,
                ComplianceStandard.HIPAA,
            ],
            AuditEventType.AUTHENTICATION_SUCCESS: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.AUTHENTICATION_FAILURE: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.AUTHORIZATION_DENIED: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.SYSTEM_SHUTDOWN: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
            AuditEventType.SYSTEM_STARTUP: [
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ],
        }

    @contextmanager
    def audit_context(self, **context_kwargs) -> Generator[AuditContext]:
        """Context manager for setting audit context."""
        if not hasattr(self.context_stack, "contexts"):
            self.context_stack.contexts = []
        current_context = self._get_current_context()
        merged_context = {**current_context.to_dict(), **context_kwargs}
        new_context = AuditContext(**merged_context)
        self.context_stack.contexts.append(new_context)
        try:
            yield new_context
        finally:
            self.context_stack.contexts.pop()

    def _get_current_context(self) -> AuditContext:
        """Get current audit context."""
        if not hasattr(self.context_stack, "contexts"):
            self.context_stack.contexts = []
        if self.context_stack.contexts:
            context = self.context_stack.contexts[-1]
            assert isinstance(context, AuditContext), "Context must be AuditContext instance"
            return context
        return AuditContext()

    def log_event(
        self,
        event_type: AuditEventType,
        description: str,
        severity: AuditSeverity = AuditSeverity.INFORMATIONAL,
        resource_affected: str | None = None,
        action_taken: str | None = None,
        outcome: str | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> str:
        """Log an audit event."""
        event_id = str(uuid7())
        compliance_tags = self.compliance_mapping.get(event_type, [])
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(UTC),
            event_type=event_type,
            severity=severity,
            description=description,
            context=self._get_current_context(),
            resource_affected=resource_affected,
            action_taken=action_taken,
            outcome=outcome,
            additional_data=additional_data or {},
            compliance_tags=compliance_tags,
        )
        self.storage.store_event(event)
        log_level = {
            AuditSeverity.INFORMATIONAL: logging.INFO,
            AuditSeverity.LOW: logging.INFO,
            AuditSeverity.MEDIUM: logging.WARNING,
            AuditSeverity.HIGH: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }[severity]
        logger.log(log_level, f"AUDIT: {event_type.value} - {description}")
        return event_id

    def log_container_event(
        self,
        event_type: AuditEventType,
        container_id: str,
        language: str,
        description: str,
        **kwargs,
    ) -> str:
        """Log container-related audit event."""
        with self.audit_context(container_id=container_id, language=language):
            return self.log_event(
                event_type=event_type,
                description=description,
                resource_affected=container_id,
                **kwargs,
            )

    def log_test_execution_event(
        self,
        event_type: AuditEventType,
        execution_id: str,
        container_id: str,
        language: str,
        test_type: str,
        description: str,
        **kwargs,
    ) -> str:
        """Log test execution audit event."""
        with self.audit_context(
            execution_id=execution_id,
            container_id=container_id,
            language=language,
            test_type=test_type,
        ):
            return self.log_event(
                event_type=event_type,
                description=description,
                resource_affected=f"execution:{execution_id}",
                **kwargs,
            )

    def log_security_event(
        self,
        event_type: AuditEventType,
        description: str,
        severity: AuditSeverity = AuditSeverity.HIGH,
        **kwargs,
    ) -> str:
        """Log security-related audit event."""
        return self.log_event(
            event_type=event_type, description=description, severity=severity, **kwargs
        )

    def query_events(self, query: AuditQuery) -> list[AuditEvent]:
        """Query audit events."""
        return self.storage.query_events(query)

    def generate_compliance_report(
        self, standard: ComplianceStandard, start_time: datetime, end_time: datetime
    ) -> dict[str, Any]:
        """Generate compliance report for specified standard."""
        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            compliance_standards=[standard],
            limit=10000,
        )
        events = self.query_events(query)
        event_counts: dict[str, int] = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        total_events = len(events)
        security_events = len(
            [e for e in events if e.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]]
        )
        return {
            "compliance_standard": standard.value,
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "total_events": total_events,
            "security_events": security_events,
            "event_breakdown": event_counts,
            "severity_breakdown": {
                severity.value: len([e for e in events if e.severity == severity])
                for severity in AuditSeverity
            },
            "containers_affected": len(
                set(e.context.container_id for e in events if e.context.container_id)
            ),
            "languages_involved": list(
                set(e.context.language for e in events if e.context.language)
            ),
        }

    def verify_log_integrity(self, hours: int = 24) -> dict[str, Any]:
        """Verify integrity of audit logs."""
        start_time = datetime.now(UTC) - timedelta(hours=hours)
        query = AuditQuery(start_time=start_time, limit=10000)
        events = self.query_events(query)
        total_events = len(events)
        valid_events = 0
        invalid_events = 0
        for event in events:
            if event.verify_integrity():
                valid_events += 1
            else:
                invalid_events += 1
        return {
            "verification_timestamp": datetime.now(UTC).isoformat(),
            "time_period_hours": hours,
            "total_events": total_events,
            "valid_events": valid_events,
            "invalid_events": invalid_events,
            "integrity_percentage": (valid_events / total_events * 100 if total_events > 0 else 0),
        }


_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        import os

        log_dir = Path(os.getenv("AUDIT_LOG_DIR", "/var/log/test_execution/audit"))
        encryption_key = os.getenv("AUDIT_ENCRYPTION_KEY")
        _audit_logger = AuditLogger(log_dir, encryption_key)
    return _audit_logger


def setup_audit_logging(log_directory: Path, encryption_key: str | None = None) -> AuditLogger:
    """Setup audit logging with custom configuration."""
    global _audit_logger
    _audit_logger = AuditLogger(log_directory, encryption_key)
    return _audit_logger


def log_audit_event(event_type: AuditEventType, description: str, **kwargs) -> str:
    """Convenience function to log audit event."""
    return get_audit_logger().log_event(event_type, description, **kwargs)


def log_container_audit(
    event_type: AuditEventType,
    container_id: str,
    language: str,
    description: str,
    **kwargs,
) -> str:
    """Convenience function to log container audit event."""
    return get_audit_logger().log_container_event(
        event_type, container_id, language, description, **kwargs
    )


def log_security_audit(event_type: AuditEventType, description: str, **kwargs) -> str:
    """Convenience function to log security audit event."""
    return get_audit_logger().log_security_event(event_type, description, **kwargs)
