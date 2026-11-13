"""
Error tracking and failure analysis logging for test execution infrastructure.

This module provides comprehensive error tracking, failure pattern analysis,
and automated root cause investigation for container-based test execution.
"""

import json
import logging
import re
import threading
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorSeverity(StrEnum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(StrEnum):
    """Error categories for classification."""

    CONTAINER_RUNTIME = "container_runtime"
    LANGUAGE_COMPILATION = "language_compilation"
    TEST_EXECUTION = "test_execution"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_CONNECTIVITY = "network_connectivity"
    FILESYSTEM_ACCESS = "filesystem_access"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    SECURITY_VIOLATION = "security_violation"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class FailurePattern(StrEnum):
    """Common failure patterns."""

    INTERMITTENT = "intermittent"
    CASCADING = "cascading"
    RESOURCE_RELATED = "resource_related"
    ENVIRONMENT_SPECIFIC = "environment_specific"
    LOAD_DEPENDENT = "load_dependent"
    TIME_DEPENDENT = "time_dependent"
    CONFIGURATION_DRIFT = "configuration_drift"


@dataclass
class ErrorSignature:
    """Error signature for pattern matching."""

    error_type: str
    message_pattern: str
    stack_trace_pattern: str | None = None
    context_patterns: dict[str, str] = field(default_factory=dict)

    def matches(self, error: "ErrorEvent") -> bool:
        """Check if error matches this signature."""
        # Check error type
        if self.error_type and error.error_type != self.error_type:
            return False

        # Check message pattern
        if self.message_pattern and not re.search(
            self.message_pattern, error.message, re.IGNORECASE
        ):
            return False

        # Check stack trace pattern
        if self.stack_trace_pattern and error.stack_trace:
            if not re.search(self.stack_trace_pattern, error.stack_trace, re.IGNORECASE):
                return False

        # Check context patterns
        for key, pattern in self.context_patterns.items():
            context_value = error.context.get(key, "")
            if not re.search(pattern, str(context_value), re.IGNORECASE):
                return False

        return True


@dataclass
class ErrorEvent:
    """Individual error event."""

    timestamp: datetime
    error_id: str
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    container_id: str | None = None
    language: str | None = None
    test_type: str | None = None
    execution_id: str | None = None
    stack_trace: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_id": self.error_id,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "container_id": self.container_id,
            "language": self.language,
            "test_type": self.test_type,
            "execution_id": self.execution_id,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass
class ErrorPattern:
    """Detected error pattern."""

    pattern_id: str
    pattern_type: FailurePattern
    signature: ErrorSignature
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    affected_containers: list[str]
    affected_languages: list[str]
    resolution_suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "signature": {
                "error_type": self.signature.error_type,
                "message_pattern": self.signature.message_pattern,
                "stack_trace_pattern": self.signature.stack_trace_pattern,
                "context_patterns": self.signature.context_patterns,
            },
            "occurrence_count": self.occurrence_count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "affected_containers": self.affected_containers,
            "affected_languages": self.affected_languages,
            "resolution_suggestions": self.resolution_suggestions,
        }


@dataclass
class FailureAnalysis:
    """Analysis of failure patterns and trends."""

    analysis_id: str
    analysis_timestamp: datetime
    time_window: timedelta
    total_errors: int
    error_rate: float
    top_error_categories: list[tuple[ErrorCategory, int]]
    top_error_messages: list[tuple[str, int]]
    language_breakdown: dict[str, int]
    container_breakdown: dict[str, int]
    temporal_patterns: dict[str, list[int]]
    detected_patterns: list[ErrorPattern]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "analysis_id": self.analysis_id,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "time_window_hours": self.time_window.total_seconds() / 3600,
            "total_errors": self.total_errors,
            "error_rate": self.error_rate,
            "top_error_categories": [
                (category.value, count) for category, count in self.top_error_categories
            ],
            "top_error_messages": self.top_error_messages,
            "language_breakdown": self.language_breakdown,
            "container_breakdown": self.container_breakdown,
            "temporal_patterns": self.temporal_patterns,
            "detected_patterns": [pattern.to_dict() for pattern in self.detected_patterns],
            "recommendations": self.recommendations,
        }


class ErrorClassifier:
    """Classifies errors into categories and determines severity."""

    def __init__(self):
        """Initialize the ErrorClassifier service.

        This initializes the ErrorClassifier with required dependencies
        and configuration for service operations.
        """
        self.classification_rules = self._load_classification_rules()

    def _load_classification_rules(
        self,
    ) -> dict[ErrorCategory, list[dict[str, Any]]]:
        """Load error classification rules."""
        return {
            ErrorCategory.CONTAINER_RUNTIME: [
                {
                    "pattern": r"container.*not found",
                    "severity": ErrorSeverity.HIGH,
                },
                {"pattern": r"docker.*error", "severity": ErrorSeverity.HIGH},
                {
                    "pattern": r"container.*failed to start",
                    "severity": ErrorSeverity.CRITICAL,
                },
                {
                    "pattern": r"oci runtime error",
                    "severity": ErrorSeverity.HIGH,
                },
                {
                    "pattern": r"container.*killed",
                    "severity": ErrorSeverity.MEDIUM,
                },
            ],
            ErrorCategory.LANGUAGE_COMPILATION: [
                {
                    "pattern": r"compilation.*failed",
                    "severity": ErrorSeverity.HIGH,
                },
                {"pattern": r"syntax error", "severity": ErrorSeverity.MEDIUM},
                {
                    "pattern": r"undefined reference",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"cannot resolve symbol",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"module not found",
                    "severity": ErrorSeverity.MEDIUM,
                },
            ],
            ErrorCategory.TEST_EXECUTION: [
                {"pattern": r"test.*failed", "severity": ErrorSeverity.MEDIUM},
                {
                    "pattern": r"assertion.*failed",
                    "severity": ErrorSeverity.LOW,
                },
                {
                    "pattern": r"test.*timeout",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"test runner.*error",
                    "severity": ErrorSeverity.HIGH,
                },
            ],
            ErrorCategory.RESOURCE_EXHAUSTION: [
                {
                    "pattern": r"out of memory",
                    "severity": ErrorSeverity.CRITICAL,
                },
                {
                    "pattern": r"memory limit exceeded",
                    "severity": ErrorSeverity.CRITICAL,
                },
                {"pattern": r"disk full", "severity": ErrorSeverity.CRITICAL},
                {
                    "pattern": r"cpu limit exceeded",
                    "severity": ErrorSeverity.HIGH,
                },
                {
                    "pattern": r"resource quota exceeded",
                    "severity": ErrorSeverity.HIGH,
                },
            ],
            ErrorCategory.NETWORK_CONNECTIVITY: [
                {
                    "pattern": r"connection refused",
                    "severity": ErrorSeverity.HIGH,
                },
                {
                    "pattern": r"network.*unreachable",
                    "severity": ErrorSeverity.HIGH,
                },
                {
                    "pattern": r"timeout.*connection",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"dns.*resolution.*failed",
                    "severity": ErrorSeverity.MEDIUM,
                },
            ],
            ErrorCategory.FILESYSTEM_ACCESS: [
                {
                    "pattern": r"permission denied",
                    "severity": ErrorSeverity.HIGH,
                },
                {
                    "pattern": r"file not found",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"directory not found",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"read-only file system",
                    "severity": ErrorSeverity.HIGH,
                },
            ],
            ErrorCategory.DEPENDENCY_RESOLUTION: [
                {
                    "pattern": r"dependency.*not found",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"version conflict",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"package.*not found",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"import.*failed",
                    "severity": ErrorSeverity.MEDIUM,
                },
            ],
            ErrorCategory.SECURITY_VIOLATION: [
                {"pattern": r"access denied", "severity": ErrorSeverity.HIGH},
                {
                    "pattern": r"security.*violation",
                    "severity": ErrorSeverity.CRITICAL,
                },
                {"pattern": r"unauthorized", "severity": ErrorSeverity.HIGH},
                {"pattern": r"forbidden", "severity": ErrorSeverity.HIGH},
            ],
            ErrorCategory.TIMEOUT: [
                {"pattern": r"timeout", "severity": ErrorSeverity.MEDIUM},
                {
                    "pattern": r"deadline exceeded",
                    "severity": ErrorSeverity.MEDIUM,
                },
                {
                    "pattern": r"operation.*timed out",
                    "severity": ErrorSeverity.MEDIUM,
                },
            ],
            ErrorCategory.CONFIGURATION: [
                {
                    "pattern": r"configuration.*invalid",
                    "severity": ErrorSeverity.HIGH,
                },
                {
                    "pattern": r"missing.*configuration",
                    "severity": ErrorSeverity.HIGH,
                },
                {
                    "pattern": r"invalid.*parameter",
                    "severity": ErrorSeverity.MEDIUM,
                },
            ],
        }

    def classify_error(
        self, error_message: str, context: dict[str, Any]
    ) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error into category and severity."""
        for category, rules in self.classification_rules.items():
            for rule in rules:
                if re.search(rule["pattern"], error_message, re.IGNORECASE):
                    return category, rule["severity"]

        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM


class ErrorPatternDetector:
    """Detects patterns in error occurrences."""

    def __init__(self):
        """Initialize the PatternDetector service.

        This initializes the PatternDetector with required dependencies
        and configuration for service operations.
        """
        self.known_patterns: dict[str, ErrorPattern] = {}
        self.signature_cache: dict[str, ErrorSignature] = {}

    def detect_patterns(self, errors: list[ErrorEvent]) -> list[ErrorPattern]:
        """Detect patterns in error events."""
        patterns = []

        # Group errors by similarity
        error_groups = self._group_similar_errors(errors)

        for group_signature, group_errors in error_groups.items():
            if len(group_errors) >= 2:  # Pattern requires at least 2 occurrences
                pattern = self._create_pattern(group_signature, group_errors)
                patterns.append(pattern)

        return patterns

    def _group_similar_errors(self, errors: list[ErrorEvent]) -> dict[str, list[ErrorEvent]]:
        """Group similar errors together."""
        groups = defaultdict(list)

        for error in errors:
            signature = self._generate_signature(error)
            signature_key = f"{signature.error_type}|{signature.message_pattern}"
            groups[signature_key].append(error)

        return groups

    def _generate_signature(self, error: ErrorEvent) -> ErrorSignature:
        """Generate error signature for pattern matching."""
        # Normalize error message by removing specific values
        normalized_message = self._normalize_message(error.message)

        # Extract stack trace pattern if available
        stack_trace_pattern = None
        if error.stack_trace:
            stack_trace_pattern = self._extract_stack_trace_pattern(error.stack_trace)

        return ErrorSignature(
            error_type=error.error_type,
            message_pattern=normalized_message,
            stack_trace_pattern=stack_trace_pattern,
            context_patterns={
                "language": error.language or "",
                "category": error.category.value,
            },
        )

    def _normalize_message(self, message: str) -> str:
        """Normalize error message by removing specific values."""
        # Replace specific values with placeholders
        normalized = re.sub(r"\b\d+\b", "NUM", message)
        normalized = re.sub(
            r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
            "UUID",
            normalized,
        )
        normalized = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "DATE", normalized)
        normalized = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "TIME", normalized)
        normalized = re.sub(r"\b/[^\s]+\b", "PATH", normalized)

        return normalized

    def _extract_stack_trace_pattern(self, stack_trace: str) -> str:
        """Extract pattern from stack trace."""
        # Get the first few lines of stack trace
        lines = stack_trace.split("\n")[:5]

        # Normalize function names and line numbers
        normalized_lines = []
        for line in lines:
            # Remove specific line numbers and file paths
            normalized = re.sub(r":\d+", ":NUM", line)
            normalized = re.sub(r"\b/[^\s]+\.py\b", "FILE.py", normalized)
            normalized = re.sub(r"\b/[^\s]+\.java\b", "FILE.java", normalized)
            normalized = re.sub(r"\b/[^\s]+\.go\b", "FILE.go", normalized)
            normalized_lines.append(normalized)

        return "\n".join(normalized_lines)

    def _create_pattern(self, signature_key: str, errors: list[ErrorEvent]) -> ErrorPattern:
        """Create error pattern from grouped errors."""
        first_error = errors[0]
        signature = self._generate_signature(first_error)

        # Determine pattern type
        pattern_type = self._determine_pattern_type(errors)

        # Extract affected resources
        affected_containers = list(set(e.container_id for e in errors if e.container_id))
        affected_languages = list(set(e.language for e in errors if e.language))

        # Generate resolution suggestions
        resolution_suggestions = self._generate_resolution_suggestions(
            first_error.category, signature
        )

        return ErrorPattern(
            pattern_id=f"pattern_{hash(signature_key)}",
            pattern_type=pattern_type,
            signature=signature,
            occurrence_count=len(errors),
            first_seen=min(e.timestamp for e in errors),
            last_seen=max(e.timestamp for e in errors),
            affected_containers=affected_containers,
            affected_languages=affected_languages,
            resolution_suggestions=resolution_suggestions,
        )

    def _determine_pattern_type(self, errors: list[ErrorEvent]) -> FailurePattern:
        """Determine the type of failure pattern."""
        if len(errors) < 2:
            return FailurePattern.INTERMITTENT

        # Check temporal distribution
        timestamps = [e.timestamp for e in errors]
        time_diffs = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)
        ]

        # Check for cascading failures (rapid succession)
        if any(diff < 60 for diff in time_diffs):  # Less than 1 minute apart
            return FailurePattern.CASCADING

        # Check for resource-related patterns
        resource_categories = [
            ErrorCategory.RESOURCE_EXHAUSTION,
            ErrorCategory.TIMEOUT,
        ]
        if any(e.category in resource_categories for e in errors):
            return FailurePattern.RESOURCE_RELATED

        # Check for environment-specific patterns
        languages = set(e.language for e in errors if e.language)
        if len(languages) == 1:
            return FailurePattern.ENVIRONMENT_SPECIFIC

        # Check for time-dependent patterns
        hours = [e.timestamp.hour for e in errors]
        if len(set(hours)) <= 3:  # Errors concentrated in few hours
            return FailurePattern.TIME_DEPENDENT

        return FailurePattern.INTERMITTENT

    def _generate_resolution_suggestions(
        self, category: ErrorCategory, signature: ErrorSignature
    ) -> list[str]:
        """Generate resolution suggestions based on error category."""
        suggestions = {
            ErrorCategory.CONTAINER_RUNTIME: [
                "Check Docker daemon status",
                "Verify container image availability",
                "Review container resource limits",
                "Check container networking configuration",
            ],
            ErrorCategory.LANGUAGE_COMPILATION: [
                "Verify source code syntax",
                "Check dependency versions",
                "Review compilation flags",
                "Validate language runtime version",
            ],
            ErrorCategory.TEST_EXECUTION: [
                "Review test configuration",
                "Check test data availability",
                "Verify test environment setup",
                "Increase test timeout limits",
            ],
            ErrorCategory.RESOURCE_EXHAUSTION: [
                "Increase container memory limits",
                "Optimize resource usage",
                "Implement resource monitoring",
                "Scale up infrastructure",
            ],
            ErrorCategory.NETWORK_CONNECTIVITY: [
                "Check network configuration",
                "Verify DNS resolution",
                "Review firewall rules",
                "Test network connectivity",
            ],
            ErrorCategory.FILESYSTEM_ACCESS: [
                "Check file permissions",
                "Verify filesystem mount status",
                "Review security policies",
                "Check available disk space",
            ],
            ErrorCategory.DEPENDENCY_RESOLUTION: [
                "Update dependency versions",
                "Check package repositories",
                "Review dependency conflicts",
                "Verify package installation",
            ],
            ErrorCategory.SECURITY_VIOLATION: [
                "Review security policies",
                "Check access permissions",
                "Verify authentication",
                "Audit security configuration",
            ],
            ErrorCategory.TIMEOUT: [
                "Increase timeout limits",
                "Optimize operation performance",
                "Check resource availability",
                "Review network latency",
            ],
            ErrorCategory.CONFIGURATION: [
                "Validate configuration syntax",
                "Check configuration completeness",
                "Review default values",
                "Verify configuration file permissions",
            ],
        }

        return suggestions.get(category, ["Review error details and context"])


class ErrorTracker:
    """Main error tracking and analysis service."""

    def __init__(self, retention_days: int = 30):
        """Initialize the ErrorTracker service.

        This initializes the ErrorTracker with required dependencies
        and configuration for service operations.

        Args:
            retention_days (int): Retention days
        """
        self.retention_days = retention_days
        self.error_events: list[ErrorEvent] = []
        self.classifier = ErrorClassifier()
        self.pattern_detector = ErrorPatternDetector()
        self.lock = threading.Lock()

        # Error statistics
        self.error_counts: dict[ErrorCategory, int] = defaultdict(int)
        self.language_error_counts: dict[str, int] = defaultdict(int)
        self.container_error_counts: dict[str, int] = defaultdict(int)

    def track_error(
        self,
        error_type: str,
        message: str,
        container_id: str | None = None,
        language: str | None = None,
        test_type: str | None = None,
        execution_id: str | None = None,
        stack_trace: str | None = None,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Track a new error event."""
        # Classify error
        category, severity = self.classifier.classify_error(message, context or {})

        # Create error event
        error_event = ErrorEvent(
            timestamp=datetime.now(UTC),
            error_id=f"error_{int(datetime.now(UTC).timestamp() * 1000000)}",
            error_type=error_type,
            message=message,
            severity=severity,
            category=category,
            container_id=container_id,
            language=language,
            test_type=test_type,
            execution_id=execution_id,
            stack_trace=stack_trace,
            context=context or {},
            metadata=metadata or {},
        )

        with self.lock:
            self.error_events.append(error_event)

            # Update statistics
            self.error_counts[category] += 1
            if language:
                self.language_error_counts[language] += 1
            if container_id:
                self.container_error_counts[container_id] += 1

            # Cleanup old events
            self._cleanup_old_events()

        logger.error(f"Error tracked: {error_event.error_id} - {category.value} - {message}")

        return error_event.error_id

    def track_exception(
        self,
        exception: Exception,
        container_id: str | None = None,
        language: str | None = None,
        test_type: str | None = None,
        execution_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Track an exception with full stack trace."""
        error_type = type(exception).__name__
        message = str(exception)
        stack_trace = traceback.format_exc()

        return self.track_error(
            error_type=error_type,
            message=message,
            container_id=container_id,
            language=language,
            test_type=test_type,
            execution_id=execution_id,
            stack_trace=stack_trace,
            context=context,
        )

    def analyze_failures(self, hours: int = 24) -> FailureAnalysis:
        """Analyze failure patterns over specified time period."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        with self.lock:
            recent_errors = [error for error in self.error_events if error.timestamp >= cutoff_time]

        if not recent_errors:
            return FailureAnalysis(
                analysis_id=f"analysis_{int(datetime.now(UTC).timestamp())}",
                analysis_timestamp=datetime.now(UTC),
                time_window=timedelta(hours=hours),
                total_errors=0,
                error_rate=0.0,
                top_error_categories=[],
                top_error_messages=[],
                language_breakdown={},
                container_breakdown={},
                temporal_patterns={},
                detected_patterns=[],
                recommendations=[],
            )

        # Basic statistics
        total_errors = len(recent_errors)
        error_rate = total_errors / hours

        # Top error categories
        category_counts = Counter(error.category for error in recent_errors)
        top_categories = category_counts.most_common(5)

        # Top error messages
        message_counts = Counter(error.message for error in recent_errors)
        top_messages = message_counts.most_common(5)

        # Language breakdown
        language_counts = Counter(error.language for error in recent_errors if error.language)

        # Container breakdown
        container_counts = Counter(
            error.container_id for error in recent_errors if error.container_id
        )

        # Temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(recent_errors)

        # Detect patterns
        detected_patterns = self.pattern_detector.detect_patterns(recent_errors)

        # Generate recommendations
        recommendations = self._generate_recommendations(recent_errors, detected_patterns)

        return FailureAnalysis(
            analysis_id=f"analysis_{int(datetime.now(UTC).timestamp())}",
            analysis_timestamp=datetime.now(UTC),
            time_window=timedelta(hours=hours),
            total_errors=total_errors,
            error_rate=error_rate,
            top_error_categories=top_categories,
            top_error_messages=top_messages,
            language_breakdown=dict(language_counts),
            container_breakdown=dict(container_counts),
            temporal_patterns=temporal_patterns,
            detected_patterns=detected_patterns,
            recommendations=recommendations,
        )

    def _analyze_temporal_patterns(self, errors: list[ErrorEvent]) -> dict[str, list[int]]:
        """Analyze temporal patterns in error occurrences."""
        hourly_counts = [0] * 24
        daily_counts = [0] * 7

        for error in errors:
            hourly_counts[error.timestamp.hour] += 1
            daily_counts[error.timestamp.weekday()] += 1

        return {"hourly": hourly_counts, "daily": daily_counts}

    def _generate_recommendations(
        self, errors: list[ErrorEvent], patterns: list[ErrorPattern]
    ) -> list[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []

        # High-frequency error recommendations
        if len(errors) > 100:
            recommendations.append("High error rate detected. Consider scaling up infrastructure.")

        # Category-specific recommendations
        category_counts = Counter(error.category for error in errors)
        top_category = category_counts.most_common(1)[0][0] if category_counts else None

        if top_category == ErrorCategory.RESOURCE_EXHAUSTION:
            recommendations.append(
                "Resource exhaustion is the primary issue. Increase container limits."
            )
        elif top_category == ErrorCategory.CONTAINER_RUNTIME:
            recommendations.append("Container runtime issues detected. Check Docker daemon health.")
        elif top_category == ErrorCategory.LANGUAGE_COMPILATION:
            recommendations.append(
                "Compilation errors are frequent. Review code quality and dependencies."
            )

        # Pattern-based recommendations
        if len(patterns) > 5:
            recommendations.append(
                "Multiple error patterns detected. Consider systematic debugging."
            )

        # Language-specific recommendations
        language_counts = Counter(error.language for error in errors if error.language)
        if language_counts:
            top_language = language_counts.most_common(1)[0][0]
            recommendations.append(
                f"Most errors in {top_language}. Focus optimization on this language."
            )

        return recommendations

    def _cleanup_old_events(self) -> None:
        """Remove old error events beyond retention period."""
        cutoff_time = datetime.now(UTC) - timedelta(days=self.retention_days)
        self.error_events = [error for error in self.error_events if error.timestamp >= cutoff_time]

    def get_error_statistics(self) -> dict[str, Any]:
        """Get current error statistics."""
        with self.lock:
            return {
                "total_errors": len(self.error_events),
                "error_categories": dict(self.error_counts),
                "language_breakdown": dict(self.language_error_counts),
                "container_breakdown": dict(self.container_error_counts),
                "recent_errors": len(
                    [
                        e
                        for e in self.error_events
                        if e.timestamp >= datetime.now(UTC) - timedelta(hours=1)
                    ]
                ),
            }

    def get_error_by_id(self, error_id: str) -> ErrorEvent | None:
        """Get specific error event by ID."""
        with self.lock:
            for error in self.error_events:
                if error.error_id == error_id:
                    return error
        return None

    def export_error_data(self, hours: int = 24) -> str:
        """Export error data as JSON."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        with self.lock:
            recent_errors = [
                error.to_dict() for error in self.error_events if error.timestamp >= cutoff_time
            ]

        return json.dumps(
            {
                "export_timestamp": datetime.now(UTC).isoformat(),
                "time_window_hours": hours,
                "error_count": len(recent_errors),
                "errors": recent_errors,
            },
            indent=2,
        )


# Global error tracker instance
_error_tracker = None


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


def track_error(
    error_type: str,
    message: str,
    container_id: str | None = None,
    language: str | None = None,
    **kwargs,
) -> str:
    """Convenience function to track an error."""
    return get_error_tracker().track_error(
        error_type=error_type,
        message=message,
        container_id=container_id,
        language=language,
        **kwargs,
    )


def track_exception(
    exception: Exception,
    container_id: str | None = None,
    language: str | None = None,
    **kwargs,
) -> str:
    """Convenience function to track an exception."""
    return get_error_tracker().track_exception(
        exception=exception,
        container_id=container_id,
        language=language,
        **kwargs,
    )
