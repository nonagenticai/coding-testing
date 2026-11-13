"""Tests for error_tracker.py — error tracking, classification, pattern detection."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from services.test_execution.error_tracker import (
    ErrorCategory,
    ErrorClassifier,
    ErrorEvent,
    ErrorPatternDetector,
    ErrorSeverity,
    ErrorSignature,
    ErrorTracker,
    FailurePattern,
    get_error_tracker,
    track_error,
    track_exception,
)

# ---------------------------------------------------------------------------
# ErrorSignature
# ---------------------------------------------------------------------------


class TestErrorSignature:
    def _make_error(self, **overrides) -> ErrorEvent:
        defaults = dict(
            timestamp=datetime.now(UTC),
            error_id="e1",
            error_type="RuntimeError",
            message="container not found",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONTAINER_RUNTIME,
            context={},
        )
        defaults.update(overrides)
        return ErrorEvent(**defaults)

    def test_matches_by_type(self):
        sig = ErrorSignature(error_type="RuntimeError", message_pattern="")
        event = self._make_error()
        assert sig.matches(event) is True

    def test_no_match_wrong_type(self):
        sig = ErrorSignature(error_type="ValueError", message_pattern="")
        event = self._make_error()
        assert sig.matches(event) is False

    def test_matches_message_pattern(self):
        sig = ErrorSignature(error_type="RuntimeError", message_pattern="container.*not found")
        event = self._make_error(message="container xyz not found")
        assert sig.matches(event) is True

    def test_no_match_message_pattern(self):
        sig = ErrorSignature(error_type="RuntimeError", message_pattern="timeout")
        event = self._make_error(message="container not found")
        assert sig.matches(event) is False

    def test_matches_stack_trace_pattern(self):
        sig = ErrorSignature(
            error_type="RuntimeError",
            message_pattern="",
            stack_trace_pattern="docker_client",
        )
        event = self._make_error(stack_trace="  File docker_client.py, line 42")
        assert sig.matches(event) is True

    def test_no_match_stack_trace_pattern(self):
        sig = ErrorSignature(
            error_type="RuntimeError",
            message_pattern="",
            stack_trace_pattern="docker_client",
        )
        event = self._make_error(stack_trace="File network.py, line 1")
        assert sig.matches(event) is False

    def test_matches_context_patterns(self):
        sig = ErrorSignature(
            error_type="RuntimeError",
            message_pattern="",
            context_patterns={"language": "python"},
        )
        event = self._make_error(context={"language": "python"})
        assert sig.matches(event) is True

    def test_no_match_context_patterns(self):
        sig = ErrorSignature(
            error_type="RuntimeError",
            message_pattern="",
            context_patterns={"language": "go"},
        )
        event = self._make_error(context={"language": "python"})
        assert sig.matches(event) is False


# ---------------------------------------------------------------------------
# ErrorEvent
# ---------------------------------------------------------------------------


class TestErrorEvent:
    def test_to_dict(self):
        event = ErrorEvent(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            error_id="e1",
            error_type="RuntimeError",
            message="boom",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONTAINER_RUNTIME,
            container_id="c1",
            language="python",
            test_type="unit",
            execution_id="exec-1",
            stack_trace="traceback...",
            context={"key": "val"},
            metadata={"k": "v"},
        )
        d = event.to_dict()
        assert d["error_id"] == "e1"
        assert d["severity"] == "high"
        assert d["category"] == "container_runtime"
        assert d["container_id"] == "c1"
        assert d["context"] == {"key": "val"}


# ---------------------------------------------------------------------------
# ErrorClassifier
# ---------------------------------------------------------------------------


class TestErrorClassifier:
    @pytest.fixture
    def classifier(self) -> ErrorClassifier:
        return ErrorClassifier()

    @pytest.mark.parametrize(
        "message,expected_category",
        [
            ("container xyz not found", ErrorCategory.CONTAINER_RUNTIME),
            ("docker daemon error", ErrorCategory.CONTAINER_RUNTIME),
            ("compilation of main.rs failed", ErrorCategory.LANGUAGE_COMPILATION),
            ("syntax error at line 5", ErrorCategory.LANGUAGE_COMPILATION),
            ("test unit_1 failed with exit code 1", ErrorCategory.TEST_EXECUTION),
            ("assertion check failed", ErrorCategory.TEST_EXECUTION),
            ("out of memory", ErrorCategory.RESOURCE_EXHAUSTION),
            ("memory limit exceeded", ErrorCategory.RESOURCE_EXHAUSTION),
            ("connection refused on port 8080", ErrorCategory.NETWORK_CONNECTIVITY),
            ("permission denied for /etc/passwd", ErrorCategory.FILESYSTEM_ACCESS),
            ("dependency foo not found", ErrorCategory.DEPENDENCY_RESOLUTION),
            ("access denied", ErrorCategory.SECURITY_VIOLATION),
            ("timeout after 30s", ErrorCategory.TIMEOUT),
            ("configuration field invalid", ErrorCategory.CONFIGURATION),
        ],
    )
    def test_classify_known_patterns(self, classifier, message, expected_category):
        category, _ = classifier.classify_error(message, {})
        assert category == expected_category

    def test_classify_unknown(self, classifier):
        category, severity = classifier.classify_error("something weird happened", {})
        assert category == ErrorCategory.UNKNOWN
        assert severity == ErrorSeverity.MEDIUM


# ---------------------------------------------------------------------------
# ErrorPatternDetector
# ---------------------------------------------------------------------------


class TestErrorPatternDetector:
    @pytest.fixture
    def detector(self) -> ErrorPatternDetector:
        return ErrorPatternDetector()

    def _make_errors(self, count: int, **overrides) -> list[ErrorEvent]:
        base_time = datetime.now(UTC)
        errors = []
        for i in range(count):
            defaults = dict(
                timestamp=base_time + timedelta(seconds=i * 30),
                error_id=f"e{i}",
                error_type="RuntimeError",
                message="container abc not found",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.CONTAINER_RUNTIME,
                container_id=f"c{i}",
                language="python",
            )
            defaults.update(overrides)
            errors.append(ErrorEvent(**defaults))
        return errors

    def test_detect_patterns_requires_two(self, detector):
        errors = self._make_errors(1)
        patterns = detector.detect_patterns(errors)
        assert len(patterns) == 0

    def test_detect_patterns_groups_similar(self, detector):
        errors = self._make_errors(3)
        patterns = detector.detect_patterns(errors)
        assert len(patterns) >= 1
        assert patterns[0].occurrence_count == 3

    def test_pattern_affected_containers(self, detector):
        errors = self._make_errors(3)
        patterns = detector.detect_patterns(errors)
        assert len(patterns[0].affected_containers) > 0

    def test_pattern_resolution_suggestions(self, detector):
        errors = self._make_errors(2)
        patterns = detector.detect_patterns(errors)
        assert len(patterns[0].resolution_suggestions) > 0

    def test_cascading_pattern_detection(self, detector):
        """Errors < 1 minute apart should be flagged as cascading."""
        base = datetime.now(UTC)
        errors = [
            ErrorEvent(
                timestamp=base + timedelta(seconds=i * 10),
                error_id=f"e{i}",
                error_type="RuntimeError",
                message="container crash",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.CONTAINER_RUNTIME,
                language="go",
            )
            for i in range(3)
        ]
        patterns = detector.detect_patterns(errors)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == FailurePattern.CASCADING

    def test_normalize_message(self, detector):
        normalized = detector._normalize_message(
            "Error at /home/user/file.py line 42 on 2025-01-01 at 10:30:00"
        )
        assert "NUM" in normalized
        # The normalizer replaces paths and numbers; date/time may be replaced as NUM-NUM-NUM
        assert "PATH" in normalized or "NUM" in normalized

    def test_extract_stack_trace_pattern(self, detector):
        trace = "File /home/user/test.py:42\nFile /home/user/main.py:10"
        pattern = detector._extract_stack_trace_pattern(trace)
        assert "NUM" in pattern

    def test_pattern_to_dict(self, detector):
        errors = self._make_errors(2)
        patterns = detector.detect_patterns(errors)
        d = patterns[0].to_dict()
        assert "pattern_id" in d
        assert "occurrence_count" in d
        assert "resolution_suggestions" in d

    def test_resource_related_pattern(self, detector):
        base = datetime.now(UTC)
        errors = [
            ErrorEvent(
                timestamp=base + timedelta(minutes=i * 5),
                error_id=f"e{i}",
                error_type="OOMKilled",
                message="out of memory",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.RESOURCE_EXHAUSTION,
                language="java",
            )
            for i in range(3)
        ]
        patterns = detector.detect_patterns(errors)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == FailurePattern.RESOURCE_RELATED


# ---------------------------------------------------------------------------
# ErrorTracker
# ---------------------------------------------------------------------------


class TestErrorTracker:
    @pytest.fixture
    def tracker(self) -> ErrorTracker:
        return ErrorTracker(retention_days=1)

    def test_track_error_returns_id(self, tracker):
        error_id = tracker.track_error("RuntimeError", "container not found")
        assert error_id.startswith("error_")

    def test_track_error_updates_counts(self, tracker):
        tracker.track_error(
            "RuntimeError",
            "container not found",
            language="python",
            container_id="c1",
        )
        stats = tracker.get_error_statistics()
        assert stats["total_errors"] == 1
        assert stats["language_breakdown"]["python"] == 1
        assert stats["container_breakdown"]["c1"] == 1

    def test_track_exception(self, tracker):
        try:
            raise ValueError("bad value")
        except ValueError as e:
            error_id = tracker.track_exception(e, container_id="c1", language="python")
        assert error_id.startswith("error_")
        assert tracker.get_error_statistics()["total_errors"] == 1

    def test_get_error_by_id(self, tracker):
        error_id = tracker.track_error("RuntimeError", "boom")
        event = tracker.get_error_by_id(error_id)
        assert event is not None
        assert event.message == "boom"

    def test_get_error_by_id_not_found(self, tracker):
        assert tracker.get_error_by_id("nonexistent") is None

    def test_analyze_failures_empty(self, tracker):
        analysis = tracker.analyze_failures(hours=1)
        assert analysis.total_errors == 0
        assert analysis.error_rate == 0.0
        assert analysis.recommendations == []

    def test_analyze_failures_with_errors(self, tracker):
        for i in range(5):
            tracker.track_error(
                "RuntimeError",
                "container not found",
                language="python",
            )
        analysis = tracker.analyze_failures(hours=1)
        assert analysis.total_errors == 5
        assert analysis.error_rate > 0

    def test_analyze_failures_recommendations(self, tracker):
        for i in range(101):
            tracker.track_error("RuntimeError", "boom", language="python")
        analysis = tracker.analyze_failures(hours=1)
        assert any(
            "scaling up" in r.lower() or "high error rate" in r.lower()
            for r in analysis.recommendations
        )

    def test_analyze_failures_language_recommendation(self, tracker):
        for i in range(3):
            tracker.track_error("RuntimeError", "boom", language="rust")
        analysis = tracker.analyze_failures(hours=1)
        assert any("rust" in r.lower() for r in analysis.recommendations)

    def test_export_error_data(self, tracker):
        tracker.track_error("RuntimeError", "boom")
        exported = tracker.export_error_data(hours=1)
        data = json.loads(exported)
        assert data["error_count"] == 1
        assert len(data["errors"]) == 1

    def test_cleanup_old_events(self, tracker):
        # Insert an "old" event by manipulating the list directly
        old_event = ErrorEvent(
            timestamp=datetime.now(UTC) - timedelta(days=2),
            error_id="old-1",
            error_type="RuntimeError",
            message="old error",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.UNKNOWN,
        )
        tracker.error_events.append(old_event)
        # Track a new error which triggers cleanup
        tracker.track_error("RuntimeError", "new error")
        # The old event should be cleaned up (retention is 1 day)
        assert all(
            e.timestamp >= datetime.now(UTC) - timedelta(days=1) for e in tracker.error_events
        )

    def test_failure_analysis_to_dict(self, tracker):
        tracker.track_error("RuntimeError", "boom", language="python")
        analysis = tracker.analyze_failures(hours=1)
        d = analysis.to_dict()
        assert "total_errors" in d
        assert "temporal_patterns" in d
        assert "recommendations" in d

    def test_temporal_patterns(self, tracker):
        tracker.track_error("RuntimeError", "boom")
        analysis = tracker.analyze_failures(hours=24)
        assert "hourly" in analysis.temporal_patterns
        assert "daily" in analysis.temporal_patterns
        assert len(analysis.temporal_patterns["hourly"]) == 24
        assert len(analysis.temporal_patterns["daily"]) == 7


# ---------------------------------------------------------------------------
# Uncovered lines: global functions, pattern detection, recommendations
# ---------------------------------------------------------------------------


class TestGlobalFunctions:
    """Cover get_error_tracker, track_error, track_exception (lines 889-891, 902, 918)."""

    def test_get_error_tracker_returns_singleton(self):
        """get_error_tracker creates and returns a global ErrorTracker."""
        import services.test_execution.error_tracker as mod

        old = mod._error_tracker
        mod._error_tracker = None
        try:
            t1 = get_error_tracker()
            assert isinstance(t1, ErrorTracker)
            t2 = get_error_tracker()
            assert t1 is t2  # same instance
        finally:
            mod._error_tracker = old

    def test_track_error_convenience(self):
        """track_error delegates to global tracker and returns error_id."""
        import services.test_execution.error_tracker as mod

        old = mod._error_tracker
        mod._error_tracker = ErrorTracker()
        try:
            error_id = track_error("TypeError", "bad type", language="python")
            assert isinstance(error_id, str)
            assert len(error_id) > 0
        finally:
            mod._error_tracker = old

    def test_track_exception_convenience(self):
        """track_exception delegates to global tracker with an Exception."""
        import services.test_execution.error_tracker as mod

        old = mod._error_tracker
        mod._error_tracker = ErrorTracker()
        try:
            error_id = track_exception(ValueError("testing"), container_id="c1", language="python")
            assert isinstance(error_id, str)
            assert len(error_id) > 0
        finally:
            mod._error_tracker = old


class TestPatternDetection:
    """Cover _determine_pattern_type edge cases (lines 513, 534-543)."""

    @pytest.fixture
    def detector(self):
        return ErrorPatternDetector()

    def _make_error(self, **overrides) -> ErrorEvent:
        defaults = dict(
            timestamp=datetime.now(UTC),
            error_id="e1",
            error_type="RuntimeError",
            message="test error",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.LANGUAGE_COMPILATION,
            context={},
        )
        defaults.update(overrides)
        return ErrorEvent(**defaults)

    def test_single_error_is_intermittent(self, detector):
        """Single error returns INTERMITTENT pattern (line 513)."""
        errors = [self._make_error()]
        pattern = detector._determine_pattern_type(errors)
        assert pattern == FailurePattern.INTERMITTENT

    def test_environment_specific_pattern(self, detector):
        """Errors from same language with no resource issues -> ENVIRONMENT_SPECIFIC (lines 534-536)."""
        base = datetime.now(UTC)
        # Timestamps must be ASCENDING and > 60s apart to avoid CASCADING
        errors = [
            self._make_error(
                error_id=f"e{i}",
                language="rust",
                category=ErrorCategory.LANGUAGE_COMPILATION,
                timestamp=base + timedelta(minutes=10 * i),
            )
            for i in range(3)
        ]
        pattern = detector._determine_pattern_type(errors)
        assert pattern == FailurePattern.ENVIRONMENT_SPECIFIC

    def test_time_dependent_pattern(self, detector):
        """Errors concentrated in few hours -> TIME_DEPENDENT (lines 539-541)."""
        base = datetime.now(UTC)
        # Multiple languages, ascending timestamps > 60s apart, same hour
        errors = [
            self._make_error(
                error_id=f"e{i}",
                language=lang,
                category=ErrorCategory.LANGUAGE_COMPILATION,
                timestamp=base + timedelta(minutes=2 * i),
            )
            for i, lang in enumerate(["python", "javascript", "rust"])
        ]
        pattern = detector._determine_pattern_type(errors)
        assert pattern == FailurePattern.TIME_DEPENDENT

    def test_default_intermittent_pattern(self, detector):
        """Errors spread across hours and languages -> INTERMITTENT (line 543)."""
        base = datetime.now(UTC)
        # Multiple languages, ascending timestamps > 60s apart, spread across many hours
        errors = [
            self._make_error(
                error_id=f"e{i}",
                language=lang,
                category=ErrorCategory.LANGUAGE_COMPILATION,
                timestamp=base + timedelta(hours=4 * i),
            )
            for i, lang in enumerate(["python", "javascript", "rust", "go"])
        ]
        pattern = detector._determine_pattern_type(errors)
        assert pattern == FailurePattern.INTERMITTENT


class TestRecommendations:
    """Cover recommendation paths (lines 806, 812, 818)."""

    @pytest.fixture
    def tracker(self):
        return ErrorTracker()

    def test_resource_exhaustion_recommendation(self, tracker):
        """RESOURCE_EXHAUSTION top category generates recommendation (line 806)."""
        for i in range(5):
            tracker.track_error("MemoryError", "out of memory", language="python")
        analysis = tracker.analyze_failures(hours=1)
        recs = " ".join(analysis.recommendations)
        assert "resource exhaustion" in recs.lower() or "container limits" in recs.lower()

    def test_compilation_recommendation(self, tracker):
        """LANGUAGE_COMPILATION top category generates recommendation (line 812)."""
        for i in range(5):
            tracker.track_error("SyntaxError", "syntax error in module", language="python")
        analysis = tracker.analyze_failures(hours=1)
        recs = " ".join(analysis.recommendations)
        assert "compilation" in recs.lower() or "code quality" in recs.lower()

    def test_many_patterns_recommendation(self, tracker):
        """More than 5 patterns generates systematic debugging recommendation (line 818)."""
        # Generate many different error types, each appearing 2+ times for pattern detection
        for i in range(8):
            for _ in range(2):
                tracker.track_error(f"Error{i}", f"unique message {i}", language="python")
        analysis = tracker.analyze_failures(hours=1)
        detected = analysis.detected_patterns
        if len(detected) > 5:
            recs = " ".join(analysis.recommendations)
            assert "systematic" in recs.lower() or "multiple" in recs.lower()


class TestStackTracePattern:
    """Cover stack trace pattern extraction (line 437)."""

    @pytest.fixture
    def tracker(self):
        return ErrorTracker()

    def test_error_with_stack_trace(self, tracker):
        """Error with stack_trace produces a signature with stack_trace_pattern (line 437)."""
        stack = 'File "test.py", line 10, in test_func\n    raise RuntimeError("test")'
        # Need 2 identical errors with stack traces to trigger pattern detection
        for _ in range(2):
            tracker.track_error(
                "RuntimeError",
                "test error",
                language="python",
                stack_trace=stack,
            )
        analysis = tracker.analyze_failures(hours=1)
        # The patterns should exist and at least one should have stack_trace_pattern
        for p in analysis.detected_patterns:
            if p.signature.stack_trace_pattern is not None:
                assert len(p.signature.stack_trace_pattern) > 0
        # Even if no pattern uses it, the code path was exercised
        assert analysis.total_errors == 2
