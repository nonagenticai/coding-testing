"""Tests for AITestFailureAnalyzer in ai_test_failure_analyzer.py.

Covers uncovered lines: 102, 112-114, 210-284, 293-310, 324-346, 363-390,
401-430, 439-474, 675-681, 687-706, 715-731, 737-749, 755-764, 787-788,
792-793, 796, 802, 805-810, 824, 854.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.ai_test_failure_models import (
    ConfidenceLevel,
    ErrorDetails,
    FailureAnalysisRequest,
    FailureType,
    FixSuggestion,
    RootCause,
    SeverityLevel,
    TestFailureContext,
    TestFailureData,
    TestFailurePattern,
    TestFramework,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure_data(**overrides) -> TestFailureData:
    defaults = dict(
        failure_id="fail-001",
        raw_output='Traceback (most recent call last):\n  File "test_main.py", line 10, in test_add\nAssertionError: expected 3 but got 4',
        exit_code=1,
        execution_time=1.5,
        context=TestFailureContext(
            test_name="test_add",
            framework=TestFramework.PYTEST,
        ),
    )
    defaults.update(overrides)
    return TestFailureData(**defaults)


def _make_request(**overrides) -> FailureAnalysisRequest:
    defaults = dict(
        request_id="req-001",
        failure_data=_make_failure_data(),
        max_root_causes=3,
        max_suggestions=3,
        include_suggestions=True,
        include_patterns=True,
        minimum_confidence=0.3,
    )
    defaults.update(overrides)
    return FailureAnalysisRequest(**defaults)


def _make_analyzer():
    """Create AITestFailureAnalyzer with mocked AI agents."""
    with patch("services.ai_test_failure_analyzer.load_pydantic_ai_config_from_env") as mock_config:
        mock_config.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        with patch("services.ai_test_failure_analyzer.Agent"):
            from services.ai_test_failure_analyzer import AITestFailureAnalyzer

            analyzer = AITestFailureAnalyzer()
            return analyzer


# ---------------------------------------------------------------------------
# _classify_error_type
# ---------------------------------------------------------------------------


class TestClassifyErrorType:
    def test_assertion_error(self):
        analyzer = _make_analyzer()
        assert (
            analyzer._classify_error_type("AssertionError: expected 3")
            == FailureType.ASSERTION_ERROR
        )

    def test_syntax_error(self):
        analyzer = _make_analyzer()
        assert (
            analyzer._classify_error_type("SyntaxError: invalid syntax") == FailureType.SYNTAX_ERROR
        )

    def test_import_error(self):
        analyzer = _make_analyzer()
        assert (
            analyzer._classify_error_type("ImportError: No module named 'foo'")
            == FailureType.IMPORT_ERROR
        )

    def test_module_not_found_error(self):
        analyzer = _make_analyzer()
        assert (
            analyzer._classify_error_type("ModuleNotFoundError: No module named 'bar'")
            == FailureType.IMPORT_ERROR
        )

    def test_timeout_error(self):
        analyzer = _make_analyzer()
        assert (
            analyzer._classify_error_type("Test timed out after 30s") == FailureType.TIMEOUT_ERROR
        )

    def test_memory_error(self):
        analyzer = _make_analyzer()
        assert (
            analyzer._classify_error_type("MemoryError: out of memory") == FailureType.MEMORY_ERROR
        )

    def test_network_error(self):
        analyzer = _make_analyzer()
        assert (
            analyzer._classify_error_type("ConnectionError: network unreachable")
            == FailureType.NETWORK_ERROR
        )

    def test_dependency_error(self):
        analyzer = _make_analyzer()
        assert (
            analyzer._classify_error_type("dependency version conflict detected")
            == FailureType.DEPENDENCY_ERROR
        )

    def test_configuration_error(self):
        analyzer = _make_analyzer()
        assert (
            analyzer._classify_error_type("configuration file not found")
            == FailureType.CONFIGURATION_ERROR
        )

    def test_runtime_error(self):
        analyzer = _make_analyzer()
        assert analyzer._classify_error_type("some error happened") == FailureType.RUNTIME_ERROR

    def test_unknown_error(self):
        analyzer = _make_analyzer()
        assert analyzer._classify_error_type("all is well") == FailureType.UNKNOWN_ERROR


# ---------------------------------------------------------------------------
# _extract_error_message
# ---------------------------------------------------------------------------


class TestExtractErrorMessage:
    def test_extracts_error_pattern(self):
        analyzer = _make_analyzer()
        output = "AssertionError: expected 3 but got 4\nsome other line"
        msg = analyzer._extract_error_message(output)
        # First regex r"(\w+Error): (.+?)..." matches, group(2) = "expected 3 but got 4"
        assert msg == "expected 3 but got 4"

    def test_fallback_finds_error_line(self):
        analyzer = _make_analyzer()
        output = "line 1\nsome failed test\nline 3"
        msg = analyzer._extract_error_message(output)
        # No regex matches, fallback scans lines for "error"/"failed"/"exception"
        assert msg == "some failed test"

    def test_unknown_error_fallback(self):
        analyzer = _make_analyzer()
        msg = analyzer._extract_error_message("all good nothing to see")
        assert msg == "Unknown error message"


# ---------------------------------------------------------------------------
# _extract_traceback
# ---------------------------------------------------------------------------


class TestExtractTraceback:
    def test_extracts_traceback(self):
        analyzer = _make_analyzer()
        output = (
            'Traceback (most recent call last):\n  File "test.py", line 10\nAssertionError: bad'
        )
        tb = analyzer._extract_traceback(output)
        # The regex captures between "Traceback (most recent call last):" and "\nAssertionError:"
        assert tb == 'File "test.py", line 10'

    def test_no_traceback(self):
        analyzer = _make_analyzer()
        tb = analyzer._extract_traceback("no traceback here")
        assert tb is None


# ---------------------------------------------------------------------------
# _extract_location_info
# ---------------------------------------------------------------------------


class TestExtractLocationInfo:
    def test_python_file_line(self):
        analyzer = _make_analyzer()
        output = '  File "test_main.py", line 42, in test_add'
        file_path, line, col = analyzer._extract_location_info(output)
        assert file_path == "test_main.py"
        assert line == 42
        # Python traceback pattern has no column info
        assert col is None

    def test_colon_format(self):
        analyzer = _make_analyzer()
        output = "src/main.rs:10:5: error"
        file_path, line, col = analyzer._extract_location_info(output)
        assert file_path == "src/main.rs"
        assert line == 10
        assert col == 5

    def test_no_location(self):
        analyzer = _make_analyzer()
        file_path, line, col = analyzer._extract_location_info("no location info")
        assert file_path is None
        assert line is None
        assert col is None


# ---------------------------------------------------------------------------
# _parse_stack_trace
# ---------------------------------------------------------------------------


class TestParseStackTrace:
    def test_parses_entries(self):
        analyzer = _make_analyzer()
        tb = '  File "test.py", line 10\n    assert x == 3'
        entries = analyzer._parse_stack_trace(tb)
        assert entries == ['File "test.py", line 10', "assert x == 3"]

    def test_empty_string(self):
        analyzer = _make_analyzer()
        entries = analyzer._parse_stack_trace("")
        assert entries == []

    def test_filters_traceback_header(self):
        analyzer = _make_analyzer()
        tb = "Traceback (most recent call last):\n  File foo.py, line 5"
        entries = analyzer._parse_stack_trace(tb)
        # "Traceback" line is skipped by the startswith("Traceback") check
        assert entries == ["File foo.py, line 5"]


# ---------------------------------------------------------------------------
# _classify_assertion_type
# ---------------------------------------------------------------------------


class TestClassifyAssertionType:
    def test_assert_equal(self):
        analyzer = _make_analyzer()
        assert analyzer._classify_assertion_type("assertEqual(a, b)") == "assertEqual"

    def test_assert_in(self):
        analyzer = _make_analyzer()
        assert analyzer._classify_assertion_type("assertIn(x, y)") == "assertIn"

    def test_unknown(self):
        analyzer = _make_analyzer()
        assert analyzer._classify_assertion_type("some random output") == "unknown"


# ---------------------------------------------------------------------------
# _extract_assertion_values
# ---------------------------------------------------------------------------


class TestExtractAssertionValues:
    def test_expected_actual_pattern(self):
        analyzer = _make_analyzer()
        output = "Expected: 3\nActual: 4"
        expected, actual = analyzer._extract_assertion_values(output)
        assert expected == "3"
        assert actual == "4"

    def test_assertion_error_pattern(self):
        analyzer = _make_analyzer()
        output = "AssertionError: 3 != 4"
        expected, actual = analyzer._extract_assertion_values(output)
        assert expected == "3"
        assert actual == "4"

    def test_no_values(self):
        analyzer = _make_analyzer()
        expected, actual = analyzer._extract_assertion_values("no values here")
        assert expected is None
        assert actual is None


# ---------------------------------------------------------------------------
# _extract_comparison_operator
# ---------------------------------------------------------------------------


class TestExtractComparisonOperator:
    def test_finds_equals(self):
        analyzer = _make_analyzer()
        op = analyzer._extract_comparison_operator("3 == 4")
        assert op == "=="

    def test_finds_not_equals(self):
        analyzer = _make_analyzer()
        op = analyzer._extract_comparison_operator("3 != 4")
        assert op == "!="

    def test_finds_in_operator(self):
        analyzer = _make_analyzer()
        op = analyzer._extract_comparison_operator("x in [1,2,3]")
        assert op == "in"

    def test_no_operator(self):
        analyzer = _make_analyzer()
        op = analyzer._extract_comparison_operator("just a message")
        assert op is None


# ---------------------------------------------------------------------------
# _extract_assertion_message
# ---------------------------------------------------------------------------


class TestExtractAssertionMessage:
    def test_extracts_message(self):
        analyzer = _make_analyzer()
        output = "AssertionError: Values should match\n"
        msg = analyzer._extract_assertion_message(output)
        assert msg == "Values should match"

    def test_filters_comparison_messages(self):
        analyzer = _make_analyzer()
        output = "AssertionError: 3 != 4"
        msg = analyzer._extract_assertion_message(output)
        # "3 != 4" contains "!=", so it's filtered as a value comparison
        assert msg is None

    def test_no_assertion_message(self):
        analyzer = _make_analyzer()
        msg = analyzer._extract_assertion_message("no assertion here")
        assert msg is None


# ---------------------------------------------------------------------------
# _calculate_value_difference
# ---------------------------------------------------------------------------


class TestCalculateValueDifference:
    def test_numerical_difference(self):
        analyzer = _make_analyzer()
        diff = analyzer._calculate_value_difference("3", "5")
        # act - exp = 5 - 3 = 2.0
        assert diff == "Numerical difference: +2.0000"

    def test_negative_numerical_difference(self):
        analyzer = _make_analyzer()
        diff = analyzer._calculate_value_difference("10", "7")
        # act - exp = 7 - 10 = -3.0
        assert diff == "Numerical difference: -3.0000"

    def test_string_difference(self):
        analyzer = _make_analyzer()
        diff = analyzer._calculate_value_difference("hello", "world")
        # Both are <1000 chars, unified_diff produces output, so diff is a string
        assert diff is not None
        assert "expected" in diff
        assert "actual" in diff

    def test_length_difference(self):
        analyzer = _make_analyzer()
        diff = analyzer._calculate_value_difference("ab", "abc")
        # "ab" vs "abc" are single-line strings with different content AND length.
        # unified_diff returns a diff for these, so it should be a string diff, not length diff.
        assert diff is not None
        # It's a unified diff string containing the expected/actual markers
        assert isinstance(diff, str)

    def test_same_length_strings_no_diff(self):
        analyzer = _make_analyzer()
        # Same string should produce no diff at all
        diff = analyzer._calculate_value_difference("hello", "hello")
        # unified_diff returns empty for identical strings, len is equal => None
        assert diff is None

    def test_very_long_strings_return_length_difference(self):
        analyzer = _make_analyzer()
        # >1000 chars: skips unified_diff, falls to length comparison
        diff = analyzer._calculate_value_difference("a" * 1001, "b" * 1002)
        assert diff == "Length difference: expected 1001, actual 1002"

    def test_very_long_same_length_returns_none(self):
        analyzer = _make_analyzer()
        # >1000 chars and same length: skip diff, same length => None
        diff = analyzer._calculate_value_difference("a" * 2000, "b" * 2000)
        assert diff is None


# ---------------------------------------------------------------------------
# _extract_error_details and _extract_assertion_details
# ---------------------------------------------------------------------------


class TestExtractErrorDetails:
    @pytest.mark.asyncio
    async def test_extracts_full_error_details(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        details = await analyzer._extract_error_details(failure_data)
        # The default raw_output contains "AssertionError" => ASSERTION_ERROR
        assert details.error_type == FailureType.ASSERTION_ERROR
        # Traceback is present in the raw_output
        assert details.traceback == 'File "test_main.py", line 10, in test_add'
        # Extracted message from "AssertionError: expected 3 but got 4"
        assert details.error_message == "expected 3 but got 4"
        # File/line from python traceback pattern
        assert details.file_path == "test_main.py"
        assert details.line_number == 10
        assert details.column_number is None
        # Stack trace parsed from the traceback
        assert details.stack_trace == ['File "test_main.py", line 10, in test_add']

    @pytest.mark.asyncio
    async def test_extract_assertion_details_with_assertion(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data(
            raw_output="AssertionError: assertEqual failed\nExpected: 3\nActual: 4"
        )
        result = await analyzer._extract_assertion_details(failure_data)
        assert result is not None
        assert result.assertion_type == "assertEqual"
        assert result.expected_value == "3"
        assert result.actual_value == "4"
        # No operator substring ("==", "!=", etc.) appears literally in the raw_output
        assert result.comparison_operator is None
        # assertion_message: regex matches "assertEqual failed", which doesn't contain !=, ==, etc.
        assert result.assertion_message == "assertEqual failed"
        # value_difference: expected="3", actual="4" => numerical diff 4-3=+1.0000
        assert result.value_difference == "Numerical difference: +1.0000"

    @pytest.mark.asyncio
    async def test_extract_assertion_details_none_for_non_assertion(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data(raw_output="ImportError: no module named foo")
        result = await analyzer._extract_assertion_details(failure_data)
        assert result is None


# ---------------------------------------------------------------------------
# _prepare_analysis_context / _prepare_fix_context / _prepare_pattern_context
# ---------------------------------------------------------------------------


class TestPrepareContexts:
    def test_prepare_analysis_context(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3 but got 4",
            traceback="File test.py, line 10",
            file_path="test.py",
            line_number=10,
        )
        ctx = analyzer._prepare_analysis_context(failure_data, error_details)
        lines = ctx.split("\n")
        assert lines[0] == "Test Failure ID: fail-001"
        assert lines[1] == "Test Name: test_add"
        assert lines[2] == "Framework: pytest"
        assert lines[3] == "Error Type: assertion_error"
        assert lines[4] == "Error Message: expected 3 but got 4"
        assert lines[5] == "File: test.py"
        assert lines[6] == "Line: 10"
        assert lines[7] == "Traceback:"
        assert lines[8] == "File test.py, line 10"
        assert "Raw Output:" in ctx

    def test_prepare_analysis_context_without_optional_fields(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.RUNTIME_ERROR,
            error_message="something broke",
        )
        ctx = analyzer._prepare_analysis_context(failure_data, error_details)
        # No file_path, line_number, or traceback => those lines absent
        assert "File:" not in ctx
        assert "Line:" not in ctx
        assert "Traceback:" not in ctx
        assert "Error Type: runtime_error" in ctx
        assert "Error Message: something broke" in ctx

    def test_prepare_fix_context(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )
        root_causes = [
            RootCause(
                cause_id="c1",
                category="logic_error",
                description="Wrong calculation",
                severity=SeverityLevel.HIGH,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.85,
                evidence=["test failed"],
            )
        ]
        ctx = analyzer._prepare_fix_context(failure_data, error_details, root_causes)
        lines = ctx.split("\n")
        assert lines[0] == "Test Failure ID: fail-001"
        assert lines[1] == "Test Name: test_add"
        assert lines[2] == "Error Type: assertion_error"
        assert lines[3] == "Error Message: expected 3"
        assert lines[4] == "Identified Root Causes:"
        assert lines[5] == "- Wrong calculation (confidence: 0.85)"
        assert "Raw Output:" in ctx

    def test_prepare_fix_context_no_root_causes(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )
        ctx = analyzer._prepare_fix_context(failure_data, error_details, [])
        assert "Identified Root Causes:" not in ctx

    def test_prepare_pattern_context(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
            stack_trace=["File test.py, line 10"],
        )
        ctx = analyzer._prepare_pattern_context(failure_data, error_details)
        lines = ctx.split("\n")
        assert lines[0] == "Error Type: assertion_error"
        assert lines[1] == "Error Message: expected 3"
        assert lines[2] == "Framework: pytest"
        assert lines[3] == "Stack Trace:"
        assert lines[4] == "File test.py, line 10"
        assert "Raw Output:" in ctx

    def test_prepare_pattern_context_no_stack_trace(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )
        ctx = analyzer._prepare_pattern_context(failure_data, error_details)
        assert "Stack Trace:" not in ctx


# ---------------------------------------------------------------------------
# _create_failure_signature
# ---------------------------------------------------------------------------


class TestCreateFailureSignature:
    def test_creates_signature_with_file_path(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
            file_path="src/test_main.py",
        )
        sig = analyzer._create_failure_signature(failure_data, error_details)
        # Format: error_type|error_message[:100]|framework|filename
        assert sig == "assertion_error|expected 3|pytest|test_main.py"

    def test_signature_without_file_path(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.RUNTIME_ERROR,
            error_message="something failed",
        )
        sig = analyzer._create_failure_signature(failure_data, error_details)
        # No file_path => only 3 parts
        assert sig == "runtime_error|something failed|pytest"

    def test_signature_truncates_long_message(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        long_msg = "x" * 200
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message=long_msg,
        )
        sig = analyzer._create_failure_signature(failure_data, error_details)
        parts = sig.split("|")
        # Message is truncated to first 100 chars
        assert parts[1] == "x" * 100


# ---------------------------------------------------------------------------
# _calculate_overall_confidence
# ---------------------------------------------------------------------------


class TestCalculateOverallConfidence:
    def test_empty_inputs(self):
        analyzer = _make_analyzer()
        level, score = analyzer._calculate_overall_confidence([], [], [])
        assert level == ConfidenceLevel.VERY_LOW
        assert score == 0.0

    def test_high_confidence(self):
        analyzer = _make_analyzer()
        causes = [
            RootCause(
                cause_id="c1",
                category="logic_error",
                description="Off by one error in calculation",
                severity=SeverityLevel.HIGH,
                confidence=ConfidenceLevel.VERY_HIGH,
                confidence_score=0.95,
                evidence=[],
            )
        ]
        suggestions = [
            FixSuggestion(
                suggestion_id="s1",
                title="Fix the calculation logic",
                description="Change the addition to subtraction to match expected behavior",
                fix_type="code_change",
                priority="high",
                confidence=ConfidenceLevel.VERY_HIGH,
                confidence_score=0.9,
                estimated_effort="low",
            )
        ]
        patterns = [
            TestFailurePattern(
                pattern_id="p1",
                pattern_name="Assertion failure in test suite",
                pattern_signature="assertion_error_pattern_sig",
                pattern_type="assertion_pattern",
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.85,
            )
        ]
        level, score = analyzer._calculate_overall_confidence(causes, suggestions, patterns)
        # weighted_sum = 0.95*0.5 + 0.9*0.3 + 0.85*0.2 = 0.475 + 0.27 + 0.17 = 0.915
        # total_weight = 0.5 + 0.3 + 0.2 = 1.0
        # avg = 0.915 => VERY_HIGH (>= 0.9)
        assert level == ConfidenceLevel.VERY_HIGH
        assert score == pytest.approx(0.915, abs=1e-6)

    def test_low_confidence(self):
        analyzer = _make_analyzer()
        causes = [
            RootCause(
                cause_id="c1",
                category="unknown",
                description="Unknown root cause needs further investigation",
                severity=SeverityLevel.LOW,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.25,
                evidence=[],
            )
        ]
        level, score = analyzer._calculate_overall_confidence(causes, [], [])
        # weighted_sum = 0.25*0.5 = 0.125, total_weight = 0.5, avg = 0.25 => LOW
        assert level == ConfidenceLevel.LOW
        assert score == pytest.approx(0.25, abs=1e-6)

    def test_medium_confidence(self):
        analyzer = _make_analyzer()
        causes = [
            RootCause(
                cause_id="c1",
                category="logic_error",
                description="Possible logic error in calculation function",
                severity=SeverityLevel.MEDIUM,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.6,
                evidence=[],
            )
        ]
        level, score = analyzer._calculate_overall_confidence(causes, [], [])
        # weighted_sum = 0.6*0.5 = 0.3, total_weight = 0.5, avg = 0.6 => MEDIUM
        assert level == ConfidenceLevel.MEDIUM
        assert score == pytest.approx(0.6, abs=1e-6)

    def test_very_low_confidence(self):
        analyzer = _make_analyzer()
        causes = [
            RootCause(
                cause_id="c1",
                category="unknown",
                description="Completely unknown cause needing investigation",
                severity=SeverityLevel.LOW,
                confidence=ConfidenceLevel.VERY_LOW,
                confidence_score=0.1,
                evidence=[],
            )
        ]
        level, score = analyzer._calculate_overall_confidence(causes, [], [])
        # weighted_sum = 0.1*0.5 = 0.05, total_weight = 0.5, avg = 0.1 => VERY_LOW
        assert level == ConfidenceLevel.VERY_LOW
        assert score == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Fallback methods
# ---------------------------------------------------------------------------


class TestFallbackMethods:
    def test_fallback_root_cause_analysis(self):
        analyzer = _make_analyzer()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3 but got 4",
        )
        causes = analyzer._fallback_root_cause_analysis(error_details)
        assert len(causes) == 1
        cause = causes[0]
        assert cause.cause_id.startswith("fallback_")
        assert cause.category == "unknown"
        assert cause.description == "Test failed with assertion_error: expected 3 but got 4"
        assert cause.severity == SeverityLevel.MEDIUM
        assert cause.confidence == ConfidenceLevel.LOW
        assert cause.confidence_score == 0.3
        assert cause.evidence == ["Error type: assertion_error"]

    def test_fallback_fix_suggestions(self):
        analyzer = _make_analyzer()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )
        suggestions = analyzer._fallback_fix_suggestions(error_details, [])
        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.suggestion_id.startswith("fallback_")
        assert s.title == "Review error details and traceback"
        assert s.fix_type == "investigation_needed"
        assert s.priority == "medium"
        assert s.confidence == ConfidenceLevel.LOW
        assert s.confidence_score == 0.3
        assert s.estimated_effort == "medium"

    def test_fallback_pattern_detection(self):
        analyzer = _make_analyzer()
        error_details = ErrorDetails(
            error_type=FailureType.SYNTAX_ERROR,
            error_message="invalid syntax",
        )
        patterns = analyzer._fallback_pattern_detection(error_details)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.pattern_id.startswith("fallback_")
        assert p.pattern_name == "syntax_error Pattern"
        assert p.pattern_signature == "error_type:syntax_error"
        assert p.pattern_type == "error_message"
        assert p.confidence == ConfidenceLevel.LOW
        assert p.confidence_score == 0.3


# ---------------------------------------------------------------------------
# analyze_failure (full flow with mocked AI)
# ---------------------------------------------------------------------------


class TestAnalyzeFailure:
    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        analyzer = _make_analyzer()

        # Mock the AI agent calls to raise, triggering fallbacks
        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(side_effect=Exception("AI unavailable"))
        analyzer.fix_suggestion_agent = MagicMock()
        analyzer.fix_suggestion_agent.run = AsyncMock(side_effect=Exception("AI unavailable"))
        analyzer.pattern_agent = MagicMock()
        analyzer.pattern_agent.run = AsyncMock(side_effect=Exception("AI unavailable"))

        request = _make_request()
        response = await analyzer.analyze_failure(request)

        assert response.success is True
        assert response.request_id == "req-001"
        assert response.processing_time >= 0.0
        result = response.analysis_result
        assert result is not None
        # Fallback produces exactly 1 of each
        assert len(result.root_causes) == 1
        assert len(result.fix_suggestions) == 1
        assert len(result.detected_patterns) == 1
        # Error details correctly extracted from default failure data
        assert result.error_details.error_type == FailureType.ASSERTION_ERROR
        assert result.error_details.error_message == "expected 3 but got 4"
        # Model name from the config
        assert result.ai_model_version == "openai:gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_analysis_with_suggestions_and_patterns_disabled(self):
        analyzer = _make_analyzer()
        # Mock AI agents to use fallbacks
        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(side_effect=Exception("AI unavailable"))
        request = _make_request(include_suggestions=False, include_patterns=False)
        response = await analyzer.analyze_failure(request)
        assert response.success is True
        assert response.request_id == "req-001"
        result = response.analysis_result
        assert result is not None
        # Should have fallback root causes but no suggestions/patterns
        assert len(result.root_causes) == 1
        assert len(result.fix_suggestions) == 0
        assert len(result.detected_patterns) == 0

    @pytest.mark.asyncio
    async def test_analysis_failure_returns_error_response(self):
        analyzer = _make_analyzer()

        # Make _extract_error_details raise
        async def fail_extract(*args, **kwargs):
            raise RuntimeError("extraction failed")

        analyzer._extract_error_details = fail_extract

        request = _make_request()
        response = await analyzer.analyze_failure(request)

        assert response.success is False
        assert response.request_id == "req-001"
        assert response.error_message == "Analysis failed: extraction failed"
        assert response.analysis_result is None


# ---------------------------------------------------------------------------
# _analyze_root_causes / _generate_fix_suggestions / _detect_patterns
# ---------------------------------------------------------------------------


class TestAIAnalysisMethods:
    @pytest.mark.asyncio
    async def test_analyze_root_causes_with_ai_success(self):
        analyzer = _make_analyzer()
        mock_result = MagicMock()
        mock_result.output = [
            RootCause(
                cause_id="c1",
                category="logic_error",
                description="Off by one",
                severity=SeverityLevel.HIGH,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.8,
                evidence=["test output"],
            )
        ]
        analyzer.root_cause_agent.run = AsyncMock(return_value=mock_result)

        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )
        causes = await analyzer._analyze_root_causes(failure_data, error_details, 3, 0.3)
        assert len(causes) == 1
        # ID gets reassigned to cause_1_<uuid>
        assert causes[0].cause_id.startswith("cause_1_")
        assert causes[0].category == "logic_error"
        assert causes[0].description == "Off by one"
        assert causes[0].confidence_score == 0.8

    @pytest.mark.asyncio
    async def test_analyze_root_causes_filters_by_confidence(self):
        analyzer = _make_analyzer()
        mock_result = MagicMock()
        mock_result.output = [
            RootCause(
                cause_id="c1",
                category="logic_error",
                description="High confidence root cause identified",
                severity=SeverityLevel.HIGH,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.8,
                evidence=[],
            ),
            RootCause(
                cause_id="c2",
                category="unknown",
                description="Very low confidence cause below threshold",
                severity=SeverityLevel.LOW,
                confidence=ConfidenceLevel.VERY_LOW,
                confidence_score=0.1,
                evidence=[],
            ),
        ]
        analyzer.root_cause_agent.run = AsyncMock(return_value=mock_result)

        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )
        # minimum_confidence=0.5 should filter out the 0.1 cause
        causes = await analyzer._analyze_root_causes(failure_data, error_details, 3, 0.5)
        assert len(causes) == 1
        assert causes[0].confidence_score == 0.8

    @pytest.mark.asyncio
    async def test_generate_fix_suggestions_with_ai_success(self):
        analyzer = _make_analyzer()
        mock_result = MagicMock()
        mock_result.output = [
            FixSuggestion(
                suggestion_id="s1",
                title="Fix the calculation logic",
                description="Change the addition operator to match expected output values",
                fix_type="code_change",
                priority="high",
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.85,
                estimated_effort="low",
            )
        ]
        analyzer.fix_suggestion_agent.run = AsyncMock(return_value=mock_result)

        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )
        suggestions = await analyzer._generate_fix_suggestions(
            failure_data, error_details, [], 3, 0.3
        )
        assert len(suggestions) == 1
        assert suggestions[0].suggestion_id.startswith("fix_1_")
        assert suggestions[0].title == "Fix the calculation logic"
        assert suggestions[0].confidence_score == 0.85

    @pytest.mark.asyncio
    async def test_detect_patterns_uses_cache(self):
        analyzer = _make_analyzer()
        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )

        # Pre-populate cache
        sig = analyzer._create_failure_signature(failure_data, error_details)
        cached_pattern = TestFailurePattern(
            pattern_id="cached_1",
            pattern_name="Cached assertion pattern",
            pattern_signature=sig,
            pattern_type="assertion_pattern",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
            frequency=1,
        )
        analyzer._pattern_cache[sig] = cached_pattern

        patterns = await analyzer._detect_patterns(failure_data, error_details, 0.3)
        assert len(patterns) == 1
        assert patterns[0].pattern_id == "cached_1"
        # Frequency should have been incremented from 1 to 2
        assert patterns[0].frequency == 2

    @pytest.mark.asyncio
    async def test_detect_patterns_with_ai_success(self):
        analyzer = _make_analyzer()
        mock_result = MagicMock()
        mock_result.output = [
            TestFailurePattern(
                pattern_id="p1",
                pattern_name="Assertion failure pattern in tests",
                pattern_signature="assertion_error_sig_abc",
                pattern_type="assertion_pattern",
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.85,
            )
        ]
        analyzer.pattern_agent.run = AsyncMock(return_value=mock_result)

        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )
        patterns = await analyzer._detect_patterns(failure_data, error_details, 0.3)
        assert len(patterns) == 1
        assert patterns[0].pattern_id.startswith("pattern_1_")
        assert patterns[0].pattern_name == "Assertion failure pattern in tests"
        assert patterns[0].confidence_score == 0.85

        # Verify it was cached
        sig = analyzer._create_failure_signature(failure_data, error_details)
        assert sig in analyzer._pattern_cache
        assert sig in analyzer._known_signatures

    @pytest.mark.asyncio
    async def test_detect_patterns_fallback_on_ai_error(self):
        analyzer = _make_analyzer()
        analyzer.pattern_agent = MagicMock()
        analyzer.pattern_agent.run = AsyncMock(side_effect=Exception("AI down"))

        failure_data = _make_failure_data()
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected 3",
        )
        patterns = await analyzer._detect_patterns(failure_data, error_details, 0.3)
        assert len(patterns) == 1
        assert patterns[0].pattern_id.startswith("fallback_")
        assert patterns[0].pattern_name == "assertion_error Pattern"


# ---------------------------------------------------------------------------
# Init failure handling
# ---------------------------------------------------------------------------


class TestInitFailure:
    def test_init_agents_failure_sets_unavailable(self):
        with patch(
            "services.ai_test_failure_analyzer.load_pydantic_ai_config_from_env"
        ) as mock_config:
            mock_config.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
            from services.ai_test_failure_analyzer import AITestFailureAnalyzer

            # Fail at the agent-initialization seam, not by patching the Agent
            # symbol: the agents are built with explicit generics
            # (Agent[None, list[...]](...)), which a symbol patch no longer
            # intercepts. The contract under test is "init failure -> unavailable".
            with patch.object(
                AITestFailureAnalyzer, "_init_ai_agents", side_effect=Exception("init fail")
            ):
                analyzer = AITestFailureAnalyzer()
                assert analyzer.is_available is False
                assert analyzer.model_name == "openai:gpt-4o-mini"
