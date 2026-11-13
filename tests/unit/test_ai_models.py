"""Tests for AI model files: ai_test_failure_models, container_models, pydantic_ai_models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

# ── ai_test_failure_models ──────────────────────────────────────────────────
from models.ai_test_failure_models import (
    AIAnalysisResult,
    AssertionDetails,
    ConfidenceLevel,
    ErrorDetails,
    FailureAnalysisRequest,
    FailureAnalysisResponse,
    FailureType,
    FixSuggestion,
    RootCause,
    SeverityLevel,
    TestFailureContext,
    TestFailureData,
    TestFailurePattern,
    TestFileMetadata,
    TestFramework,
)

# ── StrEnum values ───────────────────────────────────────────────────────────


class TestEnums:
    def test_test_framework_values(self):
        assert TestFramework.PYTEST == "pytest"
        assert TestFramework.JEST == "jest"
        assert TestFramework.UNKNOWN == "unknown"
        assert TestFramework.UNITTEST == "unittest"
        assert TestFramework.MOCHA == "mocha"
        assert TestFramework.JUNIT == "junit"
        assert TestFramework.XUNIT == "xunit"
        assert TestFramework.RSPEC == "rspec"
        # Verify total count of members
        assert len(TestFramework) == 8

    def test_failure_type_values(self):
        assert FailureType.ASSERTION_ERROR == "assertion_error"
        assert FailureType.TIMEOUT_ERROR == "timeout_error"
        assert FailureType.UNKNOWN_ERROR == "unknown_error"
        assert FailureType.SYNTAX_ERROR == "syntax_error"
        assert FailureType.IMPORT_ERROR == "import_error"
        assert FailureType.RUNTIME_ERROR == "runtime_error"
        assert FailureType.MEMORY_ERROR == "memory_error"
        assert FailureType.NETWORK_ERROR == "network_error"
        assert FailureType.DEPENDENCY_ERROR == "dependency_error"
        assert FailureType.CONFIGURATION_ERROR == "configuration_error"
        assert len(FailureType) == 10

    def test_severity_level_values(self):
        assert SeverityLevel.CRITICAL == "critical"
        assert SeverityLevel.HIGH == "high"
        assert SeverityLevel.MEDIUM == "medium"
        assert SeverityLevel.LOW == "low"
        assert SeverityLevel.INFO == "info"
        assert len(SeverityLevel) == 5

    def test_confidence_level_values(self):
        assert ConfidenceLevel.VERY_HIGH == "very_high"
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.VERY_LOW == "very_low"
        assert len(ConfidenceLevel) == 5


# ── TestFileMetadata ─────────────────────────────────────────────────────────


class TestTestFileMetadata:
    def test_minimal(self):
        m = TestFileMetadata(file_path="tests/test_foo.py")
        assert m.file_path == "tests/test_foo.py"
        assert m.framework == "unknown"
        assert m.file_size is None
        assert m.line_count is None
        assert m.last_modified is None
        assert m.test_count is None

    def test_full(self):
        m = TestFileMetadata(
            file_path="tests/test_foo.py",
            file_size=1024,
            line_count=50,
            framework=TestFramework.PYTEST,
            test_count=10,
        )
        assert m.file_path == "tests/test_foo.py"
        assert m.file_size == 1024
        assert m.line_count == 50
        assert m.framework == "pytest"
        assert m.test_count == 10

    def test_empty_path_rejected(self):
        with pytest.raises(ValidationError):
            TestFileMetadata(file_path="")

    def test_negative_file_size_rejected(self):
        with pytest.raises(ValidationError):
            TestFileMetadata(file_path="a.py", file_size=-1)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            TestFileMetadata(file_path="a.py", extra_field="bad")


# ── TestFailureContext ───────────────────────────────────────────────────────


class TestTestFailureContext:
    def test_minimal(self):
        c = TestFailureContext(test_name="test_add")
        assert c.test_name == "test_add"
        assert c.framework == "unknown"
        assert c.test_class is None
        assert c.test_method is None
        assert c.test_file is None
        assert c.source_file is None
        assert c.test_line_number is None
        assert c.execution_environment is None

    def test_full(self):
        c = TestFailureContext(
            test_name="test_add",
            test_class="MathTests",
            test_method="test_add",
            source_file="src/math.py",
            test_line_number=42,
            framework=TestFramework.PYTEST,
        )
        assert c.test_name == "test_add"
        assert c.test_class == "MathTests"
        assert c.test_method == "test_add"
        assert c.source_file == "src/math.py"
        assert c.test_line_number == 42
        assert c.framework == "pytest"

    def test_invalid_line_number(self):
        with pytest.raises(ValidationError):
            TestFailureContext(test_name="t", test_line_number=0)


# ── TestFailureData ──────────────────────────────────────────────────────────


def _make_failure_data(**overrides):
    defaults = {
        "failure_id": "fail-001",
        "raw_output": "AssertionError: 1 != 2",
        "context": {"test_name": "test_something"},
    }
    defaults.update(overrides)
    return TestFailureData(**defaults)


class TestTestFailureData:
    def test_minimal(self):
        before = datetime.now(UTC)
        d = _make_failure_data()
        after = datetime.now(UTC)
        assert d.failure_id == "fail-001"
        assert d.raw_output == "AssertionError: 1 != 2"
        assert d.context.test_name == "test_something"
        assert d.exit_code is None
        assert d.execution_time is None
        assert d.memory_usage is None
        # timestamp should be a recent UTC datetime
        assert before <= d.timestamp <= after

    def test_bad_failure_id_chars(self):
        with pytest.raises(ValidationError, match="failure_id"):
            _make_failure_data(failure_id="bad id!")

    def test_valid_failure_id_chars(self):
        d = _make_failure_data(failure_id="abc-123_XYZ")
        assert d.failure_id == "abc-123_XYZ"

    def test_exit_code_and_timing(self):
        d = _make_failure_data(exit_code=1, execution_time=3.5, memory_usage=1024)
        assert d.exit_code == 1
        assert d.execution_time == 3.5
        assert d.memory_usage == 1024

    def test_negative_execution_time_rejected(self):
        with pytest.raises(ValidationError):
            _make_failure_data(execution_time=-1.0)


# ── ErrorDetails ─────────────────────────────────────────────────────────────


class TestErrorDetails:
    def test_minimal(self):
        e = ErrorDetails(error_type=FailureType.ASSERTION_ERROR, error_message="1!=2")
        assert e.error_type == "assertion_error"
        assert e.error_message == "1!=2"
        assert e.traceback is None
        assert e.file_path is None
        assert e.line_number is None
        assert e.column_number is None
        assert e.stack_trace is None

    def test_full(self):
        e = ErrorDetails(
            error_type=FailureType.RUNTIME_ERROR,
            error_message="boom",
            traceback="Traceback...",
            file_path="src/a.py",
            line_number=10,
            column_number=5,
            stack_trace=["frame1", "frame2"],
        )
        assert e.error_type == "runtime_error"
        assert e.error_message == "boom"
        assert e.traceback == "Traceback..."
        assert e.file_path == "src/a.py"
        assert e.line_number == 10
        assert e.column_number == 5
        assert e.stack_trace == ["frame1", "frame2"]

    def test_invalid_line_zero(self):
        with pytest.raises(ValidationError):
            ErrorDetails(
                error_type=FailureType.SYNTAX_ERROR,
                error_message="x",
                line_number=0,
            )


# ── AssertionDetails ─────────────────────────────────────────────────────────


class TestAssertionDetails:
    def test_defaults(self):
        a = AssertionDetails()
        assert a.assertion_type == "unknown"
        assert a.expected_value is None
        assert a.actual_value is None
        assert a.comparison_operator is None
        assert a.assertion_message is None
        assert a.value_difference is None

    def test_full(self):
        a = AssertionDetails(
            assertion_type="assertEqual",
            expected_value="5",
            actual_value="3",
            comparison_operator="==",
            assertion_message="values should match",
            value_difference="2",
        )
        assert a.assertion_type == "assertEqual"
        assert a.expected_value == "5"
        assert a.actual_value == "3"
        assert a.comparison_operator == "=="
        assert a.assertion_message == "values should match"
        assert a.value_difference == "2"


# ── RootCause ────────────────────────────────────────────────────────────────


def _make_root_cause(**overrides):
    defaults = {
        "cause_id": "rc-1",
        "category": "logic_error",
        "description": "Off-by-one error in boundary check",
        "severity": SeverityLevel.HIGH,
        "confidence": ConfidenceLevel.HIGH,
        "confidence_score": 0.85,
    }
    defaults.update(overrides)
    return RootCause(**defaults)


class TestRootCause:
    def test_valid(self):
        rc = _make_root_cause()
        assert rc.cause_id == "rc-1"
        assert rc.category == "logic_error"
        assert rc.description == "Off-by-one error in boundary check"
        assert rc.severity == "high"
        assert rc.confidence == "high"
        assert rc.confidence_score == 0.85
        assert rc.evidence == []
        assert rc.related_patterns is None

    def test_bad_cause_id(self):
        with pytest.raises(ValidationError, match="cause_id"):
            _make_root_cause(cause_id="bad id!")

    def test_confidence_mismatch_very_high(self):
        with pytest.raises(ValidationError, match="VERY_HIGH"):
            _make_root_cause(confidence=ConfidenceLevel.VERY_HIGH, confidence_score=0.5)

    def test_confidence_mismatch_high(self):
        with pytest.raises(ValidationError, match="HIGH"):
            _make_root_cause(confidence=ConfidenceLevel.HIGH, confidence_score=0.3)

    def test_confidence_mismatch_medium(self):
        with pytest.raises(ValidationError, match="MEDIUM"):
            _make_root_cause(confidence=ConfidenceLevel.MEDIUM, confidence_score=0.2)

    def test_confidence_mismatch_low(self):
        with pytest.raises(ValidationError, match="LOW"):
            _make_root_cause(confidence=ConfidenceLevel.LOW, confidence_score=0.1)

    def test_confidence_mismatch_very_low(self):
        with pytest.raises(ValidationError, match="VERY_LOW"):
            _make_root_cause(confidence=ConfidenceLevel.VERY_LOW, confidence_score=0.5)

    def test_confidence_very_high_valid(self):
        rc = _make_root_cause(confidence=ConfidenceLevel.VERY_HIGH, confidence_score=0.95)
        assert rc.confidence == "very_high"
        assert rc.confidence_score == 0.95

    def test_confidence_medium_valid(self):
        rc = _make_root_cause(confidence=ConfidenceLevel.MEDIUM, confidence_score=0.6)
        assert rc.confidence == "medium"
        assert rc.confidence_score == 0.6

    def test_confidence_low_valid(self):
        rc = _make_root_cause(confidence=ConfidenceLevel.LOW, confidence_score=0.3)
        assert rc.confidence == "low"
        assert rc.confidence_score == 0.3

    def test_confidence_very_low_valid(self):
        rc = _make_root_cause(confidence=ConfidenceLevel.VERY_LOW, confidence_score=0.1)
        assert rc.confidence == "very_low"
        assert rc.confidence_score == 0.1

    def test_with_evidence_and_patterns(self):
        rc = _make_root_cause(
            evidence=["stack trace shows NullPointerException"],
            related_patterns=["NPE_PATTERN"],
        )
        assert rc.evidence == ["stack trace shows NullPointerException"]
        assert rc.related_patterns == ["NPE_PATTERN"]


# ── FixSuggestion ────────────────────────────────────────────────────────────


def _make_fix_suggestion(**overrides):
    defaults = {
        "suggestion_id": "sug-1",
        "title": "Fix the boundary check",
        "description": "Change the comparison operator from < to <= to include edge case",
        "fix_type": "code_change",
        "priority": "high",
        "confidence": ConfidenceLevel.HIGH,
        "confidence_score": 0.8,
        "estimated_effort": "low",
    }
    defaults.update(overrides)
    return FixSuggestion(**defaults)


class TestFixSuggestion:
    def test_valid(self):
        s = _make_fix_suggestion()
        assert s.suggestion_id == "sug-1"
        assert s.title == "Fix the boundary check"
        assert s.description == "Change the comparison operator from < to <= to include edge case"
        assert s.fix_type == "code_change"
        assert s.priority == "high"
        assert s.confidence == "high"
        assert s.confidence_score == 0.8
        assert s.estimated_effort == "low"
        assert s.code_examples is None
        assert s.files_to_modify is None
        assert s.prerequisites is None
        assert s.potential_side_effects is None
        assert s.verification_steps is None

    def test_bad_suggestion_id(self):
        with pytest.raises(ValidationError, match="suggestion_id"):
            _make_fix_suggestion(suggestion_id="bad id!")

    def test_full(self):
        s = _make_fix_suggestion(
            code_examples=["if x <= y:"],
            files_to_modify=["src/math.py"],
            prerequisites=["review docs"],
            potential_side_effects=["may affect perf"],
            verification_steps=["run tests"],
        )
        assert s.code_examples == ["if x <= y:"]
        assert s.files_to_modify == ["src/math.py"]
        assert s.prerequisites == ["review docs"]
        assert s.potential_side_effects == ["may affect perf"]
        assert s.verification_steps == ["run tests"]


# ── TestFailurePattern ───────────────────────────────────────────────────────


class TestTestFailurePattern:
    def test_valid(self):
        p = TestFailurePattern(
            pattern_id="pat-1",
            pattern_name="Null pointer pattern",
            pattern_signature="NPE at line X in module Y",
            pattern_type="error_message",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.85,
        )
        assert p.pattern_id == "pat-1"
        assert p.pattern_name == "Null pointer pattern"
        assert p.pattern_signature == "NPE at line X in module Y"
        assert p.pattern_type == "error_message"
        assert p.frequency == 1
        assert p.confidence == "high"
        assert p.confidence_score == 0.85
        assert p.associated_failures is None
        assert p.typical_causes is None
        assert p.common_fixes is None

    def test_full(self):
        p = TestFailurePattern(
            pattern_id="pat-2",
            pattern_name="Timeout pattern matching",
            pattern_signature="TimeoutError consistently at function Z",
            pattern_type="timing_pattern",
            frequency=5,
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            associated_failures=["f1", "f2"],
            typical_causes=["network latency"],
            common_fixes=["increase timeout"],
        )
        assert p.pattern_id == "pat-2"
        assert p.pattern_name == "Timeout pattern matching"
        assert p.pattern_type == "timing_pattern"
        assert p.frequency == 5
        assert p.confidence == "medium"
        assert p.confidence_score == 0.6
        assert p.associated_failures == ["f1", "f2"]
        assert p.typical_causes == ["network latency"]
        assert p.common_fixes == ["increase timeout"]


# ── AIAnalysisResult ─────────────────────────────────────────────────────────


def _make_analysis_result(**overrides):
    failure_data = _make_failure_data()
    error_details = ErrorDetails(error_type=FailureType.ASSERTION_ERROR, error_message="1!=2")
    defaults = {
        "analysis_id": "ana-1",
        "failure_data": failure_data,
        "error_details": error_details,
        "analysis_confidence": ConfidenceLevel.HIGH,
        "analysis_confidence_score": 0.85,
    }
    defaults.update(overrides)
    return AIAnalysisResult(**defaults)


class TestAIAnalysisResult:
    def test_minimal(self):
        before = datetime.now(UTC)
        a = _make_analysis_result()
        after = datetime.now(UTC)
        assert a.analysis_id == "ana-1"
        assert a.failure_data.failure_id == "fail-001"
        assert a.error_details.error_type == "assertion_error"
        assert a.error_details.error_message == "1!=2"
        assert a.analysis_confidence == "high"
        assert a.analysis_confidence_score == 0.85
        assert a.assertion_details is None
        assert a.root_causes == []
        assert a.fix_suggestions == []
        assert a.detected_patterns == []
        assert a.processing_time is None
        assert a.ai_model_version is None
        assert a.additional_metadata is None
        # timestamp should be a recent UTC datetime
        assert before <= a.timestamp <= after

    def test_bad_analysis_id(self):
        with pytest.raises(ValidationError, match="analysis_id"):
            _make_analysis_result(analysis_id="bad id!")

    def test_root_causes_sorted(self):
        rc1 = _make_root_cause(
            cause_id="rc-low", confidence_score=0.75, confidence=ConfidenceLevel.HIGH
        )
        rc2 = _make_root_cause(
            cause_id="rc-high", confidence_score=0.88, confidence=ConfidenceLevel.HIGH
        )
        a = _make_analysis_result(root_causes=[rc1, rc2])
        assert a.root_causes[0].cause_id == "rc-high"
        assert a.root_causes[0].confidence_score == 0.88
        assert a.root_causes[1].cause_id == "rc-low"
        assert a.root_causes[1].confidence_score == 0.75

    def test_fix_suggestions_sorted(self):
        s1 = _make_fix_suggestion(suggestion_id="s-low", confidence_score=0.75)
        s2 = _make_fix_suggestion(suggestion_id="s-high", confidence_score=0.88)
        a = _make_analysis_result(fix_suggestions=[s1, s2])
        assert a.fix_suggestions[0].suggestion_id == "s-high"
        assert a.fix_suggestions[0].confidence_score == 0.88
        assert a.fix_suggestions[1].suggestion_id == "s-low"
        assert a.fix_suggestions[1].confidence_score == 0.75

    def test_with_assertion_details(self):
        a = _make_analysis_result(assertion_details=AssertionDetails(assertion_type="assertEqual"))
        assert a.assertion_details.assertion_type == "assertEqual"
        assert a.assertion_details.expected_value is None
        assert a.assertion_details.actual_value is None

    def test_with_metadata(self):
        a = _make_analysis_result(
            processing_time=1.5,
            ai_model_version="v1.0",
            additional_metadata={"key": "value"},
        )
        assert a.processing_time == 1.5
        assert a.ai_model_version == "v1.0"
        assert a.additional_metadata == {"key": "value"}


# ── FailureAnalysisRequest ───────────────────────────────────────────────────


class TestFailureAnalysisRequest:
    def test_minimal(self):
        r = FailureAnalysisRequest(
            request_id="req-1",
            failure_data=_make_failure_data(),
        )
        assert r.request_id == "req-1"
        assert r.failure_data.failure_id == "fail-001"
        assert r.include_patterns is True
        assert r.include_suggestions is True
        assert r.max_root_causes == 5
        assert r.max_suggestions == 10
        assert r.minimum_confidence == 0.1
        assert r.analysis_options is None

    def test_bad_request_id(self):
        with pytest.raises(ValidationError, match="request_id"):
            FailureAnalysisRequest(
                request_id="bad id!",
                failure_data=_make_failure_data(),
            )

    def test_custom_limits(self):
        r = FailureAnalysisRequest(
            request_id="req-2",
            failure_data=_make_failure_data(),
            max_root_causes=3,
            max_suggestions=5,
            minimum_confidence=0.5,
            include_patterns=False,
        )
        assert r.request_id == "req-2"
        assert r.max_root_causes == 3
        assert r.max_suggestions == 5
        assert r.minimum_confidence == 0.5
        assert r.include_patterns is False
        assert r.include_suggestions is True


# ── FailureAnalysisResponse ──────────────────────────────────────────────────


class TestFailureAnalysisResponse:
    def test_success(self):
        before = datetime.now(UTC)
        r = FailureAnalysisResponse(
            request_id="req-1",
            success=True,
            processing_time=2.5,
        )
        after = datetime.now(UTC)
        assert r.request_id == "req-1"
        assert r.success is True
        assert r.processing_time == 2.5
        assert r.analysis_result is None
        assert r.error_message is None
        assert r.warnings is None
        assert before <= r.timestamp <= after

    def test_failure(self):
        r = FailureAnalysisResponse(
            request_id="req-1",
            success=False,
            error_message="AI unavailable",
            processing_time=0.1,
            warnings=["partial data"],
        )
        assert r.request_id == "req-1"
        assert r.success is False
        assert r.error_message == "AI unavailable"
        assert r.processing_time == 0.1
        assert r.warnings == ["partial data"]


# ── container_models ────────────────────────────────────────────────────────

from models.container_models import (  # noqa: E402
    AnalysisConfig,
    ContainerInfo,
    ContainerSpec,
    ContainerStatus,
    GitHubRepoRequest,
)


class TestContainerStatus:
    def test_values(self):
        assert ContainerStatus.CREATING == "creating"
        assert ContainerStatus.RUNNING == "running"
        assert ContainerStatus.STOPPED == "stopped"
        assert ContainerStatus.ERROR == "error"
        assert ContainerStatus.REMOVING == "removing"
        assert len(ContainerStatus) == 5


class TestAnalysisConfig:
    def test_defaults(self):
        c = AnalysisConfig()
        assert c.include_tests is None
        assert c.language_hint is None
        assert c.exclude_patterns is None
        assert c.include_patterns is None

    def test_full(self):
        c = AnalysisConfig(
            include_tests=True,
            language_hint="python",
            exclude_patterns=["*.pyc"],
            include_patterns=["*.py"],
        )
        assert c.include_tests is True
        assert c.language_hint == "python"
        assert c.exclude_patterns == ["*.pyc"]
        assert c.include_patterns == ["*.py"]


class TestGitHubRepoRequest:
    def test_valid_url(self):
        r = GitHubRepoRequest(repo_url="https://github.com/owner/repo")
        assert r.repo_url == "https://github.com/owner/repo"
        assert r.branch == "main"
        assert r.container_name is None
        assert r.description is None
        assert r.access_token is None
        assert r.user_id is None
        assert r.force_rebuild is False
        assert r.analysis_config is None
        assert r.pr_number is None
        assert r.provision_type == "branch"

    def test_pr_url_parsing(self):
        r = GitHubRepoRequest(repo_url="https://github.com/owner/repo/pull/42")
        assert r.pr_number == 42
        assert r.provision_type == "pull_request"
        assert r.repo_url == "https://github.com/owner/repo"

    def test_invalid_url(self):
        with pytest.raises(ValidationError):
            GitHubRepoRequest(repo_url="not-a-url")

    def test_empty_url(self):
        with pytest.raises(ValidationError):
            GitHubRepoRequest(repo_url="")

    def test_with_optional_fields(self):
        r = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo",
            branch="develop",
            container_name="myContainer",
            description="desc",
            force_rebuild=True,
        )
        assert r.branch == "develop"
        assert r.container_name == "myContainer"
        assert r.description == "desc"
        assert r.force_rebuild is True

    def test_container_name_pattern(self):
        with pytest.raises(ValidationError):
            GitHubRepoRequest(
                repo_url="https://github.com/o/r",
                container_name="123bad",  # must start with letter
            )

    def test_user_id_pattern(self):
        r = GitHubRepoRequest(
            repo_url="https://github.com/o/r",
            user_id="user-123",
        )
        assert r.user_id == "user-123"

    def test_with_analysis_config(self):
        r = GitHubRepoRequest(
            repo_url="https://github.com/o/r",
            analysis_config=AnalysisConfig(language_hint="python"),
        )
        assert r.analysis_config.language_hint == "python"
        assert r.analysis_config.include_tests is None

    def test_branch_validation_empty(self):
        with pytest.raises(ValidationError):
            GitHubRepoRequest(repo_url="https://github.com/o/r", branch="")

    def test_pr_number_negative(self):
        with pytest.raises(ValidationError):
            GitHubRepoRequest(
                repo_url="https://github.com/o/r",
                pr_number=-1,
            )


class TestContainerInfo:
    def test_minimal(self):
        c = ContainerInfo(
            container_name="test-c",
            api_endpoint="http://localhost:8080",
            codebase_path="/code",
            repo_url="https://github.com/o/r",
            branch="main",
            status=ContainerStatus.RUNNING,
        )
        assert c.container_name == "test-c"
        assert c.api_endpoint == "http://localhost:8080"
        assert c.codebase_path == "/code"
        assert c.repo_url == "https://github.com/o/r"
        assert c.branch == "main"
        assert c.status == ContainerStatus.RUNNING
        # public_endpoint defaults to api_endpoint via model_validator
        assert c.public_endpoint == "http://localhost:8080"
        assert c.internal_endpoint == ""
        assert c.code_graph_id == ""
        assert c.description is None
        assert c.pr_number is None
        assert c.provision_type == "branch"
        assert c.graph_initialized is False
        assert c.tenant_id is None
        assert c.user_id is None

    def test_sync_endpoint_fields(self):
        c = ContainerInfo(
            container_name="c",
            api_endpoint="http://a",
            internal_endpoint="http://i",
            public_endpoint="http://p",
            codebase_path="/x",
            repo_url="https://github.com/o/r",
            branch="main",
            status=ContainerStatus.RUNNING,
        )
        assert c.api_endpoint == "http://a"
        assert c.internal_endpoint == "http://i"
        assert c.public_endpoint == "http://p"

    def test_public_inherits_api(self):
        c = ContainerInfo(
            container_name="c",
            api_endpoint="http://api",
            codebase_path="/x",
            repo_url="https://github.com/o/r",
            branch="main",
            status=ContainerStatus.RUNNING,
        )
        assert c.public_endpoint == "http://api"
        assert c.api_endpoint == "http://api"

    def test_api_endpoint_inherits_from_internal(self):
        """Covers line 240: api_endpoint inherits from internal_endpoint.

        Validator order: public_endpoint gets set to api_endpoint first (both empty),
        then api_endpoint falls back to internal_endpoint. So public_endpoint stays empty.
        """
        c = ContainerInfo(
            container_name="c",
            api_endpoint="",
            internal_endpoint="http://internal",
            codebase_path="/x",
            repo_url="https://github.com/o/r",
            branch="main",
            status=ContainerStatus.RUNNING,
        )
        assert c.api_endpoint == "http://internal"
        # public_endpoint was set before api_endpoint was updated, so it stays empty
        assert c.public_endpoint == ""

    def test_optional_fields(self):
        c = ContainerInfo(
            container_name="c",
            api_endpoint="http://a",
            codebase_path="/x",
            repo_url="https://github.com/o/r",
            branch="main",
            status=ContainerStatus.CREATING,
            description="test",
            pr_number=5,
            user_id="u1",
            tenant_id="t1",
            graph_initialized=True,
        )
        assert c.status == ContainerStatus.CREATING
        assert c.description == "test"
        assert c.pr_number == 5
        assert c.user_id == "u1"
        assert c.tenant_id == "t1"
        assert c.graph_initialized is True


class TestContainerSpec:
    def test_valid(self):
        s = ContainerSpec(name="test", image="python:3.13", port=8080)
        assert s.name == "test"
        assert s.image == "python:3.13"
        assert s.port == 8080
        assert s.command is None
        assert s.working_dir is None
        assert s.environment == {}
        assert s.volumes == {}

    def test_with_options(self):
        s = ContainerSpec(
            name="test",
            image="python:3.13",
            port=8080,
            command="pytest",
            working_dir="/app",
            environment={"KEY": "val"},
            volumes={"/host": "/container"},
        )
        assert s.command == "pytest"
        assert s.working_dir == "/app"
        assert s.environment == {"KEY": "val"}
        assert s.volumes == {"/host": "/container"}

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ContainerSpec(name="", image="python:3.13", port=8080)

    def test_whitespace_name_rejected(self):
        with pytest.raises(ValidationError):
            ContainerSpec(name="   ", image="python:3.13", port=8080)


# ── pydantic_ai_models ──────────────────────────────────────────────────────

from models.pydantic_ai_models import (  # noqa: E402
    AgentConfiguration,
    AgentMetrics,
    AIAnalysisRequest,
    AIAnalysisResponse,
    CodeQualityIssue,
    FailureAnalysisResult,
    FailureCategory,
    FailureSeverity,
    PerformanceAnalysis,
    PydanticAIAnalysisType,
    PydanticAITestFailureContext,
    SecurityAnalysis,
)


class TestPydanticAIEnums:
    def test_analysis_type(self):
        assert PydanticAIAnalysisType.FAILURE_ANALYSIS == "failure_analysis"
        assert PydanticAIAnalysisType.CODE_QUALITY == "code_quality"
        assert PydanticAIAnalysisType.PERFORMANCE == "performance"
        assert PydanticAIAnalysisType.SECURITY == "security"
        assert PydanticAIAnalysisType.DEPENDENCY == "dependency"
        assert PydanticAIAnalysisType.RACE_CONDITION == "race_condition"
        assert PydanticAIAnalysisType.MEMORY_LEAK == "memory_leak"
        assert PydanticAIAnalysisType.TIMEOUT == "timeout"
        assert PydanticAIAnalysisType.COMPILATION_ERROR == "compilation_error"
        assert PydanticAIAnalysisType.RUNTIME_ERROR == "runtime_error"
        assert len(PydanticAIAnalysisType) == 10

    def test_failure_severity(self):
        assert FailureSeverity.CRITICAL == "critical"
        assert FailureSeverity.HIGH == "high"
        assert FailureSeverity.MEDIUM == "medium"
        assert FailureSeverity.LOW == "low"
        assert FailureSeverity.INFO == "info"
        assert len(FailureSeverity) == 5

    def test_failure_category(self):
        assert FailureCategory.SYNTAX_ERROR == "syntax_error"
        assert FailureCategory.TYPE_ERROR == "type_error"
        assert FailureCategory.LOGIC_ERROR == "logic_error"
        assert FailureCategory.RUNTIME_ERROR == "runtime_error"
        assert FailureCategory.ASSERTION_ERROR == "assertion_error"
        assert FailureCategory.TIMEOUT_ERROR == "timeout_error"
        assert FailureCategory.MEMORY_ERROR == "memory_error"
        assert FailureCategory.DEPENDENCY_ERROR == "dependency_error"
        assert FailureCategory.CONFIGURATION_ERROR == "configuration_error"
        assert FailureCategory.ENVIRONMENT_ERROR == "environment_error"
        assert FailureCategory.UNKNOWN == "unknown"
        assert len(FailureCategory) == 11


class TestPydanticAITestFailureContext:
    def test_minimal(self):
        before = datetime.now(UTC)
        c = PydanticAITestFailureContext(
            test_id="t1",
            language="python",
            test_type="unit",
            exit_code=1,
            execution_time_ms=100.0,
        )
        after = datetime.now(UTC)
        assert c.test_id == "t1"
        assert c.language == "python"
        assert c.test_type == "unit"
        assert c.exit_code == 1
        assert c.execution_time_ms == 100.0
        assert c.stdout == ""
        assert c.stderr == ""
        assert c.container_name is None
        assert c.source_code is None
        assert c.test_code is None
        assert c.dependencies == {}
        assert c.environment == {}
        assert c.resource_usage is None
        assert before <= c.created_at <= after

    def test_negative_execution_time_rejected(self):
        with pytest.raises(ValidationError):
            PydanticAITestFailureContext(
                test_id="t1",
                language="python",
                test_type="unit",
                exit_code=0,
                execution_time_ms=-1.0,
            )

    def test_full(self):
        c = PydanticAITestFailureContext(
            test_id="t1",
            language="python",
            test_type="integration",
            exit_code=1,
            stdout="output",
            stderr="error",
            execution_time_ms=500.0,
            container_name="c1",
            source_code="def foo(): pass",
            test_code="def test_foo(): assert True",
            dependencies={"pytest": "7.0"},
            environment={"CI": "true"},
            resource_usage={"memory_mb": 128},
        )
        assert c.test_id == "t1"
        assert c.test_type == "integration"
        assert c.stdout == "output"
        assert c.stderr == "error"
        assert c.execution_time_ms == 500.0
        assert c.container_name == "c1"
        assert c.source_code == "def foo(): pass"
        assert c.test_code == "def test_foo(): assert True"
        assert c.dependencies == {"pytest": "7.0"}
        assert c.environment == {"CI": "true"}
        assert c.resource_usage == {"memory_mb": 128}

    def test_unusual_exit_code_accepted(self):
        """Exit codes outside 0-255 are accepted but logged as warnings."""
        c = PydanticAITestFailureContext(
            test_id="t1",
            language="python",
            test_type="unit",
            exit_code=-1,
            execution_time_ms=100.0,
        )
        assert c.exit_code == -1


class TestFailureAnalysisResult:
    def _make(self, **overrides):
        defaults = {
            "analysis_id": "a1",
            "test_id": "t1",
            "analysis_type": PydanticAIAnalysisType.FAILURE_ANALYSIS,
            "category": FailureCategory.LOGIC_ERROR,
            "severity": FailureSeverity.HIGH,
            "confidence": 0.85,
            "root_cause": "Logic error in calculation",
            "analysis_duration_ms": 100.0,
            "ai_model_provider": "openai",
            "ai_model_name": "gpt-4",
        }
        defaults.update(overrides)
        return FailureAnalysisResult(**defaults)

    def test_minimal(self):
        before = datetime.now(UTC)
        r = self._make()
        after = datetime.now(UTC)
        assert r.analysis_id == "a1"
        assert r.test_id == "t1"
        assert r.analysis_type == "failure_analysis"
        assert r.category == "logic_error"
        assert r.severity == "high"
        assert r.confidence == 0.85
        assert r.root_cause == "Logic error in calculation"
        assert r.analysis_duration_ms == 100.0
        assert r.ai_model_provider == "openai"
        assert r.ai_model_name == "gpt-4"
        assert r.symptoms == []
        assert r.potential_causes == []
        assert r.recommendations == []
        assert r.code_fixes == []
        assert r.related_issues == []
        assert r.raw_response is None
        assert r.metadata == {}
        assert before <= r.created_at <= after

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            self._make(confidence=1.5)
        with pytest.raises(ValidationError):
            self._make(confidence=-0.1)

    def test_full(self):
        r = self._make(
            symptoms=["assertion failed"],
            potential_causes=["wrong formula"],
            recommendations=["fix formula"],
            code_fixes=["return a + b"],
            related_issues=["#123"],
            raw_response="raw",
            metadata={"key": "val"},
        )
        assert r.symptoms == ["assertion failed"]
        assert r.potential_causes == ["wrong formula"]
        assert r.recommendations == ["fix formula"]
        assert r.code_fixes == ["return a + b"]
        assert r.related_issues == ["#123"]
        assert r.raw_response == "raw"
        assert r.metadata == {"key": "val"}


class TestCodeQualityIssue:
    def test_valid(self):
        i = CodeQualityIssue(
            issue_id="i1",
            category="style",
            severity=FailureSeverity.LOW,
            message="line too long",
        )
        assert i.issue_id == "i1"
        assert i.category == "style"
        assert i.severity == "low"
        assert i.message == "line too long"
        assert i.file_path is None
        assert i.line_number is None
        assert i.column_number is None
        assert i.code_snippet is None
        assert i.suggested_fix is None
        assert i.fix_confidence is None
        assert i.rule_id is None
        assert i.tool_name is None

    def test_line_number_zero_rejected(self):
        with pytest.raises(ValidationError):
            CodeQualityIssue(
                issue_id="i1",
                category="bug",
                severity=FailureSeverity.HIGH,
                message="error",
                line_number=0,
            )

    def test_full(self):
        i = CodeQualityIssue(
            issue_id="i1",
            category="style",
            severity=FailureSeverity.INFO,
            message="long line",
            file_path="src/a.py",
            line_number=10,
            column_number=80,
            code_snippet="x = 1",
            suggested_fix="break line",
            fix_confidence=0.9,
            rule_id="E501",
            tool_name="ruff",
        )
        assert i.issue_id == "i1"
        assert i.severity == "info"
        assert i.file_path == "src/a.py"
        assert i.line_number == 10
        assert i.column_number == 80
        assert i.code_snippet == "x = 1"
        assert i.suggested_fix == "break line"
        assert i.fix_confidence == 0.9
        assert i.rule_id == "E501"
        assert i.tool_name == "ruff"


class TestPerformanceAnalysis:
    def test_valid(self):
        p = PerformanceAnalysis(analysis_id="p1", execution_time_ms=500.0)
        assert p.analysis_id == "p1"
        assert p.execution_time_ms == 500.0
        assert p.memory_usage_mb is None
        assert p.cpu_usage_percent is None
        assert p.bottlenecks == []
        assert p.optimization_suggestions == []
        assert p.performance_score is None
        assert p.baseline_comparison is None

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            PerformanceAnalysis(analysis_id="p1", execution_time_ms=100.0, performance_score=11.0)

    def test_full(self):
        p = PerformanceAnalysis(
            analysis_id="p1",
            execution_time_ms=500.0,
            memory_usage_mb=128.0,
            cpu_usage_percent=50.0,
            bottlenecks=["slow loop"],
            optimization_suggestions=["use vectorization"],
            performance_score=7.5,
            baseline_comparison={"previous": 600.0},
        )
        assert p.analysis_id == "p1"
        assert p.execution_time_ms == 500.0
        assert p.memory_usage_mb == 128.0
        assert p.cpu_usage_percent == 50.0
        assert p.bottlenecks == ["slow loop"]
        assert p.optimization_suggestions == ["use vectorization"]
        assert p.performance_score == 7.5
        assert p.baseline_comparison == {"previous": 600.0}


class TestSecurityAnalysis:
    def test_valid(self):
        s = SecurityAnalysis(analysis_id="s1", risk_level=FailureSeverity.LOW)
        assert s.analysis_id == "s1"
        assert s.risk_level == "low"
        assert s.vulnerabilities == []
        assert s.security_score is None
        assert s.recommendations == []
        assert s.cve_references == []
        assert s.compliance_issues == []

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            SecurityAnalysis(analysis_id="s1", risk_level=FailureSeverity.HIGH, security_score=-1.0)

    def test_full(self):
        s = SecurityAnalysis(
            analysis_id="s1",
            risk_level=FailureSeverity.CRITICAL,
            vulnerabilities=["SQL injection"],
            security_score=2.0,
            recommendations=["use parameterized queries"],
            cve_references=["CVE-2024-1234"],
            compliance_issues=["OWASP A1"],
        )
        assert s.analysis_id == "s1"
        assert s.risk_level == "critical"
        assert s.vulnerabilities == ["SQL injection"]
        assert s.security_score == 2.0
        assert s.recommendations == ["use parameterized queries"]
        assert s.cve_references == ["CVE-2024-1234"]
        assert s.compliance_issues == ["OWASP A1"]


class TestAIAnalysisRequest:
    def _make_context(self):
        return PydanticAITestFailureContext(
            test_id="t1",
            language="python",
            test_type="unit",
            exit_code=1,
            execution_time_ms=100.0,
        )

    def test_defaults(self):
        r = AIAnalysisRequest(context=self._make_context())
        assert r.context.test_id == "t1"
        assert r.ai_model_provider == "openai"
        assert r.ai_model_name == "gpt-4"
        assert r.max_tokens == 2000
        assert r.temperature == 0.1
        assert r.timeout_seconds == 30
        assert r.include_code_fixes is True
        assert r.include_performance_analysis is True
        assert r.include_security_analysis is True
        assert r.analysis_types == []
        assert r.metadata == {}

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            AIAnalysisRequest(context=self._make_context(), temperature=3.0)

    def test_max_tokens_bounds(self):
        with pytest.raises(ValidationError):
            AIAnalysisRequest(context=self._make_context(), max_tokens=0)
        with pytest.raises(ValidationError):
            AIAnalysisRequest(context=self._make_context(), max_tokens=50000)


class TestAIAnalysisResponse:
    def test_success(self):
        before = datetime.now(UTC)
        r = AIAnalysisResponse(
            request_id="r1",
            success=True,
            total_analysis_time_ms=200.0,
        )
        after = datetime.now(UTC)
        assert r.request_id == "r1"
        assert r.success is True
        assert r.total_analysis_time_ms == 200.0
        assert r.failure_analysis is None
        assert r.code_quality_issues == []
        assert r.performance_analysis is None
        assert r.security_analysis is None
        assert r.ai_model_usage == {}
        assert r.error_message is None
        assert r.warnings == []
        assert r.metadata == {}
        assert before <= r.created_at <= after

    def test_failure(self):
        r = AIAnalysisResponse(
            request_id="r1",
            success=False,
            total_analysis_time_ms=50.0,
            error_message="timeout",
            warnings=["slow"],
        )
        assert r.request_id == "r1"
        assert r.success is False
        assert r.total_analysis_time_ms == 50.0
        assert r.error_message == "timeout"
        assert r.warnings == ["slow"]


class TestAgentConfiguration:
    def test_minimal(self):
        c = AgentConfiguration(
            agent_id="agent-1",
            ai_model_provider="openai",
            ai_model_name="gpt-4",
        )
        assert c.agent_id == "agent-1"
        assert c.ai_model_provider == "openai"
        assert c.ai_model_name == "gpt-4"
        assert c.api_key is None
        assert c.base_url is None
        assert c.max_retries == 3
        assert c.timeout_seconds == 30
        assert c.rate_limit_rpm == 100
        assert c.rate_limit_tpm == 10000
        assert c.temperature == 0.1
        assert c.max_tokens == 2000
        assert c.system_prompt is None
        assert c.enabled is True
        assert c.metadata == {}

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            AgentConfiguration(
                agent_id="a",
                ai_model_provider="x",
                ai_model_name="y",
                temperature=2.5,
            )


class TestAgentMetrics:
    def test_defaults(self):
        m = AgentMetrics(agent_id="a1")
        assert m.agent_id == "a1"
        assert m.total_requests == 0
        assert m.successful_requests == 0
        assert m.failed_requests == 0
        assert m.average_response_time_ms == 0.0
        assert m.total_tokens_used == 0
        assert m.total_cost_usd == 0.0
        assert m.last_request_at is None
        assert m.error_rate == 0.0
        assert m.uptime_percent == 100.0
        assert m.metadata == {}

    def test_error_rate_bounds(self):
        with pytest.raises(ValidationError):
            AgentMetrics(agent_id="a1", error_rate=1.5)

    def test_uptime_bounds(self):
        with pytest.raises(ValidationError):
            AgentMetrics(agent_id="a1", uptime_percent=101.0)

    def test_full(self):
        now = datetime.now(UTC)
        m = AgentMetrics(
            agent_id="a1",
            total_requests=100,
            successful_requests=90,
            failed_requests=10,
            average_response_time_ms=150.0,
            total_tokens_used=5000,
            total_cost_usd=0.50,
            last_request_at=now,
            error_rate=0.1,
            uptime_percent=99.9,
            metadata={"region": "us"},
        )
        assert m.agent_id == "a1"
        assert m.total_requests == 100
        assert m.successful_requests == 90
        assert m.failed_requests == 10
        assert m.average_response_time_ms == 150.0
        assert m.total_tokens_used == 5000
        assert m.total_cost_usd == 0.50
        assert m.last_request_at == now
        assert m.error_rate == 0.1
        assert m.uptime_percent == 99.9
        assert m.metadata == {"region": "us"}
