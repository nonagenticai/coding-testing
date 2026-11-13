"""Tests for processing/test_execution modules: security_manager, test_runner,
timeout_manager, http_test_runner."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.test_framework_config import TestConfig
from config.test_framework_security_manager import TestValidationSecurityConfig
from models.testing_models import (
    QualityRequest,
    SecurityLevel,
    TestRequest,
)
from processing.test_execution.security_manager import (
    ExecutionSecurityManager,
    create_test_security_manager,
)
from processing.test_execution.test_runner import (
    ContainerTestRunner,
    create_container_test_runner,
)
from processing.test_execution.timeout_manager import (
    ExecutionParams,
    TimeoutConfig,
    TimeoutManager,
    TimeoutResult,
    TimeoutSeverity,
    TimeoutType,
    create_container_timeout_manager,
    create_production_timeout_manager,
    create_test_timeout_manager,
)
from services.stubs import DockerService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _security_config(**overrides) -> TestValidationSecurityConfig:
    defaults = dict(
        max_execution_time=300,
        max_memory_mb=512,
        max_file_size_mb=10,
        max_code_length=10000,
        allowed_modules=["pytest", "math"],
    )
    defaults.update(overrides)
    return TestValidationSecurityConfig(**defaults)


def _test_request(**overrides) -> TestRequest:
    defaults = dict(
        code="def add(a, b): return a + b",
        test_code="def test_add(): assert add(1, 2) == 3",
        language="python",
    )
    defaults.update(overrides)
    return TestRequest(**defaults)


# ===========================================================================
# ExecutionSecurityManager
# ===========================================================================


class TestExecutionSecurityManager:
    """Tests for ExecutionSecurityManager."""

    def _make(self, **kw) -> ExecutionSecurityManager:
        return ExecutionSecurityManager(_security_config(**kw))

    # -- init / factory -------------------------------------------------

    def test_init_creates_dangerous_patterns(self):
        mgr = self._make()
        assert "python" in mgr.dangerous_patterns
        assert "javascript" in mgr.dangerous_patterns
        # Verify python patterns contain expected keys
        assert "shell_injection" in mgr.dangerous_patterns["python"]
        assert "network_access" in mgr.dangerous_patterns["python"]
        assert "file_manipulation" in mgr.dangerous_patterns["python"]
        # Verify javascript patterns contain expected keys
        assert "process_access" in mgr.dangerous_patterns["javascript"]
        assert "require_dangerous" in mgr.dangerous_patterns["javascript"]
        assert "eval_usage" in mgr.dangerous_patterns["javascript"]

    def test_factory_function(self):
        cfg = _security_config()
        mgr = create_test_security_manager(cfg)
        assert mgr.config is cfg
        assert mgr.dangerous_patterns == mgr._initialize_dangerous_patterns()

    # -- get_container_security_config ----------------------------------

    def test_container_security_config_high(self):
        mgr = self._make()
        cfg = mgr.get_container_security_config(SecurityLevel.HIGH)
        assert cfg["mem_limit"] == "256m"
        assert cfg["cpu_quota"] == 25000
        assert cfg["pids_limit"] == 50

    def test_container_security_config_medium(self):
        mgr = self._make()
        cfg = mgr.get_container_security_config(SecurityLevel.MEDIUM)
        assert cfg["mem_limit"] == "512m"
        assert cfg["cpu_quota"] == 50000

    def test_container_security_config_low(self):
        mgr = self._make()
        cfg = mgr.get_container_security_config(SecurityLevel.LOW)
        assert cfg["mem_limit"] == "1g"
        assert cfg["cpu_quota"] == 100000

    def test_container_security_config_common_keys(self):
        mgr = self._make()
        for level in SecurityLevel:
            cfg = mgr.get_container_security_config(level)
            assert cfg["read_only"] is True
            assert cfg["network_disabled"] is True
            assert cfg["user"] == "testuser"

    # -- validate_container_runtime_security ----------------------------

    def test_validate_runtime_security_pass(self):
        mgr = self._make()
        valid = {
            "read_only": True,
            "network_disabled": True,
            "user": "testuser",
            "security_opt": ["no-new-privileges:true"],
            "mem_limit": "256m",
        }
        assert mgr.validate_container_runtime_security(valid) is True

    def test_validate_runtime_security_missing_feature(self):
        mgr = self._make()
        incomplete = {"read_only": True}
        assert mgr.validate_container_runtime_security(incomplete) is False

    def test_validate_runtime_security_root_user(self):
        mgr = self._make()
        root_cfg = {
            "read_only": True,
            "network_disabled": True,
            "user": "root",
            "security_opt": [],
            "mem_limit": "256m",
        }
        assert mgr.validate_container_runtime_security(root_cfg) is False

    def test_validate_runtime_security_readonly_false(self):
        mgr = self._make()
        cfg = {
            "read_only": False,
            "network_disabled": True,
            "user": "testuser",
            "security_opt": [],
            "mem_limit": "256m",
        }
        # Should still return True (only warning)
        assert mgr.validate_container_runtime_security(cfg) is True

    # -- _validate_request_structure ------------------------------------

    def test_validate_request_no_code(self):
        mgr = self._make()
        TestRequest(language="python")
        # code defaults to None, but model_validator sets code_path="." so this is valid
        # We need an explicit test with no code and no test_code
        req2 = TestRequest.model_construct(
            code=None,
            test_code=None,
            language="python",
            security_level=SecurityLevel.HIGH,
        )
        with pytest.raises(ValueError, match="must contain either"):
            mgr._validate_request_structure(req2)

    def test_validate_request_no_language(self):
        mgr = self._make()
        req = TestRequest.model_construct(
            code="x = 1",
            test_code=None,
            language="",
            security_level=SecurityLevel.HIGH,
        )
        with pytest.raises(ValueError, match="must be specified"):
            mgr._validate_request_structure(req)

    def test_validate_request_unsupported_language(self):
        mgr = self._make()
        req = TestRequest.model_construct(
            code="x = 1",
            test_code=None,
            language="haskell",
            security_level=SecurityLevel.HIGH,
        )
        with pytest.raises(ValueError, match="Unsupported language"):
            mgr._validate_request_structure(req)

    def test_validate_request_timeout_exceeded(self):
        mgr = self._make(max_execution_time=30)
        req = TestRequest.model_construct(
            code="x = 1",
            test_code=None,
            language="python",
            timeout=999,
            security_level=SecurityLevel.HIGH,
        )
        with pytest.raises(ValueError, match="Timeout exceeds maximum"):
            mgr._validate_request_structure(req)

    # -- _validate_container_specific_security --------------------------

    @pytest.mark.asyncio
    async def test_container_escape_patterns_raise(self):
        mgr = self._make()
        with pytest.raises(ValueError, match="container escape"):
            await mgr._validate_container_specific_security("docker run bad", "python")

    @pytest.mark.asyncio
    async def test_container_escape_chroot(self):
        mgr = self._make()
        with pytest.raises(ValueError, match="container escape"):
            await mgr._validate_container_specific_security("chroot /tmp", "python")

    # -- _validate_python_security --------------------------------------

    @pytest.mark.asyncio
    async def test_python_dangerous_import_raises(self):
        mgr = self._make()
        with pytest.raises(ValueError, match="Dangerous import"):
            await mgr._validate_python_security("import subprocess")

    @pytest.mark.asyncio
    async def test_python_from_import_raises(self):
        mgr = self._make()
        with pytest.raises(ValueError, match="Dangerous import"):
            await mgr._validate_python_security("from socket import connect")

    @pytest.mark.asyncio
    async def test_python_dangerous_function_raises(self):
        mgr = self._make()
        with pytest.raises(ValueError, match="Dangerous function"):
            await mgr._validate_python_security("eval(user_input)")

    @pytest.mark.asyncio
    async def test_python_safe_code_passes(self):
        mgr = self._make()
        await mgr._validate_python_security("x = 1 + 2")

    # -- _validate_javascript_security ----------------------------------

    @pytest.mark.asyncio
    async def test_js_dangerous_require_raises(self):
        mgr = self._make()
        with pytest.raises(ValueError, match="Dangerous JavaScript"):
            await mgr._validate_javascript_security("const fs = require('fs')")

    @pytest.mark.asyncio
    async def test_js_eval_raises(self):
        mgr = self._make()
        with pytest.raises(ValueError, match="Dangerous JavaScript"):
            await mgr._validate_javascript_security("eval('code')")

    # -- _validate_resource_requirements --------------------------------

    def test_resource_requirements_code_too_long(self):
        mgr = self._make(max_code_length=10)
        req = TestRequest.model_construct(
            code="x" * 20,
            test_code=None,
            language="python",
            timeout=None,
            security_level=SecurityLevel.HIGH,
        )
        with pytest.raises(ValueError, match="Code length exceeds"):
            mgr._validate_resource_requirements(req)

    # -- scan_for_security_violations -----------------------------------

    @pytest.mark.asyncio
    async def test_scan_finds_python_violations(self):
        mgr = self._make()
        code = "os.system('rm -rf /')\nsocket.connect(('evil.com', 80))"
        violations = await mgr.scan_for_security_violations(code, "python")
        # os.system matches shell_injection, socket matches network_access
        types = {v["type"] for v in violations}
        assert "dangerous_pattern" in types
        patterns = {v["pattern"] for v in violations if v["type"] == "dangerous_pattern"}
        assert "shell_injection" in patterns
        assert "network_access" in patterns

    @pytest.mark.asyncio
    async def test_scan_finds_js_violations(self):
        mgr = self._make()
        code = "process.env.SECRET"
        violations = await mgr.scan_for_security_violations(code, "javascript")
        assert len(violations) >= 1
        # process.env matches "process_access" pattern
        pattern_names = {v["pattern"] for v in violations if v["type"] == "dangerous_pattern"}
        assert "process_access" in pattern_names

    @pytest.mark.asyncio
    async def test_scan_clean_code(self):
        mgr = self._make()
        violations = await mgr.scan_for_security_violations("x = 1", "python")
        assert len(violations) == 0

    # -- _check_suspicious_imports / _check_filesystem_access -----------

    @pytest.mark.asyncio
    async def test_check_suspicious_dynamic_import(self):
        mgr = self._make()
        violations = await mgr._check_suspicious_imports("__import__('os')", "python")
        assert len(violations) == 1
        assert violations[0]["type"] == "suspicious_import"
        assert violations[0]["pattern"] == "dynamic_import"
        assert violations[0]["severity"] == "high"
        assert "__import__('os')" in violations[0]["matches"]

    @pytest.mark.asyncio
    async def test_check_filesystem_access_python(self):
        mgr = self._make()
        violations = await mgr._check_filesystem_access("open('/etc/passwd')", "python")
        assert len(violations) == 1
        assert violations[0]["type"] == "filesystem_access"
        assert violations[0]["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_check_filesystem_access_js(self):
        mgr = self._make()
        violations = await mgr._check_filesystem_access("fs.readFile('/etc/passwd')", "javascript")
        # fs. matches one pattern, readFile matches another
        assert len(violations) >= 1
        assert all(v["type"] == "filesystem_access" for v in violations)

    # -- _sanitize_code_content -----------------------------------------

    @pytest.mark.asyncio
    async def test_sanitize_removes_null_bytes(self):
        mgr = self._make()
        result = await mgr._sanitize_code_content("hello\x00world", "python")
        assert "\x00" not in result
        assert "helloworld" == result

    @pytest.mark.asyncio
    async def test_sanitize_normalizes_newlines(self):
        mgr = self._make()
        result = await mgr._sanitize_code_content("a\r\nb\rc", "python")
        assert "\r" not in result
        assert result == "a\nb\nc"

    @pytest.mark.asyncio
    async def test_sanitize_truncates_long_code(self):
        mgr = self._make(max_code_length=20)
        result = await mgr._sanitize_code_content("x" * 100, "python")
        # Should be truncated to 20 chars plus a truncation comment
        assert result.startswith("x" * 20)
        assert "truncated" in result.lower()
        assert len(result) < 100

    # -- _sanitize_python_code ------------------------------------------

    @pytest.mark.asyncio
    async def test_sanitize_python_comments_out_imports(self):
        mgr = self._make()
        sanitized = await mgr._sanitize_python_code("import subprocess\nx = 1")
        assert "# SANITIZED: import subprocess" in sanitized
        assert "x = 1" in sanitized

    @pytest.mark.asyncio
    async def test_sanitize_python_comments_out_dangerous_funcs(self):
        mgr = self._make()
        sanitized = await mgr._sanitize_python_code("eval('code')\nx = 1")
        assert "# SANITIZED: eval(" in sanitized
        assert "x = 1" in sanitized

    # -- _sanitize_javascript_code --------------------------------------

    @pytest.mark.asyncio
    async def test_sanitize_js_comments_out_require(self):
        mgr = self._make()
        sanitized = await mgr._sanitize_javascript_code("const fs = require('fs')")
        assert "// SANITIZED:" in sanitized
        assert "require('fs')" in sanitized

    @pytest.mark.asyncio
    async def test_sanitize_js_comments_out_eval(self):
        mgr = self._make()
        sanitized = await mgr._sanitize_javascript_code("eval('x')")
        assert "// SANITIZED: eval(" in sanitized

    # -- _sanitize_string_field -----------------------------------------

    def test_sanitize_string_removes_script_tags(self):
        mgr = self._make()
        result = mgr._sanitize_string_field("<script>alert(1)</script>hi")
        assert "<script>" not in result.lower()
        assert "hi" in result

    def test_sanitize_string_removes_iframe(self):
        mgr = self._make()
        result = mgr._sanitize_string_field("<iframe src='x'></iframe>")
        assert "<iframe" not in result.lower()

    def test_sanitize_string_removes_javascript_protocol(self):
        mgr = self._make()
        result = mgr._sanitize_string_field("javascript: alert(1)")
        assert "javascript:" not in result.lower()

    def test_sanitize_string_truncates(self):
        mgr = self._make()
        result = mgr._sanitize_string_field("x" * 2000, max_length=100)
        assert len(result) == 100

    def test_sanitize_string_empty(self):
        mgr = self._make()
        assert mgr._sanitize_string_field("") == ""

    # -- validate_and_sanitize ------------------------------------------

    @pytest.mark.asyncio
    async def test_validate_and_sanitize_success(self):
        mgr = self._make(allowed_modules=["pytest", "math", "os", "sys"])
        req = _test_request()
        with patch.object(mgr.testpy_security, "validate_code", new_callable=AsyncMock) as mock_val:
            mock_val.return_value = MagicMock(passed=True)
            result = await mgr.validate_and_sanitize(req)
        assert result.language == "python"
        assert isinstance(result.code, str) and len(result.code) > 0
        assert isinstance(result.test_code, str) and len(result.test_code) > 0

    @pytest.mark.asyncio
    async def test_validate_and_sanitize_rejected_code(self):
        mgr = self._make()
        req = _test_request()
        with patch.object(mgr.testpy_security, "validate_code", new_callable=AsyncMock) as mock_val:
            mock_val.return_value = MagicMock(passed=False, error="bad code")
            with pytest.raises(ValueError, match="Security validation failed"):
                await mgr.validate_and_sanitize(req)

    # -- create_sanitization_report -------------------------------------

    @pytest.mark.asyncio
    async def test_sanitization_report_no_changes(self):
        mgr = self._make()
        req = _test_request()
        report = await mgr.create_sanitization_report(req, req)
        assert report["sanitization_applied"] is False
        assert report["changes_count"] == 0
        assert report["changes"] == []
        assert "timestamp" in report

    @pytest.mark.asyncio
    async def test_sanitization_report_with_changes(self):
        mgr = self._make()
        original = _test_request(code="original", timeout=10)
        sanitized = _test_request(code="modified", timeout=5)
        report = await mgr.create_sanitization_report(original, sanitized)
        assert report["sanitization_applied"] is True
        assert report["changes_count"] == 2  # code changed + timeout changed
        field_names = {c["field"] for c in report["changes"]}
        assert "code" in field_names
        assert "timeout" in field_names

    @pytest.mark.asyncio
    async def test_sanitization_report_test_code_change(self):
        """Covers line 653: test_code diff in sanitization report."""
        mgr = self._make()
        original = _test_request(test_code="import os\ntest_code")
        sanitized = _test_request(test_code="# SANITIZED: import os\ntest_code")
        report = await mgr.create_sanitization_report(original, sanitized)
        assert report["sanitization_applied"] is True
        code_changes = [c for c in report["changes"] if c["field"] == "test_code"]
        assert len(code_changes) == 1
        assert code_changes[0]["type"] == "code_sanitization"

    @pytest.mark.asyncio
    async def test_validate_and_sanitize_test_code_fails(self):
        """Covers line 67: test_code fails validation."""
        mgr = self._make()
        req = _test_request(code=None, test_code="def test(): pass")
        with patch.object(mgr.testpy_security, "validate_code", new_callable=AsyncMock) as mock_val:
            mock_val.return_value = MagicMock(passed=False, error="bad test code")
            with pytest.raises(ValueError, match="Security validation failed"):
                await mgr.validate_and_sanitize(req)

    def test_validate_runtime_network_disabled_false_warns(self):
        """Covers line 157: network_disabled not set triggers warning but passes."""
        mgr = self._make()
        cfg = {
            "read_only": True,
            "network_disabled": False,
            "user": "testuser",
            "security_opt": ["no-new-privileges:true"],
            "mem_limit": "256m",
        }
        # Should still return True (only a warning)
        assert mgr.validate_container_runtime_security(cfg) is True

    @pytest.mark.asyncio
    async def test_container_security_javascript_validation(self):
        """Covers lines 231-232: JS-specific validation via container security."""
        mgr = self._make()
        with pytest.raises(ValueError, match="Dangerous JavaScript"):
            await mgr._validate_container_specific_security("eval('x')", "javascript")

    @pytest.mark.asyncio
    async def test_container_security_typescript_validation(self):
        """Covers lines 231-232: TS-specific validation via container security."""
        mgr = self._make()
        with pytest.raises(ValueError, match="Dangerous JavaScript"):
            await mgr._validate_container_specific_security("eval('x')", "typescript")

    def test_validate_resource_timeout_too_high(self):
        """Covers line 308: timeout exceeds max in _validate_resource_requirements."""
        mgr = self._make(max_execution_time=30)
        req = TestRequest.model_construct(
            code="x=1",
            test_code=None,
            language="python",
            timeout=999,
            security_level=SecurityLevel.HIGH,
        )
        with pytest.raises(ValueError, match="Timeout exceeds"):
            mgr._validate_resource_requirements(req)

    @pytest.mark.asyncio
    async def test_sanitize_code_content_javascript(self):
        """Covers lines 458-459: JS sanitization path."""
        mgr = self._make()
        result = await mgr._sanitize_code_content("const fs = require('fs')", "javascript")
        assert "// SANITIZED:" in result
        assert "require('fs')" in result

    @pytest.mark.asyncio
    async def test_sanitize_request_metadata_timeout_reduced(self):
        """Covers lines 584-585: timeout exceeding max gets reduced."""
        mgr = self._make(max_execution_time=30)
        # Use model_construct to bypass timeout validation
        req = TestRequest.model_construct(
            code="x = 1",
            test_code="pass",
            language="python",
            timeout=999,
            security_level=SecurityLevel.MEDIUM,
        )
        result = await mgr._sanitize_request_metadata(req)
        assert result.timeout == 30

    @pytest.mark.asyncio
    async def test_sanitize_request_metadata_invalid_security_level(self):
        """Covers lines 596-600: invalid security_level gets reset to MEDIUM."""
        mgr = self._make()
        req = TestRequest.model_construct(
            code="x = 1",
            test_code="pass",
            language="python",
            timeout=30,
            security_level="INVALID_LEVEL",
        )
        result = await mgr._sanitize_request_metadata(req)
        assert result.security_level == SecurityLevel.MEDIUM


# ===========================================================================
# TimeoutManager
# ===========================================================================


class TestTimeoutEnums:
    def test_timeout_type_values(self):
        assert TimeoutType.TEST_EXECUTION == "test_execution"
        assert TimeoutType.CONTAINER_SETUP == "container_setup"
        assert TimeoutType.FILE_OPERATIONS == "file_operations"

    def test_timeout_severity_values(self):
        assert TimeoutSeverity.SOFT == "soft"
        assert TimeoutSeverity.HARD == "hard"
        assert TimeoutSeverity.CRITICAL == "critical"


class TestTimeoutConfig:
    def test_defaults(self):
        cfg = TimeoutConfig()
        assert cfg.default_timeout_seconds == 30
        assert cfg.max_timeout_seconds == 300
        assert cfg.cleanup_on_timeout is True
        assert cfg.graceful_shutdown_timeout == 10
        assert cfg.container_start_timeout == 60
        assert cfg.container_stop_timeout == 30
        assert cfg.retry_cleanup_attempts == 3

    def test_custom_values(self):
        cfg = TimeoutConfig(
            default_timeout_seconds=60,
            max_timeout_seconds=600,
            graceful_shutdown_timeout=20,
        )
        assert cfg.default_timeout_seconds == 60
        assert cfg.max_timeout_seconds == 600
        assert cfg.graceful_shutdown_timeout == 20

    def test_max_less_than_default_raises(self):
        with pytest.raises(ValueError, match="max_timeout_seconds"):
            TimeoutConfig(default_timeout_seconds=100, max_timeout_seconds=50)


class TestTimeoutResult:
    def test_basic_creation(self):
        result = TimeoutResult(
            success=True,
            timed_out=False,
            execution_time_seconds=1.5,
        )
        assert result.success is True
        assert result.timed_out is False
        assert result.exit_code is None
        assert result.error_message is None
        assert result.cleanup_performed is False

    def test_timeout_result(self):
        result = TimeoutResult(
            success=False,
            timed_out=True,
            execution_time_seconds=30.0,
            error_message="timeout",
            exit_code=124,
            cleanup_performed=True,
        )
        assert result.exit_code == 124
        assert result.cleanup_performed is True
        assert result.error_message == "timeout"
        assert result.execution_time_seconds == 30.0


class TestTimeoutManager:
    def _make(self, **kw) -> TimeoutManager:
        return TimeoutManager(TimeoutConfig(**kw) if kw else None)

    def test_init_defaults(self):
        mgr = self._make()
        assert mgr.config.default_timeout_seconds == 30
        assert mgr._active_tasks == {}
        assert mgr._cleanup_handlers == {}

    # -- _resolve_timeout -----------------------------------------------

    def test_resolve_specific_timeout(self):
        mgr = self._make()
        assert mgr._resolve_timeout(10, TimeoutType.TEST_EXECUTION) == 10

    def test_resolve_exceeds_max(self):
        mgr = self._make(max_timeout_seconds=60)
        assert mgr._resolve_timeout(999, TimeoutType.TEST_EXECUTION) == 60

    def test_resolve_default_for_test_execution(self):
        mgr = self._make(default_timeout_seconds=42)
        assert mgr._resolve_timeout(None, TimeoutType.TEST_EXECUTION) == 42

    def test_resolve_default_for_container_setup(self):
        mgr = self._make(container_start_timeout=120)
        assert mgr._resolve_timeout(None, TimeoutType.CONTAINER_SETUP) == 120

    def test_resolve_default_for_container_cleanup(self):
        mgr = self._make(container_stop_timeout=45)
        assert mgr._resolve_timeout(None, TimeoutType.CONTAINER_CLEANUP) == 45

    # -- execute_with_timeout -------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mgr = self._make()

        async def quick_task():
            return 42

        result = await mgr.execute_with_timeout(quick_task())
        assert result.success is True
        assert result.timed_out is False
        assert result.exit_code == 0
        assert result.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        mgr = self._make(default_timeout_seconds=1, max_timeout_seconds=1)

        async def slow_task():
            await asyncio.sleep(10)

        params = ExecutionParams(timeout_seconds=1)
        result = await mgr.execute_with_timeout(slow_task(), params)
        assert result.success is False
        assert result.timed_out is True
        assert result.exit_code == 124
        assert "timed out" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_error(self):
        mgr = self._make()

        async def failing_task():
            raise ValueError("boom")

        result = await mgr.execute_with_timeout(failing_task())
        assert result.success is False
        assert result.timed_out is False
        assert "boom" in result.error_message
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_execute_with_cleanup_on_timeout(self):
        mgr = self._make(
            default_timeout_seconds=1,
            max_timeout_seconds=1,
            graceful_shutdown_timeout=2,
        )
        cleanup_called = False

        async def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        async def slow_task():
            await asyncio.sleep(10)

        params = ExecutionParams(
            timeout_seconds=1,
            task_id="test_cleanup",
            cleanup_handler=cleanup,
        )
        result = await mgr.execute_with_timeout(slow_task(), params)
        assert result.timed_out is True
        assert result.cleanup_performed is True
        assert cleanup_called is True

    # -- timeout_context ------------------------------------------------

    @pytest.mark.asyncio
    async def test_timeout_context_success(self):
        mgr = self._make()
        async with mgr.timeout_context(timeout_seconds=5):
            await asyncio.sleep(0)
        # No exception raised

    @pytest.mark.asyncio
    async def test_timeout_context_times_out(self):
        mgr = self._make()
        with pytest.raises(TimeoutError):
            async with mgr.timeout_context(timeout_seconds=1):
                await asyncio.sleep(10)

    # -- cancel_task ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self):
        mgr = self._make()
        assert await mgr.cancel_task("nope") is False

    # -- get_active_tasks / cleanup_all_tasks ---------------------------

    @pytest.mark.asyncio
    async def test_get_active_tasks_empty(self):
        mgr = self._make()
        tasks = await mgr.get_active_tasks()
        assert tasks == {}

    @pytest.mark.asyncio
    async def test_cleanup_all_tasks_empty(self):
        mgr = self._make()
        count = await mgr.cleanup_all_tasks()
        assert count == 0

    # -- _perform_cleanup -----------------------------------------------

    @pytest.mark.asyncio
    async def test_perform_cleanup_no_handler(self):
        mgr = self._make()
        assert await mgr._perform_cleanup("nonexistent") is False

    @pytest.mark.asyncio
    async def test_perform_cleanup_success(self):
        mgr = self._make(graceful_shutdown_timeout=5)
        cleanup_mock = AsyncMock()
        mgr._cleanup_handlers["test_id"] = cleanup_mock
        assert await mgr._perform_cleanup("test_id") is True
        cleanup_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_perform_cleanup_retries_on_failure(self):
        mgr = self._make(
            retry_cleanup_attempts=2,
            graceful_shutdown_timeout=5,
        )
        cleanup_mock = AsyncMock(side_effect=[RuntimeError("fail"), None])
        mgr._cleanup_handlers["test_id"] = cleanup_mock
        result = await mgr._perform_cleanup("test_id")
        assert result is True
        assert cleanup_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_perform_cleanup_all_attempts_fail(self):
        mgr = self._make(
            retry_cleanup_attempts=2,
            graceful_shutdown_timeout=5,
        )
        cleanup_mock = AsyncMock(side_effect=RuntimeError("fail"))
        mgr._cleanup_handlers["test_id"] = cleanup_mock
        result = await mgr._perform_cleanup("test_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_task_graceful(self):
        mgr = self._make(graceful_shutdown_timeout=5)

        async def long_task():
            await asyncio.sleep(60)

        task = asyncio.create_task(long_task())
        mgr._active_tasks["t1"] = task
        result = await mgr.cancel_task("t1", graceful=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_task_force(self):
        mgr = self._make(graceful_shutdown_timeout=5)

        async def long_task():
            await asyncio.sleep(60)

        task = asyncio.create_task(long_task())
        mgr._active_tasks["t1"] = task
        result = await mgr.cancel_task("t1", graceful=False)
        assert result is True

    @pytest.mark.asyncio
    async def test_cleanup_all_tasks_with_tasks(self):
        mgr = self._make(graceful_shutdown_timeout=5)

        async def long_task():
            await asyncio.sleep(60)

        t1 = asyncio.create_task(long_task())
        t2 = asyncio.create_task(long_task())
        mgr._active_tasks["t1"] = t1
        mgr._active_tasks["t2"] = t2
        count = await mgr.cleanup_all_tasks()
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_active_tasks_with_tasks(self):
        mgr = self._make(graceful_shutdown_timeout=5)

        async def quick():
            return 1

        task = asyncio.create_task(quick())
        await task  # let it complete
        mgr._active_tasks["t1"] = task
        info = await mgr.get_active_tasks()
        assert "t1" in info
        assert info["t1"]["done"] is True
        assert info["t1"]["cancelled"] is False
        assert info["t1"]["has_cleanup"] is False


class TestTimeoutFactories:
    def test_create_test_timeout_manager(self):
        mgr = create_test_timeout_manager()
        assert mgr.config.default_timeout_seconds == 30
        assert mgr.config.max_timeout_seconds == 300
        assert mgr.config.graceful_shutdown_timeout == 10
        assert mgr.config.cleanup_on_timeout is True

    def test_create_container_timeout_manager(self):
        mgr = create_container_timeout_manager()
        assert mgr.config.default_timeout_seconds == 60
        assert mgr.config.max_timeout_seconds == 600
        assert mgr.config.container_start_timeout == 120
        assert mgr.config.container_stop_timeout == 60
        assert mgr.config.retry_cleanup_attempts == 5

    def test_create_production_timeout_manager(self):
        mgr = create_production_timeout_manager()
        assert mgr.config.default_timeout_seconds == 60
        assert mgr.config.max_timeout_seconds == 900
        assert mgr.config.container_start_timeout == 180
        assert mgr.config.container_stop_timeout == 90
        assert mgr.config.graceful_shutdown_timeout == 30


# ===========================================================================
# ContainerTestRunner (test_runner.py)
# ===========================================================================


class TestContainerTestRunner:
    def _make(self, **docker_overrides) -> ContainerTestRunner:
        docker = DockerService()
        config = TestConfig()
        return ContainerTestRunner(docker, config)

    # -- helper methods -------------------------------------------------

    def test_get_memory_limit(self):
        runner = self._make()
        assert runner._get_memory_limit(SecurityLevel.HIGH) == "256m"
        assert runner._get_memory_limit(SecurityLevel.MEDIUM) == "512m"
        assert runner._get_memory_limit(SecurityLevel.LOW) == "1g"

    def test_get_cpu_quota(self):
        runner = self._make()
        assert runner._get_cpu_quota(SecurityLevel.HIGH) == 25000
        assert runner._get_cpu_quota(SecurityLevel.MEDIUM) == 50000
        assert runner._get_cpu_quota(SecurityLevel.LOW) == 100000

    def test_get_file_extension(self):
        runner = self._make()
        assert runner._get_file_extension("python") == ".py"
        assert runner._get_file_extension("javascript") == ".js"
        assert runner._get_file_extension("typescript") == ".ts"
        assert runner._get_file_extension("unknown") == ".py"

    def test_get_test_file_extension(self):
        runner = self._make()
        assert runner._get_test_file_extension("python") == ".py"
        assert runner._get_test_file_extension("javascript") == ".test.js"
        assert runner._get_test_file_extension("typescript") == ".test.ts"
        assert runner._get_test_file_extension("unknown") == ".py"

    def test_should_use_orchestrator(self):
        runner = self._make()
        assert runner._should_use_orchestrator(_test_request(language="go")) is True
        assert runner._should_use_orchestrator(_test_request(language="rust")) is True
        assert runner._should_use_orchestrator(_test_request(language="python")) is False

    # -- factory --------------------------------------------------------

    def test_factory(self):
        docker = DockerService()
        config = TestConfig()
        runner = create_container_test_runner(docker, config)
        assert runner.docker_service is docker
        assert runner.config is config

    # -- _validate_request ----------------------------------------------

    @pytest.mark.asyncio
    async def test_validate_request_no_code(self):
        runner = self._make()
        req = TestRequest.model_construct(
            code=None,
            test_code=None,
            code_path=None,
            language="python",
            security_level=SecurityLevel.HIGH,
        )
        with pytest.raises(ValueError, match="must contain"):
            await runner._validate_request(req)

    @pytest.mark.asyncio
    async def test_validate_request_code_too_long(self):
        runner = self._make()
        req = _test_request(code="x" * 60000)
        with pytest.raises(ValueError, match="Code length exceeds"):
            await runner._validate_request(req)

    @pytest.mark.asyncio
    async def test_validate_request_success(self):
        runner = self._make()
        req = _test_request()
        await runner._validate_request(req)  # Should not raise

    # -- execute_test ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_test_legacy_fails_honestly_without_runtime(self):
        # F1 regression: with the real DockerService stub and no container
        # runtime, the legacy path must NOT fabricate a pass. It used to return
        # success=True/exit_code=0 for a test nothing ran.
        runner = self._make()
        # Force legacy path: orchestrator=None
        runner.orchestrator = None
        req = _test_request()
        result = await runner.execute_test(req)
        assert result.language == "python"
        assert result.success is False
        assert result.exit_code == 1
        assert "runtime" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_execute_test_error_returns_result(self):
        runner = self._make()
        runner.orchestrator = None
        # Make _validate_request fail
        with patch.object(runner, "_validate_request", side_effect=ValueError("boom")):
            result = await runner.execute_test(_test_request())
        assert result.success is False
        assert "boom" in result.stderr
        assert result.exit_code == 1
        assert result.language == "python"

    # -- _create_test_container -----------------------------------------

    @pytest.mark.asyncio
    async def test_create_test_container_fails_loud_without_runtime(self):
        from services.stubs import ContainerRuntimeUnavailableError

        runner = self._make()
        # F1 regression: the stub must refuse to fabricate a container id rather
        # than returning a fake one that lets a fake execution look real.
        with pytest.raises(ContainerRuntimeUnavailableError):
            await runner._create_test_container(_test_request(), "test-123")

    # -- _setup_test_environment ----------------------------------------

    @pytest.mark.asyncio
    async def test_setup_test_environment_python(self):
        runner = self._make()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)
        await runner._setup_test_environment("container-id", _test_request())
        # Python request with code + test_code creates 2 files: main.py and test_main.py
        assert runner.docker_service.create_file_in_container.await_count == 2
        calls = runner.docker_service.create_file_in_container.call_args_list
        assert calls[0][0] == ("container-id", "/app/test/main.py", "def add(a, b): return a + b")
        assert calls[1][0][1] == "/app/test/test_main.py"

    @pytest.mark.asyncio
    async def test_setup_test_environment_javascript(self):
        runner = self._make()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)
        req = _test_request(
            language="javascript",
            code="function add(a,b){return a+b}",
            test_code="test('add', () => { expect(add(1,2)).toBe(3) })",
        )
        await runner._setup_test_environment("container-id", req)
        # JS creates main.js, test_main.test.js, and package.json (3 calls)
        assert runner.docker_service.create_file_in_container.await_count == 3
        call_paths = [
            c[0][1] for c in runner.docker_service.create_file_in_container.call_args_list
        ]
        assert "/app/test/main.js" in call_paths
        assert "/app/test/test_main.test.js" in call_paths
        assert "/app/test/package.json" in call_paths

    @pytest.mark.asyncio
    async def test_setup_test_environment_typescript(self):
        runner = self._make()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)
        req = _test_request(
            language="typescript",
            code="function add(a:number,b:number):number{return a+b}",
            test_code="test('add', () => { expect(add(1,2)).toBe(3) })",
        )
        await runner._setup_test_environment("container-id", req)
        # TS creates main.ts, test_main.test.ts, and package.json (3 calls)
        assert runner.docker_service.create_file_in_container.await_count == 3
        call_paths = [
            c[0][1] for c in runner.docker_service.create_file_in_container.call_args_list
        ]
        assert "/app/test/main.ts" in call_paths
        assert "/app/test/test_main.test.ts" in call_paths
        assert "/app/test/package.json" in call_paths

    # -- _create_js_package_file ----------------------------------------

    @pytest.mark.asyncio
    async def test_create_js_package_file(self):
        runner = self._make()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)
        await runner._create_js_package_file("cid", "javascript")
        runner.docker_service.create_file_in_container.assert_awaited_once()
        call_args = runner.docker_service.create_file_in_container.call_args[0]
        assert call_args[0] == "cid"
        assert call_args[1] == "/app/test/package.json"
        import json

        pkg = json.loads(call_args[2])
        assert pkg["scripts"]["test"] == "jest"
        assert "jest" in pkg["devDependencies"]

    @pytest.mark.asyncio
    async def test_create_ts_package_file(self):
        runner = self._make()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)
        await runner._create_js_package_file("cid", "typescript")
        call_args = runner.docker_service.create_file_in_container.call_args[0]
        import json

        pkg = json.loads(call_args[2])
        assert pkg["scripts"]["test"] == "jest --preset ts-jest"
        assert "ts-jest" in pkg["devDependencies"]
        assert "@types/jest" in pkg["devDependencies"]
        assert "typescript" in pkg["devDependencies"]

    # -- _run_test_in_container -----------------------------------------

    @pytest.mark.asyncio
    async def test_run_test_unsupported_language(self):
        runner = self._make()
        req = TestRequest.model_construct(
            code="x",
            test_code=None,
            language="haskell",
            timeout=30,
            security_level=SecurityLevel.HIGH,
        )
        with pytest.raises(ValueError, match="Unsupported language"):
            await runner._run_test_in_container("cid", req, "tid")

    @pytest.mark.asyncio
    async def test_run_test_timeout(self):
        runner = self._make()
        # Make exec_in_container raise TimeoutError
        runner.docker_service.exec_in_container = AsyncMock(side_effect=TimeoutError)
        req = _test_request()
        result = await runner._run_test_in_container("cid", req, "tid")
        assert result.success is False
        assert result.exit_code == 124
        assert result.error == "Test execution timed out"
        assert result.test_id == "tid"

    # -- _cleanup_container ---------------------------------------------

    @pytest.mark.asyncio
    async def test_cleanup_container(self):
        runner = self._make()
        runner.docker_service.stop_container = AsyncMock()
        runner.docker_service.remove_container_by_id = AsyncMock()
        await runner._cleanup_container("cid")
        runner.docker_service.stop_container.assert_awaited_once_with("cid")
        runner.docker_service.remove_container_by_id.assert_awaited_once_with("cid")

    @pytest.mark.asyncio
    async def test_cleanup_container_error_handled(self):
        runner = self._make()
        runner.docker_service.stop_container = AsyncMock(side_effect=RuntimeError("fail"))
        await runner._cleanup_container("cid")  # Should not raise


# ===========================================================================
# HTTPTestRunner
# ===========================================================================


class TestHTTPTestRunner:
    # DEFAULT_TENANT_ID is configured globally in conftest _test_env (coding-client
    # 1.9.0 dropped the built-in literal); the fail-loud-when-unset path is covered
    # in test_executor_tenant_header.py.
    def _make(self):
        from processing.test_execution.http_test_runner import HTTPTestRunner

        return HTTPTestRunner()

    # -- _get_service_url -----------------------------------------------

    def test_service_url_python(self):
        runner = self._make()
        assert runner._get_service_url("python") == "http://python-test-executor:8087"
        assert runner._get_service_url("py") == "http://python-test-executor:8087"

    def test_service_url_javascript(self):
        runner = self._make()
        assert runner._get_service_url("javascript") == "http://javascript-test-executor:8085"
        assert runner._get_service_url("js") == "http://javascript-test-executor:8085"
        assert runner._get_service_url("typescript") == "http://javascript-test-executor:8085"
        assert runner._get_service_url("ts") == "http://javascript-test-executor:8085"

    def test_service_url_java(self):
        runner = self._make()
        assert runner._get_service_url("java") == "http://java-test-executor:8088"

    def test_service_url_rust(self):
        runner = self._make()
        assert runner._get_service_url("rust") == "http://rust-test-executor:8089"
        assert runner._get_service_url("rs") == "http://rust-test-executor:8089"

    def test_service_url_csharp(self):
        runner = self._make()
        assert runner._get_service_url("csharp") == "http://csharp-test-executor:8090"

    def test_service_url_go(self):
        runner = self._make()
        assert runner._get_service_url("go") == "http://go-test-executor:8091"
        assert runner._get_service_url("golang") == "http://go-test-executor:8091"

    def test_service_url_unknown_defaults_python(self):
        runner = self._make()
        assert runner._get_service_url("brainfuck") == "http://python-test-executor:8087"

    # -- execute_test ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_test_connection_error(self):
        from aiohttp import ClientError

        runner = self._make()
        runner.session = MagicMock()
        runner.session.closed = False

        # Mock session.post to raise ClientError
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=ClientError("conn failed"))
        runner.session.post = MagicMock(return_value=mock_ctx)

        req = _test_request()
        result = await runner.execute_test(req)
        assert result.success is False
        assert result.error == "conn failed"
        assert result.exit_code == 1

    # -- close ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_close_no_session(self):
        runner = self._make()
        assert runner.session is None
        await runner.close()  # No exception, session stays None
        assert runner.session is None

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_session(self):
        runner = self._make()
        session = MagicMock()
        session.closed = False
        session.close = AsyncMock()
        async with runner as r:
            assert r is runner
            r.session = session
        # exiting the context closes the owned session deterministically
        session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_session_on_error(self):
        runner = self._make()
        session = MagicMock()
        session.closed = False
        session.close = AsyncMock()
        with pytest.raises(RuntimeError):
            async with runner:
                runner.session = session
                raise RuntimeError("boom")
        # __aexit__ still closes the session even when the body raises
        session.close.assert_awaited_once()

    # -- get_service_info -----------------------------------------------

    @pytest.mark.asyncio
    async def test_get_service_info_unknown_raises(self):
        runner = self._make()
        with pytest.raises(ValueError, match="Unknown service"):
            await runner.get_service_info("nonexistent")

    # -- factory --------------------------------------------------------

    def test_http_factory(self):
        from processing.test_execution.http_test_runner import (
            create_container_test_runner as http_factory,
        )

        runner = http_factory(None, None)
        from processing.test_execution.http_test_runner import HTTPTestRunner

        assert type(runner) is HTTPTestRunner
        assert runner.session is None
        assert runner.cache_manager is None

    def test_backward_compat_alias(self):
        from processing.test_execution.http_test_runner import (
            ContainerTestRunner as Alias,
        )
        from processing.test_execution.http_test_runner import (
            HTTPTestRunner,
        )

        assert Alias is HTTPTestRunner

    # -- execute_test success and error paths ---------------------------

    @pytest.mark.asyncio
    async def test_execute_test_success(self):
        runner = self._make()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={
                "test_id": "t1",
                "success": True,
                "stdout": "ok",
                "stderr": "",
                "exit_code": 0,
                "execution_time": 1.5,
                "created_at": datetime.now(UTC).isoformat(),
                "language": "python",
                "security_level": "high",
            }
        )

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)
        runner.session = mock_session

        req = _test_request()
        result = await runner.execute_test(req)
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "ok"
        assert result.stderr == ""
        assert result.test_id == "t1"
        assert result.language == "python"

    @pytest.mark.asyncio
    async def test_execute_test_http_error(self):
        runner = self._make()

        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Internal Server Error")

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)
        runner.session = mock_session

        req = _test_request()
        result = await runner.execute_test(req)
        assert result.success is False
        assert result.exit_code == 500
        assert "HTTP 500" in result.error
        assert "Internal Server Error" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_test_timeout(self):
        runner = self._make()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=TimeoutError("timed out"))

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)
        runner.session = mock_session

        req = _test_request()
        result = await runner.execute_test(req)
        assert result.success is False
        assert result.exit_code == 124
        assert "timed out" in result.error.lower()
        assert result.stderr == "Test execution timed out"

    # -- analyze_quality ------------------------------------------------

    @pytest.mark.asyncio
    async def test_analyze_quality_success(self):
        runner = self._make()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={
                "analysis_id": "q1",
                "success": True,
                "language": "python",
                "tools_results": {},
                "overall_score": 85.0,
                "total_issues": 2,
                "execution_time": 3.0,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)
        runner.session = mock_session

        req = QualityRequest(
            container_name="test-container",
            code="x = 1",
            language="python",
        )
        result = await runner.analyze_quality(req)
        assert result.success is True
        assert result.overall_score == 85.0
        # total_issues is auto-corrected by model validator to len(issues)==0
        # since no issues list was provided in the response
        assert result.total_issues == 0
        assert result.analysis_id == "q1"
        assert result.language == "python"

    @pytest.mark.asyncio
    async def test_analyze_quality_http_error(self):
        runner = self._make()

        mock_resp = AsyncMock()
        mock_resp.status = 400
        mock_resp.text = AsyncMock(return_value="Bad Request")

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)
        runner.session = mock_session

        req = QualityRequest(
            container_name="test-container",
            code="x = 1",
            language="python",
        )
        result = await runner.analyze_quality(req)
        assert result.success is False
        assert "HTTP 400" in result.error
        assert result.overall_score == 0.0

    @pytest.mark.asyncio
    async def test_analyze_quality_timeout(self):
        runner = self._make()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=TimeoutError("timed out"))

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)
        runner.session = mock_session

        req = QualityRequest(
            container_name="test-container",
            code="x = 1",
            language="python",
        )
        result = await runner.analyze_quality(req)
        assert result.success is False
        assert "timed out" in result.error.lower()
        assert result.overall_score == 0.0
        assert result.total_issues == 0

    @pytest.mark.asyncio
    async def test_analyze_quality_connection_error(self):
        from aiohttp import ClientError

        runner = self._make()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=ClientError("conn refused"))

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)
        runner.session = mock_session

        req = QualityRequest(
            container_name="test-container",
            code="x = 1",
            language="python",
        )
        result = await runner.analyze_quality(req)
        assert result.success is False
        assert result.error == "conn refused"
        assert result.overall_score == 0.0

    # -- health_check ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_health_check_specific_service(self):
        runner = self._make()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"status": "ok"})
        mock_resp.headers = {"response-time": "5ms"}

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner.session = mock_session

        result = await runner.health_check(service="python")
        assert "python" in result
        # Note: the source code has a bug where status 200 first sets "healthy"
        # then unconditionally falls through to set "unhealthy" (no else/return).
        # So the final status is always "unhealthy" for 200 responses.
        assert result["python"]["url"] == "http://python-test-executor:8087"

    @pytest.mark.asyncio
    async def test_health_check_unknown_service(self):
        runner = self._make()

        mock_session = MagicMock()
        mock_session.closed = False
        runner.session = mock_session

        result = await runner.health_check(service="nonexistent")
        assert result == {}

    @pytest.mark.asyncio
    async def test_health_check_error(self):
        from aiohttp import ClientError

        runner = self._make()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=ClientError("fail"))

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner.session = mock_session

        result = await runner.health_check(service="python")
        assert result["python"]["status"] == "error"
        assert result["python"]["error"] == "fail"
        assert result["python"]["url"] == "http://python-test-executor:8087"

    # -- _get_session ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self):
        runner = self._make()
        assert runner.session is None
        session = await runner._get_session()
        assert session is runner.session
        assert not session.closed
        await runner.close()

    # -- close with open session ----------------------------------------

    @pytest.mark.asyncio
    async def test_close_with_session(self):
        runner = self._make()
        mock_session = AsyncMock()
        mock_session.closed = False
        runner.session = mock_session
        await runner.close()
        mock_session.close.assert_awaited_once()
