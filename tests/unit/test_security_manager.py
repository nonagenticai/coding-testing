"""Tests for ExecutionSecurityManager in security_manager.py."""

from __future__ import annotations

import pytest

from config.test_framework_security_manager import (
    SecurityViolationType,
    TestValidationSecurityConfig,
)
from models.testing_models import SecurityLevel, TestRequest, TestSecurityLevel
from services.test_execution.security_manager import (
    ContainerSecurityConfig,
    ContainerSecurityViolationType,
    ExecutionSecurityManager,
    create_execution_security_manager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def security_config():
    return TestValidationSecurityConfig(
        max_execution_time=300,
        max_memory_mb=512,
        max_file_size_mb=10,
        max_code_length=50000,
        allowed_modules=["pytest", "unittest", "math"],
    )


@pytest.fixture
def manager(security_config):
    return ExecutionSecurityManager(security_config)


@pytest.fixture
def python_request():
    return TestRequest(
        code="import math\ndef add(a,b): return a+b",
        test_code="import pytest\ndef test_add(): assert add(1,2)==3",
        language="python",
        timeout=60,
        security_level=TestSecurityLevel.MEDIUM,
    )


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class TestContainerSecurityViolationType:
    def test_values(self):
        assert ContainerSecurityViolationType.DANGEROUS_PATTERN == "dangerous_pattern"
        assert ContainerSecurityViolationType.NETWORK_OPERATION == "network_operation"
        assert ContainerSecurityViolationType.RESOURCE_LIMIT_EXCEEDED == "resource_limit_exceeded"


# ---------------------------------------------------------------------------
# ContainerSecurityConfig
# ---------------------------------------------------------------------------


class TestContainerSecurityConfig:
    def test_defaults(self):
        cfg = ContainerSecurityConfig()
        assert cfg.max_memory_mb == 512
        assert cfg.max_cpu_quota == 50000
        assert cfg.network_disabled is True
        assert cfg.read_only_filesystem is True
        assert cfg.drop_all_capabilities is True
        assert cfg.no_new_privileges is True

    def test_allowed_modules_validation_empty(self):
        with pytest.raises(ValueError, match="At least one module"):
            ContainerSecurityConfig(allowed_modules=[])

    def test_allowed_modules_dedup(self):
        cfg = ContainerSecurityConfig(allowed_modules=["pytest", "pytest", "math"])
        assert len(cfg.allowed_modules) == 2

    def test_cpu_quota_validation(self):
        with pytest.raises(ValueError):
            ContainerSecurityConfig(max_cpu_quota=0)

    def test_cpu_quota_upper_bound(self):
        with pytest.raises(ValueError):
            ContainerSecurityConfig(max_cpu_quota=100001)


# ---------------------------------------------------------------------------
# ExecutionSecurityManager init
# ---------------------------------------------------------------------------


class TestManagerInit:
    def test_creates_container_config(self, manager):
        assert isinstance(manager.container_config, ContainerSecurityConfig)

    def test_loads_dangerous_patterns(self, manager):
        assert "python" in manager.dangerous_patterns
        assert "go" in manager.dangerous_patterns
        assert "rust" in manager.dangerous_patterns
        assert len(manager.dangerous_patterns["python"]) > 0

    def test_factory_function(self):
        cfg = ContainerSecurityConfig(allowed_modules=["pytest"])
        mgr = create_execution_security_manager(cfg)
        assert isinstance(mgr, ExecutionSecurityManager)


# ---------------------------------------------------------------------------
# validate_and_sanitize
# ---------------------------------------------------------------------------


class TestValidateAndSanitize:
    @pytest.mark.asyncio
    async def test_valid_python_request(self, manager, python_request):
        result = await manager.validate_and_sanitize(python_request)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_code_too_long(self, manager):
        req = TestRequest(
            code="x" * 50001,
            language="python",
            timeout=60,
        )
        result = await manager.validate_and_sanitize(req)
        assert result.passed is False
        assert result.violation_type == SecurityViolationType.CODE_LENGTH

    @pytest.mark.asyncio
    async def test_timeout_exceeds_max(self, manager):
        # TestRequest.timeout has a Pydantic max of 300, so use a lower config
        mgr = ExecutionSecurityManager(
            TestValidationSecurityConfig(max_execution_time=60, allowed_modules=["pytest"])
        )
        req = TestRequest(code="x=1", language="python", timeout=61)
        result = await mgr.validate_and_sanitize(req)
        assert result.passed is False
        assert result.violation_type == SecurityViolationType.EXECUTION_TIME

    @pytest.mark.asyncio
    async def test_no_code_passes(self, manager):
        req = TestRequest(language="python")
        result = await manager.validate_and_sanitize(req)
        assert result.passed is True


# ---------------------------------------------------------------------------
# _validate_code_security — dangerous pattern matching
# ---------------------------------------------------------------------------


class TestDangerousPatterns:
    @pytest.mark.asyncio
    async def test_python_os_system(self, manager):
        result = await manager._validate_code_security("os.system('ls')", "python")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_python_subprocess(self, manager):
        result = await manager._validate_code_security("import subprocess", "python")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_python_eval(self, manager):
        result = await manager._validate_code_security("eval('1+1')", "python")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_go_exec(self, manager):
        result = await manager._validate_code_security("exec.Command('ls')", "go")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_rust_unsafe(self, manager):
        result = await manager._validate_code_security("unsafe { ptr::null() }", "rust")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_java_runtime_exec(self, manager):
        result = await manager._validate_code_security("Runtime.getRuntime().exec(cmd)", "java")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_csharp_process_start(self, manager):
        result = await manager._validate_code_security("Process.Start('cmd')", "csharp")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_cpp_system(self, manager):
        result = await manager._validate_code_security('system("rm -rf /")', "cpp")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_js_eval(self, manager):
        result = await manager._validate_code_security("eval('alert(1)')", "javascript")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_safe_python_code(self, manager):
        code = "import math\ndef f(x): return math.sqrt(x)"
        result = await manager._validate_code_security(code, "python")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unknown_language_passes(self, manager):
        result = await manager._validate_code_security("anything", "elixir")
        assert result.passed is True


# ---------------------------------------------------------------------------
# _validate_python_code — import checking
# ---------------------------------------------------------------------------


class TestValidatePythonCode:
    @pytest.mark.asyncio
    async def test_allowed_import(self, manager):
        result = await manager._validate_python_code("import pytest")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unauthorized_import(self, manager):
        result = await manager._validate_python_code("import os")
        assert result.passed is False
        assert result.violation_type == SecurityViolationType.UNAUTHORIZED_IMPORT

    @pytest.mark.asyncio
    async def test_unauthorized_from_import(self, manager):
        result = await manager._validate_python_code("from os import path")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_syntax_error(self, manager):
        result = await manager._validate_python_code("def f(:")
        assert result.passed is False
        assert result.violation_type == SecurityViolationType.SYNTAX_ERROR


# ---------------------------------------------------------------------------
# _validate_javascript_code
# ---------------------------------------------------------------------------


class TestValidateJavaScriptCode:
    @pytest.mark.asyncio
    async def test_dangerous_eval(self, manager):
        result = await manager._validate_javascript_code("eval('x')")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_dangerous_function_constructor(self, manager):
        result = await manager._validate_javascript_code("new Function('return 1')")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_set_timeout(self, manager):
        result = await manager._validate_javascript_code("setTimeout(fn, 100)")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_safe_code(self, manager):
        result = await manager._validate_javascript_code("const x = 1 + 2;")
        assert result.passed is True


# ---------------------------------------------------------------------------
# _validate_go_code
# ---------------------------------------------------------------------------


class TestValidateGoCode:
    @pytest.mark.asyncio
    async def test_unsafe_import(self, manager):
        result = await manager._validate_go_code('import "unsafe"')
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_syscall_import(self, manager):
        result = await manager._validate_go_code('import "syscall"')
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_safe_code(self, manager):
        result = await manager._validate_go_code('import "fmt"\nfunc main() {}')
        assert result.passed is True


# ---------------------------------------------------------------------------
# _validate_rust_code
# ---------------------------------------------------------------------------


class TestValidateRustCode:
    @pytest.mark.asyncio
    async def test_unsafe_block(self, manager):
        result = await manager._validate_rust_code("unsafe { }")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_safe_code(self, manager):
        result = await manager._validate_rust_code('fn main() { println!("hi"); }')
        assert result.passed is True


# ---------------------------------------------------------------------------
# _validate_java_code
# ---------------------------------------------------------------------------


class TestValidateJavaCode:
    @pytest.mark.asyncio
    async def test_runtime_exec(self, manager):
        result = await manager._validate_java_code("Runtime.getRuntime")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_system_exit(self, manager):
        result = await manager._validate_java_code("System.exit(0)")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_safe_code(self, manager):
        result = await manager._validate_java_code("int x = 1 + 2;")
        assert result.passed is True


# ---------------------------------------------------------------------------
# _validate_csharp_code
# ---------------------------------------------------------------------------


class TestValidateCSharpCode:
    @pytest.mark.asyncio
    async def test_process_start(self, manager):
        result = await manager._validate_csharp_code('Process.Start("cmd")')
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_environment_exit(self, manager):
        result = await manager._validate_csharp_code("Environment.Exit(0)")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_marshal(self, manager):
        result = await manager._validate_csharp_code("Marshal.Copy()")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_safe_code(self, manager):
        result = await manager._validate_csharp_code("var x = 1;")
        assert result.passed is True


# ---------------------------------------------------------------------------
# _validate_cpp_code
# ---------------------------------------------------------------------------


class TestValidateCppCode:
    @pytest.mark.asyncio
    async def test_system_call(self, manager):
        result = await manager._validate_cpp_code('system("ls")')
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_popen(self, manager):
        result = await manager._validate_cpp_code('popen("cmd", "r")')
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_safe_code(self, manager):
        result = await manager._validate_cpp_code("int x = 42;")
        assert result.passed is True


# ---------------------------------------------------------------------------
# get_container_security_config
# ---------------------------------------------------------------------------


class TestGetContainerSecurityConfig:
    def test_high_security(self, manager):
        cfg = manager.get_container_security_config(SecurityLevel.HIGH)
        assert cfg["read_only"] is True
        assert cfg["network_disabled"] is True
        assert "ALL" in cfg["cap_drop"]
        # Memory should be capped at 256
        mem_value = int(cfg["mem_limit"].rstrip("m"))
        assert mem_value <= 256

    def test_medium_security(self, manager):
        cfg = manager.get_container_security_config(SecurityLevel.MEDIUM)
        assert "mem_limit" in cfg
        assert cfg["read_only"] is True

    def test_low_security(self, manager):
        cfg = manager.get_container_security_config(SecurityLevel.LOW)
        assert cfg["read_only"] is False
        mem_value = int(cfg["mem_limit"].rstrip("m"))
        assert mem_value > 512


# ---------------------------------------------------------------------------
# validate_container_runtime_security
# ---------------------------------------------------------------------------


class TestValidateContainerRuntimeSecurity:
    def test_valid_config(self, manager):
        config = {
            "network_disabled": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "user": "testuser",
            "mem_limit": "256m",
            "cpu_quota": 25000,
        }
        assert manager.validate_container_runtime_security(config) is True

    def test_network_enabled_fails(self, manager):
        config = {
            "network_disabled": False,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "user": "testuser",
        }
        assert manager.validate_container_runtime_security(config) is False

    def test_missing_cap_drop_fails(self, manager):
        config = {
            "network_disabled": True,
            "cap_drop": [],
            "security_opt": ["no-new-privileges:true"],
            "user": "testuser",
        }
        assert manager.validate_container_runtime_security(config) is False

    def test_missing_no_new_privileges_fails(self, manager):
        config = {
            "network_disabled": True,
            "cap_drop": ["ALL"],
            "security_opt": [],
            "user": "testuser",
        }
        assert manager.validate_container_runtime_security(config) is False

    def test_wrong_user_fails(self, manager):
        config = {
            "network_disabled": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "user": "root",
        }
        assert manager.validate_container_runtime_security(config) is False

    def test_memory_exceeds_max_fails(self, manager):
        config = {
            "network_disabled": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "user": "testuser",
            "mem_limit": "99999m",
        }
        assert manager.validate_container_runtime_security(config) is False

    def test_cpu_exceeds_max_fails(self, manager):
        config = {
            "network_disabled": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "user": "testuser",
            "cpu_quota": 999999,
        }
        assert manager.validate_container_runtime_security(config) is False

    def test_invalid_mem_format_fails(self, manager):
        config = {
            "network_disabled": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "user": "testuser",
            "mem_limit": "abc",
        }
        assert manager.validate_container_runtime_security(config) is False


# ---------------------------------------------------------------------------
# get_security_report
# ---------------------------------------------------------------------------


class TestValidateAndSanitizeCodePaths:
    """Tests for validate_and_sanitize returning failed code/test_code results."""

    @pytest.mark.asyncio
    async def test_dangerous_source_code_fails_validation(self, manager):
        """Covers line 238: code_result returned when code fails."""
        req = TestRequest(
            code="os.system('rm -rf /')",
            language="python",
            timeout=60,
        )
        result = await manager.validate_and_sanitize(req)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_dangerous_test_code_fails_validation(self, manager):
        """Covers line 246: test_result returned when test_code fails."""
        req = TestRequest(
            code="x = 1",
            test_code="import subprocess; subprocess.call('ls')",
            language="python",
            timeout=60,
        )
        result = await manager.validate_and_sanitize(req)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_validate_and_sanitize_exception_handler(self, security_config):
        """Covers lines 259-261: exception in validate_and_sanitize."""
        mgr = ExecutionSecurityManager(security_config)
        # Force an exception by making code property raise
        req = TestRequest(code="x=1", language="python")
        # Monkey-patch dangerous_patterns to raise on access
        original = mgr.dangerous_patterns
        mgr.dangerous_patterns = property(lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
        # Since dangerous_patterns is an instance attr, we need a different approach
        mgr.dangerous_patterns = original

        # Force exception via _validate_code_security raising
        async def _raise(*a, **k):
            raise RuntimeError("forced")

        mgr._validate_code_security = _raise
        result = await mgr.validate_and_sanitize(req)
        assert result.passed is False
        assert "Security validation error" in result.error


class TestValidateCodeSecurityLanguageDispatching:
    """Test _validate_code_security dispatches to language-specific validators.
    Covers lines 280, 284, 286, 288, 290 via safe code paths.
    """

    @pytest.mark.asyncio
    async def test_javascript_dispatch_safe(self, manager):
        """Covers line 280."""
        result = await manager._validate_code_security("const x = 1;", "javascript")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_go_dispatch_safe(self, manager):
        """Covers line 282 (already covered but reinforces line 282->_validate_go_code)."""
        result = await manager._validate_code_security('func main() { fmt.Println("hi") }', "go")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_rust_dispatch_safe(self, manager):
        """Covers line 284."""
        result = await manager._validate_code_security("fn main() { let x = 1; }", "rust")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_java_dispatch_safe(self, manager):
        """Covers line 286."""
        result = await manager._validate_code_security("int x = 42;", "java")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_csharp_dispatch_safe(self, manager):
        """Covers line 288."""
        result = await manager._validate_code_security("var x = 42;", "csharp")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_cpp_dispatch_safe(self, manager):
        """Covers line 290."""
        result = await manager._validate_code_security("int x = 42;", "cpp")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_validate_code_security_exception_handler(self, manager):
        """Covers lines 294-296: exception in _validate_code_security."""

        # Force exception by making _validate_python_code raise
        async def _raise(*a, **k):
            raise RuntimeError("forced error")

        manager._validate_python_code = _raise
        result = await manager._validate_code_security("x = 1", "python")
        assert result.passed is False
        assert "Code validation error" in result.error


class TestValidateContainerRuntimeSecurityException:
    """Covers lines 505-507: exception handler in validate_container_runtime_security."""

    def test_exception_returns_false(self, manager):
        """Force an exception by passing a config that causes error in processing."""
        # A config where mem_limit is not a string will trigger TypeError
        config = {
            "network_disabled": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "user": "testuser",
            "mem_limit": 12345,  # int, not str — .rstrip() will raise AttributeError
        }
        result = manager.validate_container_runtime_security(config)
        assert result is False


class TestGetSecurityReport:
    def test_report_structure(self, manager):
        report = manager.get_security_report()
        assert report["security_manager"] == "ExecutionSecurityManager"
        assert "config" in report
        assert "dangerous_patterns" in report
        assert "allowed_modules" in report
        assert "security_features" in report
        assert report["config"]["max_memory_mb"] == 512
        assert isinstance(report["dangerous_patterns"]["python"], int)
