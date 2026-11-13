"""
Test Container Security Manager - Phase 1 Implementation
Advanced security validation and enforcement for multi-language test execution containers.
"""

from __future__ import annotations

import ast
import logging
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from config.test_framework_security_manager import (
    SecurityCheckResult,
    SecurityViolationType,
    TestValidationSecurityConfig,
)
from models.testing_models import SecurityLevel, TestRequest

logger = logging.getLogger(__name__)


class ContainerSecurityViolationType(StrEnum):
    """Container-specific security violation types."""

    DANGEROUS_PATTERN = "dangerous_pattern"
    FILE_OPERATION = "file_operation"
    NETWORK_OPERATION = "network_operation"
    SYSTEM_CALL = "system_call"
    UNSAFE_CONTAINER_CONFIG = "unsafe_container_config"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    UNAUTHORIZED_RUNTIME = "unauthorized_runtime"


class ContainerSecurityConfig(BaseModel):
    """Container-specific security configuration."""

    # Resource limits
    max_memory_mb: int = Field(default=512, gt=0, description="Maximum memory in MB")
    max_cpu_quota: int = Field(default=50000, gt=0, le=100000, description="Maximum CPU quota")
    max_execution_time: int = Field(
        default=300, gt=0, description="Maximum execution time in seconds"
    )
    max_file_size_mb: int = Field(default=10, gt=0, description="Maximum file size in MB")
    max_code_length: int = Field(
        default=50000, gt=0, description="Maximum code length in characters"
    )

    # Security restrictions
    network_disabled: bool = Field(default=True, description="Whether network should be disabled")
    read_only_filesystem: bool = Field(
        default=True, description="Whether filesystem should be read-only"
    )
    drop_all_capabilities: bool = Field(
        default=True, description="Whether to drop all Linux capabilities"
    )
    no_new_privileges: bool = Field(
        default=True, description="Whether to prevent privilege escalation"
    )

    # Allowed operations
    allowed_modules: list[str] = Field(
        default_factory=lambda: ["pytest", "unittest", "math"],
        description="Allowed Python modules",
    )
    allowed_file_operations: list[str] = Field(
        default_factory=lambda: ["read", "write_temp"],
        description="Allowed file operations",
    )
    allowed_network_operations: list[str] = Field(
        default_factory=list, description="Allowed network operations"
    )

    # Language-specific patterns
    dangerous_patterns: dict[str, list[str]] = Field(
        default_factory=dict, description="Dangerous code patterns by language"
    )

    @field_validator("allowed_modules")
    @classmethod
    def validate_allowed_modules(cls, v: list[str]) -> list[str]:
        """Validate and deduplicate allowed modules."""
        if not v:
            raise ValueError("At least one module must be allowed")
        return list(set(v))

    @field_validator("max_cpu_quota")
    @classmethod
    def validate_cpu_quota(cls, v: int) -> int:
        """Validate CPU quota is within Docker limits."""
        if v <= 0 or v > 100000:
            raise ValueError("CPU quota must be between 1 and 100000")
        return v


class ExecutionSecurityManager:
    """
    Security manager for test execution containers.

    Provides comprehensive security validation and enforcement for multi-language
    test execution environments, including code analysis, container configuration
    validation, and runtime security monitoring.
    """

    def __init__(self, config: TestValidationSecurityConfig | ContainerSecurityConfig):
        """Initialize container security manager."""
        self.config = config
        self.container_config = self._create_container_config()
        self.dangerous_patterns = self._load_dangerous_patterns()
        logger.info("ExecutionSecurityManager initialized")

    def _create_container_config(self) -> ContainerSecurityConfig:
        """Create container security configuration from base config."""
        return ContainerSecurityConfig(
            max_memory_mb=getattr(self.config, "max_memory_mb", 512),
            max_execution_time=getattr(self.config, "max_execution_time", 300),
            max_file_size_mb=getattr(self.config, "max_file_size_mb", 10),
            max_code_length=getattr(self.config, "max_code_length", 50000),
            allowed_modules=getattr(self.config, "allowed_modules", ["pytest", "unittest", "math"]),
        )

    def _load_dangerous_patterns(self) -> dict[str, list[str]]:
        """Load dangerous code patterns for each language."""
        return {
            "go": [
                r"os\.Exec",
                r"exec\.Command",
                r"syscall\.",
                r"unsafe\.",
                r"//go:linkname",
                r"runtime\.GC",
                r"runtime\.GOMAXPROCS",
                r"net\.",
                r"http\.",
            ],
            "rust": [
                r"unsafe\s*{",
                r"std::process::Command",
                r"std::ptr::",
                r"libc::",
                r"#\[no_mangle\]",
                r"std::fs::",
                r"std::net::",
                r"tokio::net::",
            ],
            "java": [
                r"Runtime\.getRuntime\(\)\.exec",
                r"ProcessBuilder",
                r"System\.exit",
                r"native\s+",
                r"sun\.",
                r"java\.io\.File",
                r"java\.net\.",
                r"java\.nio\.file\.",
            ],
            "csharp": [
                r"Process\.Start",
                r"DllImport",
                r"unsafe\s*{",
                r"Marshal\.",
                r"Environment\.Exit",
                r"System\.IO\.File",
                r"System\.Net\.",
                r"System\.Diagnostics\.",
            ],
            "cpp": [
                r"system\s*\(",
                r"exec[lv]p?\s*\(",
                r"popen\s*\(",
                r"#include\s*<windows\.h>",
                r"asm\s*\(",
                r"<fstream>",
                r"<iostream>",
                r"<socket\.h>",
            ],
            "python": [
                r"os\.system",
                r"os\.popen",
                r"subprocess\.",
                r"eval\s*\(",
                r"exec\s*\(",
                r"__import__",
                r"open\s*\(",
                r"socket\.",
                r"urllib\.",
                r"requests\.",
            ],
            "javascript": [
                r"eval\s*\(",
                r"Function\s*\(",
                r"require\s*\(",
                r"process\.",
                r"fs\.",
                r"http\.",
                r"https\.",
                r"net\.",
            ],
            "typescript": [
                r"eval\s*\(",
                r"Function\s*\(",
                r"require\s*\(",
                r"process\.",
                r"fs\.",
                r"http\.",
                r"https\.",
                r"net\.",
            ],
        }

    async def validate_and_sanitize(self, request: TestRequest) -> SecurityCheckResult:
        """
        Validate and sanitize test request for security compliance.

        Args:
            request: Test execution request to validate

        Returns:
            SecurityCheckResult: Result of security validation
        """
        try:
            logger.debug(f"Validating test request for {request.language}")

            # Basic validation
            if request.code and len(request.code) > self.container_config.max_code_length:
                return SecurityCheckResult(
                    passed=False,
                    error=f"Code exceeds maximum length ({self.container_config.max_code_length})",
                    violation_type=SecurityViolationType.CODE_LENGTH,
                )

            # Validate source code
            if request.code:
                code_result = await self._validate_code_security(request.code, request.language)
                if not code_result.passed:
                    return code_result

            # Validate test code if present
            if request.test_code:
                test_result = await self._validate_code_security(
                    request.test_code, request.language
                )
                if not test_result.passed:
                    return test_result

            # Validate execution parameters
            if request.timeout and request.timeout > self.container_config.max_execution_time:
                return SecurityCheckResult(
                    passed=False,
                    error=f"Timeout exceeds maximum ({self.container_config.max_execution_time}s)",
                    violation_type=SecurityViolationType.EXECUTION_TIME,
                )

            logger.debug("Test request validation successful")
            return SecurityCheckResult(passed=True)

        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return SecurityCheckResult(passed=False, error=f"Security validation error: {str(e)}")

    async def _validate_code_security(self, code: str, language: str) -> SecurityCheckResult:
        """Validate code for security issues."""
        try:
            # Check for dangerous patterns
            patterns = self.dangerous_patterns.get(language, [])
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return SecurityCheckResult(
                        passed=False,
                        error=f"Dangerous pattern detected: {pattern}",
                        violation_type=SecurityViolationType.SYNTAX_ERROR,
                    )

            # Language-specific validation
            if language == "python":
                return await self._validate_python_code(code)
            elif language in ["javascript", "typescript"]:
                return await self._validate_javascript_code(code)
            elif language == "go":
                return await self._validate_go_code(code)
            elif language == "rust":
                return await self._validate_rust_code(code)
            elif language == "java":
                return await self._validate_java_code(code)
            elif language == "csharp":
                return await self._validate_csharp_code(code)
            elif language == "cpp":
                return await self._validate_cpp_code(code)

            return SecurityCheckResult(passed=True)

        except Exception as e:
            logger.error(f"Code security validation failed: {e}")
            return SecurityCheckResult(passed=False, error=f"Code validation error: {str(e)}")

    async def _validate_python_code(self, code: str) -> SecurityCheckResult:
        """Validate Python code for security issues."""
        try:
            # Check syntax
            ast.parse(code)

            # Check imports
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split(".")[0]
                        if module_name not in self.container_config.allowed_modules:
                            return SecurityCheckResult(
                                passed=False,
                                error=f"Unauthorized import: {module_name}",
                                violation_type=SecurityViolationType.UNAUTHORIZED_IMPORT,
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split(".")[0]
                        if module_name not in self.container_config.allowed_modules:
                            return SecurityCheckResult(
                                passed=False,
                                error=f"Unauthorized import from: {module_name}",
                                violation_type=SecurityViolationType.UNAUTHORIZED_IMPORT,
                            )

            return SecurityCheckResult(passed=True)

        except SyntaxError as e:
            return SecurityCheckResult(
                passed=False,
                error=f"Python syntax error: {str(e)}",
                violation_type=SecurityViolationType.SYNTAX_ERROR,
            )

    async def _validate_javascript_code(self, code: str) -> SecurityCheckResult:
        """Validate JavaScript/TypeScript code for security issues."""
        # Check for dangerous function calls
        dangerous_functions = ["eval", "Function", "setTimeout", "setInterval"]
        for func in dangerous_functions:
            if re.search(rf"\b{func}\s*\(", code):
                return SecurityCheckResult(
                    passed=False,
                    error=f"Dangerous function call: {func}",
                    violation_type=SecurityViolationType.UNAUTHORIZED_IMPORT,
                )

        return SecurityCheckResult(passed=True)

    async def _validate_go_code(self, code: str) -> SecurityCheckResult:
        """Validate Go code for security issues."""
        # Check for unsafe imports
        unsafe_imports = ["unsafe", "syscall", "os/exec"]
        for imp in unsafe_imports:
            if re.search(rf'import\s+.*"{imp}"', code):
                return SecurityCheckResult(
                    passed=False,
                    error=f"Unsafe import: {imp}",
                    violation_type=SecurityViolationType.UNAUTHORIZED_IMPORT,
                )

        return SecurityCheckResult(passed=True)

    async def _validate_rust_code(self, code: str) -> SecurityCheckResult:
        """Validate Rust code for security issues."""
        # Check for unsafe blocks
        if re.search(r"unsafe\s*{", code):
            return SecurityCheckResult(
                passed=False,
                error="Unsafe block detected",
                violation_type=SecurityViolationType.UNAUTHORIZED_IMPORT,
            )

        return SecurityCheckResult(passed=True)

    async def _validate_java_code(self, code: str) -> SecurityCheckResult:
        """Validate Java code for security issues."""
        # Check for dangerous method calls
        dangerous_methods = [
            "Runtime.getRuntime",
            "System.exit",
            "ProcessBuilder",
        ]
        for method in dangerous_methods:
            if method in code:
                return SecurityCheckResult(
                    passed=False,
                    error=f"Dangerous method call: {method}",
                    violation_type=SecurityViolationType.UNAUTHORIZED_IMPORT,
                )

        return SecurityCheckResult(passed=True)

    async def _validate_csharp_code(self, code: str) -> SecurityCheckResult:
        """Validate C# code for security issues."""
        # Check for dangerous method calls
        dangerous_methods = ["Process.Start", "Environment.Exit", "Marshal."]
        for method in dangerous_methods:
            if method in code:
                return SecurityCheckResult(
                    passed=False,
                    error=f"Dangerous method call: {method}",
                    violation_type=SecurityViolationType.UNAUTHORIZED_IMPORT,
                )

        return SecurityCheckResult(passed=True)

    async def _validate_cpp_code(self, code: str) -> SecurityCheckResult:
        """Validate C++ code for security issues."""
        # Check for dangerous function calls
        dangerous_functions = ["system", "exec", "popen", "asm"]
        for func in dangerous_functions:
            if re.search(rf"\b{func}\s*\(", code):
                return SecurityCheckResult(
                    passed=False,
                    error=f"Dangerous function call: {func}",
                    violation_type=SecurityViolationType.UNAUTHORIZED_IMPORT,
                )

        return SecurityCheckResult(passed=True)

    def get_container_security_config(self, security_level: SecurityLevel) -> dict[str, Any]:
        """Get container security configuration based on security level."""
        base_config: dict[str, Any] = {
            "read_only": self.container_config.read_only_filesystem,
            "network_disabled": self.container_config.network_disabled,
            "cap_drop": (["ALL"] if self.container_config.drop_all_capabilities else []),
            "security_opt": (
                ["no-new-privileges:true"] if self.container_config.no_new_privileges else []
            ),
            "user": "testuser",
            "tmpfs": {"/tmp": "noexec,nosuid,size=100m"},
        }

        # Adjust based on security level
        if security_level == SecurityLevel.HIGH:
            base_config.update(
                {
                    "mem_limit": f"{min(self.container_config.max_memory_mb, 256)}m",
                    "cpu_quota": min(self.container_config.max_cpu_quota, 25000),
                    "tmpfs": {"/tmp": "noexec,nosuid,size=50m"},
                }
            )
        elif security_level == SecurityLevel.MEDIUM:
            base_config.update(
                {
                    "mem_limit": f"{self.container_config.max_memory_mb}m",
                    "cpu_quota": self.container_config.max_cpu_quota,
                }
            )
        elif security_level == SecurityLevel.LOW:
            base_config.update(
                {
                    "mem_limit": f"{self.container_config.max_memory_mb * 2}m",
                    "cpu_quota": min(self.container_config.max_cpu_quota * 2, 100000),
                    "read_only": False,
                }
            )

        return base_config

    def validate_container_runtime_security(self, config: dict[str, Any]) -> bool:
        """Validate container runtime security configuration."""
        try:
            # Check required security options
            if config.get("network_disabled") is not True:
                logger.warning("Network is not disabled in container config")
                return False

            if "ALL" not in config.get("cap_drop", []):
                logger.warning("Not all capabilities are dropped")
                return False

            if "no-new-privileges:true" not in config.get("security_opt", []):
                logger.warning("No new privileges option not set")
                return False

            if config.get("user") != "testuser":
                logger.warning("Container not running as testuser")
                return False

            # Check resource limits
            mem_limit = config.get("mem_limit", "")
            if mem_limit:
                try:
                    mem_value = int(mem_limit.rstrip("mM"))
                    if mem_value > self.container_config.max_memory_mb:
                        logger.warning(
                            f"Memory limit {mem_value}MB exceeds maximum {self.container_config.max_memory_mb}MB"
                        )
                        return False
                except ValueError:
                    logger.warning(f"Invalid memory limit format: {mem_limit}")
                    return False

            cpu_quota = config.get("cpu_quota", 0)
            if cpu_quota > self.container_config.max_cpu_quota:
                logger.warning(
                    f"CPU quota {cpu_quota} exceeds maximum {self.container_config.max_cpu_quota}"
                )
                return False

            logger.debug("Container runtime security validation passed")
            return True

        except Exception as e:
            logger.error(f"Container security validation failed: {e}")
            return False

    def get_security_report(self) -> dict[str, Any]:
        """Get comprehensive security report."""
        return {
            "security_manager": "ExecutionSecurityManager",
            "version": "1.0.0",
            "config": {
                "max_memory_mb": self.container_config.max_memory_mb,
                "max_cpu_quota": self.container_config.max_cpu_quota,
                "max_execution_time": self.container_config.max_execution_time,
                "max_file_size_mb": self.container_config.max_file_size_mb,
                "max_code_length": self.container_config.max_code_length,
                "network_disabled": self.container_config.network_disabled,
                "read_only_filesystem": self.container_config.read_only_filesystem,
                "drop_all_capabilities": self.container_config.drop_all_capabilities,
                "no_new_privileges": self.container_config.no_new_privileges,
            },
            "dangerous_patterns": {
                lang: len(patterns) for lang, patterns in self.dangerous_patterns.items()
            },
            "allowed_modules": self.container_config.allowed_modules,
            "security_features": [
                "Code pattern analysis",
                "Import validation",
                "Syntax checking",
                "Container hardening",
                "Resource limits",
                "Runtime security validation",
            ],
        }


def create_execution_security_manager(
    config: ContainerSecurityConfig,
) -> ExecutionSecurityManager:
    """Factory function to create ExecutionSecurityManager."""
    return ExecutionSecurityManager(config)
