"""Test framework security manager stub."""

import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SecurityViolationType(StrEnum):
    """Types of security violations."""

    DANGEROUS_IMPORT = "dangerous_import"
    DANGEROUS_FUNCTION = "dangerous_function"
    FILE_SYSTEM_ACCESS = "file_system_access"
    NETWORK_ACCESS = "network_access"
    PROCESS_EXECUTION = "process_execution"
    CODE_INJECTION = "code_injection"
    RESOURCE_ABUSE = "resource_abuse"
    UNAUTHORIZED_IMPORT = "unauthorized_import"
    SYNTAX_ERROR = "syntax_error"
    EXECUTION_TIME = "execution_time"
    CODE_LENGTH = "code_length"


class SecurityCheckResult(BaseModel):
    """Result of a security check."""

    is_safe: bool = Field(default=True, description="Whether the code passed security checks")
    passed: bool = Field(default=True, description="Alias for is_safe")
    violations: list[dict[str, Any]] = Field(
        default_factory=list, description="List of security violations"
    )
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall risk score")
    checked_patterns: int = Field(default=0, description="Number of patterns checked")
    error: str | None = Field(default=None, description="Error message if check failed")
    violation_type: SecurityViolationType | None = Field(
        default=None, description="Type of security violation"
    )


class TestValidationSecurityConfig(BaseModel):
    """Security configuration for test validation."""

    max_execution_time: int = Field(default=300, description="Max execution time in seconds")
    max_memory_mb: int = Field(default=512, description="Max memory in MB")
    max_file_size_mb: int = Field(default=10, description="Max file size in MB")
    max_code_length: int = Field(default=10000, description="Max code length in characters")
    allow_network: bool = Field(default=False, description="Allow network access")
    allow_file_write: bool = Field(default=False, description="Allow file system writes")
    allowed_modules: list[str] = Field(
        default_factory=lambda: ["pytest", "math", "os", "sys"],
        description="Allowed Python modules",
    )
    blocked_imports: list[str] = Field(
        default_factory=lambda: ["subprocess", "shutil", "ctypes"],
        description="Blocked Python imports",
    )
    security_level: str = Field(default="medium", description="Security level")


class SecurityManager:
    """Stub security manager for test framework.

    Provides basic security validation for test execution.
    In production, this delegates to coding-security service.
    """

    def __init__(
        self,
        logger_or_config: Any = None,
        config: TestValidationSecurityConfig | None = None,
    ):
        # Support both SecurityManager(config) and SecurityManager(logger, config)
        if isinstance(logger_or_config, TestValidationSecurityConfig):
            self.config = logger_or_config
            self._logger = None
        elif config is not None:
            self._logger = logger_or_config
            self.config = config
        else:
            self._logger = logger_or_config
            self.config = TestValidationSecurityConfig()
        logger.info("SecurityManager stub initialized")

    async def validate_code(self, code: str, language: str = "python") -> SecurityCheckResult:
        """Validate code for security issues (basic stub)."""
        return SecurityCheckResult(is_safe=True, passed=True, checked_patterns=0)

    async def check_imports(self, code: str) -> SecurityCheckResult:
        """Check code imports for dangerous patterns."""
        return SecurityCheckResult(is_safe=True, passed=True, checked_patterns=0)

    def check_code_safety(self, code: str) -> SecurityCheckResult:
        """Check code safety synchronously (basic stub)."""
        return SecurityCheckResult(is_safe=True, passed=True, checked_patterns=0)

    def get_sandbox_config(self) -> dict[str, Any]:
        """Get sandbox configuration for test execution."""
        return {
            "network_access": self.config.allow_network,
            "file_system": "read_only" if not self.config.allow_file_write else "read_write",
            "max_execution_time": self.config.max_execution_time,
            "max_memory_mb": self.config.max_memory_mb,
        }
