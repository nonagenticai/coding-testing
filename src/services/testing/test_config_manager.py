"""Test configuration management for the testing service."""

import logging
from pathlib import Path
from typing import Any

from config.test_framework_config import TestConfig, TestFrameworkSecurityConfig
from models.security_models import (
    SecurityValidationRequest,
)
from models.testing_models import (
    SecurityLevel,
    TestRequest,
)
from services.security_service import SystemSecurityService as SecurityService

logger = logging.getLogger(__name__)


class TestConfigManager:
    """
    Manages test configuration, security validation, and environment setup.
    """

    def __init__(
        self,
        security_service: SecurityService,
        testpy_config: TestConfig | None = None,
        security_config: TestFrameworkSecurityConfig | None = None,
    ):
        self.security_service = security_service
        self.testpy_config = testpy_config or TestConfig()
        self.security_config = security_config or TestFrameworkSecurityConfig()

    async def validate_security(self, request: TestRequest) -> None:
        """Validate request against security policies using the SecurityService."""
        code_to_validate = request.code or ""
        test_code_to_validate = request.test_code or ""

        # Validate code length using SecurityConfig
        total_code = code_to_validate + test_code_to_validate
        if len(total_code) > self.security_config.max_code_length:
            raise ValueError(
                f"Code length exceeds maximum of {self.security_config.max_code_length}"
            )

        # Map TestRequest security level to SecurityLevel enum
        security_level_map = {
            "HIGH": SecurityLevel.HIGH,
            "MEDIUM": SecurityLevel.MEDIUM,
            "LOW": SecurityLevel.LOW,
        }

        security_level = security_level_map.get(
            (
                request.security_level.value
                if hasattr(request.security_level, "value")
                else str(request.security_level).upper()
            ),
            SecurityLevel.MEDIUM,
        )

        # Validate main code using SecurityService
        if code_to_validate:
            await self._validate_code_security(
                code_to_validate, request.language, security_level, "main code"
            )

        # Validate test code using SecurityService
        if test_code_to_validate:
            await self._validate_code_security(
                test_code_to_validate, request.language, security_level, "test code"
            )

    async def _validate_code_security(
        self, code: str, language: str, security_level: SecurityLevel, code_type: str
    ) -> None:
        """Validate a piece of code using SecurityService."""
        security_request = SecurityValidationRequest(
            code=code,
            language=language,
            security_level=security_level,  # type: ignore[invalid-argument-type]
            context=None,
        )

        security_result = await self.security_service.validate_code_security(security_request)

        if not security_result.is_safe:
            violation_details = [
                f"{v.type}: {v.description} (line {v.line_number})"
                # Show first 3 violations
                for v in security_result.violations[:3]
            ]
            raise ValueError(
                f"{code_type.capitalize()} security validation failed. Violations: {'; '.join(violation_details)}"
            )

    def detect_language(self, file_path: str | None, language_hint: str | None) -> str:
        """Detect programming language from file path or hint."""
        if language_hint:
            return language_hint.lower()

        if file_path:
            path = Path(file_path)
            extension_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".jsx": "javascript",
                ".tsx": "typescript",
            }
            return extension_map.get(path.suffix.lower(), "python")

        return "python"

    def get_quality_tools(self, language: str, requested_tools: list[str] | None) -> list[str]:
        """Get quality tools for language."""
        default_tools = {
            "python": ["ruff", "ruff-format", "ty"],
            "javascript": ["eslint", "prettier"],
            "typescript": ["eslint", "prettier", "tsc"],
        }

        available_tools = default_tools.get(language, [])

        if requested_tools:
            # Filter to only requested tools that are available
            return [tool for tool in requested_tools if tool in available_tools]

        return available_tools

    def get_container_image(self, language: str) -> str:
        """Get appropriate container image for language."""
        image_map = {
            "python": "codegraph/test-executor-python:latest",
            "javascript": "code-graph-test-node:latest",
            "typescript": "code-graph-test-node:latest",
        }
        return image_map.get(language, "codegraph/test-executor-python:latest")

    def get_memory_limit(self, security_level: SecurityLevel) -> str:
        """Get memory limit based on security level."""
        limits = {
            SecurityLevel.HIGH: "256m",
            SecurityLevel.MEDIUM: "512m",
            SecurityLevel.LOW: "1g",
        }
        return limits.get(security_level, "512m")

    def prepare_test_files(self, request: TestRequest) -> dict[str, str]:
        """Prepare files for container execution."""
        files = {}

        if request.code:
            files["main.py"] = request.code

        if request.test_code:
            files["test_main.py"] = request.test_code

        return files

    def get_test_command(self, language: str, test_files: dict[str, str]) -> list[str]:
        """Get test execution command based on language and files."""
        commands = {
            "python": ["python", "-m", "pytest", "-v", "--tb=short"],
            "javascript": ["npm", "test"],
            "typescript": ["npm", "test"],
        }

        # Get base command for language
        base_command = commands.get(language, ["python", "-m", "pytest"])

        # Add specific test files if they exist
        if language == "python" and "test_main.py" in test_files:
            base_command.append("test_main.py")

        return base_command

    def create_container_config(
        self, request: TestRequest, test_id: str, image_name: str
    ) -> dict[str, Any]:
        """Create container configuration for test execution."""
        return {
            "image": image_name,
            "language": request.language,
            "mem_limit": self.get_memory_limit(request.security_level),
            "cpu_quota": 50000,
            "network_disabled": True,
            "read_only": False,  # Need write access for test files
            "labels": {"test_id": test_id, "type": "test_execution"},
            "working_dir": "/app/test",
        }

    def get_tool_commands(self, language: str) -> dict[str, list[str]]:
        """Get tool commands for quality analysis."""
        return {
            "python": {
                "ruff": ["ruff", "check", "--output-format", "json", "--extend-select", "I"],
                "ruff-format": ["ruff", "format", "--check", "--diff"],
                "ty": ["ty", "check", "--output-format", "gitlab"],
            },
            "javascript": {
                "eslint": ["eslint", "--format=json"],
                "prettier": ["prettier", "--check"],
            },
            "typescript": {
                "eslint": ["eslint", "--format=json"],
                "prettier": ["prettier", "--check"],
                "tsc": ["tsc", "--noEmit"],
            },
        }.get(language, {})

    def get_file_extension_for_language(self, language: str) -> str:
        """Get file extension for a given language."""
        extension_map = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
        }
        return extension_map.get(language, ".py")

    def discover_files(
        self, directory_path: str, patterns: list[str], recursive: bool
    ) -> list[Path]:
        """Discover files matching patterns in directory."""
        base_path = Path(directory_path)

        if not base_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        files_found: list[Path] = []

        for pattern in patterns:
            if recursive:
                files_found.extend(base_path.rglob(pattern))
            else:
                files_found.extend(base_path.glob(pattern))

        # Remove duplicates and sort
        unique_files = list(set(files_found))
        unique_files.sort()

        return unique_files

    def find_test_file(self, file_path: Path) -> Path | None:
        """Find corresponding test file for a source file."""
        # Common test file patterns
        patterns = [
            f"test_{file_path.stem}.py",
            f"{file_path.stem}_test.py",
            f"test_{file_path.stem}.js",
            f"{file_path.stem}.test.js",
        ]

        # Check in same directory
        for pattern in patterns:
            test_file = file_path.parent / pattern
            if test_file.exists():
                return test_file

        # Check in tests directory
        tests_dir = file_path.parent / "tests"
        if tests_dir.exists():
            for pattern in patterns:
                test_file = tests_dir / pattern
                if test_file.exists():
                    return test_file

        return None

    def validate_configuration(self) -> dict[str, Any]:
        """Validate the current configuration."""
        validation_results = {
            "testpy_config": self.testpy_config is not None,
            "security_config": self.security_config is not None,
            "security_service": self.security_service is not None,
            "max_code_length": (
                self.security_config.max_code_length if self.security_config else "not configured"
            ),
        }

        return {
            "valid": all(v for k, v in validation_results.items() if isinstance(v, bool)),
            "details": validation_results,
        }
