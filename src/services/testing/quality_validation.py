# /src/app/services/testing/quality_validation.py
"""
Quality Validation Utilities Module
Handles validation of requests, files, fixes, and quality analysis results.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.core_config_settings import get_settings
from models.testing_models import QualityIssue, QualityRequest, QualityResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        is_valid (bool): Whether validation passed.
        errors (List[str]): List of validation errors.
        warnings (List[str]): List of validation warnings.
        metadata (Dict[str, Any]): Additional validation metadata.
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    metadata: dict[str, Any]


@dataclass
class FixValidationResult:
    """Result of fix validation.

    Attributes:
        can_apply (bool): Whether fixes can be applied.
        safe_fixes (List[str]): List of safe fix identifiers.
        risky_fixes (List[str]): List of risky fix identifiers.
        blocked_fixes (List[str]): List of blocked fix identifiers.
        confidence_score (float): Overall confidence in fixes.
    """

    can_apply: bool
    safe_fixes: list[str]
    risky_fixes: list[str]
    blocked_fixes: list[str]
    confidence_score: float


class QualityRequestValidator:
    """Validates quality analysis requests."""

    def __init__(self):
        self.settings = get_settings()
        self.supported_languages = {
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
        }
        self.max_file_size_mb = 50
        self.allowed_extensions = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".java",
            ".go",
            ".rs",
        }

    async def validate_request(self, request: QualityRequest) -> ValidationResult:
        """Validate a quality analysis request.

        Args:
            request: Quality analysis request to validate.

        Returns:
            Validation result with errors and warnings.
        """
        errors = []
        warnings = []
        metadata = {}

        # Validate code path
        path_validation = await self._validate_code_path(request.code_path)
        if not path_validation.is_valid:
            errors.extend(path_validation.errors)
        warnings.extend(path_validation.warnings)
        metadata.update(path_validation.metadata)

        # Validate language
        if request.language:
            lang_validation = self._validate_language(request.language)
            if not lang_validation.is_valid:
                errors.extend(lang_validation.errors)
            warnings.extend(lang_validation.warnings)

        # Validate tools
        if request.tools:
            tools_validation = self._validate_tools(request.tools, request.language)
            if not tools_validation.is_valid:
                errors.extend(tools_validation.errors)
            warnings.extend(tools_validation.warnings)

        # Validate fix options
        if request.fix_issues:
            fix_validation = self._validate_fix_options(request)
            if not fix_validation.is_valid:
                errors.extend(fix_validation.errors)
            warnings.extend(fix_validation.warnings)

        # Overall validation
        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, metadata=metadata
        )

    async def _validate_code_path(self, code_path: str | None) -> ValidationResult:
        """Validate code path."""
        errors = []
        warnings = []
        metadata = {}

        if not code_path:
            errors.append("Code path is required")
            return ValidationResult(False, errors, warnings, metadata)

        path = Path(code_path)

        # Check if path exists
        if not path.exists():
            errors.append(f"Code path does not exist: {code_path}")
            return ValidationResult(False, errors, warnings, metadata)

        # Check if it's a file
        if not path.is_file():
            errors.append(f"Code path must be a file: {code_path}")
            return ValidationResult(False, errors, warnings, metadata)

        # Check file extension
        if path.suffix.lower() not in self.allowed_extensions:
            warnings.append(f"File extension {path.suffix} may not be supported")

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            errors.append(f"File too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)")

        # Check file permissions
        is_readable = os.access(path, os.R_OK)
        if not is_readable:
            errors.append(f"File is not readable: {code_path}")

        # Metadata
        metadata.update(
            {
                "file_size_mb": file_size_mb,
                "file_extension": path.suffix,
                "file_name": path.name,
                "is_readable": is_readable,
            }
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _validate_language(self, language: str | None) -> ValidationResult:
        """Validate programming language."""
        errors = []
        warnings = []
        metadata = {}

        if not language:
            warnings.append("No language specified, will auto-detect")
            return ValidationResult(True, errors, warnings, metadata)

        if language.lower() not in self.supported_languages:
            errors.append(f"Unsupported language: {language}")
            metadata["supported_languages"] = list(self.supported_languages)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _validate_tools(self, tools: list[str] | None, language: str | None) -> ValidationResult:
        """Validate requested tools."""
        errors = []
        warnings = []
        metadata = {}

        if not tools:
            warnings.append("No tools specified, will use defaults")
            return ValidationResult(True, errors, warnings, metadata)

        # Basic tool validation (would be enhanced with actual tool registry)
        known_tools = {
            "python": {
                "ruff",
                "ruff-format",
                "ty",
                "bandit",
            },
            "javascript": {"eslint", "prettier", "jshint"},
            "typescript": {"eslint", "prettier", "tsc"},
        }

        if language and language.lower() in known_tools:
            available_tools = known_tools[language.lower()]
            unknown_tools = set(tools) - available_tools

            if unknown_tools:
                errors.append(f"Unknown tools for {language}: {list(unknown_tools)}")

            metadata["available_tools"] = list(available_tools)
            metadata["requested_tools"] = tools

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _validate_fix_options(self, request: QualityRequest) -> ValidationResult:
        """Validate fix options."""
        errors = []
        warnings = []
        metadata = {}

        if request.fix_issues:
            warnings.append("Automatic fixes enabled - code will be modified")

            # Check if we have backup options
            if not hasattr(request, "create_backup") or not request.create_backup:
                warnings.append("No backup option specified - original code may be lost")

        return ValidationResult(
            is_valid=True,  # Fix options are warnings, not errors
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


class QualityFileValidator:
    """Validates files for quality analysis."""

    def __init__(self):
        self.max_line_length = 10000
        self.max_lines = 50000
        self.suspicious_patterns = [
            b"\x00",  # Null bytes
            b"\xff\xfe",  # UTF-16 LE BOM
            b"\xfe\xff",  # UTF-16 BE BOM
        ]

    async def validate_file(self, file_path: str | Path) -> ValidationResult:
        """Validate a file for quality analysis.

        Args:
            file_path: Path to file to validate.

        Returns:
            Validation result.
        """
        errors = []
        warnings = []
        metadata = {}

        path = Path(file_path)

        try:
            # Read file content
            content = path.read_bytes()

            # Check for binary content
            if self._is_binary_file(content):
                errors.append("File appears to be binary")
                return ValidationResult(False, errors, warnings, metadata)

            # Convert to text for analysis
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text_content = content.decode("latin-1")
                    warnings.append("File encoding detected as latin-1, may have issues")
                except UnicodeDecodeError:
                    errors.append("Unable to decode file content")
                    return ValidationResult(False, errors, warnings, metadata)

            # Validate file structure
            structure_validation = self._validate_file_structure(text_content)
            if not structure_validation.is_valid:
                errors.extend(structure_validation.errors)
            warnings.extend(structure_validation.warnings)
            metadata.update(structure_validation.metadata)

            # Check for security issues
            security_validation = self._validate_file_security(text_content)
            warnings.extend(security_validation.warnings)
            metadata.update(security_validation.metadata)

        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _is_binary_file(self, content: bytes) -> bool:
        """Check if file content is binary."""
        # Check for null bytes or other binary indicators
        if any(pattern in content for pattern in self.suspicious_patterns):
            return True

        # Check percentage of non-text characters
        try:
            decoded = content.decode("utf-8")
            # Count printable characters
            printable_count = sum(1 for c in decoded if c.isprintable() or c.isspace())
            ratio = printable_count / len(decoded) if decoded else 0
            return ratio < 0.95
        except UnicodeDecodeError:
            return True

    def _validate_file_structure(self, content: str) -> ValidationResult:
        """Validate file structure."""
        errors = []
        warnings = []
        metadata = {}

        lines = content.split("\n")

        # Check line count
        if len(lines) > self.max_lines:
            errors.append(f"File too large: {len(lines)} lines (max: {self.max_lines})")

        # Check line lengths
        long_lines = [i for i, line in enumerate(lines, 1) if len(line) > self.max_line_length]
        if long_lines:
            warnings.append(f"Long lines detected at: {long_lines[:5]}")

        # Check for very short files
        if len(lines) < 3:
            warnings.append("File is very short, may not need analysis")

        # Metadata
        metadata.update(
            {
                "line_count": len(lines),
                "character_count": len(content),
                "average_line_length": (
                    sum(len(line) for line in lines) / len(lines) if lines else 0
                ),
                "long_lines_count": len(long_lines),
            }
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _validate_file_security(self, content: str) -> ValidationResult:
        """Validate file for security concerns."""
        warnings = []
        metadata = {}

        # Check for potential secrets (basic patterns)
        secret_patterns = [
            "password",
            "secret",
            "key",
            "token",
            "api_key",
            "private_key",
            "access_token",
            "auth_token",
        ]

        found_patterns = []
        content_lower = content.lower()

        for pattern in secret_patterns:
            if pattern in content_lower:
                found_patterns.append(pattern)

        if found_patterns:
            warnings.append(f"Potential secrets detected: {found_patterns}")

        metadata["security_patterns"] = found_patterns

        return ValidationResult(
            is_valid=True,  # Security issues are warnings
            errors=[],
            warnings=warnings,
            metadata=metadata,
        )


class QualityFixValidator:
    """Validates quality fixes before application."""

    def __init__(self):
        self.safe_tools = {"ruff", "ruff-format", "prettier"}
        self.risky_tools = {"ty", "bandit"}
        self.min_confidence_threshold = 0.7

    async def validate_fixes(
        self, issues: list[QualityIssue], language: str
    ) -> FixValidationResult:
        """Validate proposed fixes.

        Args:
            issues: List of quality issues with potential fixes.
            language: Programming language.

        Returns:
            Fix validation result.
        """
        safe_fixes = []
        risky_fixes = []
        blocked_fixes = []
        confidence_scores = []

        for issue in issues:
            if not issue.suggested_fix:
                continue

            fix_id = f"{issue.tool}:{issue.rule_id}:{issue.line_number}"

            # Validate based on tool safety
            if issue.tool in self.safe_tools:
                safe_fixes.append(fix_id)
                confidence_scores.append(issue.fix_confidence or 0.9)
            elif issue.tool in self.risky_tools:
                risky_fixes.append(fix_id)
                confidence_scores.append(issue.fix_confidence or 0.5)
            else:
                # Check confidence
                confidence = issue.fix_confidence or 0.0
                if confidence >= self.min_confidence_threshold:
                    safe_fixes.append(fix_id)
                elif confidence >= 0.4:
                    risky_fixes.append(fix_id)
                else:
                    blocked_fixes.append(fix_id)
                confidence_scores.append(confidence)

        # Calculate overall confidence
        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        )

        # Determine if fixes can be applied
        can_apply = len(safe_fixes) > 0 or (len(risky_fixes) > 0 and overall_confidence > 0.6)

        return FixValidationResult(
            can_apply=can_apply,
            safe_fixes=safe_fixes,
            risky_fixes=risky_fixes,
            blocked_fixes=blocked_fixes,
            confidence_score=overall_confidence,
        )

    def validate_fix_safety(self, fix_description: str, tool: str) -> float:
        """Validate safety of a specific fix.

        Args:
            fix_description: Description of the fix.
            tool: Tool proposing the fix.

        Returns:
            Safety score (0.0 to 1.0).
        """
        base_score = 0.8 if tool in self.safe_tools else 0.5

        # Check for dangerous operations
        dangerous_keywords = ["delete", "remove", "drop", "truncate", "exec", "eval"]
        if any(keyword in fix_description.lower() for keyword in dangerous_keywords):
            base_score *= 0.5

        # Check for file operations
        file_keywords = ["write", "overwrite", "modify", "change"]
        if any(keyword in fix_description.lower() for keyword in file_keywords):
            base_score *= 0.8

        return min(1.0, base_score)


class QualityResultValidator:
    """Validates quality analysis results."""

    def __init__(self):
        self.min_execution_time = 0.001  # Minimum reasonable execution time
        self.max_execution_time = 3600  # Maximum reasonable execution time (1 hour)

    async def validate_result(self, result: QualityResult) -> ValidationResult:
        """Validate a quality analysis result.

        Args:
            result: Quality analysis result to validate.

        Returns:
            Validation result.
        """
        errors = []
        warnings = []
        metadata = {}

        # Validate basic structure
        structure_validation = self._validate_result_structure(result)
        if not structure_validation.is_valid:
            errors.extend(structure_validation.errors)
        warnings.extend(structure_validation.warnings)

        # Validate score consistency
        score_validation = self._validate_score_consistency(result)
        if not score_validation.is_valid:
            errors.extend(score_validation.errors)
        warnings.extend(score_validation.warnings)

        # Validate timing
        timing_validation = self._validate_execution_timing(result)
        warnings.extend(timing_validation.warnings)

        # Validate issues
        issues_validation = self._validate_issues(result.issues)
        warnings.extend(issues_validation.warnings)
        metadata.update(issues_validation.metadata)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _validate_result_structure(self, result: QualityResult) -> ValidationResult:
        """Validate result structure."""
        errors = []
        warnings = []

        # Check required fields
        if not result.analysis_id:
            errors.append("Missing analysis_id")

        if not isinstance(result.overall_score, (int, float)):
            errors.append("Invalid overall_score type")
        elif not 0 <= result.overall_score <= 100:
            errors.append(f"overall_score out of range: {result.overall_score}")

        if not isinstance(result.total_issues, int) or result.total_issues < 0:
            errors.append("Invalid total_issues")

        # Check consistency
        if len(result.issues) != result.total_issues:
            warnings.append(
                f"Inconsistent issue count: {len(result.issues)} vs {result.total_issues}"
            )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings, metadata={}
        )

    def _validate_score_consistency(self, result: QualityResult) -> ValidationResult:
        """Validate score consistency with issues."""
        errors = []
        warnings = []

        # High score with many critical issues
        critical_issues = len([i for i in result.issues if i.severity == "error"])

        if result.overall_score > 8.0 and critical_issues > 5:
            warnings.append(
                f"High score ({result.overall_score}) with many critical issues ({critical_issues})"
            )

        # Low score with no issues
        if result.overall_score < 3.0 and result.total_issues == 0:
            warnings.append(f"Low score ({result.overall_score}) with no issues")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings, metadata={}
        )

    def _validate_execution_timing(self, result: QualityResult) -> ValidationResult:
        """Validate execution timing."""
        warnings = []

        if result.execution_time < self.min_execution_time:
            warnings.append(f"Execution time suspiciously low: {result.execution_time}s")
        elif result.execution_time > self.max_execution_time:
            warnings.append(f"Execution time very high: {result.execution_time}s")

        return ValidationResult(
            is_valid=True,  # Timing issues are warnings
            errors=[],
            warnings=warnings,
            metadata={},
        )

    def _validate_issues(self, issues: list[QualityIssue]) -> ValidationResult:
        """Validate quality issues."""
        warnings = []
        metadata = {}

        # Check for duplicate issues
        issue_signatures = []
        for issue in issues:
            signature = f"{issue.tool}:{issue.file_path}:{issue.line_number}:{issue.rule_id}"
            issue_signatures.append(signature)

        duplicates = len(issue_signatures) - len(set(issue_signatures))
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate issues")

        # Check severity distribution
        severity_counts = {}
        for issue in issues:
            severity = issue.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        metadata["severity_distribution"] = severity_counts

        return ValidationResult(is_valid=True, errors=[], warnings=warnings, metadata=metadata)


class QualityCacheValidator:
    """Validates cached quality results."""

    def __init__(self):
        self.max_cache_age_hours = 24

    def generate_cache_key(self, request: QualityRequest) -> str:
        """Generate cache key for quality request.

        Args:
            request: Quality analysis request.

        Returns:
            Cache key string.
        """
        # Include file content hash for cache invalidation
        try:
            file_content = Path(request.code_path or "").read_text()
            content_hash = hashlib.sha256(file_content.encode()).hexdigest()[:16]
        except Exception:
            content_hash = "unknown"

        key_data = (
            f"{request.code_path}:{sorted(request.tools or [])}:{request.language}:{content_hash}"
        )

        return hashlib.sha256(key_data.encode()).hexdigest()

    def validate_cached_result(
        self, cached_data: dict[str, Any], request: QualityRequest
    ) -> ValidationResult:
        """Validate cached quality result.

        Args:
            cached_data: Cached result data.
            request: Current request.

        Returns:
            Validation result.
        """
        errors = []
        warnings = []
        metadata = {}

        # Check age
        if "created_at" in cached_data:
            try:
                cached_time = datetime.fromisoformat(cached_data["created_at"])
                age_hours = (datetime.now(UTC) - cached_time).total_seconds() / 3600

                if age_hours > self.max_cache_age_hours:
                    errors.append(f"Cached result too old: {age_hours:.1f} hours")

                metadata["cache_age_hours"] = age_hours
            except ValueError as e:
                logger.warning(
                    f"Failed to parse cached timestamp {cached_data['created_at']!r}: {e}"
                )
                warnings.append("Invalid cached timestamp")
            except Exception:
                logger.exception(
                    f"Unexpected error parsing cached timestamp from {cached_data.get('created_at')!r}"
                )
                warnings.append("Invalid cached timestamp")

        # Check language consistency
        if "language" in cached_data and request.language:
            if cached_data["language"] != request.language:
                errors.append("Language mismatch with cached result")

        # Check tools consistency
        if "tools_results" in cached_data and request.tools:
            cached_tools = set(cached_data["tools_results"].keys())
            requested_tools = set(request.tools)

            if cached_tools != requested_tools:
                warnings.append("Tool set differs from cached result")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


# Factory functions for easy access
def get_request_validator() -> QualityRequestValidator:
    """Get a quality request validator."""
    return QualityRequestValidator()


def get_file_validator() -> QualityFileValidator:
    """Get a file validator."""
    return QualityFileValidator()


def get_fix_validator() -> QualityFixValidator:
    """Get a fix validator."""
    return QualityFixValidator()


def get_result_validator() -> QualityResultValidator:
    """Get a result validator."""
    return QualityResultValidator()


def get_cache_validator() -> QualityCacheValidator:
    """Get a cache validator."""
    return QualityCacheValidator()
