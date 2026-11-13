"""
Security Management Service - Wrapper for the main SecurityService
Provides backward compatibility and testing-specific functionality
"""

from __future__ import annotations

from typing import Any

from models.security_models import (
    SecurityValidationRequest,
)
from services.security_service import create_security_service
from services.stubs import RedisCacheManager


class SecurityManagementService:
    """
    Handles security validation and sandboxing for code execution.
    This is a wrapper around the main SecurityService for testing-specific needs.
    """

    def __init__(self, cache_manager: RedisCacheManager | None = None):
        """Initialize the SecurityManagementService."""
        self.security_service = create_security_service(cache_manager)

    async def validate_code_security(self, code: str, language: str) -> dict[str, Any]:
        """
        Validate code security and return a streamlined result format.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            Dict with validation results
        """
        from models.security_models import SecurityLevel

        request = SecurityValidationRequest(
            code=code,
            language=language,
            security_level=SecurityLevel.MEDIUM,
            context=None,
        )

        result = await self.security_service.validate_code_security(request)

        # Return streamlined format for backward compatibility
        return {
            "is_safe": result.is_safe,
            "risk_score": result.risk_score,
            "violations": len(result.violations),
            "violation_details": [
                {
                    "type": v.type,
                    "severity": v.severity.value,
                    "line": v.line_number,
                    "description": v.description,
                }
                for v in result.violations
            ],
            "sandbox_config": result.sandbox_config,
        }

    async def create_sandbox_environment(
        self, language: str, security_level: str = "medium"
    ) -> dict[str, Any]:
        """
        Generate sandbox configuration for code execution.

        Args:
            language: Programming language
            security_level: Security level (low, medium, high)

        Returns:
            Sandbox configuration dictionary
        """
        return self.security_service.generate_sandbox_config(language, security_level)

    async def validate_imports(self, code: str, language: str) -> dict[str, Any]:
        """
        Validate imports in the code.

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            Import validation results
        """
        result = await self.security_service.validate_imports(code, language)

        return {
            "safe_imports": result.is_safe,
            "dangerous_imports": len(result.violations),
            "import_details": result.violations,
        }


# Factory function for creating SecurityManagementService instances
def create_security_management_service(
    cache_manager: RedisCacheManager | None = None,
) -> SecurityManagementService:
    """Create a SecurityManagementService instance."""
    return SecurityManagementService(cache_manager=cache_manager)
