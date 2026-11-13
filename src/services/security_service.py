"""Security service stub.

Provides the SystemSecurityService interface. In the microservice architecture,
security validation is handled by coding-security service.
"""

import logging
from typing import Any

from models.security_models import (
    SecurityLevel,
    SecurityValidationResult,
)

logger = logging.getLogger(__name__)


class SystemSecurityService:
    """Stub for system security service.

    Provides basic security validation. Full validation
    is handled by the coding-security microservice.
    """

    def __init__(self, config: Any = None):
        self.config = config if isinstance(config, dict) else {}
        logger.info("SystemSecurityService stub initialized")

    async def validate_code(
        self,
        code: str,
        language: str,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        context: dict[str, Any] | None = None,
    ) -> SecurityValidationResult:
        """Validate code security (basic stub)."""
        return SecurityValidationResult(
            is_safe=True,
            violations=[],
            risk_score=0.0,
            sandbox_config=self.get_sandbox_config(security_level),
        )

    def get_sandbox_config(
        self, security_level: SecurityLevel = SecurityLevel.MEDIUM
    ) -> dict[str, Any]:
        """Get sandbox configuration for a security level."""
        configs = {
            SecurityLevel.LOW: {
                "network_access": True,
                "file_system": "read_write",
                "max_execution_time": 300,
            },
            SecurityLevel.MEDIUM: {
                "network_access": False,
                "file_system": "read_only",
                "max_execution_time": 120,
            },
            SecurityLevel.HIGH: {
                "network_access": False,
                "file_system": "none",
                "max_execution_time": 60,
            },
            SecurityLevel.CRITICAL: {
                "network_access": False,
                "file_system": "none",
                "max_execution_time": 30,
            },
        }
        return configs.get(security_level, configs[SecurityLevel.MEDIUM])

    async def validate_code_security(
        self,
        code_or_request: Any,
        language: str = "python",
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
    ) -> SecurityValidationResult:
        """Validate code security. Accepts either a string or SecurityValidationRequest."""
        if hasattr(code_or_request, "code"):
            return await self.validate_code(
                code_or_request.code,
                getattr(code_or_request, "language", language),
                getattr(code_or_request, "security_level", security_level),
            )
        return await self.validate_code(code_or_request, language, security_level)

    async def validate_imports(
        self, code: str, language: str | None = None, allowed_imports: list[str] | None = None
    ) -> SecurityValidationResult:
        """Validate imports in code (stub)."""
        return SecurityValidationResult(
            is_safe=True, violations=[], risk_score=0.0, sandbox_config=None
        )

    def generate_sandbox_config(
        self, language: Any = None, security_level: Any = None
    ) -> dict[str, Any]:
        """Generate sandbox configuration."""
        level = (
            security_level if isinstance(security_level, SecurityLevel) else SecurityLevel.MEDIUM
        )
        return self.get_sandbox_config(level)

    async def scan_container(self, container_name: str, scan_type: str = "full") -> dict[str, Any]:
        """Scan container for security issues (stub)."""
        logger.warning(f"SecurityService stub: scan_container called for {container_name}")
        return {"container_name": container_name, "vulnerabilities": [], "scan_type": scan_type}


_security_service: SystemSecurityService | None = None


def create_security_service(config: Any = None, cache_manager: Any = None) -> SystemSecurityService:
    """Create a SystemSecurityService instance."""
    global _security_service
    if _security_service is None:
        _security_service = SystemSecurityService(config=config)
    return _security_service


def get_security_service() -> SystemSecurityService:
    """Get the SystemSecurityService singleton."""
    return create_security_service()
