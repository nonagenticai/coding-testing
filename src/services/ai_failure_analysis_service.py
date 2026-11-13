"""AI failure analysis service stub.

Provides the AIFailureAnalysisService interface for test failure analysis.
"""

import logging
from typing import Any

from models.ai_test_failure_models import (
    FailureAnalysisRequest,
    FailureAnalysisResponse,
)

logger = logging.getLogger(__name__)


class AIFailureAnalysisService:
    """Stub for AI-powered test failure analysis service.

    Provides the interface for analyzing test failures using AI.
    In the microservice, this delegates to the coding-ai service.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        max_concurrent_analyses: int = 3,
        analysis_timeout: float = 60.0,
    ):
        self.config = config or {}
        self.max_concurrent_analyses = max_concurrent_analyses
        self.analysis_timeout = analysis_timeout
        self._cache: dict[str, Any] = {}
        logger.info("AIFailureAnalysisService stub initialized")

    async def analyze_failure(
        self,
        request: FailureAnalysisRequest,
    ) -> FailureAnalysisResponse:
        """Analyze a test failure (stub - returns basic response)."""
        from datetime import UTC, datetime

        logger.warning("AIFailureAnalysisService stub: analyze_failure called")
        return FailureAnalysisResponse(
            request_id=request.request_id,
            success=False,
            error_message="AI failure analysis not available in stub mode",
            processing_time=0.0,
            timestamp=datetime.now(UTC),
        )

    async def analyze_failures_batch(
        self,
        requests: list[FailureAnalysisRequest],
    ) -> list[FailureAnalysisResponse]:
        """Analyze multiple test failures."""
        return [await self.analyze_failure(req) for req in requests]

    async def analyze_test_failure(
        self,
        request: FailureAnalysisRequest,
    ) -> FailureAnalysisResponse:
        """Analyze a test failure (alias for analyze_failure)."""
        return await self.analyze_failure(request)

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of the analysis service."""
        return {
            "service": "ai_failure_analysis",
            "status": "stub",
            "message": "AIFailureAnalysisService stub mode",
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._cache.clear()

    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._cache.clear()


_ai_failure_analysis_service: AIFailureAnalysisService | None = None


def get_ai_failure_analysis_service() -> AIFailureAnalysisService:
    """Get AIFailureAnalysisService singleton."""
    global _ai_failure_analysis_service
    if _ai_failure_analysis_service is None:
        _ai_failure_analysis_service = AIFailureAnalysisService()
    return _ai_failure_analysis_service
