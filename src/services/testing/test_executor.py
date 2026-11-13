"""Core test execution logic for the testing service."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

from config.test_framework_config import TestRunner
from models.testing_models import (
    TestRequest,
    TestResult,
)
from services.test_execution.intelligent_failure_analyzer import (
    IntelligentFailureAnalyzer,
)

logger = logging.getLogger(__name__)


class TestExecutor:
    """
    Core test execution logic that orchestrates test runs using
    integrated TestPy components and intelligent analysis.
    """

    def __init__(
        self,
        test_runner: TestRunner,
        intelligent_analyzer: IntelligentFailureAnalyzer | None = None,
    ):
        self.test_runner = test_runner
        self.intelligent_analyzer = intelligent_analyzer

    async def execute_test(
        self,
        request: TestRequest,
        container_executor,
        config_manager,
        result_processor,
    ) -> TestResult:
        """Execute test using integrated TestPy components."""
        test_id = str(uuid7())
        start_time = datetime.now(UTC)

        try:
            logger.info(f"Starting test execution {test_id}")

            # Security validation using config manager
            await config_manager.validate_security(request)

            # Execute test in container
            result = await container_executor.execute_in_container(request, test_id)

            # Run intelligent failure analysis if test failed
            if not result.success and self.intelligent_analyzer:
                result = await self._run_intelligent_analysis(result, request, test_id)

            # Process result through result processor
            result = await result_processor.process_result(result, request)

            return result

        except Exception as e:
            logger.error(f"Test execution {test_id} failed: {str(e)}")
            return self._create_error_result(test_id, start_time, request, str(e))

    async def _run_intelligent_analysis(
        self, result: TestResult, request: TestRequest, test_id: str
    ) -> TestResult:
        """Run intelligent failure analysis on failed test."""
        if self.intelligent_analyzer is None:
            return result
        logger.info(f"Running intelligent analysis for failed test {test_id}")

        try:
            analysis_response = await self.intelligent_analyzer.analyze_test_failure(
                test_result=result,
                source_code=request.code,
                test_code=request.test_code,
                dependencies=request.dependencies or {},
                environment={},
                resource_usage=None,
            )

            # Add analysis results to test result metadata
            if analysis_response.success and analysis_response.failure_analysis:
                analysis_summary = {
                    "analysis_id": analysis_response.failure_analysis.analysis_id,
                    "category": analysis_response.failure_analysis.category,
                    "severity": analysis_response.failure_analysis.severity,
                    "confidence": analysis_response.failure_analysis.confidence,
                    "root_cause": analysis_response.failure_analysis.root_cause,
                    "recommendations": analysis_response.failure_analysis.recommendations,
                    "code_fixes": analysis_response.failure_analysis.code_fixes,
                    "model_provider": analysis_response.failure_analysis.ai_model_provider,
                }

                # Update test result with analysis
                result.test_metadata = result.test_metadata or {}
                result.test_metadata["intelligent_analysis"] = analysis_summary

                # Add quality issues if available
                if analysis_response.code_quality_issues:
                    result.test_metadata["code_quality_issues"] = [
                        {
                            "category": issue.category,
                            "severity": issue.severity,
                            "message": issue.message,
                            "suggested_fix": issue.suggested_fix,
                            "fix_confidence": issue.fix_confidence,
                        }
                        for issue in analysis_response.code_quality_issues
                    ]

                # Add performance analysis if available
                if analysis_response.performance_analysis:
                    result.test_metadata["performance_analysis"] = {
                        "bottlenecks": analysis_response.performance_analysis.bottlenecks,
                        "optimization_suggestions": analysis_response.performance_analysis.optimization_suggestions,
                        "performance_score": analysis_response.performance_analysis.performance_score,
                    }

                # Add security analysis if available
                if analysis_response.security_analysis:
                    result.test_metadata["security_analysis"] = {
                        "vulnerabilities": analysis_response.security_analysis.vulnerabilities,
                        "security_score": analysis_response.security_analysis.security_score,
                        "risk_level": analysis_response.security_analysis.risk_level,
                        "recommendations": analysis_response.security_analysis.recommendations,
                    }

                logger.info(
                    f"Intelligent analysis completed for test {test_id}: {analysis_response.failure_analysis.category}"
                )

            else:
                logger.warning(
                    f"Intelligent analysis failed for test {test_id}: {analysis_response.error_message}"
                )
                result.test_metadata = result.test_metadata or {}
                result.test_metadata["intelligent_analysis_error"] = analysis_response.error_message

        except Exception as e:
            logger.error(f"Intelligent analysis exception for test {test_id}: {e}")
            result.test_metadata = result.test_metadata or {}
            result.test_metadata["intelligent_analysis_error"] = str(e)

        return result

    def _create_error_result(
        self, test_id: str, start_time: datetime, request: TestRequest, error: str
    ) -> TestResult:
        """Create an error TestResult."""
        return TestResult(
            test_id=test_id,
            success=False,
            stdout="",
            stderr=error,
            exit_code=1,
            execution_time=(datetime.now(UTC) - start_time).total_seconds(),
            created_at=start_time,
            language=request.language,
            security_level=request.security_level,
            coverage_data=None,
            test_metadata=None,
            test_type=None,
            container_name=None,
            quality_report=None,
            error=error,
        )

    async def analyze_test_failure(
        self,
        test_result: TestResult,
        source_code: str | None = None,
        test_code: str | None = None,
        dependencies: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        resource_usage: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Analyze a test failure using intelligent AI analysis."""
        if not self.intelligent_analyzer:
            logger.warning("Intelligent analyzer is not available")
            return None

        try:
            response = await self.intelligent_analyzer.analyze_test_failure(
                test_result=test_result,
                source_code=source_code,
                test_code=test_code,
                dependencies=dependencies,
                environment=environment,
                resource_usage=resource_usage,
            )

            if response.success:
                return response.model_dump()
            else:
                logger.error(f"Analysis failed: {response.error_message}")
                return None

        except Exception as e:
            logger.error(f"Failed to analyze test failure: {e}")
            return None

    async def batch_analyze_failures(
        self,
        test_results: list[TestResult],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """Analyze multiple test failures concurrently."""
        if not self.intelligent_analyzer:
            logger.warning("Intelligent analyzer is not available")
            return []

        try:
            responses = await self.intelligent_analyzer.batch_analyze_failures(
                test_results=test_results,
                max_concurrent=max_concurrent,
            )

            return [response.model_dump() for response in responses]

        except Exception as e:
            logger.error(f"Failed to batch analyze failures: {e}")
            return []

    async def get_analysis_summary(self) -> dict[str, Any]:
        """Get summary of analysis service status and metrics."""
        if not self.intelligent_analyzer:
            return {
                "service": "test_executor",
                "intelligent_analysis": "disabled",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        try:
            return await self.intelligent_analyzer.get_analysis_summary()
        except Exception as e:
            logger.error(f"Failed to get analysis summary: {e}")
            return {
                "service": "test_executor",
                "intelligent_analysis": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
