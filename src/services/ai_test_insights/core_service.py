"""AI analysis core service module for AI Test Insights Service.

This module contains the main AITestInsightsService class and core AI agent
initialization logic for PydanticAI integration.
"""

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid7

from models.testing_models import QualityResult, TestResult
from services.ai_test_insights.config import (
    DEFAULT_CONFIG,
    AITestInsightsConfig,
    TestInsightType,
)
from services.ai_test_insights.insight_generation import TestInsightGenerator
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestInsightAnalysis,
    TestInsightRecommendation,
    TestPatternInsight,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)
from services.ai_test_insights.pattern_recognition import TestPatternDetector
from services.ai_test_insights.report_formatter import TestReportFormatter

# Try to import pydantic_ai, but make it optional. Same shape as
# services/test_execution/pydantic_ai_agents.py: the type checker sees the real
# import; the runtime fallback binds None only when the package is absent.
if TYPE_CHECKING:
    from pydantic_ai import Agent, RunContext

    HAS_PYDANTIC_AI: bool
else:
    try:
        from pydantic_ai import Agent, RunContext

        HAS_PYDANTIC_AI = True
    except ImportError:
        HAS_PYDANTIC_AI = False
        Agent: Any = None
        RunContext: Any = None

logger = logging.getLogger(__name__)


class AITestInsightsService:
    """AI-powered test insights service using PydanticAI."""

    def __init__(
        self,
        cache_manager: Any | None = None,
        ai_insights_engine=None,
        config: AITestInsightsConfig = DEFAULT_CONFIG,
    ):
        """Initialize the AI test insights service.

        Args:
            cache_manager: Optional cache manager for storing results
            ai_insights_engine: Optional AI insights engine
            config: Configuration for the service
        """
        self.cache_manager = cache_manager
        self.ai_insights_engine = ai_insights_engine
        self.config = config

        # Initialize modular components
        self.insight_generator = TestInsightGenerator(config)
        self.pattern_detector = TestPatternDetector(config)
        self.report_formatter = TestReportFormatter(config, cache_manager)

        # Initialize PydanticAI agents
        self._init_ai_agents()

        logger.info("AI Test Insights Service initialized with modular components")

    def _init_ai_agents(self) -> None:
        """Initialize PydanticAI agents for different types of analysis."""

        if not HAS_PYDANTIC_AI or not self.config.enable_ai_features:
            logger.warning(
                "pydantic_ai not available or AI features disabled - AI agents will be disabled"
            )
            self.performance_agent = None
            self.reliability_agent = None
            self.coverage_agent = None
            self.quality_agent = None
            self.pattern_agent = None
            self.recommendation_agent = None
            return

        try:
            # Performance analysis agent
            self.performance_agent = Agent(
                model="test",
                output_type=TestPerformanceMetrics,
                instructions="""
                You are an expert performance analyst for software testing.
                Analyze test execution data to identify performance bottlenecks,
                slow tests, and resource usage patterns.

                Provide actionable insights for improving test performance including:
                - Identification of slow tests and bottlenecks
                - Resource usage optimization recommendations
                - Performance trend analysis
                - Specific improvement suggestions

                Base your analysis on execution times, memory usage, CPU usage,
                and other performance metrics provided.
                """,
            )

            # Reliability analysis agent
            self.reliability_agent = Agent(
                model="test",
                output_type=TestReliabilityMetrics,
                instructions="""
                You are an expert reliability analyst for software testing.
                Analyze test execution results to identify reliability issues,
                flaky tests, and failure patterns.

                Provide insights for improving test reliability including:
                - Identification of flaky and unreliable tests
                - Common failure pattern analysis
                - Stability scoring and trend analysis
                - Specific reliability improvement recommendations

                Base your analysis on test success rates, failure patterns,
                error types, and execution consistency.
                """,
            )

            # Coverage analysis agent
            self.coverage_agent = Agent(
                model="test",
                output_type=TestCoverageInsights,
                instructions="""
                You are an expert test coverage analyst.
                Analyze test coverage data to identify gaps, redundancies,
                and optimization opportunities.

                Provide insights for improving test coverage including:
                - Critical path coverage analysis
                - Identification of coverage gaps
                - Redundant test detection
                - Specific test recommendations

                Base your analysis on line coverage, branch coverage,
                function coverage, and code structure information.
                """,
            )

            # Quality analysis agent
            self.quality_agent = Agent(
                model="test",
                output_type=TestQualityInsights,
                instructions="""
                You are an expert code quality analyst for testing.
                Analyze code quality metrics and test quality to identify
                improvement opportunities.

                Provide insights for improving code and test quality including:
                - Quality issue prioritization
                - Automated fix identification
                - Quality debt estimation
                - Specific improvement recommendations

                Base your analysis on quality tool results, code complexity,
                test maintainability, and industry best practices.
                """,
            )

            # Pattern detection agent
            self.pattern_agent = Agent(
                model="test",
                output_type=list[TestPatternInsight],
                instructions="""
                You are an expert pattern recognition analyst for software testing.
                Analyze test execution data to identify patterns, anti-patterns,
                and recurring issues.

                Detect and analyze patterns including:
                - Unit test patterns and anti-patterns
                - Integration test patterns
                - Performance test patterns
                - Flaky test patterns
                - Coverage patterns
                - Security test patterns

                Provide confidence scores, impact assessments, and specific
                recommendations for each pattern detected.
                """,
            )

            # Recommendation agent
            self.recommendation_agent = Agent(
                model="test",
                output_type=list[TestInsightRecommendation],
                instructions="""
                You are an expert testing consultant providing actionable
                recommendations for test improvement.

                Generate prioritized recommendations based on:
                - Performance analysis results
                - Reliability metrics
                - Coverage insights
                - Quality analysis
                - Detected patterns

                Provide specific, actionable recommendations with:
                - Clear implementation steps
                - Effort estimates
                - Expected impact
                - Priority levels

                Focus on recommendations that will have the highest impact
                on test quality, reliability, and maintainability.
                """,
            )

            logger.info("Successfully initialized all AI agents")

        except Exception as e:
            logger.warning(f"Failed to initialize AI agents: {e}. AI features will be disabled.")
            self.performance_agent = None
            self.reliability_agent = None
            self.coverage_agent = None
            self.quality_agent = None
            self.pattern_agent = None
            self.recommendation_agent = None

    async def _analyze_with_llm(
        self,
        content: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> dict[str, Any] | None:
        """Analyze content using inference-router LLM plane.

        Args:
            content: Content to analyze.
            system_prompt: System prompt that is expected to instruct the
                model to emit JSON.
            temperature: Unused — kept for backward signature compatibility;
                inference-router does not honour per-request sampling
                overrides today.
            max_tokens: Unused — same rationale as ``temperature``.

        Returns:
            Parsed JSON response or None if LLM unavailable.
        """
        del temperature, max_tokens  # accepted for back-compat, not forwarded

        if not self.config.use_ml_backend or not self.config.enable_ai_features:
            return None

        try:
            import json

            from services.ai_test_insights.llm_agent import AITestInsightsAgent

            agent = AITestInsightsAgent()
            text = await agent.execute(prompt=content, system_prompt=system_prompt)
            text = str(text).strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except json.JSONDecodeError as exc:
            logger.warning(f"AITestInsights LLM response was not valid JSON: {exc}")
            return None
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(f"LLM analysis failed: {exc}")
            return None

    async def analyze_test_insights(
        self,
        test_results: list[TestResult],
        quality_results: list[QualityResult],
        time_period: str = "recent",
        include_predictions: bool = True,
    ) -> TestInsightAnalysis:
        """Analyze test insights using modular components.

        Args:
            test_results: List of test execution results
            quality_results: List of quality analysis results
            time_period: Time period for the analysis
            include_predictions: Whether to include failure predictions

        Returns:
            Comprehensive test insight analysis
        """
        logger.info("Starting AI test insights analysis")

        try:
            # Use the report formatter's comprehensive analysis method
            analysis = await self.report_formatter.create_comprehensive_analysis(
                test_results=test_results,
                quality_results=quality_results,
                time_period=time_period,
                include_predictions=include_predictions,
            )

            logger.info("AI test insights analysis completed successfully")

            return analysis

        except Exception as e:
            logger.error(f"AI test insights analysis failed: {e}")
            raise

    async def comprehensive_analysis(
        self,
        test_results: list[TestResult],
        quality_results: list[QualityResult],
        code_changes: list[str] | None = None,
        time_period: str = "recent",
        include_predictions: bool = True,
        specialized_analyzers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive analysis using all specialized analyzers.

        Args:
            test_results: List of test execution results
            quality_results: List of quality analysis results
            code_changes: Optional list of code changes
            time_period: Time period for the analysis
            include_predictions: Whether to include failure predictions
            specialized_analyzers: Optional specialized analyzers

        Returns:
            Comprehensive analysis report
        """
        logger.info("Starting comprehensive test analysis")

        try:
            # Use the report formatter's comprehensive report generation
            comprehensive_report = await self.report_formatter.generate_comprehensive_report(
                test_results=test_results,
                quality_results=quality_results,
                code_changes=code_changes,
                time_period=time_period,
                include_predictions=include_predictions,
                specialized_analyzers=specialized_analyzers,
            )

            logger.info("Comprehensive analysis completed successfully")

            return comprehensive_report

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise

    async def predict_test_failures(
        self, test_results: list[TestResult], code_changes: list[str] | None = None
    ) -> dict[str, float]:
        """Predict likelihood of test failures based on historical data.

        Args:
            test_results: List of test execution results
            code_changes: Optional list of code changes

        Returns:
            Dictionary mapping test groups to failure probabilities
        """
        # Simple prediction based on historical success rates
        predictions = {}

        # Group results by test/language
        test_groups: dict[str, list[TestResult]] = {}
        for result in test_results:
            key = f"{result.language}_{result.test_type or 'unknown'}"
            if key not in test_groups:
                test_groups[key] = []
            test_groups[key].append(result)

        # Calculate failure probability for each group
        for key, group_results in test_groups.items():
            success_rate = sum(1 for r in group_results if r.success) / len(group_results)
            failure_probability = 1 - success_rate

            # Adjust based on recent trends
            recent_results = group_results[-5:]  # Last 5 results
            if recent_results:
                recent_success_rate = sum(1 for r in recent_results if r.success) / len(
                    recent_results
                )
                # Weight recent results more heavily
                failure_probability = 0.7 * failure_probability + 0.3 * (1 - recent_success_rate)

            predictions[key] = min(1.0, max(0.0, failure_probability))

        return predictions

    async def generate_test_recommendations(
        self, analysis: TestInsightAnalysis, focus_areas: list[str] | None = None
    ) -> list[TestInsightRecommendation]:
        """Generate additional test recommendations based on analysis.

        Args:
            analysis: The test insight analysis
            focus_areas: Optional focus areas for recommendations

        Returns:
            List of additional recommendations
        """
        additional_recommendations = []

        # Generate recommendations based on health score
        if analysis.overall_health_score < 70:
            additional_recommendations.append(
                TestInsightRecommendation(
                    recommendation_id=f"health_{str(uuid7())[:8]}",
                    type=TestInsightType.RECOMMENDATIONS,
                    priority="high",
                    confidence=0.85,
                    title="Comprehensive Test Suite Overhaul",
                    description="Test suite health score is below acceptable threshold",
                    rationale=f"Health score of {analysis.overall_health_score:.1f} indicates multiple issues",
                    implementation_steps=[
                        "Conduct comprehensive test suite audit",
                        "Prioritize high-impact improvements",
                        "Implement systematic test improvements",
                        "Establish ongoing monitoring",
                    ],
                    estimated_effort="high",
                    expected_impact="Improve overall test health by 40-60%",
                    supporting_metrics={
                        "health_score": analysis.overall_health_score,
                        "improvement_potential": analysis.improvement_potential,
                    },
                    affected_components=["entire_test_suite"],
                )
            )

        # Generate recommendations based on patterns
        high_impact_patterns = [p for p in analysis.detected_patterns if p.impact_score > 7]
        if high_impact_patterns:
            additional_recommendations.append(
                TestInsightRecommendation(
                    recommendation_id=f"pattern_{str(uuid7())[:8]}",
                    type=TestInsightType.PATTERNS,
                    priority="high",
                    confidence=0.8,
                    title="Address High-Impact Test Patterns",
                    description="High-impact patterns detected that require immediate attention",
                    rationale=f"Found {len(high_impact_patterns)} high-impact patterns",
                    implementation_steps=[
                        "Analyze high-impact patterns in detail",
                        "Develop targeted improvement plans",
                        "Implement pattern-specific solutions",
                        "Monitor pattern resolution",
                    ],
                    estimated_effort="medium",
                    expected_impact="Resolve major test suite issues",
                    supporting_metrics={"high_impact_patterns": len(high_impact_patterns)},
                    affected_components=["test_patterns", "test_reliability"],
                )
            )

        return additional_recommendations

    async def get_cached_analysis(self, analysis_id: str) -> TestInsightAnalysis | None:
        """Get cached analysis results.

        Args:
            analysis_id: The analysis ID to retrieve

        Returns:
            Cached analysis or None if not found
        """
        return await self.report_formatter.get_cached_analysis(analysis_id)

    async def get_cached_comprehensive_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        """Get cached comprehensive analysis results.

        Args:
            analysis_id: The analysis ID to retrieve

        Returns:
            Cached comprehensive analysis or None if not found
        """
        return await self.report_formatter.get_cached_comprehensive_analysis(analysis_id)

    def get_config(self) -> AITestInsightsConfig:
        """Get the current configuration.

        Returns:
            Current configuration instance
        """
        return self.config

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        # Reinitialize components if configuration changed
        self.insight_generator = TestInsightGenerator(self.config)
        self.pattern_detector = TestPatternDetector(self.config)
        self.report_formatter = TestReportFormatter(self.config, self.cache_manager)


# Factory function for dependency injection
def create_ai_test_insights_service(
    cache_manager: Any | None = None,
    ai_insights_engine=None,
    config: AITestInsightsConfig = DEFAULT_CONFIG,
) -> AITestInsightsService:
    """Create AI test insights service.

    Args:
        cache_manager: Optional cache manager
        ai_insights_engine: Optional AI insights engine
        config: Configuration for the service

    Returns:
        AITestInsightsService instance
    """
    return AITestInsightsService(
        cache_manager=cache_manager,
        ai_insights_engine=ai_insights_engine,
        config=config,
    )
