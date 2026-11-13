"""Report formatting and output module for AI Test Insights Service.

This module contains logic for formatting and generating comprehensive
test insight reports and analysis summaries.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

from models.testing_models import QualityResult, TestResult
from services.ai_test_insights.config import (
    DEFAULT_CONFIG,
    AITestInsightsConfig,
    calculate_health_score,
    calculate_improvement_potential,
)
from services.ai_test_insights.insight_generation import TestInsightGenerator
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestInsightAnalysis,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)
from services.ai_test_insights.pattern_recognition import TestPatternDetector

logger = logging.getLogger(__name__)


class TestReportFormatter:
    """Formats and generates comprehensive test insight reports."""

    def __init__(
        self,
        config: AITestInsightsConfig = DEFAULT_CONFIG,
        cache_manager: Any | None = None,
    ):
        """Initialize the report formatter.

        Args:
            config: Configuration for the AI test insights service
            cache_manager: Optional cache manager for storing results
        """
        self.config = config
        self.cache_manager = cache_manager
        self.insight_generator = TestInsightGenerator(config)
        self.pattern_detector = TestPatternDetector(config)

    async def create_comprehensive_analysis(
        self,
        test_results: list[TestResult],
        quality_results: list[QualityResult],
        time_period: str = "recent",
        include_predictions: bool = True,
    ) -> TestInsightAnalysis:
        """Create a comprehensive test insight analysis.

        Args:
            test_results: List of test execution results
            quality_results: List of quality analysis results
            time_period: Time period for the analysis
            include_predictions: Whether to include failure predictions

        Returns:
            Comprehensive test insight analysis
        """
        analysis_id = str(uuid7())

        logger.info(f"Creating comprehensive test analysis {analysis_id}")

        try:
            # Prepare analysis data
            analysis_data = await self._prepare_analysis_data(test_results, quality_results)

            # Run analysis tasks in parallel
            analysis_tasks = [
                self.insight_generator.analyze_performance(analysis_data),
                self.insight_generator.analyze_reliability(analysis_data),
                self.insight_generator.analyze_coverage(analysis_data),
                self.insight_generator.analyze_quality(analysis_data),
                self.pattern_detector.detect_patterns(analysis_data),
            ]

            # Wait for all analysis tasks to complete
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Unpack results with error handling
            performance_metrics = self._handle_result(
                results[0], "performance", TestPerformanceMetrics
            )
            reliability_metrics = self._handle_result(
                results[1], "reliability", TestReliabilityMetrics
            )
            coverage_insights = self._handle_result(results[2], "coverage", TestCoverageInsights)
            quality_insights = self._handle_result(results[3], "quality", TestQualityInsights)
            detected_patterns = self._handle_result(results[4], "patterns", list)

            # Generate recommendations
            recommendations = await self.insight_generator.generate_recommendations(
                performance_metrics,
                reliability_metrics,
                coverage_insights,
                quality_insights,
                detected_patterns,
            )

            # Calculate overall health score
            health_score = calculate_health_score(
                performance_metrics.performance_score,
                reliability_metrics.reliability_score,
                coverage_insights.overall_coverage_score,
                quality_insights.combined_quality_score,
                self.config.analysis_weights,
            )

            # Calculate improvement potential
            improvement_potential = calculate_improvement_potential(
                performance_metrics.performance_score,
                reliability_metrics.reliability_score,
                coverage_insights.overall_coverage_score,
                quality_insights.combined_quality_score,
                sum(p.impact_score for p in detected_patterns),
            )

            # Create comprehensive analysis
            analysis = TestInsightAnalysis(
                analysis_id=analysis_id,
                analyzed_period=time_period,
                performance_metrics=performance_metrics,
                reliability_metrics=reliability_metrics,
                coverage_insights=coverage_insights,
                quality_insights=quality_insights,
                detected_patterns=detected_patterns,
                recommendations=recommendations,
                overall_health_score=health_score,
                improvement_potential=improvement_potential,
            )

            # Cache results if cache manager is available
            if self.cache_manager:
                await self._cache_analysis_results(analysis_id, analysis)

            logger.info(f"Comprehensive test analysis {analysis_id} completed successfully")

            return analysis

        except Exception as e:
            logger.error(f"Failed to create comprehensive analysis {analysis_id}: {e}")
            raise

    async def generate_comprehensive_report(
        self,
        test_results: list[TestResult],
        quality_results: list[QualityResult],
        code_changes: list[str] | None = None,
        time_period: str = "recent",
        include_predictions: bool = True,
        specialized_analyzers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive report using all available analyzers.

        Args:
            test_results: List of test execution results
            quality_results: List of quality analysis results
            code_changes: Optional list of code changes
            time_period: Time period for the analysis
            include_predictions: Whether to include failure predictions
            specialized_analyzers: Optional specialized analyzers

        Returns:
            Comprehensive report dictionary
        """
        analysis_id = str(uuid7())

        logger.info(f"Generating comprehensive report {analysis_id}")

        try:
            # Create core test insights analysis
            test_insights = await self.create_comprehensive_analysis(
                test_results=test_results,
                quality_results=quality_results,
                time_period=time_period,
                include_predictions=include_predictions,
            )

            # Prepare analysis tasks
            analysis_tasks = []

            # Use specialized analyzers if provided
            if specialized_analyzers:
                if "pattern_analyzer" in specialized_analyzers:
                    analysis_tasks.append(
                        specialized_analyzers["pattern_analyzer"].analyze_patterns(
                            test_results=test_results, quality_results=quality_results
                        )
                    )

                if "performance_analyzer" in specialized_analyzers:
                    analysis_tasks.append(
                        specialized_analyzers["performance_analyzer"].analyze_performance(
                            test_results
                        )
                    )

                if "quality_analyzer" in specialized_analyzers:
                    analysis_tasks.append(
                        specialized_analyzers["quality_analyzer"].analyze_quality_metrics(
                            quality_results=quality_results, test_results=test_results
                        )
                    )

                if "failure_prediction_service" in specialized_analyzers and include_predictions:
                    analysis_tasks.append(
                        specialized_analyzers["failure_prediction_service"].predict_failures(
                            test_results=test_results, code_changes=code_changes
                        )
                    )

                if "recommendation_engine" in specialized_analyzers:
                    # This will be called after other analyses complete
                    pass

            # Execute specialized analyses if any
            specialized_results = {}
            if analysis_tasks:
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

                analyzer_names = []
                if specialized_analyzers:
                    if "pattern_analyzer" in specialized_analyzers:
                        analyzer_names.append("pattern_analysis")
                    if "performance_analyzer" in specialized_analyzers:
                        analyzer_names.append("performance_analysis")
                    if "quality_analyzer" in specialized_analyzers:
                        analyzer_names.append("quality_analysis")
                    if (
                        "failure_prediction_service" in specialized_analyzers
                        and include_predictions
                    ):
                        analyzer_names.append("failure_predictions")

                for i, result in enumerate(results):
                    if i < len(analyzer_names) and not isinstance(result, Exception):
                        specialized_results[analyzer_names[i]] = (
                            result.model_dump() if hasattr(result, "model_dump") else result  # type: ignore[call-non-callable]  # ty: ignore[call-non-callable]
                        )

            # Generate comprehensive recommendations if recommendation engine is available
            comprehensive_recommendations = None
            if specialized_analyzers and "recommendation_engine" in specialized_analyzers:
                try:
                    comprehensive_recommendations = await specialized_analyzers[
                        "recommendation_engine"
                    ].generate_recommendations(
                        test_insights=test_insights,
                        pattern_analysis=specialized_results.get("pattern_analysis"),
                        performance_analysis=specialized_results.get("performance_analysis"),
                        quality_analysis=specialized_results.get("quality_analysis"),
                        test_results=test_results,
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate comprehensive recommendations: {e}")

            # Compile comprehensive report
            comprehensive_report = {
                "analysis_id": analysis_id,
                "analysis_type": "comprehensive",
                "analyzed_at": datetime.now(UTC).isoformat(),
                "input_summary": {
                    "test_results_count": len(test_results),
                    "quality_results_count": len(quality_results),
                    "code_changes": len(code_changes) if code_changes else 0,
                    "time_period": time_period,
                    "predictions_included": include_predictions,
                },
                "core_analysis": {"test_insights": test_insights.model_dump()},
                "specialized_analyses": specialized_results,
                "recommendations": {
                    "core_recommendations": [r.model_dump() for r in test_insights.recommendations],
                    "comprehensive_recommendations": (
                        comprehensive_recommendations.model_dump()
                        if comprehensive_recommendations
                        else None
                    ),
                },
                "summary": self._generate_summary(
                    test_insights, specialized_results, comprehensive_recommendations
                ),
                "metadata": {
                    "config": self.config.to_dict(),
                    "generation_timestamp": datetime.now(UTC).isoformat(),
                    "format_version": "1.0",
                },
            }

            # Cache comprehensive results
            if self.cache_manager:
                await self._cache_comprehensive_analysis(analysis_id, comprehensive_report)

            logger.info(f"Comprehensive report {analysis_id} generated successfully")

            return comprehensive_report

        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            raise

    async def format_analysis_summary(self, analysis: TestInsightAnalysis) -> dict[str, Any]:
        """Format a concise analysis summary.

        Args:
            analysis: The test insight analysis to summarize

        Returns:
            Formatted analysis summary
        """
        summary = {
            "analysis_id": analysis.analysis_id,
            "health_overview": {
                "overall_score": analysis.overall_health_score,
                "health_grade": analysis.health_grade,
                "improvement_potential": analysis.improvement_potential,
            },
            "key_metrics": {
                "performance_score": analysis.performance_metrics.performance_score,
                "reliability_score": analysis.reliability_metrics.reliability_score,
                "coverage_score": analysis.coverage_insights.overall_coverage_score,
                "quality_score": analysis.quality_insights.combined_quality_score,
            },
            "critical_issues": {
                "slow_tests": analysis.performance_metrics.slow_tests_count,
                "flaky_tests": analysis.reliability_metrics.flaky_tests_count,
                "critical_quality_issues": len(analysis.quality_insights.critical_issues),
                "high_impact_patterns": len(
                    [p for p in analysis.detected_patterns if p.impact_score > 7]
                ),
            },
            "top_recommendations": [
                {
                    "id": rec.recommendation_id,
                    "title": rec.title,
                    "priority": rec.priority,
                    "confidence": rec.confidence,
                    "expected_impact": rec.expected_impact,
                }
                for rec in analysis.recommendations[:5]  # Top 5 recommendations
            ],
            "analysis_metadata": {
                "created_at": analysis.created_at.isoformat(),
                "analyzed_period": analysis.analyzed_period,
                "total_patterns_detected": len(analysis.detected_patterns),
                "total_recommendations": len(analysis.recommendations),
            },
        }

        return summary

    async def generate_markdown_report(self, analysis: TestInsightAnalysis) -> str:
        """Generate a markdown-formatted report.

        Args:
            analysis: The test insight analysis to format

        Returns:
            Markdown-formatted report string
        """
        md_lines = []

        # Header
        md_lines.append("# Test Insights Analysis Report")
        md_lines.append(f"**Analysis ID:** `{analysis.analysis_id}`")
        md_lines.append(f"**Generated:** {analysis.created_at.isoformat()}")
        md_lines.append(f"**Period:** {analysis.analyzed_period}")
        md_lines.append("")

        # Executive Summary
        md_lines.append("## Executive Summary")
        md_lines.append(
            f"- **Overall Health Score:** {analysis.overall_health_score:.1f}/100 (Grade: {analysis.health_grade})"
        )
        md_lines.append(f"- **Improvement Potential:** {analysis.improvement_potential:.1f}/100")
        md_lines.append("")

        # Key Metrics
        md_lines.append("## Key Metrics")
        md_lines.append("| Metric | Score | Status |")
        md_lines.append("|--------|-------|--------|")
        md_lines.append(
            f"| Performance | {analysis.performance_metrics.performance_score:.1f}/100 | {'✅' if analysis.performance_metrics.performance_score >= 80 else '⚠️' if analysis.performance_metrics.performance_score >= 60 else '❌'} |"
        )
        md_lines.append(
            f"| Reliability | {analysis.reliability_metrics.reliability_score:.1f}/100 | {'✅' if analysis.reliability_metrics.reliability_score >= 80 else '⚠️' if analysis.reliability_metrics.reliability_score >= 60 else '❌'} |"
        )
        md_lines.append(
            f"| Coverage | {analysis.coverage_insights.overall_coverage_score:.1f}/100 | {'✅' if analysis.coverage_insights.overall_coverage_score >= 80 else '⚠️' if analysis.coverage_insights.overall_coverage_score >= 60 else '❌'} |"
        )
        md_lines.append(
            f"| Quality | {analysis.quality_insights.combined_quality_score:.1f}/100 | {'✅' if analysis.quality_insights.combined_quality_score >= 80 else '⚠️' if analysis.quality_insights.combined_quality_score >= 60 else '❌'} |"
        )
        md_lines.append("")

        # Critical Issues
        md_lines.append("## Critical Issues")
        if analysis.performance_metrics.slow_tests_count > 0:
            md_lines.append(
                f"- **{analysis.performance_metrics.slow_tests_count} slow tests** detected"
            )
        if analysis.reliability_metrics.flaky_tests_count > 0:
            md_lines.append(
                f"- **{analysis.reliability_metrics.flaky_tests_count} flaky tests** detected"
            )
        if analysis.quality_insights.critical_issues:
            md_lines.append(
                f"- **{len(analysis.quality_insights.critical_issues)} critical quality issues**"
            )

        high_impact_patterns = [p for p in analysis.detected_patterns if p.impact_score > 7]
        if high_impact_patterns:
            md_lines.append(f"- **{len(high_impact_patterns)} high-impact patterns** detected")

        if not any(
            [
                analysis.performance_metrics.slow_tests_count,
                analysis.reliability_metrics.flaky_tests_count,
                analysis.quality_insights.critical_issues,
                high_impact_patterns,
            ]
        ):
            md_lines.append("- No critical issues detected ✅")
        md_lines.append("")

        # Top Recommendations
        md_lines.append("## Top Recommendations")
        for i, rec in enumerate(analysis.recommendations[:5], 1):
            priority_emoji = (
                "🔴" if rec.priority == "high" else "🟡" if rec.priority == "medium" else "🟢"
            )
            md_lines.append(f"{i}. {priority_emoji} **{rec.title}**")
            md_lines.append(f"   - *Priority:* {rec.priority.title()}")
            md_lines.append(f"   - *Confidence:* {rec.confidence:.0%}")
            md_lines.append(f"   - *Expected Impact:* {rec.expected_impact}")
            md_lines.append("")

        # Detected Patterns
        if analysis.detected_patterns:
            md_lines.append("## Detected Patterns")
            for pattern in analysis.detected_patterns:
                confidence_emoji = (
                    "🟢"
                    if pattern.confidence >= 0.8
                    else "🟡"
                    if pattern.confidence >= 0.6
                    else "🔴"
                )
                md_lines.append(
                    f"- {confidence_emoji} **{pattern.pattern_type.value.replace('_', ' ').title()}**"
                )
                md_lines.append(f"  - *Description:* {pattern.description}")
                md_lines.append(f"  - *Confidence:* {pattern.confidence:.0%}")
                md_lines.append(f"  - *Impact Score:* {pattern.impact_score:.1f}/10")
                md_lines.append("")

        # Detailed Metrics (collapsed section)
        md_lines.append("<details>")
        md_lines.append("<summary>Detailed Metrics</summary>")
        md_lines.append("")
        md_lines.append("### Performance Metrics")
        md_lines.append(
            f"- Average Execution Time: {analysis.performance_metrics.average_execution_time:.2f}s"
        )
        md_lines.append(
            f"- P95 Execution Time: {analysis.performance_metrics.p95_execution_time:.2f}s"
        )
        md_lines.append(f"- Slow Tests: {analysis.performance_metrics.slow_tests_count}")
        md_lines.append("")
        md_lines.append("### Reliability Metrics")
        md_lines.append(f"- Success Rate: {analysis.reliability_metrics.overall_success_rate:.1%}")
        md_lines.append(
            f"- Stability Score: {analysis.reliability_metrics.stability_score:.1f}/100"
        )
        md_lines.append("")
        md_lines.append("### Coverage Insights")
        md_lines.append(f"- Line Coverage: {analysis.coverage_insights.line_coverage:.1f}%")
        md_lines.append(f"- Branch Coverage: {analysis.coverage_insights.branch_coverage:.1f}%")
        md_lines.append("")
        md_lines.append("### Quality Insights")
        md_lines.append(
            f"- Test Quality Score: {analysis.quality_insights.test_quality_score:.1f}/100"
        )
        md_lines.append(f"- Quality Debt: {analysis.quality_insights.quality_debt:.1f} hours")
        md_lines.append("")
        md_lines.append("</details>")

        return "\n".join(md_lines)

    async def _prepare_analysis_data(
        self, test_results: list[TestResult], quality_results: list[QualityResult]
    ) -> dict[str, Any]:
        """Prepare data for analysis."""
        # Extract performance data
        execution_times = [r.execution_time for r in test_results if r.execution_time is not None]
        success_rates = [r.success for r in test_results]
        error_messages = [r.stderr for r in test_results if r.stderr]

        # Extract quality data
        quality_scores = [r.overall_score for r in quality_results if r.overall_score is not None]
        quality_issues = []
        for qr in quality_results:
            if hasattr(qr, "issues") and qr.issues:
                quality_issues.extend(qr.issues)

        # Group by language and test type
        results_by_language = {}
        for result in test_results:
            if result.language not in results_by_language:
                results_by_language[result.language] = []
            results_by_language[result.language].append(result)

        # Calculate basic statistics
        if execution_times:
            avg_execution_time = sum(execution_times) / len(execution_times)
            execution_times_sorted = sorted(execution_times)
            median_execution_time = execution_times_sorted[len(execution_times_sorted) // 2]
            p95_execution_time = execution_times_sorted[int(0.95 * len(execution_times_sorted))]
            p99_execution_time = execution_times_sorted[int(0.99 * len(execution_times_sorted))]
        else:
            avg_execution_time = median_execution_time = p95_execution_time = p99_execution_time = 0

        overall_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0

        return {
            "test_results": test_results,
            "quality_results": quality_results,
            "execution_times": execution_times,
            "success_rates": success_rates,
            "error_messages": error_messages,
            "quality_scores": quality_scores,
            "quality_issues": quality_issues,
            "results_by_language": results_by_language,
            "avg_execution_time": avg_execution_time,
            "median_execution_time": median_execution_time,
            "p95_execution_time": p95_execution_time,
            "p99_execution_time": p99_execution_time,
            "overall_success_rate": overall_success_rate,
            "total_tests": len(test_results),
            "total_quality_analyses": len(quality_results),
        }

    def _handle_result(self, result: Any, analysis_type: str, expected_type: type) -> Any:
        """Handle analysis result with error handling."""
        if isinstance(result, Exception):
            logger.warning(f"{analysis_type} analysis failed: {result}")
            # Return appropriate default based on expected type
            if expected_type is list:
                return []
            # For other types, use the config's result handler
            from services.ai_test_insights.config import AnalysisResultHandler

            return AnalysisResultHandler.handle_analysis_result(
                result, analysis_type, expected_type, logger
            )

        return result

    def _generate_summary(
        self,
        test_insights: TestInsightAnalysis,
        specialized_results: dict[str, Any],
        comprehensive_recommendations: Any | None,
    ) -> dict[str, Any]:
        """Generate a summary of all analysis results."""
        summary = {
            "overall_health_score": test_insights.overall_health_score,
            "health_grade": test_insights.health_grade,
            "improvement_potential": test_insights.improvement_potential,
            "core_metrics": {
                "performance_score": test_insights.performance_metrics.performance_score,
                "reliability_score": test_insights.reliability_metrics.reliability_score,
                "coverage_score": test_insights.coverage_insights.overall_coverage_score,
                "quality_score": test_insights.quality_insights.combined_quality_score,
            },
            "critical_counts": {
                "slow_tests": test_insights.performance_metrics.slow_tests_count,
                "flaky_tests": test_insights.reliability_metrics.flaky_tests_count,
                "critical_quality_issues": len(test_insights.quality_insights.critical_issues),
                "high_impact_patterns": len(
                    [p for p in test_insights.detected_patterns if p.impact_score > 7]
                ),
            },
            "recommendations_summary": {
                "total_core_recommendations": len(test_insights.recommendations),
                "high_priority_recommendations": len(
                    [r for r in test_insights.recommendations if r.priority == "high"]
                ),
                "has_comprehensive_recommendations": comprehensive_recommendations is not None,
            },
        }

        # Add specialized analysis summaries
        if "performance_analysis" in specialized_results:
            perf_data = specialized_results["performance_analysis"]
            if isinstance(perf_data, dict) and "performance_score" in perf_data:
                summary["specialized_performance_score"] = perf_data["performance_score"]

        if "quality_analysis" in specialized_results:
            qual_data = specialized_results["quality_analysis"]
            if isinstance(qual_data, dict) and "overall_quality_score" in qual_data:
                summary["specialized_quality_score"] = qual_data["overall_quality_score"]

        if "pattern_analysis" in specialized_results:
            pattern_data = specialized_results["pattern_analysis"]
            if isinstance(pattern_data, dict) and "overall_pattern_health" in pattern_data:
                summary["pattern_health"] = pattern_data["overall_pattern_health"]

        if "failure_predictions" in specialized_results:
            pred_data = specialized_results["failure_predictions"]
            if isinstance(pred_data, dict) and "overall_risk_level" in pred_data:
                summary["failure_risk"] = pred_data["overall_risk_level"]

        if comprehensive_recommendations and hasattr(
            comprehensive_recommendations, "total_recommendations"
        ):
            summary["total_comprehensive_recommendations"] = (
                comprehensive_recommendations.total_recommendations
            )
            if hasattr(comprehensive_recommendations, "critical_recommendations"):
                summary["critical_comprehensive_recommendations"] = len(
                    comprehensive_recommendations.critical_recommendations
                )

        return summary

    async def _cache_analysis_results(
        self, analysis_id: str, analysis: TestInsightAnalysis
    ) -> None:
        """Cache analysis results."""
        if not self.cache_manager:
            return

        try:
            cache_key = f"test_insights:{analysis_id}"
            cache_data = analysis.model_dump(exclude_computed_fields=True)

            await self.cache_manager.set(cache_key, cache_data, ttl=self.config.cache_ttl)

        except Exception as e:
            logger.warning(f"Failed to cache analysis results: {e}")

    async def _cache_comprehensive_analysis(
        self, analysis_id: str, comprehensive_report: dict[str, Any]
    ) -> None:
        """Cache comprehensive analysis results."""
        if not self.cache_manager:
            return

        try:
            cache_key = f"comprehensive_test_analysis:{analysis_id}"

            await self.cache_manager.set(
                cache_key, comprehensive_report, ttl=self.config.comprehensive_cache_ttl
            )

        except Exception as e:
            logger.warning(f"Failed to cache comprehensive analysis: {e}")

    async def get_cached_analysis(self, analysis_id: str) -> TestInsightAnalysis | None:
        """Get cached analysis results."""
        if not self.cache_manager:
            return None

        try:
            cache_key = f"test_insights:{analysis_id}"
            cached_data = await self.cache_manager.get(cache_key)

            if cached_data:
                # Strip computed fields that cannot be passed to the constructor
                computed = set(TestInsightAnalysis.model_computed_fields)
                for sub_model_field in (
                    "performance_metrics",
                    "reliability_metrics",
                    "coverage_insights",
                    "quality_insights",
                ):
                    sub_data = cached_data.get(sub_model_field)
                    if isinstance(sub_data, dict):
                        field_type = TestInsightAnalysis.model_fields[sub_model_field].annotation
                        for cf in getattr(field_type, "model_computed_fields", {}):
                            sub_data.pop(cf, None)
                clean = {k: v for k, v in cached_data.items() if k not in computed}
                return TestInsightAnalysis(**clean)

        except Exception as e:
            logger.warning(f"Failed to retrieve cached analysis: {e}")

        return None

    async def get_cached_comprehensive_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        """Get cached comprehensive analysis results."""
        if not self.cache_manager:
            return None

        try:
            cache_key = f"comprehensive_test_analysis:{analysis_id}"
            cached_data = await self.cache_manager.get(cache_key)

            return cached_data

        except Exception as e:
            logger.warning(f"Failed to retrieve cached comprehensive analysis: {e}")
            return None


# Factory function
def create_test_report_formatter(
    config: AITestInsightsConfig = DEFAULT_CONFIG, cache_manager: Any | None = None
) -> TestReportFormatter:
    """Create a test report formatter instance.

    Args:
        config: Configuration for the formatter
        cache_manager: Optional cache manager

    Returns:
        TestReportFormatter instance
    """
    return TestReportFormatter(config=config, cache_manager=cache_manager)
