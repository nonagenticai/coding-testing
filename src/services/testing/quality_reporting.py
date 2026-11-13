# /src/app/services/testing/quality_reporting.py
"""
Quality Reporting and Dashboards Module
Handles generation of quality reports and dashboard data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from models.testing_models import QualityIssue, QualityResult
from services.testing.quality_metrics import (
    QualityMetrics,
    QualityScoreCalculator,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityReportSection:
    """A section within a quality report.

    Attributes:
        title (str): Section title.
        content (Dict[str, Any]): Section content.
        charts (List[Dict[str, Any]]): Chart data for visualization.
        priority (str): Section priority level.
    """

    title: str
    content: dict[str, Any]
    charts: list[dict[str, Any]]
    priority: str = "medium"


@dataclass
class QualityReport:
    """Complete quality report structure.

    Attributes:
        report_id (str): Unique report identifier.
        title (str): Report title.
        generated_at (datetime): Report generation timestamp.
        analysis_id (str): Associated analysis identifier.
        summary (Dict[str, Any]): Executive summary.
        sections (List[QualityReportSection]): Report sections.
        metadata (Dict[str, Any]): Additional report metadata.
    """

    report_id: str
    title: str
    generated_at: datetime
    analysis_id: str
    summary: dict[str, Any]
    sections: list[QualityReportSection]
    metadata: dict[str, Any]


class QualityReportGenerator:
    """Generates comprehensive quality reports."""

    def __init__(self):
        self.score_calculator = QualityScoreCalculator()

    async def generate_comprehensive_report(
        self,
        quality_result: QualityResult,
        include_charts: bool = True,
        report_format: str = "detailed",
    ) -> QualityReport:
        """Generate a comprehensive quality report.

        Args:
            quality_result: Quality analysis result to report on.
            include_charts: Whether to include chart data.
            report_format: Format level (summary, detailed, comprehensive).

        Returns:
            Complete quality report.
        """
        from uuid import uuid7

        report_id = str(uuid7())

        # Calculate comprehensive metrics
        metrics = self.score_calculator.calculate_comprehensive_metrics(
            quality_result.tools_results, quality_result.issues
        )

        # Generate sections based on format
        sections = []

        if report_format in ["detailed", "comprehensive"]:
            sections.extend(
                [
                    await self._create_executive_summary_section(quality_result, metrics),
                    await self._create_issues_analysis_section(quality_result.issues),
                    await self._create_tools_performance_section(quality_result.tools_results),
                    await self._create_recommendations_section(quality_result, metrics),
                ]
            )

        if report_format == "comprehensive":
            sections.extend(
                [
                    await self._create_technical_details_section(quality_result),
                    await self._create_trends_section(quality_result),
                    await self._create_benchmarks_section(metrics, quality_result.language),
                ]
            )

        # Create summary
        summary = await self._create_report_summary(quality_result, metrics)

        return QualityReport(
            report_id=report_id,
            title=f"Quality Analysis Report - {quality_result.language.title()}",
            generated_at=datetime.now(UTC),
            analysis_id=quality_result.analysis_id,
            summary=summary,
            sections=sections,
            metadata={
                "format": report_format,
                "include_charts": include_charts,
                "language": quality_result.language,
                "tools_used": list(quality_result.tools_results.keys()),
                "total_execution_time": quality_result.execution_time,
            },
        )

    async def _create_executive_summary_section(
        self, result: QualityResult, metrics: QualityMetrics
    ) -> QualityReportSection:
        """Create executive summary section."""

        # Determine overall assessment
        if metrics.overall_score >= 8.0:
            assessment = "Excellent"
            status_color = "green"
        elif metrics.overall_score >= 6.0:
            assessment = "Good"
            status_color = "yellow"
        elif metrics.overall_score >= 4.0:
            assessment = "Needs Improvement"
            status_color = "orange"
        else:
            assessment = "Poor"
            status_color = "red"

        content = {
            "overall_score": metrics.overall_score,
            "quality_grade": metrics.quality_grade,
            "assessment": assessment,
            "status_color": status_color,
            "total_issues": metrics.total_issues,
            "critical_issues": metrics.severity_breakdown.get("error", 0),
            "improvement_potential": f"{metrics.improvement_potential}%",
            "analysis_duration": f"{result.execution_time:.2f}s",
            "key_findings": await self._extract_key_findings(result.issues),
            "priority_actions": await self._identify_priority_actions(result.issues),
        }

        charts = [
            {
                "type": "gauge",
                "title": "Quality Score",
                "data": {
                    "value": metrics.overall_score,
                    "max": 10,
                    "color": status_color,
                },
            },
            {
                "type": "donut",
                "title": "Issues by Severity",
                "data": metrics.severity_breakdown,
            },
        ]

        return QualityReportSection(
            title="Executive Summary", content=content, charts=charts, priority="high"
        )

    async def _create_issues_analysis_section(
        self, issues: list[QualityIssue]
    ) -> QualityReportSection:
        """Create detailed issues analysis section."""

        # Group issues by various dimensions
        by_severity = {}
        by_tool = {}
        by_rule = {}
        by_file = {}

        for issue in issues:
            # By severity
            severity = issue.severity
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # By tool
            tool = issue.tool
            by_tool[tool] = by_tool.get(tool, 0) + 1

            # By rule
            rule_id = issue.rule_id or "unknown"
            by_rule[rule_id] = by_rule.get(rule_id, 0) + 1

            # By file
            file_path = issue.file_path or "unknown"
            by_file[file_path] = by_file.get(file_path, 0) + 1

        # Find top issues
        top_rules = sorted(by_rule.items(), key=lambda x: x[1], reverse=True)[:10]
        most_affected_files = sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:5]

        content = {
            "total_issues": len(issues),
            "by_severity": by_severity,
            "by_tool": by_tool,
            "by_rule": dict(top_rules),
            "most_affected_files": dict(most_affected_files),
            "fixable_issues": len([i for i in issues if i.suggested_fix]),
            "high_confidence_fixes": len(
                [i for i in issues if i.fix_confidence and i.fix_confidence > 0.8]
            ),
        }

        charts = [
            {"type": "bar", "title": "Issues by Tool", "data": by_tool},
            {"type": "bar", "title": "Top Issue Types", "data": dict(top_rules[:5])},
        ]

        return QualityReportSection(
            title="Issues Analysis", content=content, charts=charts, priority="high"
        )

    async def _create_tools_performance_section(
        self, tools_results: dict[str, Any]
    ) -> QualityReportSection:
        """Create tools performance analysis section."""

        total_execution_time = sum(
            result.get("execution_time", 0) for result in tools_results.values()
        )

        successful_tools = [
            tool for tool, result in tools_results.items() if result.get("success", False)
        ]

        failed_tools = [
            tool for tool, result in tools_results.items() if not result.get("success", False)
        ]

        # Performance metrics
        performance_data = {}
        for tool, result in tools_results.items():
            performance_data[tool] = {
                "execution_time": result.get("execution_time", 0),
                "success": result.get("success", False),
                "issues_found": result.get("issues_count", 0),
                "efficiency": (
                    result.get("issues_count", 0) / max(result.get("execution_time", 1), 0.1)
                ),
            }

        content = {
            "total_tools": len(tools_results),
            "successful_tools": len(successful_tools),
            "failed_tools": len(failed_tools),
            "total_execution_time": total_execution_time,
            "average_execution_time": total_execution_time / max(len(tools_results), 1),
            "performance_data": performance_data,
            "success_rate": len(successful_tools) / max(len(tools_results), 1) * 100,
        }

        charts = [
            {
                "type": "bar",
                "title": "Tool Execution Time",
                "data": {tool: data["execution_time"] for tool, data in performance_data.items()},
            },
            {
                "type": "scatter",
                "title": "Tool Efficiency (Issues/Second)",
                "data": {tool: data["efficiency"] for tool, data in performance_data.items()},
            },
        ]

        return QualityReportSection(
            title="Tools Performance", content=content, charts=charts, priority="medium"
        )

    async def _create_recommendations_section(
        self, result: QualityResult, metrics: QualityMetrics
    ) -> QualityReportSection:
        """Create recommendations section."""

        recommendations = []

        # Score-based recommendations
        if metrics.overall_score < 5.0:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "immediate",
                    "title": "Address Critical Issues",
                    "description": "Focus on fixing error-level issues first to improve code stability.",
                    "effort": "medium",
                    "impact": "high",
                }
            )

        # Tool-specific recommendations
        error_issues = [i for i in result.issues if i.severity == "error"]
        if error_issues:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "code_quality",
                    "title": "Fix Critical Errors",
                    "description": f"Address {len(error_issues)} critical errors that may cause runtime issues.",
                    "effort": "high",
                    "impact": "high",
                }
            )

        # Automation recommendations
        fixable_issues = [i for i in result.issues if i.suggested_fix]
        if fixable_issues:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "automation",
                    "title": "Apply Automated Fixes",
                    "description": f"Auto-fix {len(fixable_issues)} issues using automated tools.",
                    "effort": "low",
                    "impact": "medium",
                }
            )

        # Process improvements
        if metrics.improvement_potential > 20:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "process",
                    "title": "Implement Pre-commit Hooks",
                    "description": "Set up automated quality checks to prevent future issues.",
                    "effort": "medium",
                    "impact": "high",
                }
            )

        content = {
            "total_recommendations": len(recommendations),
            "high_priority": len([r for r in recommendations if r["priority"] == "high"]),
            "quick_wins": len([r for r in recommendations if r["effort"] == "low"]),
            "recommendations": recommendations,
        }

        charts = [
            {
                "type": "matrix",
                "title": "Effort vs Impact Matrix",
                "data": {
                    "items": [
                        {
                            "name": rec["title"],
                            "effort": {"low": 1, "medium": 2, "high": 3}[rec["effort"]],
                            "impact": {"low": 1, "medium": 2, "high": 3}[rec["impact"]],
                        }
                        for rec in recommendations
                    ]
                },
            }
        ]

        return QualityReportSection(
            title="Recommendations", content=content, charts=charts, priority="high"
        )

    async def _create_technical_details_section(
        self, result: QualityResult
    ) -> QualityReportSection:
        """Create technical details section."""

        content = {
            "analysis_id": result.analysis_id,
            "language": result.language,
            "timestamp": result.created_at.isoformat(),
            "execution_details": {
                "total_time": result.execution_time,
                "success": result.success,
                "error": result.error,
            },
            "tools_configuration": result.tools_results,
            "environment": {
                "analysis_engine": "coding-analysis",
                "version": "1.0.0",
            },
        }

        return QualityReportSection(
            title="Technical Details", content=content, charts=[], priority="low"
        )

    async def _create_trends_section(self, result: QualityResult) -> QualityReportSection:
        """Create trends analysis section."""

        # This would integrate with historical data
        content = {
            "current_score": result.overall_score,
            "trend_analysis": "Not available - no historical data",
            "baseline_comparison": "Not available - no baseline set",
            "recommendations": [
                "Establish baseline metrics",
                "Track quality over time",
                "Set up continuous monitoring",
            ],
        }

        return QualityReportSection(
            title="Quality Trends", content=content, charts=[], priority="low"
        )

    async def _create_benchmarks_section(
        self, metrics: QualityMetrics, language: str
    ) -> QualityReportSection:
        """Create benchmarks comparison section."""

        from services.testing.quality_metrics import QualityBenchmarkComparator

        comparator = QualityBenchmarkComparator()
        benchmark_result = comparator.compare_to_benchmarks(metrics.overall_score, language)

        content = {
            "current_score": metrics.overall_score,
            "quality_level": benchmark_result["quality_level"],
            "estimated_percentile": benchmark_result["estimated_percentile"],
            "gap_to_next_level": benchmark_result["gap_to_next_level"],
            "next_level_score": benchmark_result["next_level_score"],
            "language_benchmarks": benchmark_result["benchmarks"],
        }

        charts = [
            {
                "type": "gauge",
                "title": "Benchmark Comparison",
                "data": {
                    "value": metrics.overall_score,
                    "benchmarks": benchmark_result["benchmarks"],
                },
            }
        ]

        return QualityReportSection(
            title="Benchmark Comparison",
            content=content,
            charts=charts,
            priority="medium",
        )

    async def _create_report_summary(
        self, result: QualityResult, metrics: QualityMetrics
    ) -> dict[str, Any]:
        """Create high-level report summary."""

        return {
            "overall_score": metrics.overall_score,
            "quality_grade": metrics.quality_grade,
            "total_issues": metrics.total_issues,
            "critical_issues": metrics.severity_breakdown.get("error", 0),
            "success": result.success,
            "execution_time": result.execution_time,
            "tools_used": len(result.tools_results),
            "language": result.language,
            "improvement_potential": metrics.improvement_potential,
        }

    async def _extract_key_findings(self, issues: list[QualityIssue]) -> list[str]:
        """Extract key findings from issues."""
        findings = []

        # Most common issue types
        rule_counts = {}
        for issue in issues:
            rule_id = issue.rule_id or "unknown"
            rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

        if rule_counts:
            top_rule = max(rule_counts.items(), key=lambda x: x[1])
            findings.append(f"Most common issue: {top_rule[0]} ({top_rule[1]} occurrences)")

        # Critical issues
        critical_count = len([i for i in issues if i.severity == "error"])
        if critical_count > 0:
            findings.append(f"{critical_count} critical errors require immediate attention")

        # Fixable issues
        fixable_count = len([i for i in issues if i.suggested_fix])
        if fixable_count > 0:
            findings.append(f"{fixable_count} issues can be automatically fixed")

        return findings[:5]  # Limit to top 5 findings

    async def _identify_priority_actions(self, issues: list[QualityIssue]) -> list[str]:
        """Identify priority actions based on issues."""
        actions = []

        # Critical errors first
        critical_issues = [i for i in issues if i.severity == "error"]
        if critical_issues:
            actions.append("Fix critical errors to prevent runtime failures")

        # Security issues
        security_issues = [i for i in issues if i.tool == "bandit"]
        if security_issues:
            actions.append("Address security vulnerabilities")

        # Type issues
        type_issues = [i for i in issues if i.tool == "ty"]
        if type_issues:
            actions.append("Improve type safety")

        # Automated fixes
        auto_fixable = [
            i for i in issues if i.suggested_fix and i.fix_confidence and i.fix_confidence > 0.8
        ]
        if auto_fixable:
            actions.append("Apply automated formatting and style fixes")

        return actions[:3]  # Top 3 priority actions


class QualityDashboardData:
    """Generates dashboard data for quality visualization."""

    def __init__(self):
        self.score_calculator = QualityScoreCalculator()

    async def generate_dashboard_summary(
        self, quality_results: list[QualityResult]
    ) -> dict[str, Any]:
        """Generate summary data for quality dashboard.

        Args:
            quality_results: List of quality analysis results.

        Returns:
            Dashboard summary data.
        """
        if not quality_results:
            return {
                "total_analyses": 0,
                "average_score": 0.0,
                "trend": "no_data",
                "issues_summary": {},
                "tools_usage": {},
            }

        # Calculate aggregated metrics
        total_analyses = len(quality_results)
        average_score = sum(r.overall_score for r in quality_results) / total_analyses

        # Aggregate issues
        all_issues = []
        for result in quality_results:
            all_issues.extend(result.issues)

        # Issues summary
        issues_by_severity = {}
        issues_by_tool = {}

        for issue in all_issues:
            # By severity
            severity = issue.severity
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1

            # By tool
            tool = issue.tool
            issues_by_tool[tool] = issues_by_tool.get(tool, 0) + 1

        # Tools usage
        tools_usage = {}
        for result in quality_results:
            for tool in result.tools_results.keys():
                tools_usage[tool] = tools_usage.get(tool, 0) + 1

        # Trend calculation (simplified)
        if len(quality_results) >= 2:
            recent_score = quality_results[-1].overall_score
            previous_score = quality_results[-2].overall_score
            trend = (
                "improving"
                if recent_score > previous_score
                else "declining"
                if recent_score < previous_score
                else "stable"
            )
        else:
            trend = "insufficient_data"

        return {
            "total_analyses": total_analyses,
            "average_score": round(average_score, 2),
            "trend": trend,
            "issues_summary": {
                "total": len(all_issues),
                "by_severity": issues_by_severity,
                "by_tool": issues_by_tool,
            },
            "tools_usage": tools_usage,
            "last_analysis": (
                quality_results[-1].created_at.isoformat() if quality_results else None
            ),
        }

    async def generate_detailed_charts(
        self, quality_results: list[QualityResult]
    ) -> list[dict[str, Any]]:
        """Generate detailed chart data for dashboard.

        Args:
            quality_results: List of quality analysis results.

        Returns:
            List of chart configurations.
        """
        charts = []

        if not quality_results:
            return charts

        # Score trend chart
        score_trend = {
            "type": "line",
            "title": "Quality Score Trend",
            "data": {
                "labels": [r.created_at.strftime("%Y-%m-%d") for r in quality_results],
                "values": [r.overall_score for r in quality_results],
            },
        }
        charts.append(score_trend)

        # Issues by severity (aggregated)
        all_issues = []
        for result in quality_results:
            all_issues.extend(result.issues)

        severity_counts = {}
        for issue in all_issues:
            severity = issue.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        if severity_counts:
            severity_chart = {
                "type": "donut",
                "title": "Issues by Severity",
                "data": severity_counts,
            }
            charts.append(severity_chart)

        # Tools performance
        tools_performance = {}
        for result in quality_results:
            for tool, tool_result in result.tools_results.items():
                if tool not in tools_performance:
                    tools_performance[tool] = {
                        "total_time": 0,
                        "total_issues": 0,
                        "runs": 0,
                    }

                tools_performance[tool]["total_time"] += tool_result.get("execution_time", 0)
                tools_performance[tool]["total_issues"] += tool_result.get("issues_count", 0)
                tools_performance[tool]["runs"] += 1

        if tools_performance:
            avg_times = {
                tool: data["total_time"] / data["runs"] for tool, data in tools_performance.items()
            }

            performance_chart = {
                "type": "bar",
                "title": "Average Tool Execution Time",
                "data": avg_times,
            }
            charts.append(performance_chart)

        return charts


# Export functions for easy access
async def generate_quality_report(
    quality_result: QualityResult, format_type: str = "detailed"
) -> QualityReport:
    """Generate a quality report."""
    generator = QualityReportGenerator()
    return await generator.generate_comprehensive_report(quality_result, report_format=format_type)


async def export_report_to_json(report: QualityReport) -> str:
    """Export report to JSON format."""
    return json.dumps(asdict(report), default=str, indent=2)


async def export_report_to_html(report: QualityReport) -> str:
    """Export report to HTML format."""
    # Basic HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .summary {{ background: #e9ecef; }}
            .score {{ font-size: 2em; font-weight: bold; color: #28a745; }}
            .issues {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>Generated: {generated_at}</p>
            <p>Analysis ID: {analysis_id}</p>
        </div>

        <div class="section summary">
            <h2>Summary</h2>
            <div class="score">Score: {overall_score}/10</div>
            <p>Total Issues: <span class="issues">{total_issues}</span></p>
        </div>

        {sections_html}
    </body>
    </html>
    """

    # Generate sections HTML
    sections_html = ""
    for section in report.sections:
        sections_html += f"""
        <div class="section">
            <h2>{section.title}</h2>
            <pre>{json.dumps(section.content, indent=2)}</pre>
        </div>
        """

    return html_template.format(
        title=report.title,
        generated_at=report.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        analysis_id=report.analysis_id,
        overall_score=report.summary.get("overall_score", 0),
        total_issues=report.summary.get("total_issues", 0),
        sections_html=sections_html,
    )
