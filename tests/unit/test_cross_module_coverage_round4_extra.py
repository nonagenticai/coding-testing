"""Round 4 extra coverage tests - second wave targeting remaining uncovered lines.

Focus: testing_utils get_execution_service, report_formatter specialized analyzers,
health_monitor HTTP check, intelligent_test_orchestrator batch paths,
quality_management analyze_quality, test_runner legacy paths,
intelligent_test_integration service methods, recommendation engine generation methods,
core_service LLM paths.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure docker stub is available for health_monitor imports
if "docker" not in sys.modules:
    _docker_stub = types.ModuleType("docker")
    _docker_errors = types.ModuleType("docker.errors")
    _docker_errors.NotFound = type("NotFound", (Exception,), {})
    _docker_errors.APIError = type("APIError", (Exception,), {})
    _docker_errors.ContainerError = type("ContainerError", (Exception,), {})
    _docker_stub.errors = _docker_errors
    _docker_stub.from_env = MagicMock()
    sys.modules["docker"] = _docker_stub
    sys.modules["docker.errors"] = _docker_errors


# ---------------------------------------------------------------------------
# testing_utils.py - get_execution_service (lines 36-200)
# ---------------------------------------------------------------------------


class TestGetExecutionServiceR4:
    """Cover get_execution_service function which creates stub services."""

    @pytest.mark.asyncio
    async def test_get_execution_service_returns_stub_with_correct_capabilities(self):
        """When RedisCacheManager init fails, get_execution_service falls back to
        the stub that exposes _get_supported_tools and test_orchestrator."""
        from routers.testing_modules.testing_utils import get_execution_service
        from services.stubs import RedisCacheManager

        # Force cache_manager=None -> HTTPException -> stub path
        with patch.object(RedisCacheManager, "__init__", side_effect=RuntimeError("no redis")):
            service = await get_execution_service()
        # The stub must expose the same interface as CodeExecutionService
        assert hasattr(service, "_get_supported_tools")
        assert hasattr(service, "test_orchestrator")
        # The stub's python tools must include the complete set
        python_tools = service._get_supported_tools("python")
        assert "ruff-format" in python_tools
        assert "ty" in python_tools
        assert "ruff" in python_tools
        assert "bandit" in python_tools
        for retired in ("black", "isort", "mypy", "pylint", "flake8", "autoflake"):
            assert retired not in python_tools
        assert len(python_tools) == 5

    @pytest.mark.asyncio
    async def test_get_execution_service_stub_tools_per_language(self):
        """Verify the stub returns correct tools for each supported language."""
        from routers.testing_modules.testing_utils import get_execution_service
        from services.stubs import RedisCacheManager

        with patch.object(RedisCacheManager, "__init__", side_effect=RuntimeError("no redis")):
            service = await get_execution_service()

        python_tools = service._get_supported_tools("python")
        assert "ruff-format" in python_tools
        assert "ty" in python_tools
        assert "ruff" in python_tools
        assert len(python_tools) == 5

        go_tools = service._get_supported_tools("go")
        assert "golangci-lint" in go_tools
        assert "gofmt" in go_tools
        assert "staticcheck" in go_tools
        assert len(go_tools) == 7

        rust_tools = service._get_supported_tools("rust")
        assert "clippy" in rust_tools
        assert "rustfmt" in rust_tools
        assert "cargo-audit" in rust_tools
        assert len(rust_tools) == 12

        js_tools = service._get_supported_tools("javascript")
        assert js_tools == ["eslint", "prettier", "jshint"]

        ts_tools = service._get_supported_tools("typescript")
        assert ts_tools == ["eslint", "prettier", "tsc", "tslint"]

        java_tools = service._get_supported_tools("java")
        assert len(java_tools) == 10
        assert "checkstyle" in java_tools
        assert "jacoco" in java_tools

        csharp_tools = service._get_supported_tools("csharp")
        assert csharp_tools == ["dotnet-format", "dotnet-analyzers", "security-code-scan"]

        cpp_tools = service._get_supported_tools("cpp")
        assert cpp_tools == ["clang-format", "clang-tidy", "cppcheck", "cpplint", "valgrind"]

        # Unknown language returns empty list
        unknown_tools = service._get_supported_tools("brainfuck")
        assert unknown_tools == []

    @pytest.mark.asyncio
    async def test_get_execution_service_stub_orchestrator(self):
        """Verify the stub orchestrator returns correct languages and test types."""
        from routers.testing_modules.testing_utils import get_execution_service
        from services.stubs import RedisCacheManager

        with patch.object(RedisCacheManager, "__init__", side_effect=RuntimeError("no redis")):
            service = await get_execution_service()

        languages = await service.test_orchestrator.get_supported_languages()
        assert set(languages) == {
            "go",
            "rust",
            "java",
            "csharp",
            "cpp",
            "python",
            "nodejs",
            "javascript",
            "typescript",
        }

        python_types = await service.test_orchestrator.get_supported_test_types("python")
        assert python_types == ["unit", "integration", "coverage", "security"]

        go_types = await service.test_orchestrator.get_supported_test_types("go")
        assert go_types == ["unit", "benchmark", "coverage", "race", "security", "vet"]

        rust_types = await service.test_orchestrator.get_supported_test_types("rust")
        assert rust_types == ["unit", "doc", "benchmark", "coverage", "security", "clippy", "fmt"]

        # Unknown language falls back to default
        unknown_types = await service.test_orchestrator.get_supported_test_types("brainfuck")
        assert unknown_types == ["unit", "quality"]


# ---------------------------------------------------------------------------
# report_formatter.py - specialized analyzers path (lines 194-310)
# ---------------------------------------------------------------------------


class TestReportFormatterSpecializedR4:
    """Cover generate_comprehensive_report with specialized analyzers."""

    @pytest.mark.asyncio
    async def test_with_pattern_analyzer(self):
        """Test with pattern_analyzer in specialized_analyzers (lines 195-200)."""
        from models.testing_models import QualityResult, TestResult
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter()

        test_result = TestResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time=1.0,
            language="python",
            security_level="medium",
            created_at=datetime.now(UTC),
            container_name="c1",
        )
        quality_result = QualityResult(
            analysis_id="q1",
            success=True,
            overall_score=8.0,
            total_issues=0,
            issues=[],
            tools_results={},
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )

        mock_pattern_analyzer = AsyncMock()
        mock_pattern_result = MagicMock()
        mock_pattern_result.model_dump.return_value = {
            "pattern": "data",
            "overall_pattern_health": 85,
        }
        mock_pattern_analyzer.analyze_patterns.return_value = mock_pattern_result

        mock_perf_analyzer = AsyncMock()
        mock_perf_result = MagicMock()
        mock_perf_result.model_dump.return_value = {"performance_score": 90}
        mock_perf_analyzer.analyze_performance.return_value = mock_perf_result

        mock_quality_analyzer = AsyncMock()
        mock_quality_result = MagicMock()
        mock_quality_result.model_dump.return_value = {"overall_quality_score": 88}
        mock_quality_analyzer.analyze_quality_metrics.return_value = mock_quality_result

        mock_failure_pred = AsyncMock()
        mock_failure_pred.predict_failures.return_value = MagicMock()
        mock_failure_pred.predict_failures.return_value.model_dump.return_value = {
            "overall_risk_level": "low"
        }

        mock_rec_engine = AsyncMock()
        mock_rec_engine.generate_recommendations.return_value = MagicMock()
        mock_rec_engine.generate_recommendations.return_value.model_dump.return_value = {}
        mock_rec_engine.generate_recommendations.return_value.total_recommendations = 5
        mock_rec_engine.generate_recommendations.return_value.critical_recommendations = ["crit1"]

        report = await formatter.generate_comprehensive_report(
            test_results=[test_result],
            quality_results=[quality_result],
            code_changes=["change1"],
            specialized_analyzers={
                "pattern_analyzer": mock_pattern_analyzer,
                "performance_analyzer": mock_perf_analyzer,
                "quality_analyzer": mock_quality_analyzer,
                "failure_prediction_service": mock_failure_pred,
                "recommendation_engine": mock_rec_engine,
            },
        )
        # Verify specialized analyses were populated from the mock return values
        assert report["specialized_analyses"]["pattern_analysis"] == {
            "pattern": "data",
            "overall_pattern_health": 85,
        }
        assert report["specialized_analyses"]["performance_analysis"] == {"performance_score": 90}
        assert report["specialized_analyses"]["quality_analysis"] == {"overall_quality_score": 88}
        assert report["specialized_analyses"]["failure_predictions"] == {
            "overall_risk_level": "low"
        }
        # Comprehensive recommendations should be populated from mock rec engine
        assert report["recommendations"]["comprehensive_recommendations"] == {}
        # Summary should include the specialized scores
        assert report["summary"]["specialized_performance_score"] == 90
        assert report["summary"]["specialized_quality_score"] == 88
        assert report["summary"]["pattern_health"] == 85
        assert report["summary"]["failure_risk"] == "low"
        assert report["summary"]["total_comprehensive_recommendations"] == 5
        assert report["summary"]["critical_comprehensive_recommendations"] == 1
        # Verify report structure
        assert report["analysis_type"] == "comprehensive"
        assert report["input_summary"]["test_results_count"] == 1
        assert report["input_summary"]["quality_results_count"] == 1
        assert report["input_summary"]["code_changes"] == 1

    @pytest.mark.asyncio
    async def test_generate_summary_with_specialized(self):
        """Test _generate_summary with all specialized results (lines 569-635)."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter()

        # Create a minimal mock analysis
        mock_analysis = MagicMock()
        mock_analysis.overall_health_score = 85.0
        mock_analysis.health_grade = "B"
        mock_analysis.improvement_potential = 15.0
        mock_analysis.performance_metrics.performance_score = 80.0
        mock_analysis.performance_metrics.slow_tests_count = 2
        mock_analysis.reliability_metrics.reliability_score = 90.0
        mock_analysis.reliability_metrics.flaky_tests_count = 1
        mock_analysis.coverage_insights.overall_coverage_score = 75.0
        mock_analysis.quality_insights.combined_quality_score = 85.0
        mock_analysis.quality_insights.critical_issues = []
        mock_analysis.detected_patterns = []
        mock_analysis.recommendations = []

        specialized_results = {
            "performance_analysis": {"performance_score": 90},
            "quality_analysis": {"overall_quality_score": 88},
            "pattern_analysis": {"overall_pattern_health": 85},
            "failure_predictions": {"overall_risk_level": "low"},
        }

        mock_recs = MagicMock()
        mock_recs.total_recommendations = 5
        mock_recs.critical_recommendations = ["crit1"]

        summary = formatter._generate_summary(mock_analysis, specialized_results, mock_recs)
        # Verify core metrics were extracted from mock
        assert summary["overall_health_score"] == 85.0
        assert summary["health_grade"] == "B"
        assert summary["improvement_potential"] == 15.0
        assert summary["core_metrics"]["performance_score"] == 80.0
        assert summary["core_metrics"]["reliability_score"] == 90.0
        assert summary["core_metrics"]["coverage_score"] == 75.0
        assert summary["core_metrics"]["quality_score"] == 85.0
        # Critical counts
        assert summary["critical_counts"]["slow_tests"] == 2
        assert summary["critical_counts"]["flaky_tests"] == 1
        assert summary["critical_counts"]["critical_quality_issues"] == 0
        assert summary["critical_counts"]["high_impact_patterns"] == 0
        # Specialized results
        assert summary["specialized_performance_score"] == 90
        assert summary["specialized_quality_score"] == 88
        assert summary["pattern_health"] == 85
        assert summary["failure_risk"] == "low"
        assert summary["total_comprehensive_recommendations"] == 5
        assert summary["critical_comprehensive_recommendations"] == 1

    @pytest.mark.asyncio
    async def test_cache_comprehensive_analysis_stores_with_correct_key(self):
        """Verify _cache_comprehensive_analysis calls cache.set with the correct key and data."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        mock_cache = AsyncMock()
        formatter = TestReportFormatter(cache_manager=mock_cache)
        report_data = {"summary": "test report", "score": 85}
        await formatter._cache_comprehensive_analysis("analysis-42", report_data)

        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == "comprehensive_test_analysis:analysis-42"
        assert call_args[0][1] == report_data
        # TTL should be comprehensive_cache_ttl from config
        assert call_args[1]["ttl"] == formatter.config.comprehensive_cache_ttl

    @pytest.mark.asyncio
    async def test_cache_comprehensive_analysis_no_cache_manager(self):
        """Verify _cache_comprehensive_analysis is a no-op when cache_manager is None."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter(cache_manager=None)
        # Should not raise
        await formatter._cache_comprehensive_analysis("id1", {"test": "data"})
        # Confirm no cache_manager means no operation (no attribute error)
        assert formatter.cache_manager is None

    @pytest.mark.asyncio
    async def test_cache_comprehensive_analysis_handles_cache_error(self):
        """Verify _cache_comprehensive_analysis swallows cache exceptions gracefully."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        mock_cache = AsyncMock()
        mock_cache.set.side_effect = Exception("cache error")
        formatter = TestReportFormatter(cache_manager=mock_cache)
        # Should not raise even though cache.set fails
        await formatter._cache_comprehensive_analysis("id1", {"test": "data"})
        mock_cache.set.assert_called_once()
        # Verify the correct key was attempted
        assert mock_cache.set.call_args[0][0] == "comprehensive_test_analysis:id1"

    @pytest.mark.asyncio
    async def test_prepare_analysis_data_empty(self):
        """Test _prepare_analysis_data with empty lists (lines 500-551)."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter()
        data = await formatter._prepare_analysis_data([], [])
        assert data["total_tests"] == 0
        assert data["avg_execution_time"] == 0
        assert data["median_execution_time"] == 0
        assert data["p95_execution_time"] == 0
        assert data["p99_execution_time"] == 0
        assert data["overall_success_rate"] == 0
        assert data["total_quality_analyses"] == 0
        assert data["test_results"] == []
        assert data["quality_results"] == []
        assert data["execution_times"] == []
        assert data["success_rates"] == []
        assert data["error_messages"] == []
        assert data["quality_scores"] == []
        assert data["quality_issues"] == []
        assert data["results_by_language"] == {}

    @pytest.mark.asyncio
    async def test_prepare_analysis_data_with_data(self):
        """Test _prepare_analysis_data with test data (lines 505-528)."""
        from models.testing_models import QualityResult, TestResult
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter()
        test_results = [
            TestResult(
                test_id=f"t{i}",
                success=i < 3,
                exit_code=0 if i < 3 else 1,
                stdout="ok",
                stderr="" if i < 3 else "error output",
                execution_time=float(i + 1),
                language="python",
                security_level="medium",
                created_at=datetime.now(UTC),
                container_name="c1",
            )
            for i in range(5)
        ]
        quality_results = [
            QualityResult(
                analysis_id="q1",
                success=True,
                overall_score=8.0,
                total_issues=0,
                issues=[],
                tools_results={},
                execution_time=0.5,
                created_at=datetime.now(UTC),
                language="python",
            ),
        ]
        data = await formatter._prepare_analysis_data(test_results, quality_results)
        assert data["total_tests"] == 5
        assert data["total_quality_analyses"] == 1
        # Execution times: [1.0, 2.0, 3.0, 4.0, 5.0], avg = 3.0
        assert data["avg_execution_time"] == 3.0
        # Median of [1.0, 2.0, 3.0, 4.0, 5.0] -> index 2 -> 3.0
        assert data["median_execution_time"] == 3.0
        # 3 successes out of 5 = 0.6
        assert data["overall_success_rate"] == 0.6
        # Results grouped by language
        assert "python" in data["results_by_language"]
        assert len(data["results_by_language"]["python"]) == 5
        # Quality scores extracted
        assert data["quality_scores"] == [8.0]
        # Error messages from failed tests (i=3,4 have stderr "error output")
        assert len(data["error_messages"]) == 2
        assert all(msg == "error output" for msg in data["error_messages"])


# ---------------------------------------------------------------------------
# health_monitor.py - Docker health checker (lines 303-343)
# ---------------------------------------------------------------------------


class TestDockerHealthCheckerR4:
    """Cover DockerHealthChecker check_health paths."""

    @pytest.mark.asyncio
    async def test_docker_health_check_running_healthy(self):
        """Test DockerHealthChecker with running healthy container (lines 169-198)."""
        from services.test_execution.health_monitor import (
            DockerHealthChecker,
            HealthCheckConfig,
            HealthCheckType,
            HealthStatus,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        checker = DockerHealthChecker(config)

        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 200, "percpu_usage": [100, 100]},
                "system_cpu_usage": 10000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100},
                "system_cpu_usage": 9000,
            },
            "memory_stats": {"usage": 100_000_000, "limit": 512_000_000},
        }
        checker.docker_client = MagicMock()
        checker.docker_client.containers.get.return_value = mock_container

        result = await checker.check_health("container123")
        assert result.status == HealthStatus.HEALTHY
        assert result.check_type == HealthCheckType.LIVENESS
        assert result.response_time_ms >= 0
        # CPU: (200-100)/(10000-9000) * 2 * 100 = 20%
        assert result.metadata["cpu_usage_percent"] == pytest.approx(20.0)
        # Memory: 100M/512M * 100 = 19.53%
        assert result.metadata["memory_usage_percent"] == pytest.approx(19.53125)
        assert result.metadata["container_status"] == "running"
        assert "running normally" in result.message

    @pytest.mark.asyncio
    async def test_docker_health_check_exited(self):
        """Test DockerHealthChecker with exited container (lines 205-209)."""
        from services.test_execution.health_monitor import (
            DockerHealthChecker,
            HealthCheckConfig,
            HealthCheckType,
            HealthStatus,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        checker = DockerHealthChecker(config)

        mock_container = MagicMock()
        mock_container.status = "exited"
        mock_container.attrs = {"State": {"ExitCode": 137}}
        checker.docker_client = MagicMock()
        checker.docker_client.containers.get.return_value = mock_container

        result = await checker.check_health("container123")
        assert result.status == HealthStatus.UNHEALTHY
        assert "137" in result.message
        assert result.metadata["exit_code"] == 137
        assert result.metadata["container_status"] == "exited"

    @pytest.mark.asyncio
    async def test_docker_health_check_starting(self):
        """Test DockerHealthChecker with created/restarting container (lines 200-203)."""
        from services.test_execution.health_monitor import (
            DockerHealthChecker,
            HealthCheckConfig,
            HealthCheckType,
            HealthStatus,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        checker = DockerHealthChecker(config)

        mock_container = MagicMock()
        mock_container.status = "created"
        checker.docker_client = MagicMock()
        checker.docker_client.containers.get.return_value = mock_container

        result = await checker.check_health("container123")
        assert result.status == HealthStatus.STARTING
        assert result.metadata["container_status"] == "created"
        assert "starting" in result.message.lower()

    @pytest.mark.asyncio
    async def test_docker_health_check_unknown_status(self):
        """Test DockerHealthChecker with unknown status (lines 211-214)."""
        from services.test_execution.health_monitor import (
            DockerHealthChecker,
            HealthCheckConfig,
            HealthCheckType,
            HealthStatus,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        checker = DockerHealthChecker(config)

        mock_container = MagicMock()
        mock_container.status = "paused"
        checker.docker_client = MagicMock()
        checker.docker_client.containers.get.return_value = mock_container

        result = await checker.check_health("container123")
        assert result.status == HealthStatus.UNKNOWN
        assert "paused" in result.message
        assert result.metadata["container_status"] == "paused"

    @pytest.mark.asyncio
    async def test_docker_health_check_not_found(self):
        """Test DockerHealthChecker with NotFound error (lines 216-219)."""
        import docker

        from services.test_execution.health_monitor import (
            DockerHealthChecker,
            HealthCheckConfig,
            HealthCheckType,
            HealthStatus,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        checker = DockerHealthChecker(config)
        checker.docker_client = MagicMock()
        checker.docker_client.containers.get.side_effect = docker.errors.NotFound("nf")

        result = await checker.check_health("container123")
        assert result.status == HealthStatus.UNHEALTHY
        assert "not found" in result.message.lower()
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_docker_health_check_api_error(self):
        """Test DockerHealthChecker with APIError (lines 221-228)."""
        import docker

        from services.test_execution.health_monitor import (
            DockerHealthChecker,
            HealthCheckConfig,
            HealthCheckType,
            HealthStatus,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        checker = DockerHealthChecker(config)
        checker.docker_client = MagicMock()
        checker.docker_client.containers.get.side_effect = docker.errors.APIError("api error")

        result = await checker.check_health("container123")
        assert result.status == HealthStatus.UNKNOWN
        assert "api error" in result.metadata["error"]
        assert "Health check failed" in result.message

    def test_calculate_cpu_usage_invalid_data(self):
        """Test _calculate_cpu_usage with bad data (lines 241-262)."""
        from services.test_execution.health_monitor import (
            DockerHealthChecker,
            HealthCheckConfig,
            HealthCheckType,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        checker = DockerHealthChecker(config)
        assert checker._calculate_cpu_usage({}) == 0.0
        assert checker._calculate_memory_usage({}) == 0.0
        # Also verify with partial data (missing keys)
        assert checker._calculate_cpu_usage({"cpu_stats": {}}) == 0.0
        assert checker._calculate_memory_usage({"memory_stats": {}}) == 0.0

    @pytest.mark.asyncio
    async def test_docker_health_check_degraded(self):
        """Test DockerHealthChecker with high resource usage (lines 191-192)."""
        from services.test_execution.health_monitor import (
            DockerHealthChecker,
            HealthCheckConfig,
            HealthCheckType,
            HealthStatus,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        checker = DockerHealthChecker(config)

        mock_container = MagicMock()
        mock_container.status = "running"
        # Simulate extremely high CPU and memory usage (both > 95%)
        # CPU: (9800-0)/(10000-0) * 1 * 100 = 98%
        # Memory: 500M/512M * 100 = 97.66%
        mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 9800, "percpu_usage": [9800]},
                "system_cpu_usage": 10000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 0},
                "system_cpu_usage": 0,
            },
            "memory_stats": {"usage": 500_000_000, "limit": 512_000_000},
        }
        checker.docker_client = MagicMock()
        checker.docker_client.containers.get.return_value = mock_container

        result = await checker.check_health("container123")
        # With CPU at 98% and memory at 97.66%, both > 95, status should be DEGRADED
        assert result.status == HealthStatus.DEGRADED
        assert "under stress" in result.message
        assert result.metadata["cpu_usage_percent"] == pytest.approx(98.0)
        assert result.metadata["memory_usage_percent"] == pytest.approx(
            500_000_000 / 512_000_000 * 100
        )


# ---------------------------------------------------------------------------
# intelligent_test_orchestrator.py batch paths (lines 248-320)
# ---------------------------------------------------------------------------


class TestIntelligentOrchestratorBatchR4:
    """Cover batch_execute_tests and optimization paths."""

    @pytest.mark.asyncio
    async def test_execute_test_failure_path(self):
        """Test execute_test failure path (lines 248-270)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionRequest,
            IntelligentTestOrchestrator,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())

        # Mock to force failure
        orch.base_orchestrator = AsyncMock()
        orch.base_orchestrator.execute_tests.side_effect = Exception("execution failed")
        orch.ai_coordinator = AsyncMock()
        orch.ai_coordinator.execution_history = []
        orch.ai_coordinator.record_execution = AsyncMock()

        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
            enable_ai_optimization=False,
        )

        result = await orch.execute_test(req)
        assert result.success is False
        assert result.exit_code == 1
        assert "execution failed" in result.stderr
        assert result.stdout == ""
        assert result.ai_analysis == {"error": "AI analysis failed due to execution error"}
        assert result.container_name == "c1"
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT

    @pytest.mark.asyncio
    async def test_batch_execute_with_exception_results(self):
        """Test batch_execute_tests with exceptions (lines 305-320)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionRequest,
            IntelligentTestOrchestrator,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())
        orch.max_concurrent_tests = 10

        # Mock to return failure
        orch.base_orchestrator = AsyncMock()
        orch.base_orchestrator.execute_tests.side_effect = Exception("batch fail")
        orch.ai_coordinator = AsyncMock()
        orch.ai_coordinator.execution_history = []
        orch.ai_coordinator.record_execution = AsyncMock()
        orch.ai_coordinator.analyze_patterns = AsyncMock(return_value=[])
        orch.ai_coordinator.generate_recommendations = AsyncMock(return_value=[])
        orch.ai_coordinator.optimize_schedule = AsyncMock(return_value=[])
        orch.ai_coordinator.analyze_failure = AsyncMock()

        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
            enable_ai_optimization=False,
            enable_failure_analysis=False,
            enable_pattern_analysis=False,
        )

        results = await orch.batch_execute_tests([req])
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].exit_code == 1
        assert "batch fail" in results[0].stderr
        assert results[0].language == TestLanguage.PYTHON
        assert results[0].test_type == OrchestratorType.UNIT

    @pytest.mark.asyncio
    async def test_optimize_test_request_with_history(self):
        """Test _optimize_test_request with historical data (lines 363-394)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionRequest,
            IntelligentTestOrchestrator,
        )
        from services.test_execution.pydantic_ai_agents import ExecutionMetrics
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())

        # Add historical data - average execution is 5000ms
        orch.ai_coordinator.execution_history = [
            ExecutionMetrics(
                test_id="t1",
                language="python",
                test_type="unit",
                execution_time_ms=5000.0,
                exit_code=0,
                success=True,
                memory_usage_mb=256.0,
            ),
        ]

        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
            enable_ai_optimization=True,
        )

        result = await orch._optimize_test_request(req)
        # Result should be a TestExecutionRequest (base class, not Intelligent)
        from services.test_execution.test_orchestrator import TestExecutionRequest

        assert isinstance(result, TestExecutionRequest)
        # Resource limits should have memory set based on historical 256MB * 1.2 = 307.2 -> 307m
        assert isinstance(result.resource_limits, dict)
        assert result.resource_limits["memory"] == "307m"

    @pytest.mark.asyncio
    async def test_optimize_test_request_without_ai(self):
        """Test _optimize_test_request with AI optimization disabled."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionRequest,
            IntelligentTestOrchestrator,
        )
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionRequest,
            TestLanguage,
        )

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())

        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
            enable_ai_optimization=False,
        )

        result = await orch._optimize_test_request(req)
        assert isinstance(result, TestExecutionRequest)
        # When AI optimization is disabled, request is passed through unmodified
        assert result.source_code == "print(1)"
        assert result.container_name == "c1"


# ---------------------------------------------------------------------------
# test_runner.py - _execute_with_orchestrator path (lines 143-164)
# ---------------------------------------------------------------------------


class TestContainerTestRunnerExtraR4:
    """Cover execute_with_orchestrator and legacy paths."""

    @pytest.mark.asyncio
    async def test_execute_with_orchestrator(self):
        """Test _execute_with_orchestrator (lines 138-185)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import TestRequest
        from processing.test_execution.test_runner import ContainerTestRunner
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionResult,
            TestLanguage,
        )

        mock_docker = MagicMock()
        runner = ContainerTestRunner(mock_docker, TestConfig())

        mock_result = TestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100.0,
            language=TestLanguage.GO,
            test_type=OrchestratorType.UNIT,
            container_name="c1",
            created_at=datetime.now(UTC),
            container_id="docker_123",
            performance_metrics={},
            security_issues=[],
            coverage_report=None,
        )
        runner.orchestrator = AsyncMock()
        runner.orchestrator.execute_tests = AsyncMock(return_value=mock_result)

        request = MagicMock(spec=TestRequest)
        request.language = "go"
        request.code = "package main"
        request.test_code = 'package main\nimport "testing"'
        request.timeout = 300
        request.security_level = "medium"
        request.container_name = "c1"

        start_time = datetime.now(UTC)
        result = await runner._execute_with_orchestrator(
            request,
            "test_id_1",
            start_time,
        )
        assert result.success is True
        assert result.language == "go"
        assert result.test_id == "test_id_1"
        assert result.exit_code == 0
        assert result.stdout == "ok"
        assert result.stderr == ""
        # execution_time_ms (100.0) / 1000.0 = 0.1 seconds
        assert result.execution_time == 0.1
        assert result.created_at == start_time
        assert result.container_name == "docker_123"
        assert result.test_metadata["container_id"] == "docker_123"
        assert result.test_metadata["test_type"] == "unit"

    @pytest.mark.asyncio
    async def test_execute_legacy_with_js(self):
        """Test _execute_legacy with JavaScript (lines 298-302, 337-340)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import TestRequest
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = AsyncMock()
        mock_docker.get_language_container_image.return_value = "node:20"
        mock_docker.create_test_container.return_value = "docker_js_1"
        mock_docker.create_file_in_container.return_value = True
        mock_docker.exec_in_container.return_value = {
            "exit_code": 0,
            "stdout": "tests passed",
            "stderr": "",
        }
        mock_docker.install_tools_in_container = AsyncMock()

        runner = ContainerTestRunner(mock_docker, TestConfig())
        runner.orchestrator = None  # Force legacy path

        request = MagicMock(spec=TestRequest)
        request.language = "javascript"
        request.code = "function add(a, b) { return a + b; }"
        request.test_code = "test('add', () => expect(add(1,2)).toBe(3))"
        request.code_path = None
        request.test_path = None
        request.timeout = 30
        request.security_level = "medium"
        request.container_name = None

        result = await runner._execute_legacy(
            request,
            "test_js_1",
            datetime.now(UTC),
        )
        assert result.success is True
        assert result.test_id == "test_js_1"
        assert result.language == "javascript"
        assert result.exit_code == 0
        assert result.stdout == "tests passed"

    @pytest.mark.asyncio
    async def test_execute_legacy_timeout(self):
        """Test _execute_legacy with timeout (lines 370-390)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import TestRequest
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = AsyncMock()
        mock_docker.get_language_container_image.return_value = "python:3.11"
        mock_docker.create_test_container.return_value = "docker_py_1"
        mock_docker.create_file_in_container.return_value = True
        mock_docker.install_tools_in_container = AsyncMock()
        mock_docker.exec_in_container.side_effect = TimeoutError("test timed out")

        runner = ContainerTestRunner(mock_docker, TestConfig())
        runner.orchestrator = None

        request = MagicMock(spec=TestRequest)
        request.language = "python"
        request.code = "x = 1"
        request.test_code = "assert x == 1"
        request.code_path = None
        request.test_path = None
        request.timeout = 1
        request.security_level = "medium"
        request.container_name = None

        result = await runner._execute_legacy(
            request,
            "test_timeout_1",
            datetime.now(UTC),
        )
        assert result.success is False
        assert result.exit_code == 124
        assert result.test_id == "test_timeout_1"
        assert result.language == "python"

    @pytest.mark.asyncio
    async def test_create_js_package_file(self):
        """Test _create_js_package_file (lines 304-324)."""
        import json

        from config.test_framework_config import TestConfig
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = AsyncMock()
        runner = ContainerTestRunner(mock_docker, TestConfig())

        # Test JavaScript package.json
        await runner._create_js_package_file("docker_123", "javascript")
        mock_docker.create_file_in_container.assert_called_once()
        call_args = mock_docker.create_file_in_container.call_args
        assert call_args[0][0] == "docker_123"
        assert call_args[0][1] == "/app/test/package.json"
        pkg = json.loads(call_args[0][2])
        assert pkg["scripts"]["test"] == "jest"
        assert "jest" in pkg["devDependencies"]
        assert "ts-jest" not in pkg["devDependencies"]

        # Test TypeScript package.json
        mock_docker.reset_mock()
        await runner._create_js_package_file("docker_123", "typescript")
        mock_docker.create_file_in_container.assert_called_once()
        call_args = mock_docker.create_file_in_container.call_args
        pkg_ts = json.loads(call_args[0][2])
        assert "ts-jest" in pkg_ts["scripts"]["test"]
        assert "ts-jest" in pkg_ts["devDependencies"]
        assert "@types/jest" in pkg_ts["devDependencies"]
        assert "typescript" in pkg_ts["devDependencies"]


# ---------------------------------------------------------------------------
# intelligent_test_integration.py - service methods (lines 256-259, 471+)
# ---------------------------------------------------------------------------


class TestIntelligentIntegrationServiceR4:
    """Cover IntelligentTestExecutionService methods."""

    @pytest.mark.asyncio
    async def test_execute_test_not_running(self):
        """Test execute_test when not running (line 209)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import TestRequest
        from services.test_execution.intelligent_test_integration import (
            IntelligentTestExecutionService,
            TestExecutionError,
        )

        service = IntelligentTestExecutionService(MagicMock(), TestConfig())
        service.is_running = False

        request = TestRequest(
            code="print(1)",
            language="python",
            container_name="c1",
        )
        with pytest.raises(TestExecutionError, match="not running") as exc_info:
            await service.execute_test(request)
        assert exc_info.value.error_type == "service_not_running"

    @pytest.mark.asyncio
    async def test_execute_batch_not_running(self):
        """Test execute_batch when not running (line 232)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_integration import (
            IntelligentTestExecutionService,
            TestExecutionError,
        )

        service = IntelligentTestExecutionService(MagicMock(), TestConfig())
        service.is_running = False

        with pytest.raises(TestExecutionError, match="not running") as exc_info:
            await service.execute_batch([])
        assert exc_info.value.error_type == "service_not_running"

    @pytest.mark.asyncio
    async def test_convert_test_request(self):
        """Test _convert_test_request (lines 384-448)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import TestRequest
        from services.test_execution.intelligent_test_integration import (
            IntelligentTestExecutionService,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        service = IntelligentTestExecutionService(MagicMock(), TestConfig())

        request = TestRequest(
            code="print(1)",
            language="python",
            test_code="assert True",
            container_name="c1",
        )
        result = await service._convert_test_request(request)
        assert result.source_code == "print(1)"
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        assert result.container_name == "c1"
        assert result.timeout_seconds == 300  # default
        # test_files should contain the test code
        assert "test_main.py" in result.test_files
        assert result.test_files["test_main.py"] == "assert True"
        # AI settings should use service defaults
        assert result.enable_ai_optimization == service.settings.enable_ai_optimization
        assert result.priority_level == 5
        assert result.resource_weight == 1.0

    @pytest.mark.asyncio
    async def test_convert_test_request_unsupported(self):
        """Test _convert_test_request with unsupported language (line 402-404)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_integration import (
            IntelligentTestExecutionService,
            TestExecutionError,
        )

        service = IntelligentTestExecutionService(MagicMock(), TestConfig())

        # Use a mock since TestRequest validates language
        request = MagicMock()
        request.language = "brainfuck"
        request.code = "x"
        request.test_code = None
        request.timeout = 300
        request.security_level = None
        request.test_type = None
        request.container_name = "c1"

        with pytest.raises(TestExecutionError, match="conversion failed") as exc_info:
            await service._convert_test_request(request)
        assert exc_info.value.error_type == "request_conversion_failure"

    @pytest.mark.asyncio
    async def test_validate_test_request_empty_code(self):
        """Test _validate_test_request with empty code (line 454-455)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_integration import (
            IntelligentTestExecutionRequest,
            IntelligentTestExecutionService,
            TestExecutionError,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        service = IntelligentTestExecutionService(MagicMock(), TestConfig())
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="  ",  # whitespace only
            container_name="c1",
        )
        with pytest.raises(TestExecutionError, match="empty") as exc_info:
            await service._validate_test_request(req)
        assert exc_info.value.error_type == "invalid_request"

    @pytest.mark.asyncio
    async def test_validate_test_request_timeout_too_long(self):
        """Test _validate_test_request with too long timeout (lines 457-461)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_integration import (
            IntelligentTestExecutionRequest,
            IntelligentTestExecutionService,
            TestExecutionError,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        service = IntelligentTestExecutionService(MagicMock(), TestConfig())
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
            timeout_seconds=9999,
        )
        with pytest.raises(TestExecutionError, match="Timeout exceeds") as exc_info:
            await service._validate_test_request(req)
        assert exc_info.value.error_type == "timeout_too_long"

    @pytest.mark.asyncio
    async def test_managed_execution_context(self):
        """Test managed_execution context manager (lines 691-698)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_integration import (
            IntelligentTestExecutionService,
        )

        service = IntelligentTestExecutionService(MagicMock(), TestConfig())
        # Mock start/stop to avoid real orchestrator
        service.orchestrator = AsyncMock()
        service.orchestrator.start = AsyncMock()
        service.orchestrator.stop = AsyncMock()

        async with service.managed_execution() as svc:
            assert svc is service
            assert svc.is_running is True

        assert svc.is_running is False
        service.orchestrator.start.assert_called_once()
        service.orchestrator.stop.assert_called_once()


# ---------------------------------------------------------------------------
# intelligent_test_recommendation_engine.py - from insights (lines 413-627)
# ---------------------------------------------------------------------------


class TestRecommendationEngineFromInsightsR4:
    """Cover _generate_from_test_insights and _generate_from_pattern_analysis."""

    @pytest.mark.asyncio
    async def test_generate_from_test_insights(self):
        """Test _generate_from_test_insights (lines 408-577)."""
        from services.intelligent_test_recommendation_engine import (
            IntelligentRecommendationEngine,
        )

        engine = IntelligentRecommendationEngine()

        # Create a mock TestInsightAnalysis
        mock_insights = MagicMock()
        mock_insights.performance_metrics.slow_tests_count = 5
        mock_insights.performance_metrics.performance_score = 60.0
        mock_insights.reliability_metrics.flaky_tests_count = 3
        mock_insights.reliability_metrics.overall_success_rate = 0.85
        mock_insights.coverage_insights.overall_coverage_score = 70.0
        mock_insights.coverage_insights.coverage_gaps = ["gap1", "gap2"]

        recs = await engine._generate_from_test_insights(mock_insights)
        # slow_tests_count > 0 -> perf rec, flaky > 0 -> reliability rec, coverage < 80 -> coverage rec
        assert len(recs) == 3

        # Verify performance recommendation
        perf_rec = next(r for r in recs if r.recommendation_id == "perf_optimize_slow_tests")
        assert perf_rec.priority.value == "high"
        assert perf_rec.impact_score == 8.0
        assert perf_rec.confidence == 0.9
        assert perf_rec.supporting_evidence["slow_tests_count"] == 5
        assert perf_rec.supporting_evidence["performance_score"] == 60.0
        assert perf_rec.estimated_effort == "medium"

        # Verify reliability recommendation
        rel_rec = next(r for r in recs if r.recommendation_id == "rel_fix_flaky_tests")
        assert rel_rec.priority.value == "critical"
        assert rel_rec.impact_score == 9.0
        assert rel_rec.confidence == 0.95
        assert rel_rec.supporting_evidence["flaky_tests_count"] == 3
        assert rel_rec.supporting_evidence["success_rate"] == 0.85

        # Verify coverage recommendation
        cov_rec = next(r for r in recs if r.recommendation_id == "cov_improve_coverage")
        assert cov_rec.priority.value == "medium"
        assert cov_rec.impact_score == 7.0
        assert cov_rec.confidence == 0.8
        assert cov_rec.supporting_evidence["current_coverage"] == 70.0
        assert cov_rec.supporting_evidence["coverage_gaps"] == 2

    @pytest.mark.asyncio
    async def test_generate_from_test_insights_no_issues(self):
        """Test _generate_from_test_insights with no issues returns empty list."""
        from services.intelligent_test_recommendation_engine import (
            IntelligentRecommendationEngine,
        )

        engine = IntelligentRecommendationEngine()

        mock_insights = MagicMock()
        mock_insights.performance_metrics.slow_tests_count = 0
        mock_insights.reliability_metrics.flaky_tests_count = 0
        mock_insights.coverage_insights.overall_coverage_score = 95.0

        recs = await engine._generate_from_test_insights(mock_insights)
        assert recs == []

    @pytest.mark.asyncio
    async def test_generate_from_pattern_analysis(self):
        """Test _generate_from_pattern_analysis (lines 579-627)."""
        from services.intelligent_test_recommendation_engine import (
            IntelligentRecommendationEngine,
        )

        engine = IntelligentRecommendationEngine()

        mock_analysis = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.pattern_id = "p1"
        mock_pattern.pattern_name = "High Failure Rate"
        mock_pattern.description = "Many tests failing"
        mock_pattern.impact_score = 9.0
        mock_pattern.confidence_score = 0.9
        mock_pattern.pattern_type = "failure"
        mock_pattern.occurrence_count = 10
        mock_pattern.frequency_ratio = 0.5
        mock_pattern.recommendations = ["Fix it", "Test more"]
        mock_analysis.critical_patterns = [mock_pattern]

        recs = await engine._generate_from_pattern_analysis(mock_analysis)
        assert len(recs) == 1
        rec = recs[0]
        assert rec.recommendation_id == "pattern_p1"
        assert rec.title == "Address High Failure Rate"
        assert rec.priority.value == "critical"
        assert rec.impact_score == 9.0
        assert rec.confidence == 0.9
        assert rec.implementation_steps == ["Fix it", "Test more"]
        assert rec.supporting_evidence["pattern_type"] == "failure"
        assert rec.supporting_evidence["occurrence_count"] == 10
        assert rec.supporting_evidence["frequency_ratio"] == 0.5
        assert rec.estimated_effort == "high"

    @pytest.mark.asyncio
    async def test_generate_from_performance_analysis(self):
        """Test _generate_from_performance_analysis (lines 629-677)."""
        from services.intelligent_test_recommendation_engine import (
            IntelligentRecommendationEngine,
        )

        engine = IntelligentRecommendationEngine()

        mock_analysis = MagicMock()
        mock_rec = MagicMock()
        mock_rec.recommendation_id = "r1"
        mock_rec.title = "Optimize tests"
        mock_rec.description = "Make tests faster"
        mock_rec.priority = "high"
        mock_rec.expected_improvement = 30.0
        mock_rec.confidence = 0.85
        mock_rec.implementation_steps = ["step1"]
        mock_rec.estimated_effort = "medium"
        mock_rec.dependencies = []
        mock_rec.supporting_evidence = {}
        mock_rec.risks = []
        mock_analysis.optimization_recommendations = [mock_rec]

        recs = await engine._generate_from_performance_analysis(mock_analysis)
        assert len(recs) == 1
        rec = recs[0]
        assert rec.recommendation_id == "perf_opt_r1"
        assert rec.title == "Optimize tests"
        assert rec.description == "Make tests faster"
        assert rec.priority.value == "high"
        # impact_score = min(10.0, 30.0/10) = 3.0
        assert rec.impact_score == 3.0
        assert rec.confidence == 0.85
        assert rec.implementation_steps == ["step1"]
        assert rec.estimated_effort == "medium"
        assert rec.expected_benefit == "Improve performance by 30.0%"

    @pytest.mark.asyncio
    async def test_generate_from_quality_analysis(self):
        """Test _generate_from_quality_analysis (lines 679-727)."""
        from services.intelligent_test_recommendation_engine import (
            IntelligentRecommendationEngine,
        )

        engine = IntelligentRecommendationEngine()

        mock_analysis = MagicMock()
        mock_rec = MagicMock()
        mock_rec.recommendation_id = "q1"
        mock_rec.title = "Improve quality"
        mock_rec.description = "Better code"
        mock_rec.priority = "critical"
        mock_rec.expected_improvement = 25.0
        mock_rec.confidence = 0.9
        mock_rec.implementation_steps = ["step1"]
        mock_rec.effort_estimate = "low"
        mock_rec.prerequisites = []
        mock_rec.tools_required = []
        mock_rec.supporting_evidence = {}
        mock_rec.affected_components = ["code"]
        mock_rec.risks = []
        mock_analysis.recommendations = [mock_rec]

        recs = await engine._generate_from_quality_analysis(mock_analysis)
        assert len(recs) == 1
        rec = recs[0]
        assert rec.recommendation_id == "qual_q1"
        assert rec.title == "Improve quality"
        assert rec.priority.value == "critical"
        # impact_score = min(10.0, 25.0/10) = 2.5
        assert rec.impact_score == 2.5
        assert rec.confidence == 0.9
        assert rec.estimated_effort == "low"
        assert rec.expected_benefit == "Improve quality by 25.0%"
        assert rec.affected_components == ["code"]


# ---------------------------------------------------------------------------
# core_service.py - LLM analysis path
# ---------------------------------------------------------------------------


class TestCoreServiceLLMR4:
    """Cover _analyze_with_llm real path."""

    @pytest.mark.asyncio
    async def test_analyze_with_llm_invalid_json(self):
        """Free-form agent returns non-JSON: _analyze_with_llm returns None."""
        from services.ai_test_insights.config import AITestInsightsConfig
        from services.ai_test_insights.core_service import AITestInsightsService

        config = AITestInsightsConfig()
        config.use_ml_backend = True
        config.enable_ai_features = True
        service = AITestInsightsService(config=config)

        with patch(
            "services.ai_test_insights.llm_agent.AITestInsightsAgent.execute",
            new=AsyncMock(return_value="not json at all"),
        ):
            result = await service._analyze_with_llm("content", "prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_with_llm_runtime_error(self):
        """Runtime error from the LLM agent is swallowed and returns None."""
        from services.ai_test_insights.config import AITestInsightsConfig
        from services.ai_test_insights.core_service import AITestInsightsService

        config = AITestInsightsConfig()
        config.use_ml_backend = True
        config.enable_ai_features = True
        service = AITestInsightsService(config=config)

        with patch(
            "services.ai_test_insights.llm_agent.AITestInsightsAgent.execute",
            new=AsyncMock(side_effect=RuntimeError("LLM error")),
        ):
            result = await service._analyze_with_llm("content", "prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_with_llm_valid_json(self):
        """When the LLM returns a JSON object, it is parsed and returned."""
        from services.ai_test_insights.config import AITestInsightsConfig
        from services.ai_test_insights.core_service import AITestInsightsService

        config = AITestInsightsConfig()
        config.use_ml_backend = True
        config.enable_ai_features = True
        service = AITestInsightsService(config=config)

        with patch(
            "services.ai_test_insights.llm_agent.AITestInsightsAgent.execute",
            new=AsyncMock(return_value='{"insight": "ok"}'),
        ):
            result = await service._analyze_with_llm("content", "prompt")
            assert result == {"insight": "ok"}

    @pytest.mark.asyncio
    async def test_analyze_with_llm_disabled(self):
        """Test _analyze_with_llm returns None when ML backend is disabled."""
        from services.ai_test_insights.config import AITestInsightsConfig
        from services.ai_test_insights.core_service import AITestInsightsService

        config = AITestInsightsConfig()
        config.use_ml_backend = False
        config.enable_ai_features = False
        service = AITestInsightsService(config=config)

        result = await service._analyze_with_llm("content", "prompt")
        assert result is None
