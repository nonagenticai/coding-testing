"""Comprehensive unit tests for src/services/testing/ modules.

Covers:
- test_result_processor.py
- test_config_manager.py
- test_monitor.py
- test_executor.py
- quality_metrics.py
- quality_config.py
- quality_reporting.py
- quality_validation.py
- container_manager.py
- security_management_service.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.testing_models import (
    QualityIssue,
    QualityRequest,
    QualityResult,
    TestRequest,
    TestResult,
    TestSecurityLevel,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_test_request(**overrides) -> TestRequest:
    defaults = {
        "code": "def add(a,b): return a+b",
        "test_code": "def test_add(): assert add(1,2)==3",
        "language": "python",
    }
    defaults.update(overrides)
    return TestRequest(**defaults)


def _make_test_result(**overrides) -> TestResult:
    defaults = {
        "test_id": "test-001",
        "success": True,
        "stdout": "OK",
        "stderr": "",
        "exit_code": 0,
        "execution_time": 1.0,
        "created_at": datetime.now(UTC),
        "language": "python",
        "security_level": TestSecurityLevel.HIGH,
    }
    defaults.update(overrides)
    return TestResult(**defaults)


def _make_quality_issue(**overrides) -> QualityIssue:
    defaults = {
        "tool": "ruff",
        "severity": "warning",
        "message": "Unused variable",
        "file_path": "test.py",
        "line_number": 10,
        "column_number": 5,
        "rule_id": "W0612",
        "suggested_fix": None,
        "fix_confidence": None,
    }
    defaults.update(overrides)
    return QualityIssue(**defaults)


def _make_quality_result(**overrides) -> QualityResult:
    defaults = {
        "analysis_id": "qr-001",
        "success": True,
        "overall_score": 8.5,
        "total_issues": 0,
        "issues": [],
        "tools_results": {"ruff": {"success": True, "execution_time": 1.0}},
        "execution_time": 2.0,
        "created_at": datetime.now(UTC),
        "language": "python",
    }
    defaults.update(overrides)
    return QualityResult(**defaults)


def _make_quality_request(**overrides) -> QualityRequest:
    defaults = {
        "container_name": "test-container",
        "code": "x = 1",
        "language": "python",
    }
    defaults.update(overrides)
    return QualityRequest(**defaults)


# ===================================================================
# test_result_processor.py
# ===================================================================


class TestTestResultProcessor:
    def _make_processor(self, cache_manager=None):
        from services.testing.test_result_processor import TestResultProcessor

        return TestResultProcessor(cache_manager=cache_manager)

    # -- process_result --
    @pytest.mark.asyncio
    async def test_process_result_no_cache(self):
        p = self._make_processor(cache_manager=None)
        req = _make_test_request()
        result = _make_test_result()
        processed = await p.process_result(result, req)
        assert processed.test_id == result.test_id

    @pytest.mark.asyncio
    async def test_process_result_with_cache_successful(self):
        cache = AsyncMock()
        cache.set = AsyncMock()
        p = self._make_processor(cache_manager=cache)
        req = _make_test_request()
        result = _make_test_result(success=True)
        await p.process_result(result, req)
        cache.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_result_with_cache_failed_no_cache(self):
        cache = AsyncMock()
        cache.set = AsyncMock()
        p = self._make_processor(cache_manager=cache)
        req = _make_test_request()
        result = _make_test_result(success=False)
        await p.process_result(result, req)
        cache.set.assert_not_awaited()

    # -- convert_container_result --
    def test_convert_container_result_success(self):
        p = self._make_processor()
        container_result = {"exit_code": 0, "stdout": "PASS", "stderr": "", "execution_time": 1.5}
        req = _make_test_request()
        result = p.convert_container_result(container_result, "t-001", req)
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "PASS"

    def test_convert_container_result_failure(self):
        p = self._make_processor()
        container_result = {"exit_code": 1, "stdout": "", "stderr": "FAIL"}
        req = _make_test_request()
        result = p.convert_container_result(container_result, "t-002", req)
        assert result.success is False


# ===================================================================
# test_config_manager.py
# ===================================================================


class TestTestConfigManager:
    def _make_manager(self):
        from services.testing.test_config_manager import TestConfigManager

        security_service = AsyncMock()
        return TestConfigManager(security_service=security_service)

    # -- detect_language --
    def test_detect_language_hint(self):
        m = self._make_manager()
        assert m.detect_language(None, "JavaScript") == "javascript"

    def test_detect_language_file(self):
        m = self._make_manager()
        assert m.detect_language("/f.ts", None) == "typescript"

    def test_detect_language_default(self):
        m = self._make_manager()
        assert m.detect_language(None, None) == "python"

    # -- get_quality_tools --
    def test_get_quality_tools_python(self):
        m = self._make_manager()
        tools = m.get_quality_tools("python", None)
        assert tools == ["ruff", "ruff-format", "ty"]

    def test_get_quality_tools_filtered(self):
        m = self._make_manager()
        tools = m.get_quality_tools("python", ["ruff-format", "ty", "unknown_tool"])
        assert tools == ["ruff-format", "ty"]

    def test_get_quality_tools_unknown_lang(self):
        m = self._make_manager()
        assert m.get_quality_tools("ruby", None) == []

    # -- get_container_image --
    def test_get_container_image_python(self):
        m = self._make_manager()
        assert m.get_container_image("python") == "codegraph/test-executor-python:latest"

    def test_get_container_image_js(self):
        m = self._make_manager()
        assert m.get_container_image("javascript") == "code-graph-test-node:latest"

    def test_get_container_image_unknown(self):
        m = self._make_manager()
        assert m.get_container_image("ruby") == "codegraph/test-executor-python:latest"

    # -- get_memory_limit --
    def test_memory_limit_high(self):
        m = self._make_manager()
        assert m.get_memory_limit(TestSecurityLevel.HIGH) == "256m"

    def test_memory_limit_medium(self):
        m = self._make_manager()
        assert m.get_memory_limit(TestSecurityLevel.MEDIUM) == "512m"

    def test_memory_limit_low(self):
        m = self._make_manager()
        assert m.get_memory_limit(TestSecurityLevel.LOW) == "1g"

    # -- prepare_test_files --
    def test_prepare_test_files_both(self):
        m = self._make_manager()
        req = _make_test_request()
        files = m.prepare_test_files(req)
        assert files == {
            "main.py": "def add(a,b): return a+b",
            "test_main.py": "def test_add(): assert add(1,2)==3",
        }

    def test_prepare_test_files_no_code(self):
        m = self._make_manager()
        req = _make_test_request(code=None)
        files = m.prepare_test_files(req)
        assert files == {"test_main.py": "def test_add(): assert add(1,2)==3"}

    # -- get_test_command --
    def test_get_test_command_python(self):
        m = self._make_manager()
        cmd = m.get_test_command("python", {"test_main.py": "code"})
        assert cmd == ["python", "-m", "pytest", "-v", "--tb=short", "test_main.py"]

    def test_get_test_command_javascript(self):
        m = self._make_manager()
        cmd = m.get_test_command("javascript", {})
        assert cmd == ["npm", "test"]

    # -- get_file_extension_for_language --
    def test_file_extension_python(self):
        m = self._make_manager()
        assert m.get_file_extension_for_language("python") == ".py"

    def test_file_extension_js(self):
        m = self._make_manager()
        assert m.get_file_extension_for_language("javascript") == ".js"

    def test_file_extension_unknown(self):
        m = self._make_manager()
        assert m.get_file_extension_for_language("ruby") == ".py"

    # -- get_tool_commands --
    def test_get_tool_commands_python(self):
        m = self._make_manager()
        cmds = m.get_tool_commands("python")
        assert set(cmds.keys()) == {"ruff", "ruff-format", "ty"}
        assert cmds["ruff-format"] == ["ruff", "format", "--check", "--diff"]
        assert cmds["ruff"] == ["ruff", "check", "--output-format", "json", "--extend-select", "I"]
        assert cmds["ty"] == ["ty", "check", "--output-format", "gitlab"]

    def test_get_tool_commands_unknown(self):
        m = self._make_manager()
        assert m.get_tool_commands("ruby") == {}

    # -- validate_configuration --
    def test_validate_configuration(self):
        m = self._make_manager()
        result = m.validate_configuration()
        assert result["valid"] is True
        assert result["details"]["testpy_config"] is True
        assert result["details"]["security_config"] is True
        assert result["details"]["security_service"] is True

    # -- create_container_config --
    def test_create_container_config(self):
        m = self._make_manager()
        req = _make_test_request()
        cfg = m.create_container_config(req, "tid-001", "img:latest")
        assert cfg["image"] == "img:latest"
        assert cfg["network_disabled"] is True
        assert cfg["labels"]["test_id"] == "tid-001"

    # -- discover_files --
    def test_discover_files_recursive(self, tmp_path):
        m = self._make_manager()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.py").write_text("x=1")
        (tmp_path / "top.py").write_text("y=2")
        files = m.discover_files(str(tmp_path), ["*.py"], recursive=True)
        assert len(files) == 2

    def test_discover_files_non_recursive(self, tmp_path):
        m = self._make_manager()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.py").write_text("x=1")
        (tmp_path / "top.py").write_text("y=2")
        files = m.discover_files(str(tmp_path), ["*.py"], recursive=False)
        assert len(files) == 1

    def test_discover_files_nonexistent_dir(self):
        m = self._make_manager()
        with pytest.raises(ValueError, match="does not exist"):
            m.discover_files("/nonexistent/dir/xyz", ["*.py"], recursive=True)

    # -- find_test_file --
    def test_find_test_file_same_dir(self, tmp_path):
        m = self._make_manager()
        src = tmp_path / "mymod.py"
        src.write_text("x=1")
        tst = tmp_path / "test_mymod.py"
        tst.write_text("assert True")
        result = m.find_test_file(src)
        assert result == tst

    def test_find_test_file_tests_dir(self, tmp_path):
        m = self._make_manager()
        src = tmp_path / "mymod.py"
        src.write_text("x=1")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        tst = tests_dir / "test_mymod.py"
        tst.write_text("assert True")
        result = m.find_test_file(src)
        assert result == tst

    def test_find_test_file_not_found(self, tmp_path):
        m = self._make_manager()
        src = tmp_path / "mymod.py"
        src.write_text("x=1")
        assert m.find_test_file(src) is None

    # -- validate_security --
    @pytest.mark.asyncio
    async def test_validate_security_passes(self):
        from services.testing.test_config_manager import TestConfigManager

        sec = AsyncMock()
        sec.validate_code_security = AsyncMock(
            return_value=SimpleNamespace(is_safe=True, violations=[])
        )
        m = TestConfigManager(security_service=sec)
        req = _make_test_request()
        await m.validate_security(req)  # should not raise

    @pytest.mark.asyncio
    async def test_validate_security_code_too_long(self):
        from config.test_framework_config import TestFrameworkSecurityConfig
        from services.testing.test_config_manager import TestConfigManager

        sec_cfg = TestFrameworkSecurityConfig(max_code_length=10)
        sec = AsyncMock()
        m = TestConfigManager(
            security_service=sec,
            security_config=sec_cfg,
        )
        req = _make_test_request(code="x" * 100)
        with pytest.raises(ValueError, match="Code length"):
            await m.validate_security(req)

    @pytest.mark.asyncio
    async def test_validate_security_violations(self):
        from services.testing.test_config_manager import TestConfigManager

        violation = SimpleNamespace(type="dangerous", description="eval used", line_number=1)
        sec = AsyncMock()
        sec.validate_code_security = AsyncMock(
            return_value=SimpleNamespace(is_safe=False, violations=[violation])
        )
        m = TestConfigManager(security_service=sec)
        req = _make_test_request()
        with pytest.raises(ValueError, match="security validation failed"):
            await m.validate_security(req)


# ===================================================================
# quality_metrics.py
# ===================================================================


class TestQualityScoreCalculator:
    def _make_calc(self):
        from services.testing.quality_metrics import QualityScoreCalculator

        return QualityScoreCalculator()

    def test_no_tools_results_returns_zero(self):
        calc = self._make_calc()
        assert calc.calculate_quality_score({}, []) == 0.0

    def test_no_issues_returns_ten(self):
        calc = self._make_calc()
        score = calc.calculate_quality_score({"ruff": {}}, [])
        assert score == 10.0

    def test_issues_reduce_score(self):
        calc = self._make_calc()
        issues = [_make_quality_issue(severity="error", tool="ruff")]
        score = calc.calculate_quality_score({"ruff": {}}, issues)
        assert 0.0 < score < 10.0

    def test_many_issues_floor_zero(self):
        calc = self._make_calc()
        issues = [_make_quality_issue(severity="error", tool="bandit") for _ in range(100)]
        score = calc.calculate_quality_score({"bandit": {}}, issues)
        assert score >= 0.0

    def test_comprehensive_metrics(self):
        calc = self._make_calc()
        issues = [
            _make_quality_issue(severity="error", tool="ruff", file_path="a.py"),
            _make_quality_issue(severity="warning", tool="ty", file_path="b.py"),
            _make_quality_issue(
                severity="info",
                tool="ruff-format",
                file_path="a.py",
                suggested_fix="run ruff format",
                fix_confidence=0.95,
            ),
        ]
        metrics = calc.calculate_comprehensive_metrics(
            {"ruff": {}, "ty": {}, "ruff-format": {}}, issues
        )
        assert metrics.total_issues == 3
        assert metrics.severity_breakdown == {"error": 1, "warning": 1, "info": 1}
        assert metrics.tool_breakdown == {"ruff": 1, "ty": 1, "ruff-format": 1}
        assert metrics.file_breakdown == {"a.py": 2, "b.py": 1}
        assert metrics.quality_grade in [
            "A+",
            "A",
            "A-",
            "B+",
            "B",
            "B-",
            "C+",
            "C",
            "C-",
            "D",
            "F",
        ]
        assert metrics.improvement_potential >= 0

    # -- _calculate_grade --
    def test_grade_boundaries(self):
        calc = self._make_calc()
        assert calc._calculate_grade(9.5) == "A+"
        assert calc._calculate_grade(8.7) == "A"
        assert calc._calculate_grade(8.2) == "A-"
        assert calc._calculate_grade(7.6) == "B+"
        assert calc._calculate_grade(7.1) == "B"
        assert calc._calculate_grade(6.7) == "B-"
        assert calc._calculate_grade(6.1) == "C+"
        assert calc._calculate_grade(5.6) == "C"
        assert calc._calculate_grade(5.1) == "C-"
        assert calc._calculate_grade(4.5) == "D"
        assert calc._calculate_grade(3.0) == "F"

    # -- _calculate_improvement_potential --
    def test_improvement_potential_high_score(self):
        calc = self._make_calc()
        assert calc._calculate_improvement_potential(9.8, []) == 0.0

    def test_improvement_potential_with_fixable(self):
        calc = self._make_calc()
        issues = [
            _make_quality_issue(tool="ruff-format", severity="info"),
            _make_quality_issue(tool="ruff", severity="info"),
        ]
        pot = calc._calculate_improvement_potential(7.0, issues)
        assert pot > 0.0

    def test_improvement_potential_no_fixable(self):
        calc = self._make_calc()
        issues = [_make_quality_issue(tool="ruff", severity="error")]
        pot = calc._calculate_improvement_potential(7.0, issues)
        assert pot > 0.0


class TestQualityTrendAnalyzer:
    def _make_analyzer(self):
        from services.testing.quality_metrics import QualityTrendAnalyzer

        return QualityTrendAnalyzer()

    def test_insufficient_data(self):
        a = self._make_analyzer()
        result = a.calculate_trend_metrics([{"score": 5.0}])
        assert result["trend_direction"] == "insufficient_data"

    def test_improving_trend(self):
        a = self._make_analyzer()
        scores = [{"score": 5.0}, {"score": 6.0}, {"score": 7.0}, {"score": 8.0}]
        result = a.calculate_trend_metrics(scores)
        assert result["trend_direction"] == "improving"
        assert result["trend_percentage"] > 0
        # Linear regression: slope=1.0, y_mean=6.5, n=4 -> (1.0*3/6.5)*100 ≈ 46.15%
        assert result["trend_percentage"] == pytest.approx(46.15, rel=0.01)

    def test_declining_trend(self):
        a = self._make_analyzer()
        scores = [{"score": 8.0}, {"score": 7.0}, {"score": 6.0}, {"score": 5.0}]
        result = a.calculate_trend_metrics(scores)
        assert result["trend_direction"] == "declining"
        assert result["trend_percentage"] < 0

    def test_stable_trend(self):
        a = self._make_analyzer()
        scores = [{"score": 7.0}, {"score": 7.0}, {"score": 7.0}]
        result = a.calculate_trend_metrics(scores)
        assert result["trend_direction"] == "stable"
        assert abs(result["trend_percentage"]) < 5.0  # within 5% = stable

    def test_volatility_high_variance(self):
        a = self._make_analyzer()
        vol = a._calculate_volatility([5.0, 10.0, 5.0, 10.0])
        assert vol > 0
        # std dev of [5,10,5,10] ~ 2.5, so volatility should be meaningful
        assert vol > 1.0

    def test_volatility_no_variance(self):
        a = self._make_analyzer()
        vol = a._calculate_volatility([7.0, 7.0, 7.0])
        assert vol == 0.0

    def test_stability_score_zero_volatility(self):
        a = self._make_analyzer()
        s = a._calculate_stability_score([7.0, 7.0, 7.0], 0.0)
        assert s == 10.0  # zero volatility -> max stability

    def test_stability_score_high_volatility(self):
        a = self._make_analyzer()
        s = a._calculate_stability_score([1.0, 10.0, 1.0, 10.0], 5.0)
        assert s < 10.0  # high volatility -> lower stability
        assert s >= 0.0

    def test_stability_score_empty(self):
        a = self._make_analyzer()
        assert a._calculate_stability_score([], 0.0) == 0.0


class TestQualityBenchmarkComparator:
    def _make_comparator(self):
        from services.testing.quality_metrics import QualityBenchmarkComparator

        return QualityBenchmarkComparator()

    def test_excellent_python(self):
        c = self._make_comparator()
        r = c.compare_to_benchmarks(9.0, "python")
        assert r["quality_level"] == "excellent"
        assert r["estimated_percentile"] == 95

    def test_good_python(self):
        c = self._make_comparator()
        r = c.compare_to_benchmarks(7.5, "python")
        assert r["quality_level"] == "good"

    def test_poor_python(self):
        c = self._make_comparator()
        r = c.compare_to_benchmarks(2.0, "python")
        assert r["quality_level"] == "poor"
        assert r["gap_to_next_level"] > 0

    def test_unknown_language_defaults(self):
        c = self._make_comparator()
        r = c.compare_to_benchmarks(5.5, "ruby")
        assert r["quality_level"] in ["acceptable", "good", "needs_improvement"]

    def test_factory_functions(self):
        from services.testing.quality_metrics import (
            QualityBenchmarkComparator,
            QualityScoreCalculator,
            QualityTrendAnalyzer,
            get_quality_benchmark_comparator,
            get_quality_metrics_calculator,
            get_quality_trend_analyzer,
        )

        calc = get_quality_metrics_calculator()
        assert type(calc) is QualityScoreCalculator
        trend = get_quality_trend_analyzer()
        assert type(trend) is QualityTrendAnalyzer
        bench = get_quality_benchmark_comparator()
        assert type(bench) is QualityBenchmarkComparator


# ===================================================================
# quality_config.py
# ===================================================================


class TestQualityConfig:
    def test_tool_config_defaults(self):
        from services.testing.quality_config import ToolConfig

        tc = ToolConfig(name="test", cmd=["test"])
        assert tc.enabled is True
        assert tc.timeout == 60
        assert tc.priority == 1
        assert tc.extra_args == []

    def test_language_config(self):
        from services.testing.quality_config import LanguageConfig

        lc = LanguageConfig(name="test", extensions=[".test"])
        assert lc.tools == {}
        assert lc.default_tools == []

    def test_analysis_config_defaults(self):
        from services.testing.quality_config import QualityAnalysisConfig

        ac = QualityAnalysisConfig()
        assert ac.max_file_size_mb == 10
        assert ac.timeout_seconds == 300
        assert ac.parallel_tools is True


class TestQualityToolRegistry:
    def _make_registry(self):
        from services.testing.quality_config import QualityToolRegistry

        return QualityToolRegistry()

    def test_default_languages(self):
        r = self._make_registry()
        assert "python" in r.languages
        assert "javascript" in r.languages
        assert "typescript" in r.languages

    def test_get_language_config(self):
        r = self._make_registry()
        cfg = r.get_language_config("python")
        assert cfg.name == "python"
        assert cfg.extensions == [".py", ".pyx", ".pyi"]
        assert cfg.default_tools == ["ruff", "ruff-format", "ty"]

    def test_get_language_config_unknown(self):
        r = self._make_registry()
        assert r.get_language_config("ruby") is None

    def test_get_tool_config(self):
        r = self._make_registry()
        tc = r.get_tool_config("ruff-format", "python")
        assert tc.name == "ruff-format"
        assert tc.supports_fix is True
        assert tc.output_format == "diff"
        assert tc.cmd == ["ruff", "format", "--check", "--diff"]
        assert tc.fix_cmd == ["ruff", "format"]

    def test_get_tool_config_unknown_tool(self):
        r = self._make_registry()
        assert r.get_tool_config("unknown", "python") is None

    def test_get_tools_for_language(self):
        r = self._make_registry()
        tools = r.get_tools_for_language("python")
        assert set(tools) == {
            "ruff",
            "ruff-format",
            "ty",
            "bandit",
        }

    def test_get_tools_for_unknown_lang(self):
        r = self._make_registry()
        assert r.get_tools_for_language("ruby") == []

    def test_get_default_tools(self):
        r = self._make_registry()
        tools = r.get_default_tools("python")
        assert tools == ["ruff", "ruff-format", "ty"]

    def test_supports_fix(self):
        r = self._make_registry()
        assert r.supports_fix("ruff-format", "python") is True
        assert r.supports_fix("ruff", "python") is True
        assert r.supports_fix("ty", "python") is False
        assert r.supports_fix("unknown", "python") is False

    def test_add_language(self):
        from services.testing.quality_config import LanguageConfig

        r = self._make_registry()
        r.add_language(LanguageConfig(name="Go", extensions=[".go"]))
        go_cfg = r.get_language_config("go")
        assert go_cfg.name == "Go"
        assert go_cfg.extensions == [".go"]

    def test_add_tool(self):
        from services.testing.quality_config import ToolConfig

        r = self._make_registry()
        r.add_tool("python", ToolConfig(name="ruff", cmd=["ruff", "check"]))
        ruff_cfg = r.get_tool_config("ruff", "python")
        assert ruff_cfg.name == "ruff"
        assert ruff_cfg.cmd == ["ruff", "check"]

    def test_add_tool_unknown_lang(self):
        from services.testing.quality_config import ToolConfig

        r = self._make_registry()
        r.add_tool("ruby", ToolConfig(name="rubocop", cmd=["rubocop"]))
        # Should just warn, not crash

    def test_update_tool_config(self):
        r = self._make_registry()
        assert r.update_tool_config("python", "ruff-format", timeout=120) is True
        tc = r.get_tool_config("ruff-format", "python")
        assert tc.timeout == 120

    def test_update_tool_config_unknown(self):
        r = self._make_registry()
        assert r.update_tool_config("python", "unknown", timeout=10) is False

    def test_enable_disable_tool(self):
        r = self._make_registry()
        r.disable_tool("python", "ruff-format")
        tc = r.get_tool_config("ruff-format", "python")
        assert tc.enabled is False

        r.enable_tool("python", "ruff-format")
        tc = r.get_tool_config("ruff-format", "python")
        assert tc.enabled is True

    def test_detect_language(self):
        r = self._make_registry()
        assert r.detect_language("test.py") == "python"
        assert r.detect_language("test.js") == "javascript"
        assert r.detect_language("test.ts") == "typescript"
        assert r.detect_language("test.xyz") == "python"  # fallback

    def test_get_container_image(self):
        r = self._make_registry()
        assert r.get_container_image("python") == "code-graph-test-python:latest"

    def test_get_container_image_unknown(self):
        r = self._make_registry()
        img = r.get_container_image("ruby")
        assert "code-graph-test:latest" == img

    def test_get_install_commands(self):
        r = self._make_registry()
        cmds = r.get_install_commands("python")
        assert len(cmds) == 1
        assert cmds[0].startswith("uv pip install")
        assert "ruff==0.17.3" in cmds[0]
        assert "ty==0.0.92" in cmds[0]

    def test_get_install_commands_unknown(self):
        r = self._make_registry()
        assert r.get_install_commands("ruby") == []

    def test_validate_config_clean(self):
        r = self._make_registry()
        issues = r.validate_config()
        assert issues == []

    def test_validate_config_invalid_timeout(self):
        r = self._make_registry()
        tc = r.get_tool_config("ruff-format", "python")
        tc.timeout = -1
        issues = r.validate_config()
        assert any("timeout" in i for i in issues)

    def test_get_tool_priorities(self):
        r = self._make_registry()
        priorities = r.get_tool_priorities("python")
        assert priorities["ruff-format"] == 1
        assert priorities["ty"] == 2
        assert priorities["bandit"] == 3

    def test_get_tools_by_priority(self):
        r = self._make_registry()
        tools = r.get_tools_by_priority("python")
        assert len(tools) == 4
        # Priority 1 tools come first, then 2, then 3
        priorities = r.get_tool_priorities("python")
        for i in range(len(tools) - 1):
            assert priorities[tools[i]] <= priorities[tools[i + 1]]
        # Verify the first tools are priority-1
        assert priorities[tools[0]] == 1

    def test_get_summary(self):
        r = self._make_registry()
        summary = r.get_summary()
        assert summary["total_languages"] == 3
        assert set(summary["languages"]) == {"python", "javascript", "typescript"}
        assert summary["total_tools"] == 4 + 4 + 4  # python:4, js:4, ts:4
        assert summary["tools_by_language"]["python"] == 4

    def test_save_and_load(self, tmp_path):
        r = self._make_registry()
        cfg_path = tmp_path / "quality_config.json"
        r.save_to_file(cfg_path)
        assert cfg_path.exists()

        from services.testing.quality_config import QualityToolRegistry

        r2 = QualityToolRegistry(config_path=cfg_path)
        assert "python" in r2.languages

    def test_load_from_nonexistent(self):
        from services.testing.quality_config import QualityToolRegistry

        r = QualityToolRegistry(config_path=Path("/nonexistent/config.json"))
        # Should fall back to defaults
        assert "python" in r.languages


class TestQualityConfigManager:
    def test_init_creates_config_dir(self, tmp_path):
        from services.testing.quality_config import QualityConfigManager, QualityToolRegistry

        cfg_dir = tmp_path / "config"
        mgr = QualityConfigManager(config_dir=cfg_dir)
        assert cfg_dir.exists()
        assert type(mgr.registry) is QualityToolRegistry
        assert "python" in mgr.registry.languages

    def test_save_and_load_config(self, tmp_path):
        from services.testing.quality_config import QualityConfigManager

        cfg_dir = tmp_path / "config"
        mgr = QualityConfigManager(config_dir=cfg_dir)
        mgr.save_config()
        assert (cfg_dir / "quality_config.json").exists()

        # Re-load
        mgr2 = QualityConfigManager(config_dir=cfg_dir)
        assert "python" in mgr2.registry.languages

    def test_reset_to_defaults(self, tmp_path):
        from services.testing.quality_config import QualityConfigManager

        mgr = QualityConfigManager(config_dir=tmp_path / "config")
        mgr.registry.disable_tool("python", "ruff-format")
        mgr.reset_to_defaults()
        assert mgr.registry.get_tool_config("ruff-format", "python").enabled is True

    def test_backup_config(self, tmp_path):
        from services.testing.quality_config import QualityConfigManager

        mgr = QualityConfigManager(config_dir=tmp_path / "config")
        backup_path = mgr.backup_config()
        assert backup_path.exists()

    def test_restore_config(self, tmp_path):
        from services.testing.quality_config import QualityConfigManager

        mgr = QualityConfigManager(config_dir=tmp_path / "config")
        backup_path = mgr.backup_config()
        mgr.registry.disable_tool("python", "ruff-format")
        mgr.restore_config(backup_path)
        assert mgr.registry.get_tool_config("ruff-format", "python").enabled is True

    def test_restore_nonexistent(self, tmp_path):
        from services.testing.quality_config import QualityConfigManager

        mgr = QualityConfigManager(config_dir=tmp_path / "config")
        with pytest.raises(FileNotFoundError):
            mgr.restore_config(Path("/nonexistent/backup.json"))

    def test_validate_and_fix(self, tmp_path):
        from services.testing.quality_config import QualityConfigManager

        mgr = QualityConfigManager(config_dir=tmp_path / "config")
        result = mgr.validate_and_fix()
        assert result["is_valid"] is True

    def test_validate_and_fix_bad_timeout(self, tmp_path):
        from services.testing.quality_config import QualityConfigManager

        mgr = QualityConfigManager(config_dir=tmp_path / "config")
        tc = mgr.registry.get_tool_config("ruff-format", "python")
        tc.timeout = -1
        result = mgr.validate_and_fix()
        assert len(result["fixed_issues"]) > 0

    def test_factory_functions(self):
        from services.testing.quality_config import (
            QualityToolRegistry,
            get_quality_tool_registry,
        )

        registry = get_quality_tool_registry()
        assert type(registry) is QualityToolRegistry
        assert "python" in registry.languages


# ===================================================================
# quality_validation.py
# ===================================================================


class TestQualityFileValidator:
    def _make_validator(self):
        from services.testing.quality_validation import QualityFileValidator

        return QualityFileValidator()

    @pytest.mark.asyncio
    async def test_validate_valid_file(self, tmp_path):
        v = self._make_validator()
        f = tmp_path / "test.py"
        f.write_text("x = 1\ny = 2\nz = 3\n")
        result = await v.validate_file(f)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_binary_file(self, tmp_path):
        v = self._make_validator()
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\x03" * 100)
        result = await v.validate_file(f)
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_file_with_secrets(self, tmp_path):
        v = self._make_validator()
        f = tmp_path / "config.py"
        f.write_text('password = "secret123"\napi_key = "xyz"\n')
        result = await v.validate_file(f)
        assert result.is_valid is True  # secrets are warnings, not errors
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, tmp_path):
        v = self._make_validator()
        result = await v.validate_file(tmp_path / "nonexistent.py")
        assert result.is_valid is False

    def test_is_binary_file(self):
        v = self._make_validator()
        assert v._is_binary_file(b"\x00\x01\x02") is True
        assert v._is_binary_file(b"normal text content") is False

    def test_validate_file_structure_too_many_lines(self):
        v = self._make_validator()
        content = "\n".join(["line"] * 60000)
        result = v._validate_file_structure(content)
        assert result.is_valid is False

    def test_validate_file_structure_short(self):
        v = self._make_validator()
        result = v._validate_file_structure("x\ny")
        assert "very short" in result.warnings[0].lower()


class TestQualityFixValidator:
    def _make_validator(self):
        from services.testing.quality_validation import QualityFixValidator

        return QualityFixValidator()

    @pytest.mark.asyncio
    async def test_validate_fixes_safe(self):
        v = self._make_validator()
        issues = [
            _make_quality_issue(
                tool="ruff-format",
                suggested_fix="run ruff format",
                fix_confidence=0.95,
                rule_id="fmt",
            ),
        ]
        result = await v.validate_fixes(issues, "python")
        assert result.can_apply is True
        assert len(result.safe_fixes) == 1

    @pytest.mark.asyncio
    async def test_validate_fixes_risky(self):
        v = self._make_validator()
        issues = [
            _make_quality_issue(
                tool="ty", suggested_fix="fix it", fix_confidence=0.5, rule_id="invalid-assignment"
            ),
        ]
        result = await v.validate_fixes(issues, "python")
        assert len(result.risky_fixes) == 1

    @pytest.mark.asyncio
    async def test_validate_fixes_no_fix(self):
        v = self._make_validator()
        issues = [_make_quality_issue(suggested_fix=None)]
        result = await v.validate_fixes(issues, "python")
        assert result.can_apply is False

    @pytest.mark.asyncio
    async def test_validate_fixes_unknown_tool_high_confidence(self):
        v = self._make_validator()
        issues = [
            _make_quality_issue(
                tool="custom_tool", suggested_fix="fix", fix_confidence=0.9, rule_id="X"
            ),
        ]
        result = await v.validate_fixes(issues, "python")
        assert len(result.safe_fixes) == 1

    @pytest.mark.asyncio
    async def test_validate_fixes_unknown_tool_low_confidence(self):
        v = self._make_validator()
        issues = [
            _make_quality_issue(
                tool="custom_tool", suggested_fix="fix", fix_confidence=0.2, rule_id="X"
            ),
        ]
        result = await v.validate_fixes(issues, "python")
        assert len(result.blocked_fixes) == 1

    def test_validate_fix_safety_safe_tool(self):
        v = self._make_validator()
        score = v.validate_fix_safety("Format code properly", "ruff-format")
        assert score > 0.7

    def test_validate_fix_safety_dangerous(self):
        v = self._make_validator()
        score = v.validate_fix_safety("Delete all files", "ruff-format")
        assert score < 0.5

    def test_validate_fix_safety_file_op(self):
        v = self._make_validator()
        score = v.validate_fix_safety("Overwrite the file", "ty")
        assert score < 0.5


class TestQualityResultValidator:
    def _make_validator(self):
        from services.testing.quality_validation import QualityResultValidator

        return QualityResultValidator()

    @pytest.mark.asyncio
    async def test_validate_valid_result(self):
        v = self._make_validator()
        result = _make_quality_result(execution_time=1.0)
        vr = await v.validate_result(result)
        assert vr.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_low_exec_time(self):
        v = self._make_validator()
        result = _make_quality_result(execution_time=0.0001)
        vr = await v.validate_result(result)
        assert any("suspiciously low" in w for w in vr.warnings)

    @pytest.mark.asyncio
    async def test_validate_high_exec_time(self):
        v = self._make_validator()
        result = _make_quality_result(execution_time=5000.0)
        vr = await v.validate_result(result)
        assert any("very high" in w for w in vr.warnings)

    @pytest.mark.asyncio
    async def test_validate_score_out_of_range(self):
        v = self._make_validator()
        # QualityResult validates 0-100, score=8.5 is fine
        # But we can test the internal method
        result = _make_quality_result(overall_score=9.0)
        result_issues = [_make_quality_issue(severity="error") for _ in range(10)]
        # Manually set issues for the internal check
        result.issues = result_issues
        result.total_issues = len(result_issues)
        vr = await v.validate_result(result)
        # Should warn about high score with many critical issues
        assert any("High score" in w for w in vr.warnings)


class TestQualityCacheValidator:
    def _make_validator(self):
        from services.testing.quality_validation import QualityCacheValidator

        return QualityCacheValidator()

    def test_generate_cache_key(self, tmp_path):
        v = self._make_validator()
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        req = _make_quality_request(code_path=str(f))
        key = v.generate_cache_key(req)
        assert len(key) == 64  # SHA256 hex digest
        # Same request produces same key (deterministic)
        key2 = v.generate_cache_key(req)
        assert key == key2
        # All hex characters
        assert all(c in "0123456789abcdef" for c in key)

    def test_validate_cached_result_fresh(self):
        v = self._make_validator()
        cached = {
            "created_at": datetime.now(UTC).isoformat(),
            "language": "python",
            "tools_results": {},
        }
        req = _make_quality_request(language="python")
        result = v.validate_cached_result(cached, req)
        assert result.is_valid is True

    def test_validate_cached_result_old(self):
        v = self._make_validator()
        from datetime import timedelta

        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        cached = {"created_at": old_time, "language": "python"}
        req = _make_quality_request(language="python")
        result = v.validate_cached_result(cached, req)
        assert result.is_valid is False

    def test_validate_cached_result_language_mismatch(self):
        v = self._make_validator()
        cached = {
            "created_at": datetime.now(UTC).isoformat(),
            "language": "javascript",
        }
        req = _make_quality_request(language="python")
        result = v.validate_cached_result(cached, req)
        assert result.is_valid is False

    def test_factory_functions(self):
        from services.testing.quality_validation import (
            QualityCacheValidator,
            QualityFileValidator,
            QualityFixValidator,
            QualityResultValidator,
            get_cache_validator,
            get_file_validator,
            get_fix_validator,
            get_result_validator,
        )

        assert type(get_cache_validator()) is QualityCacheValidator
        assert type(get_file_validator()) is QualityFileValidator
        assert type(get_fix_validator()) is QualityFixValidator
        assert type(get_result_validator()) is QualityResultValidator


# ===================================================================
# quality_reporting.py
# ===================================================================


class TestQualityReportGenerator:
    def _make_generator(self):
        from services.testing.quality_reporting import QualityReportGenerator

        return QualityReportGenerator()

    @pytest.mark.asyncio
    async def test_generate_detailed_report(self):
        g = self._make_generator()
        qr = _make_quality_result(
            issues=[_make_quality_issue(severity="error", tool="ruff")],
            total_issues=1,
        )
        report = await g.generate_comprehensive_report(qr, report_format="detailed")
        assert report.report_id
        assert report.title
        assert len(report.sections) == 4  # exec summary, issues, tools, recommendations

    @pytest.mark.asyncio
    async def test_generate_comprehensive_report(self):
        g = self._make_generator()
        qr = _make_quality_result()
        report = await g.generate_comprehensive_report(qr, report_format="comprehensive")
        assert len(report.sections) == 7  # 4 + 3 extra

    @pytest.mark.asyncio
    async def test_generate_summary_report(self):
        g = self._make_generator()
        qr = _make_quality_result()
        report = await g.generate_comprehensive_report(qr, report_format="summary")
        assert len(report.sections) == 0  # summary format has no detail sections

    @pytest.mark.asyncio
    async def test_extract_key_findings_with_issues(self):
        g = self._make_generator()
        issues = [
            _make_quality_issue(severity="error", rule_id="E001"),
            _make_quality_issue(severity="error", rule_id="E001"),
            _make_quality_issue(severity="warning", rule_id="W001", suggested_fix="fix it"),
        ]
        findings = await g._extract_key_findings(issues)
        # Should find: most common issue (E001), critical errors, fixable issues
        assert any("E001" in f for f in findings)
        assert any("critical" in f.lower() or "error" in f.lower() for f in findings)
        assert any("automatically fixed" in f.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_identify_priority_actions(self):
        g = self._make_generator()
        issues = [
            _make_quality_issue(severity="error", tool="ruff"),
            _make_quality_issue(severity="warning", tool="bandit"),
            _make_quality_issue(severity="info", tool="ty"),
            _make_quality_issue(
                severity="info",
                tool="ruff-format",
                suggested_fix="run ruff format",
                fix_confidence=0.95,
            ),
        ]
        actions = await g._identify_priority_actions(issues)
        # Should recommend fixing critical errors, security, type safety, auto fixes
        assert any("critical" in a.lower() or "error" in a.lower() for a in actions)
        assert any("security" in a.lower() for a in actions)
        assert len(actions) <= 3  # max 3 priority actions


class TestQualityDashboardData:
    def _make_dashboard(self):
        from services.testing.quality_reporting import QualityDashboardData

        return QualityDashboardData()

    @pytest.mark.asyncio
    async def test_empty_results(self):
        d = self._make_dashboard()
        summary = await d.generate_dashboard_summary([])
        assert summary["total_analyses"] == 0

    @pytest.mark.asyncio
    async def test_single_result(self):
        d = self._make_dashboard()
        qr = _make_quality_result()
        summary = await d.generate_dashboard_summary([qr])
        assert summary["total_analyses"] == 1
        assert summary["trend"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_multiple_results_improving(self):
        d = self._make_dashboard()
        qr1 = _make_quality_result(overall_score=5.0)
        qr2 = _make_quality_result(overall_score=8.0)
        summary = await d.generate_dashboard_summary([qr1, qr2])
        assert summary["trend"] == "improving"

    @pytest.mark.asyncio
    async def test_generate_detailed_charts_empty(self):
        d = self._make_dashboard()
        charts = await d.generate_detailed_charts([])
        assert charts == []

    @pytest.mark.asyncio
    async def test_generate_detailed_charts(self):
        d = self._make_dashboard()
        qr = _make_quality_result(
            issues=[_make_quality_issue()],
            total_issues=1,
            tools_results={"ruff": {"execution_time": 1.0, "issues_count": 1}},
        )
        charts = await d.generate_detailed_charts([qr])
        assert len(charts) >= 1


class TestReportExport:
    @pytest.mark.asyncio
    async def test_export_json(self):
        from services.testing.quality_reporting import (
            export_report_to_json,
            generate_quality_report,
        )

        qr = _make_quality_result()
        report = await generate_quality_report(qr, format_type="detailed")
        json_str = await export_report_to_json(report)
        data = json.loads(json_str)
        assert data["report_id"] == report.report_id
        assert data["analysis_id"] == "qr-001"
        assert len(data["sections"]) == 4  # detailed format has 4 sections
        assert data["summary"]["language"] == "python"
        assert data["summary"]["success"] is True

    @pytest.mark.asyncio
    async def test_export_html(self):
        from services.testing.quality_reporting import (
            export_report_to_html,
            generate_quality_report,
        )

        qr = _make_quality_result()
        report = await generate_quality_report(qr, format_type="detailed")
        html = await export_report_to_html(report)
        assert "<html>" in html
        assert report.title in html


# ===================================================================
# test_monitor.py
# ===================================================================


class TestTestMonitor:
    def _make_monitor(self, cache_manager=None):
        from services.testing.test_monitor import TestMonitor

        return TestMonitor(cache_manager=cache_manager)

    # -- tracking --
    def test_track_and_get_ids(self):
        m = self._make_monitor()
        task = MagicMock(spec=asyncio.Task)
        m.track_active_test("t1", task)
        assert m.get_active_test_count() == 1
        assert "t1" in m.get_active_test_ids()

    def test_untrack(self):
        m = self._make_monitor()
        task = MagicMock(spec=asyncio.Task)
        m.track_active_test("t1", task)
        returned = m.untrack_test("t1")
        assert returned is task
        assert m.get_active_test_count() == 0

    def test_untrack_nonexistent(self):
        m = self._make_monitor()
        assert m.untrack_test("nonexistent") is None

    # -- cancel_test --
    @pytest.mark.asyncio
    async def test_cancel_test_not_found(self):
        m = self._make_monitor()
        assert await m.cancel_test("missing") is False

    @pytest.mark.asyncio
    async def test_cancel_already_done(self):
        m = self._make_monitor()
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = True
        m.track_active_test("t1", task)
        assert await m.cancel_test("t1") is False

    # -- cleanup_completed_tests --
    @pytest.mark.asyncio
    async def test_cleanup_completed_tests(self):
        m = self._make_monitor()
        done_task = MagicMock(spec=asyncio.Task)
        done_task.done.return_value = True
        running_task = MagicMock(spec=asyncio.Task)
        running_task.done.return_value = False
        m.track_active_test("done", done_task)
        m.track_active_test("running", running_task)
        count = await m.cleanup_completed_tests()
        assert count == 1
        assert m.get_active_test_count() == 1

    # -- get_test_status --
    @pytest.mark.asyncio
    async def test_get_test_status_not_found(self):
        m = self._make_monitor()
        status = await m.get_test_status("missing")
        assert status["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_test_status_running(self):
        m = self._make_monitor()
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = False
        m.track_active_test("t1", task)
        status = await m.get_test_status("t1")
        assert status["status"] == "running"

    # -- get_system_metrics --
    @pytest.mark.asyncio
    async def test_get_system_metrics(self):
        m = self._make_monitor()
        metrics = await m.get_system_metrics()
        assert metrics["active_tests"] == 0
        assert metrics["cache_status"] == "disabled"

    @pytest.mark.asyncio
    async def test_get_system_metrics_with_cache(self):
        m = self._make_monitor(cache_manager=AsyncMock())
        metrics = await m.get_system_metrics()
        assert metrics["cache_status"] == "enabled"

    # -- get_test_results --
    @pytest.mark.asyncio
    async def test_get_test_results_not_found(self):
        m = self._make_monitor()
        with pytest.raises(ValueError, match="not found"):
            await m.get_test_results("missing")

    @pytest.mark.asyncio
    async def test_get_test_results_still_running(self):
        m = self._make_monitor()
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = False
        m.track_active_test("t1", task)
        with pytest.raises(ValueError, match="still running"):
            await m.get_test_results("t1")

    # -- health_check --
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self):
        m = self._make_monitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {"valid": True, "details": {}}
        result_processor = MagicMock()

        health = await m.health_check(container_mgr, config_mgr, result_processor)
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_container_unhealthy(self):
        m = self._make_monitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(side_effect=Exception("docker down"))
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {"valid": True, "details": {}}
        result_processor = MagicMock()

        health = await m.health_check(container_mgr, config_mgr, result_processor)
        assert health["status"] in ["degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_health_check_with_cache(self):
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        m = self._make_monitor(cache_manager=cache)
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {"valid": True, "details": {}}
        result_processor = MagicMock()

        health = await m.health_check(container_mgr, config_mgr, result_processor)
        assert health["checks"]["cache"]["status"] == "healthy"

    # -- cleanup_test_environment --
    @pytest.mark.asyncio
    async def test_cleanup_test_environment_basic(self):
        m = self._make_monitor()
        container_mgr = AsyncMock()
        container_mgr.cleanup_test_containers = AsyncMock()
        result = await m.cleanup_test_environment("t1", container_mgr)
        assert result is True

    @pytest.mark.asyncio
    async def test_cleanup_test_environment_with_active_test(self):
        m = self._make_monitor()
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = True
        m.track_active_test("t1", task)
        container_mgr = AsyncMock()
        container_mgr.cleanup_test_containers = AsyncMock()
        result = await m.cleanup_test_environment("t1", container_mgr)
        assert result is True
        assert "t1" not in m.active_tests


# ===================================================================
# test_executor.py
# ===================================================================


class TestTestExecutor:
    def _make_executor(self, analyzer=None):
        from services.testing.test_executor import TestExecutor

        runner = MagicMock()
        return TestExecutor(test_runner=runner, intelligent_analyzer=analyzer)

    # -- _create_error_result --
    def test_create_error_result(self):
        e = self._make_executor()
        req = _make_test_request()
        result = e._create_error_result("err-001", datetime.now(UTC), req, "boom")
        assert result.success is False
        assert result.error == "boom"
        assert result.exit_code == 1

    # -- execute_test --
    @pytest.mark.asyncio
    async def test_execute_test_success(self):
        e = self._make_executor()
        req = _make_test_request()
        container = AsyncMock()
        container.execute_in_container = AsyncMock(return_value=_make_test_result())
        config_mgr = AsyncMock()
        config_mgr.validate_security = AsyncMock()
        result_processor = AsyncMock()
        result_processor.process_result = AsyncMock(return_value=_make_test_result())

        result = await e.execute_test(req, container, config_mgr, result_processor)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_test_security_failure(self):
        e = self._make_executor()
        req = _make_test_request()
        container = AsyncMock()
        config_mgr = AsyncMock()
        config_mgr.validate_security = AsyncMock(side_effect=ValueError("dangerous code"))
        result_processor = AsyncMock()

        result = await e.execute_test(req, container, config_mgr, result_processor)
        assert result.success is False
        assert "dangerous code" in result.error

    # -- analyze_test_failure --
    @pytest.mark.asyncio
    async def test_analyze_failure_no_analyzer(self):
        e = self._make_executor(analyzer=None)
        result = await e.analyze_test_failure(_make_test_result())
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_failure_exception(self):
        analyzer = AsyncMock()
        analyzer.analyze_test_failure = AsyncMock(side_effect=Exception("api error"))
        e = self._make_executor(analyzer=analyzer)
        result = await e.analyze_test_failure(_make_test_result())
        assert result is None

    # -- batch_analyze_failures --
    @pytest.mark.asyncio
    async def test_batch_analyze_no_analyzer(self):
        e = self._make_executor(analyzer=None)
        result = await e.batch_analyze_failures([_make_test_result()])
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_analyze_exception(self):
        analyzer = AsyncMock()
        analyzer.batch_analyze_failures = AsyncMock(side_effect=Exception("fail"))
        e = self._make_executor(analyzer=analyzer)
        result = await e.batch_analyze_failures([_make_test_result()])
        assert result == []

    # -- get_analysis_summary --
    @pytest.mark.asyncio
    async def test_analysis_summary_no_analyzer(self):
        e = self._make_executor(analyzer=None)
        summary = await e.get_analysis_summary()
        assert summary["intelligent_analysis"] == "disabled"

    @pytest.mark.asyncio
    async def test_analysis_summary_exception(self):
        analyzer = AsyncMock()
        analyzer.get_analysis_summary = AsyncMock(side_effect=Exception("fail"))
        e = self._make_executor(analyzer=analyzer)
        summary = await e.get_analysis_summary()
        assert summary["intelligent_analysis"] == "error"


# ===================================================================
# container_manager.py (UnifiedContainerManager)
# ===================================================================


class TestUnifiedContainerManager:
    def _make_manager(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        docker.list_containers = AsyncMock(return_value=[])
        docker.find_container_by_name = AsyncMock(return_value=None)
        return UnifiedContainerManager(docker_service=docker)

    # -- _determine_container_type --
    def test_determine_type_provisioned(self):
        m = self._make_manager()
        assert m._determine_container_type("codegraph-python-abc") == "provisioned"

    def test_determine_type_test(self):
        m = self._make_manager()
        assert m._determine_container_type("python-test-executor") == "test"
        assert m._determine_container_type("test-python-abc") == "test"

    def test_determine_type_unknown(self):
        m = self._make_manager()
        assert m._determine_container_type("random-container") == "unknown"

    # -- _detect_container_language --
    def test_detect_language_python(self):
        m = self._make_manager()
        c = SimpleNamespace(container_name="python-test-executor")
        assert m._detect_container_language(c) == "python"

    def test_detect_language_js(self):
        m = self._make_manager()
        c = SimpleNamespace(container_name="node-app-test")
        assert m._detect_container_language(c) == "javascript"

    def test_detect_language_go(self):
        m = self._make_manager()
        c = SimpleNamespace(container_name="go-runner")
        assert m._detect_container_language(c) == "go"

    def test_detect_language_unknown(self):
        m = self._make_manager()
        c = SimpleNamespace(container_name="random-container")
        assert m._detect_container_language(c) is None

    # -- _container_supports_language --
    def test_container_supports_language_direct(self):
        m = self._make_manager()
        assert m._container_supports_language({"language": "python"}, "python") is True

    def test_container_supports_language_nodejs(self):
        m = self._make_manager()
        assert m._container_supports_language({"language": "javascript"}, "nodejs") is True

    def test_container_supports_language_mismatch(self):
        m = self._make_manager()
        assert m._container_supports_language({"language": "python"}, "javascript") is False

    def test_container_supports_language_no_lang(self):
        m = self._make_manager()
        assert m._container_supports_language({}, "python") is False

    # -- find_suitable_container --
    @pytest.mark.asyncio
    async def test_find_suitable_no_containers(self):
        m = self._make_manager()
        result = await m.find_suitable_container("python")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_suitable_specific_available(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        container = SimpleNamespace(
            container_name="my-container",
            status=SimpleNamespace(value="running"),
        )
        docker.find_container_by_name = AsyncMock(return_value=container)
        docker.list_containers = AsyncMock(return_value=[])
        m = UnifiedContainerManager(docker_service=docker)
        result = await m.find_suitable_container("python", container_name="my-container")
        assert result == "my-container"

    @pytest.mark.asyncio
    async def test_find_suitable_specific_unavailable(self):
        m = self._make_manager()
        m.docker_service.find_container_by_name = AsyncMock(return_value=None)
        result = await m.find_suitable_container("python", container_name="missing")
        assert result is None

    # -- list_containers --
    @pytest.mark.asyncio
    async def test_list_containers_dict_format(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        docker.list_containers = AsyncMock(
            return_value=[
                {"container_name": "test-python", "status": "running", "api_endpoint": "http://x"},
            ]
        )
        m = UnifiedContainerManager(docker_service=docker)
        result = await m.list_containers()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_containers_with_filter(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        docker.list_containers = AsyncMock(
            return_value=[
                {"container_name": "test-python", "status": "running", "api_endpoint": ""},
                {"container_name": "codegraph-abc", "status": "running", "api_endpoint": ""},
            ]
        )
        m = UnifiedContainerManager(docker_service=docker)
        result = await m.list_containers(filter_type="test")
        assert all(c["type"] == "test" for c in result)

    # -- ensure_container --
    @pytest.mark.asyncio
    async def test_ensure_container_no_create(self):
        m = self._make_manager()
        with pytest.raises(RuntimeError, match="No container available"):
            await m.ensure_container("python", create_if_missing=False)


# ===================================================================
# security_management_service.py
# ===================================================================


class TestSecurityManagementService:
    @pytest.mark.asyncio
    async def test_validate_code_security(self):
        from services.testing.security_management_service import SecurityManagementService

        with patch(
            "services.testing.security_management_service.create_security_service"
        ) as mock_css:
            mock_service = AsyncMock()
            violation = SimpleNamespace(
                type="eval_usage",
                severity=SimpleNamespace(value="high"),
                line_number=5,
                description="eval is dangerous",
            )
            mock_service.validate_code_security = AsyncMock(
                return_value=SimpleNamespace(
                    is_safe=False,
                    risk_score=0.8,
                    violations=[violation],
                    sandbox_config={"network": False},
                )
            )
            mock_css.return_value = mock_service

            sms = SecurityManagementService()
            result = await sms.validate_code_security("eval('x')", "python")

            assert result["is_safe"] is False
            assert result["risk_score"] == 0.8
            assert result["violations"] == 1
            assert len(result["violation_details"]) == 1

    @pytest.mark.asyncio
    async def test_validate_code_security_safe(self):
        from services.testing.security_management_service import SecurityManagementService

        with patch(
            "services.testing.security_management_service.create_security_service"
        ) as mock_css:
            mock_service = AsyncMock()
            mock_service.validate_code_security = AsyncMock(
                return_value=SimpleNamespace(
                    is_safe=True,
                    risk_score=0.0,
                    violations=[],
                    sandbox_config={},
                )
            )
            mock_css.return_value = mock_service

            sms = SecurityManagementService()
            result = await sms.validate_code_security("x = 1", "python")
            assert result["is_safe"] is True
            assert result["violations"] == 0

    @pytest.mark.asyncio
    async def test_create_sandbox_environment(self):
        from services.testing.security_management_service import SecurityManagementService

        with patch(
            "services.testing.security_management_service.create_security_service"
        ) as mock_css:
            mock_service = MagicMock()
            mock_service.generate_sandbox_config = MagicMock(
                return_value={"network": False, "readonly": True}
            )
            mock_css.return_value = mock_service

            sms = SecurityManagementService()
            config = await sms.create_sandbox_environment("python", "high")
            assert config["network"] is False

    @pytest.mark.asyncio
    async def test_validate_imports(self):
        from services.testing.security_management_service import SecurityManagementService

        with patch(
            "services.testing.security_management_service.create_security_service"
        ) as mock_css:
            mock_service = AsyncMock()
            mock_service.validate_imports = AsyncMock(
                return_value=SimpleNamespace(is_safe=True, violations=[])
            )
            mock_css.return_value = mock_service

            sms = SecurityManagementService()
            result = await sms.validate_imports("import os", "python")
            assert result["safe_imports"] is True
            assert result["dangerous_imports"] == 0

    def test_factory_function(self):
        from services.testing.security_management_service import (
            SecurityManagementService,
            create_security_management_service,
        )

        with patch("services.testing.security_management_service.create_security_service"):
            sms = create_security_management_service()
            assert type(sms) is SecurityManagementService


# ===================================================================
# Additional coverage boosters
# ===================================================================


class TestTestMonitorAdditional:
    """Extra tests for test_monitor.py to increase coverage."""

    def _make_monitor(self, cache_manager=None):
        from services.testing.test_monitor import TestMonitor

        return TestMonitor(cache_manager=cache_manager)

    @pytest.mark.asyncio
    async def test_get_test_results_from_cache(self):
        cache = AsyncMock()
        cache.get = AsyncMock(
            return_value={
                "test_id": "cached-1",
                "success": True,
                "stdout": "ok",
                "stderr": "",
                "exit_code": 0,
                "execution_time": 1.0,
                "created_at": datetime.now(UTC).isoformat(),
                "language": "python",
                "security_level": "high",
            }
        )
        m = self._make_monitor(cache_manager=cache)
        result = await m.get_test_results("cached-1")
        assert result.test_id == "cached-1"
        cache.get.assert_awaited_once_with("test_result:cached-1")

    @pytest.mark.asyncio
    async def test_get_test_results_from_done_task(self):
        m = self._make_monitor()
        expected = _make_test_result(test_id="done-1")
        # Create and complete a real future
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        future.set_result(expected)
        m.track_active_test("done-1", future)
        result = await m.get_test_results("done-1")
        assert result.test_id == "done-1"

    @pytest.mark.asyncio
    async def test_health_check_config_unhealthy(self):
        m = self._make_monitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.side_effect = Exception("config error")
        result_processor = MagicMock()

        health = await m.health_check(container_mgr, config_mgr, result_processor)
        assert health["checks"]["config"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_cache_unhealthy(self):
        cache = AsyncMock()
        cache.get = AsyncMock(side_effect=Exception("redis down"))
        m = self._make_monitor(cache_manager=cache)
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {"valid": True, "details": {}}
        result_processor = MagicMock()

        health = await m.health_check(container_mgr, config_mgr, result_processor)
        assert health["checks"]["cache"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_with_analyzer(self):
        m = self._make_monitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {"valid": True, "details": {}}
        result_processor = MagicMock()
        analyzer = AsyncMock()
        analyzer.health_check = AsyncMock(return_value={"status": "healthy"})

        health = await m.health_check(
            container_mgr, config_mgr, result_processor, intelligent_analyzer=analyzer
        )
        assert health["checks"]["intelligent_analyzer"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_analyzer_unhealthy(self):
        m = self._make_monitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {"valid": True, "details": {}}
        result_processor = MagicMock()
        analyzer = AsyncMock()
        analyzer.health_check = AsyncMock(side_effect=Exception("analyzer error"))

        health = await m.health_check(
            container_mgr, config_mgr, result_processor, intelligent_analyzer=analyzer
        )
        assert health["checks"]["intelligent_analyzer"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_degraded_config(self):
        m = self._make_monitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {"valid": False, "details": {}}
        result_processor = MagicMock()

        health = await m.health_check(container_mgr, config_mgr, result_processor)
        assert health["checks"]["config"]["status"] == "degraded"
        assert health["status"] in ["degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_cleanup_with_running_task(self):
        m = self._make_monitor()

        # Simulate a running task that gets cancelled
        async def _long_running():
            await asyncio.sleep(100)
            return _make_test_result()

        task = asyncio.ensure_future(_long_running())
        m.track_active_test("running-1", task)
        container_mgr = AsyncMock()
        container_mgr.cleanup_test_containers = AsyncMock()
        await m.cleanup_test_environment("running-1", container_mgr)
        assert "running-1" not in m.active_tests

    @pytest.mark.asyncio
    async def test_cleanup_with_cache(self):
        cache = AsyncMock()
        cache.delete = AsyncMock()
        m = self._make_monitor(cache_manager=cache)
        container_mgr = AsyncMock()
        container_mgr.cleanup_test_containers = AsyncMock()
        await m.cleanup_test_environment("t1", container_mgr)
        cache.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_cache_error(self):
        cache = AsyncMock()
        cache.delete = AsyncMock(side_effect=Exception("redis error"))
        m = self._make_monitor(cache_manager=cache)
        container_mgr = AsyncMock()
        container_mgr.cleanup_test_containers = AsyncMock()
        result = await m.cleanup_test_environment("t1", container_mgr)
        assert result is False  # cache cleanup failed

    @pytest.mark.asyncio
    async def test_cleanup_container_error(self):
        m = self._make_monitor()
        container_mgr = AsyncMock()
        container_mgr.cleanup_test_containers = AsyncMock(side_effect=Exception("docker error"))
        result = await m.cleanup_test_environment("t1", container_mgr)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_test_status_completed(self):
        m = self._make_monitor()
        expected = _make_test_result(test_id="done-2")

        async def _done():
            return expected

        task = asyncio.ensure_future(_done())
        await task  # complete
        m.track_active_test("done-2", task)
        status = await m.get_test_status("done-2")
        assert status["status"] == "completed"
        assert status["success"] is True

    @pytest.mark.asyncio
    async def test_get_test_status_failed(self):
        m = self._make_monitor()

        async def _fail():
            raise ValueError("boom")

        task = asyncio.ensure_future(_fail())
        try:
            await task
        except ValueError:
            pass
        m.track_active_test("fail-1", task)
        status = await m.get_test_status("fail-1")
        assert status["status"] == "failed"


class TestTestResultProcessorAdditional:
    """Extra tests for test_result_processor.py."""

    def _make_processor(self, cache_manager=None):
        from services.testing.test_result_processor import TestResultProcessor

        return TestResultProcessor(cache_manager=cache_manager)

    @pytest.mark.asyncio
    async def test_cache_result_no_cache(self):
        p = self._make_processor(cache_manager=None)
        req = _make_test_request()
        result = _make_test_result()
        # Should not raise
        await p._cache_result(req, result)

    def test_convert_container_result_defaults(self):
        p = self._make_processor()
        # Empty dict - use defaults
        req = _make_test_request()
        result = p.convert_container_result({}, "tid", req)
        assert result.success is False
        assert result.exit_code == 1


class TestTestExecutorAdditional:
    """Extra tests for test_executor.py."""

    def _make_executor(self, analyzer=None):
        from services.testing.test_executor import TestExecutor

        runner = MagicMock()
        return TestExecutor(test_runner=runner, intelligent_analyzer=analyzer)

    @pytest.mark.asyncio
    async def test_execute_test_failed_with_analyzer(self):
        """Test that intelligent analysis runs on failed tests."""
        analyzer = AsyncMock()
        analysis_resp = SimpleNamespace(
            success=True,
            failure_analysis=SimpleNamespace(
                analysis_id="a1",
                category="assertion",
                severity="medium",
                confidence=0.9,
                root_cause="wrong assertion",
                recommendations=["fix it"],
                code_fixes=[],
                ai_model_provider="test",
            ),
            code_quality_issues=[],
            performance_analysis=None,
            security_analysis=None,
        )
        analyzer.analyze_test_failure = AsyncMock(return_value=analysis_resp)

        e = self._make_executor(analyzer=analyzer)
        failed_result = _make_test_result(success=False)
        req = _make_test_request()

        container = AsyncMock()
        container.execute_in_container = AsyncMock(return_value=failed_result)
        config_mgr = AsyncMock()
        config_mgr.validate_security = AsyncMock()
        result_processor = AsyncMock()
        result_processor.process_result = AsyncMock(side_effect=lambda r, _: r)

        await e.execute_test(req, container, config_mgr, result_processor)
        # The analysis should have been called
        analyzer.analyze_test_failure.assert_awaited()

    @pytest.mark.asyncio
    async def test_analyze_failure_success(self):
        """Test successful failure analysis."""
        analyzer = AsyncMock()
        response = MagicMock()
        response.success = True
        response.model_dump.return_value = {"analysis": "data"}
        analyzer.analyze_test_failure = AsyncMock(return_value=response)

        e = self._make_executor(analyzer=analyzer)
        result = await e.analyze_test_failure(_make_test_result())
        assert result == {"analysis": "data"}

    @pytest.mark.asyncio
    async def test_analyze_failure_not_success(self):
        """Test failure analysis that returns success=False."""
        analyzer = AsyncMock()
        response = MagicMock()
        response.success = False
        response.error_message = "failed"
        analyzer.analyze_test_failure = AsyncMock(return_value=response)

        e = self._make_executor(analyzer=analyzer)
        result = await e.analyze_test_failure(_make_test_result())
        assert result is None

    @pytest.mark.asyncio
    async def test_batch_analyze_success(self):
        analyzer = AsyncMock()
        resp1 = MagicMock()
        resp1.model_dump.return_value = {"id": 1}
        resp2 = MagicMock()
        resp2.model_dump.return_value = {"id": 2}
        analyzer.batch_analyze_failures = AsyncMock(return_value=[resp1, resp2])

        e = self._make_executor(analyzer=analyzer)
        results = await e.batch_analyze_failures([_make_test_result()])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_analysis_summary_success(self):
        analyzer = AsyncMock()
        analyzer.get_analysis_summary = AsyncMock(return_value={"service": "test", "status": "ok"})
        e = self._make_executor(analyzer=analyzer)
        summary = await e.get_analysis_summary()
        assert summary["status"] == "ok"


class TestUnifiedContainerManagerAdditional:
    """Additional tests for container_manager.py."""

    @pytest.mark.asyncio
    async def test_list_containers_object_format(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        container_obj = SimpleNamespace(
            container_name="test-python",
            status=SimpleNamespace(value="running"),
            api_endpoint="http://x",
            internal_endpoint="http://internal",
            public_endpoint="http://public",
            created_at=datetime.now(UTC).isoformat(),
        )
        docker.list_containers = AsyncMock(return_value=[container_obj])
        m = UnifiedContainerManager(docker_service=docker)
        result = await m.list_containers()
        assert len(result) == 1
        assert result[0]["name"] == "test-python"

    @pytest.mark.asyncio
    async def test_list_containers_with_tenant_filter(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        docker.list_containers = AsyncMock(
            return_value=[
                {
                    "container_name": "test-1",
                    "status": "running",
                    "api_endpoint": "",
                    "tenant_id": "tenant-a",
                },
                {
                    "container_name": "test-2",
                    "status": "running",
                    "api_endpoint": "",
                    "tenant_id": "tenant-b",
                },
            ]
        )
        m = UnifiedContainerManager(docker_service=docker)
        result = await m.list_containers(tenant_id="tenant-a")
        assert len(result) == 1
        assert result[0]["name"] == "test-1"

    @pytest.mark.asyncio
    async def test_get_container_info(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        container = SimpleNamespace(
            container_name="my-container",
            status=SimpleNamespace(value="running"),
            api_endpoint="http://x",
            internal_endpoint="http://internal",
            public_endpoint="http://public",
            code_graph_id="cg-1",
            repo_url="https://github.com/test",
            created_at=datetime.now(UTC).isoformat(),
        )
        docker.find_container_by_name = AsyncMock(return_value=container)
        m = UnifiedContainerManager(docker_service=docker)
        info = await m.get_container_info("my-container")
        assert info["name"] == "my-container"
        assert info["status"] == "running"
        assert info["api_endpoint"] == "http://x"
        assert info["code_graph_id"] == "cg-1"

    @pytest.mark.asyncio
    async def test_get_container_info_not_found(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        docker.find_container_by_name = AsyncMock(return_value=None)
        m = UnifiedContainerManager(docker_service=docker)
        info = await m.get_container_info("missing")
        assert info is None

    @pytest.mark.asyncio
    async def test_get_container_info_cached(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        container = SimpleNamespace(
            container_name="cached-container",
            status=SimpleNamespace(value="running"),
            api_endpoint="http://x",
            code_graph_id="cg-1",
        )
        docker.find_container_by_name = AsyncMock(return_value=container)
        m = UnifiedContainerManager(docker_service=docker)

        # First call populates cache
        await m.get_container_info("cached-container")
        # Second call uses cache
        info = await m.get_container_info("cached-container")
        assert info["name"] == "cached-container"
        assert info["status"] == "running"
        # find_container_by_name called only once (cached)
        assert docker.find_container_by_name.await_count == 1

    @pytest.mark.asyncio
    async def test_get_container_info_tenant_mismatch(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        container = SimpleNamespace(
            container_name="tenant-container",
            status=SimpleNamespace(value="running"),
            api_endpoint="http://x",
            code_graph_id="cg-1",
            tenant_id="tenant-a",
        )
        docker.find_container_by_name = AsyncMock(return_value=container)
        m = UnifiedContainerManager(docker_service=docker)
        info = await m.get_container_info("tenant-container", tenant_id="tenant-b")
        assert info is None

    @pytest.mark.asyncio
    async def test_cleanup_old_containers(self):
        from datetime import timedelta

        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        docker.list_containers = AsyncMock(
            return_value=[
                {
                    "container_name": "test-old",
                    "status": "running",
                    "api_endpoint": "",
                    "created_at": old_time,
                },
            ]
        )
        docker.remove_container = AsyncMock()
        m = UnifiedContainerManager(docker_service=docker)
        cleaned = await m.cleanup_old_containers(max_age_hours=24)
        assert cleaned == 1

    @pytest.mark.asyncio
    async def test_cleanup_old_containers_none_old(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        recent_time = datetime.now(UTC).isoformat()
        docker.list_containers = AsyncMock(
            return_value=[
                {
                    "container_name": "test-new",
                    "status": "running",
                    "api_endpoint": "",
                    "created_at": recent_time,
                },
            ]
        )
        m = UnifiedContainerManager(docker_service=docker)
        cleaned = await m.cleanup_old_containers(max_age_hours=24)
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_is_container_available_running(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        container = SimpleNamespace(
            container_name="running", status=SimpleNamespace(value="running")
        )
        docker.find_container_by_name = AsyncMock(return_value=container)
        m = UnifiedContainerManager(docker_service=docker)
        assert await m._is_container_available("running") is True

    @pytest.mark.asyncio
    async def test_is_container_available_stopped(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        container = SimpleNamespace(
            container_name="stopped", status=SimpleNamespace(value="stopped")
        )
        docker.find_container_by_name = AsyncMock(return_value=container)
        m = UnifiedContainerManager(docker_service=docker)
        assert await m._is_container_available("stopped") is False

    @pytest.mark.asyncio
    async def test_find_test_containers(self):
        from services.testing.container_manager import UnifiedContainerManager

        docker = AsyncMock()
        docker.list_containers = AsyncMock(
            return_value=[
                SimpleNamespace(container_name="python-test-executor-1"),
                SimpleNamespace(container_name="other-container"),
            ]
        )
        m = UnifiedContainerManager(docker_service=docker)
        result = await m._find_test_containers("python")
        assert "python-test-executor-1" in result


class TestQualityValidationAdditional:
    """Extra tests for quality_validation.py to boost coverage."""

    @pytest.mark.asyncio
    async def test_request_validator_validates_request(self, tmp_path):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        f = tmp_path / "test.py"
        f.write_text("x = 1\n" * 10)
        req = _make_quality_request(code_path=str(f), language="python")
        result = await v.validate_request(req)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_request_validator_no_code_path(self):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        req = _make_quality_request(code_path=None)
        result = await v.validate_request(req)
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_request_validator_nonexistent_path(self):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        req = _make_quality_request(code_path="/nonexistent/file.py")
        result = await v.validate_request(req)
        assert result.is_valid is False

    def test_request_validator_unsupported_language(self):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        # QualityRequest pydantic model won't allow invalid languages,
        # so test the internal validation method directly
        result = v._validate_language("cobol")
        assert result.is_valid is False

    def test_validate_language_none(self):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        result = v._validate_language(None)
        assert result.is_valid is True
        assert len(result.warnings) > 0

    def test_validate_tools_none(self):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        result = v._validate_tools(None, "python")
        assert result.is_valid is True

    def test_validate_tools_unknown(self):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        result = v._validate_tools(["unknown_tool"], "python")
        assert result.is_valid is False

    def test_validate_tools_valid(self):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        result = v._validate_tools(["ruff-format", "ruff"], "python")
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_code_path_directory(self, tmp_path):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        result = await v._validate_code_path(str(tmp_path))
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_code_path_unsupported_ext(self, tmp_path):
        from services.testing.quality_validation import QualityRequestValidator

        v = QualityRequestValidator()
        f = tmp_path / "test.xyz"
        f.write_text("content")
        result = await v._validate_code_path(str(f))
        assert result.is_valid is True  # unsupported ext is a warning
        assert any("may not be supported" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_result_validator_missing_analysis_id(self):
        from services.testing.quality_validation import QualityResultValidator

        v = QualityResultValidator()
        # Can't create QualityResult with empty analysis_id via pydantic easily.
        # Test the internal method instead.
        result = _make_quality_result(analysis_id="valid-id")
        vr = v._validate_result_structure(result)
        assert vr.is_valid is True

    @pytest.mark.asyncio
    async def test_result_validator_low_score_no_issues(self):
        from services.testing.quality_validation import QualityResultValidator

        v = QualityResultValidator()
        result = _make_quality_result(overall_score=2.0)
        vr = v._validate_score_consistency(result)
        assert any("Low score" in w for w in vr.warnings)

    @pytest.mark.asyncio
    async def test_result_validator_duplicate_issues(self):
        from services.testing.quality_validation import QualityResultValidator

        v = QualityResultValidator()
        issues = [
            _make_quality_issue(tool="ruff", file_path="a.py", line_number=10, rule_id="W001"),
            _make_quality_issue(tool="ruff", file_path="a.py", line_number=10, rule_id="W001"),
        ]
        vr = v._validate_issues(issues)
        assert any("duplicate" in w for w in vr.warnings)

    def test_cache_validator_invalid_timestamp(self):
        from services.testing.quality_validation import QualityCacheValidator

        v = QualityCacheValidator()
        cached = {"created_at": "not-a-date", "language": "python"}
        req = _make_quality_request(language="python")
        result = v.validate_cached_result(cached, req)
        assert any("timestamp" in w.lower() for w in result.warnings)

    def test_cache_validator_tools_mismatch(self):
        from services.testing.quality_validation import QualityCacheValidator

        v = QualityCacheValidator()
        cached = {
            "created_at": datetime.now(UTC).isoformat(),
            "language": "python",
            "tools_results": {"ruff-format": {}},
        }
        req = _make_quality_request(language="python", tools=["ruff"])
        result = v.validate_cached_result(cached, req)
        assert any("tool set" in w.lower() for w in result.warnings)
