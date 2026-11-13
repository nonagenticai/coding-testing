"""Unit tests for services.intelligent_testing_service module."""

from __future__ import annotations

import re
from unittest.mock import patch

from shared.container_validation import ContainerInfo, ContainerStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_container(name="test-container") -> ContainerInfo:
    return ContainerInfo(
        container_name=name,
        status=ContainerStatus.RUNNING,
    )


# ---------------------------------------------------------------------------
# Service instantiation
# ---------------------------------------------------------------------------


class TestServiceInstantiation:
    @patch("services.intelligent_testing_service.get_docker_service")
    def test_init(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        assert svc.docker_service is mock_docker.return_value

    @patch("services.intelligent_testing_service.get_docker_service")
    def test_singleton(self, mock_docker):
        import services.intelligent_testing_service as mod

        mod._service_instance = None
        svc1 = mod.get_intelligent_testing_service()
        svc2 = mod.get_intelligent_testing_service()
        assert svc1 is svc2
        # Cleanup
        mod._service_instance = None


# ---------------------------------------------------------------------------
# analyze_test_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeTestPatterns:
    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_basic_pattern_analysis(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        test_results = [
            {"test_name": "test_1", "passed": True, "execution_time": 1.0},
            {"test_name": "test_2", "passed": False, "execution_time": 6.0},
        ]
        result = await svc.analyze_test_patterns(container, test_results)

        assert result["container_name"] == "test-container"
        # One slow test (>5s) and one failed test = 2 patterns
        assert result["pattern_count"] == 2
        assert len(result["patterns_detected"]) == 2
        assert len(result["performance_patterns"]) == 1
        assert len(result["failure_patterns"]) == 1
        assert len(result["timing_patterns"]) == 0
        # 2 patterns * 0.2 = 0.4
        assert result["confidence_score"] == 0.4
        assert result["analyzed_at"] is not None

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_empty_test_results(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        result = await svc.analyze_test_patterns(container, [])

        assert result["pattern_count"] == 0
        assert result["confidence_score"] == 0.0
        assert result["patterns_detected"] == []
        assert result["performance_patterns"] == []
        assert result["failure_patterns"] == []
        assert result["timing_patterns"] == []

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_disable_categories(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        test_results = [
            {"test_name": "t", "passed": False, "execution_time": 10.0},
        ]
        options = {
            "include_performance_patterns": False,
            "include_failure_patterns": False,
            "include_timing_patterns": False,
        }
        result = await svc.analyze_test_patterns(container, test_results, options)
        assert result["pattern_count"] == 0
        assert result["patterns_detected"] == []
        assert result["confidence_score"] == 0.0


# ---------------------------------------------------------------------------
# _detect_performance_patterns
# ---------------------------------------------------------------------------


class TestDetectPerformancePatterns:
    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_slow_tests_detected(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        test_results = [
            {"execution_time": 10.0},
            {"execution_time": 1.0},
        ]
        patterns = await svc._detect_performance_patterns(container, test_results)
        assert len(patterns) == 1
        assert patterns[0]["category"] == "slow_tests"
        assert patterns[0]["test_count"] == 1
        assert patterns[0]["pattern_type"] == "performance"
        assert patterns[0]["severity"] == "medium"

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_no_slow_tests(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        test_results = [{"execution_time": 1.0}, {"execution_time": 2.0}]
        patterns = await svc._detect_performance_patterns(container, test_results)
        assert patterns == []

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_multiple_slow_tests(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        test_results = [
            {"execution_time": 6.0},
            {"execution_time": 7.0},
            {"execution_time": 1.0},
        ]
        patterns = await svc._detect_performance_patterns(container, test_results)
        assert len(patterns) == 1
        assert patterns[0]["test_count"] == 2


# ---------------------------------------------------------------------------
# _detect_failure_patterns
# ---------------------------------------------------------------------------


class TestDetectFailurePatterns:
    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_failures_detected(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        test_results = [
            {"passed": False},
            {"passed": True},
            {"passed": False},
        ]
        patterns = await svc._detect_failure_patterns(container, test_results)
        assert len(patterns) == 1
        assert patterns[0]["category"] == "test_failures"
        assert patterns[0]["test_count"] == 2
        assert patterns[0]["severity"] == "high"
        assert patterns[0]["pattern_type"] == "failure"

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_no_failures(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        test_results = [{"passed": True}]
        patterns = await svc._detect_failure_patterns(container, test_results)
        assert patterns == []


# ---------------------------------------------------------------------------
# _detect_timing_patterns
# ---------------------------------------------------------------------------


class TestDetectTimingPatterns:
    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_returns_empty(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        patterns = await svc._detect_timing_patterns(container, [{"t": 1}])
        assert patterns == []


# ---------------------------------------------------------------------------
# _calculate_pattern_confidence
# ---------------------------------------------------------------------------


class TestCalculatePatternConfidence:
    @patch("services.intelligent_testing_service.get_docker_service")
    def test_empty_patterns(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        assert svc._calculate_pattern_confidence([]) == 0.0

    @patch("services.intelligent_testing_service.get_docker_service")
    def test_single_pattern(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        assert svc._calculate_pattern_confidence([{"x": 1}]) == 0.2

    @patch("services.intelligent_testing_service.get_docker_service")
    def test_two_patterns(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        assert svc._calculate_pattern_confidence([{"x": 1}, {"x": 2}]) == 0.4

    @patch("services.intelligent_testing_service.get_docker_service")
    def test_capped_at_1(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        patterns = [{"x": i} for i in range(10)]
        assert svc._calculate_pattern_confidence(patterns) == 1.0


# ---------------------------------------------------------------------------
# generate_test_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_returns_no_invented_recommendations(self, mock_docker):
        # Was a single hard-coded "Add integration tests for API endpoints"
        # (rec_001, confidence 0.85) for every workspace. Now honest: empty +
        # computed=False.
        from services.intelligent_testing_service import (
            NOT_IMPLEMENTED_REASON,
            IntelligentTestingService,
        )

        svc = IntelligentTestingService()
        container = _make_container()
        result = await svc.generate_test_recommendations(container)

        assert result["container_name"] == "test-container"
        assert result["recommendations"] == []
        assert result["recommendation_count"] == 0
        assert result["high_priority_count"] == 0
        assert result["computed"] is False
        assert result["reason"] == NOT_IMPLEMENTED_REASON
        assert result["generated_at"] is not None

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_with_context(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        result = await svc.generate_test_recommendations(container, context={"language": "python"})
        assert result["recommendation_count"] == 0
        assert result["computed"] is False


# ---------------------------------------------------------------------------
# Fake workspace exec (emulates coding-containers find + grep)
# ---------------------------------------------------------------------------

_IMPORT_LINE = re.compile(r"^[ \t]*(from|import)[ \t]+[A-Za-z_.]")

EMS_TREE = {
    "src/__init__.py": "",
    "src/auth.py": "import os\n\ndef login():\n    return os.getpid()\n",
    "src/api.py": "from src.auth import login\n\ndef handler():\n    return login()\n",
    "src/views.py": "from src import api\n",
    "src/unrelated.py": "import json\n",
    "tests/test_auth_x.py": "import pytest\nfrom src.auth import login\n",
    "tests/test_api.py": "from src.api import handler\n",
    "tests/test_unrelated.py": "from src.unrelated import json\n",
}


class FakeExec:
    """Emulates POST /api/workspaces/{name}/exec for the find and grep argv."""

    def __init__(self, tree, overrides=None):
        self.tree = tree
        self.overrides = overrides or {}
        self.calls = []

    async def __call__(self, container_name, tenant_id, argv, timeout):
        self.calls.append((container_name, tenant_id, list(argv), timeout))
        tool = argv[0]
        if tool in self.overrides:
            return self.overrides[tool]
        if tool == "find":
            stdout = "".join(f"./{p}\n" for p in sorted(self.tree))
            return {"exit_code": 0, "stdout": stdout, "stderr": "", "truncated": False}
        if tool == "grep":
            out = []
            for p in sorted(self.tree):
                for n, line in enumerate(self.tree[p].splitlines(), 1):
                    if _IMPORT_LINE.match(line):
                        out.append(f"./{p}:{n}:{line}\n")
            return {
                "exit_code": 0 if out else 1,
                "stdout": "".join(out),
                "stderr": "",
                "truncated": False,
            }
        raise AssertionError(f"unexpected argv {argv}")


def _svc(fake):
    from services.intelligent_testing_service import IntelligentTestingService

    with patch("services.intelligent_testing_service.get_docker_service"):
        return IntelligentTestingService(workspace_exec=fake)


# ---------------------------------------------------------------------------
# analyze_test_impact (real import graph)
# ---------------------------------------------------------------------------


class TestAnalyzeTestImpact:
    async def test_ems_direct_and_indirect(self):
        fake = FakeExec(EMS_TREE)
        result = await _svc(fake).analyze_test_impact(
            _make_container(), ["src/auth.py"], tenant_id="t-1"
        )

        assert result["status"] == "computed"
        by_file = {i["test_file"]: i for i in result["impacted_tests"]}
        assert set(by_file) == {"tests/test_auth_x.py", "tests/test_api.py"}
        direct = by_file["tests/test_auth_x.py"]
        assert direct["impact_type"] == "direct"
        assert direct["impact_score"] == 10.0
        assert direct["via"] == []
        indirect = by_file["tests/test_api.py"]
        assert indirect["impact_type"] == "indirect"
        assert indirect["via"] == ["src/api.py"]
        assert indirect["impact_score"] == 5.0
        assert result["direct_impacts"] == 1
        assert result["indirect_impacts"] == 1
        # (2 / 1) * 5 capped at 10
        assert result["impact_score"] == 10.0
        # Exactly one find + one grep, tenant threaded through.
        assert [c[2][0] for c in fake.calls] == ["find", "grep"]
        assert all(c[1] == "t-1" for c in fake.calls)

    async def test_never_invents_placeholder_path(self):
        # Negative control for the old placeholder: src/auth.py -> src/auth_test.py.
        result = await _svc(FakeExec(EMS_TREE)).analyze_test_impact(
            _make_container(), ["src/auth.py"]
        )
        files = {i["test_file"] for i in result["impacted_tests"]}
        assert "src/auth_test.py" not in files
        assert files <= set(EMS_TREE)
        # The unrelated test does not import auth at all.
        assert "tests/test_unrelated.py" not in files

    async def test_without_indirect(self):
        result = await _svc(FakeExec(EMS_TREE)).analyze_test_impact(
            _make_container(), ["src/auth.py"], {"include_indirect_impacts": False}
        )
        assert result["indirect_impacts"] == 0
        assert [i["test_file"] for i in result["impacted_tests"]] == ["tests/test_auth_x.py"]

    async def test_truncated_grep_is_not_computed(self):
        fake = FakeExec(
            EMS_TREE,
            overrides={
                "grep": {
                    "exit_code": 0,
                    "stdout": "./tests/test_auth_x.py:2:from src.auth import login\n",
                    "truncated": True,
                    "bytes_omitted": 4096,
                }
            },
        )
        result = await _svc(fake).analyze_test_impact(_make_container(), ["src/auth.py"])
        assert result["status"] == "not_computed"
        assert "truncated" in result["reason"]
        assert result["impacted_tests"] == []
        assert result["impact_count"] == 0
        assert result["impact_score"] == 0.0

    async def test_grep_timeout_is_not_computed(self):
        fake = FakeExec(EMS_TREE, overrides={"grep": {"exit_code": 124, "stdout": ""}})
        result = await _svc(fake).analyze_test_impact(_make_container(), ["src/auth.py"])
        assert result["status"] == "not_computed"
        assert "timed out" in result["reason"]

    async def test_find_error_exit_is_not_computed(self):
        fake = FakeExec(
            EMS_TREE, overrides={"find": {"exit_code": 2, "stdout": "", "stderr": "boom"}}
        )
        result = await _svc(fake).analyze_test_impact(_make_container(), ["src/auth.py"])
        assert result["status"] == "not_computed"
        assert "exited 2" in result["reason"]

    async def test_find_partial_listing_exit_1_is_not_computed(self):
        """find exits 1 when it could not read a subdirectory: a partial listing."""
        fake = FakeExec(
            EMS_TREE,
            overrides={
                "find": {
                    "exit_code": 1,
                    "stdout": "/workspace/src/auth.py\n",
                    "stderr": "find: './tests': Permission denied",
                }
            },
        )
        result = await _svc(fake).analyze_test_impact(_make_container(), ["src/auth.py"])
        assert result["status"] == "not_computed"
        assert "exited 1" in result["reason"]
        assert result["impacted_tests"] == []

    async def test_grep_no_match_exit_1_is_complete(self):
        """grep exit 1 is 'no match', a complete answer: impact stays computed."""
        fake = FakeExec(EMS_TREE, overrides={"grep": {"exit_code": 1, "stdout": ""}})
        result = await _svc(fake).analyze_test_impact(_make_container(), ["src/auth.py"])
        assert result["status"] == "computed"

    async def test_exec_transport_error_is_not_computed(self):
        import httpx

        async def broken(*_a):
            raise httpx.ConnectError("refused")

        result = await _svc(broken).analyze_test_impact(_make_container(), ["src/auth.py"])
        assert result["status"] == "not_computed"
        assert "refused" in result["reason"]

    async def test_ghost_changed_file_is_unknown(self):
        result = await _svc(FakeExec(EMS_TREE)).analyze_test_impact(
            _make_container(), ["src/ghost.py"]
        )
        assert result["status"] == "computed"
        assert result["unknown_changed_files"] == ["src/ghost.py"]
        assert result["impacted_tests"] == []

    async def test_non_python_changed_file_is_unsupported(self):
        fake = FakeExec(EMS_TREE)
        result = await _svc(fake).analyze_test_impact(_make_container(), ["web/app.ts"])
        assert result["status"] == "computed"
        assert result["unsupported_changed_files"] == ["web/app.ts"]
        assert result["impacted_tests"] == []
        assert fake.calls == []  # nothing Python to analyse -> no exec

    async def test_changed_test_file_impacts_itself(self):
        result = await _svc(FakeExec(EMS_TREE)).analyze_test_impact(
            _make_container(), ["tests/test_api.py"]
        )
        assert [(i["test_file"], i["impact_type"]) for i in result["impacted_tests"]] == [
            ("tests/test_api.py", "direct")
        ]

    async def test_empty_changed_files(self):
        fake = FakeExec(EMS_TREE)
        result = await _svc(fake).analyze_test_impact(_make_container(), [])
        assert result["status"] == "computed"
        assert result["impact_count"] == 0
        assert result["impact_score"] == 0.0
        assert result["impacted_tests"] == []
        assert fake.calls == []

    async def test_container_tenant_preferred_for_exec(self):
        fake = FakeExec(EMS_TREE)
        container = ContainerInfo(container_name="test-container", tenant_id="owner")
        await _svc(fake).analyze_test_impact(container, ["src/auth.py"], tenant_id="caller")
        assert {c[1] for c in fake.calls} == {"owner"}

    async def test_response_model_accepts_result(self):
        from models.intelligent_testing_models import TestImpactAnalysisResponse

        result = await _svc(FakeExec(EMS_TREE)).analyze_test_impact(
            _make_container(), ["src/auth.py", "README.md", "src/ghost.py"]
        )
        resp = TestImpactAnalysisResponse(**result)
        assert resp.status == "computed"
        assert resp.unsupported_changed_files == ["README.md"]
        assert resp.unknown_changed_files == ["src/ghost.py"]


# ---------------------------------------------------------------------------
# select_smart_tests (derived from analyze_test_impact)
# ---------------------------------------------------------------------------


class TestSelectSmartTests:
    async def test_selection_comes_from_impact(self):
        result = await _svc(FakeExec(EMS_TREE)).select_smart_tests(
            _make_container(), ["src/auth.py"]
        )
        assert result["selection_strategy"] == "impact_based"
        assert result["selected_tests"] == ["tests/test_auth_x.py", "tests/test_api.py"]
        assert result["selection_count"] == 2
        assert result["computed"] is True
        assert result["reason"] is None
        # No timing/coverage data exists -> not estimated (was 2s/test, ratio%).
        assert result["estimated_execution_time"] is None
        assert result["coverage_estimate"] is None

    async def test_max_tests_limit(self):
        result = await _svc(FakeExec(EMS_TREE)).select_smart_tests(
            _make_container(), ["src/auth.py"], selection_options={"max_tests": 1}
        )
        assert result["selected_tests"] == ["tests/test_auth_x.py"]

    async def test_not_computed_propagates(self):
        fake = FakeExec(EMS_TREE, overrides={"grep": {"exit_code": 0, "truncated": True}})
        result = await _svc(fake).select_smart_tests(_make_container(), ["src/auth.py"])
        assert result["selected_tests"] == []
        assert result["computed"] is False
        assert "truncated" in result["reason"]

    async def test_empty_changed_files(self):
        from models.intelligent_testing_models import SmartTestSelectionResponse

        result = await _svc(FakeExec(EMS_TREE)).select_smart_tests(_make_container(), [])
        assert result["selection_count"] == 0
        assert result["selected_tests"] == []
        SmartTestSelectionResponse(**result)


# ---------------------------------------------------------------------------
# prioritize_tests
# ---------------------------------------------------------------------------


class TestPrioritizeTests:
    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_basic_prioritization(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        test_files = ["test_a.py", "test_b.py"]
        result = await svc.prioritize_tests(container, test_files)

        assert result["total_tests"] == 2
        assert len(result["prioritized_tests"]) == 2
        # Should be sorted descending by priority_score
        scores = [t["priority_score"] for t in result["prioritized_tests"]]
        assert scores == sorted(scores, reverse=True)
        # Default criteria: impact + failure_history + execution_time
        assert result["criteria_used"] == ["impact", "failure_history", "execution_time"]
        # Constant failure_history/execution_time addends: flagged, not hidden.
        assert result["computed"] is False
        assert result["reason"] == "not implemented: heuristic placeholder"

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_impact_criterion_boosts_related(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        test_files = ["test_app.py", "test_other.py"]
        options = {
            "prioritization_criteria": ["impact"],
            "changed_files": ["app"],
        }
        result = await svc.prioritize_tests(container, test_files, options)
        # test_app.py contains "app" -> gets impact boost of 5.0
        # test_other.py does not match -> 0.0
        top = result["prioritized_tests"][0]
        assert top["test_file"] == "test_app.py"
        assert top["priority_score"] == 5.0
        bottom = result["prioritized_tests"][1]
        assert bottom["test_file"] == "test_other.py"
        assert bottom["priority_score"] == 0.0


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


class TestHelperMethods:
    @patch("services.intelligent_testing_service.get_docker_service")
    def test_calculate_impact_score_empty(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        assert svc._calculate_impact_score([], ["a.py"]) == 0.0

    @patch("services.intelligent_testing_service.get_docker_service")
    def test_calculate_impact_score(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        impacts = [{"test_file": "t1"}, {"test_file": "t2"}]
        score = svc._calculate_impact_score(impacts, ["a.py"])
        # (2/1)*5 = 10 -> capped at 10
        assert score == 10.0

    @patch("services.intelligent_testing_service.get_docker_service")
    def test_calculate_impact_score_proportional(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        impacts = [{"test_file": "t1"}]
        score = svc._calculate_impact_score(impacts, ["a.py", "b.py"])
        # (1/2)*5 = 2.5
        assert score == 2.5

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_calculate_test_priority(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        score = await svc._calculate_test_priority(
            container,
            "test_app.py",
            ["impact", "failure_history", "execution_time"],
            ["app"],
        )
        # impact=5 (matches), failure_history=2, execution_time=1 -> 8
        assert score == 8.0

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_calculate_test_priority_no_match(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        score = await svc._calculate_test_priority(
            container,
            "test_other.py",
            ["impact", "failure_history", "execution_time"],
            ["app"],
        )
        # impact=0 (no match), failure_history=2, execution_time=1 -> 3
        assert score == 3.0

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_select_tests_by_strategy_impact(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        impact_analysis = {
            "impacted_tests": [
                {"test_file": "low.py", "impact_score": 1},
                {"test_file": "high.py", "impact_score": 10},
            ]
        }
        result = await svc._select_tests_by_strategy(
            container, [], impact_analysis, "impact_based", None
        )
        assert result == ["high.py", "low.py"]

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_select_tests_by_strategy_default(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        impact_analysis = {
            "impacted_tests": [
                {"test_file": "a.py", "impact_score": 1},
                {"test_file": "b.py", "impact_score": 5},
            ]
        }
        result = await svc._select_tests_by_strategy(
            container, [], impact_analysis, "random_strategy", None
        )
        # Default strategy: keep order
        assert result == ["a.py", "b.py"]

    @patch("services.intelligent_testing_service.get_docker_service")
    async def test_select_tests_with_max(self, mock_docker):
        from services.intelligent_testing_service import IntelligentTestingService

        svc = IntelligentTestingService()
        container = _make_container()
        impact_analysis = {
            "impacted_tests": [{"test_file": f"t{i}.py", "impact_score": i} for i in range(10)]
        }
        result = await svc._select_tests_by_strategy(
            container, [], impact_analysis, "impact_based", 3
        )
        assert len(result) == 3
        # Top 3 by impact_score descending: t9, t8, t7
        assert result == ["t9.py", "t8.py", "t7.py"]
