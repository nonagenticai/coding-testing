"""
Intelligent Testing Service for Container-Scoped Operations.

This service provides container-scoped intelligent testing capabilities including:
- Pattern recognition and analysis
- Smart test recommendations
- Test impact analysis
- Test selection and prioritization
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from coding_client.clients import CodingWorkspacesClient, ServiceError

from auth.outbound import optional_tenant_headers, service_token
from config.settings import get_settings
from services.python_import_graph import (
    PRUNED_DIRS,
    build_import_graph,
    compute_test_impacts,
    parse_find_output,
)
from services.stubs import get_docker_service
from shared.container_validation import ContainerInfo

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = "/workspace"
EXEC_TIMEOUT_SECONDS = 60
NOT_IMPLEMENTED_REASON = "not implemented: heuristic placeholder"

# (container_name, tenant_id, argv, timeout) -> coding-containers ExecResponse
WorkspaceExec = Callable[[str, str | None, list[str], int], Awaitable[Any]]


class WorkspaceReadError(RuntimeError):
    """The workspace could not be read completely; analysis is not computed."""


async def _containers_exec(
    container_name: str, tenant_id: str | None, argv: list[str], timeout: int
) -> Any:
    """Run ``argv`` in the workspace via coding-containers exec."""
    containers_url = get_settings().CONTAINERS_SERVICE_URL
    if not containers_url:
        raise WorkspaceReadError("CONTAINERS_SERVICE_URL is not configured")
    headers = optional_tenant_headers(tenant_id, service_token=service_token())
    # Client read timeout a touch looser than the server-side command budget so
    # coding-containers' own timeout fires first and returns a real exit code.
    client = CodingWorkspacesClient(base_url=containers_url, timeout=float(timeout) + 10.0)
    try:
        return await client.exec(
            container_name,
            headers=headers,
            command=argv,
            workdir=WORKSPACE_ROOT,
            timeout=timeout,
        )
    finally:
        await client.close()


def _find_argv() -> list[str]:
    prune: list[str] = ["("]
    for i, d in enumerate(PRUNED_DIRS):
        if i:
            prune.append("-o")
        prune += ["-name", d]
    prune.append(")")
    return [
        "find",
        ".",
        "-type",
        "d",
        *prune,
        "-prune",
        "-o",
        "-type",
        "f",
        "-name",
        "*.py",
        "-print",
    ]


def _grep_argv() -> list[str]:
    return [
        "grep",
        "-rnE",
        "--include=*.py",
        *[f"--exclude-dir={d}" for d in PRUNED_DIRS],
        "^[[:space:]]*(from|import)[[:space:]]+[A-Za-z_.]",
        ".",
    ]


class IntelligentTestingService:
    """Service for container-scoped intelligent testing operations."""

    def __init__(self, workspace_exec: WorkspaceExec | None = None):
        """Initialize the intelligent testing service.

        Args:
            workspace_exec: Injectable exec transport (tests pass a fake);
                defaults to coding-containers ``POST /api/workspaces/{name}/exec``.
        """
        self.docker_service = get_docker_service()
        self._workspace_exec: WorkspaceExec = workspace_exec or _containers_exec
        logger.info("IntelligentTestingService initialized")

    async def analyze_test_patterns(
        self,
        container: ContainerInfo,
        test_results: list[dict[str, Any]],
        analysis_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze test patterns within a container context.

        Args:
            container: Container information
            test_results: Historical test results for pattern detection
            analysis_options: Optional analysis configuration

        Returns:
            Dictionary containing detected patterns and analysis results
        """
        logger.info(f"Analyzing test patterns for container '{container.container_name}'")

        options = analysis_options or {}
        include_performance = options.get("include_performance_patterns", True)
        include_failure = options.get("include_failure_patterns", True)
        include_timing = options.get("include_timing_patterns", True)

        patterns_detected = []
        performance_patterns = []
        failure_patterns = []
        timing_patterns = []

        # Analyze performance patterns
        if include_performance and test_results:
            performance_patterns = await self._detect_performance_patterns(container, test_results)
            patterns_detected.extend(performance_patterns)

        # Analyze failure patterns
        if include_failure and test_results:
            failure_patterns = await self._detect_failure_patterns(container, test_results)
            patterns_detected.extend(failure_patterns)

        # Analyze timing patterns
        if include_timing and test_results:
            timing_patterns = await self._detect_timing_patterns(container, test_results)
            patterns_detected.extend(timing_patterns)

        confidence_score = self._calculate_pattern_confidence(patterns_detected)

        return {
            "container_name": container.container_name,
            "patterns_detected": patterns_detected,
            "pattern_count": len(patterns_detected),
            "performance_patterns": performance_patterns,
            "failure_patterns": failure_patterns,
            "timing_patterns": timing_patterns,
            "confidence_score": confidence_score,
            # Performance/failure patterns are counts over the caller's own
            # test_results; timing detection does not exist and the confidence
            # is a pattern-count heuristic, so the response says so.
            "computed": True,
            "reason": (
                "partial: timing patterns not implemented; "
                "confidence_score is a pattern-count heuristic, not a calibrated probability"
            ),
            "analyzed_at": datetime.now(UTC),
        }

    async def generate_test_recommendations(
        self,
        container: ContainerInfo,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate intelligent test recommendations for a container.

        Args:
            container: Container information
            context: Optional context for better recommendations

        Returns:
            Dictionary containing test recommendations
        """
        logger.info(f"Generating test recommendations for container '{container.container_name}'")

        # There is no recommendation engine behind this method. It used to
        # return one hard-coded "Add integration tests for API endpoints"
        # (confidence 0.85) for every workspace regardless of its contents —
        # an invented finding a conveyor worker could act on. It now returns
        # nothing and says why.
        return {
            "container_name": container.container_name,
            "recommendations": [],
            "recommendation_count": 0,
            "high_priority_count": 0,
            "medium_priority_count": 0,
            "low_priority_count": 0,
            "computed": False,
            "reason": NOT_IMPLEMENTED_REASON,
            "generated_at": datetime.now(UTC),
        }

    async def select_smart_tests(
        self,
        container: ContainerInfo,
        changed_files: list[str],
        selection_options: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Select tests from the real import-graph impact analysis.

        Every selected test is an impacted test from :meth:`analyze_test_impact`
        (so it exists in the workspace). When that analysis is not computed the
        selection is empty and ``computed`` is False. Execution-time and
        coverage estimates have no data source and are returned as None.

        Args:
            container: Container information
            changed_files: List of files that have changed
            selection_options: Optional selection configuration
            tenant_id: Tenant used to address the workspace exec

        Returns:
            Dictionary containing selected tests and analysis
        """
        logger.info(
            f"Performing smart test selection for container '{container.container_name}' "
            f"with {len(changed_files)} changed files"
        )

        options = selection_options or {}
        strategy = options.get("selection_strategy", "impact_based")
        max_tests = options.get("max_tests")
        include_related = options.get("include_related_tests", True)

        # Perform impact analysis
        impact_analysis = await self.analyze_test_impact(
            container,
            changed_files,
            {"include_indirect_impacts": include_related},
            tenant_id=tenant_id,
        )

        # Select tests based on strategy
        selected_tests = await self._select_tests_by_strategy(
            container, changed_files, impact_analysis, strategy, max_tests
        )

        computed = impact_analysis.get("status") == "computed"
        return {
            "container_name": container.container_name,
            "selected_tests": selected_tests,
            "selection_count": len(selected_tests),
            "selection_strategy": strategy,
            "impact_analysis": impact_analysis,
            # No timing history or coverage data exists to estimate from; the
            # old "2s per test" and "tests/changed ratio" figures were invented.
            "estimated_execution_time": None,
            "coverage_estimate": None,
            "computed": computed,
            "reason": None if computed else impact_analysis.get("reason"),
            "selected_at": datetime.now(UTC),
        }

    async def analyze_test_impact(
        self,
        container: ContainerInfo,
        changed_files: list[str],
        analysis_options: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze the impact of code changes on tests from the real import graph.

        Reads the live workspace through coding-containers exec (one ``find``
        for ``*.py``, one ``grep`` for import lines) and resolves the imports
        to workspace files (see :mod:`services.python_import_graph`). Direct
        impacts are test files importing a changed file; indirect ones reach it
        through up to 3 intermediate modules (``via``). Every returned
        ``test_file`` exists in the workspace listing.

        If the workspace cannot be read completely (exec error, timeout,
        non-0/1 exit, truncated output) the result has ``status:
        "not_computed"`` with a ``reason`` and empty lists — never guessed
        paths. Changed Python paths absent from the workspace are listed in
        ``unknown_changed_files``; non-Python files in
        ``unsupported_changed_files``.

        Args:
            container: Container information
            changed_files: List of files that have changed
            analysis_options: Optional analysis configuration
            tenant_id: Tenant used to address the workspace exec (falls back to
                the container's owning tenant)

        Returns:
            Dictionary containing impact analysis results
        """
        logger.info(
            f"Analyzing test impact for container '{container.container_name}' with {len(changed_files)} changed files"
        )

        options = analysis_options or {}
        include_indirect = options.get("include_indirect_impacts", True)

        result: dict[str, Any] = {
            "container_name": container.container_name,
            "changed_files": changed_files,
            "impacted_tests": [],
            "impact_count": 0,
            "direct_impacts": 0,
            "indirect_impacts": 0,
            "impact_score": 0.0,
            "status": "computed",
            "reason": None,
            "unknown_changed_files": [],
            "unsupported_changed_files": [],
            "analyzed_at": datetime.now(UTC),
        }

        python_changed = [f for f in changed_files if f.strip().endswith(".py")]
        files: set[str] = set()
        grep_stdout = ""
        if python_changed:
            exec_tenant = container.tenant_id or tenant_id
            try:
                files = parse_find_output(
                    await self._read_workspace(container, exec_tenant, _find_argv(), "find"),
                    WORKSPACE_ROOT,
                )
                grep_stdout = await self._read_workspace(
                    container, exec_tenant, _grep_argv(), "grep", ok_exit_codes=(0, 1)
                )
            except WorkspaceReadError as e:
                logger.warning("Test impact not computed for '%s': %s", container.container_name, e)
                result["status"] = "not_computed"
                result["reason"] = str(e)
                return result

        graph = build_import_graph(files, grep_stdout, WORKSPACE_ROOT)
        impacts = compute_test_impacts(
            graph, changed_files, include_indirect=include_indirect, workdir=WORKSPACE_ROOT
        )
        all_impacts = impacts["direct"] + impacts["indirect"]
        result.update(
            impacted_tests=all_impacts,
            impact_count=len(all_impacts),
            direct_impacts=len(impacts["direct"]),
            indirect_impacts=len(impacts["indirect"]),
            impact_score=self._calculate_impact_score(all_impacts, changed_files),
            unknown_changed_files=impacts["unknown_changed_files"],
            unsupported_changed_files=impacts["unsupported_changed_files"],
        )
        return result

    async def _read_workspace(
        self,
        container: ContainerInfo,
        tenant_id: str | None,
        argv: list[str],
        label: str,
        ok_exit_codes: tuple[int, ...] = (0,),
    ) -> str:
        """Run a read-only listing command; raise WorkspaceReadError unless complete.

        Only *ok_exit_codes* count as a complete read: grep passes (0, 1) since 1
        means "no match", but find exits 1 when it could not descend into a
        directory, i.e. a PARTIAL listing. Anything else — including 124
        (timeout) and a truncated stream — means we did not see the whole
        workspace, and a partial import graph silently drops impacted tests.
        """
        try:
            payload = await self._workspace_exec(
                container.container_name, tenant_id, argv, EXEC_TIMEOUT_SECONDS
            )
        except (ServiceError, httpx.HTTPError) as e:
            raise WorkspaceReadError(f"{label} exec failed: {e}") from e
        if not isinstance(payload, dict):
            raise WorkspaceReadError(f"{label} exec returned no result")
        exit_code = payload.get("exit_code")
        if payload.get("truncated"):
            raise WorkspaceReadError(
                f"{label} output truncated ({payload.get('bytes_omitted', 0)} bytes omitted)"
            )
        if exit_code == 124:
            raise WorkspaceReadError(f"{label} timed out")
        if exit_code not in ok_exit_codes:
            raise WorkspaceReadError(
                f"{label} exited {exit_code}: {payload.get('stderr', '')[:200]}"
            )
        return str(payload.get("stdout") or "")

    async def prioritize_tests(
        self,
        container: ContainerInfo,
        test_files: list[str],
        prioritization_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Prioritize tests based on various criteria.

        Args:
            container: Container information
            test_files: List of test files to prioritize
            prioritization_options: Optional prioritization configuration

        Returns:
            Dictionary containing prioritized tests
        """
        logger.info(
            f"Prioritizing {len(test_files)} tests for container '{container.container_name}'"
        )

        options = prioritization_options or {}
        criteria = options.get(
            "prioritization_criteria", ["impact", "failure_history", "execution_time"]
        )
        changed_files = options.get("changed_files", [])

        # Calculate priority scores for each test
        prioritized_tests = []
        for test_file in test_files:
            priority_score = await self._calculate_test_priority(
                container, test_file, criteria, changed_files
            )
            prioritized_tests.append(
                {
                    "test_file": test_file,
                    "priority_score": priority_score,
                    "criteria": criteria,
                }
            )

        # Sort by priority score (descending)
        prioritized_tests.sort(key=lambda x: x["priority_score"], reverse=True)

        return {
            "container_name": container.container_name,
            "prioritized_tests": prioritized_tests,
            "total_tests": len(prioritized_tests),
            "criteria_used": criteria,
            # failure_history / execution_time add constants (no history is
            # stored) and "impact" is a path-substring match, not the import
            # graph — the ordering is a heuristic, and the response says so.
            "computed": False,
            "reason": NOT_IMPLEMENTED_REASON,
            "prioritized_at": datetime.now(UTC),
        }

    # Private helper methods

    async def _detect_performance_patterns(
        self, container: ContainerInfo, test_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect performance-related patterns in test results."""
        patterns = []

        # Example: Detect slow tests
        slow_threshold = 5.0  # seconds
        slow_tests = [r for r in test_results if r.get("execution_time", 0) > slow_threshold]

        if slow_tests:
            patterns.append(
                {
                    "pattern_type": "performance",
                    "category": "slow_tests",
                    "description": f"Detected {len(slow_tests)} slow tests (>{slow_threshold}s)",
                    "severity": "medium",
                    "test_count": len(slow_tests),
                }
            )

        return patterns

    async def _detect_failure_patterns(
        self, container: ContainerInfo, test_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect failure-related patterns in test results."""
        patterns = []

        # Example: Detect flaky tests
        # This would require historical data to identify tests that fail intermittently
        failed_tests = [r for r in test_results if not r.get("passed", True)]

        if failed_tests:
            patterns.append(
                {
                    "pattern_type": "failure",
                    "category": "test_failures",
                    "description": f"Detected {len(failed_tests)} failing tests",
                    "severity": "high",
                    "test_count": len(failed_tests),
                }
            )

        return patterns

    async def _detect_timing_patterns(
        self, container: ContainerInfo, test_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect timing-related patterns in test results."""
        patterns = []

        # Not implemented: no timing detection exists. analyze_test_patterns
        # reports this in its ``reason`` rather than implying "no patterns".

        return patterns

    def _calculate_pattern_confidence(self, patterns: list[dict[str, Any]]) -> float:
        """Calculate overall confidence score for pattern detection."""
        if not patterns:
            return 0.0

        # Simple confidence calculation based on pattern count and severity
        # More sophisticated ML-based confidence could be implemented
        base_confidence = min(1.0, len(patterns) * 0.2)
        return round(base_confidence, 2)

    async def _select_tests_by_strategy(
        self,
        container: ContainerInfo,
        changed_files: list[str],
        impact_analysis: dict[str, Any],
        strategy: str,
        max_tests: int | None,
    ) -> list[str]:
        """Select tests based on the specified strategy."""
        impacted_tests = impact_analysis.get("impacted_tests", [])

        if strategy == "impact_based":
            # Select tests based on impact score
            sorted_tests = sorted(
                impacted_tests, key=lambda x: x.get("impact_score", 0), reverse=True
            )
        else:
            # Default: all impacted tests
            sorted_tests = impacted_tests

        # Extract test file paths
        selected = [t.get("test_file") for t in sorted_tests if t.get("test_file")]

        # Limit if max_tests specified
        if max_tests and len(selected) > max_tests:
            selected = selected[:max_tests]

        return selected

    def _calculate_impact_score(
        self, impacts: list[dict[str, Any]], changed_files: list[str]
    ) -> float:
        """Calculate overall impact score."""
        if not impacts:
            return 0.0

        # Simple scoring based on impact count and file count
        score = min(10.0, (len(impacts) / max(1, len(changed_files))) * 5.0)
        return round(score, 2)

    async def _calculate_test_priority(
        self,
        container: ContainerInfo,
        test_file: str,
        criteria: list[str],
        changed_files: list[str],
    ) -> float:
        """Calculate priority score for a test file."""
        score = 0.0

        # Apply each criterion
        for criterion in criteria:
            if criterion == "impact":
                # Higher score if test relates to changed files
                if any(cf in test_file for cf in changed_files):
                    score += 5.0
            elif criterion == "failure_history":
                # Heuristic placeholder (no failure history stored); flagged via
                # computed=False in prioritize_tests.
                score += 2.0
            elif criterion == "execution_time":
                # Heuristic placeholder (no timing history stored); flagged via
                # computed=False in prioritize_tests.
                score += 1.0

        return round(score, 2)


# Service instance management
_service_instance: IntelligentTestingService | None = None


def get_intelligent_testing_service() -> IntelligentTestingService:
    """Get or create the intelligent testing service instance."""
    global _service_instance

    if _service_instance is None:
        _service_instance = IntelligentTestingService()

    return _service_instance
