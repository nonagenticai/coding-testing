"""pytest-cov coverage of a python workspace run as one :class:`CoverageReport`.

The run's pytest gets ``--cov --cov-fail-under=0 --cov-report=json:<run_dir>/coverage.json``
(``--cov-fail-under=0``: a project's own ``fail_under`` must not turn a measured run
red). The raw report never leaves the workspace: ``services.coverage_summary`` reduces
it there (:func:`summary_argv`) and :func:`parse_coverage` maps the summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from models.unified_test_models import CoverageMetric, CoverageReport, FileCoverage
from services.coverage_summary import MISSING_EXIT

MAX_FILES = 2000
_SUMMARY_PY = (Path(__file__).resolve().parent / "coverage_summary.py").read_text(encoding="utf-8")


def coverage_args(path: str) -> list[str]:
    """pytest arguments that write coverage.py's JSON report to ``path``."""
    return ["--cov", "--cov-fail-under=0", f"--cov-report=json:{path}"]


def summary_argv(path: str, workdir: str) -> list[str]:
    """Exec argv that prints the summary of the report at ``path``."""
    return ["python3", "-I", "-c", _SUMMARY_PY, path, workdir, str(MAX_FILES)]


def _metric(value: Any) -> CoverageMetric | None:
    if value is None:
        return None
    covered, total = value
    return CoverageMetric(
        covered=covered, total=total, percent=round(covered * 100.0 / total, 2) if total else None
    )


def parse_coverage(exit_code: int, stdout: str, stderr: str) -> CoverageReport:
    """The summary exec's output as a report; ``computed`` false (with why) otherwise."""
    if exit_code == MISSING_EXIT:
        return CoverageReport(
            computed=False,
            error=f"{stderr.strip()} (is pytest-cov in the project's dev dependencies?)",
        )
    if exit_code != 0:
        return CoverageReport(
            computed=False, error=f"coverage summary failed (exit {exit_code}): {stderr[-300:]}"
        )
    try:
        doc = json.loads(stdout)
        totals = doc["totals"]
        files = [
            FileCoverage(
                path=path,
                lines=_metric(m.get("lines")),
                functions=_metric(m.get("functions")),
                branches=_metric(m.get("branches")),
            )
            for path, m in doc["files"]
        ]
        return CoverageReport(
            computed=True,
            lines=_metric(totals["lines"]),
            functions=_metric(totals.get("functions")),
            branches=_metric(totals.get("branches")),
            files=files,
            files_clipped=doc["files_total"] - len(files),
        )
    except (ValueError, KeyError, TypeError, ValidationError) as e:
        return CoverageReport(computed=False, error=f"coverage summary unreadable: {e}"[:1000])
