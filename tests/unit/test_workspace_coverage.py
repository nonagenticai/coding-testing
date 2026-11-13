"""Coverage on python workspace test runs: the REAL pytest + pytest-cov run in a local
workspace (tests/local_workspace.py), the REAL in-workspace summary, the real grade."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from models.unified_test_models import WorkspaceTestRunRequest
from services import workspace_test_runner as wtr
from services.coverage_report import parse_coverage
from services.workspace_test_runner import WorkspaceTestRunner
from tests.local_workspace import LocalWorkspace

pytestmark = pytest.mark.usefixtures("pg_db")


def _project(root: Path, *, cov_dep: bool = True) -> None:
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "demo"
            version = "0"
            [tool.coverage.run]
            source = ["demo"]
            branch = true
            [tool.coverage.report]
            fail_under = 99
            """
        )
    )
    (root / "demo.py").write_text(
        textwrap.dedent(
            """
            def used(x):
                if x:
                    return 1
                return 2

            def unused():
                return 3
            """
        )
    )
    (root / "test_demo.py").write_text(
        "import demo\n\ndef test_used():\n    assert demo.used(1) == 1\n"
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> LocalWorkspace:
    root = tmp_path / "ws"
    root.mkdir()
    _project(root)
    monkeypatch.setattr(wtr, "RUN_ROOT", str(tmp_path / "runs"))
    return LocalWorkspace(root)


async def test_python_run_reports_coverage(workspace) -> None:
    req = WorkspaceTestRunRequest(coverage=True)
    run = await WorkspaceTestRunner(workspace).run("ws1", "t1", req)
    r = run.result
    assert r is not None and r.success, r.error if r else run
    cov = r.coverage
    assert cov is not None and cov.computed, cov
    # demo.py: 6 statements, `return 2` and `return 3` never ran (fail_under=99 is ignored).
    assert cov.lines is not None and (cov.lines.covered, cov.lines.total) == (4, 6)
    assert cov.functions is not None and (cov.functions.covered, cov.functions.total) == (1, 2)
    assert cov.branches is not None and cov.branches.covered < cov.branches.total
    assert [f.path for f in cov.files] == ["demo.py"] and cov.files_clipped == 0


async def test_run_without_coverage_has_none(workspace) -> None:
    run = await WorkspaceTestRunner(workspace).run("ws1", "t1", WorkspaceTestRunRequest())
    assert run.result is not None and run.result.success and run.result.coverage is None


async def test_missing_report_fails_the_run(workspace, monkeypatch) -> None:
    # Negative control: the report path is never written (pytest-cov told nothing).
    monkeypatch.setattr(wtr, "coverage_args", lambda path: [])
    run = await WorkspaceTestRunner(workspace).run(
        "ws1", "t1", WorkspaceTestRunRequest(coverage=True)
    )
    r = run.result
    assert r is not None and not r.success
    assert r.coverage is not None and not r.coverage.computed
    assert "no coverage report" in (r.error or "") and "pytest-cov" in (r.error or "")


def test_unreadable_summary_is_not_computed() -> None:
    assert not parse_coverage(0, "not json", "").computed
    assert "exit 1" in (parse_coverage(1, "", "boom").error or "")


def test_rust_coverage_is_the_runner_preset() -> None:
    assert WorkspaceTestRunRequest(coverage=True).rust_presets() == ["test", "coverage"]
    assert WorkspaceTestRunRequest().rust_presets() == ["test"]
