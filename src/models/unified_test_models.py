"""One result format for a workspace test run, whatever the framework.

``UnifiedTestResult`` is what the conveyor's computed definition of done reads:
pytest (a Python source repo, run in the workspace and parsed from its JUnit XML)
and cargo-nextest (a Rust target repo, run by a throwaway Rust runner Job) both
land here, so "tests green" means the same thing on both sides of a rewrite.

The verdict is ``success``; ``passed``/``failed``/``skipped``/``errored`` are the
per-status COUNTS. A run that executed zero tests is never a success.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Framework = Literal["pytest", "cargo-nextest"]
WorkspaceLanguage = Literal["python", "rust"]
CaseStatus = Literal["passed", "failed", "skipped", "errored"]
RunStatus = Literal["queued", "running", "completed", "failed"]
#: Rust runner presets (a Rust run only).
RustPreset = Literal["test", "clippy", "fmt", "audit", "deny", "coverage"]
RUST_PRESETS: tuple[str, ...] = ("test", "clippy", "fmt", "audit", "deny", "coverage")

#: The checkout root inside every coding workspace.
WORKSPACE_ROOT = "/workspace"


class UnifiedTestCase(BaseModel):
    """One test case as the framework reported it."""

    name: str = Field(..., description="Fully qualified test name (classname::name)")
    status: CaseStatus
    message: str | None = Field(None, description="Failure/error/skip message, if any")
    duration: float | None = Field(None, description="Seconds, as reported by the framework")


class CoverageMetric(BaseModel):
    covered: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    percent: float | None = Field(None, description="covered / total x 100; null for total 0")


class FileCoverage(BaseModel):
    path: str = Field(..., description="Relative to the run's workdir when under it")
    lines: CoverageMetric | None = None
    functions: CoverageMetric | None = None
    branches: CoverageMetric | None = None


class CoverageReport(BaseModel):
    """pytest-cov coverage of one python run; a metric not measured is null."""

    tool: Literal["pytest-cov"] = "pytest-cov"
    computed: bool = Field(..., description="The report was written, read back and parsed")
    error: str | None = Field(None, description="Why no coverage was computed")
    lines: CoverageMetric | None = None
    functions: CoverageMetric | None = Field(
        None, description="Functions with at least one covered line (coverage.py >= 7.6)"
    )
    branches: CoverageMetric | None = Field(None, description="Only when branch coverage ran")
    files: list[FileCoverage] = Field(default_factory=list)
    files_clipped: int = Field(0, ge=0, description="Files beyond the per-run cap, not listed")


class UnifiedTestResult(BaseModel):
    """Framework-independent result of one workspace test run."""

    framework: Framework
    language: WorkspaceLanguage
    success: bool = Field(
        ...,
        description=(
            "The computed verdict: exit code 0, no failed or errored case, at least one "
            "case actually ran, and the report was read back complete"
        ),
    )
    passed: int = Field(0, ge=0, description="Count of passed cases")
    failed: int = Field(0, ge=0, description="Count of failed cases")
    skipped: int = Field(0, ge=0, description="Count of skipped/ignored cases")
    errored: int = Field(0, ge=0, description="Count of errored cases")
    total: int = Field(0, ge=0, description="passed + failed + skipped + errored")
    duration: float = Field(0.0, ge=0, description="Run duration in seconds")
    cases: list[UnifiedTestCase] = Field(default_factory=list)
    raw_tail: str = Field("", description="Tail of the command's stdout+stderr")
    exit_code: int | None = Field(None, description="Test command exit code (None: no exit)")
    command: list[str] = Field(
        default_factory=list, description="argv that ran (DATABASE_URL redacted)"
    )
    workdir: str = Field(WORKSPACE_ROOT, description="Absolute directory the command ran in")
    with_postgres: bool = False
    report_found: bool = Field(False, description="A JUnit report was read back and parsed")
    error: str | None = Field(None, description="Why the run is not a success, when known")
    coverage: CoverageReport | None = Field(
        None,
        description="python: set when the run asked for coverage (a requested but uncomputed "
        "coverage fails the run); rust coverage is the runner's `coverage` preset in `presets`",
    )
    presets: dict[str, Any] | None = Field(
        None,
        description="rust: the runner's per-preset documents (test, clippy, fmt, audit, deny, "
        "coverage) as reported",
    )
    tool_versions: dict[str, str] = Field(
        default_factory=dict, description="rust: tool versions the runner ran"
    )


class WorkspaceTestRunRequest(BaseModel):
    """Run the workspace's own test suite, language-dispatched.

    Every field is optional: with an empty body the language is detected from the
    checkout root (``Cargo.toml`` -> a Rust runner Job; ``pyproject.toml``/``uv.lock``
    -> pytest via uv in the workspace) and the run is synchronous.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    language: WorkspaceLanguage | None = Field(
        None, description="Force the language; omitted = detect from the workdir's manifests"
    )
    workdir: str = Field(
        ".",
        max_length=512,
        description="Checkout-relative subdirectory to run in (e.g. 'services/api')",
    )
    with_postgres: bool = Field(
        False,
        description=(
            "python: run `ws-pg` first (starts Postgres 18 on 127.0.0.1 in the workspace) "
            "and export the DATABASE_URL it prints to the test command; rust: the runner "
            "Job gets a postgres:18 sidecar and DATABASE_URL"
        ),
    )
    run_async: bool = Field(
        False,
        alias="async",
        description=(
            "Return 202 with a run_id immediately and run the suite as a coding-containers "
            "async exec job; poll GET .../runs/{run_id}"
        ),
    )
    timeout: int = Field(
        300,
        ge=1,
        le=14400,
        description="Seconds. Synchronous runs are capped at 300 (the exec route's ceiling)",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        max_length=32,
        description=(
            "Extra arguments appended to the test command: pytest (python), cargo nextest "
            "(rust, via the runner's NEXTEST_EXTRA_ARGS), e.g. ['-k', 'api'] / ['-E', 'test(foo)']"
        ),
    )
    coverage: bool = Field(
        False,
        description=(
            "Measure coverage: python -> pytest-cov in the workspace (`--cov`, JSON report "
            "reduced to totals + per file in `result.coverage`; the project needs pytest-cov); "
            "rust -> adds the runner's `coverage` preset (cargo-llvm-cov, in `result.presets`)"
        ),
    )
    presets: list[RustPreset] | None = Field(
        None,
        min_length=1,
        max_length=6,
        description=(
            "rust only: runner presets to run in addition to 'test' (always run). Each "
            "preset must be clean, and at least one test must pass, for success"
        ),
    )

    @field_validator("workdir")
    @classmethod
    def _workdir_relative(cls, v: str) -> str:
        v = v.strip() or "."
        if v.startswith(("/", "-")) or ".." in v.split("/") or "\x00" in v:
            raise ValueError("workdir must be a checkout-relative path without '..'")
        return posixpath.normpath(v)

    @field_validator("extra_args")
    @classmethod
    def _no_nul(cls, v: list[str]) -> list[str]:
        if any("\x00" in a or len(a) > 1024 for a in v):
            raise ValueError("extra_args entries must be <=1024 chars without NUL")
        return v

    def rust_presets(self) -> list[str]:
        """The presets of a Rust run, in a stable order, without duplicates; always ``test``."""
        chosen = {"test", *(self.presets or [])}  # test always runs: zero tests is never success
        if self.coverage:
            chosen.add("coverage")
        return [p for p in RUST_PRESETS if p in chosen]

    def absolute_workdir(self) -> str:
        """The workdir as an absolute path under the checkout root."""
        if self.workdir == ".":
            return WORKSPACE_ROOT
        return f"{WORKSPACE_ROOT}/{self.workdir}"


class WorkspaceTestRunLinks(BaseModel):
    status: str = Field(..., description="Path of this run's status route")


class WorkspaceTestRun(BaseModel):
    """A workspace test run: finished (sync) or a pollable async job."""

    run_id: str
    container_name: str
    status: RunStatus = Field(
        ..., description="queued | running | completed (result set) | failed (result set)"
    )
    result: UnifiedTestResult | None = Field(
        None, description="Set once the run has ended (status completed or failed)"
    )
    links: WorkspaceTestRunLinks | None = None


#: ``<nonce>.<coding-containers job id>`` for async runs (``<nonce>.rte`` for an async
#: Rust run, a task of this service); ``<nonce>`` for sync runs.
RUN_ID_RE = re.compile(r"^(?P<nonce>[0-9a-f]{32})(?:\.(?P<job>[A-Za-z0-9_-]{1,64}))?$")
