"""Request/response models for the performance comparison (source(s) vs target under load)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from models.parity_behaviour_models import (
    MAX_CORPORA,
    BehaviourCorpus,
    ServiceUnderTest,
    _check_corpora,
    _check_headers,
    _relative,
)

RunStatus = Literal["queued", "running", "completed", "failed"]


class PerfThresholds(BaseModel):
    """When the target counts as a regression against its source (target / source ratios)."""

    model_config = ConfigDict(extra="forbid")

    p50_ratio: float = Field(1.25, gt=0, description="Flag when target p50 > source p50 x this")
    p95_ratio: float = Field(1.25, gt=0, description="Same for p95")
    p99_ratio: float = Field(1.5, gt=0, description="Same for p99")
    throughput_ratio: float = Field(
        0.8, gt=0, description="Flag when target rps < source rps x this"
    )
    rss_ratio: float = Field(1.5, gt=0, description="Flag when target peak RSS > source x this")
    error_rate_delta: float = Field(
        0.0, ge=0, le=1, description="Flag when target error rate - source error rate > this"
    )
    min_route_samples: int = Field(
        20,
        ge=1,
        description="Fewest completed requests per side for a latency comparison; fewer "
        "makes the run not computed (throughput is compared regardless)",
    )


class PerfRequest(BaseModel):
    """Replay the corpora under load against every source and the single target.

    N sources -> 1 target: one corpus per source (each with its ``source``); the
    target is started fresh per corpus with the corpus's ``route_map``. Shorthand
    for one source: ``source`` + ``cases``/``cases_file``. The expected status per
    case: a corpus's ``recording`` / ``recording_run_id``, else the case's
    ``expected_status``, else what the source answered in its first warmup pass.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    target: ServiceUnderTest
    corpora: list[BehaviourCorpus] | None = Field(None, min_length=1, max_length=MAX_CORPORA)
    source: ServiceUnderTest | None = Field(None, description="Single-source shorthand")
    source_id: str = Field("source", description="Single-source shorthand: its id")
    cases_file: str | None = Field(None, max_length=1024)
    cases: list[dict[str, Any]] | None = Field(None, max_length=5000)
    case_ids: list[str] | None = Field(
        None, max_length=5000, description="Load only these case names (default: every HTTP case)"
    )
    concurrency: list[int] = Field(
        default_factory=lambda: [1, 8],
        min_length=1,
        max_length=8,
        description="Concurrency levels, each measured on its own",
    )
    duration_seconds: float | None = Field(
        None, gt=0, le=600, description="Per level. Default 10 when `requests` is not set"
    )
    requests: int | None = Field(
        None, ge=1, le=1_000_000, description="Per level (total over all workers)"
    )
    warmup_rounds: int = Field(
        1,
        ge=1,
        le=20,
        description="Sequential passes before measuring (the first is the correctness "
        "reference: statuses + upstream responses for the load phase)",
    )
    default_headers: dict[str, str] = Field(default_factory=dict)
    thresholds: PerfThresholds = Field(default_factory=PerfThresholds)
    request_timeout: int = Field(30, ge=1, le=600, description="Seconds per request")
    timeout: int = Field(3600, ge=1, le=14400, description="Seconds for the whole driver job")

    @field_validator("concurrency")
    @classmethod
    def _levels(cls, v: list[int]) -> list[int]:
        if any(c < 1 or c > 256 for c in v) or len(set(v)) != len(v):
            raise ValueError("concurrency levels are distinct integers in 1..256")
        return v

    @field_validator("cases_file")
    @classmethod
    def _cases_file(cls, v: str | None) -> str | None:
        return None if v is None else _relative(v, "cases_file")

    @field_validator("default_headers")
    @classmethod
    def _headers(cls, v: dict[str, str]) -> dict[str, str]:
        return _check_headers(v)

    @model_validator(mode="after")
    def _shape(self) -> PerfRequest:
        if self.duration_seconds is not None and self.requests is not None:
            raise ValueError("send at most one of duration_seconds or requests")
        if self.duration_seconds is None and self.requests is None:
            self.duration_seconds = 10.0
        if self.corpora is None:
            if self.source is None or (self.cases is None) == (self.cases_file is None):
                raise ValueError("send `corpora`, or `source` with exactly one of cases/cases_file")
            self.corpora = [
                BehaviourCorpus(
                    source_id=self.source_id,
                    cases=self.cases,
                    cases_file=self.cases_file,
                    source=self.source,
                )
            ]
        elif self.source is not None or self.cases is not None or self.cases_file is not None:
            raise ValueError("with corpora, source and cases go inside each corpus")
        for c in self.corpora:
            if c.source is None:
                raise ValueError(f"corpus {c.source_id!r} needs a source to compare against")
        _check_corpora(self.corpora)
        return self


class PerfStats(BaseModel):
    count: int = Field(0, description="Completed responses (any status)")
    errors: int = Field(0, description="status_mismatches + transport_errors")
    status_mismatches: int = Field(0, description="Responses whose status differs from the corpus")
    transport_errors: int = 0
    throughput_rps: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    max_ms: float | None = None
    mean_ms: float | None = None


class PerfLevel(BaseModel):
    concurrency: int
    requests_sent: int | None = Field(None, description="Requests started at this level")
    wall_s: float
    client_cpu_ratio: float
    load_generator_saturated: bool = Field(
        ..., description="The (GIL-bound) load generator used ~a whole core: rps is a floor"
    )
    overall: PerfStats
    routes: dict[str, PerfStats] = Field(default_factory=dict, description="By 'METHOD /path'")


class PerfSide(BaseModel):
    key: str = ""
    role: str | None = None
    source_id: str | None = None
    ok: bool = False
    error: str | None = None
    levels: list[PerfLevel] = Field(default_factory=list)
    peak_rss_kb: int | None = Field(
        None, description="Largest VmHWM over the service's process tree; null = unknown"
    )
    cases_used: list[str] = Field(default_factory=list)
    cases_skipped: list[dict[str, Any]] = Field(default_factory=list)
    warmup_status_mismatches: list[dict[str, Any]] = Field(
        default_factory=list, description="Sequential pass: cases whose status != the expected"
    )
    upstream_table_misses: int = 0
    service_exited: int | None = None
    log_tail: str = ""


class PerfComparison(BaseModel):
    source_id: str
    concurrency: int
    route: str = Field(..., description="'*' = all routes")
    source: PerfStats
    target: PerfStats
    throughput_ratio: float | None = Field(None, description="target / source")
    p50_ratio: float | None = None
    p95_ratio: float | None = None
    p99_ratio: float | None = None
    max_ratio: float | None = None
    source_error_rate: float | None = None
    target_error_rate: float | None = None
    saturated: bool = Field(
        False, description="The load generator was saturated on a side: no verdict"
    )
    under_sampled: bool = Field(
        False, description="A side completed fewer than min_route_samples: no latency verdict"
    )


class PerfRegression(BaseModel):
    source_id: str
    concurrency: int | None = Field(None, description="None for peak RSS")
    route: str
    metric: Literal["p50", "p95", "p99", "throughput", "rss", "error_rate"]
    source: float | None
    target: float | None
    ratio: float | None = None
    threshold: float


class PerfRss(BaseModel):
    source_id: str
    source_kb: int | None = None
    target_kb: int | None = None
    ratio: float | None = None


class PerfResult(BaseModel):
    computed: bool = Field(
        ...,
        description="Every side ran every level with enough samples; false = no verdict",
    )
    passed: bool = Field(False, description="computed and no regression")
    error: str | None = None
    sides: dict[str, PerfSide] = Field(default_factory=dict)
    comparisons: list[PerfComparison] = Field(default_factory=list)
    rss: list[PerfRss] = Field(default_factory=list)
    regressions: list[PerfRegression] = Field(default_factory=list)
    thresholds: PerfThresholds
    exit_code: int | None = None
    driver_log_tail: str = ""


class PerfRunLinks(BaseModel):
    status: str


class PerfRun(BaseModel):
    run_id: str
    container_name: str
    status: RunStatus
    result: PerfResult | None = None
    links: PerfRunLinks | None = None
