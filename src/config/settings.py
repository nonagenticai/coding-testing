"""Application settings using Pydantic BaseSettings.

All service URLs default to empty strings; the production validator (`model_post_init`)
raises if any required URL is missing. Real values come from the coding-testing
Deployment env block in gitops-production (clusters/strato/services/coding-testing/).

See `gitops-production/docs/config-externalization-policy.md` and the canonical
reference at `inference-router/src/settings.py`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

from coding_client.config import BaseMicroserviceSettings
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict


class Settings(BaseMicroserviceSettings):
    """Configuration for the coding-testing service.

    URLs intentionally default to "" — they MUST be provided by env in production.
    `model_post_init` raises in production if any `_REQUIRED_IN_PRODUCTION` field
    is empty. Tests / local dev set ENVIRONMENT to anything other than
    `"production"` to bypass the validator.
    """

    model_config = SettingsConfigDict(
        # Read raw env vars (no prefix) so existing Deployment env names
        # (PYTHON_TEST_SERVICE_URL, CONTAINERS_SERVICE_URL, …) keep working.
        env_prefix="",
        case_sensitive=True,
        extra="ignore",
    )

    SERVICE_NAME: str = "coding-testing"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    DATABASE_URL: str = ""
    REDIS_URL: str = ""

    # Test executor URLs — placeholders; real values come from
    # gitops-production coding-testing-config ConfigMap.
    PYTHON_TEST_SERVICE_URL: str = ""
    JAVASCRIPT_TEST_SERVICE_URL: str = ""
    QUALITY_ANALYSIS_SERVICE_URL: str = ""
    JAVA_TEST_SERVICE_URL: str = ""
    RUST_TEST_SERVICE_URL: str = ""
    CSHARP_TEST_SERVICE_URL: str = ""
    GO_TEST_SERVICE_URL: str = ""
    CPP_TEST_SERVICE_URL: str = ""

    # Rust test runs: one k8s Job per run (services/k8s_jobs.py). The Job pulls its
    # archive from TESTING_INTERNAL_URL + /internal/test-runs/{run_id}/archive. Unset
    # image or URL: rust runs fail with the reason (python runs are unaffected).
    # Names match gitops services/coding-testing (configmap + deployment).
    RUST_RUNNER_IMAGE: str = ""
    TEST_RUNS_NAMESPACE: str = "coding-test-runs"
    TESTING_INTERNAL_URL: str = ""
    RUNNER_CPU_REQUEST: str = "1"
    RUNNER_CPU_LIMIT: str = "4"
    RUNNER_MEMORY_REQUEST: str = "2Gi"
    RUNNER_MEMORY_LIMIT: str = "8Gi"
    # with_postgres rust runs: the runner Job's postgres:18 sidecar. Deployments that mirror
    # Docker Hub internally override this with their own registry.
    RUNNER_POSTGRES_IMAGE: str = "postgres:18"

    # Security service (coding-security).
    SECURITY_SERVICE_URL: str = ""

    # Cross-service URLs.
    CONTAINERS_SERVICE_URL: str = ""

    # License introspection (license-authz / licenses service).
    LICENSE_INTROSPECT_URL: str = ""

    # LLM plane — inference-router via coding-client BaseLLMAgent. coding_client
    # also reads these env vars directly; we surface them here for diagnostics.
    LLM_API_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    LLM_MODE: str = "inference_router"
    LLM_API_TIMEOUT_SECONDS: float = Field(default=300.0)
    # Must name inference-router PROVIDERS, not models. The historical
    # "gemma,deepseek,claude" default named zero usable pairs: `gemma` is a
    # MODEL served by the `pgx-local` provider and is not a provider name in the
    # router catalog at all, and the `claude` provider ships `enabled: false`.
    # Both were therefore dropped at resolution and the chain collapsed to the
    # single hop ["deepseek"] — a "fallback chain" with no fallback.
    # `pgx-local,deepseek` mirrors inference-router's live
    # `default_chain: [pgx-local, deepseek]` (see gitops-production
    # clusters/strato/services/inference-router/providers-configmap.yaml): free
    # local vLLM primary, paid DeepSeek failover while pgx-local is DEGRADED.
    LLM_ROUTING_DEFAULT_FALLBACK_CHAIN: str = "pgx-local,deepseek"

    # Orchestration self-heal: a suite/plan CAS-marked running/executing but never
    # cleared (pod crash / OOM / node loss mid-run) would otherwise stay wedged
    # forever, permanently blocking re-execution. A new execute attempt reclaims
    # the row once it has been stuck longer than this threshold. A live run
    # periodically renews `updated_at` via the lease heartbeat (heartbeat_suite /
    # heartbeat_plan), so only a genuinely dead run ever crosses the threshold —
    # this only needs to comfortably exceed the heartbeat interval plus some slack
    # (interval defaults to a quarter of this value). The 1h default stays safe.
    ORCHESTRATION_STALE_RUNNING_SECONDS: int = Field(default=3600)

    # POST /api/parity/schema-diff connects from THIS pod to two caller-supplied
    # DSNs. That is a local/test tool: in production it would let any caller make
    # coding-testing open connections to arbitrary hosts (pgbouncer included), so
    # it is allowed ONLY when this flag is set AND ENVIRONMENT is allowlisted
    # (development/dev/test/testing/local/ci; routers.parity_schema).
    # The workspace route (/api/parity/{container}/schema-diff) is unaffected.
    PARITY_DSN_DIFF_ENABLED: bool = False

    @field_validator("QUALITY_ANALYSIS_SERVICE_URL")
    @classmethod
    def _quality_url_is_a_base_url(cls, value: str) -> str:
        """Every caller appends ``/api/quality/...`` itself, so this must be the host base.

        The gitops ConfigMap carries ``.../api/quality`` and the callers produced
        ``/api/quality/api/quality/analyze-sync`` (403 from coding-analysis), so
        every quality-check and quality-scan answered 500. Accept both forms.
        """
        value = value.rstrip("/")
        if value.endswith("/api/quality"):
            value = value[: -len("/api/quality")]
        return value

    # Fields whose absence should fail loud at startup in production. Anything
    # that is purely diagnostic (LLM_API_KEY, optional integrations) stays out.
    # The shared base model_post_init enforces these; _ENV_ATTR points at the
    # uppercase ENVIRONMENT field.
    _SERVICE_NAME: ClassVar[str] = "coding-testing"
    _ENV_ATTR: ClassVar[str] = "ENVIRONMENT"
    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "DATABASE_URL",
        "REDIS_URL",
        "PYTHON_TEST_SERVICE_URL",
        "JAVASCRIPT_TEST_SERVICE_URL",
        "QUALITY_ANALYSIS_SERVICE_URL",
        "JAVA_TEST_SERVICE_URL",
        "RUST_TEST_SERVICE_URL",
        "CSHARP_TEST_SERVICE_URL",
        "GO_TEST_SERVICE_URL",
        "CPP_TEST_SERVICE_URL",
        "SECURITY_SERVICE_URL",
        "CONTAINERS_SERVICE_URL",
        "LICENSE_INTROSPECT_URL",
        "LLM_API_BASE_URL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
