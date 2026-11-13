"""Tests for config.settings."""


def test_settings_defaults(settings):
    """Settings have correct identity defaults."""
    assert settings.SERVICE_NAME == "coding-testing"
    assert settings.VERSION == "1.0.0"


def test_settings_service_urls(settings):
    """Test executor URLs come from env (fixture sets matching values)."""
    assert "python-test-executor" in settings.PYTHON_TEST_SERVICE_URL
    assert "javascript-test-executor" in settings.JAVASCRIPT_TEST_SERVICE_URL
    assert "java-test-executor" in settings.JAVA_TEST_SERVICE_URL
    assert "rust-test-executor" in settings.RUST_TEST_SERVICE_URL


def test_settings_cross_service_urls(settings):
    """Cross-service URLs come from env."""
    assert "coding-containers" in settings.CONTAINERS_SERVICE_URL


def test_settings_llm_plane(settings):
    """Settings expose LLM-plane (inference-router) configuration."""
    assert "inference-router" in settings.LLM_API_BASE_URL
    assert settings.LLM_MODE == "inference_router"
    assert settings.LLM_API_TIMEOUT_SECONDS > 0
    assert settings.LLM_ROUTING_DEFAULT_FALLBACK_CHAIN == "pgx-local,deepseek"


def test_settings_url_defaults_are_empty_when_env_unset(monkeypatch):
    """When env vars are absent, all URL defaults are empty strings.

    The autouse `_test_env` fixture normally injects test URLs; we strip them
    here to verify the bare model has no hardcoded URL fallbacks.
    """
    from config.settings import Settings

    for var in (
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
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")
    s = Settings()
    assert s.PYTHON_TEST_SERVICE_URL == ""
    assert s.CONTAINERS_SERVICE_URL == ""
    assert s.SECURITY_SERVICE_URL == ""
    assert s.LICENSE_INTROSPECT_URL == ""
    assert s.LLM_API_BASE_URL == ""


def test_production_validator_raises_when_required_url_missing(monkeypatch):
    """ENVIRONMENT=production + missing required URL must fail loud at startup."""
    import pytest

    from config.settings import Settings

    for var in (
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
        "DATABASE_URL",
        "REDIS_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="missing required fields in production"):
        Settings()


def test_get_settings_returns_instance():
    """get_settings returns a Settings instance."""
    from config.settings import Settings, get_settings

    s = get_settings()
    assert isinstance(s, Settings)


# Providers declared in inference-router's catalog
# (gitops-production/clusters/strato/services/inference-router/providers-configmap.yaml).
# `gemma` is deliberately absent: it is a MODEL served by `pgx-local`, never a
# provider name. `claude` is present in the catalog but ships `enabled: false`.
ROUTER_ENABLED_PROVIDERS = frozenset({"pgx-local", "deepseek", "xai"})
ROUTER_DISABLED_PROVIDERS = frozenset({"claude"})


def test_default_fallback_chain_is_routable_multi_hop(monkeypatch):
    """The SHIPPED default must be >=2 distinct, real, ENABLED providers.

    Regression pin. The previous default "gemma,deepseek,claude" named zero
    usable pairs: `gemma` is not a provider name at all (it is the model
    pgx-local serves) and `claude` is `enabled: false`. Both were dropped at
    resolution, collapsing the chain to the single hop ["deepseek"] — no
    failover. model-registry now rejects such a chain at registration with
    HTTP 422, so a regression here breaks the nightly coding-llm-bootstrap Job.
    """
    from config.settings import Settings

    monkeypatch.delenv("LLM_ROUTING_DEFAULT_FALLBACK_CHAIN", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")
    raw = Settings().LLM_ROUTING_DEFAULT_FALLBACK_CHAIN
    chain = [hop.strip() for hop in raw.split(",") if hop.strip()]

    known = ROUTER_ENABLED_PROVIDERS | ROUTER_DISABLED_PROVIDERS
    unknown = [hop for hop in chain if hop not in known]
    assert not unknown, f"chain names non-providers (models?): {unknown}"

    disabled = [hop for hop in chain if hop in ROUTER_DISABLED_PROVIDERS]
    assert not disabled, f"chain names disabled providers: {disabled}"

    assert len(set(chain)) == len(chain), f"chain has duplicate hops: {chain}"
    assert len(set(chain)) >= 2, f"chain has no failover hop: {chain}"


def test_default_fallback_chain_matches_router_default_chain(monkeypatch):
    """Pin the exact value against inference-router's live `default_chain`."""
    from config.settings import Settings

    monkeypatch.delenv("LLM_ROUTING_DEFAULT_FALLBACK_CHAIN", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")
    assert Settings().LLM_ROUTING_DEFAULT_FALLBACK_CHAIN == "pgx-local,deepseek"


def test_quality_url_is_normalised_to_the_host_base(monkeypatch):
    """Callers append ``/api/quality/...``; the gitops value carries it too (doubled path -> 403)."""
    from config.settings import Settings

    monkeypatch.setenv("ENVIRONMENT", "test")
    base = "http://coding-analysis.coding-analysis.svc.cluster.local:8688"
    for raw in (base, base + "/", base + "/api/quality", base + "/api/quality/"):
        monkeypatch.setenv("QUALITY_ANALYSIS_SERVICE_URL", raw)
        assert Settings().QUALITY_ANALYSIS_SERVICE_URL == base
    monkeypatch.setenv("QUALITY_ANALYSIS_SERVICE_URL", "")
    assert Settings().QUALITY_ANALYSIS_SERVICE_URL == ""
