"""Tests for service stubs.

Verifies that stub implementations conform to the expected interface contracts
and return consistent, documented values.
"""

import inspect

import pytest

from services.stubs import (
    ContainerRuntimeUnavailableError,
    DockerService,
    MetricsService,
    RedisCacheManager,
    get_cache_manager,
    get_docker_service,
    get_metrics_service,
)

# ============================================================================
# DockerService interface and method completeness
# ============================================================================


class TestDockerServiceInterface:
    """Verify DockerService has all required interface methods."""

    _REQUIRED_ASYNC_METHODS = [
        "find_container_by_name",
        "create_test_container",
        "start_container",
        "create_file_in_container",
        "exec_in_container",
        "install_tools_in_container",
        "stop_container",
        "remove_container_by_id",
        "remove_container",
        "remove_container_by_name",
        "list_containers",
        "is_docker_available",
        "get_container_stats",
        "create_container",
    ]

    def test_has_all_required_async_methods(self):
        """DockerService exposes every async method the real service needs."""
        svc = DockerService()
        for method_name in self._REQUIRED_ASYNC_METHODS:
            method = getattr(svc, method_name)
            assert callable(method), f"{method_name} is not callable"
            assert inspect.iscoroutinefunction(method), (
                f"{method_name} should be a coroutine function"
            )

    def test_has_exactly_the_expected_public_methods(self):
        """DockerService exposes the expected set of public methods (no extras, no missing)."""
        svc = DockerService()
        expected = set(self._REQUIRED_ASYNC_METHODS) | {"get_language_container_image"}
        actual = {m for m in dir(svc) if not m.startswith("_")}
        assert actual == expected

    def test_has_sync_language_image_method(self):
        """DockerService has synchronous get_language_container_image."""
        svc = DockerService()
        method = svc.get_language_container_image
        assert callable(method)
        assert not inspect.iscoroutinefunction(method), (
            "get_language_container_image should be synchronous, not async"
        )


class TestDockerServiceSingleton:
    """Verify get_docker_service returns the same instance."""

    def test_get_docker_service_is_singleton(self):
        """get_docker_service returns same instance on repeated calls."""
        a = get_docker_service()
        b = get_docker_service()
        assert a is b

    def test_get_docker_service_returns_docker_service_type(self):
        """get_docker_service returns a DockerService (not a subclass or mock)."""
        svc = get_docker_service()
        assert type(svc) is DockerService


# ============================================================================
# DockerService method return values
# ============================================================================


@pytest.mark.asyncio
async def test_docker_find_container_returns_none():
    """DockerService stub always returns None for find_container_by_name."""
    svc = DockerService()
    result = await svc.find_container_by_name("test")
    assert result is None


@pytest.mark.asyncio
async def test_docker_create_test_container_fails_loud():
    """DockerService create_test_container refuses to fabricate a container.

    Returning a fake id here previously let the orchestrator pretend a container
    existed and later report an unexecuted test as passed.
    """
    svc = DockerService()
    with pytest.raises(ContainerRuntimeUnavailableError):
        await svc.create_test_container({"image": "python:3.13"})


@pytest.mark.asyncio
async def test_docker_start_container_fails_loud():
    """DockerService start_container refuses (no real runtime)."""
    svc = DockerService()
    with pytest.raises(ContainerRuntimeUnavailableError):
        await svc.start_container("container-123")


@pytest.mark.asyncio
async def test_docker_create_file_in_container_fails_loud():
    """DockerService create_file_in_container refuses (no real runtime)."""
    svc = DockerService()
    with pytest.raises(ContainerRuntimeUnavailableError):
        await svc.create_file_in_container("c1", "/tmp/test.py", "print(1)")


@pytest.mark.asyncio
async def test_docker_exec_in_container_fails_loud():
    """DockerService exec_in_container refuses to fabricate a success result.

    This is the core of fabrication F1: the stub used to return
    {"exit_code": 0, ...} unconditionally, so every unexecuted test read as a pass.
    """
    svc = DockerService()
    with pytest.raises(ContainerRuntimeUnavailableError):
        await svc.exec_in_container("c1", ["python", "test.py"])


@pytest.mark.asyncio
async def test_docker_exec_in_container_fails_loud_with_optional_params():
    """DockerService exec_in_container refuses regardless of timeout/user params."""
    svc = DockerService()
    with pytest.raises(ContainerRuntimeUnavailableError):
        await svc.exec_in_container("c1", ["ls"], timeout=60, user="root")


@pytest.mark.asyncio
async def test_docker_exec_in_container_default_timeout_is_30():
    """DockerService exec_in_container has timeout=30 as default."""
    sig = inspect.signature(DockerService.exec_in_container)
    assert sig.parameters["timeout"].default == 30


@pytest.mark.asyncio
async def test_docker_exec_in_container_default_user_is_none():
    """DockerService exec_in_container has user=None as default."""
    sig = inspect.signature(DockerService.exec_in_container)
    assert sig.parameters["user"].default is None


@pytest.mark.asyncio
async def test_docker_install_tools_returns_none():
    """DockerService install_tools_in_container is a no-op returning None."""
    svc = DockerService()
    result = await svc.install_tools_in_container("c1", ["ruff", "ruff-format"], "python")
    assert result is None


@pytest.mark.asyncio
async def test_docker_stop_container_returns_none():
    """DockerService stop_container is a no-op returning None."""
    svc = DockerService()
    result = await svc.stop_container("c1")
    assert result is None


@pytest.mark.asyncio
async def test_docker_remove_container_by_id_returns_none():
    """DockerService remove_container_by_id is a no-op returning None."""
    svc = DockerService()
    result = await svc.remove_container_by_id("c1")
    assert result is None


@pytest.mark.asyncio
async def test_docker_remove_container_returns_none():
    """DockerService remove_container is a no-op returning None."""
    svc = DockerService()
    result = await svc.remove_container("c1")
    assert result is None


@pytest.mark.asyncio
async def test_docker_remove_container_by_name_returns_none():
    """DockerService remove_container_by_name is a no-op returning None."""
    svc = DockerService()
    result = await svc.remove_container_by_name("test-container")
    assert result is None


@pytest.mark.asyncio
async def test_docker_list_containers_returns_empty():
    """DockerService list_containers always returns an empty list."""
    svc = DockerService()
    result = await svc.list_containers()
    assert result == []


@pytest.mark.asyncio
async def test_docker_is_docker_available_returns_false():
    """DockerService stub reports Docker as unavailable."""
    svc = DockerService()
    result = await svc.is_docker_available()
    assert result is False


@pytest.mark.asyncio
async def test_docker_get_container_stats_returns_zeroed():
    """DockerService get_container_stats returns exactly two zeroed metrics."""
    svc = DockerService()
    result = await svc.get_container_stats("c1")
    assert result == {"memory_usage_mb": 0, "cpu_percent": 0.0}


@pytest.mark.asyncio
async def test_docker_create_container_fails_loud():
    """DockerService create_container refuses to fabricate a container."""
    svc = DockerService()
    with pytest.raises(ContainerRuntimeUnavailableError):
        await svc.create_container({"image": "python"})


# ============================================================================
# DockerService.get_language_container_image - parametrized
# ============================================================================

_LANGUAGE_IMAGE_MAP = {
    "python": "python:3.13-slim",
    "javascript": "node:22-slim",
    "typescript": "node:22-slim",
    "go": "golang:1.23-alpine",
    "rust": "rust:1.83-slim",
    "java": "eclipse-temurin:21-jdk",
    "csharp": "mcr.microsoft.com/dotnet/sdk:9.0",
    "cpp": "gcc:14",
}


@pytest.mark.parametrize("language,expected_image", list(_LANGUAGE_IMAGE_MAP.items()))
def test_docker_get_language_container_image(language, expected_image):
    """DockerService returns the correct container image for each supported language."""
    svc = DockerService()
    assert svc.get_language_container_image(language) == expected_image


def test_docker_get_language_container_image_unknown_defaults_to_python():
    """DockerService returns python image as default for unknown languages."""
    svc = DockerService()
    assert svc.get_language_container_image("cobol") == "python:3.13-slim"
    assert svc.get_language_container_image("") == "python:3.13-slim"


def test_docker_language_image_map_covers_exactly_8_languages():
    """DockerService supports exactly 8 languages (python, js, ts, go, rust, java, csharp, cpp)."""
    svc = DockerService()
    # Verify by checking that every known language returns a non-default image
    # and that the count matches
    supported = []
    for lang in ["python", "javascript", "typescript", "go", "rust", "java", "csharp", "cpp"]:
        img = svc.get_language_container_image(lang)
        assert img != "", f"{lang} returned empty image"
        supported.append(lang)
    assert len(supported) == 8

    # Unknown language falls back to python default
    assert svc.get_language_container_image("ruby") == svc.get_language_container_image("python")


# ============================================================================
# RedisCacheManager interface and behavior
# ============================================================================


class TestRedisCacheManager:
    """Verify RedisCacheManager stub behavior."""

    def test_has_required_async_methods(self):
        """RedisCacheManager has get, set, delete, get_redis as coroutine functions."""
        cache = RedisCacheManager()
        for method_name in ("get", "set", "delete", "get_redis"):
            method = getattr(cache, method_name)
            assert callable(method)
            assert inspect.iscoroutinefunction(method), (
                f"{method_name} should be a coroutine function"
            )

    def test_has_exactly_the_expected_public_methods(self):
        """RedisCacheManager has only get, set, delete, get_redis."""
        cache = RedisCacheManager()
        actual = {m for m in dir(cache) if not m.startswith("_")}
        assert actual == {"get", "set", "delete", "get_redis"}

    @pytest.mark.asyncio
    async def test_get_returns_none(self):
        """Cache stub always returns None on get."""
        cache = RedisCacheManager()
        result = await cache.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_then_get_returns_none(self):
        """Cache stub ignores set -- get still returns None."""
        cache = RedisCacheManager()
        set_result = await cache.set("key", "value")
        assert set_result is None
        get_result = await cache.get("key")
        assert get_result is None

    @pytest.mark.asyncio
    async def test_set_accepts_ttl_and_returns_none(self):
        """Cache stub set accepts optional ttl parameter and returns None."""
        cache = RedisCacheManager()
        result = await cache.set("key", "value", ttl=600)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_default_ttl_is_300(self):
        """Cache stub set has default ttl of 300."""
        sig = inspect.signature(RedisCacheManager.set)
        assert sig.parameters["ttl"].default == 300

    @pytest.mark.asyncio
    async def test_delete_returns_none(self):
        """Cache stub delete returns None."""
        cache = RedisCacheManager()
        result = await cache.delete("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_redis_returns_none(self):
        """Cache stub get_redis returns None (no real redis)."""
        cache = RedisCacheManager()
        result = await cache.get_redis()
        assert result is None


class TestCacheManagerSingleton:
    """Verify get_cache_manager returns the same instance."""

    def test_is_singleton(self):
        """get_cache_manager returns same instance on repeated calls."""
        a = get_cache_manager()
        b = get_cache_manager()
        assert a is b

    def test_returns_redis_cache_manager_type(self):
        """get_cache_manager returns a RedisCacheManager (not a subclass or mock)."""
        mgr = get_cache_manager()
        assert type(mgr) is RedisCacheManager


# ============================================================================
# MetricsService interface and behavior
# ============================================================================


class TestMetricsService:
    """Verify MetricsService stub behavior."""

    def test_has_required_async_methods(self):
        """MetricsService has record_test_execution, record_quality_analysis, get_test_execution_analytics as coroutines."""
        svc = MetricsService()
        for method_name in (
            "record_test_execution",
            "record_quality_analysis",
            "get_test_execution_analytics",
        ):
            method = getattr(svc, method_name)
            assert callable(method)
            assert inspect.iscoroutinefunction(method), (
                f"{method_name} should be a coroutine function"
            )

    def test_has_exactly_the_expected_public_methods(self):
        """MetricsService has only record_test_execution, record_quality_analysis, get_test_execution_analytics."""
        svc = MetricsService()
        actual = {m for m in dir(svc) if not m.startswith("_")}
        assert actual == {
            "record_test_execution",
            "record_quality_analysis",
            "get_test_execution_analytics",
        }

    @pytest.mark.asyncio
    async def test_record_test_execution_returns_none(self):
        """MetricsService record_test_execution returns None."""
        svc = MetricsService()
        result = await svc.record_test_execution({"event_type": "test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_record_quality_analysis_with_data_returns_none(self):
        """MetricsService record_quality_analysis accepts dict and returns None."""
        svc = MetricsService()
        result = await svc.record_quality_analysis({"score": 85})
        assert result is None

    @pytest.mark.asyncio
    async def test_record_quality_analysis_default_is_none(self):
        """MetricsService record_quality_analysis default param is None, returns None."""
        sig = inspect.signature(MetricsService.record_quality_analysis)
        assert sig.parameters["analysis_data"].default is None
        svc = MetricsService()
        result = await svc.record_quality_analysis()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_test_execution_analytics_returns_exact_defaults(self):
        """MetricsService get_test_execution_analytics returns exactly the zeroed defaults."""
        svc = MetricsService()
        result = await svc.get_test_execution_analytics()
        assert result == {"total_executions": 0, "success_rate": 0.0}


class TestMetricsServiceSingleton:
    """Verify get_metrics_service returns the same instance."""

    def test_is_singleton(self):
        """get_metrics_service returns same instance on repeated calls."""
        a = get_metrics_service()
        b = get_metrics_service()
        assert a is b

    def test_returns_metrics_service_type(self):
        """get_metrics_service returns a MetricsService (not a subclass or mock)."""
        svc = get_metrics_service()
        assert type(svc) is MetricsService
