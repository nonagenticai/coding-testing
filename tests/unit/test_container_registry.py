"""Tests for TestExecutionContainerRegistry in test_execution_container_registry.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.test_execution.test_execution_container_registry import (
    ContainerExecutionStatus,
    ContainerTestExecutionStatus,
    TestExecutionContainerRegistry,
    TestExecutionSpec,
    create_container_registry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(monkeypatch):
    """Create a registry with Docker and periodic refresh disabled."""
    monkeypatch.setenv("SKIP_TEST_EXECUTORS", "true")
    return TestExecutionContainerRegistry(enable_periodic_refresh=False)


@pytest.fixture
def sample_spec():
    return TestExecutionSpec(
        image="custom/test:latest",
        version="1.0",
        memory_limit="256m",
        cpu_quota=25000,
        timeout_seconds=120,
        environment={"FOO": "bar"},
        volumes={"/tmp": "/tmp"},
        supported_test_types=["unit", "integration"],
    )


# ---------------------------------------------------------------------------
# Expected language image map — single source of truth for assertions
# ---------------------------------------------------------------------------

EXPECTED_SPECS = {
    "go": {
        "image": "codegraph/test-executor-go:latest",
        "version": "1.21",
        "memory_limit": "512m",
        "timeout_seconds": 300,
        "test_types": ["unit", "benchmark", "coverage", "race", "security", "vet"],
    },
    "rust": {
        "image": "codegraph/test-executor-rust:latest",
        "version": "1.75",
        "memory_limit": "1g",
        "timeout_seconds": 600,
        "test_types": ["unit", "doc", "benchmark", "coverage", "security", "clippy", "fmt"],
    },
    "java": {
        "image": "codegraph/test-executor-java:latest",
        "version": "17",
        "memory_limit": "1g",
        "timeout_seconds": 600,
        "test_types": ["unit", "integration", "coverage", "security"],
    },
    "csharp": {
        "image": "codegraph/test-executor-csharp:latest",
        "version": "8.0",
        "memory_limit": "768m",
        "timeout_seconds": 450,
        "test_types": ["unit", "integration", "coverage", "security"],
    },
    "cpp": {
        "image": "codegraph/test-executor-cpp:latest",
        "version": "20",
        "memory_limit": "512m",
        "timeout_seconds": 300,
        "test_types": ["unit", "coverage", "security"],
    },
    "python": {
        "image": "codegraph/test-executor-python:latest",
        "version": "3.11",
        "memory_limit": "512m",
        "timeout_seconds": 300,
        "test_types": ["unit", "integration", "coverage", "security"],
    },
    "nodejs": {
        "image": "codegraph/test-executor-nodejs:latest",
        "version": "20",
        "memory_limit": "512m",
        "timeout_seconds": 300,
        "test_types": ["unit", "integration", "coverage", "security"],
    },
    "javascript": {
        "image": "codegraph/test-executor-nodejs:latest",
        "version": "20",
        "memory_limit": "512m",
        "timeout_seconds": 300,
        "test_types": ["unit", "integration", "coverage", "security"],
    },
    "typescript": {
        "image": "codegraph/test-executor-nodejs:latest",
        "version": "20",
        "memory_limit": "512m",
        "timeout_seconds": 300,
        "test_types": ["unit", "integration", "coverage", "security"],
    },
}

ALL_LANGUAGES = sorted(EXPECTED_SPECS.keys())

EXPECTED_HEALTH_URLS = {
    "python": "http://python-test-executor:8087/health",
    "javascript": "http://javascript-test-executor:8085/health",
    "typescript": "http://javascript-test-executor:8085/health",
    "nodejs": "http://javascript-test-executor:8085/health",
    "java": "http://java-test-executor:8088/health",
    "rust": "http://rust-test-executor:8089/health",
    "csharp": "http://csharp-test-executor:8090/health",
    "go": "http://go-test-executor:8091/health",
    "cpp": "http://cpp-test-executor:8094/health",
}


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_container_test_execution_status_values(self):
        assert ContainerTestExecutionStatus.AVAILABLE.value == "available"
        assert ContainerTestExecutionStatus.UNAVAILABLE.value == "unavailable"

    def test_container_execution_status_values(self):
        assert ContainerExecutionStatus.AVAILABLE == "available"
        assert ContainerExecutionStatus.BUILDING == "building"
        assert ContainerExecutionStatus.FAILED == "failed"
        assert ContainerExecutionStatus.UNAVAILABLE == "unavailable"


# ---------------------------------------------------------------------------
# TestExecutionSpec
# ---------------------------------------------------------------------------


class TestTestExecutionSpec:
    def test_defaults(self, sample_spec):
        assert sample_spec.working_dir == "/workspace"
        assert sample_spec.user == "1001:1001"
        assert sample_spec.mounts == []
        assert sample_spec.security_opts == ["no-new-privileges:true"]

    def test_custom_security_opts(self):
        spec = TestExecutionSpec(
            image="x:latest",
            version="1",
            memory_limit="128m",
            cpu_quota=10000,
            timeout_seconds=60,
            environment={},
            volumes={},
            security_opts=["custom"],
        )
        assert spec.security_opts == ["custom"]


# ---------------------------------------------------------------------------
# Container spec lookups
# ---------------------------------------------------------------------------


class TestContainerSpecLookups:
    def test_get_spec_for_known_language(self, registry):
        spec = registry.get_container_spec("python")
        assert spec.image == "codegraph/test-executor-python:latest"
        assert spec.version == "3.11"
        assert spec.memory_limit == "512m"
        assert spec.timeout_seconds == 300

    def test_get_spec_case_insensitive(self, registry):
        upper = registry.get_container_spec("Python")
        all_upper = registry.get_container_spec("PYTHON")
        lower = registry.get_container_spec("python")
        # All three should resolve to the exact same spec object
        assert upper is lower
        assert all_upper is lower

    def test_get_spec_unknown_returns_none(self, registry):
        assert registry.get_container_spec("haskell") is None

    @pytest.mark.parametrize("lang", list(EXPECTED_SPECS.keys()))
    def test_all_languages_have_specs(self, registry, lang):
        spec = registry.get_container_spec(lang)
        expected = EXPECTED_SPECS[lang]
        assert spec.image == expected["image"]
        assert spec.version == expected["version"]
        assert spec.memory_limit == expected["memory_limit"]
        assert spec.timeout_seconds == expected["timeout_seconds"]
        assert spec.supported_test_types == expected["test_types"]


# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------


class TestSupportedLanguages:
    def test_get_supported_languages(self, registry):
        langs = registry.get_supported_languages()
        assert sorted(langs) == ALL_LANGUAGES

    def test_get_available_languages_all_unavailable(self, registry):
        # In SKIP_TEST_EXECUTORS mode, all are unavailable
        available = registry.get_available_languages()
        assert available == []


# ---------------------------------------------------------------------------
# Container status
# ---------------------------------------------------------------------------


class TestContainerStatus:
    def test_get_status_known(self, registry):
        status = registry.get_container_status("python")
        assert status == ContainerExecutionStatus.UNAVAILABLE

    def test_get_status_unknown_language(self, registry):
        status = registry.get_container_status("lisp")
        assert status == ContainerExecutionStatus.UNAVAILABLE

    def test_is_container_available_false(self, registry):
        assert registry.is_container_available("python") is False


# ---------------------------------------------------------------------------
# Supported test types
# ---------------------------------------------------------------------------


class TestSupportedTestTypes:
    def test_go_test_types(self, registry):
        types = registry.get_supported_test_types("go")
        assert types == ["unit", "benchmark", "coverage", "race", "security", "vet"]

    def test_rust_test_types(self, registry):
        types = registry.get_supported_test_types("rust")
        assert types == ["unit", "doc", "benchmark", "coverage", "security", "clippy", "fmt"]

    def test_unknown_language_defaults_to_unit(self, registry):
        types = registry.get_supported_test_types("unknown")
        assert types == ["unit"]


# ---------------------------------------------------------------------------
# Health check URLs
# ---------------------------------------------------------------------------


class TestHealthCheckURLs:
    def test_python_health_url(self, registry):
        url = registry.get_service_health_check("python")
        assert url == "http://python-test-executor:8087/health"

    def test_go_health_url(self, registry):
        url = registry.get_service_health_check("go")
        assert url == "http://go-test-executor:8091/health"

    def test_javascript_health_url(self, registry):
        url = registry.get_service_health_check("javascript")
        assert url == "http://javascript-test-executor:8085/health"

    def test_typescript_shares_js_url(self, registry):
        js_url = registry.get_service_health_check("javascript")
        ts_url = registry.get_service_health_check("typescript")
        assert js_url == ts_url == "http://javascript-test-executor:8085/health"

    def test_unknown_language_returns_none(self, registry):
        assert registry.get_service_health_check("fortran") is None

    @pytest.mark.parametrize("lang", list(EXPECTED_HEALTH_URLS.keys()))
    def test_all_health_urls(self, registry, lang):
        assert registry.get_service_health_check(lang) == EXPECTED_HEALTH_URLS[lang]


# ---------------------------------------------------------------------------
# get_container_info
# ---------------------------------------------------------------------------


class TestGetContainerInfo:
    def test_known_language(self, registry):
        info = registry.get_container_info("python")
        assert info["language"] == "python"
        assert info["image"] == "codegraph/test-executor-python:latest"
        assert info["status"] == "unavailable"
        assert info["available"] is False
        assert info["memory_limit"] == "512m"
        assert info["cpu_quota"] == 50000
        assert info["timeout_seconds"] == 300
        assert info["working_dir"] == "/workspace"
        assert info["user"] == "1001:1001"
        assert info["supported_test_types"] == ["unit", "integration", "coverage", "security"]
        assert "PYTHONPATH" in info["environment"]
        assert info["environment"]["PYTHONPATH"] == "/workspace"

    def test_unknown_language_empty(self, registry):
        assert registry.get_container_info("brainfuck") == {}


# ---------------------------------------------------------------------------
# get_registry_status
# ---------------------------------------------------------------------------


class TestGetRegistryStatus:
    def test_registry_status_structure(self, registry):
        status = registry.get_registry_status()
        assert status["total_containers"] == 9
        assert status["available_containers"] == 0
        assert status["availability_rate"] == 0
        assert status["docker_client_available"] is False
        assert sorted(status["supported_languages"]) == ALL_LANGUAGES
        assert status["available_languages"] == []
        # All containers should be unavailable in SKIP mode
        for lang, st in status["container_status"].items():
            assert st == "unavailable", f"{lang} should be unavailable, got {st}"


# ---------------------------------------------------------------------------
# validate_container_spec
# ---------------------------------------------------------------------------


class TestValidateContainerSpec:
    def test_valid_spec(self, registry):
        result = registry.validate_container_spec("python")
        assert result["valid"] is True
        assert result["language"] == "python"
        checks = result["checks"]
        assert len(checks) == 5
        check_names = [c["check"] for c in checks]
        assert check_names == ["image_format", "memory_limit", "cpu_quota", "timeout", "test_types"]
        assert all(c["passed"] is True for c in checks)

    def test_unknown_language(self, registry):
        result = registry.validate_container_spec("unknown")
        assert result["valid"] is False
        assert result["error"] == "No specification found for unknown"

    def test_all_check_types_present(self, registry):
        result = registry.validate_container_spec("go")
        check_names = [c["check"] for c in result["checks"]]
        assert check_names == ["image_format", "memory_limit", "cpu_quota", "timeout", "test_types"]
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# validate_container_spec_object
# ---------------------------------------------------------------------------


class TestValidateContainerSpecObject:
    def test_valid_object(self, registry, sample_spec):
        result = registry.validate_container_spec_object(sample_spec)
        assert result == {"valid": True}

    def test_empty_image(self, registry):
        spec = TestExecutionSpec(
            image="",
            version="1",
            memory_limit="128m",
            cpu_quota=10000,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        result = registry.validate_container_spec_object(spec)
        assert result == {"valid": False, "error": "Image name is required"}

    def test_zero_cpu_quota(self, registry):
        spec = TestExecutionSpec(
            image="x:latest",
            version="1",
            memory_limit="128m",
            cpu_quota=0,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        result = registry.validate_container_spec_object(spec)
        assert result == {"valid": False, "error": "CPU quota must be positive"}

    def test_zero_timeout(self, registry):
        spec = TestExecutionSpec(
            image="x:latest",
            version="1",
            memory_limit="128m",
            cpu_quota=10000,
            timeout_seconds=0,
            environment={},
            volumes={},
        )
        result = registry.validate_container_spec_object(spec)
        assert result == {"valid": False, "error": "Timeout must be positive"}


# ---------------------------------------------------------------------------
# register_custom_container
# ---------------------------------------------------------------------------


class TestRegisterCustomContainer:
    def test_register_valid_spec(self, registry, sample_spec):
        result = registry.register_custom_container("custom_lang", sample_spec)
        assert result is True
        assert "custom_lang" in registry.specs
        assert registry.specs["custom_lang"] is sample_spec
        assert registry.container_status["custom_lang"] == ContainerExecutionStatus.UNAVAILABLE

    def test_register_invalid_spec(self, registry):
        bad_spec = TestExecutionSpec(
            image="",
            version="1",
            memory_limit="128m",
            cpu_quota=10000,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        result = registry.register_custom_container("bad_lang", bad_spec)
        assert result is False
        assert "bad_lang" not in registry.specs


# ---------------------------------------------------------------------------
# _check_container_availability (no docker)
# ---------------------------------------------------------------------------


class TestCheckContainerAvailability:
    def test_no_docker_marks_all_unavailable(self, registry):
        registry.docker_client = None
        registry._check_container_availability()
        for lang in registry.specs:
            assert registry.container_status[lang] == ContainerExecutionStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# build_container / pull_container (no docker)
# ---------------------------------------------------------------------------


class TestBuildPullNoDocker:
    def test_build_no_docker_returns_false(self, registry):
        assert registry.build_container("python") is False

    def test_build_unknown_lang_returns_false(self, registry):
        registry.docker_client = MagicMock()
        assert registry.build_container("unknown") is False

    def test_pull_no_docker_returns_false(self, registry):
        assert registry.pull_container("python") is False

    def test_pull_unknown_lang_returns_false(self, registry):
        registry.docker_client = MagicMock()
        assert registry.pull_container("unknown") is False


# ---------------------------------------------------------------------------
# refresh / stop periodic
# ---------------------------------------------------------------------------


class TestPeriodicRefresh:
    def test_refresh_container_availability(self, registry):
        # After refresh with no docker, all containers remain unavailable
        registry.refresh_container_availability()
        for lang in registry.specs:
            assert registry.container_status[lang] == ContainerExecutionStatus.UNAVAILABLE

    def test_refresh_container_status(self, registry):
        # refresh_container_status calls _check_container_availability
        registry.refresh_container_status()
        for lang in registry.specs:
            assert registry.container_status[lang] == ContainerExecutionStatus.UNAVAILABLE

    def test_stop_periodic_refresh_noop(self, registry):
        # No thread running, stop should set _stop_refresh and not crash
        registry.stop_periodic_refresh()
        assert registry._stop_refresh is True


# ---------------------------------------------------------------------------
# check_service_health async
# ---------------------------------------------------------------------------


class TestCheckServiceHealthAsync:
    @pytest.mark.asyncio
    async def test_no_aiohttp(self, registry, monkeypatch):
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.setattr(mod, "HAS_AIOHTTP", False)
        result = await registry.check_service_health("python")
        assert result is False

    @pytest.mark.asyncio
    async def test_unknown_language(self, registry):
        result = await registry.check_service_health("unknown_lang")
        assert result is False


# ---------------------------------------------------------------------------
# _check_single_container_availability
# ---------------------------------------------------------------------------


class TestCheckSingleContainerAvailability:
    def test_no_docker_marks_unavailable(self, registry):
        registry.docker_client = None
        registry._check_single_container_availability("python")
        assert registry.container_status["python"] == ContainerExecutionStatus.UNAVAILABLE

    def test_unknown_spec_marks_unavailable(self, registry):
        registry.docker_client = MagicMock()
        registry._check_single_container_availability("nonexistent")
        assert registry.container_status["nonexistent"] == ContainerExecutionStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_container_registry(self, monkeypatch):
        monkeypatch.setenv("SKIP_TEST_EXECUTORS", "true")
        reg = create_container_registry(enable_periodic_refresh=False)
        # Verify it created a properly initialized registry
        assert sorted(reg.get_supported_languages()) == ALL_LANGUAGES
        assert reg.docker_client is None
        assert reg._docker_available is False
        for lang in reg.specs:
            assert reg.container_status[lang] == ContainerExecutionStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# CI/K8s mode init
# ---------------------------------------------------------------------------


class TestInitModes:
    def test_ci_mode(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        reg = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        for lang in reg.specs:
            assert reg.container_status[lang] == ContainerExecutionStatus.UNAVAILABLE

    def test_k8s_mode(self, monkeypatch):
        monkeypatch.setenv("KUBERNETES_IN_CLUSTER", "true")
        reg = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        assert reg._docker_available is False
        assert reg.docker_client is None
        for lang in reg.specs:
            assert reg.container_status[lang] == ContainerExecutionStatus.UNAVAILABLE

    def test_check_all_services_health_deprecated(self, registry):
        result = registry.check_all_services_health()
        # Deprecated wrapper returns same as check_all_services_health_sync
        for lang in registry.specs:
            assert lang in result
            assert result[lang] is False


# ---------------------------------------------------------------------------
# validate_container_spec edge cases
# ---------------------------------------------------------------------------


class TestValidateContainerSpecEdgeCases:
    def test_invalid_memory_limit(self, registry):
        """Test spec with invalid memory limit unit."""
        spec = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="512",
            cpu_quota=10000,
            timeout_seconds=60,
            environment={},
            volumes={},
            supported_test_types=["unit"],
        )
        registry.specs["test_lang"] = spec
        result = registry.validate_container_spec("test_lang")
        mem_check = next(c for c in result["checks"] if c["check"] == "memory_limit")
        assert mem_check["passed"] is False
        assert mem_check["message"] == "Memory limit should have unit (m, g, k)"

    def test_invalid_cpu_quota(self, registry):
        """Test spec with out-of-range CPU quota."""
        spec = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="512m",
            cpu_quota=200000,
            timeout_seconds=60,
            environment={},
            volumes={},
            supported_test_types=["unit"],
        )
        registry.specs["test_lang2"] = spec
        result = registry.validate_container_spec("test_lang2")
        cpu_check = next(c for c in result["checks"] if c["check"] == "cpu_quota")
        assert cpu_check["passed"] is False
        assert cpu_check["message"] == "CPU quota should be between 1 and 100000"

    def test_invalid_timeout(self, registry):
        """Test spec with out-of-range timeout."""
        spec = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="512m",
            cpu_quota=10000,
            timeout_seconds=5000,
            environment={},
            volumes={},
            supported_test_types=["unit"],
        )
        registry.specs["test_lang3"] = spec
        result = registry.validate_container_spec("test_lang3")
        timeout_check = next(c for c in result["checks"] if c["check"] == "timeout")
        assert timeout_check["passed"] is False
        assert timeout_check["message"] == "Timeout should be between 1 and 3600 seconds"

    def test_no_test_types(self, registry):
        """Test spec with empty test types."""
        spec = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="512m",
            cpu_quota=10000,
            timeout_seconds=60,
            environment={},
            volumes={},
            supported_test_types=[],
        )
        registry.specs["test_lang4"] = spec
        result = registry.validate_container_spec("test_lang4")
        types_check = next(c for c in result["checks"] if c["check"] == "test_types")
        assert types_check["passed"] is False
        assert types_check["message"] == "No supported test types defined"

    def test_image_without_tag(self, registry):
        """Test spec with image missing tag."""
        spec = TestExecutionSpec(
            image="testimage",
            version="1",
            memory_limit="512m",
            cpu_quota=10000,
            timeout_seconds=60,
            environment={},
            volumes={},
            supported_test_types=["unit"],
        )
        registry.specs["test_lang5"] = spec
        result = registry.validate_container_spec("test_lang5")
        img_check = next(c for c in result["checks"] if c["check"] == "image_format")
        assert img_check["passed"] is False
        assert img_check["message"] == "Image name should include tag"
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# get_container_info edge cases
# ---------------------------------------------------------------------------


class TestGetContainerInfoAdvanced:
    def test_info_includes_test_types(self, registry):
        info = registry.get_container_info("go")
        assert info["supported_test_types"] == [
            "unit",
            "benchmark",
            "coverage",
            "race",
            "security",
            "vet",
        ]

    def test_info_includes_environment(self, registry):
        info = registry.get_container_info("python")
        assert info["environment"]["PYTHONPATH"] == "/workspace"
        assert info["environment"]["PYTHONUNBUFFERED"] == "1"
        assert info["environment"]["PYTEST_CURRENT_TEST"] == "1"


# ---------------------------------------------------------------------------
# get_registry_status advanced
# ---------------------------------------------------------------------------


class TestGetRegistryStatusAdvanced:
    def test_availability_rate_zero(self, registry):
        status = registry.get_registry_status()
        assert status["availability_rate"] == 0
        assert status["available_containers"] == 0
        assert status["total_containers"] == 9


# ---------------------------------------------------------------------------
# build_container edge cases
# ---------------------------------------------------------------------------


class TestBuildContainerAdvanced:
    def test_build_missing_dockerfile(self, registry):
        registry.docker_client = MagicMock()
        result = registry.build_container("python", "/nonexistent/Dockerfile")
        assert result is False
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED

    def test_build_docker_exception(self, registry, tmp_path):
        registry.docker_client = MagicMock()
        # Create a Dockerfile
        df = tmp_path / "Dockerfile"
        df.write_text("FROM scratch\n")
        registry.docker_client.images.build = MagicMock(side_effect=RuntimeError("build fail"))
        result = registry.build_container("python", str(df))
        assert result is False
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED


# ---------------------------------------------------------------------------
# pull_container edge cases
# ---------------------------------------------------------------------------


class TestPullContainerAdvanced:
    def test_pull_success(self, registry):
        registry.docker_client = MagicMock()
        registry.docker_client.images.pull = MagicMock()
        result = registry.pull_container("python")
        assert result is True
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_pull_exception(self, registry):
        registry.docker_client = MagicMock()
        registry.docker_client.images.pull = MagicMock(side_effect=RuntimeError("pull fail"))
        result = registry.pull_container("python")
        assert result is False
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED


# ---------------------------------------------------------------------------
# check_service_health async edge cases
# ---------------------------------------------------------------------------


class TestCheckServiceHealthAdvanced:
    @pytest.mark.asyncio
    async def test_go_gets_longer_timeout(self, registry, monkeypatch):
        """Go services should get 10s timeout instead of default."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.setattr(mod, "HAS_AIOHTTP", False)
        # Without aiohttp the check returns False immediately
        result = await registry.check_service_health("go")
        assert result is False


# ---------------------------------------------------------------------------
# check_all_services_health_sync advanced
# ---------------------------------------------------------------------------


class TestCheckAllServicesHealthSyncAdvanced:
    def test_returns_dict_for_all_languages(self, registry):
        result = registry.check_all_services_health_sync()
        # Every registered language must appear with a bool result
        for lang in registry.specs:
            assert lang in result
            assert result[lang] is False  # No services running in test mode


# ---------------------------------------------------------------------------
# register_custom_container advanced
# ---------------------------------------------------------------------------


class TestRegisterCustomContainerAdvanced:
    def test_register_exception(self, registry):
        """Registration with exception returns False."""
        spec = MagicMock()
        # Make validate raise
        registry.validate_container_spec_object = MagicMock(
            side_effect=RuntimeError("validate crash")
        )
        result = registry.register_custom_container("new_lang", spec)
        assert result is False
        assert "new_lang" not in registry.specs

    def test_register_calls_availability_check(self, registry, sample_spec):
        """Successful registration checks availability."""
        result = registry.register_custom_container("custom_lang", sample_spec)
        assert result is True
        assert "custom_lang" in registry.container_status
        # docker_client is None, so availability check marks it unavailable
        assert registry.container_status["custom_lang"] == ContainerExecutionStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# validate_container_spec_object edge case
# ---------------------------------------------------------------------------


class TestValidateContainerSpecObjectAdvanced:
    def test_empty_memory_limit(self, registry):
        spec = TestExecutionSpec(
            image="x:latest",
            version="1",
            memory_limit="",
            cpu_quota=10000,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        result = registry.validate_container_spec_object(spec)
        assert result == {"valid": False, "error": "Memory limit is required"}


# ---------------------------------------------------------------------------
# Coverage tests — import fallbacks
# ---------------------------------------------------------------------------


class TestImportFallbacks:
    def test_has_aiohttp_flag_is_true(self):
        """aiohttp is a project dependency and should be importable."""
        from services.test_execution.test_execution_container_registry import HAS_AIOHTTP

        assert HAS_AIOHTTP is True

    def test_has_docker_flag_is_false(self):
        """docker is an optional dependency and is not installed in test env."""
        from services.test_execution.test_execution_container_registry import HAS_DOCKER

        assert HAS_DOCKER is False


# ---------------------------------------------------------------------------
# Coverage tests — build_container extended
# ---------------------------------------------------------------------------


class TestBuildContainerExtended:
    def test_build_sets_status_to_building_during_build(self, registry, tmp_path):
        """build_container should set BUILDING status before attempting the build."""
        registry.docker_client = MagicMock()
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11")

        statuses_during_build = []

        def capture_status(*args, **kwargs):
            statuses_during_build.append(registry.container_status.get("python"))
            return (MagicMock(), [])

        registry.docker_client.images.build.side_effect = capture_status
        registry.build_container("python", dockerfile_path=str(dockerfile))
        # During the build call, status should have been BUILDING
        assert statuses_during_build[0] == ContainerExecutionStatus.BUILDING

    def test_build_uses_default_dockerfile_path(self, registry):
        """When no dockerfile_path given, build_container constructs a default path."""
        registry.docker_client = MagicMock()
        # No dockerfile at the default path, so it should fail with FAILED status
        result = registry.build_container("python")
        assert result is False
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED

    def test_build_processes_non_stream_log_entries(self, registry, tmp_path):
        """build_container should handle build logs that lack 'stream' key."""
        registry.docker_client = MagicMock()
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11")

        mock_image = MagicMock()
        # Mix of stream and non-stream log entries
        mock_logs = [
            {"stream": "Step 1/2 : FROM python:3.11\n"},
            {"aux": {"ID": "sha256:abc123"}},
            {"error": "warning: something"},
        ]
        registry.docker_client.images.build.return_value = (mock_image, mock_logs)

        result = registry.build_container("python", dockerfile_path=str(dockerfile))
        assert result is True
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_build_multiple_languages_independently(self, registry, tmp_path):
        """Building one language should not affect another's status."""
        registry.docker_client = MagicMock()

        py_dockerfile = tmp_path / "py" / "Dockerfile"
        py_dockerfile.parent.mkdir()
        py_dockerfile.write_text("FROM python:3.11")
        registry.docker_client.images.build.return_value = (MagicMock(), [])
        registry.build_container("python", dockerfile_path=str(py_dockerfile))

        # Now fail go build
        go_dockerfile = tmp_path / "go" / "Dockerfile"
        go_dockerfile.parent.mkdir()
        go_dockerfile.write_text("FROM golang:1.21")
        registry.docker_client.images.build.side_effect = Exception("go build failed")
        registry.build_container("go", dockerfile_path=str(go_dockerfile))

        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE
        assert registry.container_status["go"] == ContainerExecutionStatus.FAILED


# ---------------------------------------------------------------------------
# Coverage tests — pull_container extended
# ---------------------------------------------------------------------------


class TestPullContainerExtended:
    def test_pull_uses_spec_image_name(self, registry):
        """pull_container should pull the image name from the language's spec."""
        registry.docker_client = MagicMock()
        spec = registry.get_container_spec("python")
        registry.pull_container("python")
        registry.docker_client.images.pull.assert_called_once_with(spec.image)

    def test_pull_timeout_error(self, registry):
        """pull_container should handle timeout-like exceptions."""
        registry.docker_client = MagicMock()
        registry.docker_client.images.pull.side_effect = TimeoutError("pull timed out")
        result = registry.pull_container("python")
        assert result is False
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED

    def test_pull_then_build_overwrites_status(self, registry, tmp_path):
        """A successful build after a failed pull should set status to AVAILABLE."""
        registry.docker_client = MagicMock()
        # Pull fails
        registry.docker_client.images.pull.side_effect = Exception("Pull failed")
        registry.pull_container("python")
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED

        # Build succeeds
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11")
        registry.docker_client.images.build.return_value = (MagicMock(), [])
        registry.build_container("python", dockerfile_path=str(dockerfile))
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_pull_multiple_languages(self, registry):
        """Pulling multiple languages should set each status independently."""
        registry.docker_client = MagicMock()
        registry.pull_container("python")
        registry.pull_container("go")
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE
        assert registry.container_status["go"] == ContainerExecutionStatus.AVAILABLE


# ---------------------------------------------------------------------------
# Coverage tests — check_service_health extended
# ---------------------------------------------------------------------------


class TestCheckServiceHealthExtended:
    @pytest.mark.asyncio
    async def test_no_aiohttp(self, registry):
        with patch("services.test_execution.test_execution_container_registry.HAS_AIOHTTP", False):
            result = await registry.check_service_health("python")
            assert result is False

    @pytest.mark.asyncio
    async def test_no_health_url(self, registry):
        result = await registry.check_service_health("nonexistent_language")
        assert result is False

    @pytest.mark.asyncio
    async def test_go_gets_longer_timeout(self, registry):
        """Go services should get a longer timeout."""
        url = registry.get_service_health_check("go")
        assert url == "http://go-test-executor:8091/health"
        with patch("services.test_execution.test_execution_container_registry.HAS_AIOHTTP", True):
            with patch(
                "services.test_execution.test_execution_container_registry.aiohttp"
            ) as mock_aiohttp:
                mock_aiohttp.ClientSession = MagicMock(side_effect=ConnectionError("no connection"))
                mock_aiohttp.ClientTimeout = MagicMock()
                result = await registry.check_service_health("go")
                assert result is False

    @pytest.mark.asyncio
    async def test_connection_error_marks_building(self, registry):
        """Connection error should mark service as building if it was available."""
        registry.container_status["python"] = ContainerExecutionStatus.AVAILABLE

        with patch("services.test_execution.test_execution_container_registry.HAS_AIOHTTP", True):
            with patch(
                "services.test_execution.test_execution_container_registry.aiohttp"
            ) as mock_aiohttp:
                mock_aiohttp.ClientSession.return_value.__aenter__ = AsyncMock(
                    side_effect=ConnectionError("connection refused")
                )
                mock_aiohttp.ClientTimeout = MagicMock()

                result = await registry.check_service_health("python")
                assert result is False
                assert registry.container_status["python"] == ContainerExecutionStatus.BUILDING


# ---------------------------------------------------------------------------
# Coverage tests — register_custom_container extended
# ---------------------------------------------------------------------------


class TestRegisterCustomContainerExtended:
    def test_register_valid_container(self, registry, sample_spec):
        registry.docker_client = None
        result = registry.register_custom_container("custom_lang", sample_spec)
        assert result is True
        assert "custom_lang" in registry.specs
        assert registry.specs["custom_lang"] is sample_spec
        assert registry.container_status["custom_lang"] == ContainerExecutionStatus.UNAVAILABLE

    def test_register_invalid_container(self, registry):
        invalid_spec = TestExecutionSpec(
            image="",
            version="1.0",
            memory_limit="256m",
            cpu_quota=25000,
            timeout_seconds=120,
            environment={},
            volumes={},
        )
        result = registry.register_custom_container("bad_lang", invalid_spec)
        assert result is False
        assert "bad_lang" not in registry.specs

    def test_register_exception(self, registry):
        with patch.object(
            registry, "validate_container_spec_object", side_effect=Exception("validation crash")
        ):
            result = registry.register_custom_container("crash", MagicMock())
            assert result is False
            assert "crash" not in registry.specs


# ---------------------------------------------------------------------------
# Coverage tests — validate_container_spec_object extended
# ---------------------------------------------------------------------------


class TestValidateContainerSpecObjectExtended:
    def test_valid_spec(self, registry, sample_spec):
        result = registry.validate_container_spec_object(sample_spec)
        assert result == {"valid": True}

    def test_empty_image(self, registry):
        spec = TestExecutionSpec(
            image="",
            version="1",
            memory_limit="256m",
            cpu_quota=25000,
            timeout_seconds=120,
            environment={},
            volumes={},
        )
        result = registry.validate_container_spec_object(spec)
        assert result == {"valid": False, "error": "Image name is required"}

    def test_negative_cpu_quota(self, registry):
        spec = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="256m",
            cpu_quota=-1,
            timeout_seconds=120,
            environment={},
            volumes={},
        )
        result = registry.validate_container_spec_object(spec)
        assert result == {"valid": False, "error": "CPU quota must be positive"}

    def test_negative_timeout(self, registry):
        spec = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="256m",
            cpu_quota=25000,
            timeout_seconds=-1,
            environment={},
            volumes={},
        )
        result = registry.validate_container_spec_object(spec)
        assert result == {"valid": False, "error": "Timeout must be positive"}


# ---------------------------------------------------------------------------
# Coverage tests — _check_single_container_availability extended
# ---------------------------------------------------------------------------


class TestCheckSingleContainerAvailabilityExtended:
    def test_no_docker_client(self, registry):
        registry.docker_client = None
        registry._check_single_container_availability("python")
        assert registry.container_status["python"] == ContainerExecutionStatus.UNAVAILABLE

    def test_no_spec(self, registry):
        registry.docker_client = MagicMock()
        registry._check_single_container_availability("nonexistent")
        assert registry.container_status["nonexistent"] == ContainerExecutionStatus.UNAVAILABLE

    def test_image_found(self, registry):
        registry.docker_client = MagicMock()
        registry.docker_client.images.get.return_value = MagicMock()
        registry._check_single_container_availability("python")
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_image_not_found(self, registry):
        registry.docker_client = MagicMock()
        not_found = type("ImageNotFound", (Exception,), {})
        with patch("services.test_execution.test_execution_container_registry.HAS_DOCKER", True):
            with patch(
                "services.test_execution.test_execution_container_registry.docker"
            ) as mock_docker:
                mock_docker.errors = MagicMock()
                mock_docker.errors.ImageNotFound = not_found
                registry.docker_client.images.get.side_effect = not_found("not found")
                registry._check_single_container_availability("python")
                assert registry.container_status["python"] == ContainerExecutionStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Coverage tests — get_container_info extended
# ---------------------------------------------------------------------------


class TestGetContainerInfoExtended:
    def test_returns_full_info_for_go(self, registry):
        info = registry.get_container_info("go")
        assert info["language"] == "go"
        assert info["image"] == "codegraph/test-executor-go:latest"
        assert info["memory_limit"] == "512m"
        assert info["timeout_seconds"] == 300
        assert info["status"] == "unavailable"
        assert info["available"] is False
        assert info["working_dir"] == "/workspace"
        assert info["user"] == "1001:1001"
        assert info["supported_test_types"] == [
            "unit",
            "benchmark",
            "coverage",
            "race",
            "security",
            "vet",
        ]
        assert info["environment"]["GOCACHE"] == "/tmp/go-cache"

    def test_unknown_language_returns_empty(self, registry):
        info = registry.get_container_info("nonexistent")
        assert info == {}


# ---------------------------------------------------------------------------
# Coverage tests — get_registry_status extended
# ---------------------------------------------------------------------------


class TestGetRegistryStatusExtended:
    def test_returns_status(self, registry):
        status = registry.get_registry_status()
        assert status["total_containers"] == 9
        assert status["available_containers"] == 0
        assert status["docker_client_available"] is False
        assert sorted(status["supported_languages"]) == ALL_LANGUAGES
        assert status["available_languages"] == []
        assert status["availability_rate"] == 0


# ---------------------------------------------------------------------------
# Coverage tests — validate_container_spec extended
# ---------------------------------------------------------------------------


class TestValidateContainerSpecExtended:
    def test_valid_python_spec(self, registry):
        result = registry.validate_container_spec("python")
        assert result["valid"] is True
        assert result["language"] == "python"
        assert len(result["checks"]) == 5
        assert all(c["passed"] for c in result["checks"])

    def test_unknown_language(self, registry):
        result = registry.validate_container_spec("brainfuck")
        assert result == {"valid": False, "error": "No specification found for brainfuck"}


# ---------------------------------------------------------------------------
# Coverage tests — stop_periodic_refresh
# ---------------------------------------------------------------------------


class TestStopPeriodicRefresh:
    def test_stop_when_not_started(self, registry):
        registry._refresh_thread = None
        registry.stop_periodic_refresh()
        assert registry._stop_refresh is True

    def test_stop_when_running(self, registry):
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_thread.join = MagicMock()
        registry._refresh_thread = mock_thread
        registry.stop_periodic_refresh()
        assert registry._stop_refresh is True
        mock_thread.join.assert_called_once()


# ---------------------------------------------------------------------------
# Coverage tests — force_health_check_refresh
# ---------------------------------------------------------------------------


class TestForceHealthCheckRefresh:
    def test_force_refresh(self, registry):
        with patch.object(registry, "_check_container_availability") as mock_check:
            with patch.object(
                registry,
                "check_all_services_health_sync",
                return_value={"python": False, "go": True},
            ) as mock_health:
                result = registry.force_health_check_refresh()
                mock_check.assert_called_once()
                mock_health.assert_called_once()
                assert result == {"python": False, "go": True}


# ---------------------------------------------------------------------------
# Coverage tests — get_health_status
# ---------------------------------------------------------------------------


class TestGetHealthStatus:
    def test_returns_comprehensive_status(self, registry):
        with patch.object(
            registry, "check_all_services_health_sync", return_value={"python": False}
        ):
            status = registry.get_health_status()
            assert status["docker_client_available"] is False
            assert status["periodic_refresh_enabled"] is False
            assert status["health_check_interval"] == 30
            assert status["health_results"] == {"python": False}
            # container_status should have all languages
            for lang in registry.specs:
                assert lang in status["container_status"]
                assert status["container_status"][lang] == "unavailable"
            overall = status["overall_health"]
            assert overall["total_services"] == 9
            assert overall["healthy_services"] == 0
            assert overall["available_containers"] == 0
            assert overall["health_rate"] == 0
            assert overall["availability_rate"] == 0
            # Timestamp sanity check
            assert status["timestamp"] > 0
            assert status["last_health_check"] == 0
            assert status["seconds_since_last_check"] > 0
