"""Tests for TestOrchestrator class in test_orchestrator.py."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.test_framework_config import TestConfig
from services.stubs import DockerService
from services.test_execution.test_orchestrator import (
    OrchestratorType,
    TestExecutionRequest,
    TestExecutionResult,
    TestLanguage,
    TestOrchestrator,
    create_test_orchestrator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docker_service():
    """Create a mock DockerService."""
    svc = AsyncMock(spec=DockerService)
    svc.create_test_container = AsyncMock(return_value="container-123")
    svc.start_container = AsyncMock()
    svc.create_file_in_container = AsyncMock(return_value=True)
    svc.exec_in_container = AsyncMock(
        return_value={
            "exit_code": 0,
            "stdout": "PASS\nok  test-module  0.5s",
            "stderr": "",
        }
    )
    svc.stop_container = AsyncMock()
    svc.remove_container_by_id = AsyncMock()
    return svc


@pytest.fixture
def config():
    return TestConfig()


@pytest.fixture
def orchestrator(docker_service, config):
    return TestOrchestrator(docker_service, config)


@pytest.fixture
def base_request():
    return TestExecutionRequest(
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        source_code="def add(a, b): return a + b",
        test_files={"test_main.py": "def test_add(): assert add(1,2)==3"},
        timeout_seconds=60,
    )


# ---------------------------------------------------------------------------
# Construction & factory
# ---------------------------------------------------------------------------


class TestNoFabricationWithRealStub:
    """F1 regression: with the REAL (non-mocked) DockerService stub and no real
    container runtime, the orchestration/execute path must fail honestly and can
    NEVER report an unexecuted test as passed."""

    @pytest.mark.asyncio
    async def test_real_stub_reports_failure_not_fabricated_pass(self, config, base_request):
        from services.stubs import get_docker_service

        orch = TestOrchestrator(get_docker_service(), config)
        orch.use_k8s = False  # force the Docker-stub path (not the K8s executor)

        result = await orch.execute_tests(base_request)

        # The whole point of F1: nothing executed, so this MUST NOT be a pass.
        assert result.success is False
        assert result.exit_code != 0
        assert result.container_id is None
        assert result.stderr, "an honest failure must explain why nothing ran"
        assert "runtime" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_real_stub_exec_raises_rather_than_returning_zero(self):
        from services.stubs import (
            ContainerRuntimeUnavailableError,
            get_docker_service,
        )

        with pytest.raises(ContainerRuntimeUnavailableError):
            await get_docker_service().exec_in_container("c1", ["python", "-m", "pytest"])


class TestOrchestratorInit:
    """Tests for __init__ and factory function."""

    def test_init_sets_docker_service(self, orchestrator, docker_service):
        assert orchestrator.docker_service is docker_service

    def test_init_creates_container_registry(self, orchestrator):
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        assert isinstance(orchestrator.container_registry, TestExecutionContainerRegistry)
        # Registry has specs for all supported languages
        supported = orchestrator.container_registry.get_supported_languages()
        assert "go" in supported
        assert "python" in supported
        assert "rust" in supported

    def test_init_creates_security_manager(self, orchestrator):
        from services.test_execution.security_manager import ExecutionSecurityManager

        assert isinstance(orchestrator.security_manager, ExecutionSecurityManager)

    def test_init_active_containers_empty(self, orchestrator):
        assert orchestrator.active_containers == {}

    def test_init_container_cache_empty(self, orchestrator):
        assert orchestrator.container_cache == {}

    def test_init_k8s_disabled_by_default(self, orchestrator):
        assert orchestrator.use_k8s is False
        assert orchestrator._k8s_executor is None

    def test_factory_function(self, docker_service, config):
        orch = create_test_orchestrator(docker_service, config)
        assert orch.docker_service is docker_service
        assert orch.config is config
        assert orch.active_containers == {}


# ---------------------------------------------------------------------------
# _get_test_command
# ---------------------------------------------------------------------------


class TestGetTestCommand:
    """Verify language/test-type command selection."""

    def test_go_unit(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.GO, OrchestratorType.UNIT)
        assert cmd == ["/usr/local/bin/go-test-runner.sh", "unit"]

    def test_go_benchmark(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.GO, OrchestratorType.BENCHMARK)
        assert cmd == ["/usr/local/bin/go-test-runner.sh", "benchmark"]

    def test_go_coverage(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.GO, OrchestratorType.COVERAGE)
        assert cmd == ["/usr/local/bin/go-test-runner.sh", "coverage"]

    def test_go_race(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.GO, OrchestratorType.RACE)
        assert cmd == ["/usr/local/bin/go-test-runner.sh", "race"]

    def test_go_security(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.GO, OrchestratorType.SECURITY)
        assert cmd == ["/usr/local/bin/go-test-runner.sh", "security"]

    def test_go_vet(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.GO, OrchestratorType.VET)
        assert cmd == ["/usr/local/bin/go-test-runner.sh", "vet"]

    def test_rust_unit(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.RUST, OrchestratorType.UNIT)
        assert cmd == ["/usr/local/bin/rust-test-runner.sh", "unit"]

    def test_rust_clippy(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.RUST, OrchestratorType.CLIPPY)
        assert cmd == ["/usr/local/bin/rust-test-runner.sh", "clippy"]

    def test_rust_fmt(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.RUST, OrchestratorType.FMT)
        assert cmd == ["/usr/local/bin/rust-test-runner.sh", "fmt"]

    def test_rust_doc(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.RUST, OrchestratorType.DOC)
        assert cmd == ["/usr/local/bin/rust-test-runner.sh", "doc"]

    def test_python_unit(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.PYTHON, OrchestratorType.UNIT)
        assert cmd == ["python", "-m", "pytest", "-v", "--tb=short"]

    def test_python_coverage(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.PYTHON, OrchestratorType.COVERAGE)
        assert cmd == ["python", "-m", "pytest", "--cov=.", "--cov-report=json"]

    def test_javascript_unit(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.JAVASCRIPT, OrchestratorType.UNIT)
        assert cmd == ["npm", "test"]

    def test_typescript_unit(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.TYPESCRIPT, OrchestratorType.UNIT)
        assert cmd == ["npm", "test"]

    def test_javascript_coverage(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.JAVASCRIPT, OrchestratorType.COVERAGE)
        assert cmd == ["npm", "run", "test:coverage"]

    def test_typescript_coverage(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.TYPESCRIPT, OrchestratorType.COVERAGE)
        assert cmd == ["npm", "run", "test:coverage"]

    def test_unknown_combination_returns_none(self, orchestrator):
        """An unmapped pair must yield NO command at all.

        It used to return ["echo", "No test command defined"]; `echo` exits 0
        and the verdict is `exit_code == 0`, so the suite passed having run
        nothing. There must be no command here, and in particular no command
        that exits 0.
        """
        cmd = orchestrator._get_test_command(TestLanguage.JAVA, OrchestratorType.UNIT)
        assert cmd is None

    def test_rust_benchmark(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.RUST, OrchestratorType.BENCHMARK)
        assert cmd == ["/usr/local/bin/rust-test-runner.sh", "benchmark"]

    def test_rust_coverage(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.RUST, OrchestratorType.COVERAGE)
        assert cmd == ["/usr/local/bin/rust-test-runner.sh", "coverage"]

    def test_rust_security(self, orchestrator):
        cmd = orchestrator._get_test_command(TestLanguage.RUST, OrchestratorType.SECURITY)
        assert cmd == ["/usr/local/bin/rust-test-runner.sh", "security"]


# ---------------------------------------------------------------------------
# _get_file_extension
# ---------------------------------------------------------------------------


class TestGetFileExtension:
    """Verify file extension mapping."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            ("go", ".go"),
            ("rust", ".rs"),
            ("java", ".java"),
            ("csharp", ".cs"),
            ("cpp", ".cpp"),
            ("python", ".py"),
            ("javascript", ".js"),
            ("typescript", ".ts"),
        ],
    )
    def test_known_languages(self, orchestrator, lang, ext):
        assert orchestrator._get_file_extension(lang) == ext

    def test_unknown_language_returns_txt(self, orchestrator):
        assert orchestrator._get_file_extension("haskell") == ".txt"


# ---------------------------------------------------------------------------
# Dependency formatting helpers
# ---------------------------------------------------------------------------


class TestDependencyFormatting:
    """Test Go / Rust dependency formatting helpers."""

    def test_go_deps_empty(self, orchestrator):
        assert orchestrator._format_go_dependencies({}) == ""

    def test_go_deps_single(self, orchestrator):
        result = orchestrator._format_go_dependencies({"github.com/stretchr/testify": "v1.9.0"})
        assert result == "require github.com/stretchr/testify v1.9.0"

    def test_go_deps_multiple(self, orchestrator):
        result = orchestrator._format_go_dependencies(
            {
                "github.com/stretchr/testify": "v1.9.0",
                "github.com/google/uuid": "v1.6.0",
            }
        )
        assert "require github.com/stretchr/testify v1.9.0" in result
        assert "require github.com/google/uuid v1.6.0" in result
        # Two deps should produce two lines joined by \n
        lines = result.split("\n")
        assert len(lines) == 2

    def test_rust_deps_empty(self, orchestrator):
        assert orchestrator._format_rust_dependencies({}) == ""

    def test_rust_deps_single(self, orchestrator):
        result = orchestrator._format_rust_dependencies({"serde": "1.0"})
        assert result == 'serde = "1.0"'

    def test_rust_deps_multiple(self, orchestrator):
        result = orchestrator._format_rust_dependencies(
            {
                "serde": "1.0",
                "tokio": "1.35",
            }
        )
        assert 'serde = "1.0"' in result
        assert 'tokio = "1.35"' in result
        lines = result.split("\n")
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# execute_tests – happy path
# ---------------------------------------------------------------------------


class TestExecuteTestsHappyPath:
    """Tests for the execute_tests main flow."""

    @pytest.mark.asyncio
    async def test_returns_success_result(self, orchestrator, base_request, docker_service):
        result = await orchestrator.execute_tests(base_request)
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "PASS\nok  test-module  0.5s"
        assert result.stderr == ""
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        assert result.container_id == "container-123"

    @pytest.mark.asyncio
    async def test_creates_container(self, orchestrator, base_request, docker_service):
        await orchestrator.execute_tests(base_request)
        docker_service.create_test_container.assert_called_once()
        # Verify the container config passed to create_test_container
        call_args = docker_service.create_test_container.call_args[0][0]
        assert call_args["working_dir"] == "/workspace"
        assert call_args["language"] == "python"
        assert call_args["labels"]["service"] == "test_execution"

    @pytest.mark.asyncio
    async def test_starts_container(self, orchestrator, base_request, docker_service):
        await orchestrator.execute_tests(base_request)
        docker_service.start_container.assert_called_once_with("container-123")

    @pytest.mark.asyncio
    async def test_cleans_up_container(self, orchestrator, base_request, docker_service):
        await orchestrator.execute_tests(base_request)
        docker_service.stop_container.assert_called_once_with("container-123")
        docker_service.remove_container_by_id.assert_called_once_with("container-123")

    @pytest.mark.asyncio
    async def test_active_containers_cleaned_after_run(self, orchestrator, base_request):
        await orchestrator.execute_tests(base_request)
        assert orchestrator.active_containers == {}

    @pytest.mark.asyncio
    async def test_source_file_written(self, orchestrator, base_request, docker_service):
        await orchestrator.execute_tests(base_request)
        calls = docker_service.create_file_in_container.call_args_list
        # Find the call that writes the main source file
        main_calls = [c for c in calls if c.args[1] == "/workspace/main.py"]
        assert len(main_calls) == 1
        assert main_calls[0].args[0] == "container-123"
        assert main_calls[0].args[2] == "def add(a, b): return a + b"

    @pytest.mark.asyncio
    async def test_test_files_written(self, orchestrator, base_request, docker_service):
        await orchestrator.execute_tests(base_request)
        calls = docker_service.create_file_in_container.call_args_list
        test_calls = [c for c in calls if c.args[1] == "/workspace/test_main.py"]
        assert len(test_calls) == 1
        assert test_calls[0].args[2] == "def test_add(): assert add(1,2)==3"

    @pytest.mark.asyncio
    async def test_exec_command_uses_correct_test_runner(
        self, orchestrator, base_request, docker_service
    ):
        await orchestrator.execute_tests(base_request)
        docker_service.exec_in_container.assert_called_once()
        call_args = docker_service.exec_in_container.call_args
        assert call_args.args[0] == "container-123"
        assert call_args.args[1] == ["python", "-m", "pytest", "-v", "--tb=short"]
        assert call_args.kwargs["timeout"] == 60


# ---------------------------------------------------------------------------
# execute_tests – error path
# ---------------------------------------------------------------------------


class TestExecuteTestsErrors:
    """Tests for error handling in execute_tests."""

    @pytest.mark.asyncio
    async def test_validation_timeout_too_long(self, orchestrator):
        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x = 1",
            timeout_seconds=601,
        )
        result = await orchestrator.execute_tests(req)
        assert result.success is False
        assert result.exit_code == 1
        # Pydantic validation may reject before our check, or our check catches it
        assert "timeout" in result.stderr.lower() or "Timeout" in result.stderr

    @pytest.mark.asyncio
    async def test_validation_code_too_large(self, orchestrator):
        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x" * 100001,
            timeout_seconds=60,
        )
        result = await orchestrator.execute_tests(req)
        assert result.success is False
        assert result.exit_code == 1
        assert "exceeds" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_container_creation_failure(self, orchestrator, base_request, docker_service):
        docker_service.create_test_container.side_effect = RuntimeError("boom")
        result = await orchestrator.execute_tests(base_request)
        assert result.success is False
        assert result.exit_code == 1
        assert "boom" in result.stderr
        assert result.container_id is None

    @pytest.mark.asyncio
    async def test_exec_timeout(self, orchestrator, base_request, docker_service):
        docker_service.exec_in_container.side_effect = TimeoutError("timed out")
        result = await orchestrator.execute_tests(base_request)
        # The timeout is caught inside _execute_in_container and returns exit_code 124
        assert result.exit_code == 124
        assert result.success is False
        assert "timed out" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_cleanup_still_runs_on_error(self, orchestrator, base_request, docker_service):
        docker_service.exec_in_container.side_effect = RuntimeError("exec fail")
        result = await orchestrator.execute_tests(base_request)
        docker_service.stop_container.assert_called_once_with("container-123")
        docker_service.remove_container_by_id.assert_called_once_with("container-123")
        assert result.success is False
        assert "exec fail" in result.stderr


# ---------------------------------------------------------------------------
# _collect_results branches
# ---------------------------------------------------------------------------


class TestCollectResults:
    """Test _collect_results with different test types."""

    @pytest.mark.asyncio
    async def test_coverage_type_collects_report(self, orchestrator, docker_service):
        req = TestExecutionRequest(
            language=TestLanguage.GO,
            test_type=OrchestratorType.COVERAGE,
            source_code="package main",
            timeout_seconds=60,
        )
        docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "", "stderr": ""}
        )
        result = await orchestrator.execute_tests(req)
        assert result.success is True
        assert result.exit_code == 0
        assert result.language == TestLanguage.GO
        assert result.test_type == OrchestratorType.COVERAGE

    @pytest.mark.asyncio
    async def test_benchmark_type(self, orchestrator, docker_service):
        req = TestExecutionRequest(
            language=TestLanguage.GO,
            test_type=OrchestratorType.BENCHMARK,
            source_code="package main",
            timeout_seconds=60,
        )
        docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "", "stderr": ""}
        )
        result = await orchestrator.execute_tests(req)
        assert result.success is True
        assert result.exit_code == 0
        assert result.language == TestLanguage.GO
        assert result.test_type == OrchestratorType.BENCHMARK

    @pytest.mark.asyncio
    async def test_security_type(self, orchestrator, docker_service):
        req = TestExecutionRequest(
            language=TestLanguage.GO,
            test_type=OrchestratorType.SECURITY,
            source_code="package main",
            timeout_seconds=60,
        )
        docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "", "stderr": ""}
        )
        result = await orchestrator.execute_tests(req)
        assert result.success is True
        assert result.exit_code == 0
        assert result.language == TestLanguage.GO
        assert result.test_type == OrchestratorType.SECURITY


# ---------------------------------------------------------------------------
# _parse_test_output
# ---------------------------------------------------------------------------


class TestParseTestOutput:
    """Test output parsing logic."""

    @pytest.mark.asyncio
    async def test_empty_stdout(self, orchestrator):
        assert await orchestrator._parse_test_output("", TestLanguage.GO) == []

    @pytest.mark.asyncio
    async def test_go_pass(self, orchestrator):
        results = await orchestrator._parse_test_output("PASS\nok test 0.5s", TestLanguage.GO)
        assert len(results) == 1
        assert results[0] == {"line": "PASS", "parsed": True}

    @pytest.mark.asyncio
    async def test_go_fail(self, orchestrator):
        results = await orchestrator._parse_test_output("FAIL\nFAIL test 1.0s", TestLanguage.GO)
        assert len(results) == 2
        assert results[0] == {"line": "FAIL", "parsed": True}
        assert results[1] == {"line": "FAIL test 1.0s", "parsed": True}

    @pytest.mark.asyncio
    async def test_go_mixed_output(self, orchestrator):
        output = "running tests...\nPASS\nok test 0.5s\nsome other output"
        results = await orchestrator._parse_test_output(output, TestLanguage.GO)
        assert len(results) == 1
        assert results[0]["line"] == "PASS"

    @pytest.mark.asyncio
    async def test_rust_test_result(self, orchestrator):
        output = "test result: ok. 5 passed; 0 failed"
        results = await orchestrator._parse_test_output(output, TestLanguage.RUST)
        assert len(results) == 1
        assert results[0] == {"line": "test result: ok. 5 passed; 0 failed", "parsed": True}

    @pytest.mark.asyncio
    async def test_rust_no_test_result_line(self, orchestrator):
        output = "compiling crate...\nrunning 3 tests"
        results = await orchestrator._parse_test_output(output, TestLanguage.RUST)
        assert results == []

    @pytest.mark.asyncio
    async def test_python_is_noop(self, orchestrator):
        results = await orchestrator._parse_test_output("PASS", TestLanguage.PYTHON)
        assert results == []

    @pytest.mark.asyncio
    async def test_javascript_is_noop(self, orchestrator):
        results = await orchestrator._parse_test_output("Tests passed", TestLanguage.JAVASCRIPT)
        assert results == []


# ---------------------------------------------------------------------------
# get_supported_languages / get_supported_test_types
# ---------------------------------------------------------------------------


class TestSupportedMethods:
    """Tests for get_supported_languages and get_supported_test_types."""

    @pytest.mark.asyncio
    async def test_get_supported_languages(self, orchestrator):
        langs = await orchestrator.get_supported_languages()
        expected = {
            "go",
            "rust",
            "java",
            "csharp",
            "cpp",
            "python",
            "nodejs",
            "javascript",
            "typescript",
        }
        assert set(langs) == expected

    @pytest.mark.asyncio
    async def test_get_supported_test_types_go(self, orchestrator):
        types = await orchestrator.get_supported_test_types("go")
        assert types == ["unit", "benchmark", "coverage", "race", "security", "vet"]

    @pytest.mark.asyncio
    async def test_get_supported_test_types_rust(self, orchestrator):
        types = await orchestrator.get_supported_test_types("rust")
        assert types == ["unit", "doc", "benchmark", "coverage", "security", "clippy", "fmt"]

    @pytest.mark.asyncio
    async def test_get_supported_test_types_unknown(self, orchestrator):
        types = await orchestrator.get_supported_test_types("brainfuck")
        # No container spec, returns []
        assert types == []

    @pytest.mark.asyncio
    async def test_get_supported_test_types_python(self, orchestrator):
        types = await orchestrator.get_supported_test_types("python")
        assert types == ["unit", "integration", "coverage", "security"]

    @pytest.mark.asyncio
    async def test_get_supported_test_types_java(self, orchestrator):
        types = await orchestrator.get_supported_test_types("java")
        assert types == ["unit", "integration", "coverage", "security"]

    @pytest.mark.asyncio
    async def test_get_supported_test_types_javascript(self, orchestrator):
        types = await orchestrator.get_supported_test_types("javascript")
        assert types == ["unit", "integration", "coverage", "security"]


# ---------------------------------------------------------------------------
# _create_language_config_files
# ---------------------------------------------------------------------------


class TestCreateLanguageConfigFiles:
    """Test language-specific config file creation."""

    @pytest.mark.asyncio
    async def test_go_creates_go_mod(self, orchestrator, docker_service):
        req = TestExecutionRequest(
            language=TestLanguage.GO,
            test_type=OrchestratorType.UNIT,
            source_code="package main",
            timeout_seconds=60,
        )
        await orchestrator._create_language_config_files("c1", req)
        docker_service.create_file_in_container.assert_called_once()
        call = docker_service.create_file_in_container.call_args
        assert call.args[0] == "c1"
        assert call.args[1] == "/workspace/go.mod"
        content = call.args[2]
        assert "module test-module" in content
        assert "go 1.21" in content

    @pytest.mark.asyncio
    async def test_rust_creates_cargo_toml(self, orchestrator, docker_service):
        req = TestExecutionRequest(
            language=TestLanguage.RUST,
            test_type=OrchestratorType.UNIT,
            source_code="fn main() {}",
            timeout_seconds=60,
        )
        await orchestrator._create_language_config_files("c1", req)
        docker_service.create_file_in_container.assert_called_once()
        call = docker_service.create_file_in_container.call_args
        assert call.args[0] == "c1"
        assert call.args[1] == "/workspace/Cargo.toml"
        content = call.args[2]
        assert 'name = "test-crate"' in content
        assert 'edition = "2021"' in content

    @pytest.mark.asyncio
    async def test_javascript_creates_package_json(self, orchestrator, docker_service):
        req = TestExecutionRequest(
            language=TestLanguage.JAVASCRIPT,
            test_type=OrchestratorType.UNIT,
            source_code="module.exports = {}",
            timeout_seconds=60,
        )
        await orchestrator._create_language_config_files("c1", req)
        docker_service.create_file_in_container.assert_called_once()
        call = docker_service.create_file_in_container.call_args
        assert call.args[0] == "c1"
        assert call.args[1] == "/workspace/package.json"
        import json

        content = json.loads(call.args[2])
        assert content["name"] == "test-project"
        assert content["scripts"]["test"] == "jest"
        assert "jest" in content["devDependencies"]
        # JavaScript should NOT have ts-jest
        assert "ts-jest" not in content["devDependencies"]

    @pytest.mark.asyncio
    async def test_typescript_creates_package_json_with_ts_deps(self, orchestrator, docker_service):
        req = TestExecutionRequest(
            language=TestLanguage.TYPESCRIPT,
            test_type=OrchestratorType.UNIT,
            source_code="export const x = 1;",
            timeout_seconds=60,
        )
        await orchestrator._create_language_config_files("c1", req)
        call = docker_service.create_file_in_container.call_args
        import json

        content = json.loads(call.args[2])
        assert content["scripts"]["test"] == "jest --preset ts-jest"
        assert content["devDependencies"]["ts-jest"] == "^29.0.0"
        assert content["devDependencies"]["@types/jest"] == "^29.0.0"
        assert content["devDependencies"]["typescript"] == "^5.0.0"

    @pytest.mark.asyncio
    async def test_python_creates_no_config(self, orchestrator, docker_service):
        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x = 1",
            timeout_seconds=60,
        )
        await orchestrator._create_language_config_files("c1", req)
        docker_service.create_file_in_container.assert_not_called()


# ---------------------------------------------------------------------------
# _build_result_from_raw (K8s path)
# ---------------------------------------------------------------------------


class TestBuildResultFromRaw:
    """Test the Kubernetes result builder."""

    @pytest.mark.asyncio
    async def test_builds_result_from_raw_success(self, orchestrator, base_request):
        raw = {"exit_code": 0, "stdout": "PASS", "stderr": ""}
        start = datetime.now(UTC)
        result = await orchestrator._build_result_from_raw(raw, base_request, "test-1", start)
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "PASS"
        assert result.stderr == ""
        assert result.test_id == "test-1"
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        assert result.container_id is None  # K8s path has no container_id
        assert result.created_at == start
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_builds_result_from_raw_failure(self, orchestrator, base_request):
        raw = {"exit_code": 1, "stdout": "", "stderr": "error"}
        start = datetime.now(UTC)
        result = await orchestrator._build_result_from_raw(raw, base_request, "test-2", start)
        assert result.success is False
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr == "error"
        assert result.test_id == "test-2"

    @pytest.mark.asyncio
    async def test_builds_result_from_raw_with_execution_time(self, orchestrator, base_request):
        raw = {"exit_code": 0, "stdout": "", "stderr": "", "execution_time_ms": 42.5}
        result = await orchestrator._build_result_from_raw(
            raw, base_request, "test-3", datetime.now(UTC)
        )
        assert result.execution_time_ms == 42.5


# ---------------------------------------------------------------------------
# _cleanup_container
# ---------------------------------------------------------------------------


class TestCleanupContainer:
    """Test container cleanup logic."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_from_active(self, orchestrator, docker_service):
        orchestrator.active_containers["t1"] = "c1"
        await orchestrator._cleanup_container("t1", "c1")
        assert "t1" not in orchestrator.active_containers
        docker_service.stop_container.assert_called_once_with("c1")
        docker_service.remove_container_by_id.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_cleanup_tolerates_errors(self, orchestrator, docker_service):
        docker_service.stop_container.side_effect = RuntimeError("stop fail")
        orchestrator.active_containers["t2"] = "c2"
        # Should not raise
        await orchestrator._cleanup_container("t2", "c2")
        # Container removal wasn't called because stop raised first
        # But active_containers should NOT have been cleaned (error path skips del)

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_key_still_works(self, orchestrator, docker_service):
        # Cleaning a test_id that's not in active_containers should not raise
        await orchestrator._cleanup_container("nonexistent", "c3")
        docker_service.stop_container.assert_called_once_with("c3")
        docker_service.remove_container_by_id.assert_called_once_with("c3")


# ---------------------------------------------------------------------------
# Coverage tests — helpers
# ---------------------------------------------------------------------------


def _make_test_request(**overrides) -> TestExecutionRequest:
    defaults = dict(
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        source_code="def add(a, b): return a + b",
        container_name="test-container",
    )
    defaults.update(overrides)
    return TestExecutionRequest(**defaults)


def _make_test_result(success=True, **overrides) -> TestExecutionResult:
    defaults = dict(
        test_id="test-1",
        success=success,
        exit_code=0 if success else 1,
        stdout="test passed" if success else "FAILED",
        stderr="" if success else "AssertionError",
        execution_time_ms=100.0,
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        container_name="test-container",
        created_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return TestExecutionResult(**defaults)


# ---------------------------------------------------------------------------
# Coverage tests — execute_suite_async
# ---------------------------------------------------------------------------


class TestExecuteSuiteAsync:
    @pytest.mark.asyncio
    async def test_parallel_execution_success(self):
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )

        suite = OrchestrationTestSuite(
            suite_id="suite-1",
            name="Test Suite",
            description="Test",
            container_name="container-1",
            tenant_id="tenant-1",
        )
        suite.tests = [_make_test_request(), _make_test_request()]

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_tests = AsyncMock(return_value=_make_test_result())

        await execute_suite_async(
            suite=suite,
            orchestrator=mock_orchestrator,
            parallel=True,
            max_workers=4,
            user_id="test-user",
        )

        assert suite.status == "completed"
        assert len(suite.results) == 2
        assert all(r.success for r in suite.results)
        assert all(r.exit_code == 0 for r in suite.results)
        assert mock_orchestrator.execute_tests.call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_execution_with_exception(self):
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )

        suite = OrchestrationTestSuite(
            suite_id="suite-2",
            name="Failing Suite",
            description="Test",
            container_name="container-1",
            tenant_id="tenant-1",
        )
        suite.tests = [_make_test_request()]

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_tests = AsyncMock(side_effect=Exception("execution failed"))

        await execute_suite_async(
            suite=suite,
            orchestrator=mock_orchestrator,
            parallel=True,
            max_workers=4,
            user_id="test-user",
        )

        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].success is False
        assert suite.results[0].exit_code == 1
        assert "execution failed" in suite.results[0].stderr

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )

        suite = OrchestrationTestSuite(
            suite_id="suite-3",
            name="Sequential Suite",
            description="Test",
            container_name="container-1",
            tenant_id="tenant-1",
        )
        suite.tests = [_make_test_request(), _make_test_request()]

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_tests = AsyncMock(return_value=_make_test_result())

        await execute_suite_async(
            suite=suite,
            orchestrator=mock_orchestrator,
            parallel=False,
            max_workers=1,
            user_id="test-user",
        )

        assert suite.status == "completed"
        assert len(suite.results) == 2
        assert all(r.success for r in suite.results)
        assert mock_orchestrator.execute_tests.call_count == 2

    @pytest.mark.asyncio
    async def test_sequential_with_exception(self):
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )

        suite = OrchestrationTestSuite(
            suite_id="suite-4",
            name="Fail Sequential",
            description="Test",
            container_name="container-1",
            tenant_id="tenant-1",
        )
        suite.tests = [_make_test_request()]

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_tests = AsyncMock(side_effect=Exception("boom"))

        await execute_suite_async(
            suite=suite,
            orchestrator=mock_orchestrator,
            parallel=False,
            max_workers=1,
            user_id="test-user",
        )

        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].success is False
        assert suite.results[0].exit_code == 1
        assert "boom" in suite.results[0].stderr
        # Sequential path preserves the original test's language/type
        assert suite.results[0].language == TestLanguage.PYTHON
        assert suite.results[0].test_type == OrchestratorType.UNIT

    @pytest.mark.asyncio
    async def test_overall_failure(self):
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )

        suite = OrchestrationTestSuite(
            suite_id="suite-5",
            name="Crash Suite",
            description="Test",
            container_name="container-1",
            tenant_id="tenant-1",
        )
        suite.tests = [_make_test_request()]

        # Make orchestrator itself fail on attribute access
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_tests = AsyncMock(
            side_effect=RuntimeError("orchestrator crashed")
        )

        await execute_suite_async(
            suite=suite,
            orchestrator=mock_orchestrator,
            parallel=True,
            max_workers=4,
            user_id="test-user",
        )

        # Should be completed (exception results are wrapped)
        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].success is False
        assert "orchestrator crashed" in suite.results[0].stderr


# ---------------------------------------------------------------------------
# Coverage tests — execute_plan_async
# ---------------------------------------------------------------------------


class TestExecutePlanAsync:
    @pytest.mark.asyncio
    async def test_plan_execution_success(self):
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            TestOrchestrationPlan,
            execute_plan_async,
        )

        suite1 = OrchestrationTestSuite(
            suite_id="s1",
            name="Suite 1",
            description="Test",
            container_name="container-1",
            tenant_id="tenant-1",
        )
        suite1.tests = [_make_test_request()]

        suite2 = OrchestrationTestSuite(
            suite_id="s2",
            name="Suite 2",
            description="Test",
            container_name="container-1",
            tenant_id="tenant-1",
        )
        suite2.tests = [_make_test_request()]

        plan = TestOrchestrationPlan("plan-1", "container-1", "tenant-1")
        plan.suites = [suite1, suite2]
        plan.parallel_groups = [["s1", "s2"]]

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_tests = AsyncMock(return_value=_make_test_result())

        await execute_plan_async(
            plan=plan,
            orchestrator=mock_orchestrator,
            user_id="test-user",
        )

        assert plan.status == "completed"
        assert suite1.status == "completed"
        assert suite2.status == "completed"
        assert len(suite1.results) == 1
        assert len(suite2.results) == 1
        assert suite1.results[0].success is True
        assert suite2.results[0].success is True

    @pytest.mark.asyncio
    async def test_plan_execution_failure(self):
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            TestOrchestrationPlan,
            execute_plan_async,
        )

        suite = OrchestrationTestSuite(
            suite_id="s1",
            name="Suite",
            description="Test",
            container_name="container-1",
            tenant_id="tenant-1",
        )
        suite.tests = [_make_test_request()]

        plan = TestOrchestrationPlan("plan-2", "container-1", "tenant-1")
        plan.suites = [suite]
        plan.parallel_groups = [["s1"]]

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_tests = AsyncMock(side_effect=Exception("boom"))

        await execute_plan_async(
            plan=plan,
            orchestrator=mock_orchestrator,
            user_id="test-user",
        )

        # Even with failed tests, plan completes
        assert plan.status == "completed"
        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].success is False
        assert "boom" in suite.results[0].stderr


# ---------------------------------------------------------------------------
# Coverage tests — get_orchestrator
# ---------------------------------------------------------------------------


class TestGetOrchestrator:
    @pytest.mark.asyncio
    async def test_creates_orchestrator(self):
        from routers.test_orchestration import get_orchestrator

        mock_docker = MagicMock()
        mock_orch = MagicMock(spec=TestOrchestrator)

        with patch("routers.test_orchestration._orchestrator", None):
            with patch("routers.test_orchestration.get_docker_service") as mock_get_docker:
                mock_get_docker.return_value = mock_docker
                with patch("routers.test_orchestration.create_test_orchestrator") as mock_create:
                    mock_create.return_value = mock_orch
                    result = await get_orchestrator()
                    assert result is mock_orch
                    mock_get_docker.assert_called_once()
                    mock_create.assert_called_once()
                    # Verify config passed to factory
                    config_arg = mock_create.call_args[0][1]
                    assert config_arg.timeout == 300
                    assert config_arg.parallel is True
                    assert config_arg.max_workers == 8


# ---------------------------------------------------------------------------
# Coverage tests — OrchestrationTestSuite and TestOrchestrationPlan classes
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_orchestration_test_suite_defaults(self):
        from routers.test_orchestration import OrchestrationTestSuite

        suite = OrchestrationTestSuite(
            suite_id="s1",
            name="test",
            description="desc",
            container_name="c1",
            tenant_id="t1",
        )
        assert suite.suite_id == "s1"
        assert suite.name == "test"
        assert suite.description == "desc"
        assert suite.container_name == "c1"
        assert suite.tenant_id == "t1"
        assert suite.status == "pending"
        assert suite.tests == []
        assert suite.results == []
        assert suite.metadata == {}
        assert isinstance(suite.created_at, datetime)

    def test_test_orchestration_plan_defaults(self):
        from routers.test_orchestration import TestOrchestrationPlan

        plan = TestOrchestrationPlan("p1", "c1", "t1")
        assert plan.plan_id == "p1"
        assert plan.container_name == "c1"
        assert plan.tenant_id == "t1"
        assert plan.status == "draft"
        assert plan.suites == []
        assert plan.execution_order == []
        assert plan.parallel_groups == []
        assert isinstance(plan.created_at, datetime)
        # dependencies should be a defaultdict(set)
        assert plan.dependencies == {}


# ---------------------------------------------------------------------------
# Coverage tests — _calculate_execution_order edge cases
# ---------------------------------------------------------------------------


class TestCalculateExecutionOrderEdgeCases:
    def test_no_dependencies(self):
        from routers.test_orchestration import _calculate_execution_order

        result = _calculate_execution_order(["a", "b", "c"], {})
        assert result == ["a", "b", "c"]

    def test_simple_dependency_chain(self):
        from routers.test_orchestration import _calculate_execution_order

        # b depends on a, so a should come before b
        deps = {"b": {"a"}}
        result = _calculate_execution_order(["a", "b"], deps)
        assert result == ["a", "b"]

    def test_dependency_outside_suite_ignored(self):
        from routers.test_orchestration import _calculate_execution_order

        deps = {"b": {"x"}}  # x is not in suite_ids
        result = _calculate_execution_order(["a", "b"], deps)
        assert set(result) == {"a", "b"}
        assert len(result) == 2

    def test_circular_dependency_doesnt_loop_forever(self):
        from routers.test_orchestration import _calculate_execution_order

        # The function uses visited set, so circular deps don't cause infinite loop
        deps = {"a": {"b"}, "b": {"a"}}
        result = _calculate_execution_order(["a", "b"], deps)
        assert set(result) == {"a", "b"}
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Coverage tests — _calculate_parallel_groups edge cases
# ---------------------------------------------------------------------------


class TestCalculateParallelGroupsEdgeCases:
    def test_empty_input(self):
        from routers.test_orchestration import _calculate_parallel_groups

        result = _calculate_parallel_groups([], {})
        assert result == []

    def test_no_dependencies_single_group(self):
        from routers.test_orchestration import _calculate_parallel_groups

        result = _calculate_parallel_groups(["a", "b", "c"], {})
        # No deps, all can run in parallel as one group
        assert len(result) == 1
        assert set(result[0]) == {"a", "b", "c"}

    def test_simple_dependency_two_groups(self):
        from routers.test_orchestration import _calculate_parallel_groups

        # b depends on a, so we need two groups: [a] then [b]
        deps = {"b": {"a"}}
        result = _calculate_parallel_groups(["a", "b"], deps)
        assert len(result) == 2
        assert "a" in result[0]
        assert "b" in result[1]

    def test_impossible_deps_breaks_deadlock(self):
        """When deps can't be satisfied, remaining items are grouped together."""
        from routers.test_orchestration import _calculate_parallel_groups

        # All depend on each other -- none have deps satisfied, so all go in one group
        deps = {"a": {"b"}, "b": {"a"}}
        result = _calculate_parallel_groups(["a", "b"], deps)
        # All items should be included across groups
        all_items = set()
        for group in result:
            all_items.update(group)
        assert all_items == {"a", "b"}
        # The deadlock-breaking code puts remaining in a single group
        assert len(result) == 1
        assert set(result[0]) == {"a", "b"}


# ---------------------------------------------------------------------------
# Defect 1 control: "no test command defined" must never be a pass.
#
# Revert guard. These tests fail if the ["echo", "No test command defined"]
# fallback is restored, because `echo` exits 0 and `success = exit_code == 0`.
# ---------------------------------------------------------------------------


class TestNoTestCommandIsNeverAPass:
    """A (language, test_type) pair with no command verified NOTHING."""

    # Measured blast radius of the old echo fallback: every pair below reported
    # a green suite having executed nothing.
    UNMAPPED_PAIRS = [
        (TestLanguage.PYTHON, OrchestratorType.INTEGRATION),
        (TestLanguage.PYTHON, OrchestratorType.SECURITY),
        (TestLanguage.PYTHON, OrchestratorType.BENCHMARK),
        (TestLanguage.JAVASCRIPT, OrchestratorType.INTEGRATION),
        (TestLanguage.TYPESCRIPT, OrchestratorType.SECURITY),
        (TestLanguage.JAVA, OrchestratorType.UNIT),
        (TestLanguage.CSHARP, OrchestratorType.UNIT),
        (TestLanguage.CPP, OrchestratorType.UNIT),
    ]

    @pytest.mark.parametrize("language,test_type", UNMAPPED_PAIRS)
    def test_unmapped_pair_has_no_command(self, orchestrator, language, test_type):
        assert orchestrator._get_test_command(language, test_type) is None

    def test_no_mapped_command_can_be_a_trivially_passing_echo(self, orchestrator):
        """Sweep EVERY (language, test_type): no command may be a bare `echo`.

        A command that always exits 0 is indistinguishable from a passing
        suite, so the sweep — not just the single JAVA/UNIT case — is the
        control.
        """
        checked = 0
        for language in TestLanguage:
            for test_type in OrchestratorType:
                cmd = orchestrator._get_test_command(language, test_type)
                checked += 1
                if cmd is None:
                    continue
                assert cmd[0] != "echo", f"{language}/{test_type} runs a no-op that exits 0: {cmd}"
        # Guard against a vacuous sweep (an empty enum would make this green).
        assert checked == len(list(TestLanguage)) * len(list(OrchestratorType))
        assert checked > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language,test_type", UNMAPPED_PAIRS)
    async def test_execute_tests_reports_not_run_never_success(
        self, orchestrator, docker_service, language, test_type
    ):
        """End-to-end: the verdict for an unmapped pair is NOT-RUN, not PASS."""
        from services.test_execution.test_orchestrator import (
            NO_TEST_COMMAND_EXIT_CODE,
            NO_TEST_COMMAND_REASON,
        )

        request = TestExecutionRequest(
            language=language,
            test_type=test_type,
            source_code="# nothing to see here",
        )

        result = await orchestrator.execute_tests(request)

        assert result.success is False, "a suite that ran no tests must not be a pass"
        assert result.not_run is True
        assert result.not_run_reason == NO_TEST_COMMAND_REASON
        assert result.exit_code == NO_TEST_COMMAND_EXIT_CODE
        assert result.exit_code != 0
        assert "No test command is defined" in result.stderr
        # And nothing was executed: no container was created or exec'd into.
        docker_service.create_test_container.assert_not_called()
        docker_service.exec_in_container.assert_not_called()

    @pytest.mark.asyncio
    async def test_mapped_pair_still_passes(self, orchestrator, base_request):
        """Negative control: a pair that DOES have a command still passes.

        Without this, making everything fail would also make the tests above
        green -- and a gate that can only go red is as useless as one that can
        only go green.
        """
        result = await orchestrator.execute_tests(base_request)
        assert result.not_run is False
        assert result.not_run_reason is None
        assert result.success is True
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_in_container_guards_independently(self, orchestrator, docker_service):
        """_execute_in_container is independently callable and must also refuse."""
        from services.test_execution.test_orchestrator import NO_TEST_COMMAND_EXIT_CODE

        request = TestExecutionRequest(
            language=TestLanguage.JAVA,
            test_type=OrchestratorType.UNIT,
            source_code="class X {}",
        )
        raw = await orchestrator._execute_in_container("container-123", request, "t-1")

        assert raw["exit_code"] == NO_TEST_COMMAND_EXIT_CODE
        assert raw["not_run"] is True
        docker_service.exec_in_container.assert_not_called()


# ---------------------------------------------------------------------------
# The Docker call site's not-run read (_collect_results) -- producer -> reader
# ---------------------------------------------------------------------------


class TestDockerCallSiteReadsNotRun:
    """Pin the hop that `execute_tests` can never exercise.

    `_execute_in_container` PRODUCES `not_run: True` for an unmapped
    (language, test_type) pair, and `_collect_results` READS it back through
    `_read_not_run`. Both halves already existed; nothing joined them, which
    is why disabling the read produced zero failures across the whole suite.

    The whole-flow path cannot cover this: `execute_tests` short-circuits at
    the `_get_test_command(...) is None` guard *before* creating a container,
    so the producer branch is only reachable by calling the method directly --
    which is exactly what its own "independently callable" docstring promises.
    """

    @staticmethod
    def _unmapped_request() -> TestExecutionRequest:
        # python maps UNIT and COVERAGE only, so INTEGRATION has no command.
        # INTEGRATION also collects no coverage/benchmark/security artefacts,
        # keeping this test to the one hop it is about.
        return TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.INTEGRATION,
            source_code="def add(a, b): return a + b",
        )

    @pytest.mark.asyncio
    async def test_not_run_survives_the_hop_into_collect_results(
        self, orchestrator, docker_service
    ):
        """not_run produced by _execute_in_container reaches the result."""
        from services.test_execution.test_orchestrator import NO_TEST_COMMAND_REASON

        request = self._unmapped_request()
        raw = await orchestrator._execute_in_container("container-123", request, "t-hop")

        # Precondition: the producer really did emit the fact. Without this the
        # assertion below could pass for the wrong reason (nothing to carry).
        assert raw["not_run"] is True

        result = await orchestrator._collect_results(
            "container-123", raw, request, "t-hop", datetime.now(UTC)
        )

        # This is the assertion that reds when the read at the Docker call site
        # is disabled (`not_run=False`).
        assert result.not_run is True
        assert result.not_run_reason == NO_TEST_COMMAND_REASON
        assert result.success is False
        docker_service.exec_in_container.assert_not_called()

    @pytest.mark.asyncio
    async def test_executor_spelling_also_survives_the_hop(self, orchestrator):
        """A raw dict using the executors' `no_tests_run` spelling propagates."""
        request = self._unmapped_request()
        raw = {
            "exit_code": 0,
            "stdout": "collected 0 items",
            "stderr": "",
            "execution_time_ms": 12.0,
            "no_tests_run": True,
        }

        result = await orchestrator._collect_results(
            "container-123", raw, request, "t-exec", datetime.now(UTC)
        )

        assert result.not_run is True
        # exit_code 0 would otherwise have been graded a pass -- this is the
        # zero-collection fake green the marker exists to stop.
        assert result.success is False

    @pytest.mark.asyncio
    async def test_camelcase_spelling_also_survives_the_hop(self, orchestrator):
        """csharp/cpp emit `noTestsRun`; it must not be read as "tests ran"."""
        request = self._unmapped_request()
        raw = {
            "exit_code": 0,
            "stdout": "No test is available",
            "stderr": "",
            "execution_time_ms": 8.0,
            "noTestsRun": True,
        }

        result = await orchestrator._collect_results(
            "container-123", raw, request, "t-camel", datetime.now(UTC)
        )

        assert result.not_run is True
        assert result.success is False

    @pytest.mark.asyncio
    async def test_all_passed_is_not_reported_as_not_run(self, orchestrator):
        """Outcome 2 of 3: a real green run must stay green and not-not-run."""
        request = self._unmapped_request()
        raw = {
            "exit_code": 0,
            "stdout": "collected 3 items\n3 passed",
            "stderr": "",
            "execution_time_ms": 30.0,
        }

        result = await orchestrator._collect_results(
            "container-123", raw, request, "t-pass", datetime.now(UTC)
        )

        assert result.not_run is False
        assert result.success is True

    @pytest.mark.asyncio
    async def test_genuine_failure_is_not_reported_as_not_run(self, orchestrator):
        """Outcome 3 of 3: a real red run is a failure, not a not-run.

        Without this leg, hard-wiring `not_run=True` at the call site would
        satisfy every other assertion here.
        """
        request = self._unmapped_request()
        raw = {
            "exit_code": 1,
            "stdout": "collected 3 items\n1 failed, 2 passed",
            "stderr": "",
            "execution_time_ms": 30.0,
        }

        result = await orchestrator._collect_results(
            "container-123", raw, request, "t-fail", datetime.now(UTC)
        )

        assert result.not_run is False
        assert result.not_run_reason is None
        assert result.success is False

    @pytest.mark.asyncio
    async def test_executor_that_omits_the_field_is_unaffected(self, orchestrator):
        """Additive and defensive: no field means no change in behaviour."""
        request = self._unmapped_request()
        raw = {"exit_code": 0, "stdout": "ok", "stderr": "", "execution_time_ms": 1.0}

        result = await orchestrator._collect_results(
            "container-123", raw, request, "t-absent", datetime.now(UTC)
        )

        assert result.not_run is False
        assert result.success is True
