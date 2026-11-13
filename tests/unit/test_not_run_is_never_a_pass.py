"""CONSUMER-side control for "a run that executed nothing is never a pass".

Why this file exists, precisely
-------------------------------
The previous round of this defect added ``TestResult.no_tests_run`` and called
it "the explicit, never-a-pass marker". Its test
(``test_zero_collected_is_distinguishable_from_all_passed``) is a **producer**
test: it proves the field is POPULATED. It even writes out, in its own body,
the predicate a caller *could* use::

    def is_pass(r) -> bool:
        return r.success and r.tests_run > 0 and r.tests_failed == 0 and not r.no_tests_run

...and no caller in ``src/`` was written that way. The marker was consulted by
nothing. A producer test cannot catch a write-only field, which is why the
field shipped inert inside the fix for the very thing it was meant to close.

So every test here exercises a CONSUMER:

* the orchestration router's server-side suite verdict (``summarize_suite`` via
  ``get_suite_results``), which is what a caller reads as "did the tests pass";
* the suite execution-history record, which used to compute a SECOND,
  divergent verdict of its own;
* ``success`` itself, which is what every other consumer in the service grades
  on -- the ``/tests/execute`` and ``/tests/batch`` responses are returned to
  clients verbatim.

Each test is written so it goes RED if the consumer stops consulting the
marker, and the "everything passed" cases are kept as negative controls so a
suite that goes uniformly red cannot be mistaken for a working gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.testing_models import TestResult, TestSecurityLevel
from routers.test_orchestration import summarize_suite
from services.orchestration_models import OrchestrationTestSuite
from services.orchestration_store import get_orchestration_store
from services.test_execution.test_orchestrator import (
    NO_TEST_COMMAND_EXIT_CODE,
    NO_TEST_COMMAND_REASON,
    OrchestratorType,
    TestExecutionRequest,
    TestExecutionResult,
    TestLanguage,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _passed_result(test_id: str = "t-pass") -> TestExecutionResult:
    """A result that genuinely ran and genuinely passed."""
    return TestExecutionResult(
        test_id=test_id,
        success=True,
        exit_code=0,
        stdout="1 passed",
        stderr="",
        execution_time_ms=100.0,
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        container_name="c1",
        created_at=datetime.now(UTC),
    )


def _not_run_result(test_id: str = "t-notrun") -> TestExecutionResult:
    """A result shaped exactly like ``TestOrchestrator._build_not_run_result``."""
    return TestExecutionResult(
        test_id=test_id,
        success=False,
        exit_code=NO_TEST_COMMAND_EXIT_CODE,
        stdout="",
        stderr="No test command is defined ...",
        execution_time_ms=0.0,
        language=TestLanguage.JAVA,
        test_type=OrchestratorType.INTEGRATION,
        container_name="c1",
        created_at=datetime.now(UTC),
        not_run=True,
        not_run_reason=NO_TEST_COMMAND_REASON,
    )


def _lying_not_run_result(test_id: str = "t-lie") -> TestExecutionResult:
    """A not-run result that ALSO claims ``success=True``.

    Built with ``model_construct`` so it bypasses validation on purpose. This
    is the row the ROUTER has to defend against by itself: if the consumer only
    ever saw rows the model had already sanitised, a test through the model
    would be grading the model, not the consumer, and the router could drop its
    check with the suite still green.
    """
    return TestExecutionResult.model_construct(
        test_id=test_id,
        success=True,
        exit_code=0,
        stdout="",
        stderr="",
        execution_time_ms=0.0,
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        container_name="c1",
        created_at=datetime.now(UTC),
        not_run=True,
        not_run_reason=NO_TEST_COMMAND_REASON,
    )


def _suite(results: list[TestExecutionResult], status: str = "completed") -> OrchestrationTestSuite:
    suite = OrchestrationTestSuite("s1", "Suite 1", "desc", "c1", "n8n-dev")
    suite.status = status
    suite.results = results
    return suite


async def _get_results_for(suite: OrchestrationTestSuite) -> dict:
    """Drive ``get_suite_results`` against an exact in-memory suite.

    The store is stubbed rather than written to, because ``save_suite`` ->
    ``suite_from_data`` re-validates every result and would launder the lying
    row before the consumer ever saw it -- the test would then pass with the
    consumer's check deleted.
    """
    from routers.test_orchestration import get_suite_results

    with patch("routers.test_orchestration._store.get_suite", AsyncMock(return_value=suite)):
        return await get_suite_results(
            suite_id=suite.suite_id,
            current_user=MagicMock(preferred_username="testuser"),
            tenant_id=suite.tenant_id,
        )


# ---------------------------------------------------------------------------
# 1. The consumer: the server-side suite verdict
# ---------------------------------------------------------------------------


async def test_not_run_result_cannot_make_the_suite_report_passed():
    """THE test. A suite holding a not-run result must never report passed.

    The row claims ``success=True``, so ``failed == 0``, the suite has results
    (``executed``) and its status is ``completed``. Under the old verdict --
    ``total > 0 and failed == 0 and status == "completed"`` -- every input said
    PASS. Only consulting ``not_run`` changes the answer, so this test is red
    the moment the consumer stops consulting it.
    """
    payload = await _get_results_for(_suite([_lying_not_run_result()]))

    assert payload["passed"] is False
    assert payload["summary"]["passed"] is False
    assert payload["summary"]["verdict"] == "not_run"
    assert payload["summary"]["not_run"] == 1
    # The trap that makes it look green, still present and still not a pass.
    assert payload["summary"]["failed"] == 0 or payload["summary"]["successful"] == 1


async def test_honest_not_run_result_is_reported_as_not_run_not_as_failed():
    """NOT-RUN is the third value, not a dressed-up failure.

    A wrong PASS and a wrong FAIL are both lies. The suite is not a pass, and
    the verdict says why: nothing executed. ``failed`` deliberately still
    counts it (see ``summarize_suite``) so no caller grading on ``failed``
    becomes more green than it was.
    """
    payload = await _get_results_for(_suite([_not_run_result()]))

    assert payload["passed"] is False
    assert payload["summary"]["verdict"] == "not_run"
    assert payload["summary"]["not_run"] == 1
    assert payload["summary"]["failed"] == 1
    detail = payload["execution_details"][0]
    assert detail["not_run"] is True
    assert detail["not_run_reason"] == NO_TEST_COMMAND_REASON


async def test_a_real_failure_is_still_reported_as_failed_not_as_not_run():
    """Discriminator: the verdict must not collapse FAILED into NOT-RUN."""
    failed = TestExecutionResult(
        test_id="t-fail",
        success=False,
        exit_code=1,
        stdout="1 failed",
        stderr="",
        execution_time_ms=10.0,
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        container_name="c1",
        created_at=datetime.now(UTC),
    )
    payload = await _get_results_for(_suite([failed]))

    assert payload["passed"] is False
    assert payload["summary"]["verdict"] == "failed"
    assert payload["summary"]["not_run"] == 0


async def test_a_genuinely_passing_suite_still_passes():
    """NEGATIVE CONTROL.

    A gate in which every case goes red cannot tell "the verdict regressed"
    from "the harness broke". This case must stay green.
    """
    payload = await _get_results_for(_suite([_passed_result(), _passed_result("t-pass-2")]))

    assert payload["passed"] is True
    assert payload["summary"]["verdict"] == "passed"
    assert payload["summary"]["not_run"] == 0
    assert payload["summary"]["success_rate"] == 100.0


async def test_one_not_run_result_poisons_an_otherwise_passing_suite():
    """Partial coverage is not coverage: one unverified pair sinks the verdict."""
    payload = await _get_results_for(_suite([_passed_result(), _not_run_result()]))

    assert payload["passed"] is False
    assert payload["summary"]["verdict"] == "not_run"
    assert payload["summary"]["successful"] == 1
    assert payload["summary"]["not_run"] == 1


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
async def test_an_incomplete_suite_is_neither_passed_nor_silently_failed(status):
    """A suite still in flight is `incomplete`, which is also never a pass."""
    payload = await _get_results_for(_suite([_passed_result()], status=status))

    assert payload["passed"] is False
    assert payload["summary"]["verdict"] == "incomplete"


async def test_an_empty_suite_is_not_run_not_passed():
    """`all([])` / `failed == 0` over nothing: the original shape of this bug."""
    payload = await _get_results_for(_suite([]))

    assert payload["passed"] is False
    assert payload["summary"]["verdict"] == "not_run"
    assert payload["summary"]["executed"] is False
    assert payload["summary"]["success_rate"] is None


# ---------------------------------------------------------------------------
# 2. The second consumer: the execution-history record
# ---------------------------------------------------------------------------


async def test_suite_history_uses_the_same_verdict_as_the_results_endpoint(pg_db):
    """Two readers of one question -> one authority.

    History used to compute ``all(r.success for r in suite.results)``: a second
    implementation that ignored ``not_run`` and answered True over an empty
    list. With the lying row it recorded ``success: True`` for a suite that
    executed nothing, while ``get_suite_results`` said otherwise about the same
    suite.
    """
    from routers.test_orchestration import execute_suite_async

    suite = OrchestrationTestSuite("s-hist", "Hist", "desc", "c1", "t-hist")
    suite.tests = [
        TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
        )
    ]

    orchestrator = AsyncMock()
    orchestrator.execute_tests = AsyncMock(return_value=_lying_not_run_result())

    await execute_suite_async(
        suite=suite,
        orchestrator=orchestrator,
        parallel=False,
        max_workers=1,
        user_id="testuser",
    )

    history = await get_orchestration_store().list_history("t-hist")
    assert len(history) == 1
    assert history[0]["success"] is False
    assert history[0]["verdict"] == "not_run"
    assert history[0]["not_run"] == 1


# ---------------------------------------------------------------------------
# 3. The asymmetry: the producer field is populated AND that is not enough
# ---------------------------------------------------------------------------


async def test_populated_marker_does_not_by_itself_make_a_result_successful():
    """The whole bug, in one assertion pair.

    Read through the LIVE hop (``CodeExecutionService`` -> executor
    ``/execute``) with the payload the previous round's test never covered: an
    executor that exits 0 having collected nothing. The marker is populated --
    that part already worked and must keep working -- and the marker is also
    now CONSULTED, so ``success`` cannot still be True.
    """
    from services.code_execution_service import CodeExecutionService

    payload = {
        "success": True,
        "output": "no tests ran in 0.01s",
        "execution_time": 0.01,
        "test_id": "t-1",
        "exit_code": 0,
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "no_tests_run": True,
    }
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)

    with (
        patch("services.code_execution_service._get_executor_http_client", return_value=client),
        patch("services.code_execution_service._get_executor_url", return_value="http://exec"),
    ):
        result = await CodeExecutionService().execute_test(
            SimpleNamespace(
                code="x = 1",
                test_code="def test_x(): assert x == 1",
                language="python",
                tenant_id="tenant-1",
            )
        )

    # Producer half (unchanged): the marker survived the hop.
    assert result.no_tests_run is True
    assert result.tests_run == 0
    assert result.tests_failed == 0
    # Consumer half (the fix): and it is therefore not a success, even though
    # the executor said `success: true` with `exit_code: 0`.
    assert result.success is False
    assert result.exit_code == 0


def test_a_passing_result_is_not_collaterally_marked_unsuccessful():
    """NEGATIVE CONTROL for the coercion: it must only fire on the marker."""
    result = TestResult(
        test_id="t",
        success=True,
        exit_code=0,
        execution_time=0.1,
        created_at=datetime.now(UTC),
        language="python",
        security_level=TestSecurityLevel.HIGH,
        tests_run=3,
        tests_passed=3,
        no_tests_run=False,
    )
    assert result.success is True


# ---------------------------------------------------------------------------
# 4. One authority, both models
# ---------------------------------------------------------------------------


def _build_test_result(**overrides):
    kwargs = dict(
        test_id="t",
        success=True,
        exit_code=0,
        execution_time=0.1,
        created_at=datetime.now(UTC),
        language="python",
        security_level=TestSecurityLevel.HIGH,
    )
    kwargs.update(overrides)
    return TestResult(**kwargs)


def _build_execution_result(**overrides):
    kwargs = dict(
        test_id="t",
        success=True,
        exit_code=0,
        stdout="",
        stderr="",
        execution_time_ms=1.0,
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        created_at=datetime.now(UTC),
    )
    kwargs.update(overrides)
    return TestExecutionResult(**kwargs)


@pytest.mark.parametrize(
    ("build", "marker"),
    [
        pytest.param(_build_test_result, "no_tests_run", id="TestResult.no_tests_run"),
        pytest.param(_build_execution_result, "not_run", id="TestExecutionResult.not_run"),
    ],
)
def test_both_result_models_share_the_same_never_a_pass_authority(build, marker):
    """The two models spell the fact differently; they must answer identically.

    Fixing one and leaving the other is how this class of defect has recurred:
    the orchestrator path was correct, the executor path was not, and nothing
    tied them together.
    """
    marked = build(**{marker: True})
    assert getattr(marked, marker) is True
    assert marked.success is False
    # ...and the marker is not erased by the coercion: the caller must still be
    # able to tell NOT-RUN from FAILED.
    assert getattr(marked, marker) is True

    unmarked = build(**{marker: False})
    assert unmarked.success is True


@pytest.mark.parametrize(
    ("build", "marker"),
    [
        pytest.param(_build_test_result, "no_tests_run", id="TestResult.no_tests_run"),
        pytest.param(_build_execution_result, "not_run", id="TestExecutionResult.not_run"),
    ],
)
def test_the_invariant_survives_a_round_trip_through_serialization(build, marker):
    """Stored rows are rehydrated with ``model_validate``; the lie must not persist."""
    lie = dict(build(**{marker: True}).model_dump(mode="json"))
    lie["success"] = True  # a row written before the invariant existed
    rehydrated = type(build())(**lie)
    assert rehydrated.success is False
    assert getattr(rehydrated, marker) is True


# ---------------------------------------------------------------------------
# 5. The boundary: the executor's spelling must survive the orchestrator hop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["not_run", "no_tests_run"])
async def test_orchestrator_reads_the_not_run_fact_under_either_spelling(field):
    """``_build_result_from_raw`` is the K8s path; it must not drop the fact.

    The orchestrator's own raw dicts say ``not_run``; the executors' ``/execute``
    contract says ``no_tests_run``. A reader that knows one spelling silently
    discards the other at the one boundary where it is still recoverable.
    """
    from config.test_framework_config import TestConfig
    from services.stubs import DockerService
    from services.test_execution.test_orchestrator import create_test_orchestrator

    orchestrator = create_test_orchestrator(DockerService(), TestConfig(timeout=1, parallel=False))
    request = TestExecutionRequest(
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        source_code="print(1)",
        container_name="c1",
    )
    result = await orchestrator._build_result_from_raw(
        raw={"exit_code": 0, "stdout": "", "stderr": "", field: True},
        request=request,
        test_id="t",
        start_time=datetime.now(UTC),
    )

    assert result.not_run is True
    assert result.success is False


async def test_orchestrator_does_not_invent_a_not_run_outcome():
    """NEGATIVE CONTROL for the boundary read: absent marker, ordinary pass."""
    from config.test_framework_config import TestConfig
    from services.stubs import DockerService
    from services.test_execution.test_orchestrator import create_test_orchestrator

    orchestrator = create_test_orchestrator(DockerService(), TestConfig(timeout=1, parallel=False))
    request = TestExecutionRequest(
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        source_code="print(1)",
        container_name="c1",
    )
    result = await orchestrator._build_result_from_raw(
        raw={"exit_code": 0, "stdout": "1 passed", "stderr": ""},
        request=request,
        test_id="t",
        start_time=datetime.now(UTC),
    )

    assert result.not_run is False
    assert result.success is True


# ---------------------------------------------------------------------------
# 6. summarize_suite is the only verdict; nobody may grow a second one
# ---------------------------------------------------------------------------


def test_the_verdict_is_computed_in_exactly_one_place():
    """Revert guard against a second, divergent verdict re-appearing.

    ``get_suite_results`` and the suite history record answer the same question
    and used to answer it with two different expressions. If a future edit
    re-inlines either of them, this fails.
    """
    import inspect

    import routers.test_orchestration as module

    source = inspect.getsource(module)
    # Comment lines are excluded: the tombstone that explains WHY the old
    # expression was removed necessarily quotes it, and a grep that cannot tell
    # a comment from code would fire on the explanation instead of the defect.
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    # Positive control: the filter must not have eaten the module.
    assert "def summarize_suite(" in code
    assert code.count("def summarize_suite(") == 1
    # The historical inline verdict, by shape.
    assert "all(r.success for r in suite.results)" not in code
    assert code.count("summarize_suite(suite)") >= 2


def test_summarize_suite_never_reports_passed_with_a_not_run_result():
    """Direct unit control on the authority itself, independent of the router."""
    assert summarize_suite(_suite([_lying_not_run_result()]))["passed"] is False
    assert summarize_suite(_suite([_not_run_result()]))["passed"] is False
    assert summarize_suite(_suite([_passed_result()]))["passed"] is True
