"""The executor `/execute` response contract, on the coding-testing side.

Two defects of the same class lived here, and the first is why nobody could see
the second:

1. Every language executor returns the combined process stream as ``output``
   (see its ``TestResponse`` model). Both readers asked only for ``stdout`` /
   ``stderr``, so ``TestResult.stdout`` and ``.stderr`` were the empty string on
   EVERY successful call — pytest's failure report included.
2. The executors report ``tests_run`` / ``tests_passed`` / ``tests_failed`` /
   ``no_tests_run``, but ``TestResult`` is ``extra="forbid"`` and had no fields
   for them, so a run that collected ZERO tests arrived indistinguishable from a
   run in which everything passed — only ``exit_code`` survived the hop.

**There are TWO readers of this contract, and the contract is one contract.**
``CodeExecutionService`` is the live hop (``coding-testing`` →
``*_TEST_SERVICE_URL`` executors). ``HTTPTestRunner`` is the aiohttp reader of
the same ``/execute`` response; it is currently dormant — nothing in ``src/``
instantiates it, it is only exported from
``processing.test_execution.__init__`` — but "dormant" is not "cannot be
revived", and a dormant reader with no regression control is exactly how the
next person reverts a correct mapping and ships the old behaviour under a green
suite. So every test below is parametrised over BOTH readers and the mapping is
governed once, for both, rather than per-reader.

These tests fail if either mapping is reverted, in either reader.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.testing_models import TestRequest
from processing.test_execution.http_test_runner import HTTPTestRunner
from services.code_execution_service import CodeExecutionService

# A response shaped exactly like the python-test-executor's TestResponse for a
# pytest run that collected zero tests (pytest exit code 5).
NO_TESTS_COLLECTED_RESPONSE = {
    "success": False,
    "output": "no tests ran in 0.01s",
    "execution_time": 0.01,
    "test_id": "t-1",
    "exit_code": 5,
    "error": "no_tests_collected",
    "tests_run": 0,
    "tests_passed": 0,
    "tests_failed": 0,
    "no_tests_run": True,
}

ALL_PASSED_RESPONSE = {
    "success": True,
    "output": "3 passed in 0.10s",
    "execution_time": 0.1,
    "test_id": "t-2",
    "exit_code": 0,
    "error": None,
    "tests_run": 3,
    "tests_passed": 3,
    "tests_failed": 0,
    "no_tests_run": False,
}


# --------------------------------------------------------------------------
# Reader 1 — CodeExecutionService (httpx). The LIVE hop.
# --------------------------------------------------------------------------
def _httpx_client(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    return client


async def _read_via_code_execution_service(payload: dict):
    client = _httpx_client(payload)
    request = SimpleNamespace(
        code="x = 1",
        test_code="def test_x(): assert x == 1",
        language="python",
        tenant_id="tenant-1",
    )
    with (
        patch("services.code_execution_service._get_executor_http_client", return_value=client),
        patch("services.code_execution_service._get_executor_url", return_value="http://exec"),
    ):
        return await CodeExecutionService().execute_test(request)


# --------------------------------------------------------------------------
# Reader 2 — HTTPTestRunner (aiohttp). Dormant, and governed anyway.
# --------------------------------------------------------------------------
class _FakeAiohttpResponse:
    """Minimal stand-in for the aiohttp response HTTPTestRunner reads."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status = 200

    async def json(self) -> dict:
        return self._payload

    async def text(self) -> str:  # pragma: no cover - only the 200 path is used
        return ""


class _FakeAiohttpPost:
    """`session.post(...)` is used as an async context manager."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeAiohttpResponse:
        return _FakeAiohttpResponse(self._payload)

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class _FakeAiohttpSession:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.closed = False
        self.calls: list[tuple] = []

    def post(self, url: str, **kwargs: object) -> _FakeAiohttpPost:
        self.calls.append((url, kwargs))
        return _FakeAiohttpPost(self._payload)

    async def close(self) -> None:
        self.closed = True


async def _read_via_http_test_runner(payload: dict):
    request = TestRequest(
        code="x = 1",
        test_code="def test_x(): assert x == 1",
        language="python",
    )
    session = _FakeAiohttpSession(payload)
    runner = HTTPTestRunner()
    with patch.object(HTTPTestRunner, "_get_session", AsyncMock(return_value=session)):
        result = await runner.execute_test(request, tenant_id="tenant-1")
    # Guard against a silent no-op: if the runner never reached the executor the
    # assertions below would be grading a default-constructed TestResult.
    assert session.calls, "HTTPTestRunner never POSTed to the executor"
    return result


READERS = pytest.mark.parametrize(
    "read",
    [
        pytest.param(_read_via_code_execution_service, id="CodeExecutionService"),
        pytest.param(_read_via_http_test_runner, id="HTTPTestRunner"),
    ],
)


@READERS
async def test_executor_output_reaches_stdout(read):
    """`output` must land somewhere a caller can read; it used to be dropped."""
    result = await read(ALL_PASSED_RESPONSE)
    assert result.stdout == "3 passed in 0.10s"
    assert result.stdout != ""


@READERS
async def test_explicit_stdout_still_wins_when_the_executor_sends_one(read):
    """Negative control: executors that DO send stdout are not overridden.

    This one passes with or without the fix, on purpose. A suite in which every
    test goes red on the revert cannot distinguish "the mapping regressed" from
    "the harness broke".
    """
    payload = dict(ALL_PASSED_RESPONSE, stdout="real stdout", stderr="real stderr")
    result = await read(payload)
    assert result.stdout == "real stdout"
    assert result.stderr == "real stderr"


@READERS
async def test_zero_collected_is_distinguishable_from_all_passed(read):
    """`failed == 0` alone is ambiguous; the counts must disambiguate it."""
    nothing = await read(NO_TESTS_COLLECTED_RESPONSE)
    passed = await read(ALL_PASSED_RESPONSE)

    # The trap: both report zero failures.
    assert nothing.tests_failed == 0
    assert passed.tests_failed == 0

    # ...and are nonetheless tellable apart, without parsing any text.
    assert nothing.no_tests_run is True
    assert nothing.tests_run == 0
    assert nothing.success is False
    assert nothing.exit_code == 5
    assert nothing.error == "no_tests_collected"

    assert passed.no_tests_run is False
    assert passed.tests_run == 3
    assert passed.tests_passed == 3
    assert passed.success is True

    # The safe predicate a caller can now write.
    def is_pass(r) -> bool:
        return r.success and r.tests_run > 0 and r.tests_failed == 0 and not r.no_tests_run

    assert is_pass(nothing) is False
    assert is_pass(passed) is True


@READERS
async def test_counts_default_to_zero_for_executors_that_omit_them(read):
    """Older executors send no counts; the model must still validate."""
    legacy = {
        "success": True,
        "output": "ok",
        "execution_time": 0.0,
        "exit_code": 0,
    }
    result = await read(legacy)
    assert result.tests_run == 0
    assert result.no_tests_run is False
    assert result.stdout == "ok"


def test_both_readers_of_the_execute_contract_are_covered():
    """The parametrisation must not silently lose a reader.

    `http_test_runner.py`'s share of this change originally had NO regression
    control at all: every line of it could be reverted with the full suite still
    green. If a future edit drops a reader from READERS, this fails rather than
    quietly shrinking the gate back to one.
    """
    ids = {p.id for p in READERS.args[1]}
    assert ids == {"CodeExecutionService", "HTTPTestRunner"}


# --------------------------------------------------------------------------
# Spelling drift across the eight executor repos.
#
# The executors have no shared schema and have already diverged:
# python/javascript/go emit `no_tests_run`; csharp and cpp emit `noTestsRun`.
# Both readers used a bare `no_tests_run` lookup, so a csharp or cpp
# zero-collection run read as False -- byte-identical to "tests did run", which
# is the fake green the marker exists to prevent.
#
# ⚠️ These tests exist because reverting BOTH boundaries to the bare lookup left
# the whole unit suite GREEN: the alias read was exercised only at
# `_collect_results`, so a regression at either HTTP boundary was invisible.
# Every case below is parametrised over BOTH readers.
# --------------------------------------------------------------------------

CAMELCASE_NO_TESTS_RESPONSE = {
    "success": True,  # csharp/cpp can exit 0 having executed nothing
    "output": "[==========] Running 0 tests from 0 test suites.",
    "execution_time": 0.02,
    "test_id": "t-camel",
    "exit_code": 0,
    "error": None,
    "tests_run": 0,
    "tests_passed": 0,
    "tests_failed": 0,
    "noTestsRun": True,
}

UNKNOWN_RESPONSE = {
    "success": True,  # the runner exited 0...
    "output": "100% tests passed, 0 tests failed out of 1",
    "execution_time": 0.02,
    "test_id": "t-unknown",
    "exit_code": 0,
    "error": None,
    "tests_run": 1,
    "tests_passed": 0,
    "tests_failed": 0,
    # ...but cpp could not determine whether any test CASE ran.
    "no_tests_run_unknown": True,
}


@READERS
async def test_camelcase_no_tests_run_is_not_read_as_tests_ran(read):
    """`noTestsRun` (csharp/cpp) must survive the hop at BOTH boundaries."""
    result = await read(CAMELCASE_NO_TESTS_RESPONSE)
    assert result.no_tests_run is True
    # And the invariant must then bite: zero collected is never a pass, even
    # though the executor said success=True and exit_code=0.
    assert result.success is False


@READERS
async def test_unknown_collection_is_never_reported_as_a_clean_pass(read):
    """An UNMEASURABLE run must not be graded green.

    cpp cannot always tell whether any test case ran: `ctest` suppresses the
    child summary on a passing run and its own count is of registered ctest
    ENTRIES, not cases. When the executor says so, "we do not know" must resolve
    to not-run -- never a pass, and never silently a clean result. A marker no
    reader consults is the exact defect this whole line of work is about, so
    `no_tests_run_unknown` is given a consumer here rather than being emitted
    into the void.
    """
    result = await read(UNKNOWN_RESPONSE)
    assert result.no_tests_run is True
    assert result.success is False


@READERS
async def test_snake_case_no_tests_run_still_survives_both_boundaries(read):
    """The original spelling must not regress while the aliases are added."""
    result = await read(NO_TESTS_COLLECTED_RESPONSE)
    assert result.no_tests_run is True
    assert result.success is False


@READERS
async def test_an_executor_sending_no_marker_is_unaffected(read):
    """Negative control: absence of every alias must read False, not True.

    Without this, widening the alias set to something over-eager (e.g. matching
    any key containing "run") would pass every test above while breaking every
    healthy run. `tests_run` is present and non-zero here precisely to catch
    that class of mistake.
    """
    result = await read(ALL_PASSED_RESPONSE)
    assert result.no_tests_run is False
    assert result.success is True
