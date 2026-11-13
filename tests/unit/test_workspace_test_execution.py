"""POST /api/tests/execute with container_name and no inline code runs IN the workspace.

Live 2026-10-12 (cv-m2-hello, t4/test_pass.py and t4/test_fail.py): both answered
502 "Test executor error: body.code Field required" -- workspace mode forwarded
test_path to an executor that only takes inline code, so no workspace test ever ran.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from services.code_execution_service import CodeExecutionService

_EXEC = "services.intelligent_testing_service._containers_exec"


def _req(**kw):
    base = dict(
        language="python",
        container_name="cv-m2-hello",
        code=None,
        test_code=None,
        test_path="t4/test_pass.py",
        test_file_path=None,
        timeout=None,
        security_level="high",
        test_type="unit",
        tenant_id=None,
    )
    return SimpleNamespace(**{**base, **kw})


def _run(payload):
    return patch(_EXEC, AsyncMock(return_value=payload))


async def test_passing_workspace_test_is_a_pass():
    with _run({"exit_code": 0, "stdout": ".\n1 passed in 0.02s\n", "stderr": ""}) as ex:
        r = await CodeExecutionService().execute_test(_req(), tenant_id="t1")
    assert r.success is True
    assert (r.tests_passed, r.tests_failed) == (1, 0)
    argv = ex.call_args.args[2]
    assert argv[:3] == ["python", "-m", "pytest"] and argv[-1] == "t4/test_pass.py"
    assert ex.call_args.args[:2] == ("cv-m2-hello", "t1")


async def test_failing_workspace_test_is_a_failure():
    out = "F\nFAILED t4/test_fail.py::test_add_wrong - assert 4 == 5\n1 failed in 0.02s\n"
    with _run({"exit_code": 1, "stdout": out, "stderr": ""}):
        r = await CodeExecutionService().execute_test(_req(test_path="t4/test_fail.py"))
    assert r.success is False
    assert r.tests_failed == 1


async def test_nothing_collected_is_never_a_pass():
    with _run({"exit_code": 5, "stdout": "no tests ran in 0.01s\n", "stderr": ""}):
        r = await CodeExecutionService().execute_test(_req(test_path="t4"))
    assert r.success is False
    assert r.no_tests_run is True
    assert "nothing was verified" in r.error


async def test_errors_and_truncation_are_not_passes():
    with _run({"exit_code": 0, "stdout": "3 passed\n", "stderr": "", "truncated": True}):
        truncated = await CodeExecutionService().execute_test(_req())
    with _run({"exit_code": 2, "stdout": "1 passed, 2 errors in 0.1s\n", "stderr": ""}):
        errored = await CodeExecutionService().execute_test(_req())
    assert truncated.success is False
    assert errored.success is False and errored.tests_failed == 2


async def test_other_languages_and_escaping_paths_are_refused():
    with pytest.raises(HTTPException) as lang:
        await CodeExecutionService().execute_test(_req(language="go"))
    with pytest.raises(HTTPException) as escape:
        await CodeExecutionService().execute_test(_req(test_path="../etc"))
    with pytest.raises(HTTPException) as option:
        await CodeExecutionService().execute_test(_req(test_path="-pX"))
    with _run({"stdout": "1 passed"}), pytest.raises(HTTPException) as no_exit:
        await CodeExecutionService().execute_test(_req())
    assert lang.value.status_code == 422
    assert escape.value.status_code == 400
    assert option.value.status_code == 400
    assert no_exit.value.status_code == 502


async def test_inline_code_still_goes_to_the_executor():
    with (
        patch(_EXEC, AsyncMock()) as ex,
        patch("services.code_execution_service._get_executor_http_client") as client,
        patch("services.code_execution_service._get_executor_url", return_value="http://py"),
    ):
        client.return_value.post = AsyncMock(
            return_value=SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"success": True, "exit_code": 0, "output": "1 passed"},
            )
        )
        r = await CodeExecutionService().execute_test(
            _req(code="def add(a, b):\n    return a + b\n", test_code="def test_a(): pass\n")
        )
    ex.assert_not_called()
    assert r.success is True


# ── /api/tests/validate-security: a verdict, or "not computed" -- never a default "safe" ──

_SEC_CLIENT = "services.code_execution_service._get_security_http_client"
_SETTINGS = "services.code_execution_service.get_settings"


def _security_answer(body=None, *, status=200, raise_exc=None):
    import httpx

    def raise_for_status():
        if status >= 400:
            request = httpx.Request("POST", "http://sec/api/security/validate")
            raise httpx.HTTPStatusError(
                "x", request=request, response=httpx.Response(status, text="no table")
            )

    response = SimpleNamespace(raise_for_status=raise_for_status, json=lambda: body)
    client = SimpleNamespace(
        post=AsyncMock(side_effect=raise_exc) if raise_exc else AsyncMock(return_value=response)
    )
    return client


async def _validate(client, url="http://sec"):
    with (
        patch(_SEC_CLIENT, return_value=client),
        patch(_SETTINGS, return_value=SimpleNamespace(SECURITY_SERVICE_URL=url)),
    ):
        return await CodeExecutionService()._validate_code_security("x", "python")


async def test_security_verdict_is_read_from_the_validation_envelope():
    """coding-security answers {"status", "validation": {...}}: the top level has no is_safe."""
    unsafe = {
        "status": "success",
        "validation": {
            "is_safe": False,
            "violations": [{"type": "sql_injection", "line_number": 7, "description": "concat"}],
            "risk_score": 30.0,
        },
    }
    r = await _validate(_security_answer(unsafe))
    assert r["is_safe"] is False and r["computed"] is True
    assert r["violations"][0]["type"] == "sql_injection"


async def test_no_verdict_is_not_computed_never_safe():
    import httpx

    cases = [
        await _validate(_security_answer({"status": "success"})),
        await _validate(_security_answer(status=422)),
        await _validate(_security_answer(raise_exc=httpx.ConnectError("down"))),
        await _validate(_security_answer({}), url=""),
    ]
    for r in cases:
        assert r["is_safe"] is None and r["computed"] is False and r["reason"]


async def test_route_accepts_a_json_body_and_never_defaults_to_safe():
    from unittest.mock import MagicMock

    from routers.testing_modules.test_execution import SecurityValidateBody, validate_code_security

    user = MagicMock(preferred_username="u")
    service = SimpleNamespace(
        _validate_code_security=AsyncMock(
            return_value={"is_safe": None, "computed": False, "reason": "unreachable"}
        )
    )
    sqli = 'cur.execute("SELECT * FROM users WHERE id = " + uid)\n'
    r = await validate_code_security(
        code=None,
        language=None,
        body=SecurityValidateBody(code=sqli, language="python"),
        user=user,
        execution_service=service,
    )
    service._validate_code_security.assert_awaited_once_with(sqli, "python")
    assert r["is_safe"] is None and r["computed"] is False and r["reason"] == "unreachable"
