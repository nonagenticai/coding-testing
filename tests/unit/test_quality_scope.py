"""quality-check / quality-scan analyse what was asked for, or say "not computed".

Live 2026-10-12: both endpoints dropped ``code_path`` / ``scan_path`` and sent
only ``container_name`` to coding-analysis ``/api/quality/analyze-sync``, which
then analysed the WHOLE workspace (>60 s with a big ``.repos/`` clone; issues
about files nobody asked about). These tests pin:

* quality-check + code_path sends ``container_name`` + ``paths: [code_path]`` —
  never ``code_content`` (coding-analysis's ad-hoc mode has no linters and
  always came back "inconclusive: 0/6 tools succeeded", live 10:24Z);
* inline code without a workspace path is a 422, never an ad-hoc analysis;
* quality-scan + scan_path sends ``paths: [scan_path]`` and only claims
  ``scope_applied`` when coding-analysis echoes the scope back;
* every no-verdict path is ``success=False`` with an explicit error.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

import services.code_execution_service as mod

_RealAsyncClient = httpx.AsyncClient

_OK = {
    "analysis_id": "qa-1",
    "success": True,
    "overall_score": 80.0,
    "total_issues": 0,
    "issues": [],
    "tools_results": {},
    "execution_time": 0.5,
    "language": "python",
}


def _issue(path: str) -> dict[str, Any]:
    return {
        "tool": "ruff",
        "severity": "warning",
        "message": "x",
        "file_path": path,
        "line_number": 1,
        "column_number": 1,
        "rule_id": "F401",
    }


class _Analysis:
    """Records every POST to coding-analysis and answers with ``response``."""

    def __init__(self, response: httpx.Response | Exception):
        self.response = response
        self.payloads: list[dict[str, Any]] = []

    def client(self, *args: Any, **kwargs: Any) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            self.payloads.append(json.loads(request.content))
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

        return _RealAsyncClient(transport=httpx.MockTransport(handler))


class _Exec:
    def __init__(self, result: dict[str, Any] | Exception):
        self.result = result
        self.calls: list[tuple[str, str | None, list[str], int]] = []

    async def __call__(
        self, container: str, tenant: str | None, argv: list[str], timeout: int
    ) -> Any:
        self.calls.append((container, tenant, argv, timeout))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _check(**over: Any) -> SimpleNamespace:
    base = {
        "container_name": "ws1",
        "code": None,
        "code_content": None,
        "code_path": "src/app.py",
        "language": None,
        "tools": None,
        "fix_issues": False,
        "custom_rules": None,
        "tenant_id": None,
    }
    base.update(over)
    return SimpleNamespace(**base)


def _scan(**over: Any) -> SimpleNamespace:
    base = {
        "container_name": "ws1",
        "scan_path": "src",
        "language": None,
        "tools": None,
        "fix_issues": False,
        "tenant_id": None,
    }
    base.update(over)
    return SimpleNamespace(**base)


async def _run(method: str, request: Any, analysis: _Analysis, exec_: _Exec | None = None):
    svc = mod.CodeExecutionService(workspace_exec=exec_ or _Exec(RuntimeError("no exec")))
    with patch.object(mod.httpx, "AsyncClient", analysis.client):
        return await getattr(svc, method)(request, tenant_id="t1")


# --------------------------------------------------------------- quality-check


class TestQualityCheckCodePath:
    async def test_payload_is_container_plus_paths_never_code_content(self):
        body = {**_OK, "scope": ["src/app.py"], "issues": [_issue("src/app.py")]}
        analysis = _Analysis(httpx.Response(200, json=body))
        exec_ = _Exec({"exit_code": 0, "stdout": "f 120", "stderr": ""})

        result = await _run("analyze_quality", _check(), analysis, exec_)

        assert len(analysis.payloads) == 1
        payload = analysis.payloads[0]
        assert payload["container_name"] == "ws1"
        assert payload["paths"] == ["src/app.py"]
        assert payload["language"] == "python"  # inferred from .py
        assert "code_content" not in payload
        # the ONE file was checked, workspace-relative, with "--" before the path
        assert exec_.calls[0][0] == "ws1"
        assert exec_.calls[0][2][:2] == ["find", "src/app.py"]
        assert result.success is True
        assert result.scope == ["src/app.py"] and result.scope_applied is True
        assert [i.file_path for i in result.issues] == ["src/app.py"]

    async def test_old_analyser_without_echo_reports_scope_not_applied(self):
        """An older coding-analysis ignores ``paths`` and analyses the whole workspace."""
        issues = [_issue("src/app.py"), _issue("src/other.py"), _issue(".repos/x.py")]
        analysis = _Analysis(httpx.Response(200, json={**_OK, "issues": issues}))
        exec_ = _Exec({"exit_code": 0, "stdout": "f 120", "stderr": ""})
        result = await _run("analyze_quality", _check(), analysis, exec_)
        assert result.scope_applied is False
        assert [i.file_path for i in result.issues] == ["src/app.py"]
        assert result.out_of_scope_issues == 2

    @pytest.mark.parametrize(
        "req",
        [
            _check(code="x = 1\n", code_path=None, language="python"),
            _check(code_content="x = 1\n", code_path=".", language="python"),
            _check(code="x = 1\n", code_path=None, container_name=None, language="python"),
        ],
    )
    async def test_inline_code_without_workspace_path_is_422_not_ad_hoc(self, req):
        """coding-analysis's ad-hoc code_content mode can never produce a verdict."""
        analysis = _Analysis(httpx.Response(200, json=_OK))
        with pytest.raises(HTTPException) as ei:
            await _run("analyze_quality", req, analysis)
        assert ei.value.status_code == 422
        assert "inline code quality needs a workspace" in ei.value.detail
        assert analysis.payloads == []

    async def test_code_path_wins_over_inline_code(self):
        analysis = _Analysis(httpx.Response(200, json={**_OK, "scope": ["src/app.py"]}))
        exec_ = _Exec({"exit_code": 0, "stdout": "f 120", "stderr": ""})
        await _run("analyze_quality", _check(code="x = 1\n"), analysis, exec_)
        assert "code_content" not in analysis.payloads[0]
        assert analysis.payloads[0]["paths"] == ["src/app.py"]

    async def test_no_code_and_no_path_is_the_whole_workspace(self):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        result = await _run("analyze_quality", _check(code_path="."), analysis)
        assert analysis.payloads[0]["container_name"] == "ws1"
        assert "paths" not in analysis.payloads[0]
        assert result.scope is None and result.scope_applied is None

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../x.py", "src/../../x.py", "-rf.py"])
    async def test_rejects_paths_outside_the_workspace(self, bad):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        exec_ = _Exec({"exit_code": 0, "stdout": "f 120", "stderr": ""})
        with pytest.raises(HTTPException) as ei:
            await _run("analyze_quality", _check(code_path=bad), analysis, exec_)
        assert ei.value.status_code == 422
        assert exec_.calls == [] and analysis.payloads == []

    async def test_workspace_absolute_prefix_is_accepted(self):
        analysis = _Analysis(httpx.Response(200, json={**_OK, "scope": ["src/app.py"]}))
        exec_ = _Exec({"exit_code": 0, "stdout": "f 120", "stderr": ""})
        result = await _run(
            "analyze_quality", _check(code_path="/workspace/src/app.py"), analysis, exec_
        )
        assert exec_.calls[0][2][1] == "src/app.py"
        assert analysis.payloads[0]["paths"] == ["src/app.py"]
        assert result.scope == ["src/app.py"]

    async def test_missing_file_is_404_not_an_empty_pass(self):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        exec_ = _Exec(
            {
                "exit_code": 1,
                "stdout": "",
                "stderr": "find: 'src/app.py': No such file or directory",
            }
        )
        with pytest.raises(HTTPException) as ei:
            await _run("analyze_quality", _check(), analysis, exec_)
        assert ei.value.status_code == 404
        assert "not computed" in ei.value.detail
        assert analysis.payloads == []

    async def test_directory_is_422(self):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        exec_ = _Exec({"exit_code": 0, "stdout": "d 4096", "stderr": ""})
        with pytest.raises(HTTPException) as ei:
            await _run("analyze_quality", _check(code_path="src.py"), analysis, exec_)
        assert ei.value.status_code == 422
        assert analysis.payloads == []

    async def test_empty_file_is_not_computed(self):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        exec_ = _Exec({"exit_code": 0, "stdout": "f 0", "stderr": ""})
        result = await _run("analyze_quality", _check(), analysis, exec_)
        assert result.success is False and "empty" in result.error
        assert analysis.payloads == []

    async def test_unknown_extension_without_language_is_422(self):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        with pytest.raises(HTTPException) as ei:
            await _run("analyze_quality", _check(code_path="README.md"), analysis)
        assert ei.value.status_code == 422

    async def test_exec_failure_is_not_computed(self):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        exec_ = _Exec(httpx.ConnectError("containers down"))
        result = await _run("analyze_quality", _check(), analysis, exec_)
        assert result.success is False
        assert "not computed" in result.error
        assert result.total_issues == 0 and analysis.payloads == []

    async def test_exec_timeout_is_not_computed(self):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        exec_ = _Exec({"exit_code": 124, "stdout": "", "stderr": ""})
        result = await _run("analyze_quality", _check(), analysis, exec_)
        assert result.success is False and "timed out" in result.error

    async def test_analyser_422_on_paths_is_not_computed(self):
        """coding-analysis 422 ("paths match no files") is an explicit no-verdict."""
        analysis = _Analysis(httpx.Response(422, text="paths match no files"))
        exec_ = _Exec({"exit_code": 0, "stdout": "f 120", "stderr": ""})
        result = await _run("analyze_quality", _check(), analysis, exec_)
        assert result.success is False
        assert "422" in result.error and "not computed" in result.error
        assert result.scope == ["src/app.py"]


# ---------------------------------------------------------------- quality-scan


class TestQualityScanScanPath:
    async def test_sends_paths_and_trusts_only_an_echo(self):
        body = {**_OK, "scope": ["src"], "issues": [_issue("src/a.py")]}
        analysis = _Analysis(httpx.Response(200, json=body))
        result = await _run("analyze_container_quality", _scan(), analysis)
        assert analysis.payloads[0]["paths"] == ["src"]
        assert analysis.payloads[0]["container_name"] == "ws1"
        assert result.scope == ["src"] and result.scope_applied is True
        assert result.total_issues == 1

    async def test_old_analyser_without_echo_reports_scope_not_applied(self):
        """An older coding-analysis ignores ``paths`` and scans everything."""
        issues = [_issue("src/a.py"), _issue(".repos/big/x.py"), _issue("/srv/ws1/src/b.py")]
        analysis = _Analysis(httpx.Response(200, json={**_OK, "issues": issues}))
        result = await _run("analyze_container_quality", _scan(), analysis)
        assert result.scope_applied is False
        assert [i.file_path for i in result.issues] == ["src/a.py", "/srv/ws1/src/b.py"]
        assert result.out_of_scope_issues == 1
        assert result.total_issues == 2

    async def test_no_scan_path_is_the_whole_workspace(self):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        result = await _run("analyze_container_quality", _scan(scan_path=None), analysis)
        assert "paths" not in analysis.payloads[0]
        assert result.scope is None and result.scope_applied is None

    async def test_rejects_absolute_scan_path(self):
        analysis = _Analysis(httpx.Response(200, json=_OK))
        with pytest.raises(HTTPException) as ei:
            await _run("analyze_container_quality", _scan(scan_path="/app/data"), analysis)
        assert ei.value.status_code == 422


# --------------------------------------------------------- no verdict, ever


class TestNoVerdictIsNeverAPass:
    @pytest.mark.parametrize(
        "method,req",
        [
            ("analyze_quality", _check(code_path=".")),
            ("analyze_container_quality", _scan()),
        ],
    )
    @pytest.mark.parametrize(
        "response,needle",
        [
            (httpx.Response(500, text="boom"), "500"),
            (httpx.Response(401, text="no licence"), "401"),
            (httpx.ReadTimeout("slow"), "timed out"),
            (httpx.ConnectError("down"), "unreachable"),
        ],
    )
    async def test_analysis_failure_is_success_false_with_error(
        self, method, req, response, needle
    ):
        result = await _run(method, req, _Analysis(response))
        assert result.success is False
        assert needle in result.error and "not computed" in result.error
        assert result.total_issues == 0

    async def test_inconclusive_null_score_is_not_success(self):
        body = {**_OK, "overall_score": None}
        result = await _run(
            "analyze_container_quality",
            _scan(scan_path=None),
            _Analysis(httpx.Response(200, json=body)),
        )
        assert result.success is False and result.error


def test_existence_check_uses_an_allowlisted_command():
    """coding-containers refuses `stat` (400 "Command 'stat' not allowed", live 2026-10-12)."""
    import inspect

    from services.code_execution_service import CodeExecutionService

    src = inspect.getsource(CodeExecutionService._check_workspace_file)
    assert '"find"' in src and '"stat"' not in src
