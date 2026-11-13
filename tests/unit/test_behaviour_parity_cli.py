"""CLI + library behaviour parity, end to end over REAL subprocesses (no mocks).

The request model builds the driver config and spec exactly as the route does, the
real driver (in-process ``main``) runs the real fixture CLIs / the real library call
harness as subprocesses, and :func:`grade` turns the result into the verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from models.parity_behaviour_models import CompareRequest, RecordRequest, ReplayRequest
from services import behaviour_parity_driver as drv
from services import lib_call_harness
from services.behaviour_parity import (
    LIB_HARNESS_NAME,
    build_driver_config,
    build_spec,
    grade,
    lib_harness_source,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "behaviour_parity"
CLI = DATA / "cli_fixture"
LIB = DATA / "lib_fixture"


def _cli_side(script: str) -> dict:
    return {"kind": "cli", "start": [sys.executable, str(CLI / script)]}


def _lib_side() -> dict:
    return {"kind": "cli", "start": [sys.executable, "{lib_harness}"]}


def _run(tmp_path: Path, op: str, req, workdirs: dict[str, Path] | None = None) -> dict:
    """Stage like the runner (driver config + harness), run the driver, grade it."""
    sides = []
    if isinstance(req, CompareRequest):
        sides = [("source", req.source), ("target", req.target)]
    elif isinstance(req, RecordRequest):
        sides = [("source", req.source)]
    else:
        sides = [("target", req.target)]
    cfg = build_driver_config(req, sides)
    for side in cfg["sides"]:
        side["workdir"] = str((workdirs or {}).get(side["role"], tmp_path))
    run_dir = tmp_path / ("f" * 32)
    run_dir.mkdir(parents=True)
    (run_dir / "driver_config.json").write_text(json.dumps(cfg))
    (run_dir / LIB_HARNESS_NAME).write_text(lib_harness_source())
    code = drv.main([str(run_dir)])
    result = json.loads((run_dir / "result.json").read_text())
    verdict = grade(build_spec(op, req), result, exit_code=code)  # type: ignore[arg-type]
    return {"code": code, "result": result, "verdict": verdict.model_dump()}


def _by_name(verdict: dict) -> dict[str, dict]:
    return {c["name"]: c for c in verdict["cases"]}


CLI_CASES = [
    {"name": "add", "kind": "cli", "args": ["add", "2", "3"]},
    {
        "name": "info",
        "kind": "cli",
        "args": ["info", "abc"],
        "stdout_format": "json",
    },
    {"name": "upper", "kind": "cli", "args": ["upper"], "stdin": "hi there\n"},
    {"name": "fail", "kind": "cli", "args": ["fail", "boom"]},
    {"name": "warn_ignored", "kind": "cli", "args": ["warn", "w"]},
    {"name": "warn_compared", "kind": "cli", "args": ["warn", "w"], "compare_stderr": True},
    {
        "name": "write",
        "kind": "cli",
        "args": ["write", "{case_dir}"],
        "env": {"GREETING": "hello"},
    },
]


def test_cli_tool_vs_itself_is_equal(tmp_path):
    req = CompareRequest(
        cases=CLI_CASES,
        ignore_paths=["/pid"],
        source=_cli_side("tool.py"),
        target=_cli_side("tool.py"),
    )
    out = _run(tmp_path, "compare", req)
    v = out["verdict"]
    assert out["code"] == 0, out["result"]
    assert v["computed"] is True and v["equal"] is True, v["cases"]
    assert v["cases_total"] == len(CLI_CASES)
    src = out["result"]["sides"]["source"]["cases"]
    rec = {c["name"]: c for c in src}
    assert rec["add"]["stdout"] == "5\n" and rec["add"]["exit_code"] == 0
    assert rec["info"]["stdout_json"]["length"] == 3 and "stdout" not in rec["info"]
    assert rec["upper"]["stdout"] == "HI THERE\n"
    assert rec["fail"]["exit_code"] == 3 and "fatal: boom" in rec["fail"]["stderr"]
    # {case_dir} is the same path on both sides, and the case env reached the process.
    written = Path(rec["write"]["stdout"].strip())
    assert written.read_text() == "hello"
    assert written.parent == tmp_path / ("f" * 32) / "cli_cases" / "6"
    assert v["recording"]["cases"] == src
    assert out["result"]["driver_version"] == 4


def test_cli_planted_drift_is_caught(tmp_path):
    req = CompareRequest(
        cases=CLI_CASES,
        ignore_paths=["/pid"],
        source=_cli_side("tool.py"),
        target=_cli_side("tool_planted.py"),
    )
    v = _run(tmp_path, "compare", req)["verdict"]
    assert v["computed"] is True and v["equal"] is False
    cases = _by_name(v)
    assert all(c["kind"] == "cli" for c in v["cases"])
    assert cases["add"]["diffs"] == [
        {"kind": "stdout", "pointer": "/stdout", "expected": "5\n", "actual": "6\n", "detail": None}
    ]
    assert [
        (d["kind"], d["pointer"], d["expected"], d["actual"]) for d in cases["info"]["diffs"]
    ] == [("stdout", "/stdout/length", 3, 6)]
    # exit code differs; stderr differs too but is not compared without compare_stderr.
    assert [d["kind"] for d in cases["fail"]["diffs"]] == ["exit_code"]
    assert cases["fail"]["diffs"][0]["expected"] == 3 and cases["fail"]["diffs"][0]["actual"] == 4
    assert cases["warn_ignored"]["passed"] is True
    assert [d["kind"] for d in cases["warn_compared"]["diffs"]] == ["stderr"]
    assert cases["upper"]["passed"] and cases["write"]["passed"]
    assert v["cases_failed"] == 4


def test_cli_ignore_paths_matter(tmp_path):
    """Without ignoring /pid, two runs of the SAME tool differ: the JSON diff is real."""
    req = CompareRequest(
        cases=[CLI_CASES[1]], source=_cli_side("tool.py"), target=_cli_side("tool.py")
    )
    v = _run(tmp_path, "compare", req)["verdict"]
    assert v["equal"] is False
    assert [d["pointer"] for d in v["cases"][0]["diffs"]] == ["/stdout/pid"]


def test_cli_record_against_expected(tmp_path):
    cases = [
        {"name": "add", "kind": "cli", "args": ["add", "1", "1"], "expected_exit_code": 0,
         "expected_stdout": "2\n"},
        {"name": "info", "kind": "cli", "args": ["info", "xy"], "stdout_format": "json",
         "expected_exit_code": 0, "expected_stdout_json": {"name": "xy", "length": 2, "pid": 0}},
        {"name": "fail", "kind": "cli", "args": ["fail", "z"], "expected_exit_code": 3,
         "expected_stderr": "nope\n"},
    ]  # fmt: skip
    req = RecordRequest(cases=cases, ignore_paths=["/pid"], source=_cli_side("tool.py"))
    v = _run(tmp_path, "record", req)["verdict"]
    assert v["baseline"] == "expected" and v["computed"] is True
    cases_out = _by_name(v)
    assert cases_out["add"]["passed"] and cases_out["info"]["passed"]
    assert [d["kind"] for d in cases_out["fail"]["diffs"]] == ["stderr"]

    # Without expectations a record just records.
    bare = RecordRequest(cases=[CLI_CASES[0]], source=_cli_side("tool.py"))
    v = _run(tmp_path / "bare", "record", bare)["verdict"]
    assert v["baseline"] == "none" and v["equal"] is None and v["recording"]["cases"]


def test_cli_replay_against_recording(tmp_path):
    rec = _run(
        tmp_path / "a", "record", RecordRequest(cases=CLI_CASES[:4], source=_cli_side("tool.py"))
    )
    recording = rec["verdict"]["recording"]
    same = ReplayRequest(
        cases=CLI_CASES[:4],
        ignore_paths=["/pid"],
        target=_cli_side("tool.py"),
        baseline="recording",
        recording=recording,
    )
    assert _run(tmp_path / "b", "replay", same)["verdict"]["equal"] is True
    drift = ReplayRequest(
        cases=CLI_CASES[:4],
        ignore_paths=["/pid"],
        target=_cli_side("tool_planted.py"),
        baseline="recording",
        recording=recording,
    )
    v = _run(tmp_path / "c", "replay", drift)["verdict"]
    assert v["equal"] is False and v["cases_failed"] == 3  # add, info, fail


def test_cli_timeout_and_missing_binary_are_errors(tmp_path):
    cases = [{"name": "slow", "kind": "cli", "args": ["sleep", "30"], "timeout": 1}]
    req = CompareRequest(cases=cases, source=_cli_side("tool.py"), target=_cli_side("tool.py"))
    out = _run(tmp_path / "t", "compare", req)
    case = out["result"]["sides"]["source"]["cases"][0]
    assert case["exit_code"] is None and "timed out after 1s" in case["error"]
    assert case["duration_ms"] < 15000
    v = out["verdict"]
    assert v["equal"] is False and v["cases"][0]["diffs"][0]["kind"] == "baseline_error"

    missing = CompareRequest(
        cases=[CLI_CASES[0]],
        source=_cli_side("tool.py"),
        target={"kind": "cli", "start": [str(tmp_path / "no-such-binary")]},
    )
    out = _run(tmp_path / "m", "compare", missing)
    tcase = out["result"]["sides"]["target"]["cases"][0]
    assert "command failed to run" in tcase["error"]
    assert out["verdict"]["cases"][0]["diffs"][0]["kind"] == "error"


def _one(tmp_path: Path, code: str, **case) -> dict:
    """Run one cli case whose command is ``python -c <code>`` through the real driver path."""
    run_dir = tmp_path / ("e" * 32)
    run_dir.mkdir(exist_ok=True)
    start = [sys.executable, "-c", code]
    side = {"role": "source", "kind": "cli", "workdir": str(tmp_path), "start": start}
    case = {"name": "c", "kind": "cli", **case}
    return drv.run_cli_case(case, 0, side, str(run_dir), str(tmp_path), start, 10)


def test_cli_streams_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "CLI_STREAM_CAP", 8)
    code = "import sys\nprint('x' * 100)\nprint('y' * 100, file=sys.stderr)\n"
    out = _one(tmp_path, code, stdout_format="json")
    assert out["stdout"] == "x" * 8 and out["stdout_clipped"] is True
    assert out["stderr"] == "y" * 8 and out["stderr_clipped"] is True
    assert out["truncated"] is True and out["exit_code"] is None
    assert "exceeded 8 bytes" in out["error"]
    assert "not JSON" in out["stdout_json_error"]


def test_cli_flooding_command_is_killed_at_the_cap(tmp_path):
    """An endless writer is stopped at the real 1 MiB cap, not buffered until the timeout."""
    code = "import sys\nwhile True:\n    sys.stdout.write('z' * 65536)\n"
    out = _one(tmp_path, code, timeout=60)
    assert out["truncated"] is True and out["stdout_clipped"] is True
    assert len(out["stdout"]) == drv.CLI_STREAM_CAP == 1 << 20
    assert out["exit_code"] is None and "was killed" in out["error"]
    assert out["duration_ms"] < 20_000  # killed at the cap, far before the 60 s timeout


GRANDCHILD = """
import os, subprocess, sys
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                     start_new_session=True)
open(os.path.join(sys.argv[1], "grandchild.pid"), "w").write(str(p.pid))
print("parent", flush=True)
{tail}
"""


def _reap_grandchild(case_dir: Path) -> None:
    import os
    import signal

    pid = int((case_dir / "grandchild.pid").read_text())
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.mark.parametrize(
    "tail, needle",
    [
        ("import time; time.sleep(60)", "timed out after 2s"),  # parent hangs too
        ("", "a background process holds the pipe"),  # parent exits at once
    ],
)
def test_cli_setsid_grandchild_holding_the_pipe_cannot_hang_the_driver(tmp_path, tail, needle):
    import time

    code = GRANDCHILD.format(tail=tail)
    case_dir = tmp_path / ("e" * 32) / "cli_cases" / "0"
    started = time.monotonic()
    try:
        out = _one(tmp_path, code, args=["{case_dir}"], timeout=2)
    finally:
        if (case_dir / "grandchild.pid").exists():
            _reap_grandchild(case_dir)
    elapsed = time.monotonic() - started
    assert elapsed < 10, elapsed  # bounded by the 2 s case timeout (+ kill grace)
    assert needle in out["error"], out
    assert out["stdout"] == "parent\n"


SAME_GROUP_CHILD = """
import os, subprocess, sys, time
p = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # same group, NOT setsid
open(os.path.join(sys.argv[1], "grandchild.pid"), "w").write(str(p.pid))
{tail}
"""


def _alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child of an exited leader is reparented and reaped by init; until then
    # it is a zombie, which is dead for our purpose.
    import subprocess

    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


@pytest.mark.parametrize(
    "tail, needle",
    [
        ("", None),  # the leader exits at once, leaving its child behind
        ("time.sleep(60)", "timed out after 2s"),  # the leader times out
    ],
)
def test_cli_same_group_children_do_not_outlive_the_case(tmp_path, tail, needle):
    """Fails if the command does not get its own process group or the group kill is lost."""
    import time

    case_dir = tmp_path / ("e" * 32) / "cli_cases" / "0"
    out = _one(tmp_path, SAME_GROUP_CHILD.format(tail=tail), args=["{case_dir}"], timeout=2)
    pid = int((case_dir / "grandchild.pid").read_text())
    try:
        deadline = time.monotonic() + 5
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(pid), f"same-group child {pid} survived the case: {out}"
    finally:
        if _alive(pid):
            import os
            import signal

            os.kill(pid, signal.SIGKILL)
    if needle:
        assert needle in out["error"]
    else:
        assert "error" not in out and out["exit_code"] == 0


def test_kill_group_kills_a_sigterm_ignoring_child_of_a_compliant_leader(tmp_path):
    """The leader dies on SIGTERM at once; its SIGTERM-ignoring child still gets SIGKILL."""
    import subprocess
    import time

    code = SAME_GROUP_CHILD.format(tail="time.sleep(60)")
    leader = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], start_new_session=True)
    pid_file = tmp_path / "grandchild.pid"
    deadline = time.monotonic() + 10
    while not (pid_file.exists() and pid_file.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    pid = int(pid_file.read_text())
    try:
        drv._kill_group(leader, grace=5.0)
        assert leader.returncode is not None
        deadline = time.monotonic() + 5
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(pid)
    finally:
        if _alive(pid):
            import os
            import signal

            os.kill(pid, signal.SIGKILL)


def test_cli_timeout_over_the_cap_is_rejected(tmp_path):
    run_dir = tmp_path / ("c" * 32)
    run_dir.mkdir()
    side = {"role": "source", "kind": "cli", "workdir": str(tmp_path), "start": [sys.executable]}
    case = {"name": "a", "kind": "cli", "timeout": drv.CLI_MAX_TIMEOUT + 1}
    (run_dir / "driver_config.json").write_text(json.dumps({"cases": [case], "sides": [side]}))
    assert drv.main([str(run_dir)]) == 1
    assert "<= 300" in json.loads((run_dir / "result.json").read_text())["error"]


def test_cli_default_timeout_is_capped_too(tmp_path, monkeypatch):
    """request_timeout (up to 600 s) is the default per-case limit, clamped to the cap."""
    monkeypatch.setattr(drv, "CLI_MAX_TIMEOUT", 1)
    run_dir = tmp_path / ("d" * 32)
    run_dir.mkdir()
    start = [sys.executable, "-c", "import time; time.sleep(30)"]
    side = {"role": "source", "kind": "cli", "workdir": str(tmp_path), "start": start}
    out = drv.run_cli_case({"name": "c", "kind": "cli"}, 0, side, str(run_dir), str(tmp_path),
                           start, 600)  # fmt: skip
    assert "timed out after 1s" in out["error"] and out["duration_ms"] < 10_000


# ------------------------------------------------------------------ library parity


LIB_CASES = [
    {"name": "add", "kind": "cli", "args": ["mylib:add"], "stdin": {"args": [2, 3]},
     "stdout_format": "json"},
    {"name": "tags", "kind": "cli", "args": ["mylib:tags"],
     "stdin": {"args": ["a", "b"], "kwargs": {"sep": "-"}}, "stdout_format": "json"},
    {"name": "method", "kind": "cli", "args": ["mylib:Calc.double"], "stdin": {"args": [21]},
     "stdout_format": "json"},
    {"name": "noisy", "kind": "cli", "args": ["mylib:noisy"], "stdin": {"args": ["q"]},
     "stdout_format": "json"},
    {"name": "divide_zero", "kind": "cli", "args": ["mylib:divide"], "stdin": {"args": [1, 0]},
     "stdout_format": "json"},
    {"name": "non_json", "kind": "cli", "args": ["mylib:as_set"], "stdin": {"args": [1]},
     "stdout_format": "json"},
]  # fmt: skip


def test_library_vs_itself_is_equal_and_records_the_contract(tmp_path):
    req = CompareRequest(cases=LIB_CASES, source=_lib_side(), target=_lib_side())
    out = _run(tmp_path, "compare", req, {"source": LIB, "target": LIB})
    v = out["verdict"]
    assert v["computed"] is True and v["equal"] is True, v["cases"]
    rec = {c["name"]: c for c in out["result"]["sides"]["source"]["cases"]}
    assert rec["add"]["stdout_json"] == {"return": 5} and rec["add"]["exit_code"] == 0
    assert rec["tags"]["stdout_json"] == {"return": {"joined": "a-b", "count": 2}}
    assert rec["method"]["stdout_json"] == {"return": 42}
    assert rec["noisy"]["stdout_json"] == {"return": ["q", "q"]}
    assert "goes to stderr" in rec["noisy"]["stderr"]
    assert rec["divide_zero"]["stdout_json"] == {
        "raises": "ZeroDivisionError",
        "message": "division by zero",
    }
    assert rec["divide_zero"]["exit_code"] == 0
    assert rec["non_json"]["exit_code"] == 2
    assert "not JSON (set)" in rec["non_json"]["stdout_json"]["error"]


def test_library_planted_drift_is_caught(tmp_path):
    req = CompareRequest(cases=LIB_CASES, source=_lib_side(), target=_lib_side())
    v = _run(tmp_path, "compare", req, {"source": LIB, "target": LIB / "planted"})["verdict"]
    cases = _by_name(v)
    assert v["equal"] is False and v["cases_failed"] == 2
    assert [(d["pointer"], d["expected"], d["actual"]) for d in cases["add"]["diffs"]] == [
        ("/stdout/return", 5, 6)
    ]
    assert {(d["pointer"], d["actual"]) for d in cases["divide_zero"]["diffs"]} == {
        ("/stdout/raises", "ValueError"),
        ("/stdout/message", "b must not be zero"),
    }


def test_harness_errors_in_process(tmp_path, monkeypatch, capsys):
    import io

    monkeypatch.chdir(LIB)

    def call(argv: list[str], stdin: str) -> tuple[int, dict]:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
        code = lib_call_harness.main(argv)
        return code, json.loads(capsys.readouterr().out)

    assert call(["mylib:add"], '{"args": [1, 2]}') == (0, {"return": 3})
    code, out = call(["mylib:add"], "")  # no args: the CALL raises, the harness does not
    assert code == 0 and out["raises"] == "TypeError"
    for argv, stdin, needle in [
        ([], "", "usage"),
        (["mylib:add"], "not json", "stdin is not JSON"),
        (["mylib:add"], "[1]", "JSON object"),
        (["mylib:add"], '{"args": {}}', "args must be a list"),
        (["mylib"], "", "is not 'module:function'"),
        (["no_such_mod:f"], "", "cannot resolve"),
        (["mylib:missing"], "", "cannot resolve"),
        (["mylib:Calc.double.__doc__"], "", "cannot resolve"),
    ]:
        code, out = call(argv, stdin)
        assert code == 2 and needle in out["error"], (argv, out)


# ------------------------------------------------------------- driver validation


@pytest.mark.parametrize(
    "cases, side_kind, needle",
    [
        ([{"name": "a", "kind": "cli", "args": "add 1 2"}], "cli", "list of strings"),
        ([{"name": "a", "kind": "cli", "env": {"A": 1}}], "cli", "strings to strings"),
        ([{"name": "a", "kind": "cli", "timeout": 0}], "cli", "positive number"),
        ([{"name": "a", "kind": "cli", "stdout_format": "xml"}], "cli", "'text' or 'json'"),
        ([{"name": "a", "kind": "cli", "compare_stderr": "y"}], "cli", "boolean"),
        (
            [{"name": "a", "kind": "cli"}, {"name": "b", "method": "GET", "path": "/"}],
            "cli",
            "cannot mix",
        ),
        ([{"name": "a", "method": "GET", "path": "/"}], "cli", "runs only cli cases"),
        ([{"name": "a", "kind": "cli"}], "service", "need a side with kind 'cli'"),
    ],
)
def test_driver_rejects_bad_cli_configs(tmp_path, cases, side_kind, needle):
    run_dir = tmp_path / ("c" * 32)
    run_dir.mkdir()
    side = {"role": "source", "workdir": str(tmp_path), "start": [sys.executable, "-c", "0"]}
    if side_kind == "cli":
        side["kind"] = "cli"
    (run_dir / "driver_config.json").write_text(json.dumps({"cases": cases, "sides": [side]}))
    assert drv.main([str(run_dir)]) == 1
    result = json.loads((run_dir / "result.json").read_text())
    assert needle in result["error"], result["error"]


def test_harness_file_is_the_shipped_module():
    assert lib_harness_source() == Path(lib_call_harness.__file__).read_text()
    assert LIB_HARNESS_NAME == drv.LIB_HARNESS_NAME


def test_exit_is_observed_without_reaping_so_the_pgid_stays_reserved():
    """The leader stays a zombie (pid + pgid reserved) until _kill_group signals the group."""
    import os
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"], start_new_session=True)
    assert drv._wait_exit(proc, 10) == 7
    assert proc.returncode is None  # not reaped
    assert os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    drv._kill_group(proc, grace=1)
    assert proc.returncode == 7  # reaped only by _kill_group, after the group kill
    killed = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              start_new_session=True)  # fmt: skip
    os.kill(killed.pid, 9)
    assert drv._wait_exit(killed, 10) == -9
    drv._kill_group(killed, grace=1)
    assert killed.returncode == -9
