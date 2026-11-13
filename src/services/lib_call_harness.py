"""Library call harness: call ONE Python function from a JSON request, print a JSON verdict.

Like the behaviour-parity driver, THIS FILE RUNS INSIDE A CODING WORKSPACE. coding-testing
writes it next to the driver (``<run_dir>/lib_call_harness.py``) and a cli parity side
substitutes its path as ``{lib_harness}``, e.g. ``start: ["uv", "run", "python",
"{lib_harness}"]``. Standard library only, Python 3.9+.

The contract (a Rust harness binary in the target repo implements the SAME one, so a
library is compared as a CLI from one corpus)::

    argv[1]  "pkg.mod:func"  (or "pkg.mod:Class.method"; the part after ':' is an
                              attribute path)
    stdin    {"args": [...], "kwargs": {...}}   (both optional; empty stdin = no args)
    stdout   exactly ONE JSON object, then exit 0:
               {"return": <json value>}                     the call returned
               {"raises": "<ExcType>", "message": "<str>"}  the call raised
             or, exit 2, {"error": "<str>"}: the harness could not make the call or
             the return value is not JSON (a tuple becomes a list; NaN is an error)

``ExcType`` is the exception class's bare name (``ValueError``, not
``builtins.ValueError``) and ``message`` is ``str(exc)``. Anything the function prints
goes to stderr, so stdout carries only the verdict. The working directory is put first
on ``sys.path``, so a package in the side's workdir imports without being installed.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys


def _fail(message: str) -> int:
    sys.stdout.write(json.dumps({"error": message}) + "\n")
    return 2


def resolve(target: str):
    module_name, sep, attr_path = target.partition(":")
    if not sep or not module_name or not attr_path:
        raise ValueError(f"target {target!r} is not 'module:function'")
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise ValueError(f"{target} is not callable")
    return obj


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        return _fail("usage: lib_call_harness.py module:function  (request JSON on stdin)")
    raw = sys.stdin.read()
    try:
        request = json.loads(raw) if raw.strip() else {}
    except ValueError as e:
        return _fail(f"stdin is not JSON: {e}")
    if not isinstance(request, dict):
        return _fail("stdin must be a JSON object {args, kwargs}")
    call_args = request.get("args", [])
    call_kwargs = request.get("kwargs", {})
    if not isinstance(call_args, list) or not isinstance(call_kwargs, dict):
        return _fail("args must be a list and kwargs an object")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            func = resolve(args[0])
    except Exception as e:  # noqa: BLE001 - any import failure is a harness error
        return _fail(f"cannot resolve {args[0]}: {type(e).__name__}: {e}")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            value = func(*call_args, **call_kwargs)
    except Exception as e:  # noqa: BLE001 - the exception IS the recorded behaviour
        verdict: dict = {"raises": type(e).__name__, "message": str(e)}
    else:
        verdict = {"return": value}
    try:
        line = json.dumps(verdict, allow_nan=False)
    except (TypeError, ValueError) as e:
        return _fail(f"return value is not JSON ({type(value).__name__}): {e}")
    sys.stdout.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
