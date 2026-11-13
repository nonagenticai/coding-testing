"""Reduce coverage.py's JSON report to its summary INSIDE the workspace (stdlib, 3.9+).

The raw report lists every executed and missing line of every file, far too big to
read back through the exec route. This runs next to the report as
``python3 -I -c <this source> <report path> <workdir> <max files>`` and prints
only the numbers ``services.coverage_report`` turns into a ``CoverageReport``::

    {"totals": {"lines": [covered, total], "functions": [c, t] | null,
                "branches": [c, t] | null},
     "files": [[path relative to workdir, {"lines": [c, t], ...}], ...],  # <= max files
     "files_total": N}

Exit 3: no report at the path.
"""

from __future__ import annotations

import json
import sys

MISSING_EXIT = 3


def _functions(entry: dict):
    """[functions with a covered line, functions with statements]; None when not reported."""
    fns = entry.get("functions")
    if fns is None:
        return None
    covered = total = 0
    for name, fn in fns.items():
        summary = fn.get("summary") or {}
        if not name or not summary.get("num_statements"):
            continue  # "" is module-level code, not a function
        total += 1
        covered += 1 if summary.get("covered_lines") else 0
    return [covered, total]


def summarize(doc: dict, workdir: str, max_files: int) -> dict:
    prefix = workdir.rstrip("/") + "/"
    files = []
    fn_cov = fn_total = 0
    fn_seen = False
    for path, entry in (doc.get("files") or {}).items():
        s = entry.get("summary") or {}
        fns = _functions(entry)
        if fns is not None:
            fn_seen = True
            fn_cov, fn_total = fn_cov + fns[0], fn_total + fns[1]
        files.append(
            [
                path[len(prefix) :] if path.startswith(prefix) else path,
                {
                    "lines": [s.get("covered_lines", 0), s.get("num_statements", 0)],
                    "functions": fns,
                    "branches": [s["covered_branches"], s["num_branches"]]
                    if s.get("num_branches")
                    else None,
                },
            ]
        )
    t = doc.get("totals") or {}
    return {
        "totals": {
            "lines": [t.get("covered_lines", 0), t.get("num_statements", 0)],
            "functions": [fn_cov, fn_total] if fn_seen else None,
            # No branch measurement is "not measured", not 0 %.
            "branches": [t["covered_branches"], t["num_branches"]]
            if t.get("num_branches")
            else None,
        },
        "files": files[:max_files],
        "files_total": len(files),
    }


def main(argv: list) -> int:
    path, workdir, max_files = argv[0], argv[1], int(argv[2])
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        sys.stderr.write(f"no coverage report at {path}\n")
        return MISSING_EXIT
    sys.stdout.write(json.dumps(summarize(doc, workdir, max_files)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
