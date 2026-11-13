"""Python test-impact analysis over a workspace's real import graph.

Pure functions only — no I/O. The caller (``IntelligentTestingService``) runs
one ``find`` and one ``grep`` inside the workspace and hands the raw stdout
here. Everything returned is derived from those two listings; nothing is
guessed. In particular every ``test_file`` in the result is a path that the
``find`` listing contained — the property the previous placeholder
(``changed_file.replace('.py', '_test.py')``) violated by inventing paths such
as ``src/auth_test.py`` that did not exist.

Module map
----------
A file can be importable under several dotted names, depending on which
directory is on ``sys.path``. We register every file under each of these roots:

* the repository root (``src/auth.py`` -> ``src.auth``);
* ``src/`` when it exists (``src/auth.py`` -> ``auth``) — the src-layout;
* the file's *package root*: walk up from the file's directory while the
  directory contains ``__init__.py``; the first directory without one is the
  root. This is the directory pytest's default ``prepend`` import mode puts on
  ``sys.path`` (``tests/test_x.py`` with no ``tests/__init__.py`` -> ``test_x``).

A dotted name may therefore map to more than one file; all of them are kept.
That errs towards reporting a test as impacted, never towards inventing one.

Known limits (documented, not hidden)
-------------------------------------
* Imports are read line-by-line by ``grep``: a parenthesised multi-line
  ``from x import (\\n a,\\n b)`` contributes ``x`` (and ``x.a`` only when ``a``
  is on the first line). Dynamic imports (``importlib``, ``__import__``) are
  not seen.
* An ``import`` inside a docstring that happens to name a workspace module adds
  an edge. Over-inclusion, never fabrication.
* ``conftest.py`` is not imported by tests; a changed conftest impacts every
  test file in its directory subtree (pytest's own scoping rule).
"""

from __future__ import annotations

import posixpath
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Matches the grep -rn line format ``<path>:<lineno>:<text>``. The path is
# matched non-greedily up to the FIRST ``:<digits>:`` so a colon inside the
# source text cannot be mistaken for the separator.
_GREP_LINE = re.compile(r"^(?P<path>.+?):(?P<lineno>\d+):(?P<text>.*)$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

MAX_INDIRECT_HOPS = 3

# Directories pruned from both listings. Kept here so the find and grep argv
# built by the service and the documentation cannot drift apart.
PRUNED_DIRS = (".git", ".venv", "node_modules", "__pycache__")


def normalize_path(path: str, workdir: str = "/workspace") -> str:
    """Return ``path`` relative to the workspace root, posix-style, no ``./``."""
    p = path.strip().replace("\\", "/")
    prefix = workdir.rstrip("/") + "/"
    if p.startswith(prefix):
        p = p[len(prefix) :]
    while p.startswith("./"):
        p = p[2:]
    return posixpath.normpath(p) if p else p


def parse_find_output(stdout: str, workdir: str = "/workspace") -> set[str]:
    """Parse ``find`` stdout into a set of workspace-relative .py paths."""
    files: set[str] = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rel = normalize_path(line, workdir)
        if rel.endswith(".py"):
            files.add(rel)
    return files


def is_test_file(path: str) -> bool:
    """test_*.py, *_test.py, or any module under a ``tests/`` directory.

    ``__init__.py`` and ``conftest.py`` are never test files: pytest does not
    collect tests from them.
    """
    base = posixpath.basename(path)
    if base in ("__init__.py", "conftest.py"):
        return False
    if base.startswith("test_") or base.endswith("_test.py"):
        return True
    return "tests" in path.split("/")[:-1]


@dataclass
class ImportGraph:
    """Forward and reverse import edges between workspace files."""

    files: set[str]
    module_map: dict[str, set[str]] = field(default_factory=dict)
    imports: dict[str, set[str]] = field(default_factory=dict)
    importers: dict[str, set[str]] = field(default_factory=dict)


def _module_roots(files: set[str]) -> dict[str, set[str]]:
    """Map each file to the set of directories it is importable from."""
    dirs_with_init = {posixpath.dirname(f) for f in files if f.endswith("/__init__.py")}
    if "__init__.py" in files:
        dirs_with_init.add("")
    has_src = any(f.startswith("src/") for f in files)
    roots: dict[str, set[str]] = {}
    for f in files:
        r = {""}
        if has_src and f.startswith("src/"):
            r.add("src")
        d = posixpath.dirname(f)
        while d and d in dirs_with_init:
            d = posixpath.dirname(d)
        r.add(d)
        roots[f] = r
    return roots


def _dotted_name(path: str, root: str) -> str | None:
    rel = path[len(root) + 1 :] if root else path
    if not rel.endswith(".py"):
        return None
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(_IDENT.match(p) for p in parts):
        return None
    return ".".join(parts)


def build_module_map(files: set[str]) -> dict[str, set[str]]:
    """dotted module name -> workspace files importable under that name."""
    module_map: dict[str, set[str]] = {}
    for f, roots in _module_roots(files).items():
        for root in roots:
            name = _dotted_name(f, root)
            if name:
                module_map.setdefault(name, set()).add(f)
    return module_map


def _split_aliases(chunk: str) -> list[str]:
    """``a as b, c`` -> ``['a', 'c']`` (only well-formed dotted names)."""
    out = []
    for part in chunk.split(","):
        name = part.strip().split(" as ")[0].strip()
        if name and _DOTTED.match(name):
            out.append(name)
    return out


def parse_import_statement(text: str) -> list[tuple[int, str, list[str]]]:
    """Parse one source line into ``(level, module, names)`` tuples.

    ``import a.b, c`` -> ``[(0, 'a.b', []), (0, 'c', [])]``;
    ``from ..x import y, z`` -> ``[(2, 'x', ['y', 'z'])]``. Semicolon-joined
    statements are each parsed. Anything unrecognised yields nothing.
    """
    results: list[tuple[int, str, list[str]]] = []
    code = text.split("#", 1)[0]
    for stmt in code.split(";"):
        stmt = stmt.strip().rstrip("\\").strip()
        if stmt.startswith("import "):
            for name in _split_aliases(stmt[len("import ") :]):
                results.append((0, name, []))
        elif stmt.startswith("from "):
            m = re.match(r"^from\s+(\.*)([A-Za-z_][\w.]*)?\s+import\s+(.*)$", stmt)
            if not m:
                continue
            level = len(m.group(1))
            module = m.group(2) or ""
            if module and not _DOTTED.match(module):
                continue
            if not level and not module:
                continue
            names_chunk = m.group(3).strip().lstrip("(").rstrip(")")
            names = [] if names_chunk.strip() == "*" else _split_aliases(names_chunk)
            results.append((level, module, names))
    return results


def _resolve(
    importer: str,
    level: int,
    module: str,
    names: list[str],
    module_map: dict[str, set[str]],
    files: set[str],
) -> set[str]:
    targets: set[str] = set()
    if level:
        base = posixpath.dirname(importer)
        for _ in range(level - 1):
            if not base:
                return targets  # beyond the workspace root
            base = posixpath.dirname(base)
        mod_dir = posixpath.join(base, *module.split(".")) if module else base
        candidates = []
        if module:
            candidates += [mod_dir + ".py", posixpath.join(mod_dir, "__init__.py")]
        else:
            candidates.append(posixpath.join(mod_dir, "__init__.py") if mod_dir else "__init__.py")
        for name in names:
            candidates += [
                posixpath.join(mod_dir, name + ".py"),
                posixpath.join(mod_dir, name, "__init__.py"),
            ]
        targets.update(c for c in candidates if c in files)
        return targets

    # Absolute: the module itself, every parent package (Python executes each
    # package __init__ on the way down), and ``from pkg import submodule``.
    parts = module.split(".")
    for i in range(1, len(parts) + 1):
        targets |= module_map.get(".".join(parts[:i]), set())
    for name in names:
        targets |= module_map.get(f"{module}.{name}", set())
    return targets


def build_import_graph(
    files: set[str], grep_stdout: str, workdir: str = "/workspace"
) -> ImportGraph:
    """Build the workspace import graph from the ``find`` set and ``grep`` stdout."""
    graph = ImportGraph(files=set(files), module_map=build_module_map(files))
    for line in grep_stdout.splitlines():
        m = _GREP_LINE.match(line)
        if not m:
            continue
        importer = normalize_path(m.group("path"), workdir)
        if importer not in graph.files:
            continue
        for level, module, names in parse_import_statement(m.group("text").strip()):
            for target in _resolve(importer, level, module, names, graph.module_map, graph.files):
                if target == importer:
                    continue
                graph.imports.setdefault(importer, set()).add(target)
                graph.importers.setdefault(target, set()).add(importer)
    return graph


def _impact(test_file: str, source: str, via: list[str], reason: str) -> dict[str, Any]:
    hops = len(via)
    return {
        "test_file": test_file,
        "source_file": source,
        "impact_type": "direct" if hops == 0 else "indirect",
        "impact_score": round(10.0 / (hops + 1), 2),
        "hops": hops,
        "via": via,
        "reason": reason,
    }


def compute_test_impacts(
    graph: ImportGraph,
    changed_files: list[str],
    include_indirect: bool = True,
    max_hops: int = MAX_INDIRECT_HOPS,
    workdir: str = "/workspace",
) -> dict[str, Any]:
    """Compute direct and indirect test impacts for ``changed_files``.

    Returns ``direct``, ``indirect`` (lists of impact dicts, one per test file,
    keeping the shortest path), ``unknown_changed_files`` (Python paths not in
    the workspace listing) and ``unsupported_changed_files`` (non-Python).
    """
    best: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    unsupported: list[str] = []

    def offer(impact: dict[str, Any]) -> None:
        # Only files from the find listing can ever be offered — the guarantee.
        if impact["test_file"] not in graph.files:
            return
        cur = best.get(impact["test_file"])
        if cur is None or impact["hops"] < cur["hops"]:
            best[impact["test_file"]] = impact

    for raw in changed_files:
        changed = normalize_path(raw, workdir)
        if not changed.endswith(".py"):
            unsupported.append(raw)
            continue
        if changed not in graph.files:
            unknown.append(raw)
            continue

        if is_test_file(changed):
            offer(_impact(changed, changed, [], "changed_test_file"))

        if posixpath.basename(changed) == "conftest.py":
            scope = posixpath.dirname(changed)
            for f in graph.files:
                if is_test_file(f) and (not scope or f.startswith(scope + "/")):
                    offer(_impact(f, changed, [], "conftest_scope"))

        # BFS over the reverse import graph. depth = number of import edges
        # between the test and the changed file; via = the intermediates.
        limit = (max_hops + 1) if include_indirect else 1
        seen = {changed}
        queue: deque[tuple[str, list[str]]] = deque([(changed, [])])
        while queue:
            node, path = queue.popleft()
            if len(path) + 1 > limit:
                continue
            for importer in sorted(graph.importers.get(node, ())):
                if importer in seen:
                    continue
                seen.add(importer)
                via = list(reversed(path))  # nearest-to-test first
                if is_test_file(importer):
                    offer(_impact(importer, changed, via, "imports"))
                queue.append((importer, [*path, importer]))

    impacts = sorted(best.values(), key=lambda i: (i["hops"], i["test_file"]))
    return {
        "direct": [i for i in impacts if i["hops"] == 0],
        "indirect": [i for i in impacts if i["hops"] > 0],
        "unknown_changed_files": unknown,
        "unsupported_changed_files": unsupported,
    }
