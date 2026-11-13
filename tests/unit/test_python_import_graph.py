"""Unit tests for services.python_import_graph (pure, no I/O)."""

from __future__ import annotations

import re

from services.python_import_graph import (
    build_import_graph,
    build_module_map,
    compute_test_impacts,
    is_test_file,
    parse_find_output,
    parse_import_statement,
)

_IMPORT_LINE = re.compile(r"^[ \t]*(from|import)[ \t]+[A-Za-z_.]")


def _graph(tree: dict[str, str]):
    files = parse_find_output("".join(f"./{p}\n" for p in tree))
    grep = "".join(
        f"./{p}:{n}:{line}\n"
        for p, src in tree.items()
        for n, line in enumerate(src.splitlines(), 1)
        if _IMPORT_LINE.match(line)
    )
    return build_import_graph(files, grep)


def _tests(result):
    return {i["test_file"]: i for i in result["direct"] + result["indirect"]}


class TestParsing:
    def test_find_output_normalised(self):
        out = "./src/a.py\n/workspace/src/b.py\nsrc/c.txt\n\n"
        assert parse_find_output(out) == {"src/a.py", "src/b.py"}

    def test_parse_import_forms(self):
        assert parse_import_statement("import a.b as x, c") == [(0, "a.b", []), (0, "c", [])]
        assert parse_import_statement("from ..pkg import y as z, w") == [(2, "pkg", ["y", "w"])]
        assert parse_import_statement("from . import mod") == [(1, "", ["mod"])]
        assert parse_import_statement("from x import (a, b") == [(0, "x", ["a", "b"])]
        assert parse_import_statement("from x import *") == [(0, "x", [])]
        assert parse_import_statement("import os; import src.auth  # c") == [
            (0, "os", []),
            (0, "src.auth", []),
        ]
        assert parse_import_statement("from the moon we came") == []

    def test_is_test_file(self):
        assert is_test_file("tests/test_a.py")
        assert is_test_file("pkg/a_test.py")
        assert is_test_file("tests/helpers.py")
        assert not is_test_file("tests/conftest.py")
        assert not is_test_file("tests/__init__.py")
        assert not is_test_file("src/auth.py")

    def test_module_map_roots(self):
        files = {"src/__init__.py", "src/auth.py", "lib/pkg/__init__.py", "lib/pkg/m.py"}
        mm = build_module_map(files)
        assert mm["src.auth"] == {"src/auth.py"}
        assert mm["auth"] == {"src/auth.py"}  # src-layout root
        assert mm["pkg.m"] == {"lib/pkg/m.py"}  # package root (lib/ has no __init__)
        assert "lib.pkg.m" in mm  # repo-root name


class TestImpacts:
    def test_src_layout_and_relative_imports(self):
        tree = {
            "src/app/__init__.py": "",
            "src/app/core.py": "X = 1\n",
            "src/app/svc.py": "from .core import X\n",
            "tests/test_svc.py": "from app.svc import X\n",
        }
        res = compute_test_impacts(_graph(tree), ["src/app/core.py"])
        t = _tests(res)
        assert set(t) == {"tests/test_svc.py"}
        assert t["tests/test_svc.py"]["via"] == ["src/app/svc.py"]

    def test_from_package_import_submodule(self):
        tree = {
            "src/__init__.py": "",
            "src/auth.py": "",
            "tests/test_a.py": "from src import auth\n",
        }
        res = compute_test_impacts(_graph(tree), ["src/auth.py"])
        assert [i["test_file"] for i in res["direct"]] == ["tests/test_a.py"]

    def test_hop_limit_three_intermediates(self):
        tree = {
            "m0.py": "",
            "m1.py": "import m0\n",
            "m2.py": "import m1\n",
            "m3.py": "import m2\n",
            "m4.py": "import m3\n",
            "test_3hops.py": "import m3\n",
            "test_4hops.py": "import m4\n",
        }
        t = _tests(compute_test_impacts(_graph(tree), ["m0.py"]))
        assert t["test_3hops.py"]["via"] == ["m3.py", "m2.py", "m1.py"]
        assert t["test_3hops.py"]["impact_score"] == 2.5
        assert "test_4hops.py" not in t

    def test_shortest_path_kept(self):
        tree = {
            "a.py": "",
            "b.py": "import a\n",
            "test_x.py": "import a\nimport b\n",
        }
        t = _tests(compute_test_impacts(_graph(tree), ["a.py"]))
        assert t["test_x.py"]["impact_type"] == "direct"

    def test_conftest_scope(self):
        tree = {
            "tests/conftest.py": "import pytest\n",
            "tests/unit/test_a.py": "",
            "other/test_b.py": "",
        }
        t = _tests(compute_test_impacts(_graph(tree), ["tests/conftest.py"]))
        assert set(t) == {"tests/unit/test_a.py"}
        assert t["tests/unit/test_a.py"]["reason"] == "conftest_scope"

    def test_import_of_external_module_adds_no_edge(self):
        tree = {"src/a.py": "import requests\n", "tests/test_a.py": "import requests\n"}
        res = compute_test_impacts(_graph(tree), ["src/a.py"])
        assert res["direct"] == [] and res["indirect"] == []

    def test_every_result_exists_in_listing(self):
        tree = {
            "src/auth.py": "",
            "tests/test_auth.py": "from src.auth import x\n",
        }
        g = _graph(tree)
        res = compute_test_impacts(g, ["src/auth.py", "src/ghost.py", "docs/x.md"])
        assert all(i["test_file"] in g.files for i in res["direct"] + res["indirect"])
        assert res["unknown_changed_files"] == ["src/ghost.py"]
        assert res["unsupported_changed_files"] == ["docs/x.md"]
        assert "src/auth_test.py" not in _tests(res)
