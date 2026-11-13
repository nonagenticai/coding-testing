"""The registry's ruff/ty commands produce the output contract consumers parse.

Runs the exact argv from ``QualityToolRegistry`` (and ``get_tool_commands``) with the
real ruff/ty binaries from the dev environment against a seeded-defect file, so a
flag typo or an output-format change fails here rather than as an empty "clean" run
in a user's workspace.

- ruff check JSON: [{code, message, filename, location:{row,column}, ...}], exit 1.
- ty gitlab JSON: [{check_name, description "<code>: msg", severity, fingerprint,
  location:{path, positions:{begin:{line,column}}}}], exit 1.
- ty crash/bad config: exit 2 and NO JSON on stdout — a consumer must treat that as
  a tool failure, never as zero findings.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 - argv is the registry's fixed tool command
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from services.testing.quality_config import QualityToolRegistry
from services.testing.test_config_manager import TestConfigManager as _ConfigManager


def _bin(name: str) -> str:
    candidate = Path(sys.executable).parent / name
    found = str(candidate) if candidate.exists() else shutil.which(name)
    if not found:
        pytest.fail(f"{name} not installed in the dev environment (uv sync)")
    return found


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    argv = [_bin(argv[0]), *argv[1:]]
    return subprocess.run(  # nosec B603 - fixed argv
        argv, cwd=cwd, capture_output=True, text=True, timeout=120, check=False
    )


@pytest.fixture
def defective(tmp_path: Path) -> Path:
    # sys before os = unsorted (I001); both unused (F401); unused local (F841);
    # str assigned to int (ty invalid-assignment).
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "probe"\nversion = "0"\n')
    (tmp_path / "m.py").write_text(
        'import sys\nimport os\n\n\ndef f() -> None:\n    unused = 1\n\n\nx: int = "a"\n'
    )
    return tmp_path


def _registry_cmd(tool: str) -> list[str]:
    cfg = QualityToolRegistry().get_tool_config(tool, "python")
    assert cfg is not None, f"{tool} missing from the python registry"
    return cfg.cmd


def test_ruff_json_yields_findings_with_codes(defective: Path) -> None:
    proc = _run([*_registry_cmd("ruff"), "m.py"], defective)
    assert proc.returncode == 1, proc.stderr
    findings = json.loads(proc.stdout)
    codes = {f["code"] for f in findings}
    assert {"I001", "F401", "F841"} <= codes
    first = findings[0]
    assert first["filename"].endswith("m.py")
    assert isinstance(first["location"]["row"], int)
    assert isinstance(first["location"]["column"], int)
    assert first["message"]


def test_ruff_clean_file_exits_zero_with_empty_array(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("X = 1\n")
    proc = _run([*_registry_cmd("ruff"), "ok.py"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []


def test_ruff_format_check_flags_unformatted(defective: Path) -> None:
    (defective / "ugly.py").write_text("x=[1,2 ,3]\n")
    proc = _run([*_registry_cmd("ruff-format"), "ugly.py"], defective)
    assert proc.returncode == 1
    assert "x = [1, 2, 3]" in proc.stdout


def test_ty_gitlab_json_yields_findings_with_codes(defective: Path) -> None:
    proc = _run([*_registry_cmd("ty"), "m.py"], defective)
    assert proc.returncode == 1, proc.stderr
    findings = json.loads(proc.stdout)
    assert findings, "ty reported no diagnostics for a seeded type error"
    hit = next(f for f in findings if f["check_name"] == "invalid-assignment")
    assert hit["description"].startswith("invalid-assignment: ")
    assert hit["location"]["path"].endswith("m.py")
    assert hit["location"]["positions"]["begin"]["line"] == 9
    assert hit["fingerprint"]
    assert hit["severity"]


def test_ty_crash_is_a_failure_not_a_clean_run(defective: Path) -> None:
    (defective / "pyproject.toml").write_text(
        '[project]\nname = "probe"\nversion = "0"\n\n[tool.ty]\nbogus = 1\n'
    )
    proc = _run([*_registry_cmd("ty"), "m.py"], defective)
    assert proc.returncode == 2
    assert "ty failed" in proc.stderr
    with pytest.raises(json.JSONDecodeError):
        json.loads(proc.stdout)


def test_config_manager_commands_match_registry() -> None:
    cmds = _ConfigManager(security_service=AsyncMock()).get_tool_commands("python")
    for tool in ("ruff", "ruff-format", "ty"):
        assert cmds[tool] == _registry_cmd(tool)


def test_python_install_spec_is_pinned_uv() -> None:
    (cmd,) = QualityToolRegistry().get_install_commands("python")
    assert cmd.startswith("uv pip install")
    assert "ruff==0.17.3" in cmd
    assert "ty==0.0.92" in cmd
    for retired in ("black", "isort", "mypy", "pylint", "flake8", "autoflake", "pyright"):
        assert retired not in cmd.split()
