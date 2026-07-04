from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_linux_shell_wrappers_exist_and_use_sh_shebang() -> None:
    scripts = [
        "_common.sh",
        "prepare_dataset.sh",
        "prepare_timelens_inputs.sh",
        "interpolate.sh",
        "train_stage1.sh",
        "train_stage2.sh",
        "eval.sh",
        "infer.sh",
        "demo_sequence.sh",
    ]
    scripts_root = PROJECT_ROOT / "scripts"
    for name in scripts:
        path = scripts_root / name
        assert path.exists(), f"Missing script: {name}"
        assert path.read_text(encoding="utf-8").startswith("#!/usr/bin/env sh")


def test_linux_shell_wrappers_pass_sh_syntax_check() -> None:
    sh_bin = shutil.which("sh")
    if sh_bin is None:
        pytest.skip("POSIX sh is not available in this environment")
    scripts_root = PROJECT_ROOT / "scripts"
    for path in sorted(scripts_root.glob("*.sh")):
        subprocess.run([sh_bin, "-n", str(path)], check=True)


def test_required_tool_entrypoints_exist() -> None:
    tools = [
        "prepare_dataset.py",
        "prepare_timelens_inputs.py",
        "interpolate_timelens.py",
        "train.py",
        "eval.py",
        "infer.py",
        "demo_sequence.py",
        "import_swift_eye_checkpoint.py",
    ]
    tools_root = PROJECT_ROOT / "tools"
    for name in tools:
        assert (tools_root / name).exists(), f"Missing tool: {name}"


def test_mmrotate_stack_is_absent_from_project_sources() -> None:
    forbidden = ("mmrotate", "mmdet", "mmcv")
    for root_name in ("swift_hbtxr", "tools"):
        for path in (PROJECT_ROOT / root_name).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in text, f"Forbidden dependency token found in {path}: {token}"
