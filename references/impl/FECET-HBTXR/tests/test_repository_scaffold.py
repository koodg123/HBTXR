from __future__ import annotations

from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_linux_shell_wrappers_exist_and_use_sh_shebang() -> None:
    scripts = [
        "_common.sh",
        "prepare_dataset.sh",
        "prepare_facet_gsam_dataset.sh",
        "facet_prepare_reference_dataset.sh",
        "facet_train.sh",
        "facet_eval.sh",
        "facet_infer.sh",
        "train_stage1.sh",
        "train_stage2.sh",
        "fecet_compare_train_stage1.sh",
        "fecet_compare_train_stage2.sh",
        "fecet_compare_eval.sh",
        "fecet_compare_infer.sh",
        "eval.sh",
        "infer.sh",
        "visualize.sh",
        "overlay_preview.sh",
    ]
    scripts_root = PROJECT_ROOT / "scripts"
    for name in scripts:
        path = scripts_root / name
        assert path.exists(), f"Missing script: {name}"
        assert path.read_text(encoding="utf-8").startswith("#!/usr/bin/env sh")


def test_linux_shell_wrappers_pass_sh_syntax_check() -> None:
    scripts_root = PROJECT_ROOT / "scripts"
    for path in sorted(scripts_root.glob("*.sh")):
        subprocess.run(["sh", "-n", str(path)], check=True)


def test_doc_history_files_exist() -> None:
    docs = [
        "README.md",
        "UPDATE_HISTORY.md",
        "CONVERSATION_HISTORY.md",
        "FACET_QUANT_COMPARISON.md",
    ]
    docs_root = PROJECT_ROOT / "doc"
    for name in docs:
        path = docs_root / name
        assert path.exists(), f"Missing doc file: {name}"
