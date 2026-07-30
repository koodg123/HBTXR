#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Sequence

HARDWARE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_m_axi_smoke_bundle"
DEFAULT_TAR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_m_axi_smoke_bundle.tar.gz"

PACKAGE_FILES = [
    "pynq/hgtxr/README.md",
    "pynq/hgtxr/__init__.py",
    "pynq/hgtxr/hgtxr_overlay.py",
    "pynq/hgtxr/e2e_m_axi_overlay.py",
    "pynq/hgtxr/e2e_m_axi_weights.py",
    "pynq/hgtxr/run_e2e_m_axi_smoke.py",
    "pynq/hgtxr/hgtxr_e2e_m_axi.bit",
    "pynq/hgtxr/hgtxr_e2e_m_axi.hwh",
]
WEIGHT_FILES = [
    "refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
    "refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
]
SMOKE_COMMAND = (
    "PYTHONPATH=. python3 -m hgtxr.run_e2e_m_axi_smoke "
    "--weights-mode file "
    "--weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin "
    "--expect-out-raw 32 -13 26 -6 14 -11 "
    "--json-out e2e_m_axi_file_smoke.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}


def write_text_file(dst: Path, text: str) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    return {"path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}


def bundle_readme_text(src: Path) -> str:
    text = src.read_text()
    repo_only_block = (
        "Exported packed-weight smoke:\n\n"
        "    python3 hardware/tools/export_e2e_m_axi_weights.py\n"
        "    python3 -m hgtxr.run_e2e_m_axi_smoke --weights-mode file --weights-bin "
        "hardware/refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin "
        "--expect-out-raw 32 -13 26 -6 14 -11 --json-out e2e_m_axi_file_smoke.json\n\n"
    )
    return text.replace(repo_only_block, "")


def build_bundle(out_dir: Path = DEFAULT_OUT_DIR, tar_path: Path | None = DEFAULT_TAR) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    files: list[dict[str, Any]] = []
    for rel in PACKAGE_FILES:
        src = HARDWARE_ROOT / rel
        dst = out_dir / "hgtxr" / Path(rel).name
        if rel == "pynq/hgtxr/README.md":
            copied = write_text_file(dst, bundle_readme_text(src))
        else:
            copied = copy_file(src, dst)
        files.append({"source": rel, "bundle": f"hgtxr/{dst.name}", **copied})
    for rel in WEIGHT_FILES:
        src = HARDWARE_ROOT / rel
        dst = out_dir / "weights" / Path(rel).name
        files.append({"source": rel, "bundle": f"weights/{dst.name}", **copy_file(src, dst)})

    run_script = out_dir / "run_e2e_m_axi_file_smoke.sh"
    run_script.write_text("#!/usr/bin/env sh\nset -eu\n" + SMOKE_COMMAND + "\n")
    run_script.chmod(0o755)
    files.append(
        {
            "source": "generated",
            "bundle": run_script.name,
            "path": str(run_script),
            "bytes": run_script.stat().st_size,
            "sha256": sha256_file(run_script),
        }
    )

    manifest = {
        "name": "hgtxr_e2e_m_axi_smoke_bundle",
        "target": "ZCU104 PYNQ",
        "command": SMOKE_COMMAND,
        "expected_runtime_state": 2,
        "expected_out_raw": [32, -13, 26, -6, 14, -11],
        "files": files,
    }
    manifest_path = out_dir / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if tar_path is not None:
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        if tar_path.exists():
            tar_path.unlink()
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(out_dir, arcname=out_dir.name)
        manifest["tar"] = {
            "path": str(tar_path),
            "bytes": tar_path.stat().st_size,
            "sha256": sha256_file(tar_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return {"out_dir": str(out_dir), "manifest": str(manifest_path), "tar": manifest.get("tar")}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package HGTXR A2 E2E m_axi PYNQ smoke bundle.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tar", dest="tar_path", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--no-tar", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_bundle(args.out_dir, None if args.no_tar else args.tar_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
