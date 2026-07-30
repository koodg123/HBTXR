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
DEFAULT_VARIANT = "c3b-mem16"
DEFAULT_OUT_DIR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
DEFAULT_TAR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"

ARTIFACT_PREFIXES = {
    "a1": "hgtxr_e2e_axis_dma",
    "c1-par16": "hgtxr_e2e_axis_dma_par16",
    "c3b-mem16": "hgtxr_e2e_axis_dma_c3b_mem16",
    "runtime-mode-par32-search": "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16",
    "runtime-mode-par32-track": "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16",
    "par32-rom-compute-300-search": (
        "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
        "corefabric_normuram_compute_300_mem16"
    ),
    "par32-rom-compute-300-track": (
        "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
        "corefabric_normuram_compute_300_mem16"
    ),
    "par32-prefetchall4-300-search": (
        "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
        "corefabric_normuram_compute_prefetchall4_300_mem16"
    ),
    "par32-prefetchall4-300-track": (
        "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
        "corefabric_normuram_compute_prefetchall4_300_mem16"
    ),
    "par32-prefetchall4-300-hybrid-10-90": (
        "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
        "corefabric_normuram_compute_prefetchall4_300_mem16"
    ),
    "par32-patch32-dtok4-300-search": "hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16",
    "par32-patch32-dtok4-300-track": "hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16",
    "par32-patch32-dtok4-300-hybrid-10-90": "hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16",
    "vref-p0-softmax-input-x2-dsp-mixed-stream": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream",
    "vref-p0-softmax-input-x2-qkv-uram": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram",
}

PACKAGE_FILES = [
    "pynq/hgtxr/README.md",
    "pynq/hgtxr/__init__.py",
    "pynq/hgtxr/hgtxr_overlay.py",
    "pynq/hgtxr/e2e_axis_dma_overlay.py",
    "pynq/hgtxr/e2e_m_axi_overlay.py",
    "pynq/hgtxr/e2e_m_axi_weights.py",
    "pynq/hgtxr/run_e2e_axis_dma_smoke.py",
    "pynq/hgtxr/run_e2e_axis_dma_hybrid_smoke.py",
]
TOOL_FILES = [
    "tools/validate_pynq_smoke_result.py",
]
WEIGHT_FILES = [
    "refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
    "refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
]
EXPECTED_RAW = [32, -13, 26, -6, 14, -11]
PAR32_PATCH32_DTOK4_300_SEARCH_EXPECTED_RAW = [3, 3, 3, 3, 3, 3]
PAR32_PATCH32_DTOK4_300_TRACK_EXPECTED_RAW = [217, 217, 217, 217, 217, 217]
EXPECTED_RAW_BY_VARIANT = {
    "a1": EXPECTED_RAW,
    "c1-par16": EXPECTED_RAW,
    "c3b-mem16": EXPECTED_RAW,
    "runtime-mode-par32-search": None,
    "runtime-mode-par32-track": None,
    "par32-rom-compute-300-search": [-1169, -1169, -1169, -1169, -1169, -1133],
    "par32-rom-compute-300-track": [-235, -235, -235, -235, -235, -226],
    "par32-prefetchall4-300-search": [-1169, -1169, -1169, -1169, -1169, -1125],
    "par32-prefetchall4-300-track": [-235, -235, -235, -235, -235, -239],
    "par32-prefetchall4-300-hybrid-10-90": {
        "search": [-1169, -1169, -1169, -1169, -1169, -1125],
        "track": [-235, -235, -235, -235, -235, -239],
    },
    "par32-patch32-dtok4-300-search": PAR32_PATCH32_DTOK4_300_SEARCH_EXPECTED_RAW,
    "par32-patch32-dtok4-300-track": PAR32_PATCH32_DTOK4_300_TRACK_EXPECTED_RAW,
    "par32-patch32-dtok4-300-hybrid-10-90": {
        "search": PAR32_PATCH32_DTOK4_300_SEARCH_EXPECTED_RAW,
        "track": PAR32_PATCH32_DTOK4_300_TRACK_EXPECTED_RAW,
    },
    "vref-p0-softmax-input-x2-dsp-mixed-stream": [58, -51, 42, -28, 36, -41],
    "vref-p0-softmax-input-x2-qkv-uram": [58, -51, 42, -28, 36, -41],
}
EXPECTED_RUNTIME_STATE_BY_VARIANT = {
    "runtime-mode-par32-search": 0,
    "runtime-mode-par32-track": 1,
    "par32-rom-compute-300-search": 0,
    "par32-rom-compute-300-track": 1,
    "par32-prefetchall4-300-search": 0,
    "par32-prefetchall4-300-track": 1,
    "par32-prefetchall4-300-hybrid-10-90": None,
    "par32-patch32-dtok4-300-search": 0,
    "par32-patch32-dtok4-300-track": 1,
    "par32-patch32-dtok4-300-hybrid-10-90": None,
}
RUNNER_VARIANT_BY_VARIANT = {
    "runtime-mode-par32-search": "runtime-mode-par32",
    "runtime-mode-par32-track": "runtime-mode-par32",
    "par32-rom-compute-300-search": "par32-rom-compute-300",
    "par32-rom-compute-300-track": "par32-rom-compute-300",
    "par32-prefetchall4-300-search": "par32-prefetchall4-300",
    "par32-prefetchall4-300-track": "par32-prefetchall4-300",
    "par32-prefetchall4-300-hybrid-10-90": "par32-prefetchall4-300",
    "par32-patch32-dtok4-300-search": "par32-patch32-dtok4-300",
    "par32-patch32-dtok4-300-track": "par32-patch32-dtok4-300",
    "par32-patch32-dtok4-300-hybrid-10-90": "par32-patch32-dtok4-300",
}
MODE_PROFILE_BY_VARIANT = {
    "runtime-mode-par32-search": "search",
    "runtime-mode-par32-track": "track",
    "par32-rom-compute-300-search": "search",
    "par32-rom-compute-300-track": "track",
    "par32-prefetchall4-300-search": "search",
    "par32-prefetchall4-300-track": "track",
    "par32-patch32-dtok4-300-search": "search",
    "par32-patch32-dtok4-300-track": "track",
}


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


def smoke_command(variant: str) -> str:
    if variant == "par32-prefetchall4-300-hybrid-10-90":
        return (
            "PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_hybrid_smoke "
            "--variant par32-prefetchall4-300 "
            "--warmup-cycles 1 --repeat-cycles 5 "
            "--json-out e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json"
        )
    if variant == "par32-patch32-dtok4-300-hybrid-10-90":
        return (
            "PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_hybrid_smoke "
            "--variant par32-patch32-dtok4-300 "
            "--warmup-cycles 1 --repeat-cycles 5 "
            "--json-out e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_file_smoke.json"
        )
    runner_variant = RUNNER_VARIANT_BY_VARIANT.get(variant, variant)
    command = (
        "PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke "
        f"--variant {runner_variant} "
    )
    if variant in MODE_PROFILE_BY_VARIANT:
        command += (
            "--weights-mode zero "
            f"--mode-profile {MODE_PROFILE_BY_VARIANT[variant]} "
            f"--expect-runtime-state {EXPECTED_RUNTIME_STATE_BY_VARIANT[variant]} "
            "--warmup 1 --repeat 5 "
        )
        if EXPECTED_RAW_BY_VARIANT[variant] is not None:
            raw = " ".join(str(value) for value in EXPECTED_RAW_BY_VARIANT[variant])
            command += f"--expect-out-raw {raw} "
    else:
        raw = " ".join(str(value) for value in EXPECTED_RAW_BY_VARIANT[variant])
        command += (
            "--weights-mode file "
            "--weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin "
            f"--expect-out-raw {raw} "
        )
    return command + f"--json-out e2e_axis_dma_{variant.replace('-', '_')}_file_smoke.json"


def result_json_name(variant: str) -> str:
    return f"e2e_axis_dma_{variant.replace('-', '_')}_file_smoke.json"


def validation_json_name(variant: str) -> str:
    return f"e2e_axis_dma_{variant.replace('-', '_')}_file_smoke_validation.json"


def validation_preset(variant: str) -> str:
    if variant == "a1":
        return "axis-a1"
    if variant == "c1-par16":
        return "axis-c1-par16"
    if variant == "c3b-mem16":
        return "axis-c3b-mem16"
    if variant == "runtime-mode-par32-search":
        return "axis-runtime-mode-par32-search"
    if variant == "runtime-mode-par32-track":
        return "axis-runtime-mode-par32-track"
    if variant == "par32-rom-compute-300-search":
        return "axis-par32-rom-compute-300-search"
    if variant == "par32-rom-compute-300-track":
        return "axis-par32-rom-compute-300-track"
    if variant == "par32-prefetchall4-300-search":
        return "axis-par32-prefetchall4-300-search"
    if variant == "par32-prefetchall4-300-track":
        return "axis-par32-prefetchall4-300-track"
    if variant == "par32-prefetchall4-300-hybrid-10-90":
        return "axis-par32-prefetchall4-300-hybrid-10-90"
    if variant == "par32-patch32-dtok4-300-search":
        return "axis-par32-patch32-dtok4-300-search"
    if variant == "par32-patch32-dtok4-300-track":
        return "axis-par32-patch32-dtok4-300-track"
    if variant == "par32-patch32-dtok4-300-hybrid-10-90":
        return "axis-par32-patch32-dtok4-300-hybrid-10-90"
    if variant == "vref-p0-softmax-input-x2-dsp-mixed-stream":
        return "axis-vref-p0-softmax-input-x2-dsp-mixed-stream"
    if variant == "vref-p0-softmax-input-x2-qkv-uram":
        return "axis-vref-p0-softmax-input-x2-qkv-uram"
    raise ValueError(f"unsupported variant: {variant}")


def validation_command(variant: str) -> str:
    return (
        "python3 tools/validate_pynq_smoke_result.py "
        f"{result_json_name(variant)} "
        f"--preset {validation_preset(variant)} "
        f"--json-out {validation_json_name(variant)}"
    )


def bundle_readme_text(src: Path, variant: str) -> str:
    text = src.read_text()
    return (
        text
        + f"\n\nBundle-local {variant} AXIS/DMA smoke:\n\n"
        + f"    {smoke_command(variant)}\n"
        + f"\nBundle-local {variant} AXIS/DMA result validation:\n\n"
        + f"    {validation_command(variant)}\n"
    )


def artifact_files(variant: str) -> list[str]:
    prefix = ARTIFACT_PREFIXES[variant]
    return [
        f"pynq/hgtxr/{prefix}.bit",
        f"pynq/hgtxr/{prefix}.hwh",
    ]


def build_bundle(
    out_dir: Path = DEFAULT_OUT_DIR,
    tar_path: Path | None = DEFAULT_TAR,
    variant: str = DEFAULT_VARIANT,
) -> dict[str, Any]:
    if variant not in ARTIFACT_PREFIXES:
        raise ValueError(f"unsupported variant: {variant}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    files: list[dict[str, Any]] = []
    for rel in PACKAGE_FILES + artifact_files(variant):
        src = HARDWARE_ROOT / rel
        dst = out_dir / "hgtxr" / Path(rel).name
        if rel == "pynq/hgtxr/README.md":
            copied = write_text_file(dst, bundle_readme_text(src, variant))
        else:
            copied = copy_file(src, dst)
        files.append({"source": rel, "bundle": f"hgtxr/{dst.name}", **copied})
    for rel in WEIGHT_FILES:
        src = HARDWARE_ROOT / rel
        dst = out_dir / "weights" / Path(rel).name
        files.append({"source": rel, "bundle": f"weights/{dst.name}", **copy_file(src, dst)})
    for rel in TOOL_FILES:
        src = HARDWARE_ROOT / rel
        dst = out_dir / "tools" / Path(rel).name
        files.append({"source": rel, "bundle": f"tools/{dst.name}", **copy_file(src, dst)})

    command = smoke_command(variant)
    script_name = f"run_e2e_axis_dma_{variant.replace('-', '_')}_file_smoke.sh"
    run_script = out_dir / script_name
    run_script.write_text("#!/usr/bin/env sh\nset -eu\n" + command + "\n")
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
    validate_command = validation_command(variant)
    validate_script_name = f"validate_e2e_axis_dma_{variant.replace('-', '_')}_file_smoke.sh"
    validate_script = out_dir / validate_script_name
    validate_script.write_text("#!/usr/bin/env sh\nset -eu\n" + validate_command + "\n")
    validate_script.chmod(0o755)
    files.append(
        {
            "source": "generated",
            "bundle": validate_script.name,
            "path": str(validate_script),
            "bytes": validate_script.stat().st_size,
            "sha256": sha256_file(validate_script),
        }
    )

    manifest = {
        "name": "hgtxr_e2e_axis_dma_smoke_bundle",
        "target": "ZCU104 PYNQ",
        "variant": variant,
        "artifact_prefix": ARTIFACT_PREFIXES[variant],
        "command": command,
        "validation_command": validate_command,
        "expected_runtime_state": EXPECTED_RUNTIME_STATE_BY_VARIANT.get(variant, 2),
        "expected_out_raw": EXPECTED_RAW_BY_VARIANT[variant],
        "files": files,
    }
    if variant in MODE_PROFILE_BY_VARIANT:
        manifest["mode_profile"] = MODE_PROFILE_BY_VARIANT[variant]
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
        sha_path = tar_path.with_name(tar_path.name + ".sha256")
        sha_path.write_text(f"{manifest['tar']['sha256']}  {tar_path.name}\n")
        manifest["tar"]["sha256_file"] = str(sha_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return {"out_dir": str(out_dir), "manifest": str(manifest_path), "tar": manifest.get("tar")}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package HGTXR E2E AXIS/DMA PYNQ smoke bundle.")
    parser.add_argument("--variant", choices=sorted(ARTIFACT_PREFIXES), default=DEFAULT_VARIANT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tar", dest="tar_path", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--no-tar", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_bundle(args.out_dir, None if args.no_tar else args.tar_path, args.variant)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
