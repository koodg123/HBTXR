"""Inference entrypoint for the reimplemented HBTXR models (P5.7).

Replaces the old detector-coupled ``apps/infer.py`` (streaming_inference +
process_detector_prediction). Runs batched forward inference over a test manifest
and writes per-sample predicted pupil states.

    load_config -> make_model -> load_checkpoint -> build_dataloader(test,
    shuffle=False) -> predict_batch -> write predictions.csv (+ states.npz).

Run from the ``algorithm/`` root::

    python -m engine.infer.entrypoint -c configs/experiment/frame_hbtxr.yaml \
        --ckpt runs/.../final.pt -o predictions.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from engine.data.adapter import resolve_modality
from engine.infer.inference import predict_batch
from engine.model_factory import make_model
from engine.runspec.run_contract import resolve_manifest_path
from engine.tools.checkpoint import load_checkpoint
from engine.tools.load_config import load_config

if TYPE_CHECKING:  # engine.data.factory is only imported at call time (see below)
    from engine.data.factory import AdaptedLoader

_STATE_COLUMNS = ("x", "y", "a", "b", "theta")


def _build_dataloader(
    manifest_path: str,
    cfg: dict[str, Any],
    *,
    shuffle: bool,
    modality: str | None = None,
) -> AdaptedLoader:
    """Lazy proxy to ``engine.data.factory.build_dataloader``.

    ``engine.data.factory`` pulls the full HBTXR data pipeline (PIL / cv2 / h5py), which
    made merely *importing* this module — and therefore the ``hbtxr infer`` dispatch —
    fail wherever those are absent. Kept at module scope rather than inlined into
    ``run_infer`` so it stays patchable without importing the pipeline to reach it.
    """
    from engine.data.factory import build_dataloader

    return build_dataloader(manifest_path, cfg, shuffle=shuffle, modality=modality)


def _resolve_project_root(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    root = (cfg.get("experiment") or {}).get("project_root")
    return Path(root).expanduser().resolve() if root else Path.cwd()


def _sample_ids(batch: dict[str, Any], count: int) -> list[str]:
    ids = batch.get("sample_id")
    if isinstance(ids, (list, tuple)) and len(ids) == count:
        return [str(v) for v in ids]
    return [str(i) for i in range(count)]


def run_infer(
    config_path: str,
    *,
    output: str = "predictions.csv",
    ckpt: str | None = None,
    project_root: str | None = None,
    device: str | None = None,
) -> Path:
    """Batched forward inference over the test manifest; write predicted states."""
    cfg = load_config(config_path)
    root = _resolve_project_root(cfg, project_root)
    modality = resolve_modality(cfg)
    dev = device or str((cfg.get("training") or {}).get("device", "cpu"))

    model = make_model(cfg["model"])
    ckpt_path = ckpt or (cfg.get("test") or cfg.get("infer") or {}).get("ckpt_path")
    if ckpt_path:
        load_checkpoint(model, ckpt_path, map_location=dev)
    model = model.to(dev)
    model.eval()

    manifest = str(resolve_manifest_path(cfg, project_root=root, split="test"))
    loader = _build_dataloader(manifest, cfg, shuffle=False, modality=modality)

    ids: list[str] = []
    states: list[np.ndarray] = []
    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = predict_batch(model, batch, modality)
        state = out.get("state")
        if state is None:
            continue
        state_np = state.detach().cpu().numpy()
        ids.extend(_sample_ids(batch, state_np.shape[0]))
        states.append(state_np)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.concatenate(states, axis=0) if states else np.zeros((0, len(_STATE_COLUMNS)), dtype=np.float32)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *_STATE_COLUMNS])
        for sample_id, row in zip(ids, stacked):
            writer.writerow([sample_id, *(float(v) for v in row)])
    np.savez(out_path.with_suffix(".npz"), sample_id=np.asarray(ids), state=stacked)
    print(f"wrote {len(ids)} predictions -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run inference with a reimplemented HBTXR model.")
    parser.add_argument("-c", "--config", required=True, help="experiment config path (yaml/json)")
    parser.add_argument("-o", "--output", default="predictions.csv", help="predictions csv path")
    parser.add_argument("--ckpt", default=None, help="checkpoint path (overrides config)")
    parser.add_argument("--project-root", default=None, help="root for manifest resolution")
    parser.add_argument("--device", default=None, help="override inference device")
    args = parser.parse_args(argv)
    run_infer(args.config, output=args.output, ckpt=args.ckpt, project_root=args.project_root, device=args.device)


if __name__ == "__main__":
    main()
