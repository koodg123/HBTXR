from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .dataset import make_synthetic_batch
from .io import ensure_dir, write_json, write_jsonl
from .model import build_model
from .runtime import RuntimeHGTXRTracker


@torch.no_grad()
def infer(cfg: dict[str, Any], *, checkpoint: str | None = None) -> dict[str, Any]:
    device = torch.device(cfg.get("training", {}).get("device", "cpu"))
    model = build_model(cfg).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    tracker = RuntimeHGTXRTracker(model, hold_last_on_blink=bool(cfg.get("runtime", {}).get("hold_last_on_blink", True)))
    batch = make_synthetic_batch(3, input_size=tuple(cfg["data"]["input_size"]))
    rows = []
    for idx in range(batch["frame"].shape[0]):
        outputs = tracker.step(frame=batch["frame"][idx:idx + 1].to(device), event=batch["event"][idx:idx + 1].to(device), event_density=batch["event_density"][idx:idx + 1].to(device))
        rows.append({
            "sample_id": f"synthetic_{idx}",
            "runtime_state": str(outputs["runtime/state"]),
            "runtime_reason": str(outputs["runtime/reason"]),
            "ellipse_state": outputs["runtime/ellipse_state"][0].detach().cpu().tolist(),
        })
    out_dir = ensure_dir(cfg["experiment"].get("output_dir") or "runs/infer")
    write_jsonl(rows, Path(out_dir) / "runtime_trace.jsonl")
    write_json({"num_samples": len(rows), "states": [r["runtime_state"] for r in rows]}, Path(out_dir) / "summary.json")
    return {"rows": rows, "output_dir": str(out_dir)}

