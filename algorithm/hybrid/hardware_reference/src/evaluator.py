from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .dataset import make_synthetic_batch
from .io import ensure_dir, write_json
from .metrics import compute_metrics
from .model import build_model


@torch.no_grad()
def evaluate(cfg: dict[str, Any], *, checkpoint: str | None = None) -> dict[str, float]:
    device = torch.device(cfg.get("training", {}).get("device", "cpu"))
    model = build_model(cfg).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    batch = make_synthetic_batch(2, input_size=tuple(cfg["data"]["input_size"]))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    metrics = compute_metrics(batch, model.forward_train(batch))
    result = {k: float(v.cpu().item()) for k, v in metrics.items()}
    out_dir = ensure_dir(cfg["experiment"].get("output_dir") or "runs/eval")
    write_json(result, Path(out_dir) / "metrics.json")
    return result

