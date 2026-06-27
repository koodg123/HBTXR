"""Artifact loading helpers for ICCAD24-HG-PIPE quantization references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np


INT_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class HgPipeSource:
    root: Path

    @classmethod
    def from_path(cls, path: str | Path) -> "HgPipeSource":
        root = Path(path).resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        refs = root / "case" / "refs"
        if not refs.exists():
            raise FileNotFoundError(f"Missing refs directory: {refs}")
        return cls(root=root)

    @property
    def refs(self) -> Path:
        return self.root / "case" / "refs"

    @property
    def statistics(self) -> Path:
        return self.root / "statistics"


def read_ints(path: Path) -> list[int]:
    """Read C/C++ include-style comma separated integer literals."""
    return [int(match) for match in INT_RE.findall(path.read_text())]


def load_statistics(source: HgPipeSource) -> dict[str, Any]:
    type_path = source.statistics / "type.npy"
    range_path = source.statistics / "range.npy"
    return {
        "type": np.load(type_path, allow_pickle=True).item() if type_path.exists() else {},
        "range": np.load(range_path, allow_pickle=True).item() if range_path.exists() else {},
    }


def stem_to_stat_key(stem: str) -> str:
    """Map ref file stems such as attn_0_qq to statistics keys like attn0.qq."""
    attn = re.match(r"attn_(\d+)_(.+)$", stem)
    if attn:
        return f"attn{attn.group(1)}.{attn.group(2)}"
    mlp = re.match(r"mlp_(\d+)_(.+)$", stem)
    if mlp:
        return f"mlp{mlp.group(1)}.{mlp.group(2)}"
    if stem.startswith("head_"):
        return "head." + stem[len("head_") :]
    if stem.startswith("patch_embed_"):
        return "patch_embed." + stem[len("patch_embed_") :]
    return stem.replace("_", ".")


def summarize_tensor(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(sum(values)) / len(values),
    }


def metadata_for(stem: str, stats: dict[str, Any]) -> dict[str, Any]:
    key = stem_to_stat_key(stem)
    type_dict = stats.get("type", {})
    range_dict = stats.get("range", {})
    return {
        "stat_key": key,
        "input_type": type_dict.get(f"{key}.input"),
        "output_type": type_dict.get(f"{key}.output"),
        "range": range_dict.get(key, {}),
    }
