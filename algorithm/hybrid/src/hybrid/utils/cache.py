from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _digest(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:16]


def build_cache_file(root: str | Path, prefix: str, sample_id: str, payload: dict) -> Path:
    root = ensure_dir(root)
    return root / f"{prefix}_{sample_id}_{_digest(payload)}.npz"


def load_or_build_npz_array(cache_path: str | Path, key: str, builder: Callable[[], np.ndarray]) -> tuple[np.ndarray, bool]:
    p = Path(cache_path)
    if p.exists():
        arr = np.load(p)[key]
        return arr, True
    arr = builder()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{key: arr})
    return arr, False
