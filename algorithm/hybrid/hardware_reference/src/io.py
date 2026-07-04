from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def normalize_user_path(path_str: str | Path, base: str | Path | None = None) -> Path:
    raw = str(path_str).replace("\\", "/")
    if len(raw) >= 3 and raw[1] == ":" and raw[2] == "/":
        raw = "/mnt/" + raw[0].lower() + raw[2:]
    path = Path(raw)
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_cache_file(root: str | Path, prefix: str, sample_id: str, payload: dict[str, Any]) -> Path:
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Path(root) / prefix / f"{sample_id}_{digest}.npz"


def load_or_build_npz_array(cache_path: str | Path, key: str, builder: Callable[[], np.ndarray]) -> tuple[np.ndarray, bool]:
    p = Path(cache_path)
    if p.exists():
        return np.load(p)[key], True
    arr = builder()
    ensure_dir(p.parent)
    np.savez_compressed(p, **{key: arr})
    return arr, False


def angle_to_uv(theta: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    return np.cos(theta), np.sin(theta)


def uv_to_angle(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    return np.arctan2(v, u)


def xywht_to_xyabuv(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    u, v = angle_to_uv(arr[..., 4])
    return np.concatenate([arr[..., :4], np.expand_dims(u, -1), np.expand_dims(v, -1)], axis=-1)


def xyabuv_to_xywht(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    theta = uv_to_angle(arr[..., 4], arr[..., 5])
    return np.concatenate([arr[..., :4], np.expand_dims(theta, -1)], axis=-1)

