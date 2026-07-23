from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch


_WINDOWS_ABS_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
_WSL_ABS_RE = re.compile(r"^/mnt/(?P<drive>[A-Za-z])/(?P<rest>.*)$")


def _normalize_path_string(raw: str) -> str:
    text = os.path.expandvars(os.path.expanduser(str(raw)))
    if os.name != "nt":
        match = _WINDOWS_ABS_RE.match(text)
        if match:
            rest = match.group("rest").replace("\\", "/")
            return f"/mnt/{match.group('drive').lower()}/{rest}"
        return text
    match = _WSL_ABS_RE.match(text)
    if match:
        rest = match.group("rest").replace("/", "\\")
        return f"{match.group('drive').upper()}:\\{rest}"
    return text


def normalize_user_path(path_str: str | Path, base: str | Path | None = None) -> Path:
    path = Path(_normalize_path_string(str(path_str)))
    if not path.is_absolute() and base is not None:
        path = normalize_user_path(base) / path
    return Path(os.path.abspath(str(path))) if path.is_absolute() else path


def resolve_stored_path(root: str | Path, stored_path: str | Path) -> Path:
    path = normalize_user_path(stored_path)
    if path.is_absolute():
        return path
    return normalize_user_path(path, base=root)


def relativize_to(root: str | Path, path: str | Path) -> str:
    root_path = normalize_user_path(root)
    target_path = normalize_user_path(path)
    try:
        return str(target_path.relative_to(root_path))
    except ValueError:
        return str(target_path)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    return json.loads(normalize_user_path(path).read_text(encoding="utf-8"))


def write_json(obj: Any, path: str | Path) -> None:
    dst = normalize_user_path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    src = normalize_user_path(path)
    rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    dst = normalize_user_path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def build_cache_file(root: str | Path, prefix: str, sample_id: str, payload: dict[str, Any]) -> Path:
    return ensure_dir(root) / f"{prefix}_{sample_id}_{_digest(payload)}.npz"


def load_or_build_npz_array(cache_path: str | Path, key: str, builder: Callable[[], np.ndarray]) -> tuple[np.ndarray, bool]:
    cache_file = Path(cache_path)
    if cache_file.exists():
        return np.load(cache_file)[key], True
    array = builder()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **{key: array})
    return array, False


def angle_to_uv(theta: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(theta, torch.Tensor):
        return torch.sin(2.0 * theta), torch.cos(2.0 * theta)
    arr = np.asarray(theta, dtype=np.float32)
    return np.sin(2.0 * arr), np.cos(2.0 * arr)


def uv_to_angle(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    if isinstance(u, torch.Tensor) or isinstance(v, torch.Tensor):
        if not isinstance(u, torch.Tensor):
            u = torch.as_tensor(u, dtype=torch.float32)
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=torch.float32, device=u.device)
        norm = torch.clamp(torch.sqrt(u * u + v * v), min=1e-6)
        return 0.5 * torch.atan2(u / norm, v / norm)
    u_arr = np.asarray(u, dtype=np.float32)
    v_arr = np.asarray(v, dtype=np.float32)
    norm = np.maximum(np.sqrt(u_arr * u_arr + v_arr * v_arr), 1e-6)
    return 0.5 * np.arctan2(u_arr / norm, v_arr / norm)


def xywht_to_xyabuv(state: np.ndarray) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        u, v = angle_to_uv(state[..., 4])
        return torch.stack([state[..., 0], state[..., 1], state[..., 2], state[..., 3], u, v], dim=-1)
    arr = np.asarray(state, dtype=np.float32)
    u, v = angle_to_uv(arr[..., 4])
    return np.stack([arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3], u, v], axis=-1).astype(np.float32)


def xyabuv_to_xywht(state: np.ndarray) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        theta = uv_to_angle(state[..., 4], state[..., 5])
        return torch.stack([state[..., 0], state[..., 1], state[..., 2], state[..., 3], theta], dim=-1)
    arr = np.asarray(state, dtype=np.float32)
    theta = uv_to_angle(arr[..., 4], arr[..., 5])
    return np.stack([arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3], theta], axis=-1).astype(np.float32)


def legacy_track_delta_to_uv(prev_state_xyabuv: np.ndarray, cur_state_xyabuv: np.ndarray) -> np.ndarray:
    if isinstance(prev_state_xyabuv, torch.Tensor) or isinstance(cur_state_xyabuv, torch.Tensor):
        if not isinstance(prev_state_xyabuv, torch.Tensor):
            prev_state_xyabuv = torch.as_tensor(prev_state_xyabuv, dtype=torch.float32)
        if not isinstance(cur_state_xyabuv, torch.Tensor):
            cur_state_xyabuv = torch.as_tensor(cur_state_xyabuv, dtype=torch.float32, device=prev_state_xyabuv.device)
        return torch.stack(
            [
                cur_state_xyabuv[..., 0] - prev_state_xyabuv[..., 0],
                cur_state_xyabuv[..., 1] - prev_state_xyabuv[..., 1],
                torch.log(torch.clamp(cur_state_xyabuv[..., 2], min=1e-3) / torch.clamp(prev_state_xyabuv[..., 2], min=1e-3)),
                torch.log(torch.clamp(cur_state_xyabuv[..., 3], min=1e-3) / torch.clamp(prev_state_xyabuv[..., 3], min=1e-3)),
                cur_state_xyabuv[..., 4] - prev_state_xyabuv[..., 4],
                cur_state_xyabuv[..., 5] - prev_state_xyabuv[..., 5],
            ],
            dim=-1,
        )
    prev_arr = np.asarray(prev_state_xyabuv, dtype=np.float32)
    cur_arr = np.asarray(cur_state_xyabuv, dtype=np.float32)
    return np.asarray(
        [
            cur_arr[0] - prev_arr[0],
            cur_arr[1] - prev_arr[1],
            math.log(max(float(cur_arr[2]), 1e-3) / max(float(prev_arr[2]), 1e-3)),
            math.log(max(float(cur_arr[3]), 1e-3) / max(float(prev_arr[3]), 1e-3)),
            cur_arr[4] - prev_arr[4],
            cur_arr[5] - prev_arr[5],
        ],
        dtype=np.float32,
    )
