from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


def angle_to_uv(theta):
    if isinstance(theta, torch.Tensor):
        return torch.sin(2.0 * theta), torch.cos(2.0 * theta)
    arr = np.asarray(theta, dtype=np.float32)
    return np.sin(2.0 * arr), np.cos(2.0 * arr)


def uv_to_angle(u, v):
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


def xywht_to_xyabuv(state):
    if isinstance(state, torch.Tensor):
        u, v = angle_to_uv(state[..., 4])
        return torch.stack([state[..., 0], state[..., 1], state[..., 2], state[..., 3], u, v], dim=-1)
    arr = np.asarray(state, dtype=np.float32)
    u, v = angle_to_uv(arr[..., 4])
    return np.stack([arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3], u, v], axis=-1).astype(np.float32)


def xyabuv_to_xywht(state):
    if isinstance(state, torch.Tensor):
        theta = uv_to_angle(state[..., 4], state[..., 5])
        return torch.stack([state[..., 0], state[..., 1], state[..., 2], state[..., 3], theta], dim=-1)
    arr = np.asarray(state, dtype=np.float32)
    theta = uv_to_angle(arr[..., 4], arr[..., 5])
    return np.stack([arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3], theta], axis=-1).astype(np.float32)


def legacy_track_delta_to_uv(prev_state_xyabuv, cur_state_xyabuv):
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
