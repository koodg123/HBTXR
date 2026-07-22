"""state6 angle codec: (x,y,a,b,theta) <-> (x,y,a,b,u,v) round-trip."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from utils.state6 import xyabuv_to_xywht, xywht_to_xyabuv


def test_xywht_xyabuv_roundtrip_tensor():
    state = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.6]])   # theta in [0, pi/2) round-trips
    uv = xywht_to_xyabuv(state)
    assert uv.shape[-1] == 6
    back = xyabuv_to_xywht(uv)
    assert torch.allclose(back[..., :4], state[..., :4], atol=1e-5)
    assert abs(float(back[0, 4]) - 0.6) < 1e-4


def test_uv_is_unit_encoding():
    state = torch.tensor([[0.0, 0.0, 0.2, 0.1, 0.3]])
    uv = xywht_to_xyabuv(state)
    norm = float((uv[0, 4] ** 2 + uv[0, 5] ** 2) ** 0.5)
    assert abs(norm - 1.0) < 1e-4
