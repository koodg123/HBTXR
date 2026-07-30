from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional, Tuple

import numpy as np

from .hgtxr_overlay import DATA_FRAC_BITS, STATE_SHAPE, _find_ip, from_fixed16, parse_registers_from_hwh, to_fixed16

try:
    from pynq import Overlay, allocate
except Exception:  # pragma: no cover
    Overlay = None
    allocate = None

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BIT = PACKAGE_DIR / "hgtxr_e2e_m_axi.bit"
DEFAULT_HWH = PACKAGE_DIR / "hgtxr_e2e_m_axi.hwh"
FRAME_SHAPE = (256, 256)
E2E_WEIGHT_WORDS = 381024
U32_PER_256B_WORD = 8


@dataclass
class E2EMaxiRuntimeConfig:
    bitfile: Path = DEFAULT_BIT
    hwhfile: Path = DEFAULT_HWH
    ip_name: Optional[str] = None
    timeout_s: float = 60.0
    poll_s: float = 0.001
    frac_bits: int = DATA_FRAC_BITS


class HgtxrE2EMaxiOverlay:
    def __init__(self, config: E2EMaxiRuntimeConfig = E2EMaxiRuntimeConfig()):
        if Overlay is None or allocate is None:
            raise RuntimeError("pynq is required on the target board")
        self.config = config
        self.overlay = Overlay(str(config.bitfile))
        self.ip = _find_ip(self.overlay, config.ip_name)
        self.mmio = self.ip.mmio if hasattr(self.ip, "mmio") else self.ip["mmio"]
        self.register_map = getattr(self.ip, "register_map", None)
        self.hwh_registers = parse_registers_from_hwh(config.hwhfile) if Path(config.hwhfile).exists() else {}
        self.buffers = []

    def close(self) -> None:
        for buf in self.buffers:
            if hasattr(buf, "free"):
                buf.free()
        self.buffers.clear()

    def __enter__(self) -> "HgtxrE2EMaxiOverlay":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    @staticmethod
    def _phys(buf: Any) -> int:
        if hasattr(buf, "physical_address"):
            return int(buf.physical_address)
        if hasattr(buf, "device_address"):
            return int(buf.device_address)
        return int(buf.ctypes.data)

    def _alloc_frame(self, frame: Any) -> Any:
        arr = to_fixed16(frame, self.config.frac_bits).reshape(FRAME_SHAPE)
        buf = allocate(shape=FRAME_SHAPE, dtype=np.int16)
        buf[:] = arr
        self.buffers.append(buf)
        return buf

    def _alloc_weights(self, weights_u32: Optional[Any]) -> Any:
        shape = (E2E_WEIGHT_WORDS * U32_PER_256B_WORD,)
        buf = allocate(shape=shape, dtype=np.uint32)
        if weights_u32 is None:
            buf[:] = 0
        else:
            arr = np.asarray(weights_u32, dtype=np.uint32).reshape(-1)
            if arr.size > buf.size:
                if hasattr(buf, "free"):
                    buf.free()
                raise ValueError(f"weights_u32 has {arr.size} uint32 values, capacity is {buf.size}")
            buf[:] = 0
            buf[: arr.size] = arr
        self.buffers.append(buf)
        return buf

    def _alloc_zero(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        buf = allocate(shape=shape, dtype=dtype)
        buf[:] = 0
        self.buffers.append(buf)
        return buf

    def _write_addr(self, name: str, addr: int) -> None:
        low_name = name if name in self.hwh_registers else f"{name}_1"
        high_name = f"{name}_2"
        if self.register_map is not None:
            if hasattr(self.register_map, low_name):
                setattr(self.register_map, low_name, int(addr) & 0xFFFFFFFF)
                if hasattr(self.register_map, high_name):
                    setattr(self.register_map, high_name, (int(addr) >> 32) & 0xFFFFFFFF)
                return
            if hasattr(self.register_map, name):
                setattr(self.register_map, name, int(addr))
                return
        low = self.hwh_registers.get(low_name)
        if low is None:
            raise KeyError(f"Register {name} not found in HWH/register_map")
        self.mmio.write(low, int(addr) & 0xFFFFFFFF)
        high = self.hwh_registers.get(high_name, low + 4)
        self.mmio.write(high, (int(addr) >> 32) & 0xFFFFFFFF)

    def _start(self) -> None:
        if self.register_map is not None and hasattr(self.register_map, "CTRL"):
            self.register_map.CTRL.AP_START = 1
            return
        self.mmio.write(0x00, self.mmio.read(0x00) | 1)

    def _done(self) -> bool:
        if self.register_map is not None and hasattr(self.register_map, "CTRL"):
            return bool(self.register_map.CTRL.AP_DONE)
        return bool((self.mmio.read(0x00) >> 1) & 1)

    def run(self, frame: Any, weights_u32: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        frame_buf = self._alloc_frame(frame)
        weights_buf = self._alloc_weights(weights_u32)
        out_state_buf = self._alloc_zero(STATE_SHAPE, np.int16)
        runtime_state_buf = self._alloc_zero((1,), np.int32)

        self._write_addr("frame", self._phys(frame_buf))
        self._write_addr("weights", self._phys(weights_buf))
        self._write_addr("out_state", self._phys(out_state_buf))
        self._write_addr("runtime_state", self._phys(runtime_state_buf))

        for buf in [frame_buf, weights_buf, out_state_buf, runtime_state_buf]:
            if hasattr(buf, "flush"):
                buf.flush()

        self._start()
        deadline = time.monotonic() + self.config.timeout_s
        while not self._done():
            if time.monotonic() > deadline:
                raise TimeoutError("HGTXR E2E m_axi IP did not assert AP_DONE")
            time.sleep(self.config.poll_s)

        for buf in [out_state_buf, runtime_state_buf]:
            if hasattr(buf, "invalidate"):
                buf.invalidate()
        return from_fixed16(np.array(out_state_buf), self.config.frac_bits), np.array(runtime_state_buf, dtype=np.int32)
