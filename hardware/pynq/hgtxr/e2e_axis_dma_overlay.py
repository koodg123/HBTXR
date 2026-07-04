from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional, Tuple

import numpy as np

from .hgtxr_overlay import DATA_FRAC_BITS, FRAME_SHAPE, STATE_SHAPE, _find_ip, from_fixed16, parse_registers_from_hwh, to_fixed16
from .e2e_m_axi_overlay import E2E_WEIGHT_WORDS, U32_PER_256B_WORD

try:
    from pynq import Overlay, allocate
except Exception:  # pragma: no cover
    Overlay = None
    allocate = None

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BIT = PACKAGE_DIR / "hgtxr_e2e_axis_dma.bit"
DEFAULT_HWH = PACKAGE_DIR / "hgtxr_e2e_axis_dma.hwh"
AXIS_U32_PER_WORD = 8
AXIS_FRAME_WORDS = FRAME_SHAPE[0] * FRAME_SHAPE[1]


@dataclass
class E2EAxisDmaRuntimeConfig:
    bitfile: Path = DEFAULT_BIT
    hwhfile: Path = DEFAULT_HWH
    ip_name: Optional[str] = None
    dma_in_name: str = "axi_dma_in"
    dma_out_name: str = "axi_dma_out"
    timeout_s: float = 60.0
    poll_s: float = 0.001
    frac_bits: int = DATA_FRAC_BITS


class HgtxrE2EAxisDmaOverlay:
    def __init__(self, config: E2EAxisDmaRuntimeConfig = E2EAxisDmaRuntimeConfig()):
        if Overlay is None or allocate is None:
            raise RuntimeError("pynq is required on the target board")
        self.config = config
        self.overlay = Overlay(str(config.bitfile))
        self.ip = _find_ip(self.overlay, config.ip_name)
        self.mmio = self.ip.mmio if hasattr(self.ip, "mmio") else self.ip["mmio"]
        self.register_map = getattr(self.ip, "register_map", None)
        self.hwh_registers = parse_registers_from_hwh(config.hwhfile) if Path(config.hwhfile).exists() else {}
        self.dma_in = self._find_dma(config.dma_in_name)
        self.dma_out = self._find_dma(config.dma_out_name)
        self.buffers = []

    def _find_dma(self, name: str) -> Any:
        if hasattr(self.overlay, name):
            return getattr(self.overlay, name)
        ip_dict = getattr(self.overlay, "ip_dict", {})
        for key in ip_dict:
            attr = key.replace("/", "_")
            if key.endswith(name) or attr.endswith(name):
                return getattr(self.overlay, attr, ip_dict[key])
        raise RuntimeError(f"DMA {name} not found in overlay")

    def close(self) -> None:
        for buf in self.buffers:
            if hasattr(buf, "free"):
                buf.free()
        self.buffers.clear()

    def _free_buffers_since(self, start_idx: int) -> None:
        run_buffers = self.buffers[start_idx:]
        del self.buffers[start_idx:]
        for buf in run_buffers:
            if hasattr(buf, "free"):
                buf.free()

    def __enter__(self) -> "HgtxrE2EAxisDmaOverlay":
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

    def _alloc_axis_frame(self, frame: Any) -> Any:
        arr = np.asarray(frame)
        if arr.ndim != 2:
            arr = arr.reshape(FRAME_SHAPE)
        if np.issubdtype(arr.dtype, np.integer):
            raw = arr.astype(np.int64) & 0xFF
        else:
            raw = np.rint(arr.astype(np.float64) * 128.0)
            raw = np.clip(raw, 0, 255).astype(np.uint8)
        buf = allocate(shape=(int(raw.size), AXIS_U32_PER_WORD), dtype=np.uint32)
        buf[:] = 0
        buf[:, 0] = raw.reshape(-1).astype(np.uint32)
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

    def _write_scalar(self, name: str, value: int) -> None:
        if self.register_map is not None and hasattr(self.register_map, name):
            setattr(self.register_map, name, int(value))
            return
        offset = self.hwh_registers.get(name)
        if offset is None:
            raise KeyError(f"Register {name} not found in HWH/register_map")
        self.mmio.write(offset, int(value))

    def _start(self) -> None:
        if self.register_map is not None and hasattr(self.register_map, "CTRL"):
            self.register_map.CTRL.AP_START = 1
            return
        self.mmio.write(0x00, self.mmio.read(0x00) | 1)

    def _done(self) -> bool:
        if self.register_map is not None and hasattr(self.register_map, "CTRL"):
            return bool(self.register_map.CTRL.AP_DONE)
        return bool((self.mmio.read(0x00) >> 1) & 1)

    @staticmethod
    def _wait_channel(channel: Any) -> None:
        if hasattr(channel, "wait"):
            channel.wait()

    def run(
        self,
        frame: Any,
        weights_u32: Optional[Any] = None,
        num_pixels_control: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        out_state, runtime_state, _metrics = self.run_profiled(
            frame, weights_u32=weights_u32, num_pixels_control=num_pixels_control
        )
        return out_state, runtime_state

    def run_profiled(
        self,
        frame: Any,
        weights_u32: Optional[Any] = None,
        num_pixels_control: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        start_idx = len(self.buffers)
        total_start = time.perf_counter()
        alloc_start = total_start
        frame_buf = self._alloc_axis_frame(frame)
        weights_buf = self._alloc_weights(weights_u32)
        out_axis_buf = self._alloc_zero((STATE_SHAPE[0], AXIS_U32_PER_WORD), np.uint32)
        runtime_state_buf = self._alloc_zero((1,), np.int32)
        alloc_done = time.perf_counter()

        frame_dma_bytes = int(np.asarray(frame_buf).nbytes)
        output_dma_bytes = int(np.asarray(out_axis_buf).nbytes)
        weights_bytes = int(np.asarray(weights_buf).nbytes)

        control_value = int(num_pixels_control) if num_pixels_control is not None else int(np.asarray(frame_buf).shape[0])
        self._write_addr("weights", self._phys(weights_buf))
        self._write_scalar("num_pixels", control_value)
        self._write_addr("runtime_state", self._phys(runtime_state_buf))

        flush_start = time.perf_counter()
        for buf in [frame_buf, weights_buf, runtime_state_buf]:
            if hasattr(buf, "flush"):
                buf.flush()
        flush_done = time.perf_counter()

        output_submit_start = time.perf_counter()
        self.dma_out.recvchannel.transfer(out_axis_buf)
        output_submit_done = time.perf_counter()
        kernel_start = time.perf_counter()
        self._start()
        input_submit_start = time.perf_counter()
        self.dma_in.sendchannel.transfer(frame_buf)
        input_submit_done = time.perf_counter()
        input_wait_start = time.perf_counter()
        self._wait_channel(self.dma_in.sendchannel)
        input_wait_done = time.perf_counter()
        output_wait_start = time.perf_counter()
        self._wait_channel(self.dma_out.recvchannel)
        output_wait_done = time.perf_counter()

        deadline = time.monotonic() + self.config.timeout_s
        ap_done_poll_start = time.perf_counter()
        while not self._done():
            if time.monotonic() > deadline:
                raise TimeoutError("HGTXR E2E AXIS IP did not assert AP_DONE")
            time.sleep(self.config.poll_s)
        ap_done_poll_done = time.perf_counter()

        invalidate_start = time.perf_counter()
        for buf in [out_axis_buf, runtime_state_buf]:
            if hasattr(buf, "invalidate"):
                buf.invalidate()
        invalidate_done = time.perf_counter()
        raw = (np.asarray(out_axis_buf)[:, 0] & 0xFFFF).astype(np.uint16).view(np.int16)
        out_state = from_fixed16(raw, 4)
        runtime_state = np.array(runtime_state_buf, dtype=np.int32)
        total_done = time.perf_counter()

        input_dma_wait_s = input_wait_done - input_wait_start
        output_dma_wait_plus_compute_s = output_wait_done - output_wait_start
        aggregate_dma_window_s = input_dma_wait_s + output_dma_wait_plus_compute_s
        metrics: dict[str, Any] = {
            "clock": "time.perf_counter",
            "latency_s": total_done - total_start,
            "batch_size": 1,
            "frame_dma_bytes": frame_dma_bytes,
            "output_dma_bytes": output_dma_bytes,
            "weights_buffer_bytes": weights_bytes,
            "total_dma_bytes": frame_dma_bytes + output_dma_bytes,
            "useful_input_payload_bytes": int(np.asarray(frame_buf).shape[0]),
            "num_pixels_control": control_value,
            "frame_shape": list(np.asarray(frame).shape),
            "allocation_s": alloc_done - alloc_start,
            "flush_s": flush_done - flush_start,
            "dma_output_submit_s": output_submit_done - output_submit_start,
            "dma_input_submit_s": input_submit_done - input_submit_start,
            "input_dma_wait_s": input_dma_wait_s,
            "output_dma_wait_plus_compute_s": output_dma_wait_plus_compute_s,
            "kernel_observed_s": output_wait_done - kernel_start,
            "ap_done_poll_s": ap_done_poll_done - ap_done_poll_start,
            "invalidate_s": invalidate_done - invalidate_start,
            "input_dma_measured_bandwidth_Bps": frame_dma_bytes / input_dma_wait_s if input_dma_wait_s > 0.0 else None,
            "output_dma_effective_bandwidth_Bps": (
                output_dma_bytes / output_dma_wait_plus_compute_s if output_dma_wait_plus_compute_s > 0.0 else None
            ),
            "aggregate_dma_effective_bandwidth_Bps": (
                (frame_dma_bytes + output_dma_bytes) / aggregate_dma_window_s if aggregate_dma_window_s > 0.0 else None
            ),
            "measurement_note": "output_dma_wait_plus_compute_s includes accelerator compute because the output DMA completes when the kernel emits state words",
        }
        self._free_buffers_since(start_idx)
        return out_state, runtime_state, metrics
