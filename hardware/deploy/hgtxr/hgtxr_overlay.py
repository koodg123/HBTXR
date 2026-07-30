from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np

try:
    from pynq import Overlay, allocate
except Exception:  # pragma: no cover
    Overlay = None
    allocate = None

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BIT = PACKAGE_DIR / "hgtxr.bit"
DEFAULT_HWH = PACKAGE_DIR / "hgtxr.hwh"
DATA_FRAC_BITS = 10
FRAME_SHAPE = (256, 256)
STATE_SHAPE = (6,)


def to_fixed16(values: Any, frac_bits: int = DATA_FRAC_BITS) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int16, copy=False)
    scaled = np.rint(arr.astype(np.float64) * float(1 << frac_bits))
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def from_fixed16(values: Any, frac_bits: int = DATA_FRAC_BITS) -> np.ndarray:
    return np.asarray(values, dtype=np.int16).astype(np.float32) / float(1 << frac_bits)


def parse_registers_from_hwh(hwh_path: Path) -> Dict[str, int]:
    root = ET.parse(hwh_path).getroot()
    regs: Dict[str, int] = {}
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag != "register":
            continue
        name = node.attrib.get("NAME") or node.attrib.get("Name") or node.attrib.get("name")
        offset_text = (
            node.attrib.get("ADDRESS_OFFSET")
            or node.attrib.get("AddressOffset")
            or node.attrib.get("addressOffset")
            or node.attrib.get("OFFSET")
            or node.attrib.get("offset")
        )
        offset = None
        if offset_text:
            try:
                offset = int(offset_text, 0)
            except ValueError:
                offset = None
        for child in node:
            child_tag = child.tag.rsplit("}", 1)[-1].lower()
            text = (child.text or "").strip()
            if child_tag in {"name", "logicalname"} and text:
                name = text
            if child_tag in {"addressoffset", "offset"} and text:
                try:
                    offset = int(text, 0)
                except ValueError:
                    pass
            if child_tag == "property":
                prop_name = (child.attrib.get("NAME") or child.attrib.get("Name") or child.attrib.get("name") or "").lower()
                prop_value = child.attrib.get("VALUE") or child.attrib.get("Value") or child.attrib.get("value")
                if prop_name in {"address_offset", "addressoffset", "offset"} and prop_value is not None:
                    try:
                        offset = int(prop_value, 0)
                    except ValueError:
                        pass
        if name and offset is not None:
            regs[name] = offset
    return regs


def _find_ip(overlay: Any, requested: Optional[str]) -> Any:
    if requested:
        return getattr(overlay, requested)
    for name in getattr(overlay, "ip_dict", {}):
        if "hgtxr" in name.lower():
            return getattr(overlay, name.replace("/", "_"), None) or overlay.ip_dict[name]
    if getattr(overlay, "ip_dict", None):
        first = next(iter(overlay.ip_dict))
        return getattr(overlay, first.replace("/", "_"), None) or overlay.ip_dict[first]
    raise RuntimeError("No IP found in overlay")


@dataclass
class RuntimeConfig:
    bitfile: Path = DEFAULT_BIT
    hwhfile: Path = DEFAULT_HWH
    ip_name: Optional[str] = None
    timeout_s: float = 30.0
    poll_s: float = 0.001
    frac_bits: int = DATA_FRAC_BITS


class HgtxrOverlay:
    def __init__(self, config: RuntimeConfig = RuntimeConfig()):
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

    def __enter__(self) -> "HgtxrOverlay":
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

    def _alloc(self, values: Any, shape: Tuple[int, ...]) -> Any:
        arr = to_fixed16(values, self.config.frac_bits).reshape(shape)
        buf = allocate(shape=shape, dtype=np.int16)
        buf[:] = arr
        self.buffers.append(buf)
        return buf

    def _alloc_zero(self, shape: Tuple[int, ...], dtype: Any = np.int16) -> Any:
        buf = allocate(shape=shape, dtype=dtype)
        buf[:] = 0
        self.buffers.append(buf)
        return buf

    def _write_addr(self, name: str, addr: int) -> None:
        # Vitis HLS exports 64-bit pointer args as name_1/name_2 registers.
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

    def run(self, frame: Any, event_pos: Any, event_neg: Any, prev_state: Any) -> Tuple[np.ndarray, np.ndarray]:
        frame_buf = self._alloc(frame, FRAME_SHAPE)
        event_pos_buf = self._alloc(event_pos, FRAME_SHAPE)
        event_neg_buf = self._alloc(event_neg, FRAME_SHAPE)
        prev_state_buf = self._alloc(prev_state, STATE_SHAPE)
        out_state_buf = self._alloc_zero(STATE_SHAPE, np.int16)
        runtime_state_buf = self._alloc_zero((1,), np.int32)

        self._write_addr("frame", self._phys(frame_buf))
        self._write_addr("event_pos", self._phys(event_pos_buf))
        self._write_addr("event_neg", self._phys(event_neg_buf))
        self._write_addr("prev_state", self._phys(prev_state_buf))
        self._write_addr("out_state", self._phys(out_state_buf))
        self._write_addr("runtime_state", self._phys(runtime_state_buf))

        for buf in [frame_buf, event_pos_buf, event_neg_buf, prev_state_buf, out_state_buf, runtime_state_buf]:
            if hasattr(buf, "flush"):
                buf.flush()

        self._start()
        deadline = time.monotonic() + self.config.timeout_s
        while not self._done():
            if time.monotonic() > deadline:
                raise TimeoutError("HGTXR IP did not assert AP_DONE")
            time.sleep(self.config.poll_s)

        for buf in [out_state_buf, runtime_state_buf]:
            if hasattr(buf, "invalidate"):
                buf.invalidate()
        return from_fixed16(np.array(out_state_buf), self.config.frac_bits), np.array(runtime_state_buf, dtype=np.int32)
