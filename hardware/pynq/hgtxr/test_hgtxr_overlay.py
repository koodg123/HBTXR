import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_DIR.parent
HARDWARE_ROOT = PACKAGE_DIR.parents[1]
TOOLS_DIR = HARDWARE_ROOT / "tools"
for path in [PACKAGE_DIR, PACKAGE_PARENT, TOOLS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hgtxr_overlay import to_fixed16, from_fixed16, parse_registers_from_hwh

from hgtxr import e2e_axis_dma_overlay as axis_dma
from hgtxr import run_e2e_axis_dma_smoke as axis_smoke
from hgtxr import e2e_m_axi_overlay as e2e
from hgtxr import e2e_m_axi_weights as weights
from hgtxr import run_e2e_m_axi_smoke as smoke
import export_e2e_m_axi_weights as export_weights


class FakeBuffer:
    next_phys = 0x100000000

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype):
        self.array = np.zeros(shape, dtype=dtype)
        self.physical_address = FakeBuffer.next_phys
        FakeBuffer.next_phys += 0x10000
        self.flush_count = 0
        self.invalidate_count = 0
        self.free_count = 0

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        return np.array(self.array, dtype=dtype, copy=copy) if copy is not None else np.asarray(self.array, dtype=dtype)

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value) -> None:
        self.array[key] = value

    @property
    def size(self) -> int:
        return int(self.array.size)

    def flush(self) -> None:
        self.flush_count += 1

    def invalidate(self) -> None:
        self.invalidate_count += 1

    def free(self) -> None:
        self.free_count += 1

    @property
    def ctypes(self):
        return self.array.ctypes


class FakeMMIO:
    def __init__(self, auto_done: bool = True):
        self.auto_done = auto_done
        self.registers = {0x00: 0x80}
        self.writes: list[tuple[int, int]] = []

    def write(self, offset: int, value: int) -> None:
        self.writes.append((offset, value))
        self.registers[offset] = value
        if offset == 0x00 and self.auto_done and (value & 1):
            self.registers[0x00] = value | 0x2

    def read(self, offset: int) -> int:
        return self.registers.get(offset, 0)


class FakeIp:
    def __init__(self, mmio: FakeMMIO):
        self.mmio = mmio


class FakeOverlay:
    def __init__(self, _bitfile: str, mmio: FakeMMIO):
        self.ip_dict = {"hgtxr_e2e_m_axi_top_0": {}}
        self.hgtxr_e2e_m_axi_top_0 = FakeIp(mmio)


class FakeDmaChannel:
    def __init__(self, name: str):
        self.name = name
        self.transfers = []
        self.wait_count = 0

    def transfer(self, buf) -> None:
        self.transfers.append(buf)

    def wait(self) -> None:
        self.wait_count += 1


class FakeDma:
    def __init__(self, name: str):
        self.name = name
        self.sendchannel = FakeDmaChannel(f"{name}.send")
        self.recvchannel = FakeDmaChannel(f"{name}.recv")


class FakeAxisOverlay:
    def __init__(self, _bitfile: str, mmio: FakeMMIO):
        self.ip_dict = {"hgtxr_e2e_axis_top_0": {}, "axi_dma_in": {}, "axi_dma_out": {}}
        self.hgtxr_e2e_axis_top_0 = FakeIp(mmio)
        self.axi_dma_in = FakeDma("axi_dma_in")
        self.axi_dma_out = FakeDma("axi_dma_out")


class E2EMaxiOverlayRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeBuffer.next_phys = 0x100000000
        self.buffers: list[FakeBuffer] = []
        self.mmio = FakeMMIO()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.overlay_patcher = mock.patch.object(e2e, "Overlay", lambda bitfile: FakeOverlay(bitfile, self.mmio))
        self.allocate_patcher = mock.patch.object(e2e, "allocate", self._allocate)
        self.overlay_patcher.start()
        self.allocate_patcher.start()
        self.addCleanup(self.overlay_patcher.stop)
        self.addCleanup(self.allocate_patcher.stop)
        self.hwh = Path(self.tmpdir.name) / "sample.hwh"
        self.hwh.write_text(
            """
            <ROOT>
              <REGISTER NAME="frame_1"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x10"/></REGISTER>
              <REGISTER NAME="frame_2"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x14"/></REGISTER>
              <REGISTER NAME="weights_1"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x1c"/></REGISTER>
              <REGISTER NAME="weights_2"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x20"/></REGISTER>
              <REGISTER NAME="out_state_1"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x28"/></REGISTER>
              <REGISTER NAME="out_state_2"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x2c"/></REGISTER>
              <REGISTER NAME="runtime_state_1"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x34"/></REGISTER>
              <REGISTER NAME="runtime_state_2"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x38"/></REGISTER>
            </ROOT>
            """
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _allocate(self, shape, dtype):
        buf = FakeBuffer(tuple(shape), np.dtype(dtype))
        self.buffers.append(buf)
        return buf

    def _runtime(self) -> e2e.HgtxrE2EMaxiOverlay:
        config = e2e.E2EMaxiRuntimeConfig(
            bitfile=Path(self.tmpdir.name) / "fake.bit",
            hwhfile=self.hwh,
            timeout_s=0.001,
            poll_s=0.0,
        )
        return e2e.HgtxrE2EMaxiOverlay(config)

    def test_e2e_m_axi_alloc_weights_defaults_to_zero(self) -> None:
        runtime = self._runtime()
        with mock.patch.object(e2e, "E2E_WEIGHT_WORDS", 2):
            buf = runtime._alloc_weights(None)
        self.assertEqual(buf.array.dtype, np.dtype(np.uint32))
        self.assertEqual(buf.array.shape, (2 * e2e.U32_PER_256B_WORD,))
        self.assertTrue(np.all(buf.array == 0))
        self.assertIn(buf, runtime.buffers)

    def test_e2e_m_axi_alloc_weights_copies_partial_and_frees_on_oversize(self) -> None:
        runtime = self._runtime()
        with mock.patch.object(e2e, "E2E_WEIGHT_WORDS", 1), mock.patch.object(e2e, "U32_PER_256B_WORD", 4):
            buf = runtime._alloc_weights(np.array([1, 0xDEADBEEF], dtype=np.uint32))
            np.testing.assert_array_equal(buf.array, np.array([1, 0xDEADBEEF, 0, 0], dtype=np.uint32))
            with self.assertRaises(ValueError):
                runtime._alloc_weights(np.arange(5, dtype=np.uint32))
        self.assertEqual(self.buffers[-1].free_count, 1)
        self.assertNotIn(self.buffers[-1], runtime.buffers)

    def test_e2e_m_axi_write_addr_and_control_bits(self) -> None:
        runtime = self._runtime()
        runtime._write_addr("frame", 0x123456789ABCDEF0)
        self.assertIn((0x10, 0x9ABCDEF0), self.mmio.writes)
        self.assertIn((0x14, 0x12345678), self.mmio.writes)
        runtime._start()
        self.assertIn((0x00, 0x81), self.mmio.writes)
        self.assertTrue(runtime._done())

    def test_e2e_m_axi_run_writes_args_flushes_invalidates_and_closes(self) -> None:
        runtime = self._runtime()
        out, runtime_state = runtime.run(np.zeros(e2e.FRAME_SHAPE, dtype=np.float32), weights_u32=np.array([1], dtype=np.uint32))

        self.assertEqual(out.shape, (6,))
        self.assertEqual(runtime_state.shape, (1,))
        self.assertEqual(len(runtime.buffers), 4)
        frame_buf, weights_buf, out_buf, state_buf = runtime.buffers
        self.assertEqual(int(weights_buf.array[0]), 1)
        self.assertEqual([buf.flush_count for buf in runtime.buffers], [1, 1, 1, 1])
        self.assertEqual(out_buf.invalidate_count, 1)
        self.assertEqual(state_buf.invalidate_count, 1)
        expected_offsets = [0x10, 0x14, 0x1C, 0x20, 0x28, 0x2C, 0x34, 0x38]
        self.assertEqual([offset for offset, _value in self.mmio.writes[:8]], expected_offsets)
        self.assertEqual(self.mmio.writes[8], (0x00, 0x81))
        self.assertEqual(frame_buf.physical_address, 0x100000000)
        runtime.close()
        self.assertEqual([buf.free_count for buf in [frame_buf, weights_buf, out_buf, state_buf]], [1, 1, 1, 1])
        self.assertEqual(runtime.buffers, [])


class E2EAxisDmaOverlayRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeBuffer.next_phys = 0x200000000
        self.buffers: list[FakeBuffer] = []
        self.mmio = FakeMMIO()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.overlay_patcher = mock.patch.object(axis_dma, "Overlay", lambda bitfile: FakeAxisOverlay(bitfile, self.mmio))
        self.allocate_patcher = mock.patch.object(axis_dma, "allocate", self._allocate)
        self.overlay_patcher.start()
        self.allocate_patcher.start()
        self.addCleanup(self.overlay_patcher.stop)
        self.addCleanup(self.allocate_patcher.stop)
        self.hwh = Path(self.tmpdir.name) / "sample_axis.hwh"
        self.hwh.write_text(
            """
            <ROOT>
              <REGISTER NAME="weights_1"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x10"/></REGISTER>
              <REGISTER NAME="weights_2"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x14"/></REGISTER>
              <REGISTER NAME="num_pixels"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x1c"/></REGISTER>
              <REGISTER NAME="runtime_state_1"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x24"/></REGISTER>
              <REGISTER NAME="runtime_state_2"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x28"/></REGISTER>
            </ROOT>
            """
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _allocate(self, shape, dtype):
        buf = FakeBuffer(tuple(shape), np.dtype(dtype))
        self.buffers.append(buf)
        return buf

    def _runtime(self) -> axis_dma.HgtxrE2EAxisDmaOverlay:
        config = axis_dma.E2EAxisDmaRuntimeConfig(
            bitfile=Path(self.tmpdir.name) / "fake_axis.bit",
            hwhfile=self.hwh,
            timeout_s=0.001,
            poll_s=0.0,
        )
        return axis_dma.HgtxrE2EAxisDmaOverlay(config)

    def test_e2e_axis_dma_frame_pack_uses_raw_low_byte_contract(self) -> None:
        runtime = self._runtime()
        frame = np.zeros(axis_dma.FRAME_SHAPE, dtype=np.float32)
        frame[0, 0] = 1.0 / 128.0
        frame[0, 1] = 255.0 / 128.0
        buf = runtime._alloc_axis_frame(frame)
        self.assertEqual(buf.array.shape, (axis_dma.AXIS_FRAME_WORDS, axis_dma.AXIS_U32_PER_WORD))
        self.assertEqual(int(buf.array[0, 0]), 1)
        self.assertEqual(int(buf.array[1, 0]), 255)
        self.assertTrue(np.all(buf.array[:, 1:] == 0))

    def test_e2e_axis_dma_run_writes_args_and_drives_dma(self) -> None:
        runtime = self._runtime()
        out, runtime_state = runtime.run(np.zeros(axis_dma.FRAME_SHAPE, dtype=np.float32), weights_u32=np.array([1], dtype=np.uint32))

        self.assertEqual(out.shape, (6,))
        self.assertEqual(runtime_state.shape, (1,))
        self.assertEqual(len(runtime.buffers), 0)
        frame_buf = runtime.dma_in.sendchannel.transfers[0]
        out_buf = runtime.dma_out.recvchannel.transfers[0]
        self.assertEqual(runtime.dma_out.recvchannel.transfers, [out_buf])
        self.assertEqual(runtime.dma_in.sendchannel.transfers, [frame_buf])
        self.assertEqual(runtime.dma_in.sendchannel.wait_count, 1)
        self.assertEqual(runtime.dma_out.recvchannel.wait_count, 1)
        expected_offsets = [0x10, 0x14, 0x1C, 0x24, 0x28]
        self.assertEqual([offset for offset, _value in self.mmio.writes[:5]], expected_offsets)
        self.assertEqual(self.mmio.writes[5], (0x00, 0x81))
        self.assertEqual(frame_buf.free_count, 1)
        self.assertEqual(out_buf.free_count, 1)


class FakeAxisSmokeRuntime:
    calls: list[tuple[np.ndarray, np.ndarray | None, int | None]] = []
    runtime_state = 2
    output = np.zeros((6,), dtype=np.float32)

    def __init__(self, config: axis_dma.E2EAxisDmaRuntimeConfig):
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def run(self, frame, weights_u32=None, num_pixels_control=None):
        out, state, _metrics = self.run_profiled(frame, weights_u32=weights_u32, num_pixels_control=num_pixels_control)
        return out, state

    def run_profiled(self, frame, weights_u32=None, num_pixels_control=None):
        FakeAxisSmokeRuntime.calls.append((np.asarray(frame), weights_u32, num_pixels_control))
        metrics = {
            "latency_s": 0.001,
            "kernel_observed_s": 0.0008,
            "output_dma_wait_plus_compute_s": 0.0007,
            "input_dma_measured_bandwidth_Bps": 1.0,
            "aggregate_dma_effective_bandwidth_Bps": 1.0,
        }
        return FakeAxisSmokeRuntime.output.copy(), np.array([FakeAxisSmokeRuntime.runtime_state], dtype=np.int32), metrics


class E2EAxisDmaSmokeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeAxisSmokeRuntime.calls = []
        FakeAxisSmokeRuntime.runtime_state = 2
        FakeAxisSmokeRuntime.output = np.zeros((6,), dtype=np.float32)

    def test_axis_dma_smoke_live_mode_passes_single_live_weight_word(self) -> None:
        with mock.patch.object(axis_smoke, "HgtxrE2EAxisDmaOverlay", FakeAxisSmokeRuntime):
            result = axis_smoke.run_smoke(axis_smoke.parse_args(["--weights-mode", "live"]))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["runtime_state"], 2)
        self.assertEqual(int(FakeAxisSmokeRuntime.calls[0][1][0]), 1)

    def test_axis_dma_smoke_c3b_variant_selects_c3b_artifacts(self) -> None:
        with mock.patch.object(axis_smoke, "HgtxrE2EAxisDmaOverlay", FakeAxisSmokeRuntime):
            result = axis_smoke.run_smoke(axis_smoke.parse_args(["--variant", "c3b-mem16", "--weights-mode", "live"]))
        self.assertEqual(result["variant"], "c3b-mem16")
        self.assertTrue(result["bitfile"].endswith("hgtxr_e2e_axis_dma_c3b_mem16.bit"))
        self.assertTrue(result["hwhfile"].endswith("hgtxr_e2e_axis_dma_c3b_mem16.hwh"))

    def test_axis_dma_runtime_mode_search_uses_search_shape_and_control(self) -> None:
        FakeAxisSmokeRuntime.runtime_state = 0
        with mock.patch.object(axis_smoke, "HgtxrE2EAxisDmaOverlay", FakeAxisSmokeRuntime):
            result = axis_smoke.run_smoke(
                axis_smoke.parse_args(
                    [
                        "--variant",
                        "runtime-mode-par32",
                        "--weights-mode",
                        "zero",
                        "--mode-profile",
                        "search",
                    ]
                )
            )
        frame, weights_u32, control = FakeAxisSmokeRuntime.calls[0]
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["expected_runtime_state"], 0)
        self.assertEqual(result["mode_profile"], "search")
        self.assertEqual(frame.shape, (128, 128))
        self.assertIsNone(weights_u32)
        self.assertEqual(control, 128 * 128)

    def test_axis_dma_runtime_mode_track_uses_track_shape_and_control(self) -> None:
        FakeAxisSmokeRuntime.runtime_state = 1
        with mock.patch.object(axis_smoke, "HgtxrE2EAxisDmaOverlay", FakeAxisSmokeRuntime):
            result = axis_smoke.run_smoke(
                axis_smoke.parse_args(
                    [
                        "--variant",
                        "runtime-mode-par32",
                        "--weights-mode",
                        "zero",
                        "--mode-profile",
                        "track",
                    ]
                )
            )
        frame, weights_u32, control = FakeAxisSmokeRuntime.calls[0]
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["expected_runtime_state"], 1)
        self.assertEqual(result["mode_profile"], "track")
        self.assertEqual(frame.shape, (64, 64))
        self.assertIsNone(weights_u32)
        self.assertEqual(control, 1)

    def test_axis_dma_par32_rom_compute_variant_checks_search_expected_raw(self) -> None:
        FakeAxisSmokeRuntime.runtime_state = 0
        FakeAxisSmokeRuntime.output = np.array([-1169, -1169, -1169, -1169, -1169, -1133], dtype=np.float32) / 16.0
        with mock.patch.object(axis_smoke, "HgtxrE2EAxisDmaOverlay", FakeAxisSmokeRuntime):
            result = axis_smoke.run_smoke(
                axis_smoke.parse_args(
                    [
                        "--variant",
                        "par32-rom-compute-300",
                        "--weights-mode",
                        "zero",
                        "--mode-profile",
                        "search",
                        "--expect-runtime-state",
                        "0",
                        "--expect-out-raw",
                        "-1169",
                        "-1169",
                        "-1169",
                        "-1169",
                        "-1169",
                        "-1133",
                    ]
                )
            )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["variant"], "par32-rom-compute-300")
        self.assertTrue(
            result["bitfile"].endswith(
                "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
                "corefabric_normuram_compute_300_mem16.bit"
            )
        )
        self.assertEqual(result["out_raw"], [-1169, -1169, -1169, -1169, -1169, -1133])

    def test_axis_dma_smoke_file_mode_loads_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "weights.bin"
            np.array([1, 2, 3], dtype=np.uint32).tofile(bin_path)
            loaded = axis_smoke.make_weights("file", bin_path)
        np.testing.assert_array_equal(loaded, np.array([1, 2, 3], dtype=np.uint32))


class FakeSmokeRuntime:
    calls: list[tuple[np.ndarray, np.ndarray | None]] = []
    runtime_state = 2

    def __init__(self, config: e2e.E2EMaxiRuntimeConfig):
        self.config = config
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        self.closed = True
        return False

    def run(self, frame, weights_u32=None):
        FakeSmokeRuntime.calls.append((np.asarray(frame), weights_u32))
        return np.zeros((6,), dtype=np.float32), np.array([FakeSmokeRuntime.runtime_state], dtype=np.int32)


class E2EMaxiSmokeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSmokeRuntime.calls = []
        FakeSmokeRuntime.runtime_state = 2

    def test_smoke_defaults_use_live_weight_runtime_state_2(self) -> None:
        args = smoke.parse_args([])
        self.assertEqual(args.weights_mode, "live")
        self.assertEqual(smoke.default_expected_runtime_state(args.weights_mode), 2)

    def test_smoke_live_mode_passes_single_live_weight_word(self) -> None:
        with mock.patch.object(smoke, "HgtxrE2EMaxiOverlay", FakeSmokeRuntime):
            result = smoke.run_smoke(smoke.parse_args(["--weights-mode", "live"]))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["runtime_state"], 2)
        self.assertEqual(int(FakeSmokeRuntime.calls[0][1][0]), 1)

    def test_smoke_zero_mode_uses_default_weight_allocation_contract(self) -> None:
        FakeSmokeRuntime.runtime_state = 1
        with mock.patch.object(smoke, "HgtxrE2EMaxiOverlay", FakeSmokeRuntime):
            result = smoke.run_smoke(smoke.parse_args(["--weights-mode", "zero", "--frame-pattern", "zero"]))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["expected_runtime_state"], 1)
        self.assertIsNone(FakeSmokeRuntime.calls[0][1])
        self.assertTrue(np.all(FakeSmokeRuntime.calls[0][0] == 0.0))

    def test_smoke_main_writes_json_and_returns_failure_on_runtime_mismatch(self) -> None:
        FakeSmokeRuntime.runtime_state = 1
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "nested" / "smoke.json"
            with mock.patch.object(smoke, "HgtxrE2EMaxiOverlay", FakeSmokeRuntime):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = smoke.main(["--weights-mode", "live", "--json-out", str(json_out)])
            payload = json.loads(json_out.read_text())
        self.assertEqual(rc, 1)
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["expected_runtime_state"], 2)
        self.assertEqual(payload["runtime_state"], 1)

    def test_smoke_golden_mode_checks_expected_output_raw(self) -> None:
        class GoldenRuntime(FakeSmokeRuntime):
            def run(self, frame, weights_u32=None):
                FakeSmokeRuntime.calls.append((np.asarray(frame), weights_u32))
                out = np.asarray(weights.EXPECTED_RAW, dtype=np.float32) / 16.0
                return out, np.array([weights.EXPECTED_RUNTIME_STATE], dtype=np.int32)

        with mock.patch.object(smoke, "HgtxrE2EMaxiOverlay", GoldenRuntime):
            result = smoke.run_smoke(smoke.parse_args(["--weights-mode", "golden"]))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["out_raw"], weights.EXPECTED_RAW)
        self.assertEqual(result["expected_out_raw"], weights.EXPECTED_RAW)
        self.assertEqual(int(FakeSmokeRuntime.calls[0][1][0] & 0xF), 1)

    def test_smoke_file_mode_loads_exported_u32_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "weights.bin"
            np.array([1, 2, 3], dtype=np.uint32).tofile(bin_path)
            loaded = smoke.make_weights("file", bin_path)
        np.testing.assert_array_equal(loaded, np.array([1, 2, 3], dtype=np.uint32))


class E2EMaxiWeightBuilderTests(unittest.TestCase):
    def test_q4_packing_lanes_and_signed_decode(self) -> None:
        buf = np.zeros((16,), dtype=np.uint32)
        weights.set_q4_weight(buf, 0, -1)
        weights.set_q4_weight(buf, 7, -8)
        weights.set_q4_weight(buf, 8, 7)
        self.assertEqual(int(buf[0]), 0x8000000F)
        self.assertEqual(int(buf[1]), 0x00000007)
        self.assertEqual(weights.get_q4_weight_raw(buf, 0), -1)
        self.assertEqual(weights.get_q4_weight_raw(buf, 7), -8)
        self.assertEqual(weights.get_q4_weight_raw(buf, 8), 7)

    def test_active196_b6_ff768_weights_match_known_offsets(self) -> None:
        buf = weights.build_active196_b6_ff768_weights()
        self.assertEqual(buf.dtype, np.dtype(np.uint32))
        self.assertEqual(buf.size, weights.E2E_WEIGHT_WORDS_256B * weights.U32_PER_256B_WORD)
        self.assertEqual(weights.get_q4_weight_raw(buf, weights.PATCH_WEIGHT_ELEM_BASE), 1)
        self.assertEqual(
            weights.get_q4_weight_raw(buf, weights.PATCH_WEIGHT_ELEM_BASE + 2 * weights.PATCH_ELEMS),
            -1,
        )
        self.assertEqual(
            weights.get_q4_weight_raw(buf, weights.HEAD_WEIGHT_ELEM_BASE + 1 * weights.EMBED + 1),
            -8,
        )
        block_base = weights.BLOCK_WEIGHT_ELEM_BASE
        self.assertEqual(weights.get_q4_weight_raw(buf, block_base + weights.BLOCK_WQ), 7)
        self.assertEqual(weights.get_q4_weight_raw(buf, block_base + weights.BLOCK_LN1_BETA), 7)
        self.assertEqual(int(buf[0] & 1), 1)
        tail_start = weights.REQUIRED_WEIGHT_WORDS_256B * weights.U32_PER_256B_WORD
        self.assertTrue(np.all(buf[tail_start:] == 0))


class E2EMaxiWeightExportTests(unittest.TestCase):
    def test_export_weights_writes_binary_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "e2e_weights"
            bin_path, manifest_path = export_weights.export_weights(prefix)
            payload = json.loads(manifest_path.read_text())
            arr = np.fromfile(bin_path, dtype=np.uint32)

        self.assertEqual(payload["format"], "raw-little-endian-uint32")
        self.assertEqual(payload["dtype"], "uint32")
        self.assertEqual(payload["shape"], [weights.E2E_WEIGHT_WORDS_256B * weights.U32_PER_256B_WORD])
        self.assertEqual(payload["expected_runtime_state"], weights.EXPECTED_RUNTIME_STATE)
        self.assertEqual(payload["expected_raw"], weights.EXPECTED_RAW)
        self.assertEqual(arr.size, weights.E2E_WEIGHT_WORDS_256B * weights.U32_PER_256B_WORD)
        self.assertEqual(weights.get_q4_weight_raw(arr, weights.PATCH_WEIGHT_ELEM_BASE), 1)
        self.assertEqual(payload["known_offset_checks"]["patch_channel2_elem0"], -1)


class HgtxrOverlayHelperTests(unittest.TestCase):
    def test_fixed16_round_trip(self) -> None:
        x = np.array([0.0, 0.5, -0.5, 1.25], dtype=np.float32)
        y = from_fixed16(to_fixed16(x), 10)
        np.testing.assert_allclose(y, x, atol=1.0 / 1024.0)

    def test_parse_registers_from_hwh_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hwh = Path(tmp) / "sample.hwh"
            hwh.write_text("<root><register><name>frame</name><addressOffset>0x10</addressOffset></register></root>")
            self.assertEqual(parse_registers_from_hwh(hwh)["frame"], 0x10)

    def test_parse_registers_from_vivado_hwh_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hwh = Path(tmp) / "sample_attr.hwh"
            hwh.write_text(
                '<ROOT><REGISTER NAME="frame_1"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="16"/></REGISTER></ROOT>'
            )
            self.assertEqual(parse_registers_from_hwh(hwh)["frame_1"], 0x10)


if __name__ == "__main__":
    unittest.main()
