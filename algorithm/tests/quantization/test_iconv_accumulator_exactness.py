"""B6: the conv accumulator must be exact at the tap counts this model actually has.

``IConv2d`` used to run its integer conv through ``F.conv2d`` on operands cast to float,
which is exact only while the accumulator stays inside float32's 2^24 integer range. The
shipped mask head is already outside it: ``Conv2d(192, 96, k=3)`` has 1728 taps and a
worst-case accumulator of 27,870,912.

The reason this went unnoticed is the reason it needs a deliberate test. Random
activations and weights produce an accumulator that grows like sqrt(taps), landing three
orders of magnitude below the worst case, so every existing random-input conv test passed
with zero error while a saturated input is off by an LSB. These tests drive the operands
to the corner on purpose and check against the arbitrary-precision ``i_ops`` golden.

The fix is an int64 im2col matmul rather than a cleverer float schedule: an int8xint8 MAC
array is integer hardware, and a kernel that reaches the right answer via a float unit can
be exact and still not be a model of the thing.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F

from quantization.i_ops import int_conv2d
from quantization.ilayers.conv import FLOAT32_EXACT_INT, IConv2d, worst_case_accumulator
from quantization.scheme import INT8


def _saturated(cin: int, k: int, cout: int = 2):
    """Operands at the int8 corner, with one tap knocked off the maximum.

    All-127 x all-127 gives an accumulator that happens to be even and therefore still
    representable at this magnitude; moving a single weight to 126 makes the exact result
    odd, which is precisely what float32 cannot hold above 2^24. The bug is an off-by-one
    in the last bit, so the input has to be built to make that bit observable.
    """
    x = torch.full((1, cin, 4, 4), 127, dtype=torch.int64)
    w = torch.full((cout, cin, k, k), 127, dtype=torch.int64)
    w[0, 0, 0, 0] = 126
    return x, w


def _iconv(w: torch.Tensor, *, act_scale: float = 1.0) -> IConv2d:
    scale = torch.ones(w.shape[0])
    return IConv2d(w.to(torch.int32), scale, act_scale, INT8, None, stride=1, padding=0)


# --- the premise: float32 really does lose the bit at the shipped tap count ----

def test_a_float_accumulator_is_inexact_at_the_mask_heads_tap_count():
    """Without this, the kernel below would be defending against nothing."""
    cin, k = 192, 3                                  # the shipped mask-head projection
    x, w = _saturated(cin, k)
    assert worst_case_accumulator(w, INT8) > FLOAT32_EXACT_INT
    naive = F.conv2d(x.float(), w.float(), stride=1).round().to(torch.int64)
    exact = F.conv2d(x.double(), w.double(), stride=1).round().to(torch.int64)
    assert int((naive - exact).abs().max()) == 1, "float32 was expected to drop a bit here"


def test_random_data_never_reaches_the_corner_that_breaks_it():
    """Why the old kernel passed every test it had: the bug is unreachable by accident."""
    gen = torch.Generator().manual_seed(3)
    x = torch.randint(-128, 128, (1, 192, 4, 4), generator=gen, dtype=torch.int64)
    w = torch.randint(-128, 128, (2, 192, 3, 3), generator=gen, dtype=torch.int64)
    reached = int(F.conv2d(x.double(), w.double(), stride=1).abs().max())
    assert reached < FLOAT32_EXACT_INT / 10, (
        f"random data reached {reached}, close enough to the limit to trip the old kernel "
        "by luck — which would make every random-input conv test an accidental guard")


# --- the kernel ---------------------------------------------------------------

def test_the_conv_matches_the_arbitrary_precision_golden_where_float32_cannot():
    """Element-wise ``==`` against ``i_ops.int_conv2d``, on the input that breaks float32."""
    x, w = _saturated(192, 3)
    acc = _iconv(w).accumulate(x)
    gold = int_conv2d(x[0].tolist(), w.tolist(), stride=1, padding=0)
    assert acc[0].tolist() == gold


@pytest.mark.parametrize("cin,k,stride", [
    (192, 3, 1),      # mask head projection: past the float limit
    (96, 3, 1),       # its 96-channel half: under it
    (1, 16, 16),      # Conv-F patch embed (non-overlapping)
    (2, 16, 16),      # Conv-E patch embed
    (8, 3, 2),        # overlapping strided
])
def test_the_kernel_is_exact_across_the_shapes_this_graph_contains(cin, k, stride):
    """One path for every conv, so the tap count stops being a correctness variable."""
    gen = torch.Generator().manual_seed(cin * 100 + k)
    x = torch.randint(-128, 128, (1, cin, 16, 16), generator=gen, dtype=torch.int64)
    w = torch.randint(-128, 128, (3, cin, k, k), generator=gen, dtype=torch.int64)
    conv = IConv2d(w.to(torch.int32), torch.ones(3), 1.0, INT8, None, stride=stride,
                   padding=0)
    gold = int_conv2d(x[0].tolist(), w.tolist(), stride=stride, padding=0)
    assert conv.accumulate(x)[0].tolist() == gold


def test_the_accumulation_never_creates_a_float_tensor():
    """The property, checked mechanically rather than by reading the source."""
    from torch.overrides import TorchFunctionMode

    offenders: list[str] = []

    class Watch(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                offenders.append(f"{getattr(func, '__name__', func)} -> {out.dtype}")
            return out

    x, w = _saturated(8, 3)
    conv = _iconv(w)                 # built outside: setup holds float SCALES by design
    with torch.no_grad(), Watch():
        conv.accumulate(x)
    assert not offenders, f"float tensors inside the integer conv: {offenders[:6]}"


def test_the_float_io_port_still_reproduces_the_fake_quant_conv():
    """``forward`` keeps its float I/O; only what happens between the ports changed."""
    from torch import nn

    torch.manual_seed(5)
    conv = nn.Conv2d(6, 4, kernel_size=3, padding=1)
    iconv = IConv2d.from_conv(conv, act_scale=0.02)
    x = torch.rand(2, 6, 9, 9) * 0.5
    got = iconv(x)
    x_int = torch.round(x / 0.02).clamp(-128, 127)
    want = F.conv2d(x_int.double() * 0.02, (iconv.weight_int.double()
                                            * iconv.weight_scale.view(-1, 1, 1, 1).double()),
                    bias=conv.bias.double(), padding=1)
    assert torch.allclose(got.double(), want, atol=1e-5)


def test_the_live_mask_head_conv_is_the_one_that_needed_this():
    """Tie the bound to the real model rather than to a number typed into a test."""
    from engine.model_factory import make_model

    model = make_model({"target": "models.frame.FrameModel"})
    proj = model.mask_head.proj
    conv = IConv2d.from_conv(proj, act_scale=0.01)
    taps = proj.in_channels * proj.kernel_size[0] * proj.kernel_size[1]
    assert taps * 127 * 127 > FLOAT32_EXACT_INT, f"{taps} taps no longer exceed the limit"
    x_int = torch.randint(-128, 128, (1, proj.in_channels, 5, 5), dtype=torch.int64,
                          generator=torch.Generator().manual_seed(1))
    padded = F.pad(x_int, (1, 1, 1, 1))
    gold = int_conv2d(padded[0].tolist(), conv.weight_int.to(torch.int64).tolist(),
                      stride=1, padding=0)
    assert conv.accumulate(padded)[0].tolist() == gold
