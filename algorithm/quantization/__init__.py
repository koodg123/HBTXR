"""quantization — HW-friendly (HG-PIPE style) quantization for the HBTXR ViT models.

Top-level flat package (run from algorithm/). Two tiers over the shared calibration
primitives:

- **Q tier** (qlayers): fake-quant modules with float I/O, STE-trainable (QAT/PTQ) —
  QLinear, QGeLU, QLayerNorm, QSoftmax.
- **I tier** (ilayers): HW-faithful integer deployment kernels, verified bit-exact
  against the i_ops golden.

Shared: scheme (dtype/range), observer + lut_calibrate (calibration), q_ops
(fake-quant primitives), i_ops (integer golden). convert wires float->Q; calibrate /
qat drive PTQ / QAT.
"""
from quantization.calibrate import calibrate, post_training_quantize
from quantization.convert import (
    calibrate_gelu_luts,
    calibrate_layernorm_luts,
    calibrate_softmax_luts,
    collect_quantizers,
    insert_fake_quant,
)
from quantization.ilayers.linear import int_linear_forward, verify_int_linear
from quantization.lut_calibrate import build_function_lut, build_gelu_lut
from quantization.q_ops import AffineFakeQuantizer, LUTFakeQuantizer
from quantization.qat import prepare_qat
from quantization.qlayers.linear import QLinear, QuantConfig
from quantization.qlayers.nonlinear import QGeLU, QLayerNorm, QSoftmax
from quantization.scheme import INT4, INT8, UINT8, QuantDtype, qrange, quantize_clamp

__all__ = [
    # scheme
    "QuantDtype",
    "INT8",
    "UINT8",
    "INT4",
    "qrange",
    "quantize_clamp",
    # q primitives / layers
    "AffineFakeQuantizer",
    "LUTFakeQuantizer",
    "QuantConfig",
    "QLinear",
    "QGeLU",
    "QLayerNorm",
    "QSoftmax",
    # convert / calibrate / qat
    "insert_fake_quant",
    "collect_quantizers",
    "calibrate_gelu_luts",
    "calibrate_layernorm_luts",
    "calibrate_softmax_luts",
    "calibrate",
    "post_training_quantize",
    "prepare_qat",
    # calibration builders
    "build_function_lut",
    "build_gelu_lut",
    # i tier
    "int_linear_forward",
    "verify_int_linear",
]
