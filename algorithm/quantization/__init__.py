"""quantization — HW-friendly (HG-PIPE style) quantization for the HBTXR ViT models.

Top-level flat package (run from algorithm/). Provides the quantization primitives
adapted from the HG-PIPE reference (references/) and applied to the reimplemented
HBTXR ViT (models.backbones.vit / models.blocks / models.heads):

- scheme: integer dtype / range / clamp semantics.
- fake_quant: QAT-ready FakeQuantizer modules (affine with STE, LUT nonlinear).
- int_ops: bit-exact integer reference kernels (table / layernorm / softmax) for
  integer inference and verification.

Later phases add insertion into HBTXR models, calibration + PTQ, QAT, and integer
inference (see the phased plan).
"""
from quantization.calibrate import calibrate, post_training_quantize
from quantization.fake_quant import AffineFakeQuantizer, LUTFakeQuantizer
from quantization.insert import QuantConfig, QuantLinear, collect_quantizers, insert_fake_quant
from quantization.lut_calibrate import build_function_lut, build_gelu_lut
from quantization.nonlinear import (
    GeLULUT,
    LayerNormLUT,
    SoftmaxLUT,
    calibrate_gelu_luts,
    calibrate_layernorm_luts,
    calibrate_softmax_luts,
)
from quantization.qat import prepare_qat
from quantization.scheme import INT4, INT8, UINT8, QuantDtype, qrange, quantize_clamp

__all__ = [
    "QuantDtype",
    "INT8",
    "UINT8",
    "INT4",
    "qrange",
    "quantize_clamp",
    "AffineFakeQuantizer",
    "LUTFakeQuantizer",
    "QuantConfig",
    "QuantLinear",
    "insert_fake_quant",
    "collect_quantizers",
    "calibrate",
    "post_training_quantize",
    "prepare_qat",
    "build_function_lut",
    "build_gelu_lut",
    "GeLULUT",
    "calibrate_gelu_luts",
    "LayerNormLUT",
    "calibrate_layernorm_luts",
    "SoftmaxLUT",
    "calibrate_softmax_luts",
]
