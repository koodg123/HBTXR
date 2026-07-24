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
from quantization.fake_quant import AffineFakeQuantizer, LUTFakeQuantizer
from quantization.insert import QuantConfig, QuantLinear, collect_quantizers, insert_fake_quant
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
]
