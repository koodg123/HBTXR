"""quantization — HW-friendly (HG-PIPE style) quantization for the HBTXR ViT models.

Top-level flat package (run from algorithm/). Two tiers over the shared calibration
primitives:

- **Q tier** (qlayers): fake-quant modules with float I/O, STE-trainable (QAT/PTQ) —
  QLinear, QGeLU, QLayerNorm, QSoftmax. The nonlinears still reduce in float.
- **I tier** (ilayers): HW-faithful integer deployment kernels, verified bit-exact
  against the i_ops golden — ILinear, IConv2d, IGeLU, ILayerNorm, ISoftmax, plus the
  scale-aligning IMatMul / IAdd / ICat / IPool.

Shared: scheme (dtype/range), observer + lut_calibrate + int_calibrate(_softmax)
(calibration), q_ops (fake-quant primitives), i_ops (integer golden). convert wires
float->Q and Q->I (``convert_model_to_integer`` for the whole graph); calibrate / qat
drive PTQ / QAT.
"""
from quantization.calibrate import calibrate, post_training_quantize
from quantization.convert import (
    ConversionReport,
    calibrate_gelu_luts,
    calibrate_int_convs,
    calibrate_int_gelus,
    calibrate_int_layernorms,
    calibrate_int_softmaxes,
    calibrate_layernorm_luts,
    calibrate_softmax_luts,
    collect_quantizers,
    convert_model_to_integer,
    convert_to_integer,
    float_modules,
    insert_fake_quant,
)
from quantization.i_ops import layernorm_quantize, softmax_quantize, table_quantize
from quantization.ilayers import (
    IAdd,
    ICat,
    IConv2d,
    IGeLU,
    ILayerNorm,
    ILinear,
    IMatMul,
    IPool,
    ISoftmax,
    QTensor,
)
from quantization.ilayers.linear import int_linear_forward, verify_int_linear
from quantization.int_calibrate import calibrate_int_layernorm
from quantization.int_calibrate_softmax import (
    build_softmax_int_payload,
    build_softmax_int_payload_from_qsoftmax,
)
from quantization.lut_calibrate import build_function_lut, build_gelu_lut
from quantization.observer import AffineObserver, MinMaxObserver, build_observer
from quantization.q_ops import AffineFakeQuantizer, LUTFakeQuantizer
from quantization.qat import prepare_qat
from quantization.qlayers.linear import QLinear, QuantConfig
from quantization.qlayers.nonlinear import QGeLU, QLayerNorm, QSoftmax
from quantization.scheme import INT4, INT8, UINT8, QuantDtype, qrange, quantize_clamp
from quantization.spec import CalibrationSpec, QuantScheme, TensorQuantSpec, apply_scale_type

__all__ = [
    # scheme
    "QuantDtype",
    "INT8",
    "UINT8",
    "INT4",
    "qrange",
    "quantize_clamp",
    # configurable spec
    "TensorQuantSpec",
    "QuantScheme",
    "CalibrationSpec",
    "apply_scale_type",
    # observers
    "MinMaxObserver",
    "AffineObserver",
    "build_observer",
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
    "convert_to_integer",
    "convert_model_to_integer",
    "ConversionReport",
    "float_modules",
    "calibrate_gelu_luts",
    "calibrate_layernorm_luts",
    "calibrate_softmax_luts",
    "calibrate_int_convs",
    "calibrate_int_gelus",
    "calibrate_int_layernorms",
    "calibrate_int_softmaxes",
    "calibrate",
    "post_training_quantize",
    "prepare_qat",
    # calibration builders
    "build_function_lut",
    "build_gelu_lut",
    "calibrate_int_layernorm",
    "build_softmax_int_payload",
    "build_softmax_int_payload_from_qsoftmax",
    # i tier
    "QTensor",
    "ILinear",
    "IConv2d",
    "IMatMul",
    "IGeLU",
    "ILayerNorm",
    "ISoftmax",
    "IAdd",
    "ICat",
    "IPool",
    "int_linear_forward",
    "verify_int_linear",
    # integer golden kernels
    "layernorm_quantize",
    "softmax_quantize",
    "table_quantize",
]
