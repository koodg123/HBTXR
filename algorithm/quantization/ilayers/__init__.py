"""ilayers — I-tier (integer) deployment modules for the HBTXR ViT.

HW-faithful integer graph: int8/uint8 storage, int32 accumulation, int64 requant
intermediate; verified bit-exact against the i_ops golden. ``ILayerNorm`` and
``ISoftmax`` are fully integer end to end (integer mean/variance/rsqrt, integer row
max/exp/row-sum/reciprocal) — unlike their Q-tier twins, which still reduce in float.

``quantization.convert.convert_model_to_integer`` installs these into a calibrated
model and reports what is still float.
"""
from quantization.ilayers.conv import IConv2d
from quantization.ilayers.int_functional import int_matmul, requant
from quantization.ilayers.layernorm import ILayerNorm
from quantization.ilayers.linear import ILinear, int_linear_forward, quantize_to_int, verify_int_linear
from quantization.ilayers.matmul import IMatMul
from quantization.ilayers.nonlinear import IGeLU
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.softmax import ISoftmax
from quantization.ilayers.tensor_ops import IAdd, ICat, IPool

__all__ = [
    "QTensor",
    "int_matmul",
    "requant",
    "quantize_to_int",
    "int_linear_forward",
    "verify_int_linear",
    "ILinear",
    "IConv2d",
    "IMatMul",
    "IGeLU",
    "ILayerNorm",
    "ISoftmax",
    "IAdd",
    "ICat",
    "IPool",
]
