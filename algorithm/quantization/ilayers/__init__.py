"""ilayers — I-tier (integer) deployment modules for the HBTXR ViT.

HW-faithful integer graph: int8/uint8 storage, int32 accumulation, int64 requant
intermediate; verified bit-exact against the i_ops golden. More modules (conv,
matmul, tensor_ops, vit) land with the integer-graph phases.
"""
from quantization.ilayers.int_functional import int_matmul, requant
from quantization.ilayers.linear import int_linear_forward, quantize_to_int, verify_int_linear
from quantization.ilayers.qtensor import QTensor

__all__ = [
    "QTensor",
    "int_matmul",
    "requant",
    "quantize_to_int",
    "int_linear_forward",
    "verify_int_linear",
]
