"""qlayers — Q-tier (fake-quant) layer modules for the HBTXR ViT.

Float I/O, STE-trainable (QAT). More module types (conv, matmul, tensor_ops) are
added with the integer graph phases.
"""
from quantization.qlayers.linear import QLinear, QuantConfig
from quantization.qlayers.nonlinear import QGeLU, QLayerNorm, QSoftmax

__all__ = ["QuantConfig", "QLinear", "QGeLU", "QLayerNorm", "QSoftmax"]
