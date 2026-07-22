"""models.blocks — ViT transformer-block primitives (paper Fig 3).

The shared low-level building blocks used by the HBTXR backbone: multi-head
attention, the position-wise MLP, and the pre-norm residual Transformer Block.
"""
from models.blocks.attention import MultiHeadAttention
from models.blocks.mlp import Mlp
from models.blocks.transformer_block import TransformerBlock

__all__ = ["MultiHeadAttention", "Mlp", "TransformerBlock"]
