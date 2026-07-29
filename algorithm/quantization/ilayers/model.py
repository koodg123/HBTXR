"""The whole detector as one integer graph — ``models/frame/model.py``'s counterpart.

``ITransformerBlock`` removed the float transport from inside a block, but
``DirectPupilDetector`` composes the stem, the backbone norm and the heads *outside* the
block, and those edges were still float32. This assembles the lot:

    image (int, on the stem's grid)
        -> IPatchEmbed   conv, per-out-channel requant, flatten to tokens
        -> IViTBackbone  block stack (bridged per edge) + final ILayerNorm
        -> IPooledMlpHead / IEllipseHead   integer mean, Linear, GeLU table, Linear
        -> QTensor(int32 accumulator, per-out-channel scale)

Between those there is one boundary in each direction and no other float:

- **the input** — the host quantizes the image once, on the stem conv's calibrated
  activation grid, and hands over integers. ``IPatchEmbed`` refuses a QTensor that is on
  any other grid (or carries a different zero-point) rather than bridging it, because a
  bridge computed per call is the per-call float ratio this whole rewrite removes.
- **the output** — one ``dequantize()`` per head. The heads end on their accumulator, so
  what crosses the boundary is int32 plus a per-out-channel scale, and the host does the
  multiply. Anything the model applies *after* that (``box_to_state``'s geometry, the
  reliability sigmoid) is host arithmetic on a real-valued answer and is left where it is.

What is NOT threaded, and why: the mask head. Its conv stack would lower the same way,
but it ends in ``F.interpolate(mode="bilinear")``, and an integer bilinear resample is an
unmade design decision (nearest? fixed-point weights at what fractional width?). It stays
on the float-I/O path, and ``float_io_heads`` names it so nobody has to infer it from the
absence of a class.
"""
from __future__ import annotations

import torch
from torch import nn

from quantization.ilayers.heads import IEllipseHead, IPooledMlpHead
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.vit import IPatchEmbed, IViTBackbone
from quantization.scheme import INT8, QuantDtype

# Heads whose bodies are not integer-threaded, with the reason, so the gap is stated
# rather than implied. Read by ``IDirectPupilDetector.float_io_heads``.
FLOAT_IO_HEADS = {
    "mask_head": "ends in F.interpolate(bilinear); the integer resample is undecided",
}


def _build_head(head: nn.Module, *, input_scale: float, dtype: QuantDtype):
    """Pick the integer head matching a converted float head, or say it is unsupported."""
    if hasattr(head, "condition_on_state"):
        return IEllipseHead(head, input_scale=input_scale, dtype=dtype)
    if hasattr(head, "regressor") or hasattr(head, "estimator"):
        return IPooledMlpHead(head, input_scale=input_scale, dtype=dtype)
    raise ValueError(
        f"{type(head).__name__} is not a pooled regression head; only the heads built "
        "from models.heads.common.mlp_head are integer-threaded")


class IDirectPupilDetector(nn.Module):
    """Stem -> backbone -> pooled heads, integer from the input port to one dequant."""

    def __init__(self, model: nn.Module, *, depth_limit: int | None = None,
                 dtype: QuantDtype = INT8) -> None:
        super().__init__()
        self.backbone = IViTBackbone(model.backbone, depth_limit=depth_limit, dtype=dtype)
        self.patch_embed = IPatchEmbed(model.patch_embed, self.backbone.input_scale,
                                       dtype=dtype)
        feature_scale = self.backbone.output_scale
        self.head = _build_head(model.head, input_scale=feature_scale, dtype=dtype)
        self.roi_head = (None if getattr(model, "roi_head", None) is None
                         else _build_head(model.roi_head, input_scale=feature_scale, dtype=dtype))
        self.reliability_head = (
            None if getattr(model, "reliability_head", None) is None
            else _build_head(model.reliability_head, input_scale=feature_scale, dtype=dtype))
        self.dtype = dtype

    # --- the two boundaries ---------------------------------------------------

    @property
    def input_scale(self) -> float:
        return self.patch_embed.input_scale

    @property
    def input_zero_point(self) -> int:
        return self.patch_embed.input_zero_point

    def quantize_input(self, image: torch.Tensor) -> QTensor:
        """The host's side of the input port: one quantization of the whole image."""
        return QTensor.quantize(image, self.input_scale, self.input_zero_point, self.dtype)

    @property
    def float_io_heads(self) -> dict[str, str]:
        """Heads of the source model this graph does not carry, and why."""
        return dict(FLOAT_IO_HEADS)

    # --- the graph ------------------------------------------------------------

    def forward(self, image_int: QTensor, *,
                anchor_state: QTensor | None = None) -> dict[str, QTensor]:
        tokens, _grid_hw = self.patch_embed(image_int)
        feats = self.backbone(tokens)
        out = {"head": self.head(feats, anchor_state)
               if isinstance(self.head, IEllipseHead) else self.head(feats)}
        if self.roi_head is not None:
            out["eye_box"] = self.roi_head(feats)
        if self.reliability_head is not None:
            out["reliability"] = self.reliability_head(feats)
        return out

    def forward_real(self, image: torch.Tensor, *,
                     anchor_state: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Convenience: quantize in, run, dequantize out — the host's view of the chip.

        The reliability sigmoid lives here rather than in the graph: it is a monotone
        squash of an already-final accumulator, so it changes nothing about what the
        accelerator computes.
        """
        state = None
        if anchor_state is not None:
            state = QTensor.quantize(anchor_state, self.head.mlp.input_scale, 0.0, self.dtype)
        raw = self.forward(self.quantize_input(image), anchor_state=state)
        out = {name: qt.dequantize() for name, qt in raw.items()}
        if "reliability" in out:
            out["reliability"] = torch.sigmoid(out["reliability"])
        return out


__all__ = ["IDirectPupilDetector", "FLOAT_IO_HEADS"]
