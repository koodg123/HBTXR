"""R1: the claims the user-facing docs make, checked against what the code does.

Every statement pinned here was, at some point, written down and then quietly falsified by
a later phase. `frame_quant.yaml` told users the attention matmuls and residual adds stay
float (D4 converted them), that tensors move between modules as float (A1 closed that),
and that `mask_head.proj` is left float and appears in `manifest['unexported']` (D2 made
padded convs convertible; `unexported` is empty). The Part C/D report still listed
`total_params` reading 0 after `export.py` had been fixed to count buffers.

That is the same defect class D0.3 found and fixed once already, which is why it gets a
test this time rather than another round of edits. A doc claim nobody can falsify
automatically is a claim that will be false again within two phases.

Two kinds of check here, and both are needed:

- **the facts**, so the doc's replacement text is anchored to measurements;
- **the retired phrasings**, so a plausible-sounding sentence that happens to be false
  cannot be reintroduced into the file a user reads first.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import nn

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiment" / "frame_quant.yaml"
ENTRYPOINT = ROOT / "quantization" / "entrypoint.py"


@pytest.fixture(scope="module")
def converted():
    from engine.model_factory import make_model
    from quantization.calibrate import post_training_quantize
    from quantization.convert import convert_model_to_integer

    torch.manual_seed(0)
    model = make_model({"target": "models.frame.FrameModel", "embed_dim": 48,
                        "patch_size": 16,
                        "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0,
                                     "cut_point": 1}})
    calib = [torch.rand(2, 1, 64, 64, generator=torch.Generator().manual_seed(7))]
    model, _ = post_training_quantize(model, calib)
    model, report = convert_model_to_integer(model, calib)
    return model.eval(), report


# --- the facts the replacement text asserts ----------------------------------

def test_the_attention_matmuls_and_residual_adds_do_convert(converted):
    """Retired claim: "the attention matmuls / residual adds stay float"."""
    from quantization.ilayers.matmul import IMatMul
    from quantization.ilayers.tensor_ops import IAdd

    model, _ = converted
    matmuls = [m for m in model.modules() if isinstance(m, IMatMul)]
    adds = [m for m in model.modules() if isinstance(m, IAdd)]
    assert len(matmuls) == 4 and len(adds) == 4, "two blocks x (2 matmuls, 2 adds)"


def test_only_inert_and_exact_constant_modules_are_left_float(converted):
    """What "left_float" actually contains, so the doc can name it exactly."""
    from models.blocks.seams import Scale

    model, report = converted
    kinds = {type(m).__name__ for m in report.left_float.values()}
    assert kinds == {"Dropout", "Scale"}, kinds
    # Dropout is an inference no-op; Scale is an exact constant multiply (2^-3 at the
    # shipped head_dim) that folds into the following requant, so quantizing it could
    # only add error. Neither is an unconverted op.
    assert all(isinstance(m, (nn.Dropout, Scale)) for m in report.left_float.values())


def test_the_padded_mask_conv_converts_and_nothing_is_unexported(converted, tmp_path):
    """Retired claim: "mask_head.proj is left float and listed in unexported"."""
    import json

    from quantization.export import export_integer_model
    from quantization.ilayers.conv import IConv2d

    model, _ = converted
    assert isinstance(model.mask_head.proj, IConv2d), "the padded 3x3 conv did not convert"
    assert tuple(model.mask_head.proj.padding) == (1, 1)
    export_integer_model(model, str(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["unexported"] == []


def test_total_params_counts_buffers_and_says_which_is_which(converted, tmp_path):
    """Retired claim: "manifest['total_params'] reads 0 on a converted model"."""
    import json

    from quantization.export import export_integer_model

    model, _ = converted
    export_integer_model(model, str(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_params"] > 0, "the field is useless again"
    # The parameters-only count IS zero on a converted model — every weight became a
    # buffer. That is a real fact, reported as its own field rather than folded into a
    # number people size deployments from.
    assert manifest["total_parameters_only"] == 0


def test_the_report_separates_deployment_coverage_from_conversion_coverage(converted):
    """R2: conversion converts the aux head too, and the count must not read as coverage.

    Nothing is wrong with converting it — the pass is general and should stay general.
    What would be wrong is "38 integer modules" standing as a statement about the
    accelerator, when 3 of them belong to a head no accelerator runs.
    """
    model, report = converted
    assert set(report.auxiliary) == {"mask_head.act", "mask_head.proj",
                                     "mask_head.to_logits"}
    assert len(report.deployed) == len(report.replaced) - len(report.auxiliary)
    assert not any(name.startswith("mask_head") for name in report.deployed)
    summary = report.summary()
    assert "on the inference path: 35" in summary
    assert "training-only (aux heads): 3" in summary


def test_which_heads_are_auxiliary_is_the_models_statement_not_the_quantizers(converted):
    """The rule has to live where the fact lives, or it becomes a second copy that drifts.

    The ROI and reliability heads are deliberately NOT auxiliary: the hybrid runtime
    scheduler reads reliability every step and the ROI box is a live cue for the search
    branch, so both are in ``HybridModel`` and both deploy. Only the mask head is not.
    """
    from engine.model_factory import make_model
    from models.frame.model import DirectPupilDetector

    assert DirectPupilDetector.AUXILIARY_HEADS == ("mask_head",)
    plain = make_model({"target": "models.frame.FrameModel"})
    names = set(plain.auxiliary_module_names())
    assert "mask_head.proj" in names
    assert not any(n.startswith(("roi_head", "reliability_head", "head.")) for n in names)

    hybrid = make_model({"target": "models.hybrid.HybridModel"})
    for deployed in ("roi_head", "search_reliability", "track_reliability"):
        assert getattr(hybrid, deployed, None) is not None, (
            f"{deployed} must be in the deployed system for it to count as non-auxiliary")


def test_a_model_that_declares_nothing_still_converts(converted):
    """A model with no auxiliary declaration must not gain an empty extra section."""
    from quantization.convert import ConversionReport

    report = ConversionReport(replaced={"a": None, "b": None})
    assert report.auxiliary == () and len(report.deployed) == 2
    assert "inference path" not in report.summary()


def test_the_deployed_model_has_no_mask_head_at_all():
    """The fact that makes ``FLOAT_IO_HEADS``'s reason true rather than an excuse.

    The mask head is stage-1 auxiliary supervision, not a runtime output, and the paper's
    full system does not instantiate one — so the integer graph omitting it is scope, and
    the integer-bilinear question it would raise never has to be answered.
    """
    from engine.model_factory import make_model
    from quantization.ilayers.model import FLOAT_IO_HEADS

    hybrid = make_model({"target": "models.hybrid.HybridModel"})
    assert not any("mask" in name for name, _ in hybrid.named_modules())
    assert "inference path" in FLOAT_IO_HEADS["mask_head"]


# --- and the retired phrasings must not come back ----------------------------

@pytest.mark.parametrize("phrase,why", [
    ("residual adds stay float", "D4 converted them; a FrameModel installs 4 IAdd"),
    ("left float and listed in", "D2 made padded convs convertible; unexported is empty"),
    ("Tensors still move between modules as", "A1 closed the transport"),
    ("the GRAPH does not", "A1 closed the transport"),
    ("the graph does not (see", "A1 closed the transport"),
])
def test_a_retired_claim_is_not_reintroduced(phrase, why):
    """These sentences were all true once. Each is now false, and each reads plausibly.

    Checked against the two files a user meets first — the example config and the
    entrypoint docstring — because that is where a stale claim does the most damage.
    """
    for path in (CONFIG, ENTRYPOINT):
        text = path.read_text(encoding="utf-8")
        assert phrase not in text, f"{path.name} says {phrase!r}, but {why}"
