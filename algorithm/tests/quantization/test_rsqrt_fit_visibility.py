"""R4: the rsqrt index's fit quality must survive calibration, and here is why it matters.

R4 set out to answer whether the two-segment rsqrt index leaves accuracy on the table.
The measurement says no — see ``docs/QUANTIZATION-PARTCD-REPORT.md`` — but getting to it
turned up something else: the Part C/D report claimed "the payload carries ``metrics`` so
the error is visible", and it did not. ``calibrate_int_layernorm`` computes a full fit
report (``rsqrt_rel_rms/p99/max``, the observed variance range, and how many rows fall
outside the index), ``ILayerNorm`` dropped it on construction, and ``export.py`` builds its
entry from the module — so nothing reached the artifact.

That matters more than the segment count. The fit quality is NOT recoverable from the
exported tables afterwards, and ``rows_above_range`` / ``rows_below_range`` are documented
as "a warning about the fit" — rows pinned to an extreme table entry. A warning nobody can
read after calibration is not a warning. It is also precisely what a trained checkpoint
(A2) will need in order to answer the segment question on a real variance distribution
instead of this one.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from quantization.calibrate import post_training_quantize
from quantization.convert import convert_model_to_integer
from quantization.export import export_integer_model
from quantization.ilayers.layernorm import ILayerNorm
from quantization.ilayers.qtensor import QTensor
from quantization.scheme import INT8

WARNING_FIELDS = ("rows_above_range", "rows_below_range")
FIT_FIELDS = ("rsqrt_rel_rms", "rsqrt_rel_p99", "rsqrt_rel_max", "var_sum_dynamic_range")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from engine.model_factory import make_model

    torch.manual_seed(0)
    model = make_model({"target": "models.frame.FrameModel", "embed_dim": 48,
                        "patch_size": 16,
                        "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0,
                                     "cut_point": 1}})
    calib = [torch.rand(2, 1, 64, 64, generator=torch.Generator().manual_seed(7))]
    model, _ = post_training_quantize(model, calib)
    model, _ = convert_model_to_integer(model, calib)
    out = tmp_path_factory.mktemp("export")
    export_integer_model(model.eval(), str(out))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entries = {name: e for name, e in manifest["modules"].items()
               if e.get("op") == "layernorm_int"}
    return model.eval(), entries


def test_every_exported_layernorm_reports_how_well_its_index_fits(exported):
    """The claim the report made and the code did not keep."""
    _model, entries = exported
    assert entries, "fixture must contain integer layernorms"
    for name, entry in entries.items():
        metrics = entry.get("metrics")
        assert metrics, f"{name} exported no fit metrics"
        for field in FIT_FIELDS + WARNING_FIELDS:
            assert field in metrics, f"{name} is missing {field}"


def test_the_out_of_range_warning_is_readable_after_export(exported):
    """``rows_above_range`` is the field that says the index did not cover the data.

    It is an int count, not a flag, because "3 rows pinned at the top entry" and "40% of
    rows pinned" are different situations and the reader has to be able to tell them apart.
    """
    _model, entries = exported
    for name, entry in entries.items():
        metrics = entry["metrics"]
        for field in WARNING_FIELDS:
            assert isinstance(metrics[field], int), f"{name}.{field} is not a count"
        assert metrics["rows"] > 0
        assert metrics[WARNING_FIELDS[0]] <= metrics["rows"]


def test_the_recorded_fit_matches_what_the_index_actually_does(exported):
    """A metric that is not derived from the deployed kernel is decoration.

    Recomputed here from the exported scalars and tables — the same lookup the golden
    performs, branch included — against the variance the module saw at calibration.
    """
    import numpy as np

    from quantization.int_calibrate import PotIndexParams, _segmented_index_error

    _model, entries = exported
    for name, entry in entries.items():
        if entry["segments"] != 2:
            continue
        metrics = entry["metrics"]
        lo, hi = metrics["var_sum_min"], metrics["var_sum_max"]
        assert lo <= hi and metrics["var_sum_dynamic_range"] == pytest.approx(
            max(hi, 1) / max(lo, 1), rel=1e-6), name

        _c1m, _c1s, b, s1, bound, _s2, _clamp = entry["scalars"]
        b_two, s1_two, bound_two = entry["scalars_two"]
        one = PotIndexParams(offset=b, shift=s1, bound=bound)
        two = PotIndexParams(offset=b_two, shift=s1_two, bound=bound_two)
        unit = (float(entry["input_scale"]) ** 2) / float(entry["channels"])
        span = np.arange(lo, hi + 1, dtype=np.int64)
        err, _ = _segmented_index_error(one, two, span, unit, 1e-6)
        # the recorded max is over the rows actually observed, a subset of [lo, hi],
        # so it must not exceed the worst case over the whole span
        assert metrics["rsqrt_rel_max"] <= float(np.abs(err).max()) + 1e-9, name

        # And the family must be internally consistent, because "0.0" is a number a
        # fabricated metric reaches for and every upper bound above accepts. This
        # subsystem has shipped exactly that once already — verify_export reported
        # max_abs_diff=0.0 for comparisons it never ran — so the lower bound is checked
        # too. rms, p99 and max are all over the SAME observed rows, so from the
        # definitions: max >= p99, max >= rms, and rms >= max / sqrt(rows).
        rms, p99, mx, rows = (metrics["rsqrt_rel_rms"], metrics["rsqrt_rel_p99"],
                              metrics["rsqrt_rel_max"], metrics["rows"])
        assert mx > 0.0, f"{name}: a linear index over a log-domain function is not exact"
        assert p99 <= mx + 1e-12 and rms <= mx + 1e-12, name
        assert rms >= mx / (rows ** 0.5) - 1e-12, (
            f"{name}: rms {rms} is below the floor {mx / rows ** 0.5} its own max implies")


def test_the_metrics_are_diagnostic_and_never_reach_the_datapath():
    """A field the kernel reads would be a field that changes the answer when dropped."""
    scalars = [44739243, 31, -38306, 6, 255, 22, 8, -38306, 6, 255]
    table = list(range(256))
    kwargs = dict(input_scale=0.01, output_scale=0.04, rsqrt_table_two=table)
    plain = ILayerNorm(scalars, table, [1] * 8, [0] * 8, **kwargs)
    annotated = ILayerNorm(scalars, table, [1] * 8, [0] * 8,
                           metrics={"rsqrt_rel_rms": 0.5}, **kwargs)
    assert plain.metrics == {} and annotated.metrics["rsqrt_rel_rms"] == 0.5
    x = QTensor(torch.arange(-4, 4, dtype=torch.int32).reshape(1, 8), scale=0.01,
                zero_point=0.0, dtype=INT8)
    assert torch.equal(plain.forward_int(x).int_data, annotated.forward_int(x).int_data)


def test_a_payload_without_metrics_still_loads():
    """Every payload written before this existed has no ``metrics`` key."""
    payload = {
        "scalars": [44739243, 31, -38306, 6, 255, 22, 8],
        "rsqrt_table": list(range(256)), "lnw": [1] * 8, "lnb": [0] * 8,
        "input_scale": 0.01, "output_scale": 0.04,
    }
    module = ILayerNorm.from_payload(payload)
    assert module.metrics == {} and module.segments == 1


def test_the_measured_fit_is_good_enough_that_a_third_segment_cannot_matter(exported):
    """R4's actual answer, pinned so it is a measurement and not a memory.

    On this fixture the observed variance dynamic range is under 2:1 — nowhere near the
    ``bound+1``:1 that D3 identified as the limit of a linear index over a log-domain
    function — and the residual rsqrt error is ~1e-4 relative. A simulated third segment
    improves that by ~1.6x, on a quantity that is already four orders of magnitude below
    the graph's ~2e-2 end-to-end error. No third segment is justified.

    The bound is asserted loosely on purpose: the point is the ORDER, and this is a
    randomly-initialised model whose variance spread is plausibly the floor. A trained
    checkpoint is what would move it, which is why the metrics now survive to the export.
    """
    _model, entries = exported
    for name, entry in entries.items():
        metrics = entry["metrics"]
        assert metrics["rsqrt_rel_rms"] < 1e-3, f"{name}: {metrics['rsqrt_rel_rms']}"
        assert metrics["var_sum_dynamic_range"] < 8.0, (
            f"{name}: dynamic range {metrics['var_sum_dynamic_range']:.1f} — wide enough "
            "that the segment count is worth revisiting")
