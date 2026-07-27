"""D5: the HG-PIPE ``.txt`` render of an integer dump.

The HLS backend ``#include``s comma-separated integer literals; it reads neither JSON
nor ``.npy``. These tests hold the render to the reference's exact byte format and to
the property that actually matters — every value round-trips, so the ``.txt`` view and
the manifest cannot drift.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from quantization.calibrate import post_training_quantize
from quantization.convert import convert_model_to_integer
from quantization.export import MANIFEST_NAME, export_integer_model
from quantization.export_txt import SCALAR_KEYS, TABLE_KEYS, read_csv_ints, write_hgpipe_txt


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    from engine.model_factory import make_model

    torch.manual_seed(0)
    model = make_model({"target": "models.frame.FrameModel", "embed_dim": 48, "patch_size": 16,
                        "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1}})
    calib = [torch.rand(2, 1, 64, 64, generator=torch.Generator().manual_seed(7))]
    model, _ = post_training_quantize(model, calib)
    model, _ = convert_model_to_integer(model, calib)
    out = tmp_path_factory.mktemp("int_export")
    manifest = export_integer_model(model, out)
    return out, manifest


def test_render_matches_the_reference_byte_format(dump):
    """One line, comma separated, TRAILING comma, no newline — a C initializer body."""
    out, _ = dump
    files = write_hgpipe_txt(out)
    csvs = [f for f in files if f.name.endswith(".txt")]
    assert csvs, "nothing was rendered"
    for path in csvs:
        raw = path.read_text(encoding="ascii")
        assert "\n" not in raw, f"{path.name} is not a single line"
        assert raw.endswith(","), f"{path.name} lacks the trailing comma"
        assert raw.count(",") == len(read_csv_ints(path)), f"{path.name} has a stray separator"


def test_the_generic_table_key_is_renamed_but_the_others_are_not(dump):
    """The reference calls the generic ``table`` ``table_m``; every other key is verbatim."""
    from quantization.export import _stem

    out, manifest = dump
    write_hgpipe_txt(out)
    names = {p.name for p in (out / "txt").glob("*.txt")}
    gelus = [(n, e) for n, e in manifest["modules"].items() if e["op"] == "gelu_lut"]
    assert gelus, "fixture has no GeLU to check the rename against"
    for name, entry in gelus:
        stem = _stem(int(entry["order"]), name)
        assert f"{stem}_table_m.txt" in names, f"{name}: generic `table` was not renamed"
        assert f"{stem}_table.txt" not in names, f"{name}: unrenamed `table` file present"
    # ... while a named table keeps its manifest key, so a reader can map it back
    assert any(n.endswith("_rsqrt_table.txt") for n in names)
    assert any(n.endswith("_exp_table.txt") for n in names)
    assert any(n.endswith("_recip_table_one.txt") for n in names)


def test_every_value_round_trips_against_the_manifest(dump):
    """The render is a VIEW: if a value differs from the manifest it is worthless."""
    out, manifest = dump
    write_hgpipe_txt(out)
    on_disk = json.loads((out / MANIFEST_NAME).read_text())
    checked = 0
    for name, entry in on_disk["modules"].items():
        from quantization.export import _stem

        stem = _stem(int(entry["order"]), name)
        for key in SCALAR_KEYS + TABLE_KEYS:
            if entry.get(key) is None:
                continue
            suffix = {"table": "table_m"}.get(key, key)
            path = out / "txt" / f"{stem}_{suffix}.txt"
            assert path.exists(), f"{name}: {key} was not rendered"
            assert read_csv_ints(path) == [int(v) for v in entry[key]], f"{name}: {key} differs"
            checked += 1
    assert checked >= 20, f"only {checked} fields checked — the fixture got thinner"


def test_index_maps_every_file_back_to_its_module_and_field(dump):
    """A .txt carries no metadata, so the mapping has to live somewhere."""
    out, _ = dump
    files = write_hgpipe_txt(out)
    index = out / "txt" / "index.tsv"
    assert index in files
    rows = index.read_text(encoding="ascii").strip().splitlines()
    assert rows[0] == "file\tmodule\top\tfield"
    listed = {r.split("\t")[0] for r in rows[1:]}
    rendered = {p.name for p in (out / "txt").glob("*.txt")}
    assert listed == rendered, f"index and directory disagree: {listed ^ rendered}"


def test_weights_are_opt_in(dump):
    """The default render is scalars + tables, as the reference produces."""
    out, _ = dump
    plain = out / "txt_plain"
    write_hgpipe_txt(out, txt_dir=plain)
    assert not list(plain.glob("*_weight.txt"))
    with_weights = out / "txt_weights"
    write_hgpipe_txt(out, txt_dir=with_weights, include_weights=True)
    produced = list(with_weights.glob("*_weight.txt"))
    assert produced, "include_weights=True rendered no weights"
    assert read_csv_ints(produced[0]), "weight render is empty"


def test_a_corrupt_dump_is_refused_before_anything_is_rendered(dump, tmp_path):
    """Rendering a dump that does not load would produce plausible-looking garbage."""
    import shutil

    import numpy as np

    out, manifest = dump
    broken = tmp_path / "broken"
    shutil.copytree(out, broken)
    weight = next(broken.glob("*.npy"))
    np.save(weight, np.zeros_like(np.load(weight)))
    with pytest.raises(ValueError):
        write_hgpipe_txt(broken, txt_dir=tmp_path / "should_not_appear")
    assert not (tmp_path / "should_not_appear").exists()
