from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from hbtxr.config.runtime_config import build_dataset_kwargs
from hbtxr.data.dataset import EVEyeHBTXRDataset, load_track_target_overrides
from hbtxr.data.loader import collate_samples


def test_load_track_target_overrides_accepts_payload_dict(tmp_path: Path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "sample_id": "s1",
                        "track_target_override_state": [1, 2, 3, 4, 0, 1],
                        "track_target_override_weight": 0.5,
                        "track_target_override_source": "teacher",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    overrides = load_track_target_overrides(path)

    assert sorted(overrides) == ["s1"]
    assert overrides["s1"]["track_target_override_source"] == "teacher"


def test_load_track_target_overrides_rejects_missing_state(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"sample_id": "s1"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="requires 6D state"):
        load_track_target_overrides(path)


def test_build_dataset_kwargs_passes_target_override_config(tmp_path: Path) -> None:
    override_path = tmp_path / "overrides.json"
    kwargs = build_dataset_kwargs(
        {
            "track_target_override_path": str(override_path),
            "allow_test_target_override": True,
        }
    )

    assert kwargs["track_target_override_path"] == str(override_path)
    assert kwargs["allow_test_target_override"] is True


def test_dataset_rejects_test_manifest_target_override(tmp_path: Path) -> None:
    manifest = tmp_path / "test_manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"overrides": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="disabled for test manifests"):
        EVEyeHBTXRDataset(str(manifest), track_target_override_path=str(overrides))


def test_collate_samples_stacks_track_target_override_tensors() -> None:
    batch = [
        {
            "sample_id": "s1",
            "meta": {"track_target_override_source": "a"},
            "track_target_override_state": torch.zeros(6),
            "track_target_override_weight": torch.tensor(1.0),
        },
        {
            "sample_id": "s2",
            "meta": {"track_target_override_source": "b"},
            "track_target_override_state": torch.ones(6),
            "track_target_override_weight": torch.tensor(0.0),
        },
    ]

    collated = collate_samples(batch)

    assert collated["sample_id"] == ["s1", "s2"]
    assert collated["track_target_override_state"].shape == (2, 6)
    assert torch.allclose(collated["track_target_override_weight"], torch.tensor([1.0, 0.0]))


def test_xr64_builder_rejects_test_split_before_loading_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path("scripts/external/build_xr64_teacher_target_overrides.py").resolve()
    spec = importlib.util.spec_from_file_location("build_xr64_teacher_target_overrides", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    monkeypatch.syspath_prepend(str(script.parent))
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--reference-config",
            str(tmp_path / "missing_config.json"),
            "--manifest",
            str(tmp_path / "test_manifest.jsonl"),
            "--split",
            "test",
            "--eval",
            f"teacher={tmp_path / 'missing_eval_rows.json'}",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )

    with pytest.raises(ValueError, match="must not be built from test split"):
        module.main()
