from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_dataset import build_session_index
from overlay_preview import run


def test_overlay_preview_generates_overlay_sheet(synthetic_workspace) -> None:
    build_session_index(canonical_root=synthetic_workspace["canonical_root"], indexes_root=synthetic_workspace["indexes_root"])
    output_dir = synthetic_workspace["project_root"] / "overlay_preview"
    result = run(
        argparse.Namespace(
            canonical_root=str(synthetic_workspace["canonical_root"]),
            indexes_root=str(synthetic_workspace["indexes_root"]),
            output_dir=str(output_dir),
            samples_per_session=1,
            seed=7,
            user=[],
            eye=[],
            session_key=[],
            max_sessions=None,
            tile_width=320,
            columns=2,
            hide_mask=False,
        )
    )

    assert result["session_count"] == 1
    output_path = Path(result["rows"][0]["output"])
    assert output_path.exists()

    image = Image.open(output_path).convert("RGB")
    arr = np.asarray(image)
    assert arr.ndim == 3
    assert np.any(arr[..., 0] != arr[..., 1]) or np.any(arr[..., 1] != arr[..., 2])
    assert (output_dir / "overlay_preview_summary.json").exists()
