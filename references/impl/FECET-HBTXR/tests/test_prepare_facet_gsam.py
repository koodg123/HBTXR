from __future__ import annotations

import csv
from pathlib import Path

from fecet_hbtxr.io import read_json, read_jsonl
from prepare_facet_gsam_dataset import prepare_facet_gsam_dataset


def _build_raw_session(raw_root: Path) -> dict[str, Path]:
    session_dir = raw_root / "user1" / "left" / "session_1_0_1"
    frames_dir = session_dir / "frames"
    events_dir = session_dir / "events"
    frames_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    frame_names = [
        "000001_1000.png",
        "000002_2000.png",
    ]
    for frame_name in frame_names:
        (frames_dir / frame_name).write_bytes(
            bytes.fromhex(
                "89504E470D0A1A0A0000000D49484452000000010000000108000000003A7E9B550000000A49444154789C6360000000020001E221BC330000000049454E44AE426082"
            )
        )

    events_text = "\n".join(
        [
            "1000 150 120 1",
            "1500 151 121 0",
            "2000 152 122 1",
        ]
    ) + "\n"
    events_path = events_dir / "events.txt"
    events_path.write_text(events_text, encoding="utf-8")
    original_events = events_text

    csv_path = session_dir / "user_1.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filename",
                "file_size",
                "file_attributes",
                "region_count",
                "region_id",
                "region_shape_attributes",
                "region_attributes",
            ]
        )
        writer.writerow(
            [
                "000001_1000.png",
                "0",
                "{}",
                "1",
                "0",
                '{"name":"ellipse","cx":150,"cy":120,"rx":20,"ry":10,"theta":0.5}',
                "{}",
            ]
        )
        writer.writerow(
            [
                "000002_2000.png",
                "0",
                "{}",
                "1",
                "0",
                '{"name":"ellipse","cx":152,"cy":122,"rx":18,"ry":9,"theta":0.25}',
                "{}",
            ]
        )

    return {
        "session_dir": session_dir,
        "events_path": events_path,
        "original_events": original_events,
    }


def test_prepare_facet_gsam_dataset_csv_only(tmp_path: Path):
    raw_root = tmp_path / "raw" / "Data_davis"
    grounded_sam_root = tmp_path / "grounded_sam"
    grounded_sam_root.mkdir(parents=True, exist_ok=True)
    raw_info = _build_raw_session(raw_root)

    project_root = tmp_path / "project"
    canonical_root = project_root / "canonical"
    manifests_root = project_root / "manifests"
    splits_root = project_root / "splits"
    facet_root = project_root / "facet_style"

    summary = prepare_facet_gsam_dataset(
        raw_root=raw_root,
        grounded_sam_root=grounded_sam_root,
        canonical_root=canonical_root,
        manifests_root=manifests_root,
        splits_root=splits_root,
        facet_root=facet_root,
        annotation_mode="csv_only",
        link_mode="copy",
        export_frames="none",
        overwrite=False,
        official_only=True,
        annotation_stride=100,
        annotation_offset=0,
        frame_limit=None,
        users=[1],
        eyes=["left"],
        split_scheme="random",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        gsam_kwargs={
            "grounding_config": grounded_sam_root / "dummy.py",
            "grounded_checkpoint": grounded_sam_root / "dummy.pth",
            "sam_checkpoint": grounded_sam_root / "dummy_sam.pth",
        },
    )

    assert summary["canonical"]["sessions_prepared"] == 1
    package_path = canonical_root / "sessions" / "user01" / "left" / "session_101" / "labels" / "session_package.json"
    assert package_path.exists()
    package = read_json(package_path)
    assert package["n_labelled_frames"] == 2

    ann_rows = read_jsonl(canonical_root / "sessions" / "user01" / "left" / "session_101" / "labels" / "frame_annotations.jsonl")
    assert len(ann_rows) == 2
    assert ann_rows[0]["annotation_source"] == "manual_csv"

    train_manifest = read_jsonl(manifests_root / "train_manifest.jsonl")
    assert len(train_manifest) == 2

    events_export = facet_root / "train" / "data" / "user01_left_session_101_events.txt"
    ellipse_export = facet_root / "train" / "ellipse" / "user01_left_session_101_ellipses.txt"
    centers_export = facet_root / "train" / "label" / "user01_left_session_101_centers.txt"
    assert events_export.exists()
    assert ellipse_export.exists()
    assert centers_export.exists()
    ellipse_lines = ellipse_export.read_text(encoding="utf-8").strip().splitlines()
    assert ellipse_lines[0].endswith("28.65")
    assert raw_info["events_path"].read_text(encoding="utf-8") == raw_info["original_events"]


def test_prepare_facet_gsam_rejects_output_inside_input_root(tmp_path: Path):
    raw_root = tmp_path / "raw" / "Data_davis"
    grounded_sam_root = tmp_path / "grounded_sam"
    grounded_sam_root.mkdir(parents=True, exist_ok=True)
    _build_raw_session(raw_root)

    try:
        prepare_facet_gsam_dataset(
            raw_root=raw_root,
            grounded_sam_root=grounded_sam_root,
            canonical_root=raw_root / "bad_output",
            manifests_root=tmp_path / "manifests",
            splits_root=tmp_path / "splits",
            facet_root=tmp_path / "facet_style",
            annotation_mode="csv_only",
            link_mode="copy",
            export_frames="none",
            overwrite=False,
            official_only=True,
            annotation_stride=100,
            annotation_offset=0,
            frame_limit=None,
            users=[1],
            eyes=["left"],
            split_scheme="random",
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
            gsam_kwargs={
                "grounding_config": grounded_sam_root / "dummy.py",
                "grounded_checkpoint": grounded_sam_root / "dummy.pth",
                "sam_checkpoint": grounded_sam_root / "dummy_sam.pth",
            },
        )
    except ValueError as exc:
        assert "must not be nested under input root" in str(exc)
    else:
        raise AssertionError("expected ValueError for nested output root")
