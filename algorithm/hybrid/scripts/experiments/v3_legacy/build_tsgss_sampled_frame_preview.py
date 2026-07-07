#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from build_tsgss_frame_event_visualization import _load_frame_rgb, _read_jsonl_rows, _resolve_path
from hbtxr.utils.io import write_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render numbered previews for all sampled TSGSS frames using the global "
            "sample-manifest order."
        ),
    )
    parser.add_argument(
        "--sample-manifest",
        type=str,
        default="tsgss/dense_annotation/sample_manifest.jsonl",
        help="Global sampled-frame manifest produced by build_tsgss_dense_annotation.py",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="tsgss/sampled_frame_preview",
        help="Output root for numbered frame previews",
    )
    parser.add_argument(
        "--focus-session-key",
        type=str,
        default="user01/left/session_102",
        help="Session key for an additional focused contact sheet and GIF",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=160,
        help="Per-frame thumbnail width",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=120,
        help="Per-frame thumbnail height",
    )
    parser.add_argument(
        "--page-cols",
        type=int,
        default=12,
        help="Number of columns in the global contact-sheet pages",
    )
    parser.add_argument(
        "--page-rows",
        type=int,
        default=10,
        help="Number of rows in the global contact-sheet pages",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=350,
        help="Per-frame duration for the focus-session GIF",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_existing_frame_path(row: dict[str, Any]) -> Path | None:
    for key in ["sample_frame_path", "raw_frame_path"]:
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        path = path if path.is_absolute() else (PROJECT_ROOT / path)
        path = path.resolve()
        if path.exists():
            return path
    return None


def _annotated_tile(
    *,
    row: dict[str, Any],
    global_index: int,
    session_local_index: int,
    session_local_total: int,
    tile_size: tuple[int, int],
) -> Image.Image:
    frame_path = _resolve_existing_frame_path(row)
    if frame_path is not None:
        image = _load_frame_rgb(frame_path)
    else:
        image = Image.new("RGB", tile_size, color=(230, 225, 220))

    tile = ImageOps.pad(image, tile_size, color=(225, 225, 225))
    tile = tile.convert("RGB")
    draw = ImageDraw.Draw(tile)

    draw.rectangle((0, 0, tile.width, 24), fill=(18, 22, 28))
    draw.rectangle((0, tile.height - 32, tile.width, tile.height), fill=(245, 241, 235))
    draw.text((8, 6), f"TSGSS #{global_index:04d}", fill=(250, 248, 244))
    draw.text((tile.width - 44, 6), f"{session_local_index}/{session_local_total}", fill=(184, 226, 255))

    footer = f"{row['session_key']} | f={int(row['frame_idx'])}"
    draw.text((8, tile.height - 22), footer, fill=(38, 36, 34))
    if frame_path is None:
        draw.rectangle((8, 36, tile.width - 8, tile.height - 40), outline=(210, 54, 42), width=3)
        draw.text((12, 42), "missing source frame", fill=(210, 54, 42))
    return tile


def _build_contact_sheet(
    *,
    tiles: list[Image.Image],
    cols: int,
    rows: int,
    header_text: str,
    save_path: Path,
) -> None:
    if not tiles:
        return
    gap = 10
    header_h = 44
    tile_w, tile_h = tiles[0].size
    canvas_w = gap + cols * (tile_w + gap)
    canvas_h = header_h + gap + rows * (tile_h + gap)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(250, 247, 241))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), header_text, fill=(34, 31, 29))
    for idx, tile in enumerate(tiles):
        x = gap + (idx % cols) * (tile_w + gap)
        y = header_h + gap + (idx // cols) * (tile_h + gap)
        canvas.paste(tile, (x, y))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def _save_gif(frames: list[Image.Image], *, save_path: Path, duration_ms: int) -> None:
    if not frames:
        return
    palette_frames = [frame.convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    palette_frames[0].save(
        save_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=[int(duration_ms)] * len(palette_frames),
        loop=0,
        optimize=False,
        disposal=2,
    )


def main() -> None:
    args = build_argparser().parse_args()
    sample_manifest_path = _resolve_path(args.sample_manifest)
    output_root = _resolve_path(args.output_root)
    if output_root.exists() and not args.overwrite:
        summary_path = output_root / "experiment_summary.json"
        if summary_path.exists():
            print(f"[REUSED] {summary_path}")
            print(summary_path.read_text(encoding="utf-8"))
            return
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl_rows(sample_manifest_path)
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["session_key"])].append(row)

    session_local_map: dict[str, dict[str, int]] = {}
    for session_key, session_rows in grouped_rows.items():
        session_local_map[session_key] = {
            str(row["frame_filename"]): idx for idx, row in enumerate(session_rows, start=1)
        }

    tile_size = (int(args.tile_width), int(args.tile_height))
    page_capacity = int(args.page_cols) * int(args.page_rows)
    all_tiles: list[Image.Image] = []
    missing_source_count = 0
    page_summaries: list[dict[str, Any]] = []

    focus_session_key = str(args.focus_session_key)
    focus_tiles: list[Image.Image] = []
    focus_numbered_dir = output_root / "focus_session" / "numbered_frames"

    for global_index, row in enumerate(rows, start=1):
        session_key = str(row["session_key"])
        session_rows = grouped_rows[session_key]
        session_local_index = int(session_local_map[session_key][str(row["frame_filename"])])
        if _resolve_existing_frame_path(row) is None:
            missing_source_count += 1
        tile = _annotated_tile(
            row=row,
            global_index=global_index,
            session_local_index=session_local_index,
            session_local_total=len(session_rows),
            tile_size=tile_size,
        )
        all_tiles.append(tile)

        if session_key == focus_session_key:
            focus_tiles.append(tile.copy())
            focus_numbered_dir.mkdir(parents=True, exist_ok=True)
            tile.save(focus_numbered_dir / f"frame_{session_local_index:02d}_tsgss_{global_index:04d}.png")

    global_pages_dir = output_root / "global_pages"
    global_pages_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(all_tiles), page_capacity):
        chunk = all_tiles[start : start + page_capacity]
        page_index = start // page_capacity + 1
        start_index = start + 1
        end_index = start + len(chunk)
        page_path = global_pages_dir / f"page_{page_index:03d}_tsgss_{start_index:04d}_{end_index:04d}.png"
        _build_contact_sheet(
            tiles=chunk,
            cols=int(args.page_cols),
            rows=int(args.page_rows),
            header_text=(
                f"TSGSS Sampled Frames | Global Order {start_index:04d}-{end_index:04d} | "
                f"{len(rows)} total frames"
            ),
            save_path=page_path,
        )
        page_summaries.append(
            {
                "page_index": page_index,
                "start_global_index": start_index,
                "end_global_index": end_index,
                "page_path": str(page_path),
            },
        )

    focus_root = output_root / "focus_session"
    focus_contact_sheet = focus_root / "sampled_frames_numbered_contact_sheet.png"
    focus_gif = focus_root / "sampled_frames_numbered.gif"
    if focus_tiles:
        _build_contact_sheet(
            tiles=focus_tiles,
            cols=min(4, len(focus_tiles)),
            rows=(len(focus_tiles) + min(4, len(focus_tiles)) - 1) // min(4, len(focus_tiles)),
            header_text=f"Focus Session {focus_session_key} | TSGSS Global Order",
            save_path=focus_contact_sheet,
        )
        _save_gif(focus_tiles, save_path=focus_gif, duration_ms=int(args.gif_duration_ms))

    quick_preview_root = output_root.parent / f"{output_root.name}_quick_preview"
    quick_preview_root.mkdir(parents=True, exist_ok=True)
    if page_summaries:
        first_page = Path(page_summaries[0]["page_path"])
        first_page_dst = quick_preview_root / "01_global_page_001.png"
        if first_page_dst.exists() or first_page_dst.is_symlink():
            first_page_dst.unlink()
        first_page_dst.symlink_to(first_page)
    if focus_contact_sheet.exists():
        focus_sheet_dst = quick_preview_root / "02_focus_session_contact_sheet.png"
        if focus_sheet_dst.exists() or focus_sheet_dst.is_symlink():
            focus_sheet_dst.unlink()
        focus_sheet_dst.symlink_to(focus_contact_sheet)
    if focus_gif.exists():
        focus_gif_dst = quick_preview_root / "03_focus_session_numbered.gif"
        if focus_gif_dst.exists() or focus_gif_dst.is_symlink():
            focus_gif_dst.unlink()
        focus_gif_dst.symlink_to(focus_gif)

    summary = {
        "experiment": "tsgss_sampled_frame_preview",
        "sample_manifest_path": str(sample_manifest_path),
        "output_root": str(output_root),
        "quick_preview_root": str(quick_preview_root),
        "n_total_sampled_frames": len(rows),
        "n_sessions": len(grouped_rows),
        "page_cols": int(args.page_cols),
        "page_rows": int(args.page_rows),
        "page_capacity": int(page_capacity),
        "n_pages": len(page_summaries),
        "missing_source_count": int(missing_source_count),
        "focus_session_key": focus_session_key,
        "focus_session_frame_count": len(focus_tiles),
        "focus_contact_sheet_path": str(focus_contact_sheet) if focus_contact_sheet.exists() else None,
        "focus_gif_path": str(focus_gif) if focus_gif.exists() else None,
        "pages": page_summaries,
    }
    write_json(summary, output_root / "experiment_summary.json")
    readme_lines = [
        "# TSGSS Sampled Frame Preview",
        "",
        "## Summary",
        f"- total sampled frames: `{summary['n_total_sampled_frames']}`",
        f"- sessions: `{summary['n_sessions']}`",
        f"- global pages: `{summary['n_pages']}`",
        f"- focus session: `{summary['focus_session_key']}`",
        f"- quick preview root: `{quick_preview_root}`",
        "",
        "## Quick Preview",
        "- `tsgss/sampled_frame_preview_quick_preview/01_global_page_001.png`",
        "- `tsgss/sampled_frame_preview_quick_preview/02_focus_session_contact_sheet.png`",
        "- `tsgss/sampled_frame_preview_quick_preview/03_focus_session_numbered.gif`",
    ]
    (output_root / "README.md").write_text("\n".join(readme_lines).rstrip() + "\n", encoding="utf-8")
    print(
        f"[DONE] frames={summary['n_total_sampled_frames']} pages={summary['n_pages']} "
        f"focus={focus_session_key} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
