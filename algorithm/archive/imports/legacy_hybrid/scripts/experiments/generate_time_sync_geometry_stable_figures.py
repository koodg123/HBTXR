from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "workspace_figures" / "time_sync_geometry_stable_figure_set"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig: plt.Figure, path: Path) -> None:
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _fit_image(path: str | Path, max_size: tuple[int, int]) -> Image.Image:
    img = _load_image(path)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def _draw_box(ax, x, y, w, h, text, facecolor, edgecolor="#222222", fontsize=11):
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fontsize, color="#111111")


def _draw_arrow(ax, x0, y0, x1, y1, color="#444444", lw=2.0):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=0, shrinkB=0),
    )


def _panel_title(ax, title: str, subtitle: str | None = None):
    ax.set_title(title, fontsize=15, loc="left", pad=12, fontweight="bold")
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color="#555555", va="bottom")


def figure_01_frame_interpolation(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    _panel_title(
        ax,
        "Time-Synchronous: Frame Interpolation (TimeLens-XL) with Adaptive TimeBin",
        "Conceptual supervisory timeline for sparse labeled frames and inserted interpolation anchors.",
    )

    ax.plot([0.6, 9.4], [1.4, 1.4], color="#444444", linewidth=2.0)
    frame_x = [1.0, 3.0, 5.0, 7.2, 9.0]
    labels = ["t0", "t1", "t2", "t3", "t4"]
    for x, label in zip(frame_x, labels):
        ax.add_patch(patches.Circle((x, 1.4), 0.08, color="#1f77b4"))
        ax.text(x, 1.07, label, ha="center", va="top", fontsize=11)
        ax.text(x, 0.78, "raw frame", ha="center", va="top", fontsize=9, color="#1f77b4")

    interp_x = [2.0, 4.1, 4.55, 6.1, 8.1]
    for idx, x in enumerate(interp_x, start=1):
        ax.add_patch(patches.Circle((x, 1.4), 0.07, color="#7fc8ff"))
        ax.text(x, 1.78, f"τ{idx}", ha="center", va="bottom", fontsize=10, color="#0f4c81")
        ax.text(x, 2.02, "interp", ha="center", va="bottom", fontsize=8, color="#0f4c81")

    bins = [
        (1.55, 0.95, 0.75, 0.34, "narrow\nTimeBin"),
        (3.55, 0.95, 1.35, 0.34, "wide\nTimeBin"),
        (5.7, 0.95, 0.75, 0.34, "narrow\nTimeBin"),
        (7.75, 0.95, 1.0, 0.34, "medium\nTimeBin"),
    ]
    for x, y, w, h, text in bins:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor="#dceeff",
                edgecolor="#4c78a8",
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=8)

    _draw_box(ax, 0.7, 2.65, 1.7, 0.62, "Sparse labeled frames", "#e8f2ff")
    _draw_box(ax, 3.0, 2.65, 1.9, 0.62, "TimeLens-XL interpolation", "#d8ebff")
    _draw_box(ax, 5.5, 2.65, 1.8, 0.62, "Adaptive TimeBin", "#d9f4e8")
    _draw_box(ax, 7.8, 2.65, 1.4, 0.62, "Aligned anchor", "#fff2cf")
    _draw_arrow(ax, 2.42, 2.96, 2.96, 2.96)
    _draw_arrow(ax, 4.92, 2.96, 5.46, 2.96)
    _draw_arrow(ax, 7.34, 2.96, 7.76, 2.96)

    ax.text(
        0.72,
        3.45,
        "Short bins near rapid motion, wider bins in quieter spans.\nInserted interpolation anchors shrink frame-event misalignment before annotation.",
        fontsize=10,
        color="#333333",
    )

    out = output_dir / "01_time_sync_frame_interpolation_timelens_xl_adaptive_timebin.png"
    _save_fig(fig, out)
    return out


def figure_02_event_interpolation(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    _panel_title(
        ax,
        "Time-Synchronous: Event Interpolation (V2E) with Adaptive Event Counts",
        "Canonical timeline stays fixed; event-count policy adapts the window width for each anchor.",
    )

    ax.plot([0.7, 9.3], [1.25, 1.25], color="#444444", linewidth=2.0)
    rng = np.random.default_rng(7)
    dense_left = np.linspace(1.0, 2.0, 14)
    sparse_mid = np.linspace(3.4, 4.2, 4)
    dense_right = np.linspace(6.0, 8.5, 18)
    times = np.concatenate([dense_left, sparse_mid, dense_right])
    heights = 0.25 + 0.55 * rng.random(len(times))
    colors = ["#2b6cb0" if i % 2 == 0 else "#ef6c57" for i in range(len(times))]
    for t, h, c in zip(times, heights, colors):
        ax.plot([t, t], [1.25, 1.25 + h], color=c, linewidth=2.0)

    anchor_x = [2.0, 4.1, 7.2]
    windows = [(1.65, 0.8, "K≈5000\nshort Δt"), (3.1, 2.0, "K≈5000\nwide Δt"), (6.65, 1.1, "K≈5000\nmedium Δt")]
    for x in anchor_x:
        ax.add_patch(patches.Circle((x, 1.25), 0.09, color="#111111"))
        ax.text(x, 0.9, "anchor", ha="center", va="top", fontsize=9)
    for x, w, text in windows:
        ax.add_patch(
            patches.Rectangle((x, 0.65), w, 1.45, fill=False, edgecolor="#10a37f", linewidth=2.2, linestyle="--")
        )
        ax.text(x + w / 2.0, 2.25, text, ha="center", va="bottom", fontsize=9, color="#106f57")

    _draw_box(ax, 0.7, 3.45, 1.55, 0.62, "Async raw events", "#f4f6f8")
    _draw_box(ax, 2.8, 3.45, 1.4, 0.62, "Adaptive counts", "#d9f4e8")
    _draw_box(ax, 4.75, 3.45, 1.35, 0.62, "V2E / synth", "#fbe8d3")
    _draw_box(ax, 6.65, 3.45, 1.95, 0.62, "Frame-synchronous event tensor", "#e8f2ff")
    _draw_arrow(ax, 2.28, 3.76, 2.76, 3.76)
    _draw_arrow(ax, 4.23, 3.76, 4.7, 3.76)
    _draw_arrow(ax, 6.15, 3.76, 6.6, 3.76)

    ax.text(
        0.72,
        4.28,
        "Dense motion spans use short windows; quiet spans expand the horizon to maintain a target count.\nThis keeps event tensors comparable while preserving a time-synchronous supervisory index.",
        fontsize=10,
        color="#333333",
    )

    out = output_dir / "02_time_sync_event_interpolation_v2e_adaptive_event_counts.png"
    _save_fig(fig, out)
    return out


def figure_03_frame_event_pairing(output_dir: Path) -> Path:
    frame_img = _fit_image(
        REPO_ROOT
        / "workspace_event_voxel_analysis/interval_5012_frame_eye_pupil_boxes_only/user01_left_session201_interval5012_frame_eye_pupil_boxes_only.png",
        (700, 420),
    )
    event_img = _fit_image(
        REPO_ROOT / "workspace_event_voxel_analysis/interval_5012_accumulated_channels/user01_session201_left_interval_5012_positive_accum.png",
        (700, 420),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    _panel_title(
        axes[0],
        "Geometry-Stable: Frame-Event Pairing",
        "Actual user01/left/session_201 interval 5012 example with one frame anchor and its paired event window.",
    )
    axes[0].imshow(frame_img)
    axes[0].axis("off")
    axes[0].text(
        0.02,
        -0.08,
        "Frame anchor t: raw frame + eye ROI + pupil box",
        transform=axes[0].transAxes,
        fontsize=10,
        color="#333333",
    )
    axes[1].imshow(event_img)
    axes[1].axis("off")
    axes[1].text(
        0.02,
        -0.08,
        "Event window W(t): positive accumulation aligned to the same anchor",
        transform=axes[1].transAxes,
        fontsize=10,
        color="#333333",
    )
    fig.text(
        0.5,
        0.03,
        "Pairing rule: frame anchor, accumulated event tensor, previous state, and annotation targets all refer to the same canonical time index.",
        ha="center",
        fontsize=11,
        color="#333333",
    )
    out = output_dir / "03_geometry_stable_frame_event_pairing.png"
    _save_fig(fig, out)
    return out


def figure_04_dense_annotation_stage1(output_dir: Path) -> Path:
    img = _fit_image(
        REPO_ROOT
        / "workspace_event_voxel_analysis/interval_5012_frame_eye_pupil_mask_overlay_blue/user01_left_session201_interval5012_frame_eye_pupil_mask_overlay_blue.png",
        (1050, 640),
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    _panel_title(
        ax,
        "Dense Annotation (Grounded-SAM): Eye Region ROI Mask / Box with Pupil Guide",
        "Stage1 training-oriented geometry supervision example.",
    )
    ax.imshow(img)
    ax.axis("off")
    ax.text(
        0.02,
        -0.08,
        "Green box: Eye Region ROI    Red box: Pupil Box guide    Blue fill: Pupil mask proxy",
        transform=ax.transAxes,
        fontsize=10,
        color="#333333",
    )
    out = output_dir / "04_dense_annotation_groundedsam_eye_roi_mask_box_stage1.png"
    _save_fig(fig, out)
    return out


def figure_05_missing_and_synthesized(output_dir: Path) -> Path:
    clean_img = _fit_image(
        REPO_ROOT
        / "workspace_session_samples_16/all48_consecutive16_dataset_construction_v1/previews/overlay_preview/clean__overlay_examples.png",
        (900, 520),
    )
    fail_img = _fit_image(
        REPO_ROOT
        / "workspace_session_samples_16/all48_consecutive16_dataset_construction_v1/previews/overlay_preview/fail_like__overlay_examples.png",
        (900, 520),
    )

    fig = plt.figure(figsize=(13.5, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    _panel_title(
        ax0,
        "Dense Annotation: Missing Label Data and Synthesized Data Annotation",
        "Top: conceptual supervisory branch; bottom: actual no-CSV completion previews from sampled construction.",
    )
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 3.5)
    ax0.axis("off")
    _draw_box(ax0, 0.4, 1.85, 1.7, 0.68, "Missing raw CSV rows", "#fde8e8")
    _draw_box(ax0, 2.55, 1.85, 1.7, 0.68, "TimeLens-XL synth frames", "#d8ebff")
    _draw_box(ax0, 4.7, 1.85, 1.9, 0.68, "Grounded-SAM pseudo labels", "#d9f4e8")
    _draw_box(ax0, 7.1, 1.85, 2.0, 0.68, "Merged canonical annotation", "#fff2cf")
    _draw_arrow(ax0, 2.14, 2.19, 2.5, 2.19)
    _draw_arrow(ax0, 4.28, 2.19, 4.65, 2.19)
    _draw_arrow(ax0, 6.64, 2.19, 7.05, 2.19)
    ax0.text(
        0.45,
        0.95,
        "Two complementary branches are kept explicit:\n"
        "1) no-CSV / missing-label completion from real frames\n"
        "2) synthetic-frame annotation after TimeLens-style interpolation\n"
        "Both feed the same geometry-stable canonical tuple with source and quality flags.",
        fontsize=10.5,
        color="#333333",
    )

    ax1.imshow(clean_img)
    ax1.axis("off")
    ax1.set_title("Actual clean completion examples", fontsize=12, loc="left")

    ax2.imshow(fail_img)
    ax2.axis("off")
    ax2.set_title("Actual fail-like / review examples", fontsize=12, loc="left")

    out = output_dir / "05_dense_annotation_missing_label_and_synthesized_annotation.png"
    _save_fig(fig, out)
    return out


def figure_06_overview_bundle(output_dir: Path, figure_paths: Iterable[Path]) -> Path:
    imgs = [_fit_image(path, (1000, 700)) for path in figure_paths]
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    titles = [
        "01 TimeLens-XL + Adaptive TimeBin",
        "02 V2E + Adaptive Event Counts",
        "03 Frame-Event Pairing",
        "04 Grounded-SAM ROI / Mask / Box",
        "05 Missing Label + Synth Annotation",
    ]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=11, loc="left")
    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.95,
        "Figure bundle summary",
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    axes[-1].text(
        0.02,
        0.8,
        "Time-Synchronous\n"
        "- Frame interpolation with adaptive time bins\n"
        "- Event interpolation / accumulation with adaptive counts\n\n"
        "Geometry-Stable\n"
        "- Frame-event pairing at one canonical index\n"
        "- Dense annotation with ROI / mask / box guidance\n"
        "- Missing-label and synthesized-data annotation branches",
        fontsize=11,
        va="top",
        color="#333333",
    )
    out = output_dir / "06_time_sync_geometry_stable_figure_bundle_overview.png"
    _save_fig(fig, out)
    return out


def write_manifest(output_dir: Path, figure_paths: list[Path]) -> Path:
    manifest = output_dir / "README.md"
    lines = [
        "# Time-Synchronous / Geometry-Stable Figure Set",
        "",
        "Generated figure bundle for:",
        "",
        "- Time-Synchronous",
        "  - Frame Interpolation (TimeLens-XL): Adaptive TimeBin",
        "  - Event Interpolation (V2E): Adaptive Event Counts",
        "- Geometry-Stable",
        "  - Frame-Event Pairing",
        "  - Dense Annotation (Grounded-SAM)",
        "    - Eye Region ROI Mask / Box Annotation: Stage1 train Pupil Box guide",
        "    - Missing Label Data and Synthesized Data Annotation",
        "",
        "Figures:",
        "",
    ]
    for path in figure_paths:
        lines.append(f"- `{path.name}`")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Every figure is exported as `PNG`, `SVG`, and `PDF`.")
    lines.append("- Figures 01/02/05-top are explanatory schematics and are mostly fully editable in `SVG/PDF`.")
    lines.append("- Figures 03/04/05-bottom use actual repository artifacts, so the layout/text remain editable but embedded example images stay raster.")
    lines.append("- Interval example for actual pairing visuals: `user01 / left / session_201 / interval 5012`.")
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    output_dir = _ensure_dir(OUTPUT_ROOT)
    figure_paths = [
        figure_01_frame_interpolation(output_dir),
        figure_02_event_interpolation(output_dir),
        figure_03_frame_event_pairing(output_dir),
        figure_04_dense_annotation_stage1(output_dir),
        figure_05_missing_and_synthesized(output_dir),
    ]
    figure_paths.append(figure_06_overview_bundle(output_dir, figure_paths))
    manifest = write_manifest(output_dir, figure_paths)
    for path in figure_paths + [manifest]:
        print(path)


if __name__ == "__main__":
    main()
