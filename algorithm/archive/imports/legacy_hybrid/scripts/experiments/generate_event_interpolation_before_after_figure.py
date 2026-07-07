from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "workspace_figures" / "event_interpolation_before_after"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_all(fig: plt.Figure, path: Path) -> None:
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_and_fit(path: str | Path, max_size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def _box(ax, x, y, w, h, text, color):
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=10)


def _arrow(ax, x0, y0, x1, y1, color="#444444"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=color, shrinkA=0, shrinkB=0),
    )


def make_main_figure(output_dir: Path) -> Path:
    raw_async = _load_and_fit(
        REPO_ROOT / "workspace_event_voxel_analysis/axis_swapped_views/user01_session201_left_interval_5012_time_x_y.png",
        (1200, 800),
    )
    frame_sync = _load_and_fit(
        REPO_ROOT / "workspace_event_voxel_analysis/interval_5012_accumulated_channels/contact_sheet.png",
        (1200, 800),
    )

    fig = plt.figure(figsize=(16, 6.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.15, 0.85], width_ratios=[1.15, 0.9, 1.15], hspace=0.1, wspace=0.2)

    ax_title = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_mid = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2])

    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.72,
        "Event Interpolation Before / After",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax_title.text(
        0.0,
        0.2,
        "Representative example: user01 / left / session_201 / interval 5012",
        fontsize=11,
        color="#555555",
        ha="left",
        va="center",
    )

    ax_left.imshow(raw_async)
    ax_left.axis("off")
    ax_left.set_title("Before: Asynchronous raw events (time, x, y)", fontsize=13, loc="left", pad=10)
    ax_left.text(
        0.02,
        -0.08,
        "Sparse and bursty points arrive off-grid.\nNo direct frame-synchronous supervision index yet.",
        transform=ax_left.transAxes,
        fontsize=10,
        color="#333333",
        va="top",
    )

    ax_mid.set_xlim(0, 10)
    ax_mid.set_ylim(0, 6)
    ax_mid.axis("off")
    ax_mid.set_title("Adaptive windowing + target anchor τ", fontsize=13, loc="left", pad=10)

    ax_mid.plot([0.8, 9.2], [1.45, 1.45], color="#444444", linewidth=1.8)
    for x, label in [(1.2, "tprev"), (4.9, "τ"), (8.5, "tnext")]:
        ax_mid.add_patch(patches.Circle((x, 1.45), 0.09, color="#111111"))
        ax_mid.text(x, 1.0, label, ha="center", va="top", fontsize=10)
    ax_mid.add_patch(
        patches.Rectangle((3.1, 0.8), 3.6, 1.3, fill=False, edgecolor="#10a37f", linewidth=2.0, linestyle="--")
    )
    ax_mid.text(4.9, 2.28, "adaptive event-count window", ha="center", va="bottom", fontsize=10, color="#106f57")
    ax_mid.text(4.9, 0.38, "select / resample events around τ", ha="center", va="top", fontsize=10, color="#333333")

    _box(ax_mid, 0.9, 3.35, 2.0, 0.7, "raw async events", "#f2f4f7")
    _box(ax_mid, 3.45, 3.35, 2.05, 0.7, "adaptive counts", "#d9f4e8")
    _box(ax_mid, 6.05, 3.35, 2.15, 0.7, "target timestamp τ", "#fff2cf")
    _arrow(ax_mid, 2.95, 3.7, 3.38, 3.7)
    _arrow(ax_mid, 5.57, 3.7, 5.98, 3.7)

    ax_mid.text(
        0.92,
        5.1,
        "Interpretation:\n"
        "- keep the canonical target timestamp fixed\n"
        "- adapt the event horizon or count budget\n"
        "- build one aligned event tensor for the same supervision anchor",
        fontsize=10.3,
        color="#333333",
        va="top",
    )

    ax_right.imshow(frame_sync)
    ax_right.axis("off")
    ax_right.set_title("After: Frame-synchronous event representation", fontsize=13, loc="left", pad=10)
    ax_right.text(
        0.02,
        -0.08,
        "Polarity-split accumulation maps now correspond to the same target index τ.\nThis is the representation paired with the frame and geometry labels.",
        transform=ax_right.transAxes,
        fontsize=10,
        color="#333333",
        va="top",
    )

    out = output_dir / "01_event_interpolation_before_after_main.png"
    _save_all(fig, out)
    return out


def make_pairing_figure(output_dir: Path) -> Path:
    frame = _load_and_fit(
        REPO_ROOT / "workspace_event_voxel_analysis/interval_5012_frame_eye_pupil_mask_overlay_blue/user01_left_session201_interval5012_frame_eye_pupil_mask_overlay_blue.png",
        (1000, 700),
    )
    event = _load_and_fit(
        REPO_ROOT / "workspace_event_voxel_analysis/interval_5012_accumulated_channels/user01_session201_left_interval_5012_positive_accum.png",
        (1000, 700),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    axes[0].imshow(frame)
    axes[0].axis("off")
    axes[0].set_title("Aligned frame / annotation anchor", fontsize=13, loc="left", pad=10)
    axes[0].text(
        0.02,
        -0.08,
        "Eye ROI, pupil box, and mask-style geometry all refer to anchor τ.",
        transform=axes[0].transAxes,
        fontsize=10,
        color="#333333",
        va="top",
    )

    axes[1].imshow(event)
    axes[1].axis("off")
    axes[1].set_title("Aligned event tensor at the same τ", fontsize=13, loc="left", pad=10)
    axes[1].text(
        0.02,
        -0.08,
        "After interpolation / resampling, frame and event can be supervised against one geometry-stable target.",
        transform=axes[1].transAxes,
        fontsize=10,
        color="#333333",
        va="top",
    )

    fig.suptitle("After interpolation: one canonical supervisory index", fontsize=17, fontweight="bold", x=0.05, y=0.98, ha="left")
    out = output_dir / "02_event_interpolation_after_pairing.png"
    _save_all(fig, out)
    return out


def write_manifest(output_dir: Path, outputs: list[Path]) -> Path:
    manifest = output_dir / "README.md"
    lines = [
        "# Event Interpolation Before / After Figures",
        "",
        "Outputs are exported as `PNG`, `SVG`, and `PDF`.",
        "",
        "Files:",
        "",
    ]
    for path in outputs:
        lines.append(f"- `{path.name}`")
    lines += [
        "",
        "Notes:",
        "- `01` shows the recommended before/after narrative:",
        "  - left: raw async event points",
        "  - middle: adaptive window + target anchor",
        "  - right: frame-synchronous event tensor",
        "- `02` shows how the post-interpolation event tensor pairs with the annotated frame target.",
        "- Actual example assets are from `user01 / left / session_201 / interval 5012`.",
    ]
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    output_dir = _ensure_dir(OUTPUT_ROOT)
    outputs = [make_main_figure(output_dir), make_pairing_figure(output_dir)]
    manifest = write_manifest(output_dir, outputs)
    for path in outputs + [manifest]:
        print(path)


if __name__ == "__main__":
    main()
