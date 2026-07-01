"""10b_shape_metrics.py — SHAPE agreement vs human GT at anchors (complements 10, which is
center-only). Center is blind to pupil/iris confusion because pupil and iris are concentric
(docs/14); this adds size + overlap metrics that DO see it:

  radius_ratio = r_equiv(det) / r_equiv(GT)   [r_equiv=sqrt(area/pi); GT uses sqrt(rx*ry)]
                 ~1.0 = pupil; >1.5 = iris/over-seg suspect
  mask IoU     = |det_mask ∩ GT_ellipse| / |det_mask ∪ GT_ellipse|

For GT-vs-U-Net and GT-vs-GSAM2 (GSAM2 mislabel-flagged frames excluded). Overall + per
motion. Proves the ROI+geom iris fix held (expect 0 iris-suspect, high IoU).
Saves results/label_shape_gt_unet_gsam2.md.
"""
import os, glob, json, csv, math
import numpy as np
import cv2
from PIL import Image

AA = "/home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis"
LAB = f"{AA}/samples/label"


def five(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return (float("nan"),) * 5 + (0,)
    return (v.mean(), np.median(v), np.percentile(v, 95), np.percentile(v, 99),
            v.std(ddof=1) if len(v) > 1 else 0.0, len(v))


def gt_ellipse_mask(cx, cy, rx, ry, theta, shape=(260, 346)):
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                (max(int(round(rx)), 1), max(int(round(ry)), 1)),
                math.degrees(theta), 0, 360, 255, -1)
    return m > 0


def read_mask(path):
    if path.lower().endswith(".gif"):
        return np.array(Image.open(path).convert("L")) > 0
    return cv2.imread(path, 0) > 0


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else float("nan")


motion = {r["key"]: r["motion"] for r in csv.DictReader(open(f"{AA}/samples/manifest_windows.csv"))}

rows = []
for gj in sorted(glob.glob(f"{LAB}/*/gsam2.json")):
    key = os.path.basename(os.path.dirname(gj))
    gt = json.load(open(f"{LAB}/{key}/gt.json"))
    ai = gt["anchor_idx"]
    cx, cy, rx, ry, th = gt["ellipse_cx_cy_rx_ry_theta"]
    gr = math.sqrt(max(rx * ry, 1e-6))
    gmask = gt_ellipse_mask(cx, cy, rx, ry, th)

    unet = {c["idx"]: c for c in json.load(open(f"{LAB}/{key}/unet_dense.json"))["unet_centers"]}
    gs = {c["idx"]: c for c in json.load(open(gj))["gsam2_centers"]}
    u, g = unet.get(ai), gs.get(ai)

    rec = dict(key=key, motion=motion[key])
    # U-Net
    if u and u.get("valid") and u.get("area"):
        rec["rr_unet"] = math.sqrt(u["area"] / math.pi) / gr
        mp = glob.glob(f"{LAB}/{key}/unet_masks/{ai:06d}_*_mask.gif")
        rec["iou_unet"] = iou(read_mask(mp[0]), gmask) if mp else float("nan")
    # GSAM2 (skip mislabel/invalid)
    if g and g.get("valid") and not g.get("mislabel", False) and g.get("area"):
        rec["rr_gsam2"] = math.sqrt(g["area"] / math.pi) / gr
        mp = glob.glob(f"{LAB}/{key}/gsam2_masks/{ai:06d}_*_mask.png")
        rec["iou_gsam2"] = iou(read_mask(mp[0]), gmask) if mp else float("nan")
    rows.append(rec)


def col(name, mot=None):
    return [r[name] for r in rows if name in r and not (mot and r["motion"] != mot)
            and not math.isnan(r[name])]


def hdr():
    print(f"    {'metric':24s} {'mean':>7s} {'median':>7s} {'p95':>7s} {'p99':>7s} {'std':>7s} {'n':>4s}")


def line(tag, v):
    m, md, p95, p99, sd, n = five(v)
    print(f"    {tag:24s} {m:7.3f} {md:7.3f} {p95:7.3f} {p99:7.3f} {sd:7.3f} {n:4d}")


print("=" * 74)
print("SHAPE metrics vs human GT (anchors)  — radius_ratio (~1=pupil, >1.5=iris) & mask IoU")
print("=" * 74)
print("\n[ OVERALL ]"); hdr()
line("radius_ratio U-Net", col("rr_unet"))
line("radius_ratio GSAM2", col("rr_gsam2"))
line("mask IoU U-Net", col("iou_unet"))
line("mask IoU GSAM2", col("iou_gsam2"))
for src in ("rr_unet", "rr_gsam2"):
    v = np.asarray(col(src)); n_iris = int((v > 1.5).sum())
    print(f"    iris-suspect(>1.5) {src:12s}: {n_iris}/{len(v)}   (max ratio {v.max():.2f})")

for m in ("fixation", "saccade", "smooth_pursuit"):
    print(f"\n[ {m} ]"); hdr()
    line("radius_ratio U-Net", col("rr_unet", m))
    line("radius_ratio GSAM2", col("rr_gsam2", m))
    line("mask IoU U-Net", col("iou_unet", m))
    line("mask IoU GSAM2", col("iou_gsam2", m))


def row_md(tag, v):
    a, b, c, d, e, n = five(v)
    return f"| {tag} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} | {e:.3f} | {n} |"


os.makedirs(f"{AA}/results", exist_ok=True)
with open(f"{AA}/results/label_shape_gt_unet_gsam2.md", "w", encoding="utf-8") as f:
    f.write("# Shape metrics vs human GT — radius_ratio & mask IoU (anchors)\n\n")
    f.write("radius_ratio = r_equiv(det)/r_equiv(GT), ~1.0=pupil, **>1.5 = iris/over-seg suspect** "
            "(center-blind; docs/14). IoU = det_mask ∩ GT-ellipse / ∪. GSAM2 mislabel-flagged excluded.\n\n")
    f.write("## OVERALL\n| metric | mean | median | p95 | p99 | std | n |\n|---|---|---|---|---|---|---|\n")
    for tag, cn in [("radius_ratio U-Net", "rr_unet"), ("radius_ratio GSAM2", "rr_gsam2"),
                    ("mask IoU U-Net", "iou_unet"), ("mask IoU GSAM2", "iou_gsam2")]:
        f.write(row_md(tag, col(cn)) + "\n")
    for src, nm in [("rr_unet", "U-Net"), ("rr_gsam2", "GSAM2")]:
        v = np.asarray(col(src))
        f.write(f"\n- **iris-suspect(ratio>1.5) {nm}: {int((v>1.5).sum())}/{len(v)}** (max {v.max():.2f})\n")
    for m in ("fixation", "saccade", "smooth_pursuit"):
        f.write(f"\n## {m}\n| metric | mean | median | p95 | p99 | std | n |\n|---|---|---|---|---|---|---|\n")
        for tag, cn in [("radius_ratio U-Net", "rr_unet"), ("radius_ratio GSAM2", "rr_gsam2"),
                        ("mask IoU U-Net", "iou_unet"), ("mask IoU GSAM2", "iou_gsam2")]:
            f.write(row_md(tag, col(cn, m)) + "\n")
    f.write("\n## notes\n- ROI+geom harness(marker fix)로 GSAM2 홍채혼동 해소 확인용. GT 타원 rasterize로 IoU.\n")
    f.write("- U-Net mask=공식 predict .gif, GSAM2 mask=ROI+geom .png(0/255).\n")
print("\n[saved] results/label_shape_gt_unet_gsam2.md")
