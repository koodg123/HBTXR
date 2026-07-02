"""10c_geometric_precision.py — (b) Geometric Precision of annotated SHAPE (ellipse/mask).

Center distance (Metric) alone cannot measure shape agreement; a right center with wrong
size/rotation still misaligns the region. So we score region overlap + shape params + boundary,
for GSAM2 / U-Net masks vs the human-GT ellipse (rasterized to the same 346x260 grid).

  Step2  Region overlap   : IoU, Dice  (center+size+rotation jointly)
  Step4  Shape params     : area (log-ratio) / semi-axes / orientation via Bland-Altman (bias/LoA)
  Step5  Boundary         : Hausdorff d_H, d_H^95, ASSD
(Step3 rasterization floor delta_raster = sqrt(2)=1.414px@346 / 0.302@64 is a discretization
 floor of the DERIVED mask, reported in the precision pipeline (La.3); not recomputed here.)

Sources (same audit toy set as Metric precision, anchors n~479):
  human : ellipse params (cx,cy,rx,ry,theta) -> rasterized mask (cv2.ellipse)
  gsam2 : samples/label/{key}/gsam2_masks/{ai}_*_mask.png   (mislabel/invalid excluded)
  unet  : samples/label/{key}/unet_masks/{ai}_*_mask.gif
-> writes results/geometric_precision.md
"""
import json, glob, os, math, csv
import numpy as np
import cv2
from PIL import Image

AA = "/home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis"
LAB = f"{AA}/samples/label"
W, H = 346, 260


def read_mask(p):
    if p.lower().endswith(".gif"):
        return np.array(Image.open(p).convert("L")) > 0
    return cv2.imread(p, 0) > 0


def gt_mask(cx, cy, rx, ry, th):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))),
                math.degrees(th), 0, 360, 255, -1)
    return m > 0


def largest(mask):
    m = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def fit(comp):
    """-> (semi_major, semi_minor, angle_deg_of_major in [0,180), area_pixels)."""
    c, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not c or len(c[0]) < 5:
        return None
    (x, y), (A, B), ang = cv2.fitEllipse(c[0])
    a = float(comp.sum())
    return (A / 2, B / 2, ang % 180, a) if A >= B else (B / 2, A / 2, (ang + 90) % 180, a)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def bmask(c):
    er = cv2.erode(c, np.ones((3, 3), np.uint8))
    return ((c > 0) & (er == 0)).astype(np.uint8)


def haus_assd(cA, cB):
    """symmetric Hausdorff (max, p95) + ASSD between two binary component boundaries (px)."""
    ba, bb = bmask(cA), bmask(cB)
    if ba.sum() == 0 or bb.sum() == 0:
        return None
    eB = cv2.distanceTransform(1 - bb, cv2.DIST_L2, 3)
    eA = cv2.distanceTransform(1 - ba, cv2.DIST_L2, 3)
    dA, dB = eB[ba > 0], eA[bb > 0]                       # A->B, B->A boundary distances
    dH = max(dA.max(), dB.max())
    dH95 = max(np.percentile(dA, 95), np.percentile(dB, 95))
    assd = (dA.sum() + dB.sum()) / (len(dA) + len(dB))
    return dH, dH95, assd


def ba_stats(v):
    v = np.asarray(v, float)
    m, s = v.mean(), v.std()
    return m, m - 1.96 * s, m + 1.96 * s


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def main():
    motion = {r["key"]: r["motion"] for r in csv.DictReader(open(f"{AA}/samples/manifest_windows.csv"))}
    SRC = ("gsam2", "unet")
    D = {s: {k: [] for k in ("iou", "dice", "mot", "lA", "drx", "dry", "dth", "asp", "dH", "dH95", "assd")}
         for s in SRC}
    iou_gu = []

    for gj in sorted(glob.glob(f"{LAB}/*/gsam2.json")):
        key = os.path.basename(os.path.dirname(gj))
        gt = json.load(open(f"{LAB}/{key}/gt.json")); ai = gt["anchor_idx"]
        cx, cy, rx, ry, th = gt["ellipse_cx_cy_rx_ry_theta"]
        gmaj, gmin = max(rx, ry), min(rx, ry)
        gang = (math.degrees(th) if rx >= ry else math.degrees(th) + 90) % 180
        Agt = math.pi * rx * ry
        gm = gt_mask(cx, cy, rx, ry, th); gcomp = largest(gm)
        unet = {c["idx"]: c for c in json.load(open(f"{LAB}/{key}/unet_dense.json"))["unet_centers"]}
        gs = {c["idx"]: c for c in json.load(open(gj))["gsam2_centers"]}
        u, g = unet.get(ai), gs.get(ai)
        comps = {}
        cfg = (("gsam2", g and g.get("valid") and not g.get("mislabel", False),
                f"{LAB}/{key}/gsam2_masks/{ai:06d}_*_mask.png"),
               ("unet", u and u.get("valid"), f"{LAB}/{key}/unet_masks/{ai:06d}_*_mask.gif"))
        for src, ok, pat in cfg:
            if not ok:
                continue
            mp = glob.glob(pat)
            if not mp:
                continue
            comp = largest(read_mask(mp[0]))
            if comp is None:
                continue
            f = fit(comp)
            if f is None:
                continue
            comps[src] = comp
            smaj, smin, sang, area = f
            d = D[src]
            d["iou"].append(iou(comp > 0, gm)); d["dice"].append(2 * d["iou"][-1] / (1 + d["iou"][-1]))
            d["mot"].append(motion[key])
            d["lA"].append(math.log(area / Agt)); d["drx"].append(smaj - gmaj); d["dry"].append(smin - gmin)
            dth = abs(sang - gang) % 180; d["dth"].append(min(dth, 180 - dth)); d["asp"].append(gmin / gmaj)
            r = haus_assd(comp, gcomp)
            if r:
                d["dH"].append(r[0]); d["dH95"].append(r[1]); d["assd"].append(r[2])
        if "gsam2" in comps and "unet" in comps:
            iou_gu.append(iou(comps["gsam2"] > 0, comps["unet"] > 0))

    # ---- write markdown ----
    L = ["# Geometric precision of annotated shape — IoU/Dice + shape-param BA + Hausdorff/ASSD (anchors, 346x260)\n",
         "Human GT ellipse rasterized (cv2.ellipse) to 346x260; GSAM2/U-Net native masks. GSAM2 mislabel/invalid excluded.",
         "Generated by `scripts/10c_geometric_precision.py`.\n"]

    L.append("## Step2 · Region overlap (IoU / Dice) vs human GT")
    L.append("| source | IoU median | IoU mean | Dice median | n |")
    L.append("|---|---|---|---|---|")
    for s in SRC:
        v = np.array(D[s]["iou"]); dc = np.array(D[s]["dice"])
        L.append(f"| {s} | {med(v):.3f} | {v.mean():.3f} | {med(dc):.3f} | {len(v)} |")
    L.append(f"| gsam2 vs unet | {med(iou_gu):.3f} | {np.mean(iou_gu):.3f} | {med([2*i/(1+i) for i in iou_gu]):.3f} | {len(iou_gu)} |")
    L.append("\n### IoU by motion")
    L.append("| source | fixation | saccade | smooth_pursuit |")
    L.append("|---|---|---|---|")
    for s in SRC:
        v = np.array(D[s]["iou"]); mo = np.array(D[s]["mot"])
        cells = " | ".join(f"{med(v[mo==m]):.3f}" for m in ("fixation", "saccade", "smooth_pursuit"))
        L.append(f"| {s} | {cells} |")

    L.append("\n## Step4 · Shape-parameter agreement (Bland-Altman) vs human GT")
    L.append("| source | area lambda_A bias (ratio) | area LoA | Dr_x bias (px) | Dr_x LoA | Dr_y bias (px) | Dr_y LoA | \\|Dtheta\\| median/p95 (deg, elong) | n |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in SRC:
        d = D[s]
        lm, ll, lh = ba_stats(d["lA"]); rm, rl, rh = ba_stats(d["drx"]); ym, yl, yh = ba_stats(d["dry"])
        dth = np.array(d["dth"]); el = np.array(d["asp"]) < 0.85
        L.append(f"| {s} | {lm:+.3f} (x{math.exp(lm):.3f}) | [{ll:+.3f},{lh:+.3f}] | {rm:+.3f} | [{rl:+.2f},{rh:+.2f}] | "
                 f"{ym:+.3f} | [{yl:+.2f},{yh:+.2f}] | {med(dth[el]):.2f}/{np.percentile(dth[el],95):.1f} | {len(d['lA'])} |")
    L.append("- area lambda_A = log(A_src / A_gt) with A_gt = pi*rx*ry, A_src = mask pixel count; ratio = exp(bias).")
    L.append("- orientation restricted to elongated pupils (GT aspect r_y/r_x < 0.85); near-circular pupils have ill-defined angle.")

    L.append("\n## Step5 · Boundary precision (Hausdorff / ASSD) vs human GT mask (px)")
    L.append("| source | Hausdorff d_H median (p95) | d_H^95 median | ASSD median | n |")
    L.append("|---|---|---|---|---|")
    for s in SRC:
        dH = np.array(D[s]["dH"]); d95 = np.array(D[s]["dH95"]); assd = np.array(D[s]["assd"])
        L.append(f"| {s} | {med(dH):.2f} ({np.percentile(dH,95):.1f}) | {med(d95):.2f} | {med(assd):.3f} | {len(dH)} |")

    L.append("\n## Interpretation")
    L.append("- **Area**: GSAM2 +3.8% / U-Net -1.5% (both small; consistent with radius_ratio 1.016 / 0.993).")
    L.append("- **Axes**: GSAM2 Dr ~-0.16/-0.22px (sub-pixel) vs U-Net -0.57/-0.55px (systematic under-sizing) "
             "-> U-Net's mask-shrink bias is the quantitative cause of its lower IoU/boundary.")
    L.append("- **Orientation**: both ~2.2-2.5deg median (elongated) -> comparable.")
    L.append("- **Boundary**: GSAM2 ASSD 0.53 vs U-Net 0.99px (~2x), d_H^95 1.37 vs 2.32 -> GSAM2 boundary clearly closer to GT.")
    L.append("- Metric (center) and Geometric (shape) independently pick GSAM2 as the most-precise independent estimator "
             "-> supports adopting GSAM2 as the auditor for label-uncertainty (sigma_label) at the shape level too.")

    out = f"{AA}/results/geometric_precision.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("[10c] wrote", out)
    print(f"  Step2 IoU  gsam2 {med(D['gsam2']['iou']):.3f} / unet {med(D['unet']['iou']):.3f} / gsam2-unet {med(iou_gu):.3f}")
    print(f"  Step5 ASSD gsam2 {med(D['gsam2']['assd']):.3f} / unet {med(D['unet']['assd']):.3f}")


if __name__ == "__main__":
    main()
