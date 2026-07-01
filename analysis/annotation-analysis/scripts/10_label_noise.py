"""
10_label_noise.py — Label Noise (& Uncertainty) vs human GT at anchors, for U-Net and GSAM2.

y_gsam2 uses a DETECTOR-FAILURE FALLBACK: where GSAM2 is a mislabel (or invalid), substitute
the U-Net center (per user request) so GSAM2 has a value on every anchor. Reports mean / median
/ p95 / p99 / std of ||y_orig - y_X||, overall and per motion (fixation / saccade / smooth).
Also: GT-vs-U-Net bias decomposition, 3-source uncertainty, and caveats. Saves results md.
"""
import os, csv, glob, json
import numpy as np

AA = "/home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis"
LAB = f"{AA}/samples/label"


def five(v):
    v = np.asarray(v, float)
    return (v.mean(), np.median(v), np.percentile(v, 95), np.percentile(v, 99),
            v.std(ddof=1), len(v))


def hdr():
    print(f"    {'source':22s} {'mean':>7s} {'median':>7s} {'p95':>7s} {'p99':>7s} {'std':>7s} {'n':>4s}")


def line(tag, v):
    m, md, p95, p99, sd, n = five(v)
    print(f"    {tag:22s} {m:7.3f} {md:7.3f} {p95:7.3f} {p99:7.3f} {sd:7.3f} {n:4d}")


motion = {}
for r in csv.DictReader(open(f"{AA}/samples/manifest_windows.csv")):
    motion[r["key"]] = r["motion"]

rows = []
n_fail = 0
for gj in sorted(glob.glob(f"{LAB}/*/gsam2.json")):
    key = os.path.basename(os.path.dirname(gj))
    gt = json.load(open(f"{LAB}/{key}/gt.json")); ai = gt["anchor_idx"]
    o = np.array(gt["ellipse_cx_cy_rx_ry_theta"][:2], float)
    unet = {c["idx"]: c for c in json.load(open(f"{LAB}/{key}/unet_dense.json"))["unet_centers"]}
    gs = {c["idx"]: c for c in json.load(open(gj))["gsam2_centers"]}
    u = unet.get(ai)
    if not (u and u.get("valid")):
        continue                                   # need U-Net (also our fallback) — none missing in practice
    uc = np.array([u["cx"], u["cy"]], float)
    g = gs.get(ai)
    fail = (g is None) or (not g.get("valid")) or g.get("mislabel", False)
    gc = uc if fail else np.array([g["cx"], g["cy"]], float)   # detector-failure -> U-Net fallback
    n_fail += int(fail)
    rows.append(dict(key=key, motion=motion[key],
                     e_unet=float(np.linalg.norm(uc - o)),
                     e_gsam2=float(np.linalg.norm(gc - o)),
                     e_gsam2_raw=(np.nan if fail else float(np.linalg.norm(gc - o))),
                     dx=float(uc[0]-o[0]), dy=float(uc[1]-o[1])))

eu = [r["e_unet"] for r in rows]; eg = [r["e_gsam2"] for r in rows]
print("="*70)
print(f"LABEL NOISE vs human GT  (anchors n={len(rows)}; GSAM2 detector-failures→U-Net: {n_fail})")
print("="*70)
print("\n[ OVERALL ]"); hdr()
line("GT vs U-Net", eu)
line("GT vs GSAM2 (fallback)", eg)

for m in ("fixation", "saccade", "smooth_pursuit"):
    print(f"\n[ {m} ]"); hdr()
    line("GT vs U-Net", [r["e_unet"] for r in rows if r["motion"] == m])
    line("GT vs GSAM2 (fallback)", [r["e_gsam2"] for r in rows if r["motion"] == m])

# extras
dx = np.array([r["dx"] for r in rows]); dy = np.array([r["dy"] for r in rows])
raw = np.array([r["e_gsam2_raw"] for r in rows]); raw = raw[~np.isnan(raw)]
print("\n[ notes ]")
print(f"  GT-vs-U-Net systematic bias Δ=({dx.mean():+.3f},{dy.mean():+.3f}) |Δ|={np.hypot(dx.mean(),dy.mean()):.3f}px "
      f"(most of the U-Net 'noise' is this offset).")
print(f"  GT-vs-GSAM2 RAW (non-failure only, n={len(raw)}): mean {raw.mean():.3f} median {np.median(raw):.3f} "
      f"p95 {np.percentile(raw,95):.3f} p99 {np.percentile(raw,99):.3f} std {raw.std(ddof=1):.3f}")
print("  caveat: U-Net trained on human masks (not independent); GSAM2 independent. "
      "y_unet=official Data_davis_predict proxy. std=sample(ddof=1).")

# save md
def row_md(tag, v):
    m, md, p95, p99, sd, n = five(v)
    return f"| {tag} | {m:.3f} | {md:.3f} | {p95:.3f} | {p99:.3f} | {sd:.3f} | {n} |"

os.makedirs(f"{AA}/results", exist_ok=True)
with open(f"{AA}/results/label_noise_gt_unet_gsam2.md", "w", encoding="utf-8") as f:
    f.write("# Label Noise vs human GT — U-Net & GSAM2 (detector-failure→U-Net fallback)\n\n")
    f.write(f"anchors n={len(rows)}. GSAM2 detector-failure(mislabel/invalid)→U-Net fallback: {n_fail}건. "
            f"단위 px(346×260), std=sample(ddof=1).\n\n")
    f.write("## OVERALL\n| source | mean | median | p95 | p99 | std | n |\n|---|---|---|---|---|---|---|\n")
    f.write(row_md("GT vs U-Net", eu) + "\n" + row_md("GT vs GSAM2 (fallback)", eg) + "\n")
    for m in ("fixation", "saccade", "smooth_pursuit"):
        f.write(f"\n## {m}\n| source | mean | median | p95 | p99 | std | n |\n|---|---|---|---|---|---|---|\n")
        f.write(row_md("GT vs U-Net", [r["e_unet"] for r in rows if r["motion"] == m]) + "\n")
        f.write(row_md("GT vs GSAM2 (fallback)", [r["e_gsam2"] for r in rows if r["motion"] == m]) + "\n")
    f.write(f"\n## notes\n- GT-vs-U-Net bias Δ=({dx.mean():+.3f},{dy.mean():+.3f}) |Δ|={np.hypot(dx.mean(),dy.mean()):.3f}px "
            f"(U-Net 노이즈 대부분이 계통 offset).\n")
    f.write(f"- GT-vs-GSAM2 RAW(non-failure, n={len(raw)}): median {np.median(raw):.3f} p95 {np.percentile(raw,95):.3f} std {raw.std(ddof=1):.3f}.\n")
    f.write("- U-Net은 사람 라벨로 학습(비독립), GSAM2는 독립. y_unet=공식 Data_davis_predict proxy.\n")
print(f"\n[saved] results/label_noise_gt_unet_gsam2.md")
