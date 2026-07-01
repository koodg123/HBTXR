"""
13_gsam2_mislabel.py — GSAM2 mislabeling analysis over the FULL per-frame run.

(1) ratio: mislabel fraction, gold (anchor vs human GT) + proxy (all frames vs U-Net dense).
(2) cause: characterize mislabels (det_score, mask area, y-position, box size) vs good;
    categorize failure modes; per-motion / per-user breakdown.
(3) solutions: evaluate candidate reject-gates (U-Net cross-check / y-position / det / area /
    combined) on the anchor gold set — how many mislabels removed vs good wrongly dropped.

Reads samples/label/{key}/{gsam2,unet_dense,gt}.json. Anchors give human GT; all frames
use U-Net dense center as an audit-vs-audit cross-check (both far from pupil => gross fail).
"""
import os, csv, glob, json, argparse
import numpy as np

AA = "/home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis"


def d(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) if a and b else np.nan


def pct(mask):
    m = np.asarray(mask, bool)
    return 100.0 * m.sum() / max(1, m.size)


def stat(v):
    v = np.asarray([x for x in v if not np.isnan(x)], float)
    if not len(v):
        return "n/a"
    return (f"n={len(v):4d} med={np.median(v):6.2f} mean={np.mean(v):6.2f} "
            f"p90={np.percentile(v,90):6.2f} p95={np.percentile(v,95):6.2f} max={v.max():6.2f}")


def main():
    LAB = f"{AA}/samples/label"
    motion, user = {}, {}
    for r in csv.DictReader(open(f"{AA}/samples/manifest_windows.csv")):
        motion[r["key"]] = r["motion"]; user[r["key"]] = r["user"]

    rows = []  # per frame
    for gj in sorted(glob.glob(f"{LAB}/*/gsam2.json")):
        key = os.path.basename(os.path.dirname(gj))
        g = json.load(open(gj))
        unet = {c["idx"]: c for c in json.load(open(f"{LAB}/{key}/unet_dense.json"))["unet_centers"]}
        gt = json.load(open(f"{LAB}/{key}/gt.json")); ai = gt["anchor_idx"]
        oc = gt["ellipse_cx_cy_rx_ry_theta"][:2]
        for c in g["gsam2_centers"]:
            if not c.get("valid"):
                continue
            i = c["idx"]
            gc = [c["cx"], c["cy"]]
            uc = None
            u = unet.get(i)
            if u and u.get("valid"):
                uc = [u["cx"], u["cy"]]
            bx = c.get("box", [0, 0, 0, 0])
            rows.append(dict(
                key=key, idx=i, motion=motion[key], user=user[key],
                is_anchor=(i == ai),
                e_orig=(d(gc, oc) if i == ai else np.nan),
                e_unet=(d(gc, uc) if uc else np.nan),
                cx=c["cx"], cy=c["cy"], area=c.get("area", 0), det=c.get("det_score", 0),
                bw=bx[2] - bx[0], bh=bx[3] - bx[1]))

    A = [r for r in rows if r["is_anchor"]]        # gold set
    print(f"[data] valid gsam2 frames={len(rows)}  anchors(gold)={len(A)}\n")

    # ---- (1) RATIO ----
    print("="*70, "\n(1) MISLABEL RATIO\n" + "="*70)
    print("Gold (anchor, ||gsam2 - human GT||):")
    eo = np.array([r["e_orig"] for r in A])
    for t in (2, 3, 5, 10, 30):
        print(f"   e_orig > {t:2d}px : {(eo>t).sum():3d}/{len(A)}  ({pct(eo>t):.2f}%)")
    print("Proxy (ALL frames, ||gsam2 - U-Net dense||):")
    eu = np.array([r["e_unet"] for r in rows]); euv = eu[~np.isnan(eu)]
    for t in (5, 10, 15, 30):
        print(f"   e_unet > {t:2d}px : {(euv>t).sum():3d}/{len(euv)}  ({pct(euv>t):.2f}%)")
    # validate proxy vs gold on anchors
    ea_u = np.array([r["e_unet"] for r in A]); ok = ~np.isnan(ea_u)
    gold_bad = eo[ok] > 10; proxy_bad = ea_u[ok] > 15
    tp = (gold_bad & proxy_bad).sum(); fp = (~gold_bad & proxy_bad).sum(); fn = (gold_bad & ~proxy_bad).sum()
    print(f"proxy(e_unet>15) vs gold(e_orig>10) on anchors: TP={tp} FP={fp} FN={fn} "
          f"(recall {tp/max(1,tp+fn):.2f}, precision {tp/max(1,tp+fp):.2f})")

    # ---- (2) CAUSE ----
    print("\n" + "="*70, "\n(2) CAUSE — mislabel(e_orig>10) vs good on anchors\n" + "="*70)
    bad = [r for r in A if r["e_orig"] > 10]; good = [r for r in A if r["e_orig"] <= 10]
    for name, S in (("MISLABEL", bad), ("GOOD", good)):
        det = np.array([r["det"] for r in S]); ar = np.array([r["area"] for r in S])
        cy = np.array([r["cy"] for r in S]); bw = np.array([r["bw"] for r in S])
        print(f"  {name:9s} n={len(S):3d} | det med={np.median(det):.3f} | area med={np.median(ar):5.0f} "
              f"| cy med={np.median(cy):5.1f} | box_w med={np.median(bw):5.1f}")
    print("  failure-mode heuristics on mislabels:")
    bottom = sum(1 for r in bad if r["cy"] > 200)
    tiny = sum(1 for r in bad if r["area"] < 400)
    lowdet = sum(1 for r in bad if r["det"] < 0.35)
    print(f"    bottom(cy>200): {bottom}/{len(bad)}  tiny(area<400): {tiny}/{len(bad)}  lowdet(det<0.35): {lowdet}/{len(bad)}")
    print("  per-motion mislabel rate (anchor, e_orig>10):")
    for m in ("fixation", "saccade", "smooth_pursuit"):
        Sm = [r for r in A if r["motion"] == m]; b = sum(1 for r in Sm if r["e_orig"] > 10)
        print(f"    {m:14s} {b:2d}/{len(Sm)} ({100*b/max(1,len(Sm)):.1f}%)")
    print("  per-user mislabel count (anchor, e_orig>10), top 6:")
    from collections import Counter
    cu = Counter(r["user"] for r in bad)
    for u, n in cu.most_common(6):
        print(f"    {u:8s} {n}")

    # ---- (3) SOLUTIONS: gate evaluation on gold ----
    print("\n" + "="*70, "\n(3) SOLUTIONS — reject-gate eval on anchor gold (mislabel=e_orig>10)\n" + "="*70)
    Aok = [r for r in A if not np.isnan(r["e_unet"])]  # need unet for cross-check gate
    goldbad = np.array([r["e_orig"] > 10 for r in Aok])
    gates = {
        "U-Net xcheck e_unet>15": np.array([r["e_unet"] > 15 for r in Aok]),
        "y-pos cy>200":           np.array([r["cy"] > 200 for r in Aok]),
        "det<0.35":               np.array([r["det"] < 0.35 for r in Aok]),
        "area<400":               np.array([r["area"] < 400 for r in Aok]),
        "det<0.35 & area<400":    np.array([(r["det"] < 0.35 and r["area"] < 400) for r in Aok]),
        "xcheck OR (det<.35&tiny)":np.array([(r["e_unet"] > 15 or (r["det"] < 0.35 and r["area"] < 400)) for r in Aok]),
    }
    nbad = goldbad.sum(); ngood = (~goldbad).sum()
    print(f"  (anchors with U-Net: {len(Aok)}; mislabels={nbad}, good={ngood})")
    print(f"  {'gate':28s} {'removes_bad':>11s} {'drops_good':>10s}")
    for name, rej in gates.items():
        rb = int((rej & goldbad).sum()); dg = int((rej & ~goldbad).sum())
        print(f"  {name:28s} {rb:3d}/{nbad:<3d}    {dg:3d}/{ngood:<3d}")


if __name__ == "__main__":
    main()
