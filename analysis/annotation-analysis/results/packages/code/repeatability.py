"""repeatability.py (Layer A — precision lower bound), 346x260 px.

La.1  GSAM2 perturbation std from gsam2.json.repeats (axis + radial)   = sigma_s_rep
La.2  Fixation frame-to-frame (F2F) center jitter, per source
        unet/gsam2 = automatic-method stability;  pred = MODEL temporal jitter
        (reported separately, NOT a GT-precision proxy); human = SKIP (sparse labels)
        -> per user decision, improved GSAM2 is the repeatability proxy.
La.3  Representation floor: human ellipse-center vs human mask-centroid. Mask is
        DERIVED (rasterized) from the ellipse -> this is a RASTERIZATION floor, not
        precision (a constant ~(-1,-1)px generation offset dominates; IoU~0.9).
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S


def la1_repeats():
    sx, sy, rad = [], [], []
    for a in S.anchors():
        if not a["repeats"] or len(a["repeats"]) < 2:
            continue
        p = np.array(a["repeats"], float)
        sx.append(float(p[:, 0].std(ddof=1)))
        sy.append(float(p[:, 1].std(ddof=1)))
        c = p.mean(0)
        rad.append(float(np.sqrt(((p - c) ** 2).sum(1)).mean()))
    return dict(sigma_x=S.five(sx), sigma_y=S.five(sy), sigma_radial=S.five(rad),
                n_repeats=(len(a["repeats"]) if a.get("repeats") else 0))


def la2_f2f():
    out = {}
    for src in ("unet", "gsam2", "pred"):
        disp = []
        for a in S.anchors():
            if a["motion"] != "fixation":
                continue
            cs = S.dense_centers(a["key"], src)
            for (i, x1, y1), (j, x2, y2) in zip(cs, cs[1:]):
                if j - i == 1:
                    disp.append(math.hypot(x2 - x1, y2 - y1))
        out[src] = S.five(disp)
    out["_note"] = "unet/gsam2=method stability; pred=model temporal jitter (not GT precision); human SKIP->GSAM2 proxy"
    return out


def la3_raster_floor():
    seen, dists, ious = set(), [], []
    for a in S.anchors():
        k = (a["subject"], a["eye"], a["session"])
        if k in seen:
            continue
        seen.add(k)
        for dst, iou in S.mask_vs_ellipse(*k):
            dists.append(dst)
            ious.append(iou)
    return dict(dist=S.five(dists), iou=S.five(ious), n_sessions=len(seen),
                interpretation="RASTERIZATION floor (mask derived from ellipse); "
                               "distance dominated by a constant ~1px generation offset, not human variability")


def main():
    la1, la2, la3 = la1_repeats(), la2_f2f(), la3_raster_floor()
    print("[La.1] GSAM2 repeats  sigma_radial median=%.3f p95=%.3f (n=%d, %d repeats/frame)"
          % (la1["sigma_radial"]["median"], la1["sigma_radial"]["p95"], la1["sigma_radial"]["n"], la1["n_repeats"]))
    print("[La.2] fixation F2F jitter median  unet=%.3f gsam2=%.3f pred=%.3f (pred=model jitter, separate)"
          % (la2["unet"]["median"], la2["gsam2"]["median"], la2["pred"]["median"]))
    print("[La.3] rasterization floor  dist median=%.3f  IoU median=%.3f (n_sessions=%d) [DERIVED mask]"
          % (la3["dist"]["median"], la3["iou"]["median"], la3["n_sessions"]))
    return dict(La1_gsam2_repeats=la1, La2_f2f_jitter=la2, La3_rasterization_floor=la3)


if __name__ == "__main__":
    main()
