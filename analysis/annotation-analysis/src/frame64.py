"""frame64.py — recompute the key precision scalars in the 64x64 model frame.

The 346x260 -> 64x64 reduction is ANISOTROPIC (x/5.406, y/4.063), so radial quantities
must be recomputed from per-axis coordinates, not rescaled by one factor. Writes
results/precision/frame64_scalars.json. Run in .venv-gsam2 (needs h5py).
"""
import os, sys, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S


def c64(xy):
    return None if xy is None else (xy[0] / S.SX, xy[1] / S.SY)


def _V(a):
    return float(np.var(a, ddof=1))


def main():
    A = list(S.anchors())
    ac = S.load_annotator_centers()
    gh = lambda a: c64(a["human"]); gs = lambda a: c64(a["gsam2"]); gu = lambda a: c64(a["unet"])
    ge = lambda a: c64(ac["ellseg"].get(a["key"]))

    def diffs(fa, fb):
        dx, dy = [], []
        for a in A:
            va, vb = fa(a), fb(a)
            if va and vb:
                dx.append(va[0] - vb[0]); dy.append(va[1] - vb[1])
        return np.array(dx), np.array(dy)

    def three_ch(fs, fu):
        dxhs, dyhs = diffs(gh, fs); dxhu, dyhu = diffs(gh, fu); dxsu, dysu = diffs(fs, fu)
        sig = {}
        for ax, (hs, hu, su) in [("x", (dxhs, dxhu, dxsu)), ("y", (dyhs, dyhu, dysu))]:
            Vhs, Vhu, Vsu = _V(hs), _V(hu), _V(su)
            sig[ax] = dict(h=0.5 * (Vhs + Vhu - Vsu), s=0.5 * (Vhs + Vsu - Vhu), u=0.5 * (Vhu + Vsu - Vhs))
        rad = lambda k: math.sqrt(max(sig["x"][k], 0) + max(sig["y"][k], 0))
        neg = min(sig["x"]["h"], sig["x"]["s"], sig["x"]["u"], sig["y"]["h"], sig["y"]["s"], sig["y"]["u"]) < 0
        return dict(sigma_h=rad("h"), sigma_s=rad("s"), sigma_u=rad("u"), neg=neg)

    dep = three_ch(gs, gu)
    indep = three_ch(gs, ge)
    dxhg, dyhg = diffs(gh, gs)
    hg_rms = float(np.sqrt((dxhg ** 2 + dyhg ** 2).mean()))
    rep = []
    for a in A:
        if a["repeats"] and len(a["repeats"]) > 1:
            p = np.array([[x / S.SX, y / S.SY] for x, y in a["repeats"]]); c = p.mean(0)
            rep.append(float(np.sqrt(((p - c) ** 2).sum(1)).mean()))
    rf = []
    seen = set()
    for a in A:
        k = (a["subject"], a["eye"], a["session"])
        if k in seen:
            continue
        seen.add(k)
        for dstv, _ in S.mask_vs_ellipse(*k):
            rf.append(dstv / S.GEO)
    Eo = []
    for a in A:
        if a["pred"]:
            h = c64(a["human"]); p = c64(a["pred"]); Eo.append(math.hypot(p[0] - h[0], p[1] - h[1]))
    Eo = np.array(Eo)

    def pn(fx):
        d = []
        for a in A:
            v = fx(a)
            if v:
                h = c64(a["human"]); d.append(math.hypot(v[0] - h[0], v[1] - h[1]))
        d = np.array(d)
        return {str(t): round(float((d <= t).mean()), 3) for t in (10, 5, 1)}

    out = dict(sigma_human_3ch=dep["sigma_h"], sigma_human_3ch_neg=dep["neg"],
               sigma_human_indep=indep["sigma_h"], sigma_human_indep_neg=indep["neg"],
               sigma_gsam2_3ch=dep["sigma_s"], sigma_unet_3ch=dep["sigma_u"],
               gsam2_repeat=float(np.median(rep)), human_gsam2_rms=hg_rms,
               rep_floor=float(np.median(rf)), E_orig_median=float(np.median(Eo)),
               E_orig_mean=float(Eo.mean()), E_orig_p95=float(np.percentile(Eo, 95)),
               reported=S.REPORTED_0181_64, Pn_gsam2=pn(gs), Pn_pred=pn(lambda a: c64(a["pred"])),
               bracket=[dep["sigma_h"], hg_rms])
    json.dump(out, open(f"{S.OUT}/frame64_scalars.json", "w"), indent=1)
    print("[frame64] sigma_human=%.3f bracket=[%.3f,%.3f] E_orig median=%.3f reported=%.4f -> frame64_scalars.json"
          % (out["sigma_human_3ch"], out["bracket"][0], out["bracket"][1], out["E_orig_median"], out["reported"]))
    return out


if __name__ == "__main__":
    main()
