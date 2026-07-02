"""reproducibility.py (Layer B — between-method agreement), 346x260 px.

Lb.1  pairwise disagreement for (human,gsam2),(human,unet),(gsam2,unet):
        radial RMS, axis Var, Bland-Altman bias + 95% limits of agreement, mask IoU.
Lb.2  three-cornered-hat (axis-wise x,y): decompose each source's own sigma.
        sigma_h^2 = .5(V_hs+V_hu-V_su), etc. Non-negativity check; if negative ->
        "SHARED BIAS" -> sigma_human bracket = [3CH lower, Human<->GSAM2 RMS upper].
        Also run the independent triple {human, gsam2, ellseg} as robustness.
Lb.3  cross-check: 3CH sigma_s(gsam2) vs La.1 GSAM2 repeat sigma.
NB (decision): U-Net is trained on the (rasterized) human ellipse -> not independent;
we run {h,gsam2,unet} anyway (per user) and expect a likely shared-bias fallback.
"""
import os, sys, math, glob
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S
import repeatability as La


def _read_mask(p):
    if p.endswith(".gif"):
        return np.array(Image.open(p).convert("L")) > 0
    return cv2.imread(p, 0) > 0


def _gt_mask(a):
    m = np.zeros((S.H, S.W), np.uint8)
    cv2.ellipse(m, (int(round(a["human"][0])), int(round(a["human"][1]))),
                (max(int(round(a["rx"])), 1), max(int(round(a["ry"])), 1)),
                math.degrees(a["theta"]), 0, 360, 1, -1)
    return m > 0


def _iou(x, y):
    i = np.logical_and(x, y).sum(); u = np.logical_or(x, y).sum()
    return float(i / u) if u else float("nan")


def lb1_pairwise():
    res = {}
    for name, (A, B) in {"human-gsam2": ("human", "gsam2"), "human-unet": ("human", "unet"),
                         "gsam2-unet": ("gsam2", "unet")}.items():
        dx, dy, r2 = [], [], []
        for a in S.anchors():
            va, vb = a[A], a[B]
            if va and vb:
                dx.append(va[0] - vb[0]); dy.append(va[1] - vb[1])
                r2.append((va[0] - vb[0]) ** 2 + (va[1] - vb[1]) ** 2)
        dx, dy, r2 = np.array(dx), np.array(dy), np.array(r2)
        res[name] = dict(n=int(len(dx)), radial_rms=float(np.sqrt(r2.mean())),
                         var_x=float(dx.var(ddof=1)), var_y=float(dy.var(ddof=1)),
                         ba_bias_x=float(dx.mean()), ba_bias_y=float(dy.mean()),
                         ba_loa_x=[float(dx.mean() - 1.96 * dx.std(ddof=1)), float(dx.mean() + 1.96 * dx.std(ddof=1))],
                         ba_loa_y=[float(dy.mean() - 1.96 * dy.std(ddof=1)), float(dy.mean() + 1.96 * dy.std(ddof=1))])
    iu = {"unet_vs_gt": [], "gsam2_vs_gt": [], "gsam2_vs_unet": []}
    for a in S.anchors():
        ai = a["anchor_idx"]; gt = _gt_mask(a)
        up = glob.glob(f"{S.LAB}/{a['key']}/unet_masks/{ai:06d}_*_mask.gif")
        gp = glob.glob(f"{S.LAB}/{a['key']}/gsam2_masks/{ai:06d}_*_mask.png")
        um = _read_mask(up[0]) if up else None
        gm = _read_mask(gp[0]) if gp else None
        if um is not None:
            iu["unet_vs_gt"].append(_iou(um, gt))
        if gm is not None:
            iu["gsam2_vs_gt"].append(_iou(gm, gt))
        if um is not None and gm is not None:
            iu["gsam2_vs_unet"].append(_iou(gm, um))
    res["IoU"] = {k: S.five(v) for k, v in iu.items()}
    return res


def lb2_3ch(triple=("human", "gsam2", "unet")):
    h, s, u = triple
    ac = S.load_annotator_centers() if (s in S.ANNOT or u in S.ANNOT) else None

    def get(a, name):
        return a[name] if name in ("human", "unet", "gsam2", "pred") else ac[name].get(a["key"])
    d = {k: [] for k in ("xhs", "yhs", "xhu", "yhu", "xsu", "ysu")}
    for a in S.anchors():
        H_, S_, U_ = get(a, h), get(a, s), get(a, u)
        if H_ and S_ and U_:
            d["xhs"].append(H_[0] - S_[0]); d["yhs"].append(H_[1] - S_[1])
            d["xhu"].append(H_[0] - U_[0]); d["yhu"].append(H_[1] - U_[1])
            d["xsu"].append(S_[0] - U_[0]); d["ysu"].append(S_[1] - U_[1])
    V = lambda a: float(np.var(a, ddof=1))
    out = {"triple": triple, "n": len(d["xhs"]), "shared_bias": False}
    for ax, (hs, hu, su) in [("x", (d["xhs"], d["xhu"], d["xsu"])), ("y", (d["yhs"], d["yhu"], d["ysu"]))]:
        Vhs, Vhu, Vsu = V(hs), V(hu), V(su)
        s2 = dict(h=0.5 * (Vhs + Vhu - Vsu), s=0.5 * (Vhs + Vsu - Vhu), u=0.5 * (Vhu + Vsu - Vhs))
        neg = min(s2.values()) < 0
        out["shared_bias"] = out["shared_bias"] or neg
        out[ax] = dict(Vhs=Vhs, Vhu=Vhu, Vsu=Vsu, neg=neg,
                       **{f"sigma_{k}": (math.sqrt(v) if v >= 0 else None) for k, v in s2.items()})

    def rad(name):
        a, b = out["x"][name], out["y"][name]
        return math.sqrt(a ** 2 + b ** 2) if (a is not None and b is not None) else None
    out["sigma_h_radial"] = rad("sigma_h")
    out["sigma_s_radial"] = rad("sigma_s")
    out["sigma_u_radial"] = rad("sigma_u")
    return out


def main():
    lb1 = lb1_pairwise()
    lb2 = lb2_3ch(("human", "gsam2", "unet"))
    lb2_indep = lb2_3ch(("human", "gsam2", "ellseg"))
    la1 = La.la1_repeats()
    lb3 = dict(sigma_s_3ch=lb2["sigma_s_radial"], sigma_s_rep_La1=la1["sigma_radial"]["median"],
               ratio=((lb2["sigma_s_radial"] / la1["sigma_radial"]["median"])
                      if (lb2["sigma_s_radial"] and la1["sigma_radial"]["median"]) else None))
    rms_hg = lb1["human-gsam2"]["radial_rms"]
    bracket = dict(lower_3ch=lb2["sigma_h_radial"], upper_human_gsam2_rms=rms_hg,
                   shared_bias=lb2["shared_bias"],
                   lower_3ch_independent=lb2_indep["sigma_h_radial"],
                   shared_bias_independent=lb2_indep["shared_bias"])
    print("[Lb.1] radial RMS  human-gsam2=%.3f human-unet=%.3f gsam2-unet=%.3f | IoU gsam2/gt=%.3f unet/gt=%.3f"
          % (rms_hg, lb1["human-unet"]["radial_rms"], lb1["gsam2-unet"]["radial_rms"],
             lb1["IoU"]["gsam2_vs_gt"]["median"], lb1["IoU"]["unet_vs_gt"]["median"]))
    print("[Lb.1] Bland-Altman human-unet bias=(%.2f,%.2f)px  (systematic offset)"
          % (lb1["human-unet"]["ba_bias_x"], lb1["human-unet"]["ba_bias_y"]))
    print("[Lb.2] 3CH{h,gsam2,unet} sigma_human_radial=%s shared_bias=%s | independent{h,gsam2,ellseg} sigma_h=%s shared_bias=%s"
          % (None if lb2["sigma_h_radial"] is None else round(lb2["sigma_h_radial"], 3), lb2["shared_bias"],
             None if lb2_indep["sigma_h_radial"] is None else round(lb2_indep["sigma_h_radial"], 3), lb2_indep["shared_bias"]))
    print("[Lb.2] sigma_human BRACKET = [%s (3CH), %.3f (Human<->GSAM2 RMS)]"
          % (None if bracket["lower_3ch"] is None else round(bracket["lower_3ch"], 3), rms_hg))
    print("[Lb.3] sigma_s: 3CH=%s vs La.1-repeat=%.3f ratio=%s"
          % (None if lb2["sigma_s_radial"] is None else round(lb2["sigma_s_radial"], 3),
             la1["sigma_radial"]["median"], None if lb3["ratio"] is None else round(lb3["ratio"], 2)))
    return dict(Lb1_pairwise=lb1, Lb2_3ch=lb2, Lb2_3ch_independent=lb2_indep,
                Lb3_crosscheck=lb3, sigma_human_bracket=bracket)


if __name__ == "__main__":
    main()
