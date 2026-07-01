"""
14_flag_gsam2_mislabel.py — flag GSAM2 mis-detections in gsam2.json (non-destructive).

Recommended gate (validated on the anchor gold set: catches 14/14 mislabels, 0 good dropped):
  mislabel = (U-Net valid at this frame AND ||y_gsam2 - y_unet|| > --xcheck-thr)
             OR (U-Net invalid AND det_score < --det-thr AND area < --area-thr)   # fallback
Uses U-Net dense center as an audit-vs-audit cross-check (no human GT) so the audit stays
independent of the frozen human labels. Adds fields to each gsam2_centers entry:
  "mislabel": bool, "mislabel_reason": str, "e_unet": float|null
Re-run 11_build_perframe.py afterwards to propagate the flag into perframe gsam2/center.json.
"""
import os, csv, glob, json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../samples")
    ap.add_argument("--xcheck-thr", type=float, default=15.0, help="||gsam2-unet|| px reject threshold")
    ap.add_argument("--det-thr", type=float, default=0.35, help="fallback det_score floor (unet invalid)")
    ap.add_argument("--area-thr", type=float, default=400.0, help="fallback mask-area floor (unet invalid)")
    a = ap.parse_args()
    LAB = os.path.join(a.out, "label")

    n_frame = n_mis = 0
    reasons = {"unet_xcheck": 0, "lowconf_tiny": 0}
    for gj in sorted(glob.glob(os.path.join(LAB, "*", "gsam2.json"))):
        key = os.path.basename(os.path.dirname(gj))
        g = json.load(open(gj))
        unet = {c["idx"]: c for c in json.load(open(os.path.join(LAB, key, "unet_dense.json")))["unet_centers"]}
        for c in g["gsam2_centers"]:
            if not c.get("valid"):
                c["mislabel"] = False; c["mislabel_reason"] = "invalid"; c["e_unet"] = None
                continue
            n_frame += 1
            u = unet.get(c["idx"])
            e_unet = None; mis = False; reason = "ok"
            if u and u.get("valid"):
                e_unet = float(np.hypot(c["cx"] - u["cx"], c["cy"] - u["cy"]))
                if e_unet > a.xcheck_thr:
                    mis, reason = True, "unet_xcheck"
            else:  # no U-Net reference -> intrinsic fallback
                if c.get("det_score", 1) < a.det_thr and c.get("area", 1e9) < a.area_thr:
                    mis, reason = True, "lowconf_tiny"
            c["mislabel"] = mis
            c["mislabel_reason"] = reason
            c["e_unet"] = None if e_unet is None else round(e_unet, 2)
            if mis:
                n_mis += 1; reasons[reason] = reasons.get(reason, 0) + 1
        with open(gj, "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=2)

    print(f"[done] flagged {n_frame} valid gsam2 frames; mislabel={n_mis} ({100*n_mis/max(1,n_frame):.2f}%)")
    print(f"  by reason: {reasons}")
    print("  -> gsam2.json updated with mislabel/mislabel_reason/e_unet. Re-run 11_build_perframe.py.")


if __name__ == "__main__":
    main()
