"""corrected_error.py (STEP 5 — corrected error & budget). y_pred is used ONLY here.

E_orig = ||y_pred - human_ellipse|| (346x260) over anchors: distribution + per-subject
(worst) + per-motion. ★ CAVEAT: samples are users 1-10 = HBTXR TRAIN subjects
(test=37-48) -> this is an OPTIMISTIC, in-training-distribution number, NOT the
subject-independent value.

Budget assembles, in the SAME 346x260 frame:
  rasterization floor (La.3) | sigma_human bracket (Lb.2) | inter-method RMS (Lb.1
  human-gsam2) | corrected E_orig median | reported 0.1812 (converted 64x64->346x260).
0.1812 is converted anisotropically: range [0.1812*4.06, 0.1812*5.41], iso-approx *4.69.
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S


def step5():
    E, subj, mot = [], [], []
    for a in S.anchors():
        if a["pred"] is None:
            continue
        E.append(math.hypot(a["pred"][0] - a["human"][0], a["pred"][1] - a["human"][1]))
        subj.append(a["subject"]); mot.append(a["motion"])
    E = np.array(E)
    per_subject = {f"user{u}": S.five([E[i] for i in range(len(E)) if subj[i] == u])
                   for u in sorted(set(subj))}
    worst = max(per_subject.items(), key=lambda kv: kv[1]["median"])
    c = S.REPORTED_0181_64
    conv = dict(value_64x64=c, iso_approx_346=round(c * S.GEO, 4),
                range_346=[round(c * S.SY, 4), round(c * S.SX, 4)])
    return dict(E_orig=S.five(E), per_subject=per_subject, worst_subject=worst[0],
                per_motion={m: S.five([E[i] for i in range(len(E)) if mot[i] == m]) for m in S.MOTIONS},
                reported_0181_converted=conv, frac_within={f"{t}px": round(float((E <= t).mean()), 4) for t in (2, 5, 10)},
                frac_gross_gt20=round(float((E > 20).mean()), 4),
                caveat="users 1-10 = HBTXR TRAIN subjects (test=37-48): OPTIMISTIC, NOT subject-independent")


def main():
    r = step5()
    e = r["E_orig"]
    print("[S5] E_orig(pred vs human GT) median=%.3f mean=%.3f p95=%.2f p99=%.2f max=%.2f (n=%d)  [OPTIMISTIC: train subj]"
          % (e["median"], e["mean"], e["p95"], e["p99"], e["max"], e["n"]))
    print("[S5] within 2/5/10px = %.0f/%.0f/%.0f%%  gross>20px=%.0f%%  worst=%s(med %.2f)"
          % (100 * r["frac_within"]["2px"], 100 * r["frac_within"]["5px"], 100 * r["frac_within"]["10px"],
             100 * r["frac_gross_gt20"], r["worst_subject"], r["per_subject"][r["worst_subject"]]["median"]))
    print("[S5] reported 0.1812(64x64) -> 346x260 iso-approx=%.2fpx range=%s"
          % (r["reported_0181_converted"]["iso_approx_346"], r["reported_0181_converted"]["range_346"]))
    return dict(STEP5=r)


if __name__ == "__main__":
    main()
