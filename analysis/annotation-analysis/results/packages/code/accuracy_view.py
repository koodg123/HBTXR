"""accuracy_view.py (Layer C — threshold / metric precision), 346x260 px.

Lc.1  P_n = fraction within {10,5,1}px of human GT, for:
        gsam2_vs_human  (inter-method agreement)
        pred_vs_human   (HBTXR model, corrected)
      abstain (no valid detection) is separated: P_n counts abstain as FAIL (over ALL
      anchors); precision is over VALID predictions only. In 346x260 the 1px threshold
      is very strict (~0.18px in the 64x64 model frame).
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S

THRS = [10, 5, 1]


def lc1_pn():
    res = {}
    for name, src in [("gsam2_vs_human", "gsam2"), ("pred_vs_human", "pred")]:
        dists, n_total, n_valid = [], 0, 0
        for a in S.anchors():
            n_total += 1
            v = a[src]
            if v is not None:
                n_valid += 1
                dists.append(math.hypot(v[0] - a["human"][0], v[1] - a["human"][1]))
        dd = np.array(dists)
        res[name] = dict(
            n_total=n_total, n_valid=n_valid, abstain=n_total - n_valid,
            Pn_over_all={str(t): round(float((dd <= t).sum()) / n_total, 4) for t in THRS},
            precision_over_valid={str(t): round(float((dd <= t).mean()), 4) for t in THRS},
            median=float(np.median(dd)))
    return res


def main():
    r = lc1_pn()
    for name in ("gsam2_vs_human", "pred_vs_human"):
        x = r[name]
        print("[Lc.1] %-16s valid=%d/%d abstain=%d | P_n(all) 10/5/1px=%.2f/%.2f/%.2f | prec(valid)=%.2f/%.2f/%.2f"
              % (name, x["n_valid"], x["n_total"], x["abstain"],
                 x["Pn_over_all"]["10"], x["Pn_over_all"]["5"], x["Pn_over_all"]["1"],
                 x["precision_over_valid"]["10"], x["precision_over_valid"]["5"], x["precision_over_valid"]["1"]))
    return dict(Lc1_Pn=r)


if __name__ == "__main__":
    main()
