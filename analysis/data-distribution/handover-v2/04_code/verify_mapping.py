"""Verify pattern->motion mapping and characterize velocity by motion.
Also checks: within saccade session 201, GT frames INSIDE the labeled saccade
window should have higher speed than those outside (independent confirmation)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import CACHE_DIR

df = pd.read_csv(os.path.join(CACHE_DIR, "velocity_at_gt.csv"))
df = df[df.speed_pxps.notna()].copy()
print(f"rows with valid speed: {len(df)}  users={sorted(df.user.unique())}")

def q(s):
    return dict(n=len(s), median=np.median(s), p25=np.percentile(s,25),
                p75=np.percentile(s,75), p90=np.percentile(s,90), frac_lt30=(s<30).mean())

print("\n==== speed (px/s) by motion (saccade sessions vs smooth sessions) ====")
for mot in ["saccade","smooth"]:
    s = df[df.motion==mot].speed_pxps
    r = q(s)
    print(f"{mot:8s}: n={r['n']:4d} median={r['median']:7.1f} p25={r['p25']:7.1f} "
          f"p75={r['p75']:7.1f} p90={r['p90']:7.1f}  frac(<30px/s)={r['frac_lt30']:.2f}")

print("\n==== saccade session (201): inside vs outside labeled saccade window ====")
s201 = df[df.code==201]
ins = s201[s201.in_seg_window].speed_pxps
out = s201[~s201.in_seg_window].speed_pxps
if len(ins): print(f"inside  window: n={len(ins):3d} median={np.median(ins):7.1f} p90={np.percentile(ins,90):7.1f}")
if len(out): print(f"outside window: n={len(out):3d} median={np.median(out):7.1f} p90={np.percentile(out,90):7.1f}")

# data-driven I-VT thresholds from saccade-session distribution
sacc_speed = df[df.motion=="saccade"].speed_pxps
v_fix = np.percentile(sacc_speed, 60)    # below -> fixation
v_sacc = np.percentile(sacc_speed, 90)   # above -> saccade
print(f"\n==== derived I-VT thresholds (px/s): v_fix(<)={v_fix:.0f}  v_sacc(>)={v_sacc:.0f} ====")

def ivt(v):
    if v < v_fix: return "fixation"
    if v > v_sacc: return "saccade"
    return "smooth"
df["ivt"] = df.speed_pxps.apply(ivt)
print("\nI-VT label distribution within each SESSION type:")
print(pd.crosstab(df.motion, df.ivt).to_string())

# velocity bins (tertiles over all)
edges = np.percentile(df.speed_pxps, [33.3, 66.6])
print(f"\n==== velocity bin edges (px/s): low<{edges[0]:.0f}<=med<{edges[1]:.0f}<=high ====")
df["vbin"] = np.where(df.speed_pxps<edges[0],"low",np.where(df.speed_pxps<edges[1],"med","high"))
print(pd.crosstab(df.motion, df.vbin).to_string())
