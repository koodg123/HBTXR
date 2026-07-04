import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import CACHE_DIR, REPORTS_DIR
from error_harness import load_labeled

df, thr = load_labeled()
os.makedirs(REPORTS_DIR, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# panel 1: velocity signature
bins = np.logspace(np.log10(0.5), np.log10(5000), 45)
for mot,c in [("saccade","#d1495b"),("smooth","#1f77b4")]:
    s = df[df.motion==mot].speed_pxps.dropna()
    ax[0].hist(s, bins=bins, alpha=0.55, label=f"{mot} session (n={len(s)})", color=c)
ax[0].set_xscale("log"); ax[0].set_xlabel("pupil speed at GT frame (px/s)")
ax[0].set_ylabel("# GT frames")
ax[0].set_title("(a) Velocity signature by session type\n(48 users; pattern1=saccade, pattern2=smooth)")
ax[0].legend(fontsize=8)

# panel 2: I-VT per-motion error (filtered <34px), median + p95
order=["fixation","saccade","smooth"]
med=[df[(df.ivt==k)&(df.err<34)].err.median() for k in order]
p95=[np.percentile(df[(df.ivt==k)&(df.err<34)].err,95) for k in order]
x=np.arange(3); w=0.38
ax[1].bar(x-w/2, med, w, label="median", color="#5b8c5a")
ax[1].bar(x+w/2, p95, w, label="p95", color="#e8a33d")
ax[1].set_xticks(x); ax[1].set_xticklabels(order)
ax[1].set_ylabel("pixel error (filtered <34px)")
ax[1].set_title("(b) Error by I-VT motion class\n(cv2 classical baseline)")
ax[1].legend(fontsize=8)

# panel 3: velocity-bin vs error (raw median, log)
order2=["low","med","high"]
medr=[df[df.vbin==k].err.median() for k in order2]
ax[2].plot(order2, medr, "o-", color="#7d3c98", lw=2)
ax[2].set_ylabel("median pixel error (raw)")
ax[2].set_xlabel("speed bin")
ax[2].set_title("(c) Error vs speed bin\n(raw median; blink-sensitive)")

plt.tight_layout()
p = os.path.join(REPORTS_DIR, "fig_velocity_mapping_verification.png")
plt.savefig(p, dpi=130); print("wrote", p)

# per-subject summary (rigor)
rows=[]
for u,g in df.groupby("user"):
    for mot in ["saccade","smooth"]:
        s=g[g.motion==mot].err
        sf=s[s<34]
        rows.append(dict(user=u,motion=mot,n=len(s),
            median=round(s.median(),2),p95=round(np.percentile(s,95),2) if len(s) else np.nan,
            median_filt=round(sf.median(),2) if len(sf) else np.nan))
ps=pd.DataFrame(rows)
ps.to_csv(os.path.join(REPORTS_DIR,"per_subject_cv2_baseline.csv"),index=False)
print("per-subject median_filt by motion (mean over 48 users):")
print(ps.groupby("motion").median_filt.agg(['mean','std','min','max']).round(3).to_string())
