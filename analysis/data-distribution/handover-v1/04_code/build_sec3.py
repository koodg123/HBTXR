"""(3) Per-subject x motion-type error distribution: split violins + delta table."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import eveye_metrics as M
from common import REPORTS_DIR
df = M.load_pe_allframes(); os.makedirs(REPORTS_DIR, exist_ok=True)

# table: per subject x motion (+ smooth-saccade delta of medians)
rows=[]
for track in ["frame","event"]:
    sub=df[df.track==track]
    for u in sorted(sub.user.unique()):
        rec={"track":track,"user":u}
        for mot in ["saccade","smooth"]:
            s=sub[(sub.user==u)&(sub.motion==mot)].err
            d=M.describe(s); rec[f"{mot}_n"]=d["n"]; rec[f"{mot}_median"]=d["median"]; rec[f"{mot}_p95"]=d["p95"]
        rec["delta_median(smooth-sacc)"]=round((rec.get("smooth_median",np.nan) or np.nan)-(rec.get("saccade_median",np.nan) or np.nan),3)
        rows.append(rec)
t3=pd.DataFrame(rows); t3.to_csv(os.path.join(REPORTS_DIR,"sec3_subject_motion.csv"),index=False)
print("=== (3) per-subject x motion (frame, head) ===")
print(t3[t3.track=="frame"].head(6).to_string(index=False))
print("\nframe: mean delta median (smooth-saccade) over subjects = %.3f px"%t3[t3.track=="frame"]["delta_median(smooth-sacc)"].mean())
print("event: mean delta median (smooth-saccade) over subjects = %.3f px"%t3[t3.track=="event"]["delta_median(smooth-sacc)"].mean())

# split-violin figure per subject (saccade left / smooth right)
def split_violin(ax, sub, track, ymax):
    users=sorted(sub.user.unique()); 
    # All first
    groups=[("All",sub)]+[(str(u),sub[sub.user==u]) for u in users]
    for i,(lab,g) in enumerate(groups):
        for mot,side,c in [("saccade",-1,"#d1495b"),("smooth",1,"#1f77b4")]:
            d=np.clip(g[g.motion==mot].err.values,0,ymax)
            if len(d)<3: continue
            vp=ax.violinplot([d],positions=[i],widths=0.9,showmedians=True)
            for b in vp['bodies']:
                m=np.mean(b.get_paths()[0].vertices[:,0])
                b.get_paths()[0].vertices[:,0]=np.clip(b.get_paths()[0].vertices[:,0], m if side>0 else -np.inf, np.inf if side>0 else m)
                b.set_facecolor(c); b.set_alpha(0.7)
            for key in ('cbars','cmins','cmaxes','cmedians'):
                if key in vp: vp[key].set_color(c); vp[key].set_linewidth(0.8)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups],fontsize=6,rotation=90)
    ax.set_ylim(0,ymax); ax.set_ylabel("center error (px)")
    ax.set_title(f"(3) Per-subject error distribution by motion — EV-Eye {track} (red=saccade, blue=smooth)")

for track,ymax in [("frame",3.0),("event",6.0)]:
    sub=df[df.track==track]
    fig,ax=plt.subplots(figsize=(15,4.5)); split_violin(ax,sub,track,ymax)
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,f"fig_sec3_subject_motion_{track}.png"),dpi=130)
print("\nwrote sec3_subject_motion.csv + fig_sec3_subject_motion_{frame,event}.png")
