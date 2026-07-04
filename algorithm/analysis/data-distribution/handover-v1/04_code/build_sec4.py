"""(4) Fig.4-style violin+box: center-distance error (top) + IoU (bottom),
All + per-subject, EV-Eye frame track. Mirrors the uploaded reference figure."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import eveye_metrics as M
from common import CACHE_DIR, REPORTS_DIR

pe = M.load_pe_allframes()
pe = pe[pe.track=="frame"]
iou = pd.read_csv(os.path.join(CACHE_DIR,"iou_frame.csv"))
users = sorted(pe.user.unique())

def series_by_group(df, col):
    out=[("All", df[col].values)]
    for u in users: out.append((str(u), df[df.user==u][col].values))
    return out

def violin_box(ax, groups, ymax, hline, ylabel, clip=True):
    pos=np.arange(len(groups))
    data=[np.clip(d,0,ymax) if clip else d for _,d in groups]
    parts=ax.violinplot(data, positions=pos, widths=0.85, showextrema=False)
    for i,b in enumerate(parts['bodies']):
        b.set_facecolor("#2a9d8f" if i==0 else "#4c8fbf"); b.set_alpha(0.65); b.set_edgecolor("none")
    bp=ax.boxplot(data, positions=pos, widths=0.18, showfliers=False, patch_artist=True,
                  medianprops=dict(color="white",lw=1.2),
                  boxprops=dict(facecolor="black",edgecolor="black"),
                  whiskerprops=dict(color="black"),capprops=dict(color="black"))
    ax.axhline(hline, ls="--", color="gray", lw=1)
    ax.set_xticks(pos); ax.set_xticklabels([g[0] for g in groups], fontsize=6)
    ax.set_ylabel(ylabel)

fig, (a0,a1)=plt.subplots(2,1,figsize=(15,7),sharex=True)
violin_box(a0, series_by_group(pe,"err"), 15.0, 5.0, "Center Distance\nError (Pixel)")
a0.set_ylim(0,15); a0.set_title("Pupil Ellipse Center Distance Error & IoU Score — EV-Eye (frame/U-Net), 48 subjects")
violin_box(a1, series_by_group(iou,"iou"), 1.0, 0.5, "IoU score", clip=False)
a1.set_ylim(0,1.0); a1.set_xlabel("Subject")
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_sec4_eveye_frame_violinbox.png"),dpi=130)

# summary table per subject
rows=[]
for u in ["All"]+users:
    pp = pe if u=="All" else pe[pe.user==u]
    ii = iou if u=="All" else iou[iou.user==u]
    rows.append(dict(subject=u, n_err=len(pp), err_median=round(np.median(pp.err),3),
                     err_p95=round(np.percentile(pp.err,95),3),
                     n_iou=len(ii), iou_median=round(np.median(ii.iou),4),
                     iou_q1=round(np.percentile(ii.iou,25),4)))
t4=pd.DataFrame(rows); t4.to_csv(os.path.join(REPORTS_DIR,"sec4_subject_summary.csv"),index=False)
print("=== (4) overall (All) ===")
print(t4[t4.subject=="All"].to_string(index=False))
print("\nframe center-dist median(All)=%.2f px | IoU median(All)=%.3f"%(np.median(pe.err),np.median(iou.iou)))
print("wrote fig_sec4_eveye_frame_violinbox.png, sec4_subject_summary.csv")
