"""Per-subject detailed tables + figures.
(1) per-subject overall mean/median/P95/P99
(2) per-subject motion-type sample COUNTS
(3) per-subject x motion-type mean/median/P95/P99
(4) figures for (1)-(3). Tracks: frame & event. Subject = user (both eyes pooled)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import eveye_metrics as M
from common import REPORTS_DIR
df = M.load_pe_allframes(); os.makedirs(REPORTS_DIR, exist_ok=True)
users = sorted(df.user.unique())
STATS = ["mean","median","p95","p99"]

# ---------- (1) per-subject overall ----------
r1=[]
for track in ["frame","event"]:
    sub=df[df.track==track]
    for u in users:
        d=M.describe(sub[sub.user==u].err); r1.append(dict(track=track,subject=u,**d))
    d=M.describe(sub.err); r1.append(dict(track=track,subject="All",**d))
t1=pd.DataFrame(r1); t1.to_csv(os.path.join(REPORTS_DIR,"tbl1_subject_overall.csv"),index=False)

# ---------- (2) per-subject motion counts ----------
r2=[]
for u in users:
    rec={"subject":u}
    for track in ["frame","event"]:
        for mot in ["saccade","smooth"]:
            rec[f"{track}_{mot}_n"]=int(((df.track==track)&(df.user==u)&(df.motion==mot)).sum())
    rec["total_n"]=int((df.user==u).sum())
    r2.append(rec)
t2=pd.DataFrame(r2); t2.to_csv(os.path.join(REPORTS_DIR,"tbl2_subject_motion_counts.csv"),index=False)

# ---------- (3) per-subject x motion mean/median/p95/p99 ----------
r3=[]
for track in ["frame","event"]:
    sub=df[df.track==track]
    for u in users:
        for mot in ["saccade","smooth"]:
            d=M.describe(sub[(sub.user==u)&(sub.motion==mot)].err)
            r3.append(dict(track=track,subject=u,motion=mot,**d))
t3=pd.DataFrame(r3); t3.to_csv(os.path.join(REPORTS_DIR,"tbl3_subject_motion_stats.csv"),index=False)
print("wrote tbl1/tbl2/tbl3 csv. rows:",len(t1),len(t2),len(t3))

# ================= FIGURES =================
# (1) per-subject overall: lines of mean/median/p95/p99 over subjects
for track in ["frame","event"]:
    s=t1[(t1.track==track)&(t1.subject!="All")].copy(); s["subject"]=s["subject"].astype(int)
    s=s.sort_values("subject"); x=s["subject"].values
    fig,ax=plt.subplots(figsize=(14,4.2))
    for st,c in [("mean","#5b8c5a"),("median","#1f77b4"),("p95","#e8a33d"),("p99","#d1495b")]:
        ax.plot(x,s[st],marker="o",ms=3,lw=1,label=st,color=c)
    allrow=t1[(t1.track==track)&(t1.subject=="All")].iloc[0]
    for st,c in [("median","#1f77b4"),("p95","#e8a33d"),("p99","#d1495b")]:
        ax.axhline(allrow[st],ls="--",lw=0.8,color=c,alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(x,fontsize=6); ax.set_xlabel("subject")
    ax.set_ylabel("center error (px)"); ax.legend(ncol=4,fontsize=8)
    ax.set_title(f"(1) Per-subject error statistics — EV-Eye {track} (dashed = All)")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,f"fig_tbl1_subject_overall_{track}.png"),dpi=130)

# (2) per-subject motion counts: stacked bars (frame track) + event overlay markers
fig,ax=plt.subplots(figsize=(14,4))
x=np.arange(len(users))
sac=[t2[t2.subject==u][f"frame_saccade_n"].iloc[0] for u in users]
smo=[t2[t2.subject==u][f"frame_smooth_n"].iloc[0] for u in users]
ax.bar(x,sac,label="saccade",color="#d1495b")
ax.bar(x,smo,bottom=sac,label="smooth",color="#1f77b4")
ax.set_xticks(x); ax.set_xticklabels(users,fontsize=6); ax.set_xlabel("subject")
ax.set_ylabel("# evaluated GT frames"); ax.legend()
ax.set_title("(2) Per-subject motion-type sample counts (frame track; saccade+smooth)")
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_tbl2_subject_motion_counts.png"),dpi=130)

# (3) per-subject x motion: 2x2 panels (mean/median/p95/p99), saccade vs smooth, per track
for track in ["frame","event"]:
    s=t3[t3.track==track]
    fig,axes=plt.subplots(2,2,figsize=(15,7),sharex=True)
    for ax,st in zip(axes.ravel(),STATS):
        for mot,c in [("saccade","#d1495b"),("smooth","#1f77b4")]:
            m=s[s.motion==mot].set_index("subject").reindex(users)
            ax.plot(np.arange(len(users)),m[st],marker="o",ms=3,lw=1,label=mot,color=c)
        ax.set_title(st); ax.set_ylabel("px"); ax.legend(fontsize=7)
        ax.set_xticks(np.arange(len(users))); ax.set_xticklabels(users,fontsize=5)
    fig.suptitle(f"(3) Per-subject x motion-type error stats — EV-Eye {track}")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,f"fig_tbl3_subject_motion_{track}.png"),dpi=130)

print("wrote figures: fig_tbl1_*, fig_tbl2_*, fig_tbl3_*")
print("\n(1) All-row:\n", t1[t1.subject=='All'].to_string(index=False))
print("\n(2) counts head:\n", t2.head(4).to_string(index=False))
