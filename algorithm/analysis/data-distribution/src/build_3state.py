"""3-state (Fixation/Smooth/Saccade) per-subject tables+figures.
(1) counts: from I-VT velocity labels (all GT frames).
(2) stats : authoritative official frame PE aligned per frame (official drops LAST 3
    labeled frames -> PE[k] <-> gt.frame_idx[k]) joined with I-VT state.  frame track.
"""
import os, sys, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import USERS, EYES, SESSIONS, PE_DIR, REPORTS_DIR, load_gt_ellipses
import eveye_metrics as M
from ivt_states import labeled_velocity

STATES=["Fixation","Smooth","Saccade"]; COL={"Fixation":"#2a9d8f","Smooth":"#1f77b4","Saccade":"#d1495b"}
vel,VSACC = labeled_velocity()
state_of = {(int(r.user),r.eye,int(r.code),int(r.frame_idx)): r.state for r in vel.itertuples()}

# ---- frame-track per-frame error WITH state (official PE aligned) ----
rows=[]
for u in USERS:
    for eye in EYES:
        for code in ["102","201","202"]:
            f=os.path.join(PE_DIR,"frame",eye,f"user{u}_{SESSIONS[code][0]}.mat")
            if not os.path.exists(f): continue
            pe=sio.loadmat(f)["matcell"].ravel().astype(float)
            gt=load_gt_ellipses(u,eye,code)
            if len(gt)==0: continue
            for k in range(min(len(pe),len(gt))):     # PE[k] <-> k-th labeled frame
                fidx=int(gt.iloc[k].frame_idx)
                st=state_of.get((u,eye,int(code),fidx))
                if st is None: continue
                rows.append(dict(user=u,eye=eye,code=int(code),frame_idx=fidx,state=st,err=float(pe[k])))
fe=pd.DataFrame(rows)
print(f"frame-track aligned w/ state: n={len(fe)} overall median={fe.err.median():.3f}px (official ~0.55 sanity)")
print(fe.groupby('state').err.median().to_string())

# ===== (1) per-subject 3-state COUNTS (from velocity labels) =====
cnt=vel.groupby(["user","state"]).size().unstack(fill_value=0).reindex(columns=STATES,fill_value=0)
cnt["total"]=cnt.sum(axis=1); cnt.loc["All"]=cnt.sum()
cnt.to_csv(os.path.join(REPORTS_DIR,"tbl_3state_counts.csv"))
print("\n(1) counts All:",cnt.loc["All",STATES].to_dict())

# ===== (2) per-subject 3-state STATS (frame track) =====
r2=[]
for u in sorted(fe.user.unique()):
    for st in STATES:
        d=M.describe(fe[(fe.user==u)&(fe.state==st)].err); r2.append(dict(subject=u,state=st,**d))
for st in STATES:
    d=M.describe(fe[fe.state==st].err); r2.append(dict(subject="All",state=st,**d))
t2=pd.DataFrame(r2); t2.to_csv(os.path.join(REPORTS_DIR,"tbl_3state_frame_stats.csv"),index=False)
print("\n(2) frame-track stats (All):")
print(t2[t2.subject=='All'][["state","n","mean","median","p95","p99"]].to_string(index=False))

# ===== FIGURES =====
users=sorted(cnt.index.drop("All"))
# (1) counts stacked bar
fig,ax=plt.subplots(figsize=(14,4)); x=np.arange(len(users)); bottom=np.zeros(len(users))
for st in STATES:
    vals=cnt.loc[users,st].values; ax.bar(x,vals,bottom=bottom,label=st,color=COL[st]); bottom+=vals
ax.set_xticks(x); ax.set_xticklabels(users,fontsize=6); ax.set_xlabel("subject"); ax.set_ylabel("# GT frames")
ax.legend(); ax.set_title(f"(1) Per-subject 3-state counts (I-VT, v_sacc={VSACC:.0f}px/s)")
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_3state_counts.png"),dpi=130)

# (2) stats: 4 panels mean/median/p95/p99, 3 state lines over subjects
fig,axes=plt.subplots(2,2,figsize=(15,7),sharex=True)
for ax,stat in zip(axes.ravel(),["mean","median","p95","p99"]):
    for st in STATES:
        s=t2[(t2.state==st)&(t2.subject!="All")].set_index("subject").reindex(users)
        ax.plot(np.arange(len(users)),s[stat],marker="o",ms=3,lw=1,label=st,color=COL[st])
    ax.set_title(stat); ax.set_ylabel("px"); ax.legend(fontsize=7)
    ax.set_xticks(np.arange(len(users))); ax.set_xticklabels(users,fontsize=5)
fig.suptitle("(2) Per-subject x 3-state center error — EV-Eye frame track (official PE)")
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_3state_frame_stats.png"),dpi=130)

# overall 3-state summary bar
fig,ax=plt.subplots(figsize=(7,4))
allt=t2[t2.subject=="All"].set_index("state").reindex(STATES); xx=np.arange(3); w=0.2
for i,(stat,c) in enumerate([("mean","#5b8c5a"),("median","#1f77b4"),("p95","#e8a33d"),("p99","#d1495b")]):
    ax.bar(xx+(i-1.5)*w,allt[stat],w,label=stat,color=c)
ax.set_xticks(xx); ax.set_xticklabels(STATES); ax.set_ylabel("px error"); ax.legend()
ax.set_title("3-state error (All subjects, EV-Eye frame)")
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_3state_overall.png"),dpi=130)
print("\nwrote tbl_3state_counts.csv, tbl_3state_frame_stats.csv, fig_3state_{counts,frame_stats,overall}.png")
