"""Per-subject tables for test set 37-48:
 (A) 4-state counts + shares (from stride-2 classification cache)
 (B) total sample counts: full frame count (all frames, no stride) + GT-annotated + classified-n"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import EYES, SESSIONS, CACHE_DIR, REPORTS_DIR, list_frames, load_gt_ellipses
USERS_TEST=list(range(37,49)); STATES=["Fixation","Saccade","Smooth","Blink"]
cache=pd.read_csv(os.path.join(CACHE_DIR,"frames_4state_test37_48.csv"))

# (A) 4-state counts + shares
tab=(cache.groupby(["user","state"]).size().unstack(fill_value=0)
        .reindex(columns=STATES,fill_value=0).reindex(USERS_TEST,fill_value=0))
tab["Classified_n"]=tab[STATES].sum(axis=1)
for s in STATES: tab[s+"_%"]=(tab[s]/tab["Classified_n"]*100).round(2)
A=tab.reset_index().rename(columns={"user":"Subject"})

# (B) full sample counts (all frames, no stride) + GT
rows=[]
for u in USERS_TEST:
    full=0; gt=0
    for e in EYES:
        for c in SESSIONS:
            full+=len(list_frames(u,e,c))
            gt+=len(load_gt_ellipses(u,e,c))
    rows.append(dict(Subject=u, Full_frames_all=full, GT_annotated=gt,
                     Classified_n_stride2=int(tab.loc[u,"Classified_n"])))
B=pd.DataFrame(rows)

# add All rows
A_all={"Subject":"All",**{s:int(tab[s].sum()) for s in STATES},"Classified_n":int(tab["Classified_n"].sum())}
for s in STATES: A_all[s+"_%"]=round(tab[s].sum()/tab["Classified_n"].sum()*100,2)
A=pd.concat([A,pd.DataFrame([A_all])],ignore_index=True)
B=pd.concat([B,pd.DataFrame([{"Subject":"All","Full_frames_all":int(B.Full_frames_all.sum()),
   "GT_annotated":int(B.GT_annotated.sum()),"Classified_n_stride2":int(B.Classified_n_stride2.sum())}])],ignore_index=True)

os.makedirs(REPORTS_DIR,exist_ok=True)
A.to_csv(os.path.join(REPORTS_DIR,"tbl_test37_48_4state_detail.csv"),index=False)
B.to_csv(os.path.join(REPORTS_DIR,"tbl_test37_48_sample_counts.csv"),index=False)
print("=== (A) per-subject 4-state counts + share% ===")
print(A.to_string(index=False))
print("\n=== (B) per-subject total sample counts ===")
print(B.to_string(index=False))

# figure: per-subject total sample counts (full vs classified vs GT)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
b=B[B.Subject!="All"]; x=np.arange(len(b)); w=0.4
fig,ax=plt.subplots(figsize=(12,4.5))
ax.bar(x-w/2,b.Full_frames_all,w,label="Full frames (all)",color="#4472C4")
ax.bar(x+w/2,b.Classified_n_stride2,w,label="Classified (stride2)",color="#ED7D31")
ax.set_xticks(x); ax.set_xticklabels(b.Subject); ax.set_xlabel("Subject (test 37-48)")
ax.set_ylabel("# frames"); ax.legend(); ax.set_title("Test 37-48 per-subject sample counts")
for i,(f,g) in enumerate(zip(b.Full_frames_all,b.GT_annotated)):
    ax.text(x[i]-w/2,f,f"{f}",ha="center",va="bottom",fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_test37_48_sample_counts.png"),dpi=130)
print("\nwrote tbl_test37_48_4state_detail.csv, tbl_test37_48_sample_counts.csv, fig_test37_48_sample_counts.png")
