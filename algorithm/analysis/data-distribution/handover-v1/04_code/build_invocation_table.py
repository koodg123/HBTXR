"""Per-subject Search/Track invocation counts & ratio (Train/Val/Test),
derived from the analyzed 4-state distribution (NOT raw tracker logs).
Mapping (hybrid search-track logic): Track-mode = Fixation+Smooth (continuous event tracking);
Search-mode = Saccade+Blink (track lost/unreliable -> re-detection/search).
Event basis = measured event 4-state (captures saccade & blink); Frame basis = frame 4-state."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
MT=os.environ.get("MT_CODES","/sessions/busy-happy-newton/mnt/DATASET/codes")
DD=os.path.join(MT,"data-distribution")
EV=os.path.join(MT,"motion_type","cache","event_4state_all48.csv")
PS=os.path.join(os.environ.get("EVEYE_ROOT","/sessions/busy-happy-newton/mnt/eveye"),"DeanDataset_full_unet","progress_state.json")
ST=["Fixation","Saccade","Smooth","Blink"]
TRAIN=set(range(1,37)); VAL={37,40,45,48}; TEST=set(range(37,49))
F2C={"session_1_0_1":"saccade","session_1_0_2":"smooth","session_2_0_1":"saccade","session_2_0_2":"smooth"}
def split_label(u): return "Train" if u in TRAIN else "Test"

def per_subject_event():
    ev=pd.read_csv(EV); g=ev.groupby("user")[ST].sum()
    return g
def per_subject_frame():
    ps=json.load(open(PS)); rows={}
    for s in ps["session_summaries"]:
        parts=s["session"].replace("\\","/").split("/")
        u=int([p for p in parts if p.startswith("user")][-1][4:]); mot=F2C[parts[-1]]
        valid=s["frames"]-s.get("skipped",0); blink=s.get("skip_no_ellipse",0)
        d=rows.setdefault(u,{k:0 for k in ST})
        d["Blink"]+=blink; d["Fixation" if mot=="saccade" else "Smooth"]+=valid
    return pd.DataFrame(rows).T.reindex(columns=ST).fillna(0)

def make(basis, g, fname):
    rows=[]
    for u in sorted(g.index):
        fx,sa,sm,bl=[float(g.loc[u,k]) for k in ST]
        track=fx+sm; search=sa+bl; tot=track+search
        rows.append(dict(Subject=int(u), Split=split_label(u), InVal=(u in VAL),
            Track_inv=int(round(track)), Search_inv=int(round(search)), Total=int(round(tot)),
            Track_pct=round(100*track/tot,2) if tot else 0,
            Search_pct=round(100*search/tot,2) if tot else 0,
            Track_to_Search=round(track/search,2) if search>0 else np.nan))
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(DD,"tables",fname),index=False)
    return df

def summary(df, basis):
    out=[]
    for sp,users in [("Train",TRAIN),("Val",VAL),("Test",TEST)]:
        d=df[df.Subject.isin(users)]; tr=d.Track_inv.sum(); se=d.Search_inv.sum(); to=tr+se
        out.append(dict(basis=basis,split=sp,n_subj=d.Subject.nunique(),
            Track_inv=int(tr),Search_inv=int(se),Total=int(to),
            Track_pct=round(100*tr/to,2),Search_pct=round(100*se/to,2),
            Track_to_Search=round(tr/se,2) if se>0 else np.nan))
    return pd.DataFrame(out)

os.makedirs(os.path.join(DD,"tables"),exist_ok=True)
ge=per_subject_event(); gf=per_subject_frame()
de=make("event",ge,"tbl_invocation_search_track_event.csv")
df_=make("frame",gf,"tbl_invocation_search_track_frame.csv")
ss=pd.concat([summary(de,"event"),summary(df_,"frame")],ignore_index=True)
ss.to_csv(os.path.join(DD,"tables","tbl_invocation_summary.csv"),index=False)
print("=== EVENT-basis per-subject (head) ===")
print(de.head(6).to_string(index=False))
print("\n=== SPLIT SUMMARY (Search/Track invocation) ===")
print(ss.to_string(index=False))
