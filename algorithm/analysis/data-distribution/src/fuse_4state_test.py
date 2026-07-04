"""Fused per-frame 4-state (test 37-48), frame timeline (same DAVIS clock as events):
  Blink   = frame UNet mask empty   (from frames_4state_test37_48.csv, reliable)
  Saccade = event burst in the frame's +-20ms window (events.txt; >4x session-median)
  else    = Fixation (saccade-session) / Smooth (smooth-session)
Parallel(2), resumable per session."""
import os, sys, glob, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import EYES, SESSIONS, CACHE_DIR, REPORTS_DIR, EVEYE_ROOT, parse_frame_filename
RAW=os.path.join(EVEYE_ROOT,"raw_data","Data_davis")
FRAME_CACHE=os.path.join(CACHE_DIR,"frames_4state_test37_48.csv")
OUT=os.path.join(CACHE_DIR,"fused_4state_test37_48.csv")
USERS_TEST=list(range(37,49)); STATES=["Fixation","Saccade","Smooth","Blink"]
WIN_US=20000; BURST=4.0
FC=pd.read_csv(FRAME_CACHE)

def one(args):
    u,e,c=args; folder,motion,_=SESSIONS[c]
    sub=FC[(FC.user==u)&(FC.eye==e)&(FC.code==int(c))]
    if len(sub)==0: return None
    fr=glob.glob(f"{RAW}/user{u}/{e}/{folder}/frames/*.png")
    idx2ts={parse_frame_filename(os.path.basename(p))[0]: parse_frame_filename(os.path.basename(p))[1] for p in fr}
    ef=f"{RAW}/user{u}/{e}/{folder}/events/events.txt"
    if not os.path.exists(ef): return None
    et=pd.read_csv(ef,sep=r'\s+',header=None,usecols=[0],names=['t'],dtype=np.int64)['t'].values
    et.sort()
    rec=dict(user=u,eye=e,code=int(c),motion=motion,Fixation=0,Saccade=0,Smooth=0,Blink=0,n=0)
    # per cached frame: event count in +-WIN
    cnts=[]; states0=[]; tss=[]
    for r in sub.itertuples():
        ts=idx2ts.get(int(r.frame_idx))
        if ts is None: continue
        lo=np.searchsorted(et,ts-WIN_US); hi=np.searchsorted(et,ts+WIN_US)
        cnts.append(hi-lo); states0.append(r.state); tss.append(ts)
    cnts=np.array(cnts); 
    if len(cnts)==0: return rec
    med=np.median(cnts[cnts>0]) if (cnts>0).any() else 1
    for cnt,st in zip(cnts,states0):
        if st=="Blink": rec["Blink"]+=1
        elif cnt> BURST*med: rec["Saccade"]+=1
        elif motion=="saccade": rec["Fixation"]+=1
        else: rec["Smooth"]+=1
    rec["n"]=len(cnts)
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=float,default=36); a=ap.parse_args()
    done=set(); rows=[]
    if os.path.exists(OUT):
        prev=pd.read_csv(OUT); rows=prev.to_dict('records'); done=set(zip(prev.user,prev.eye,prev.code.astype(str)))
    todo=[(u,e,c) for u in USERS_TEST for e in EYES for c in SESSIONS if (u,e,str(c)) not in done]
    print(f"done={len(done)} todo={len(todo)}")
    if todo:
        t0=time.time(); nd=0
        with Pool(2) as pool:
            for rec in pool.imap_unordered(one,todo):
                if rec: rows.append(rec); nd+=1
                pd.DataFrame(rows).to_csv(OUT,index=False)
                if time.time()-t0>a.budget: break
        print(f"+{nd}, remaining={len(todo)-nd}")
        if len(todo)-nd>0: return
    df=pd.DataFrame(rows); df=df[df.user.isin(USERS_TEST)]
    g=df.groupby("user")[STATES].sum(); g["Total"]=g[STATES].sum(axis=1)
    for s in STATES: g[s+"_%"]=(g[s]/g["Total"]*100).round(2)
    out=g.reset_index().rename(columns={"user":"Subject"})
    allr={"Subject":"All",**{s:int(g[s].sum()) for s in STATES},"Total":int(g["Total"].sum())}
    for s in STATES: allr[s+"_%"]=round(g[s].sum()/g["Total"].sum()*100,2)
    out=pd.concat([out,pd.DataFrame([allr])],ignore_index=True)
    out.to_csv(os.path.join(REPORTS_DIR,"tbl_fused_4state_test37_48.csv"),index=False)
    print(out.to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sub=out[out.Subject!="All"]; share=sub.set_index("Subject")[[s+"_%" for s in STATES]]
    ax=share.plot(kind="bar",stacked=True,figsize=(12,5),color=["#70AD47","#ED7D31","#5B9BD5","#FFC000"])
    ax.set_ylabel("% of frames"); ax.set_xlabel("Subject (test 37-48)"); ax.legend(STATES,ncol=4,loc="lower center",bbox_to_anchor=(0.5,1.02))
    ax.set_title("FUSED 4-state (Blink=frame-mask, Saccade=event-burst) per subject")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_fused_4state_test37_48.png"),dpi=130); print("WROTE fig")

if __name__=="__main__": main()
