"""Event-stream 4-state for test 37-48 from raw events.txt (event-rate based).
Per 20ms bin: rate r (ev/s); m=session-median rate.
  Blink   : r > 12*m   (extreme eyelid burst; approximate)
  Saccade : 4*m < r <= 12*m  (rapid motion burst -- captured at event time-res)
  else    : Fixation (saccade-session 101/201) / Smooth (smooth-session 102/202)
Parallel(2), resumable per session (stores per-session state-bin counts)."""
import os, sys, glob, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import EYES, SESSIONS, CACHE_DIR, REPORTS_DIR, EVEYE_ROOT
RAW=os.path.join(EVEYE_ROOT,"raw_data","Data_davis")
USERS_TEST=list(range(37,49)); STATES=["Fixation","Saccade","Smooth","Blink"]
CACHE=os.path.join(CACHE_DIR,"event_4state_test37_48.csv")
BIN=0.02; BURST=4.0; BLINK=12.0

def one(args):
    u,e,c=args; folder,motion,_=SESSIONS[c]
    f=f"{RAW}/user{u}/{e}/{folder}/events/events.txt"
    if not os.path.exists(f): return None
    t=pd.read_csv(f,sep=r'\s+',header=None,usecols=[0],names=['t'],dtype=np.float64)['t'].values
    rel=(t-t.min())/1e6; nb=int(rel.max()//BIN)+1
    rate=np.bincount((rel//BIN).astype(int),minlength=nb)/BIN
    m=np.median(rate); 
    rec=dict(user=u,eye=e,code=int(c),motion=motion,nbins=nb,
             Fixation=0,Saccade=0,Smooth=0,Blink=0)
    blink=rate>BLINK*m; sacc=(rate>BURST*m)&(~blink)
    base=~(blink|sacc)
    rec["Blink"]=int(blink.sum()); rec["Saccade"]=int(sacc.sum())
    if motion=="saccade": rec["Fixation"]=int(base.sum())
    else: rec["Smooth"]=int(base.sum())
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=float,default=38); a=ap.parse_args()
    os.makedirs(CACHE_DIR,exist_ok=True); os.makedirs(REPORTS_DIR,exist_ok=True)
    done=set(); rows=[]
    if os.path.exists(CACHE):
        prev=pd.read_csv(CACHE); rows=prev.to_dict('records'); done=set(zip(prev.user,prev.eye,prev.code.astype(str)))
    todo=[(u,e,c) for u in USERS_TEST for e in EYES for c in SESSIONS if (u,e,str(c)) not in done]
    print(f"done={len(done)} todo={len(todo)}")
    if todo:
        t0=time.time(); nd=0
        with Pool(2) as pool:
            for rec in pool.imap_unordered(one,todo):
                if rec: rows.append(rec); nd+=1
                pd.DataFrame(rows).to_csv(CACHE,index=False)
                if time.time()-t0>a.budget: break
        print(f"+{nd} sessions, remaining={len(todo)-nd}")
        if len(todo)-nd>0: return
    # aggregate per subject
    df=pd.DataFrame(rows); df=df[df.user.isin(USERS_TEST)]
    g=df.groupby("user")[STATES].sum(); g["Total_bins"]=g[STATES].sum(axis=1)
    for s in STATES: g[s+"_%"]=(g[s]/g["Total_bins"]*100).round(2)
    out=g.reset_index().rename(columns={"user":"Subject"})
    allr={"Subject":"All",**{s:int(g[s].sum()) for s in STATES},"Total_bins":int(g["Total_bins"].sum())}
    for s in STATES: allr[s+"_%"]=round(g[s].sum()/g["Total_bins"].sum()*100,2)
    out=pd.concat([out,pd.DataFrame([allr])],ignore_index=True)
    out.to_csv(os.path.join(REPORTS_DIR,"tbl_event_4state_test37_48.csv"),index=False)
    print(out.to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sub=out[out.Subject!="All"]; share=sub.set_index("Subject")[[s+"_%" for s in STATES]]
        ax=share.plot(kind="bar",stacked=True,figsize=(12,5),color=["#70AD47","#ED7D31","#5B9BD5","#FFC000"])
        ax.set_ylabel("% of 20ms bins"); ax.set_xlabel("Subject (test 37-48)")
        ax.legend(STATES,ncol=4,loc="lower center",bbox_to_anchor=(0.5,1.02))
        ax.set_title("EVENT-stream per-subject 4-state share (event-rate based)")
        plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_event_4state_test37_48.png"),dpi=130); print("WROTE fig")
    except Exception as ex: print("fig skip",ex)

if __name__=="__main__": main()
