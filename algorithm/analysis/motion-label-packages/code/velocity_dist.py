"""Per-subject pupil-velocity distribution (RAW, before augmentation), from the
event-track trajectory (Frame_event_pupil_track_result matcell; high temporal res).
Smooth x,y (rolling median win=9) -> speed(px/s) central diff. Subsample per session
for pooled per-subject distribution. Parallel(2), resumable per session."""
import os, sys, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.io as sio
from multiprocessing import Pool
EVR=os.environ.get("EVEYE_ROOT","/sessions/busy-happy-newton/mnt/eveye")
FE=os.path.join(EVR,"processed_data","Frame_event_pupil_track_result")
MT=os.environ.get("MT_CODES","/sessions/busy-happy-newton/mnt/DATASET/codes")
DD=os.path.join(MT,"data-distribution"); CACHE=os.path.join(DD,"cache","velocity_subsample.csv")
SESS={"101":("session_1_0_1","saccade"),"102":("session_1_0_2","smooth"),
      "201":("session_2_0_1","saccade"),"202":("session_2_0_2","smooth")}
TRAIN=set(range(1,37)); VAL={37,40,45,48}
NSUB=1500; WIN=9

def one(args):
    u,eye,code=args; folder,mot=SESS[code]
    f=os.path.join(FE,eye,f"update_20_point_user{u}_{folder}.mat")
    if not os.path.exists(f): return None
    m=sio.loadmat(f)["matcell"]; t=m[:,2].astype(float); x=m[:,3].astype(float); y=m[:,4].astype(float)
    o=np.argsort(t); t,x,y=t[o],x[o],y[o]
    xs=pd.Series(x).rolling(WIN,center=True,min_periods=1).median().values
    ys=pd.Series(y).rolling(WIN,center=True,min_periods=1).median().values
    dt=(t[2:]-t[:-2])/1e6; disp=np.hypot(xs[2:]-xs[:-2],ys[2:]-ys[:-2])
    sp=disp[dt>0]/dt[dt>0]; sp=sp[np.isfinite(sp)]
    if len(sp)==0: return None
    idx=np.random.default_rng(u*100+int(code)).choice(len(sp),min(NSUB,len(sp)),replace=False)
    sub=sp[idx]
    return pd.DataFrame(dict(user=u,eye=eye,code=int(code),motion=mot,speed=np.round(sub,2)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=float,default=38); a=ap.parse_args()
    os.makedirs(os.path.dirname(CACHE),exist_ok=True)
    done=set(); parts=[]
    if os.path.exists(CACHE):
        prev=pd.read_csv(CACHE); parts=[prev]; done=set(zip(prev.user,prev.eye,prev.code.astype(str)))
    todo=[(u,e,c) for u in range(1,49) for e in ["left","right"] for c in SESS if (u,e,str(c)) not in done]
    print(f"done={len(done)} todo={len(todo)}")
    if todo:
        t0=time.time(); nd=0
        with Pool(2) as pool:
            for r in pool.imap_unordered(one,todo):
                if r is not None: parts.append(r); nd+=1
                if nd%8==0 or time.time()-t0>a.budget:
                    pd.concat(parts,ignore_index=True).to_csv(CACHE,index=False)
                if time.time()-t0>a.budget: break
        pd.concat(parts,ignore_index=True).to_csv(CACHE,index=False)
        print(f"+{nd}, remaining≈{len(todo)-nd}")
        if len(todo)-nd>0: return
    finalize()

def finalize():
    df=pd.read_csv(CACHE)
    rows=[]
    for u in sorted(df.user.unique()):
        s=df[df.user==u].speed.values
        rows.append(dict(Subject=int(u), Split="Train" if u in TRAIN else "Test", InVal=(u in VAL),
            n=len(s), mean=round(s.mean(),1),
            p25=round(np.percentile(s,25),1), p50=round(np.percentile(s,50),1),
            p75=round(np.percentile(s,75),1), p90=round(np.percentile(s,90),1),
            p95=round(np.percentile(s,95),1), p99=round(np.percentile(s,99),1),
            frac_gt493=round(float((s>493).mean()),3)))
    t=pd.DataFrame(rows); t.to_csv(os.path.join(DD,"tables","tbl_velocity_dist_per_subject.csv"),index=False)
    print(t.to_string(index=False))
    # figure: per-subject violin of log10(speed+1), split-separated
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    users=sorted(df.user.unique())
    data=[np.log10(df[df.user==u].speed.values+1) for u in users]
    fig,ax=plt.subplots(figsize=(16,5))
    vp=ax.violinplot(data,positions=np.arange(len(users)),widths=0.85,showmedians=True)
    for i,b in enumerate(vp['bodies']): b.set_facecolor("#4472C4" if users[i] in TRAIN else "#ED7D31"); b.set_alpha(0.6)
    ax.axhline(np.log10(493+1),color="r",ls="--",lw=1,label="v_sacc=493 px/s")
    ax.axvline(35.5,color="k",ls="--",lw=1)
    ax.set_xticks(np.arange(len(users))); ax.set_xticklabels(users,fontsize=6)
    ax.set_ylabel("log10(speed+1)  [px/s]"); ax.set_xlabel("Subject (blue=Train1-36, orange=Test37-48)")
    ax.set_title("Per-subject pupil-velocity distribution (RAW, event-track; before augmentation)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(DD,"figures","fig_velocity_dist_per_subject.png"),dpi=130); print("WROTE fig")

if __name__=="__main__": main()
