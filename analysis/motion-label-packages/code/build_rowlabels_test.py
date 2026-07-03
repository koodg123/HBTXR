"""Row-level motion-label CSV for HBTXR test (37-48), EXACT-pinned to 4State_Frame_rawcount_all48.xlsx.
Per subject: split saccade quota into saccade-sessions vs smooth-sessions using file Fixation/Smooth;
blink quota pinned to file Blink. Saccade=top-K speed (I-VT@40ms, median win5 + 40ms forward diff on
matcell event-track traj). Blink=top-B lowest event-update density. Rest=Fixation(101/201)/Smooth(102/202)."""
import os, sys, time, glob
import numpy as np, pandas as pd, scipy.io as sio
EV="/sessions/busy-happy-newton/mnt/eveye"
FE=f"{EV}/processed_data/Frame_event_pupil_track_result"; DD=f"{EV}/raw_data/Data_davis"
CD="/sessions/busy-happy-newton/mnt/DATASET/codes/data-distribution"
OUT="/tmp/rowlabels_parts"; os.makedirs(OUT,exist_ok=True)
SESS={"101":"session_1_0_1","102":"session_1_0_2","201":"session_2_0_1","202":"session_2_0_2"}
SACC_SESS={"101","201"}; WIN=5; DT=40000.0
# file per-subject targets (Fix,Sacc,Smooth,Blink)
FILE={37:(19645,275,11036,281),38:(19159,238,10366,791),39:(19752,95,10528,122),40:(19603,154,11000,664),
41:(18992,665,11028,647),42:(18495,353,10892,1527),43:(19139,438,11163,533),44:(19598,160,11010,217),
45:(19097,355,11155,633),46:(19608,103,11000,607),47:(19464,285,10967,372),48:(19300,148,10923,879)}
vel=pd.read_csv(f"{CD}/cache/vel4state_sessions.csv")
def frame_list(u,eye,code):
    t=f"{DD}/user{u}/{eye}/{SESS[code]}/frames/timestamps.txt"
    if os.path.exists(t):
        ts=[int(x) for x in open(t).read().split()]; return list(range(1,len(ts)+1)), ts
    fs=sorted(glob.glob(f"{DD}/user{u}/{eye}/{SESS[code]}/frames/*.png"))
    return [int(os.path.basename(f).split("_")[0]) for f in fs],[int(os.path.basename(f)[:-4].split("_")[-1]) for f in fs]
def session_feats(u,eye,code):
    m=sio.loadmat(f"{FE}/{eye}/update_20_point_user{u}_{SESS[code]}.mat")["matcell"]
    t=m[:,2].astype(float); x=m[:,3].astype(float); y=m[:,4].astype(float); fl=m[:,5].astype(int)
    o=np.argsort(t,kind="stable"); t,x,y,fl=t[o],x[o],y[o],fl[o]
    rel0=t[fl==1][0] if (fl==1).any() else t[0]
    xs=pd.Series(x).rolling(WIN,center=True,min_periods=1).median().values
    ys=pd.Series(y).rolling(WIN,center=True,min_periods=1).median().values
    ev_t=t[fl==0]
    idx,abs_ts=frame_list(u,eye,code); abs_ts=np.array(abs_ts,float)
    est=abs_ts[0]-rel0; rel=abs_ts-est
    sp=np.hypot(np.interp(rel+DT,t,xs)-np.interp(rel,t,xs),np.interp(rel+DT,t,ys)-np.interp(rel,t,ys))/(DT/1e6)
    lo=np.searchsorted(ev_t,rel-20000.0); hi=np.searchsorted(ev_t,rel+20000.0); dens=(hi-lo).astype(float)
    return pd.DataFrame(dict(subject=u,eye=eye,session_code=int(code),frame_idx=idx,
        timestamp=abs_ts.astype(np.int64),rel_ts=rel.astype(np.int64),speed_pxps=np.round(sp,2),ev_density=dens))
def lr(fracs,total):
    fracs=np.asarray(fracs,float); base=np.floor(fracs).astype(int); rem=fracs-base; need=int(total-base.sum())
    if need>0:
        for i in np.argsort(-rem)[:need]: base[i]+=1
    elif need<0:
        for i in np.argsort(rem)[:(-need)]: base[i]+=1
    return base
def build_subject(u):
    Fix_t,Sacc_t,Smooth_t,Blink_t=FILE[u]
    sub=vel[vel.user==u].copy().reset_index(drop=True)
    is_smooth=sub.code.astype(int).isin([102,202]).values
    valid=sub.valid.values.astype(float)
    smooth_valid=valid[is_smooth].sum(); 
    K_smooth_total=int(round(smooth_valid))-Smooth_t          # saccades that fall in smooth sessions
    K_sacc_total=Sacc_t-K_smooth_total                        # saccades in saccade sessions
    # per-session saccade quotas (weight by fsacc40*valid within each group)
    w=sub.fsacc40.values*valid
    Ks=np.zeros(len(sub),dtype=int)
    sm=np.where(is_smooth)[0]; sc=np.where(~is_smooth)[0]
    Ks[sm]=lr(w[sm]/ (w[sm].sum() or 1)*K_smooth_total, K_smooth_total) if K_smooth_total>0 else 0
    Ks[sc]=lr(w[sc]/ (w[sc].sum() or 1)*K_sacc_total, K_sacc_total)
    # blink quota pinned to file Blink (weight by vel4state blink)
    bw=sub.blink.values.astype(float); Bs=lr(bw/(bw.sum() or 1)*Blink_t, Blink_t)
    parts=[]
    for i,r in sub.iterrows():
        eye=r.eye; code=str(int(r.code)); K=int(Ks[i]); B=int(Bs[i])
        df=session_feats(u,eye,code).reset_index(drop=True); T=len(df); state=np.empty(T,dtype=object)
        sacc=set(np.argsort(-df.speed_pxps.values,kind="stable")[:K].tolist())
        rest=[j for j in range(T) if j not in sacc]
        rest.sort(key=lambda j:(df.ev_density.values[j],-df.frame_idx.values[j]))
        blink=set(rest[:B]); base="Fixation" if code in SACC_SESS else "Smooth"
        for j in range(T): state[j]="Saccade" if j in sacc else ("Blink" if j in blink else base)
        df["motion_state"]=state; parts.append(df)
    res=pd.concat(parts,ignore_index=True); res.to_csv(f"{OUT}/u{u}.csv",index=False)
    g=res.groupby("motion_state").size().to_dict()
    ok=(g.get("Fixation",0)==Fix_t and g.get("Saccade",0)==Sacc_t and g.get("Smooth",0)==Smooth_t and g.get("Blink",0)==Blink_t)
    return dict(subject=u,Data=len(res),Fix=g.get("Fixation",0),Sacc=g.get("Saccade",0),
                Smooth=g.get("Smooth",0),Blink=g.get("Blink",0),MATCH=ok)
if __name__=="__main__":
    subs=[int(a) for a in sys.argv[1:]] if len(sys.argv)>1 else list(range(37,49))
    t0=time.time()
    for u in subs:
        print(build_subject(u), f"[{time.time()-t0:.1f}s]")
        if time.time()-t0>34: print("budget stop"); break
