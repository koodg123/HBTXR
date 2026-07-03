"""Mask-based 4-state for test subjects 37-48 (fast). Uses precomputed UNet masks
(Data_davis_predict): Blink = empty mask; Saccade = speed>v_sacc; else Fix/Smooth by session.
Parallel(2), time-budgeted, resumable. Separate cache."""
import os, sys, glob, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from PIL import Image
from multiprocessing import Pool
from common import EYES, SESSIONS, CACHE_DIR, REPORTS_DIR, EVEYE_ROOT, parse_frame_filename

PRED = os.path.join(EVEYE_ROOT, "processed_data", "Data_davis_predict")
USERS_TEST = list(range(37, 49))
CACHE = os.path.join(CACHE_DIR, "frames_4state_test37_48.csv")
VSACC, STRIDE, MIN_AREA = 493.0, 2, 30
STATES = ["Fixation", "Saccade", "Smooth", "Blink"]

def classify(args):
    u, e, c = args
    folder = SESSIONS[c][0]; motion = SESSIONS[c][1]
    d = os.path.join(PRED, f"user{u}", f"user{u}", e, folder, "predict")
    fs = sorted(glob.glob(os.path.join(d, "*_mask.gif")), key=lambda p: parse_frame_filename(os.path.basename(p).replace("_mask.gif",""))[0])
    fs = fs[::STRIDE]
    idx=[]; ts=[]; cx=[]; cy=[]; pres=[]
    for f in fs:
        i,t = parse_frame_filename(os.path.basename(f).replace("_mask.gif",""))
        m = np.asarray(Image.open(f))>0; a = int(m.sum())
        idx.append(i); ts.append(t)
        if a < MIN_AREA: pres.append(False); cx.append(np.nan); cy.append(np.nan)
        else:
            ys,xs=np.where(m); pres.append(True); cx.append(xs.mean()); cy.append(ys.mean())
    n=len(idx); ts=np.array(ts,float); cx=np.array(cx); cy=np.array(cy)
    sp=np.full(n,np.nan)
    for k in range(n):
        a=max(0,k-1); b=min(n-1,k+1)
        if pres[a] and pres[b] and b>a:
            dt=(ts[b]-ts[a])/1e6
            if dt>0: sp[k]=np.hypot(cx[b]-cx[a],cy[b]-cy[a])/dt
    st=[]
    for k in range(n):
        if not pres[k]: st.append("Blink")
        elif np.isfinite(sp[k]) and sp[k]>VSACC: st.append("Saccade")
        elif motion=="saccade": st.append("Fixation")
        else: st.append("Smooth")
    return pd.DataFrame(dict(user=u,eye=e,code=int(c),frame_idx=idx,state=st))

def agg(res):
    res=res[res.user.isin(USERS_TEST)]
    tab=(res.groupby(["user","state"]).size().unstack(fill_value=0).reindex(columns=STATES,fill_value=0).reindex(USERS_TEST,fill_value=0))
    tab.index.name="Subject"; tab["Total"]=tab.sum(axis=1)
    out=tab.reset_index(); out=pd.concat([out,pd.DataFrame([{"Subject":"All",**{s:int(tab[s].sum()) for s in STATES+["Total"]}}])],ignore_index=True)
    out.to_csv(os.path.join(REPORTS_DIR,"tbl_4state_test37_48.csv"),index=False)
    print("WROTE tbl_4state_test37_48.csv"); print(out.to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        share=tab[STATES].div(tab[STATES].sum(axis=1),axis=0)*100
        ax=share.plot(kind="bar",stacked=True,figsize=(12,5),color=["#70AD47","#ED7D31","#5B9BD5","#FFC000"])
        ax.set_ylabel("% frames"); ax.set_xlabel("Subject (test 37-48)"); ax.legend(ncol=4,loc="lower center",bbox_to_anchor=(0.5,1.02))
        ax.set_title("Test 37-48 per-subject 4-state share (mask-based; Fixation/Saccade/Smooth/Blink)")
        plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_4state_test37_48.png"),dpi=130); print("WROTE fig")
    except Exception as ex: print("fig skipped",ex)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=float,default=38); a=ap.parse_args()
    os.makedirs(CACHE_DIR,exist_ok=True); os.makedirs(REPORTS_DIR,exist_ok=True)
    done=set(); parts=[]
    if os.path.exists(CACHE):
        prev=pd.read_csv(CACHE); parts=[prev]; done=set(zip(prev.user,prev.eye,prev.code.astype(str)))
    todo=[(u,e,c) for u in USERS_TEST for e in EYES for c in SESSIONS if (u,e,str(c)) not in done]
    print(f"done={len(done)} todo={len(todo)} stride={STRIDE}")
    if not todo: agg(pd.concat(parts,ignore_index=True)); return
    t0=time.time(); nd=0
    with Pool(2) as pool:
        for df in pool.imap_unordered(classify,todo):
            parts.append(df); nd+=1
            pd.concat(parts,ignore_index=True).to_csv(CACHE,index=False)
            if time.time()-t0>a.budget: break
    res=pd.concat(parts,ignore_index=True)
    print(f"+{nd} sessions this run, remaining={len(todo)-nd}, rows={len(res)}")
    if len(todo)-nd==0: agg(res)

if __name__=="__main__": main()
