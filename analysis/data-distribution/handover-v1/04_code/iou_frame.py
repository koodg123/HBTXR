"""Frame-track IoU per GT frame = IoU(predicted UNet mask gif, GT mask h5).
Resumable cache: cache/iou_frame.csv. Time-budgeted for chunked runs."""
import os, sys, glob, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, h5py
from PIL import Image
from common import USERS, EYES, SESSIONS, DATA_MASK, CACHE_DIR, session_dir, load_gt_ellipses, parse_frame_filename
PRED="/sessions/busy-happy-newton/mnt/eveye/processed_data/Data_davis_predict"

def pred_map(u,eye,code):
    d=f"{PRED}/user{u}/user{u}/{eye}/{SESSIONS[code][0]}/predict"
    return {parse_frame_filename(os.path.basename(f).replace('_mask.gif',''))[0]: f
            for f in glob.glob(os.path.join(d,"*_mask.gif"))}

def iou_session(u,eye,code):
    h=f"{DATA_MASK}/{eye}/user{u}_{SESSIONS[code][0]}.h5"
    if not os.path.exists(h): return []
    gt=load_gt_ellipses(u,eye,code)
    if len(gt)==0: return []
    with h5py.File(h,'r') as f:
        lab=f['label'][:]            # (W,H,N)
    n=lab.shape[2]
    pm=pred_map(u,eye,code)
    rows=[]
    for i,(_,r) in enumerate(gt.iterrows()):
        if i>=n: break
        gmask = lab[:,:,i].T > 0.5   # -> (H,W)
        fidx=int(r.frame_idx); pf=pm.get(fidx)
        if pf is None: continue
        pmask=np.array(Image.open(pf))>0
        if pmask.shape!=gmask.shape: 
            pmask=pmask.T if pmask.T.shape==gmask.shape else pmask
        inter=np.logical_and(gmask,pmask).sum(); union=np.logical_or(gmask,pmask).sum()
        iou=float(inter/union) if union>0 else np.nan
        rows.append(dict(user=u,eye=eye,code=int(code),frame_idx=fidx,
                         motion=SESSIONS[code][1],iou=round(iou,4)))
    return rows

def build(budget=38):
    t0=time.time(); out=os.path.join(CACHE_DIR,"iou_frame.csv")
    done=set(); allrows=[]
    if os.path.exists(out):
        prev=pd.read_csv(out); allrows=[prev]
        done=set(zip(prev.user,prev.eye,prev.code.astype(str)))
    todo=[(u,e,c) for u in USERS for e in EYES for c in ["102","201","202"] if (u,e,c) not in done]
    proc=0
    for (u,e,c) in todo:
        if time.time()-t0>budget: break
        r=iou_session(u,e,c)
        if r: allrows.append(pd.DataFrame(r))
        proc+=1
    res=pd.concat(allrows,ignore_index=True) if allrows else pd.DataFrame()
    res.to_csv(out,index=False)
    return res,len(todo)-proc

if __name__=="__main__":
    res,rem=build()
    print(f"iou rows={len(res)} sessions_remaining={rem} users_done={res.user.nunique() if len(res) else 0}")
    if len(res): print(res.groupby('motion').iou.median())
