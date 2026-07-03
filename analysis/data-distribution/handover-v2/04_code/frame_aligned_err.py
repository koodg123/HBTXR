"""Frame-aligned FRAME-track center error per GT frame = ||ellipse-center(pred mask) - GT||.
Fits an ellipse to the predicted UNet mask (like EV-Eye) -> center. Same frame clock as GT.
Resumable/time-budgeted. Cache: cache/frame_aligned_err_ellipse.csv."""
import os, sys, glob, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, cv2
from PIL import Image
from common import USERS, EYES, SESSIONS, CACHE_DIR, load_gt_ellipses, parse_frame_filename
PRED="/sessions/busy-happy-newton/mnt/eveye/processed_data/Data_davis_predict"

def pmap(u,eye,code):
    d=f"{PRED}/user{u}/user{u}/{eye}/{SESSIONS[code][0]}/predict"
    return {parse_frame_filename(os.path.basename(f).replace('_mask.gif',''))[0]: f
            for f in glob.glob(os.path.join(d,"*_mask.gif"))}

def center_of(mask):
    m=(mask>0).astype(np.uint8)
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return np.nan,np.nan
    c=max(cnts,key=cv2.contourArea)
    if len(c)>=5:
        (cx,cy),_,_=cv2.fitEllipse(c); return float(cx),float(cy)
    M=cv2.moments(c)
    if M["m00"]==0: return np.nan,np.nan
    return float(M["m10"]/M["m00"]),float(M["m01"]/M["m00"])

def build(budget=38):
    t0=time.time(); out=os.path.join(CACHE_DIR,"frame_aligned_err_ellipse.csv")
    done=set(); rows=[]
    if os.path.exists(out):
        prev=pd.read_csv(out); rows=[prev]; done=set(zip(prev.user,prev.eye,prev.code.astype(str)))
    todo=[(u,e,c) for u in USERS for e in EYES for c in ["102","201","202"] if (u,e,str(c)) not in done]
    new=[]; proc=0
    for (u,eye,code) in todo:
        if time.time()-t0>budget: break
        gt=load_gt_ellipses(u,eye,code)
        if len(gt)==0: proc+=1; continue
        pm=pmap(u,eye,code)
        for _,r in gt.iterrows():
            pf=pm.get(int(r.frame_idx))
            if not pf: continue
            cx,cy=center_of(np.array(Image.open(pf)))
            err=float(np.hypot(cx-r.cx,cy-r.cy)) if np.isfinite(cx) else np.nan
            new.append(dict(user=u,eye=eye,code=int(code),frame_idx=int(r.frame_idx),track="frame",
                            pred_x=cx,pred_y=cy,err=round(err,3) if np.isfinite(err) else np.nan))
        proc+=1
    if new: rows.append(pd.DataFrame(new))
    df=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    df.to_csv(out,index=False)
    return df,len(todo)-proc

if __name__=="__main__":
    df,rem=build()
    e=df[df.err.notna()].err
    print(f"frame-aligned(ellipse) rows={len(df)} remaining={rem} median_err={e.median():.3f}px (official ~0.55)")
