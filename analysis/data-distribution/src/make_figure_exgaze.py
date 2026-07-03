import os,sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import REPORTS_DIR
df=pd.read_csv(os.path.join(REPORTS_DIR,"exgaze_user48_201_per_frame.csv"))
methods=[("err_frame","EX-Gaze\nframe"),("err_event","EX-Gaze\nevent"),("err_cv2","cv2\nbaseline")]
med=[df[m].median() for m,_ in methods]; p95=[np.percentile(df[m].dropna(),95) for m,_ in methods]
fig,ax=plt.subplots(1,2,figsize=(10,4))
x=np.arange(3); w=0.38
ax[0].bar(x-w/2,med,w,label="median",color="#5b8c5a"); ax[0].bar(x+w/2,p95,w,label="p95",color="#e8a33d")
ax[0].set_xticks(x); ax[0].set_xticklabels([l for _,l in methods]); ax[0].set_ylabel("pixel error")
ax[0].set_title("user48/left/201 (saccade session)\nREAL EX-Gaze model vs cv2"); ax[0].legend(fontsize=8)
# per-motion median (fixation vs intermediate)
piv=df.groupby("ivt")[["err_frame","err_event","err_cv2"]].median().reindex(["fixation","smooth"])
piv.index=["fixation","mid-speed"]
piv.plot(kind="bar",ax=ax[1],color=["#1f77b4","#d1495b","#999999"]); ax[1].set_ylabel("median px error")
ax[1].set_title("Error by motion class (I-VT)\n(no high-speed saccade GT in this session)")
ax[1].legend(["frame","event","cv2"],fontsize=8); plt.setp(ax[1].get_xticklabels(),rotation=0)
plt.tight_layout(); p=os.path.join(REPORTS_DIR,"fig_exgaze_user48_201.png"); plt.savefig(p,dpi=130); print("wrote",p)
