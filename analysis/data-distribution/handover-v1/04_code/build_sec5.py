"""(5) Annotation precision / label noise / label uncertainty.
GT pupil centers are integer-quantized -> uncertainty ~U(+-0.5px) per axis.
A perfect predictor still incurs an Euclidean 'floor' error. Compare to EV-Eye errors."""
import os, sys, glob, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import eveye_metrics as M
from common import USERS, EYES, SESSIONS, session_dir, REPORTS_DIR

# --- verify quantization + quality flags across all GT ---
nint=ntot=0; quals={"good":0,"frontal":0,"good_illumination":0}; nq=0
for u in USERS:
  for eye in EYES:
    for code in ["102","201","202"]:
      cs=glob.glob(os.path.join(session_dir(u,eye,code),"user_*.csv"))
      if not cs: continue
      for _,r in pd.read_csv(cs[0]).iterrows():
        if int(r["region_count"])<=0: continue
        try: sh=json.loads(r["region_shape_attributes"])
        except: continue
        if sh.get("name")!="ellipse": continue
        cx,cy=float(sh["cx"]),float(sh["cy"]); ntot+=1
        if cx==round(cx) and cy==round(cy): nint+=1
        try:
          iq=json.loads(r["region_attributes"]).get("image_quality",{})
          if iq:
            nq+=1
            for k in quals:
              if iq.get(k) in (True,"true","True"): quals[k]+=1
        except: pass

# --- quantization floor via Monte Carlo ---
rng=np.random.default_rng(0); q=rng.uniform(-0.5,0.5,size=(3_000_000,2)); fl=np.hypot(q[:,0],q[:,1])
floor=dict(mean=fl.mean(),median=np.median(fl),p95=np.percentile(fl,95))

df=M.load_pe_allframes()
rows=[]
for track in ["frame","event"]:
    e=df[df.track==track].err
    rows.append(dict(track=track, err_mean=round(e.mean(),3), err_median=round(np.median(e),3),
                     floor_mean=round(floor["mean"],3), floor_median=round(floor["median"],3),
                     pct_of_mean_at_floor=round(100*floor["mean"]/e.mean(),1),
                     ratio_median=round(np.median(e)/floor["median"],2)))
t5=pd.DataFrame(rows)
meta=pd.DataFrame([dict(metric="GT_center_integer_quantized_pct", value=round(100*nint/ntot,2)),
                   dict(metric="per_axis_label_std_px", value=round(1/np.sqrt(12),3)),
                   dict(metric="label_quality_good_pct", value=round(100*quals['good']/max(nq,1),1)),
                   dict(metric="n_annotations", value=ntot),
                   dict(metric="inter_annotator_noise", value="N/A (single annotation per frame)")])
t5.to_csv(os.path.join(REPORTS_DIR,"sec5_label_precision.csv"),index=False)
meta.to_csv(os.path.join(REPORTS_DIR,"sec5_label_meta.csv"),index=False)
print("=== (5) label precision vs reported error ===")
print(f"GT integer-quantized: {100*nint/ntot:.1f}% ({ntot} anns) | quality good={100*quals['good']/max(nq,1):.0f}%")
print(f"quantization floor: mean={floor['mean']:.3f} median={floor['median']:.3f} p95={floor['p95']:.3f} px")
print(t5.to_string(index=False))

# figure: error distribution (frame/event) vs label-quantization floor
fig,ax=plt.subplots(figsize=(8,4.5))
bins=np.linspace(0,4,80)
ax.hist(df[df.track=="frame"].err,bins=bins,density=True,alpha=0.5,label="EV-Eye frame error",color="#1f77b4")
ax.hist(df[df.track=="event"].err,bins=bins,density=True,alpha=0.4,label="EV-Eye event error",color="#d1495b")
ax.hist(fl,bins=bins,density=True,alpha=0.5,label="label quantization floor\n(perfect predictor vs int GT)",color="#777777")
ax.axvline(floor["mean"],color="k",ls="--",lw=1); ax.text(floor["mean"]+0.03,ax.get_ylim()[1]*0.9,f"floor mean {floor['mean']:.2f}px",fontsize=8)
ax.set_xlabel("center distance error (px)"); ax.set_ylabel("density")
ax.set_title("(5) Label uncertainty vs reported error\nframe-track error is comparable to the GT quantization floor"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,"fig_sec5_label_uncertainty.png"),dpi=130)
print("\nwrote sec5_label_precision.csv, sec5_label_meta.csv, fig_sec5_label_uncertainty.png")
