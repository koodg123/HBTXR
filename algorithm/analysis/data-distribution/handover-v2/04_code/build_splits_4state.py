"""Train/Val/Test per-subject 4-state tables+figures.
 BEFORE interpolation:
   Frame  : from DeanDataset_full_unet/progress_state.json (all frames, UNet no-pupil=Blink;
            valid frames -> Fixation(saccade-session)/Smooth(smooth-session); Saccade~0 @25Hz)
   Event  : from cache/event_4state_all48.csv (20ms event-rate bins)
 AFTER interpolation (EMULATED; no real interpolated data -> assumption stated):
   Blink = frame-UNet blink (unchanged); non-blink split by EVENT Fix:Sacc:Smooth ratio
           (assumes 200Hz frame interp + event densify lets frames resolve saccades like events)
Splits: Train=1-36, Val={37,40,45,48}, Test=37-48 (Val subset of Test)."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import CACHE_DIR, REPORTS_DIR
ST=["Fixation","Saccade","Smooth","Blink"]; COL=["#70AD47","#ED7D31","#5B9BD5","#FFC000"]
TRAIN=set(range(1,37)); VAL={37,40,45,48}; TEST=set(range(37,49))
FOLDER2CODE={"session_1_0_1":"101","session_1_0_2":"102","session_2_0_1":"201","session_2_0_2":"202"}
MOTION={"101":"saccade","102":"smooth","201":"saccade","202":"smooth"}
def splits_of(u):
    s=[]
    if u in TRAIN: s.append("Train")
    if u in VAL: s.append("Val")
    if u in TEST and u not in VAL: s.append("Test")  # avoid double count if you want; we keep Test full separately
    return s

# ---- FRAME before (progress_state) ----
ps=json.load(open("/sessions/busy-happy-newton/mnt/DATASET/eveye/DeanDataset_full_unet/progress_state.json"))
fr={}
for s in ps["session_summaries"]:
    parts=s["session"].replace("\\","/").split("/")
    u=int([p for p in parts if p.startswith("user")][-1][4:]); eye=parts[parts.index(f"user{u}")+1]
    folder=parts[-1]; code=FOLDER2CODE[folder]; mot=MOTION[code]
    blink=s.get("skip_no_ellipse",0); valid=s["frames"]-s.get("skipped",0)
    d=fr.setdefault(u,{k:0 for k in ST}); d["Blink"]+=blink
    d["Fixation" if mot=="saccade" else "Smooth"]+=valid  # Saccade~0 @25Hz
frame_before=pd.DataFrame([{"Subject":u,**v} for u,v in sorted(fr.items())])

# ---- EVENT before (event cache) ----
ev=pd.read_csv(os.path.join(CACHE_DIR,"event_4state_all48.csv"))
event_before=ev.groupby("user")[ST].sum().reset_index().rename(columns={"user":"Subject"})

# ---- AFTER (emulated) ----
rows=[]
for u in range(1,49):
    f=frame_before[frame_before.Subject==u]; e=event_before[event_before.Subject==u]
    if len(f)==0 or len(e)==0: continue
    f=f.iloc[0]; e=e.iloc[0]
    total=f[ST].sum(); blink=f["Blink"]; nonblink=total-blink
    edenom=e["Fixation"]+e["Saccade"]+e["Smooth"]
    if edenom<=0: rows.append({"Subject":u,**{k:f[k] for k in ST}}); continue
    rows.append({"Subject":u,
        "Fixation":nonblink*e["Fixation"]/edenom, "Saccade":nonblink*e["Saccade"]/edenom,
        "Smooth":nonblink*e["Smooth"]/edenom, "Blink":blink})
after_emul=pd.DataFrame(rows)

def finalize(df,name):
    df=df.copy(); df[ST]=df[ST].round().astype(int); df["Total"]=df[ST].sum(axis=1)
    for s in ST: df[s+"_%"]=(df[s]/df["Total"]*100).round(2)
    df["Split"]=df.Subject.apply(lambda u:"Train" if u in TRAIN else ("Test"))  # primary split label
    df["InVal"]=df.Subject.apply(lambda u: u in VAL)
    df=df[["Subject","Split","InVal"]+ST+["Total"]+[s+"_%" for s in ST]]
    df.to_csv(os.path.join(REPORTS_DIR,f"split_4state_{name}.csv"),index=False)
    return df

os.makedirs(REPORTS_DIR,exist_ok=True)
fb=finalize(frame_before,"frame_before"); eb=finalize(event_before,"event_before"); ae=finalize(after_emul,"after_emulated")

# split-level summary (share %)
def split_summary():
    out=[]
    for name,df in [("frame_before",fb),("event_before",eb),("after_emulated",ae)]:
        for split,users in [("Train",TRAIN),("Val",VAL),("Test",TEST)]:
            d=df[df.Subject.isin(users)]; tot=d[ST].sum().sum()
            out.append({"condition":name,"split":split,"n_subj":d.Subject.nunique(),
                        **{s+"_%":round(d[s].sum()/tot*100,2) for s in ST},"total":int(tot)})
    return pd.DataFrame(out)
ss=split_summary(); ss.to_csv(os.path.join(REPORTS_DIR,"split_4state_summary.csv"),index=False)
print("=== SPLIT SUMMARY (share %) ==="); print(ss.to_string(index=False))

# figures: per condition, per-subject stacked share, split-separated
for name,df in [("frame_before",fb),("event_before",eb),("after_emulated",ae)]:
    d=df.sort_values("Subject"); share=d.set_index("Subject")[[s+"_%" for s in ST]]
    ax=share.plot(kind="bar",stacked=True,figsize=(15,5),color=COL,width=0.85)
    ax.axvline(35.5,color="k",ls="--",lw=1); ax.text(17,103,"Train (1-36)",ha="center",fontsize=9)
    ax.text(41.5,103,"Test (37-48)",ha="center",fontsize=9)
    ax.set_ylabel("% frames/bins"); ax.set_xlabel("Subject"); ax.set_ylim(0,108)
    ax.legend(ST,ncol=4,loc="lower center",bbox_to_anchor=(0.5,1.04))
    ax.set_title(f"Per-subject 4-state share — {name} (Val={{37,40,45,48}}⊂Test)")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,f"fig_split_4state_{name}.png"),dpi=130)
print("\nwrote split_4state_{frame_before,event_before,after_emulated}.csv + summary + 3 figs")
