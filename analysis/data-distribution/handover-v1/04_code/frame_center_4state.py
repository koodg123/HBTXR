import scipy.io as sio, pandas as pd, numpy as np, os, time, shutil
EVR=os.environ["EVEYE_ROOT"]
FE=os.path.join(EVR,"processed_data","Frame_event_pupil_track_result")
DD="/sessions/busy-happy-newton/mnt/DATASET/codes/data-distribution"
SESS={101:"session_1_0_1",102:"session_1_0_2",201:"session_2_0_1",202:"session_2_0_2"}
SACC_SESS={101,201}; SMOOTH_SESS={102,202}; VSACC=493.0
ST=["Fixation","Saccade","Smooth","Blink"]
v=pd.read_csv(DD+"/cache/vel4state_sessions.csv")  # total(frames),blink(skip),valid

def frame_sacc(f):
    M=np.array(sio.loadmat(f)["matcell"],dtype=float)
    fa=M[M[:,5]==1]                       # frame-rate anchors (40ms, 25Hz)
    o=np.argsort(fa[:,2]); fa=fa[o]
    t=fa[:,2]/1e6; x=fa[:,3]; y=fa[:,4]
    if len(t)<3: return np.nan,np.nan,len(t)
    dt=np.diff(t)
    def frac(xx,yy):
        sp=np.hypot(np.diff(xx),np.diff(yy))/dt
        sp=sp[np.isfinite(sp)]
        return float((sp>VSACC).mean()) if len(sp) else np.nan
    raw=frac(x,y)
    xm=pd.Series(x).rolling(3,center=True,min_periods=1).median().values
    ym=pd.Series(y).rolling(3,center=True,min_periods=1).median().values
    med3=frac(xm,ym)
    return raw,med3,len(t)

cf=DD+"/cache/frame_center_sacc.csv"
if os.path.exists(cf):
    fc=pd.read_csv(cf)
else:
    rows=[]
    for _,r in v.iterrows():
        f=os.path.join(FE,r.eye,f"update_20_point_user{int(r.user)}_{SESS[int(r.code)]}.mat")
        raw,med3,nfa=(np.nan,np.nan,0)
        if os.path.exists(f): raw,med3,nfa=frame_sacc(f)
        rows.append(dict(user=int(r.user),eye=r.eye,code=int(r.code),fsacc_raw=raw,fsacc_med3=med3,n_anchor=nfa))
    fc=pd.DataFrame(rows); fc.to_csv(cf,index=False)
v=v.merge(fc,on=["user","eye","code"])

def build(metric):
    def sc(r):
        valid=int(r.valid); nb=int(r.blink)
        sf=r[metric] if np.isfinite(r[metric]) else 0.0
        ns=min(int(round(sf*valid)),valid); rem=valid-ns
        fix=rem if int(r.code) in SACC_SESS else 0
        smo=rem if int(r.code) in SMOOTH_SESS else 0
        return pd.Series(dict(Fixation=fix,Saccade=ns,Smooth=smo,Blink=nb))
    c=pd.concat([v,v.apply(sc,axis=1)],axis=1)
    agg=c.groupby("user")[ST].sum().reset_index().rename(columns={"user":"Subject"})
    agg["Split"]=agg.Subject.map(lambda u:"Train" if u<=36 else "Test")
    agg["InVal"]=agg.Subject.isin({37,40,45,48})
    agg["Total"]=agg[ST].sum(axis=1)
    for s in ST: agg[s+"_%"]=(agg[s]/agg["Total"]*100).round(2)
    return agg[["Subject","Split","InVal"]+ST+["Total"]+[s+"_%" for s in ST]]

before=build("fsacc_raw")          # literal frame-center (no smoothing)
before_m3=build("fsacc_med3")      # denoised (median-3)
MUL={"Fixation":2,"Smooth":2,"Saccade":8,"Blink":1}
after=before[["Subject","Split","InVal"]].copy()
for s in ST: after[s]=(before[s]*MUL[s]).round().astype(int)
after["Total"]=after[ST].sum(axis=1)
for s in ST: after[s+"_%"]=(after[s]/after["Total"]*100).round(2)

def ssum(df,tag):
    out=[]
    for sp,us in [("Train",set(range(1,37))),("Val",{37,40,45,48}),("Test",set(range(37,49)))]:
        d=df[df.Subject.isin(us)]; tot=d[ST].sum().sum()
        out.append({"phase":tag,"split":sp,**{s+"_%":round(100*d[s].sum()/tot,2) for s in ST},"Total":int(tot)})
    return pd.DataFrame(out)
summary=pd.concat([ssum(before,"before raw"),ssum(before_m3,"before med3"),ssum(after,"after_aug(raw)")],ignore_index=True)
print("RAW saccade%% (Train)=%.2f  MED3 saccade%% (Train)=%.2f"%(
   100*before[before.Subject<=36].Saccade.sum()/before[before.Subject<=36].Total.sum(),
   100*before_m3[before_m3.Subject<=36].Saccade.sum()/before_m3[before_m3.Subject<=36].Total.sum()))
print(summary.to_string(index=False))
print("\nbefore RAW head6:\n",before.head(6).to_string(index=False))
# write workbook (use RAW as main 'before' per literal formula; include med3 sheet)
readme=pd.DataFrame({"item":["population","Saccade method","velocity","Vsacc","Blink/total","Fix/Smooth","note"],
 "value":["Frame-rate centers @25Hz (matcell col5==1 anchors, 40ms) — counts in FRAME units",
          "Literal consecutive-frame finite difference (no event resampling)",
          "v=hypot(dx,dy)/dt, dt=actual inter-frame (~0.04s)",
          "493 px/s (90th pct saccade-session speed)",
          "progress_state frames / skip_no_ellipse (UNet no-pupil)",
          "valid-Saccade by session (101/201=Fixation,102/202=Smooth)",
          "before_4state=RAW(no smoothing); before_med3=median-3 denoised; same format as EventStream/DataAug"]})
tmp="/tmp/FrameCenter_4state.xlsx"
with pd.ExcelWriter(tmp,engine="openpyxl") as w:
    readme.to_excel(w,sheet_name="README",index=False)
    before.to_excel(w,sheet_name="before_4state",index=False)
    before_m3.to_excel(w,sheet_name="before_med3",index=False)
    after.to_excel(w,sheet_name="after_aug_4state",index=False)
    summary.to_excel(w,sheet_name="summary_split",index=False)
import openpyxl
wb=openpyxl.load_workbook(tmp)
for ws in wb.worksheets:
    ws.freeze_panes="A2"
    for cl in ws[1]: cl.font=openpyxl.styles.Font(bold=True)
wb.save(tmp)
before.to_csv(DD+"/tables/tbl_4state_framecenter_before.csv",index=False)
shutil.copy(tmp,DD+"/FrameCenter_4state_excl.xlsx")
shutil.copy(tmp,"/sessions/busy-happy-newton/mnt/outputs/FrameCenter_4state_excl.xlsx")
print("\nWROTE FrameCenter_4state_excl.xlsx")
