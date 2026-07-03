import scipy.io as sio, pandas as pd, numpy as np, os
EVR=os.environ["EVEYE_ROOT"]
FE=os.path.join(EVR,"processed_data","Frame_event_pupil_track_result")
DD="/sessions/busy-happy-newton/mnt/DATASET/codes/data-distribution"
SESS={101:"session_1_0_1",102:"session_1_0_2",201:"session_2_0_1",202:"session_2_0_2"}
SACC_SESS={101,201}  # pattern1 = saccade-session -> remainder is Fixation
SMOOTH_SESS={102,202}# pattern2 = smooth-session  -> remainder is Smooth
ST=["Fixation","Saccade","Smooth","Blink"]
v=pd.read_csv(DD+"/cache/vel4state_sessions.csv")   # user,eye,code,motion,total(frames),blink(skip),valid,fsacc40,fsacc5
# --- get n_ev (event-track sample count) per session, cached ---
cf=DD+"/cache/event_nrows.csv"
if os.path.exists(cf):
    nev=pd.read_csv(cf)
else:
    rows=[]
    for _,r in v.iterrows():
        f=os.path.join(FE,r.eye,f"update_20_point_user{int(r.user)}_{SESS[int(r.code)]}.mat")
        n=int(sio.loadmat(f)["matcell"].shape[0]) if os.path.exists(f) else 0
        rows.append(dict(user=int(r.user),eye=r.eye,code=int(r.code)))
    nev=pd.DataFrame(rows); nev.to_csv(cf,index=False)
v=v.merge(nev,on=["user","eye","code"])
# --- per-session event-stream 4-state counts ---
def sess_counts(r):
    n=int(r.n_ev)
    bfrac=(r.blink/r.total) if r.total>0 else 0.0          # frame-derived temporal blink fraction
    nb=int(round(bfrac*n))
    valid=n-nb
    sfrac=r.fsacc5 if np.isfinite(r.fsacc5) else 0.0       # event-native 5ms I-VT
    ns=min(int(round(sfrac*n)),valid)
    rem=valid-ns
    fix=rem if int(r.code) in SACC_SESS else 0
    smo=rem if int(r.code) in SMOOTH_SESS else 0
    return pd.Series(dict(Fixation=fix,Saccade=ns,Smooth=smo,Blink=nb))
c=pd.concat([v,v.apply(sess_counts,axis=1)],axis=1)
# --- aggregate per subject ---
def split_of(u):
    inval = u in {37,40,45,48}
    if u<=36: return "Train",inval
    return "Test",inval
agg=c.groupby("user")[["Fixation","Saccade","Smooth","Blink"]].sum().reset_index().rename(columns={"user":"Subject"})
agg["Split"]=agg.Subject.map(lambda u:split_of(u)[0]); agg["InVal"]=agg.Subject.map(lambda u:split_of(u)[1])
agg["Total"]=agg[ST].sum(axis=1)
for s in ST: agg[s+"_%"]=(agg[s]/agg["Total"]*100).round(2)
before=agg[["Subject","Split","InVal"]+ST+["Total"]+[s+"_%" for s in ST]].copy()
# --- after augmentation (same policy: Fix x2, Smooth x2, Saccade x8, Blink x1) ---
MUL={"Fixation":2,"Smooth":2,"Saccade":8,"Blink":1}
after=agg[["Subject","Split","InVal"]].copy()
for s in ST: after[s]=(agg[s]*MUL[s]).round().astype(int)
after["Total"]=after[ST].sum(axis=1)
for s in ST: after[s+"_%"]=(after[s]/after["Total"]*100).round(2)
# --- summary by split ---
def ssum(df,tag):
    out=[]
    for sp,us in [("Train",set(range(1,37))),("Val",{37,40,45,48}),("Test",set(range(37,49)))]:
        d=df[df.Subject.isin(us)]; tot=d[ST].sum().sum()
        out.append({"phase":tag,"split":sp,**{s+"_%":round(100*d[s].sum()/tot,2) for s in ST},"Total":int(tot)})
    return pd.DataFrame(out)
summary=pd.concat([ssum(before,"before(raw event)"),ssum(after,"after_aug")],ignore_index=True)
readme=pd.DataFrame({"item":["population","Saccade method","Blink","Fixation/Smooth","aug policy","note"],
 "value":["Event-track samples (matcell, ~300Hz) — counts in EVENT-SAMPLE units, not frames",
          "Position-velocity I-VT @ event-native 5ms window, speed>493 px/s (fsacc5)",
          "Frame UNet no-pupil temporal fraction (skip_no_ellipse/frames) mapped onto event samples",
          "valid−saccade, by session type (101/201=Fixation, 102/202=Smooth)",
          "Fixation x2, Smooth x2, Saccade x8, Blink x1 (same as DataAug_4state_excl)",
          "Mirrors DataAug_4state_excl.xlsx format but on the EVENT stream. Total=event-track samples."]})
before.to_csv(DD+"/tables/tbl_4state_event_before.csv",index=False)
after.to_csv(DD+"/tables/tbl_4state_event_after.csv",index=False)
tmp="/tmp/EventStream_4state.xlsx"
with pd.ExcelWriter(tmp,engine="openpyxl") as w:
    readme.to_excel(w,sheet_name="README",index=False)
    before.to_excel(w,sheet_name="before_4state",index=False)
    after.to_excel(w,sheet_name="after_aug_4state",index=False)
    summary.to_excel(w,sheet_name="summary_split",index=False)
import openpyxl
wb=openpyxl.load_workbook(tmp)
for ws in wb.worksheets:
    ws.freeze_panes="A2"
    for cl in ws[1]: cl.font=openpyxl.styles.Font(bold=True)
wb.save(tmp)
import shutil
shutil.copy(tmp,DD+"/EventStream_4state_excl.xlsx")
shutil.copy(tmp,"/sessions/busy-happy-newton/mnt/outputs/EventStream_4state_excl.xlsx")
print("WROTE EventStream_4state_excl.xlsx")
print("\n=== n_ev sanity (per session) total event samples =",int(v.n_ev.sum()),"avg/session=%.0f"%v.n_ev.mean())
print("\n=== summary (ratio %) ===")
print(summary.to_string(index=False))
print("\n=== before (raw event) head 8 ===")
print(before.head(8).to_string(index=False))
