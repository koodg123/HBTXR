import os, sys, numpy as np, pandas as pd
DD="/sessions/busy-happy-newton/mnt/DATASET/codes/data-distribution"
T=os.path.join(DD,"tables")
TRAIN=set(range(1,37)); VAL={37,40,45,48}
def splitlab(u): return "Train" if u in TRAIN else "Test"
ST=["Fixation","Saccade","Smooth","Blink"]

def load4(name):
    d=pd.read_csv(os.path.join(T,f"split_4state_{name}.csv"))
    keep=["Subject","Split","InVal"]+ST+["Total"]+[s+"_%" for s in ST]
    return d[keep]
fb=load4("frame_before"); eb=load4("event_before"); ae=load4("after_emulated")

# search/track after (from after_emulated)
rows=[]
for _,r in ae.iterrows():
    tr=r["Fixation"]+r["Smooth"]; se=r["Saccade"]+r["Blink"]; tot=tr+se
    rows.append(dict(Subject=int(r.Subject),Split=r.Split,InVal=r.InVal,
        Track_inv=int(tr),Search_inv=int(se),Total=int(tot),
        Track_pct=round(100*tr/tot,2),Search_pct=round(100*se/tot,2),
        Track_to_Search=round(tr/se,2) if se>0 else np.nan))
st_after=pd.DataFrame(rows)
st_before=pd.read_csv(os.path.join(T,"tbl_invocation_raw_per_subject.csv"))
veld=pd.read_csv(os.path.join(T,"tbl_velocity_dist_per_subject.csv"))

# velocity 100ms histogram -> per subject %
h=pd.read_csv(os.path.join(DD,"cache","velocity_100ms_hist.csv"))
LAB=["0-50","50-100","100-200","200-300","300-500","500-800","800-1200","1200-2000",">2000"]
g=h.groupby("user").agg({**{l:"sum" for l in LAB},"n_win":"sum","sum_sp":"sum"}).reset_index()
v100=[]
for _,r in g.iterrows():
    tot=r["n_win"]; rec=dict(Subject=int(r.user),Split=splitlab(int(r.user)),InVal=int(r.user) in VAL,
        n_win_100ms=int(tot), mean_pxps=round(r["sum_sp"]/tot,1))
    for l in LAB: rec[l+"_%"]=round(100*r[l]/tot,2)
    v100.append(rec)
v100=pd.DataFrame(v100)

def addAll(df, statecols):  # append an All row (sum counts, recompute %)
    return df  # keep per-subject; All rows added per-sheet below if needed

out=os.path.join(DD,"DataDistribution_HBTXR.xlsx")
readme=pd.DataFrame({"Sheet":[
 "1_4state_before_Frame","1_4state_before_Event","1_4state_after_emul",
 "2_searchtrack_before","2_searchtrack_after","3_velocity_per_subject","4_velocity_100ms_bins"],
 "Description":[
 "증강 전 Frame 4-state (전 프레임; Saccade~0, 25Hz 한계). Count & Ratio(%).",
 "증강 전 Event 4-state (20ms bin). Count & Ratio(%).",
 "증강 후(EMULATED) 4-state: Blink=프레임UNet, 비-blink=이벤트비율 재분배.",
 "증강 전 Search/Track invocation. Frame기준(Saccade못봄→Search=Blink) & Event기준. Count & Ratio.",
 "증강 후(EMULATED) Search/Track: Track=Fixation+Smooth, Search=Saccade+Blink.",
 "증강 전 per-subject pupil velocity 분포(event-track, 평활). percentiles, frac>493px/s.",
 "(3)velocity를 100ms 윈도우로 재계산→velocity 크기 bin(컬럼)별 % (event-track)."],
 "Notes":[
 "Split: Train1-36/Test37-48, InVal=Val{37,40,45,48}⊂Test.","",
 "EMULATED(실 보간데이터 없음): 200Hz 보간 가정.","모드-점유 추정치(실 invocation 로그 아님; user48/201만 실측).",
 "EMULATED.","event-track 미세지터로 절대값 상한; 분포형태가 핵심.",
 "100ms 윈도우 displacement/0.1s. 단위 px/s."]})

with pd.ExcelWriter(out, engine="openpyxl") as w:
    readme.to_excel(w,"README",index=False)
    fb.to_excel(w,"1_4state_before_Frame",index=False)
    eb.to_excel(w,"1_4state_before_Event",index=False)
    ae.to_excel(w,"1_4state_after_emul",index=False)
    st_before.to_excel(w,"2_searchtrack_before",index=False)
    st_after.to_excel(w,"2_searchtrack_after",index=False)
    veld.to_excel(w,"3_velocity_per_subject",index=False)
    v100.to_excel(w,"4_velocity_100ms_bins",index=False)
# light formatting
import openpyxl
wb=openpyxl.load_workbook(out)
for ws in wb.worksheets:
    ws.freeze_panes="A2"
    for c in ws[1]: c.font=openpyxl.styles.Font(bold=True)
    for col in ws.columns:
        wmax=max((len(str(c.value)) for c in col if c.value is not None),default=8)
        ws.column_dimensions[col[0].column_letter].width=min(max(wmax+1,8),22)
wb.save(out)
print("WROTE", out)
print("sheets:", [ws.title for ws in wb.worksheets])
print("\n=== 4_velocity_100ms_bins (head) ===")
print(v100.head(4).to_string(index=False))
print("\n=== 2_searchtrack_after (head) ===")
print(st_after.head(4).to_string(index=False))
