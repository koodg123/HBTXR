import scipy.io as sio, pandas as pd, numpy as np, os, shutil
FE="/sessions/busy-happy-newton/mnt/eveye/processed_data/Frame_event_pupil_track_result"
DD="/sessions/busy-happy-newton/mnt/DATASET/codes/data-distribution"
SESS={101:"session_1_0_1",102:"session_1_0_2",201:"session_2_0_1",202:"session_2_0_2"}
R=35.0; TAUS=[0.3,0.5,0.7]; TICK=5000.0  # 5ms
v=pd.read_csv(DD+"/cache/vel4state_sessions.csv")  # total=frames

def iou_circ(d):
    out=np.zeros_like(d); r2=R*R
    m=(d>0)&(d<2*R); dd=d[m]
    a=2*r2*np.arccos(np.clip(dd/(2*R),-1,1))-0.5*dd*np.sqrt(np.clip(4*r2-dd*dd,0,None))
    out[m]=a/(2*np.pi*r2-a); out[d<=0]=1.0
    return out

def sim(f):
    M=np.array(sio.loadmat(f)["matcell"],dtype=float); M=M[np.argsort(M[:,2])]
    t=M[:,2]
    xs=pd.Series(M[:,3]).rolling(9,center=True,min_periods=1).median().values
    ys=pd.Series(M[:,4]).rolling(9,center=True,min_periods=1).median().values
    g=np.arange(t.min(),t.max(),TICK)
    if len(g)<9: return None
    j=np.clip(np.searchsorted(t,g),0,len(t)-1); gx=xs[j]; gy=ys[j]; n=len(g)
    aidx=(np.arange(n)//8)*8                 # anchor = pos at last 40ms frame tick
    d=np.hypot(gx-gx[aidx],gy-gy[aidx]); iou=iou_circ(d)
    track=(np.arange(n)%8!=0)
    fid=np.arange(n)//8; nframes=fid.max()+1
    res={"n":n,"ntrack":int(track.sum()),"nframes":int(nframes)}
    for tau in TAUS:
        fail=(iou<tau)&track
        res[f"deg_{tau}"]=int(fail.sum())
        # frames with any degraded track tick
        res[f"need_{tau}"]=int(len(np.unique(fid[fail])))
    return res

rows=[]
for _,r in v.iterrows():
    f=os.path.join(FE,r.eye,f"update_20_point_user{int(r.user)}_{SESS[int(r.code)]}.mat")
    if not os.path.exists(f): continue
    s=sim(f)
    if s: s.update(user=int(r.user),frames=int(r.total)); rows.append(s)
P=pd.DataFrame(rows)
g=P.groupby("user").sum().reset_index()
g["Split"]=g.user.map(lambda u:"Train" if u<=36 else "Test"); g["InVal"]=g.user.isin({37,40,45,48})
# canonical invocation (deterministic): Search=frames, Track=7*frames, total=8*frames
g["N_Search"]=g.frames; g["N_Track"]=7*g.frames; g["N_total"]=8*g.frames
g["Search_%"]=round(100*1/8,2); g["Track_%"]=round(100*7/8,2)
for tau in TAUS:
    g[f"track_deg%@{tau}"]=(100*g[f"deg_{tau}"]/g.ntrack).round(2)
    g[f"needed_refresh%@{tau}"]=(100*g[f"need_{tau}"]/g.nframes).round(2)
inv=g[["user","Split","InVal","frames","N_Search","N_Track","N_total","Search_%","Track_%",
       "track_deg%@0.5","needed_refresh%@0.5"]].rename(columns={"user":"Subject"})
def ssum():
    out=[]
    for sp,us in [("Train",set(range(1,37))),("Val",{37,40,45,48}),("Test",set(range(37,49)))]:
        d=g[g.user.isin(us)]
        row={"split":sp,"N_Search":int(d.N_Search.sum()),"N_Track":int(d.N_Track.sum()),
             "N_total":int(d.N_total.sum()),"Search_%":12.5,"Track_%":87.5}
        for tau in TAUS:
            row[f"track_deg%@{tau}"]=round(100*d[f"deg_{tau}"].sum()/d.ntrack.sum(),2)
            row[f"needed%@{tau}"]=round(100*d[f"need_{tau}"].sum()/d.nframes.sum(),2)
        out.append(row)
    return pd.DataFrame(out)
summary=ssum()
readme=pd.DataFrame({"item":["config","Track","Search","invocation 카운트","IoU 분석","track_deg%","needed_refresh%","params","caveat"],
 "value":[
  "Track=5ms voxel(200Hz), Search=IoU게이트+40ms 자동refresh",
  "이벤트 5ms voxel, 프레임 사이 7틱/40ms",
  "프레임 입력→40ms마다만 가능. 자동refresh=매 프레임 → Search=frames",
  "N_Search=frames, N_Track=7×frames, N_total=8×frames → Search 12.5% / Track 87.5% (결정적)",
  "anchor=직전 40ms 프레임 위치, 현재 track 위치와 동공원(r=35px) IoU. τ 미만이면 품질저하",
  "track 틱 중 IoU<τ 비율 (추적이 anchor에서 벗어난 정도)",
  "한 40ms 구간 내 IoU<τ가 한 번이라도 발생→그 프레임 refresh가 '정당'. 나머지는 redundant",
  "r=35px(GT a≈37,b≈32 평균), τ∈{0.3,0.5,0.7}, voxel=5ms, frame=40ms",
  "Search는 프레임율(25Hz)에 묶여 12.5% 고정. IoU게이트는 redundant refresh 식별용. 위치=event-track 평활"]})
tmp="/tmp/Invocation_final.xlsx"
with pd.ExcelWriter(tmp,engine="openpyxl") as w:
    readme.to_excel(w,sheet_name="README",index=False)
    inv.to_excel(w,sheet_name="invocation_per_subject",index=False)
    summary.to_excel(w,sheet_name="summary_split",index=False)
import openpyxl
wb=openpyxl.load_workbook(tmp)
for ws in wb.worksheets:
    ws.freeze_panes=("D2" if ws.title=="invocation_per_subject" else "A2")
    for cl in ws[1]: cl.font=openpyxl.styles.Font(bold=True)
wb.save(tmp)
shutil.copy(tmp,DD+"/Invocation_final.xlsx"); shutil.copy(tmp,"/sessions/busy-happy-newton/mnt/outputs/Invocation_final.xlsx")
inv.to_csv(DD+"/tables/tbl_invocation_final.csv",index=False)
print("WROTE Invocation_final.xlsx")
print("\n=== summary_split ===")
print(summary.to_string(index=False))
print("\n=== invocation per-subject (Test 37-48) ===")
print(inv[inv.Split=='Test'].to_string(index=False))
