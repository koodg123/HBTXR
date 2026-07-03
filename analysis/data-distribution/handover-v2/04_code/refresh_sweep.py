import scipy.io as sio, pandas as pd, numpy as np, os, shutil
FE="/sessions/busy-happy-newton/mnt/eveye/processed_data/Frame_event_pupil_track_result"
DD="/sessions/busy-happy-newton/mnt/DATASET/codes/data-distribution"
SESS={101:"session_1_0_1",102:"session_1_0_2",201:"session_2_0_1",202:"session_2_0_2"}
R=35.0; FRAME=40000.0
TAUS=[0.3,0.5,0.7]; WS=[40,80,120,200,400,1000,10**9]   # ms ; 1e9 = inf (IoU-only)
v=pd.read_csv(DD+"/cache/vel4state_sessions.csv")
def iou_circ(d):
    out=np.zeros_like(d); r2=R*R; m=(d>0)&(d<2*R); dd=d[m]
    a=2*r2*np.arccos(np.clip(dd/(2*R),-1,1))-0.5*dd*np.sqrt(np.clip(4*r2-dd*dd,0,None))
    out[m]=a/(2*np.pi*r2-a); out[d<=0]=1.0; return out
# accumulate per subject: nframes, ntotal(=8*frames via 5ms), and per-(W,tau) N_search
acc={u:{"nframes":0,"frames":0} for u in range(1,49)}
for u in range(1,49):
    for tau in TAUS:
        for W in WS: acc[u][(W,tau)]=0
for _,r in v.iterrows():
    f=os.path.join(FE,r.eye,f"update_20_point_user{int(r.user)}_{SESS[int(r.code)]}.mat")
    if not os.path.exists(f): continue
    M=np.array(sio.loadmat(f)["matcell"],dtype=float); M=M[np.argsort(M[:,2])]; t=M[:,2]
    xs=pd.Series(M[:,3]).rolling(9,center=True,min_periods=1).median().values
    ys=pd.Series(M[:,4]).rolling(9,center=True,min_periods=1).median().values
    g=np.arange(t.min(),t.max(),FRAME)              # 40ms frame grid
    if len(g)<3: continue
    j=np.clip(np.searchsorted(t,g),0,len(t)-1); fx=xs[j]; fy=ys[j]; nf=len(g)
    d=np.hypot(np.diff(fx,prepend=fx[0]),np.diff(fy,prepend=fy[0]))   # 40ms displacement
    iou=iou_circ(d)
    u=int(r.user); acc[u]["nframes"]+=nf; acc[u]["frames"]+=int(r.total)
    k=np.arange(nf)
    for tau in TAUS:
        trig=iou<tau; trig[0]=True
        for W in WS:
            m=max(1,round(W/40.0))
            wd=(k%m==0)
            acc[u][(W,tau)]+=int((trig|wd).sum())
# build per-subject + split sweep
rows=[]
for u in range(1,49):
    a=acc[u]; F=a["nframes"]; ntot=8*F
    rec={"Subject":u,"Split":"Train" if u<=36 else "Test","InVal":u in {37,40,45,48},"frames":F,"N_total":ntot}
    for tau in TAUS:
        for W in WS:
            rec[(W,tau)]=a[(W,tau)]
    rows.append(rec)
P=pd.DataFrame(rows)
def lab(W): return "inf" if W>=10**9 else str(W)
# SWEEP summary per split
sweeprows=[]
for sp,us in [("Train",set(range(1,37))),("Val",{37,40,45,48}),("Test",set(range(37,49)))]:
    d=P[P.Subject.isin(us)]; ntot=d.N_total.sum(); base=d[(40,0.5)].sum()  # W40 search count (=frames)
    for W in WS:
        row={"split":sp,"W_ms":lab(W),"max_staleness_ms":lab(W),"N_total":int(ntot)}
        for tau in TAUS:
            ns=d[(W,tau)].sum()
            row[f"Search_tau{tau}"]=int(ns)
            row[f"Search%_tau{tau}"]=round(100*ns/ntot,2)
        row["reduction x (tau0.5 vs W40)"]=round(base/max(d[(W,0.5)].sum(),1),2)
        sweeprows.append(row)
sweep=pd.DataFrame(sweeprows)
# recommended per-subject: W=200, tau=0.5
rec_rows=[]
for u in range(1,49):
    a=acc[u]; F=a["nframes"]; ntot=8*F; ns=a[(200,0.5)]
    rec_rows.append({"Subject":u,"Split":"Train" if u<=36 else "Test","InVal":u in {37,40,45,48},
        "frames":F,"N_Search":int(ns),"N_Track":int(ntot-ns),"N_total":int(ntot),
        "Search_%":round(100*ns/ntot,2),"Track_%":round(100*(ntot-ns)/ntot,2),
        "vs_W40_search_x":round(F/max(ns,1),2)})
recdf=pd.DataFrame(rec_rows)
readme=pd.DataFrame({"item":["model","Search trigger","Track","sweep 축","metric","reduction","max_staleness","baseline W40","caveat"],
 "value":[
  "프레임(40ms)마다 search 결정. Track=5ms voxel(틱당 1추론, 총 8×frames).",
  "search[k]= (IoU(40ms 변위, r=35px)<τ : 모션 트리거) OR (k%(W/40)==0 : watchdog 강제). k=0 강제.",
  "N_Track = N_total - N_Search = 8×frames - N_Search",
  "W∈{40,80,120,200,400,1000,inf} ms × τ∈{0.3,0.5,0.7}. inf=IoU-only(모션 구동 하한).",
  "N_Search, Search%, Track%, W40 대비 절감배수",
  "reduction× = N_Search(W40)/N_Search(W) at τ0.5. W40=매 프레임 search(=frames, 12.5%).",
  "watchdog 주기 W = anchor 최악 staleness(재anchor 없이 추적 지속 최대시간). inf=모션 없으면 무한.",
  "W40·τ무관 = 12.5%/87.5% (이전 Invocation_final과 동일).",
  "IoU 트리거가 모션(saccade)은 항상 잡으므로 '놓침'은 IoU-게이트 실패 시에만. W는 드리프트/실패 안전망."]})
tmp="/tmp/Refresh_sweep.xlsx"
with pd.ExcelWriter(tmp,engine="openpyxl") as w:
    readme.to_excel(w,sheet_name="README",index=False)
    sweep.to_excel(w,sheet_name="sweep_by_split",index=False)
    recdf.to_excel(w,sheet_name="per_subject_W200_tau0.5",index=False)
import openpyxl
wb=openpyxl.load_workbook(tmp)
for ws in wb.worksheets:
    ws.freeze_panes=("D2" if ws.title.startswith("per_subject") else "A2")
    for cl in ws[1]: cl.font=openpyxl.styles.Font(bold=True)
wb.save(tmp)
shutil.copy(tmp,DD+"/Refresh_sweep.xlsx"); shutil.copy(tmp,"/sessions/busy-happy-newton/mnt/outputs/Refresh_sweep.xlsx")
sweep.to_csv(DD+"/tables/tbl_refresh_sweep.csv",index=False)
print("WROTE Refresh_sweep.xlsx")
print("\n=== sweep_by_split (Test) ===")
print(sweep[sweep.split=='Test'][["W_ms","Search%_tau0.3","Search%_tau0.5","Search%_tau0.7","reduction x (tau0.5 vs W40)"]].to_string(index=False))
print("\n=== sweep_by_split (Train) ===")
print(sweep[sweep.split=='Train'][["W_ms","Search%_tau0.5","reduction x (tau0.5 vs W40)"]].to_string(index=False))
print("\n=== per-subject W200 tau0.5 (Test) ===")
print(recdf[recdf.Split=='Test'][["Subject","N_Search","N_Track","Search_%","vs_W40_search_x"]].to_string(index=False))
