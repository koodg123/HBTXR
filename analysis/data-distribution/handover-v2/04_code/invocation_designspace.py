import scipy.io as sio, pandas as pd, numpy as np, os, time, shutil
FE="/sessions/busy-happy-newton/mnt/eveye/processed_data/Frame_event_pupil_track_result"
DD="/sessions/busy-happy-newton/mnt/DATASET/codes/data-distribution"
SESS={101:"session_1_0_1",102:"session_1_0_2",201:"session_2_0_1",202:"session_2_0_2"}
SACC_SESS={101,201}; VSACC=493.0; EWIN=5000.0
v=pd.read_csv(DD+"/cache/vel4state_sessions.csv")

def count_ep(sac,gap):
    if sac.sum()==0: return 0
    idx=np.where(sac)[0]
    return int((np.diff(idx)>gap).sum())+1

def prim(f,code,blink_frames):
    M=np.array(sio.loadmat(f)["matcell"],dtype=float)
    fa=M[M[:,5]==1]; fa=fa[np.argsort(fa[:,2])]; tf=fa[:,2]/1e6
    Ms=M[np.argsort(M[:,2])]; t=Ms[:,2]; dur=(t.max()-t.min())/1e6
    dtf=np.diff(tf); n_blink_ep=int((dtf>0.060).sum())
    xs=pd.Series(Ms[:,3]).rolling(9,center=True,min_periods=1).median().values
    ys=pd.Series(Ms[:,4]).rolling(9,center=True,min_periods=1).median().values
    g=np.arange(t.min(),t.max(),EWIN); j=np.clip(np.searchsorted(t,g),0,len(t)-1)
    ve=np.hypot(np.diff(xs[j]),np.diff(ys[j]))/(EWIN/1e6); sac=ve>VSACC
    T_sacc=int(sac.sum())*EWIN/1e6; T_blink=blink_frames*0.040
    ss=int(code) in SACC_SESS
    T_fs=max(dur-T_sacc-T_blink,0.0)
    return dict(dur=dur,T_sacc=T_sacc,T_blink=T_blink,
                T_fix=T_fs if ss else 0.0, T_smooth=0.0 if ss else T_fs,
                n_sacc_ep=count_ep(sac,1), n_sacc_ep_db=count_ep(sac,10), n_blink_ep=n_blink_ep)
rows=[]
for _,r in v.iterrows():
    f=os.path.join(FE,r.eye,f"update_20_point_user{int(r.user)}_{SESS[int(r.code)]}.mat")
    if not os.path.exists(f): continue
    p=prim(f,int(r.code),int(r.blink)); p["user"]=int(r.user); rows.append(p)
P=pd.DataFrame(rows)
ag=P.groupby("user").sum().reset_index()
ag["Split"]=ag.user.map(lambda u:"Train" if u<=36 else "Test"); ag["InVal"]=ag.user.isin({37,40,45,48})
ag=ag.rename(columns={"user":"Subject"})

# ---------- N_Track grid: cadence x idle ----------
def ntrack(row,c,idle):
    Ts,Tm,Tf,Tb=row.T_sacc,row.T_smooth,row.T_fix,row.T_blink
    if c=="adapt":
        base=Ts/0.005+Tm/0.010+Tf/0.040
        if idle=="always": return base+Tb/0.040
        return base                      # fixlow_blinkhold & skipboth: blink hold; skipboth also drops fix
    c=float(c)/1000.0   # ms -> s
    if idle=="always":            return (Ts+Tm+Tf+Tb)/c
    if idle=="fixlow_blinkhold":  return (Ts+Tm)/c + Tf/0.040
    if idle=="skipboth":          return (Ts+Tm)/c
CAD=[("5",0.005),("10",0.010),("40",0.040),("adapt",None)]
IDLE=["always","fixlow_blinkhold","skipboth"]
trk=ag[["Subject","Split","InVal"]].copy()
for cname,_ in CAD:
    for idle in IDLE:
        trk[f"T_{cname}ms_{idle}"]=ag.apply(lambda r:int(round(ntrack(r,cname,idle))),axis=1)

# ---------- N_Search grid: trigger ----------
srch=ag[["Subject","Split","InVal"]].copy()
srch["S_state"]      =(1+ag.n_blink_ep+ag.n_sacc_ep_db).astype(int)
srch["S_state_raw"]  =(1+ag.n_blink_ep+ag.n_sacc_ep).astype(int)
srch["S_frameperiodic"]=(ag.dur/0.040).round().astype(int)
srch["S_refresh1s_lowconf"]=(ag.dur/1.0 + ag.n_sacc_ep_db).round().astype(int)
srch["S_lockloss"]   =(1+ag.n_sacc_ep_db).astype(int)

# ---------- example full configs (S,T,total,ratio) ----------
def full(tag,c,idle,strig):
    T=ag.apply(lambda r:ntrack(r,c,idle),axis=1)
    S=srch[strig].astype(float)
    tot=S+T
    return pd.DataFrame({"config":tag,"Subject":ag.Subject,"Split":ag.Split,
        "N_Search":S.round().astype(int),"N_Track":T.round().astype(int),
        "N_total":tot.round().astype(int),"Search_%":(100*S/tot).round(2),"Track_%":(100*T/tot).round(2)})
EX=[("R1 rec: Track5ms+fixlow/blinkhold, Search=state","5","fixlow_blinkhold","S_state"),
    ("R2 event-aggressive: Track5ms+always, Search=lockloss","5","always","S_lockloss"),
    ("R3 frame-heavy: Track40ms+always, Search=frameperiodic","40","always","S_frameperiodic"),
    ("R4 adaptive: Track adapt+fixlow/blinkhold, Search=state","adapt","fixlow_blinkhold","S_state")]
examples=pd.concat([full(*e) for e in EX],ignore_index=True)
def exsum():
    out=[]
    for tag in [e[0] for e in EX]:
        d=examples[examples.config==tag]
        for sp,us in [("Train",set(range(1,37))),("Test",set(range(37,49)))]:
            q=d[d.Subject.isin(us)]; S=q.N_Search.sum(); T=q.N_Track.sum(); tot=S+T
            out.append({"config":tag,"split":sp,"N_Search":int(S),"N_Track":int(T),
                        "Search_%":round(100*S/tot,2),"Track_%":round(100*T/tot,2)})
    return pd.DataFrame(out)
exsummary=exsum()

primout=ag[["Subject","Split","InVal","dur","T_fix","T_smooth","T_sacc","T_blink","n_sacc_ep","n_sacc_ep_db","n_blink_ep"]].round(2)
readme=pd.DataFrame({"item":[
 "invocation 정의","Event=voxel","Frame vs Event","Track cadence(ΔTv)","Search trigger","유휴정책",
 "primitives","saccade 에피소드","N_Track 공식","N_Search 공식","caveat"],
 "value":[
 "scheduler가 추론커널을 dispatch한 횟수. 입력 sample 수 아님. tick=voxel cadence.",
 "이벤트 N개→voxel 1개→invocation 1회. Event invoc = 시간/ΔTv (voxel수), 이벤트수 아님.",
 "한 타임라인 위 택일(Search=frame, Track=event-voxel). 합산하면 이중계산.",
 "{5,10,40 ms, adaptive(sacc5/smooth10/fix40)} 4종 — 시트 N_Track_grid 열",
 "{state(init+blink복귀+saccade onset), frameperiodic(25Hz), refresh1s+lowconf, lockloss} — 시트 N_Search_grid",
 "{always, fixlow_blinkhold(고정40ms·blink0), skipboth} — N_Track_grid 열",
 "dur=녹화s, T_state=체류s, n_*_ep=에피소드수 (primitives 시트)",
 "raw=5ms run, db=50ms 병합(생리적 ~120/session에 근접). state/lockloss는 db 사용",
 "always:(Ts+Tm+Tf+Tb)/c ; fixlow_blinkhold:(Ts+Tm)/c+Tf/40ms ; skipboth:(Ts+Tm)/c ; adapt:Ts/5+Tm/10+Tf/40(+Tb/40 if always)",
 "state:1+blink_ep+sacc_ep_db ; frameperiodic:dur/40ms ; refresh1s_lowconf:dur/1s+sacc_ep_db ; lockloss:1+sacc_ep_db",
 "5ms saccade 에피소드는 과분절(raw~347/sess vs 생리 ~120). refresh주기 1s·fix저rate 40ms는 가정값(조정가능). T_blink=skip×40ms."]})
tmp="/tmp/Invocation_designspace.xlsx"
with pd.ExcelWriter(tmp,engine="openpyxl") as w:
    readme.to_excel(w,sheet_name="README",index=False)
    primout.to_excel(w,sheet_name="primitives",index=False)
    trk.to_excel(w,sheet_name="N_Track_grid",index=False)
    srch.to_excel(w,sheet_name="N_Search_grid",index=False)
    examples.to_excel(w,sheet_name="examples_fullconfig",index=False)
    exsummary.to_excel(w,sheet_name="examples_summary",index=False)
import openpyxl
wb=openpyxl.load_workbook(tmp)
for ws in wb.worksheets:
    ws.freeze_panes=("D2" if ws.title in("primitives","N_Track_grid","N_Search_grid") else "A2")
    for cl in ws[1]: cl.font=openpyxl.styles.Font(bold=True)
wb.save(tmp)
shutil.copy(tmp,DD+"/Invocation_designspace.xlsx")
shutil.copy(tmp,"/sessions/busy-happy-newton/mnt/outputs/Invocation_designspace.xlsx")
trk.to_csv(DD+"/tables/tbl_invoc_track_grid.csv",index=False)
srch.to_csv(DD+"/tables/tbl_invoc_search_grid.csv",index=False)
print("WROTE Invocation_designspace.xlsx")
print("\n=== examples_summary ===")
print(exsummary.to_string(index=False))
print("\n=== N_Track grid (Train totals, per config) ===")
tt=trk[trk.Split=='Train'][[c for c in trk.columns if c.startswith('T_')]].sum()
print(tt.to_string())
print("\n=== N_Search grid (Train totals) ===")
print(srch[srch.Split=='Train'][[c for c in srch.columns if c.startswith('S_')]].sum().to_string())
