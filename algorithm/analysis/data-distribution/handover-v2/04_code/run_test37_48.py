"""Parallel (2-core), time-budgeted, resumable 4-state run for test subjects 37-48.
Separate clean cache. Run repeatedly until done; then aggregates per-subject."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import EYES, SESSIONS, CACHE_DIR, REPORTS_DIR
import build_4state_blink as B

USERS_TEST = list(range(37, 49))
CACHE = os.path.join(CACHE_DIR, "frames_4state_blink_test37_48.csv")
VSACC, TAU, STRIDE = 493.0, 4.1, 2

def work(args):
    u, e, c = args
    df = B.classify_session(u, e, c, VSACC, TAU, STRIDE, None)
    return (u, e, c, df)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=38)
    a = ap.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(REPORTS_DIR, exist_ok=True)
    done = set(); parts = []
    if os.path.exists(CACHE):
        prev = pd.read_csv(CACHE); parts = [prev]
        done = set(zip(prev.user, prev.eye, prev.code.astype(str)))
    todo = [(u, e, c) for u in USERS_TEST for e in EYES for c in SESSIONS
            if (u, e, str(c)) not in done]
    print(f"done={len(done)} todo={len(todo)} (stride={STRIDE})")
    if not todo:
        agg(pd.concat(parts, ignore_index=True)); return
    t0 = time.time(); newdone = 0
    with Pool(2) as pool:
        for (u, e, c, df) in pool.imap_unordered(work, todo):
            if len(df): parts.append(df)
            newdone += 1
            pd.concat(parts, ignore_index=True).to_csv(CACHE, index=False)
            el = time.time() - t0
            print(f"  +{u}/{e}/{c} ({len(df)}f) | {newdone} done this run | {el:.0f}s")
            if el > a.budget: break
    res = pd.concat(parts, ignore_index=True)
    remaining = len(todo) - newdone
    print(f"this run: +{newdone} sessions, remaining={remaining}, cache rows={len(res)}")
    if remaining == 0: agg(res)

def agg(res):
    res = res[res.user.isin(USERS_TEST)]
    tab = (res.groupby(["user","state"]).size().unstack(fill_value=0)
              .reindex(columns=B.STATES, fill_value=0).reindex(USERS_TEST, fill_value=0))
    tab.index.name="Subject"; tab["Total"]=tab.sum(axis=1)
    out=tab.reset_index()
    allrow={"Subject":"All",**{s:int(tab[s].sum()) for s in B.STATES+["Total"]}}
    out=pd.concat([out,pd.DataFrame([allrow])],ignore_index=True)
    csv=os.path.join(REPORTS_DIR,"tbl_4state_blink_test37_48.csv"); out.to_csv(csv,index=False)
    print("WROTE",csv); print(out.to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        share=tab[B.STATES].div(tab[B.STATES].sum(axis=1),axis=0)*100
        ax=share.plot(kind="bar",stacked=True,figsize=(12,5),color=["#70AD47","#ED7D31","#5B9BD5","#FFC000"])
        ax.set_ylabel("% frames"); ax.set_xlabel("Subject (test 37-48)")
        ax.set_title("Test set (37-48) per-subject 4-state share (Fixation/Saccade/Smooth/Blink)")
        ax.legend(ncol=4,loc="lower center",bbox_to_anchor=(0.5,1.02))
        fig=os.path.join(REPORTS_DIR,"fig_4state_blink_test37_48.png")
        plt.tight_layout(); plt.savefig(fig,dpi=130); print("WROTE",fig)
    except Exception as ex: print("fig skipped",ex)

if __name__=="__main__": main()
