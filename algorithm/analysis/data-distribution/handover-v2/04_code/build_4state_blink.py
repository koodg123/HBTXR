"""Full-stream 4-state classification: Fixation / Saccade / Smooth / Blink.

This is the STANDARD eye-movement taxonomy (replacing the velocity-only "Reset"
4th class). Unlike the GT-frame analyses, it runs over the *entire* near-eye
frame stream, because Blink frames are eye-closed and were excluded from the
sparse GT annotations -- they only exist in the full stream.

Per frame:
  1) Blink   if no pupil is present (eyelid occludes it) -- see blink_detect.
  2) Saccade if eye open and speed > v_sacc      (standard I-VT, any session).
  3) Fixation if eye open, speed <= v_sacc, saccade-session (101/201).
  4) Smooth   if eye open, speed <= v_sacc, smooth-session  (102/202).
(The former "Reset" = high-speed frames in smooth sessions now correctly fall
under Saccade -- they are return/catch-up saccades.)

Dependencies: common.py, pupil_detect.py, blink_detect.py  (drop this file in
codes/motion_type/src/ next to them). Needs raw frames under EVEYE_ROOT, so run
on the WSL/GPU box, not the cloud sandbox.

Outputs (under codes/motion_type/):
  cache/frames_4state_blink.csv          per-frame labels (resumable per session)
  reports/tbl_4state_blink_per_subject.csv   Subject x {Fixation,Saccade,Smooth,Blink}
  reports/fig_4state_blink_per_subject.png   stacked-share figure (if matplotlib)

Usage:
  EVEYE_ROOT=/mnt/e/DATASET/eveye python3 build_4state_blink.py            # full, all users
  EVEYE_ROOT=... python3 build_4state_blink.py --users 1 2 3 --stride 1
  EVEYE_ROOT=... python3 build_4state_blink.py --stride 2 --time-budget 3600
  # thresholds (defaults are auto-calibrated / consistent with the 3-state pipeline):
  EVEYE_ROOT=... python3 build_4state_blink.py --v-sacc 493 --blink-tau 4.1
"""
import os, sys, glob, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import cv2

from common import (USERS, EYES, SESSIONS, CACHE_DIR, REPORTS_DIR,
                    session_dir, parse_frame_filename)
import blink_detect as bd

STATES = ["Fixation", "Saccade", "Smooth", "Blink"]
PERFRAME_CACHE = os.path.join(CACHE_DIR, "frames_4state_blink.csv")


def default_v_sacc():
    """v_sacc = 90th pct of saccade-session speed, from the existing GT velocity
    cache if present (keeps it identical to the validated 3-state pipeline);
    else fall back to the known value 493 px/s."""
    p = os.path.join(CACHE_DIR, "velocity_at_gt.csv")
    if os.path.exists(p):
        v = pd.read_csv(p)
        s = v.loc[v.motion == "saccade", "speed_pxps"].dropna()
        if len(s):
            return float(np.percentile(s, 90))
    return 493.0


def default_tau():
    """Blink score threshold, calibrated from GT (eye-open) detector scores."""
    p = os.path.join(CACHE_DIR, "velocity_at_gt.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        if "det_conf" in d:
            return bd.calibrate_tau(d["det_conf"].tolist())
    return 4.0


def classify_session(user, eye, code, v_sacc, tau, stride, max_frames):
    """Return a DataFrame of per-frame 4-state labels for one session."""
    motion_session = SESSIONS[code][1]  # 'saccade' or 'smooth'
    fdir = os.path.join(session_dir(user, eye, code), "frames")
    files = sorted(glob.glob(os.path.join(fdir, "*.png")),
                   key=lambda f: parse_frame_filename(f)[0])
    if stride > 1:
        files = files[::stride]
    if max_frames:
        files = files[:max_frames]
    if not files:
        return pd.DataFrame()

    idx, ts, cx, cy, score, blink0 = [], [], [], [], [], []
    for f in files:
        i, t = parse_frame_filename(f)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        ft = bd.pupil_features(img)
        idx.append(i); ts.append(t)
        cx.append(ft["cx"]); cy.append(ft["cy"]); score.append(ft["score"])
        blink0.append(bd.is_blink(ft, tau))

    # temporal hysteresis on the blink flag (remove flicker / fill 1-frame gaps)
    blink = bd.smooth_blink_runs(blink0, min_len=2, max_gap=1)

    cx = np.array(cx); cy = np.array(cy); ts = np.array(ts, float)
    n = len(idx)
    speed = np.full(n, np.nan)
    # central difference using nearest non-blink detected neighbours (<= 2 away)
    for k in range(n):
        if blink[k]:
            continue
        pk = nk = None
        for d in (1, 2):
            if pk is None and k - d >= 0 and not blink[k - d] and not np.isnan(cx[k - d]):
                pk = k - d
            if nk is None and k + d < n and not blink[k + d] and not np.isnan(cx[k + d]):
                nk = k + d
        if pk is not None and nk is not None:
            dt = (ts[nk] - ts[pk]) / 1e6
            if dt > 0:
                speed[k] = np.hypot(cx[nk] - cx[pk], cy[nk] - cy[pk]) / dt

    state = []
    for k in range(n):
        if blink[k]:
            state.append("Blink")
        elif not np.isnan(speed[k]) and speed[k] > v_sacc:
            state.append("Saccade")
        elif motion_session == "saccade":
            state.append("Fixation")
        else:
            state.append("Smooth")

    return pd.DataFrame(dict(user=user, eye=eye, code=code,
                             motion_session=motion_session, frame_idx=idx, ts=ts,
                             cx=cx, cy=cy, score=score, speed_pxps=speed, state=state))


def build(users=None, stride=1, max_frames=None, v_sacc=None, tau=None,
          time_budget=None):
    v_sacc = default_v_sacc() if v_sacc is None else v_sacc
    tau = default_tau() if tau is None else tau
    users = users or USERS
    os.makedirs(CACHE_DIR, exist_ok=True)

    done, parts = set(), []
    if os.path.exists(PERFRAME_CACHE):
        prev = pd.read_csv(PERFRAME_CACHE)
        done = set(zip(prev.user, prev.eye, prev.code.astype(str)))
        parts = [prev]

    todo = [(u, e, c) for u in users for e in EYES for c in SESSIONS
            if (u, e, str(c)) not in done]
    print(f"v_sacc={v_sacc:.0f} px/s | blink_tau={tau:.2f} | stride={stride} | "
          f"sessions to do={len(todo)}")

    t0 = time.time()
    for n, (u, e, c) in enumerate(todo, 1):
        if time_budget and time.time() - t0 > time_budget:
            print(f"[time budget hit] stopping after {n-1} sessions"); break
        df = classify_session(u, e, c, v_sacc, tau, stride, max_frames)
        if len(df):
            parts.append(df)
        if n % 20 == 0:
            print(f"  {n}/{len(todo)} sessions ({time.time()-t0:.0f}s)")
            pd.concat(parts, ignore_index=True).to_csv(PERFRAME_CACHE, index=False)

    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    res.to_csv(PERFRAME_CACHE, index=False)
    return res, v_sacc, tau


def aggregate(res):
    """Per-subject Subject x {Fixation,Saccade,Smooth,Blink} counts + All row."""
    tab = (res.groupby(["user", "state"]).size().unstack(fill_value=0)
              .reindex(columns=STATES, fill_value=0)
              .reindex(sorted(res.user.unique()), fill_value=0))
    tab.index.name = "Subject"
    tab["Total"] = tab.sum(axis=1)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = tab.reset_index()
    allrow = {"Subject": "All", **{s: int(tab[s].sum()) for s in STATES + ["Total"]}}
    out = pd.concat([out, pd.DataFrame([allrow])], ignore_index=True)
    csv = os.path.join(REPORTS_DIR, "tbl_4state_blink_per_subject.csv")
    out.to_csv(csv, index=False)
    print("wrote", csv)
    print(out.tail(13).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        share = tab[STATES].div(tab[STATES].sum(axis=1), axis=0) * 100
        ax = share.plot(kind="bar", stacked=True, figsize=(16, 5),
                        color=["#70AD47", "#ED7D31", "#5B9BD5", "#FFC000"])
        ax.set_ylabel("% of frames"); ax.set_xlabel("Subject")
        ax.set_title("EV-Eye per-subject 4-state share (Fixation / Saccade / Smooth / Blink)")
        ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
        fig = os.path.join(REPORTS_DIR, "fig_4state_blink_per_subject.png")
        plt.tight_layout(); plt.savefig(fig, dpi=130); print("wrote", fig)
    except Exception as ex:
        print("figure skipped:", ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs="*", type=int)
    ap.add_argument("--stride", type=int, default=1,
                    help="frame subsampling; keep 1-2 so brief blinks are not missed")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per session (for a quick smoke test)")
    ap.add_argument("--v-sacc", type=float, default=None)
    ap.add_argument("--blink-tau", type=float, default=None)
    ap.add_argument("--time-budget", type=float, default=None, help="seconds")
    a = ap.parse_args()
    res, v_sacc, tau = build(users=a.users, stride=a.stride, max_frames=a.max_frames,
                             v_sacc=a.v_sacc, tau=a.blink_tau, time_budget=a.time_budget)
    if len(res):
        aggregate(res)
    else:
        print("no frames processed -- check EVEYE_ROOT and raw_data/Data_davis layout")


if __name__ == "__main__":
    main()
