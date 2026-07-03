"""Sections (1) subject-wise and (2) motion-type-wise error statistics.
Outputs CSV tables + figures to reports/. center-distance error, frame & event tracks."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import eveye_metrics as M
from common import REPORTS_DIR

df = M.load_pe_allframes()
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------- (2) motion-type-wise ----------
rows = []
for track in ["frame", "event"]:
    sub = df[df.track == track]
    for mot in ["saccade", "smooth", "all"]:
        s = sub.err if mot == "all" else sub[sub.motion == mot].err
        d = M.describe(s)
        ci_mean = M.boot_ci(s, np.mean); ci_med = M.boot_ci(s, np.median)
        # between-subject CI of the mean
        psm = (sub if mot == "all" else sub[sub.motion == mot]).groupby("user").err.mean()
        sci = M.subject_level_ci(psm.values)
        rows.append(dict(track=track, motion=mot, **d,
                         mean_CI95_lo=ci_mean[0], mean_CI95_hi=ci_mean[1],
                         median_CI95_lo=ci_med[0], median_CI95_hi=ci_med[1],
                         mean_subjCI_lo=sci[0], mean_subjCI_hi=sci[1]))
t2 = pd.DataFrame(rows)
t2.to_csv(os.path.join(REPORTS_DIR, "sec2_motion_stats.csv"), index=False)
print("=== (2) motion-type-wise ===\n", t2[["track","motion","n","mean","median","p95","p99","mean_CI95_lo","mean_CI95_hi"]].to_string(index=False))

# (2) figure: mean/median/p95/p99 by motion x track
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, track in zip(axes, ["frame", "event"]):
    tt = t2[(t2.track == track) & (t2.motion != "all")].set_index("motion").reindex(["saccade", "smooth"])
    x = np.arange(2); w = 0.2
    for i, (stat, c) in enumerate([("mean","#5b8c5a"),("median","#1f77b4"),("p95","#e8a33d"),("p99","#d1495b")]):
        ax.bar(x + (i-1.5)*w, tt[stat], w, label=stat, color=c)
    ax.set_xticks(x); ax.set_xticklabels(["saccade","smooth"]); ax.set_title(f"EV-Eye {track} track")
    ax.set_ylabel("pixel error"); ax.legend(fontsize=8)
plt.suptitle("(2) Motion-type-wise pixel error"); plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "fig_sec2_motion_stats.png"), dpi=130)

# ---------- (1) subject-wise ----------
rows = []
for track in ["frame", "event"]:
    sub = df[df.track == track]
    for u in sorted(sub.user.unique()):
        su = sub[sub.user == u]
        for mot in ["saccade", "smooth", "all"]:
            s = su.err if mot == "all" else su[su.motion == mot].err
            if len(s) == 0: continue
            d = M.describe(s); ci = M.boot_ci(s, np.median)
            rows.append(dict(track=track, user=u, motion=mot, **d,
                             median_CI95_lo=ci[0], median_CI95_hi=ci[1]))
t1 = pd.DataFrame(rows)
t1.to_csv(os.path.join(REPORTS_DIR, "sec1_subject_stats.csv"), index=False)
print(f"\n=== (1) subject-wise: {len(t1)} rows (track x user x motion) -> sec1_subject_stats.csv ===")
print(t1[t1.track=="frame"].groupby("motion")[["median","p95","p99"]].mean().round(3).to_string())

# (1) figure: per-subject median (+CI) saccade vs smooth, frame track
for track in ["frame", "event"]:
    sub = t1[(t1.track == track) & (t1.motion != "all")]
    users = sorted(sub.user.unique()); x = np.arange(len(users))
    fig, ax = plt.subplots(figsize=(13, 4))
    for mot, off, c in [("saccade", -0.15, "#d1495b"), ("smooth", 0.15, "#1f77b4")]:
        m = sub[sub.motion == mot].set_index("user").reindex(users)
        yerr = np.vstack([m["median"]-m["median_CI95_lo"], m["median_CI95_hi"]-m["median"]])
        ax.errorbar(x+off, m["median"], yerr=yerr, fmt="o", ms=3, capsize=2, label=mot, color=c)
    ax.set_xticks(x); ax.set_xticklabels(users, fontsize=6); ax.set_xlabel("subject (user)")
    ax.set_ylabel("median center error (px)"); ax.legend()
    ax.set_title(f"(1) Subject-wise median center error ±95% CI — EV-Eye {track} track")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR, f"fig_sec1_subject_{track}.png"), dpi=130)
print("\nwrote figs: fig_sec2_motion_stats.png, fig_sec1_subject_frame.png, fig_sec1_subject_event.png")
