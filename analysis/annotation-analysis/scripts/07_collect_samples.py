"""
07_collect_samples.py — EV-Eye audit sample collector (GT-anchored continuous windows).

Design (confirmed with user):
  (1) window length K: fixation/smooth = 11, saccade = 7
  (2) windows per motion: fixation 45, smooth_pursuit 45, saccade 70  (~1,480 frames)
  (3) GT-keyframe CENTER-anchored windows: the center frame has original EV-Eye VIA GT;
      the K-1 neighbours are continuous frames (audit/pred only)
  (4) collect ALL events spanning each window  [ts_first .. ts_last]  (+optional pad)
  (5) record provenance (source absolute path of every copied item)
  (6) collect timestamp / label / mask / U-Net center — everything needed
  (7) copy into  samples/frame/ , samples/event/ , samples/label/

Ground truth = original EV-Eye VIA ellipse (frozen).  U-Net (Data_davis_predict) is an
AUDIT tool only and is used here (a) to derive per-frame centres and (b) to classify
fixation-vs-saccade inside session_1_0_2 by local centre velocity.  No human re-annotation.

Run on the machine that holds the dataset (local FS is fast):
  python 07_collect_samples.py --root E:/DATASET/eveye --out ../samples --dry-run
  python 07_collect_samples.py --root E:/DATASET/eveye --out ../samples           # full copy
  python 07_collect_samples.py --root E:/DATASET/eveye --out ../samples \
        --test-users 1,2,5,9 --copy-masks

Quick test:  add  --max-users 3  --dry-run  first to check the strata counts.
"""
import os, re, csv, json, glob, time, shutil, argparse, sys
import evlib as ev

try:
    import numpy as np
except Exception:
    np = None


# ----------------------------------------------------------------- helpers
def find_via_csv(session_dir):
    hits = glob.glob(ev.p(session_dir, "user_*.csv")) or glob.glob(ev.p(session_dir, "*.csv"))
    return hits[0] if hits else None


def frame_index(name):
    m = re.match(r"(\d+)_", os.path.basename(name))
    return int(m.group(1)) if m else None


def load_frames(frames_dir):
    """Return list of dicts {idx, ts, path} sorted by idx."""
    out = []
    for fp in ev.frame_files(frames_dir):
        pt = ev.parse_frame_timestamp(fp)
        if pt:
            out.append({"idx": pt[0], "ts": pt[1], "path": fp})
    out.sort(key=lambda r: r["idx"])
    return out


def predict_dir(root, user, eye, session):
    proot = ev.predict_root(root)
    for base in (ev.p(proot, user, user, eye, session, "predict"),
                 ev.p(proot, user, eye, session, "predict")):
        if os.path.isdir(base):
            return base
    return None


def predict_map(pdir):
    """idx -> gif path for a predict folder."""
    m = {}
    if not pdir:
        return m
    for fp in glob.glob(ev.p(pdir, "*_mask.gif")) + glob.glob(ev.p(pdir, "*.gif")):
        i = frame_index(fp)
        if i is not None:
            m.setdefault(i, fp)
    return m


def unet_center(gif_path):
    if np is None:
        return {"valid": False, "cx": None, "cy": None, "area": 0}
    try:
        from PIL import Image
        im = np.array(Image.open(gif_path).convert("L"))
    except Exception:
        return {"valid": False, "cx": None, "cy": None, "area": 0}
    thr = (im.max() / 2.0) if im.max() > 1 else 0
    b = im > thr
    area = int(b.sum())
    if area == 0:
        return {"valid": False, "cx": None, "cy": None, "area": 0}
    ys, xs = np.where(b)
    return {"valid": True, "cx": round(float(xs.mean()), 3),
            "cy": round(float(ys.mean()), 3), "area": area}


def local_velocity(frames, pos, pmap, half):
    """Robust |Δcentre| stats (px/frame) from U-Net centres around frames[pos].
    Excludes blink frames (U-Net mask-area collapse) so single-frame centroid
    jumps don't masquerade as saccades. Returns mean/median/max/blink_frac."""
    import math, statistics
    lo, hi = max(0, pos - half), min(len(frames), pos + half + 1)
    cs, areas = [], []
    for r in frames[lo:hi]:
        gp = pmap.get(r["idx"])
        c = unet_center(gp) if gp else {"valid": False}
        if c.get("valid"):
            cs.append(c); areas.append(c["area"])
        else:
            cs.append(None)
    if len([c for c in cs if c]) < 3:
        return None
    med_area = statistics.median(areas) if areas else 0

    def ok(c):
        return c is not None and med_area > 0 and c["area"] >= 0.4 * med_area

    blink_frac = 1.0 - (sum(1 for c in cs if ok(c)) / len(cs))
    vs = []
    for a, b in zip(cs, cs[1:]):
        if ok(a) and ok(b):
            vs.append(math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))
    if not vs:
        return None
    return {"mean": sum(vs) / len(vs), "median": statistics.median(vs),
            "max": max(vs), "n": len(vs), "blink_frac": blink_frac}


# ----------------------------------------------------------------- candidate windows
def gather_candidates(root, args):
    """Yield candidate window dicts per motion (not yet capped)."""
    davis = ev.davis_root(root)
    users = ev.list_users(davis)
    if args.test_users:
        keep = {f"user{u}" for u in args.test_users.split(",") if u.strip()}
        users = [u for u in users if u in keep]
    if args.max_users:
        users = users[: args.max_users]

    cands = {"fixation": [], "saccade": [], "smooth_pursuit": []}
    rejections = []
    half = max(args.kf // 2, 3)

    for u in users:
        for eye in ev.EYES:
            for session in ev.SESSIONS:
                sdir = ev.p(davis, u, eye, session)
                if not os.path.isdir(sdir) or session == "session_1_0_1":
                    continue  # 1_0_1 has no GT
                frames = load_frames(ev.p(sdir, "frames"))
                if not frames:
                    continue
                pos_of = {r["idx"]: i for i, r in enumerate(frames)}
                csv_path = find_via_csv(sdir)
                if not csv_path:
                    continue
                kfs = [r for r in ev.parse_via_csv(csv_path)
                       if r["region_count"] > 0 and r["ellipse"]]
                is_smooth = session in ("session_2_0_1", "session_2_0_2")
                pmap = None if is_smooth else predict_map(predict_dir(root, u, eye, session))

                for r in kfs:
                    idx = frame_index(r["filename"])
                    if idx is None or idx not in pos_of:
                        rejections.append((u, eye, session, idx, "anchor_not_in_frames"))
                        continue
                    pos = pos_of[idx]
                    # motion + K
                    if is_smooth:
                        motion, K = "smooth_pursuit", args.kf
                    else:
                        vel = local_velocity(frames, pos, pmap, half) if pmap else None
                        if vel is None:
                            rejections.append((u, eye, session, idx, "no_unet_velocity"))
                            continue
                        if vel["blink_frac"] > 0.34:
                            rejections.append((u, eye, session, idx, "blink_contaminated"))
                            continue
                        if vel["max"] >= args.vsac:
                            motion, K = "saccade", args.ks
                        elif vel["median"] < args.vfix:
                            motion, K = "fixation", args.kf
                        else:
                            rejections.append((u, eye, session, idx, "ambiguous_motion"))
                            continue
                    k = K // 2
                    if pos - k < 0 or pos + k >= len(frames):
                        rejections.append((u, eye, session, idx, "window_truncated"))
                        continue
                    win_frames = frames[pos - k: pos + k + 1]
                    # require contiguous indices (no dropped frames)
                    idxs = [f["idx"] for f in win_frames]
                    if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
                        rejections.append((u, eye, session, idx, "non_contiguous"))
                        continue
                    cands[motion].append({
                        "motion": motion, "user": u, "eye": eye, "session": session,
                        "K": K, "anchor_idx": idx, "anchor_ts": r["ts"],
                        "ellipse": r["ellipse"], "quality": r["quality"],
                        "frames": win_frames, "csv": csv_path,
                        "ts_lo": win_frames[0]["ts"], "ts_hi": win_frames[-1]["ts"],
                    })
    return cands, rejections


def select(cands, args):
    """Non-overlapping anchors per (user,eye,session), round-robin across users.
    If args.balance: equalize all motions to the min available (saccade-matched)."""
    import random
    from collections import defaultdict, deque
    rng = random.Random(args.seed)
    bags = {}
    for m, lst in cands.items():
        rng.shuffle(lst)
        used, bag = {}, []
        for w in lst:
            key = (w["user"], w["eye"], w["session"])
            if all(abs(w["anchor_idx"] - a) >= w["K"] for a in used.get(key, [])):
                used.setdefault(key, []).append(w["anchor_idx"])
                bag.append(w)
        bag.sort(key=lambda w: (w["user"], w["eye"], w["session"], w["anchor_idx"]))
        bags[m] = bag
    if args.balance:
        n = min((len(b) for b in bags.values()), default=0)
        if args.cap:
            n = min(n, args.cap)
        targets = {m: n for m in bags}
    else:
        targets = {"fixation": args.nf, "smooth_pursuit": args.nsm, "saccade": args.nsac}
    chosen = {}
    for m, bag in bags.items():
        byuser = defaultdict(deque)
        for w in bag:
            byuser[w["user"]].append(w)
        users = list(byuser.keys())
        ordered = []
        while any(byuser[u] for u in users) and len(ordered) < targets.get(m, 0):
            for u in users:
                if byuser[u]:
                    ordered.append(byuser[u].popleft())
                    if len(ordered) >= targets.get(m, 0):
                        break
        chosen[m] = ordered[: targets.get(m, 0)]
    return chosen


# ----------------------------------------------------------------- events
def slice_events(events_txt, windows, pad_us=0):
    """One pass over events.txt; return key -> dict of arrays for windows in this session."""
    ranges = [(w["ts_lo"] - pad_us, w["ts_hi"] + pad_us, w["key"]) for w in windows]
    buf = {w["key"]: {"t": [], "x": [], "y": [], "p": []} for w in windows}
    if not os.path.isfile(events_txt) or not ranges:
        return buf
    tmin = min(r[0] for r in ranges)
    tmax = max(r[1] for r in ranges)
    with open(events_txt, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                ts = int(parts[0])
            except ValueError:
                continue
            if ts < tmin:
                continue
            if ts > tmax:
                break  # events.txt is time-sorted
            x, y, pol = int(parts[1]), int(parts[2]), int(parts[3])
            for lo, hi, key in ranges:
                if lo <= ts <= hi:
                    b = buf[key]
                    b["t"].append(ts); b["x"].append(x); b["y"].append(y); b["p"].append(pol)
    return buf


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EV-Eye root, e.g. E:/DATASET/eveye")
    ap.add_argument("--out", default="../samples")
    ap.add_argument("--test-users", default="", help="comma user ids to restrict (subject-independent)")
    ap.add_argument("--kf", type=int, default=11, help="window K for fixation/smooth")
    ap.add_argument("--ks", type=int, default=7, help="window K for saccade")
    ap.add_argument("--nf", type=int, default=45, help="fixation windows")
    ap.add_argument("--nsm", type=int, default=45, help="smooth_pursuit windows")
    ap.add_argument("--nsac", type=int, default=70, help="saccade windows")
    ap.add_argument("--vfix", type=float, default=1.2, help="px/frame; median vel < vfix => fixation")
    ap.add_argument("--vsac", type=float, default=6.0, help="px/frame; max vel >= vsac => saccade")
    ap.add_argument("--cap", type=int, default=0, help="max windows/motion when balancing (0 = use min available)")
    ap.add_argument("--no-balance", dest="balance", action="store_false",
                    help="disable saccade-matched balancing (use --nf/--nsm/--nsac targets)")
    ap.set_defaults(balance=True)
    ap.add_argument("--event-pad-us", type=int, default=0, help="extra +-us around window for events")
    ap.add_argument("--copy-masks", action="store_true", help="also copy U-Net mask gifs into label/")
    ap.add_argument("--max-users", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--dry-run", action="store_true", help="select + manifest only, no copying")
    args = ap.parse_args()

    out = args.out
    for sub in ("frame", "event", "label", "meta"):
        ev.ensure_dir(ev.p(out, sub))

    print("[i] gathering candidates ...")
    cands, rejections = gather_candidates(args.root, args)
    for m in cands:
        print(f"    candidates {m}: {len(cands[m])}")
    chosen = select(cands, args)

    # assign keys
    all_windows = []
    for m, lst in chosen.items():
        for wi, w in enumerate(lst):
            w["window_id"] = wi
            w["key"] = f"{m}_{w['user']}_{w['eye']}_{w['session']}_w{wi:03d}_a{w['anchor_idx']:06d}"
            all_windows.append(w)

    print(f"\n[i] selected (balance={args.balance}):")
    strata = {}
    for w in all_windows:
        strata.setdefault(w["motion"], 0)
        strata[w["motion"]] += 1
    for m in ("fixation", "saccade", "smooth_pursuit"):
        print(f"    {m}: {strata.get(m,0)} collected  (candidates {len(cands.get(m, []))})")

    # group windows by session for one-pass event slicing
    bysession = {}
    for w in all_windows:
        bysession.setdefault((w["user"], w["eye"], w["session"]), []).append(w)

    frow, wrow = [], []
    for (u, eye, session), wins in bysession.items():
        events_txt = ev.p(ev.davis_root(args.root), u, eye, session, "events", "events.txt")
        ev_buf = {} if args.dry_run else slice_events(events_txt, wins, args.event_pad_us)
        for w in wins:
            key = w["key"]
            ldir = ev.p(out, "label", key)
            fdir = ev.p(out, "frame", key)
            if not args.dry_run:
                ev.ensure_dir(ldir); ev.ensure_dir(fdir)
            # frames + per-frame unet centre
            pmap = predict_map(predict_dir(args.root, u, eye, session))
            unet_rows = []
            for fr in w["frames"]:
                gp = pmap.get(fr["idx"])
                uc = unet_center(gp) if gp else {"valid": False, "cx": None, "cy": None, "area": 0}
                role = "anchor" if fr["idx"] == w["anchor_idx"] else "neighbor"
                dst_frame = ev.p(fdir, os.path.basename(fr["path"]))
                if not args.dry_run:
                    try:
                        shutil.copy2(fr["path"], dst_frame)
                        if args.copy_masks and gp:
                            mdir = ev.ensure_dir(ev.p(ldir, "unet_masks"))
                            shutil.copy2(gp, ev.p(mdir, os.path.basename(gp)))
                    except Exception as e:
                        print(f"    [warn] copy {fr['path']}: {e}")
                unet_rows.append({"idx": fr["idx"], "ts": fr["ts"], **uc, "src_gif": gp})
                el = w["ellipse"] if role == "anchor" else None
                frow.append({
                    "key": key, "motion": w["motion"], "role": role,
                    "user": u, "eye": eye, "session": session,
                    "frame_index": fr["idx"], "frame_ts_us": fr["ts"],
                    "has_gt": role == "anchor",
                    "gt_cx": el[0] if el else "", "gt_cy": el[1] if el else "",
                    "gt_rx": el[2] if el else "", "gt_ry": el[3] if el else "",
                    "gt_theta": el[4] if el else "",
                    "unet_valid": uc["valid"], "unet_cx": uc["cx"], "unet_cy": uc["cy"],
                    "unet_area": uc["area"],
                    "src_frame": fr["path"], "dst_frame": dst_frame,
                })
            # events
            n_events = 0
            ev_file = ev.p(out, "event", key + ".npz")
            if not args.dry_run and np is not None:
                b = ev_buf.get(key, {"t": [], "x": [], "y": [], "p": []})
                n_events = len(b["t"])
                np.savez_compressed(
                    ev_file,
                    t=np.asarray(b["t"], dtype=np.int64),
                    x=np.asarray(b["x"], dtype=np.int16),
                    y=np.asarray(b["y"], dtype=np.int16),
                    p=np.asarray(b["p"], dtype=np.int8),
                    ts_lo=w["ts_lo"], ts_hi=w["ts_hi"])
            # labels + provenance
            q = w["quality"] or {}
            if not args.dry_run:
                ev.dump_json({"anchor_idx": w["anchor_idx"], "anchor_ts": w["anchor_ts"],
                              "ellipse_cx_cy_rx_ry_theta": w["ellipse"],
                              "quality": q, "source_csv": w["csv"]},
                             ev.p(ldir, "gt.json"))
                ev.dump_json({"unet_centers": unet_rows}, ev.p(ldir, "unet_dense.json"))
                ev.dump_json({
                    "key": key, "motion": w["motion"], "K": w["K"],
                    "user": u, "eye": eye, "session": session,
                    "source": {
                        "frames_dir": ev.p(ev.davis_root(args.root), u, eye, session, "frames"),
                        "events_txt": events_txt,
                        "via_csv": w["csv"],
                        "predict_dir": predict_dir(args.root, u, eye, session),
                        "mask_h5": ev.p(ev.mask_root(args.root), eye, f"{u}_{session}.h5"),
                    },
                    "ts_lo": w["ts_lo"], "ts_hi": w["ts_hi"],
                    "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, ev.p(ldir, "provenance.json"))
            wrow.append({
                "key": key, "motion": w["motion"], "user": u, "eye": eye, "session": session,
                "K": w["K"], "anchor_index": w["anchor_idx"], "anchor_ts_us": w["anchor_ts"],
                "ts_lo": w["ts_lo"], "ts_hi": w["ts_hi"], "n_frames": len(w["frames"]),
                "n_events": n_events,
                "q_good": q.get("good"), "q_frontal": q.get("frontal"),
                "q_illum": q.get("good_illumination"),
                "event_file": ev_file, "label_dir": ldir,
            })
        print(f"  [{u}/{eye}/{session}] {len(wins)} windows done")

    # write manifests + meta
    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    write_csv(ev.p(out, "manifest_frames.csv"), frow)
    write_csv(ev.p(out, "manifest_windows.csv"), wrow)
    write_csv(ev.p(out, "meta", "rejections.csv"),
              [{"user": a, "eye": b, "session": c, "anchor_index": d, "reason": e}
               for (a, b, c, d, e) in rejections])
    ev.dump_json(vars(args), ev.p(out, "meta", "selection_config.json"))
    write_csv(ev.p(out, "meta", "strata_summary.csv"),
              [{"motion": m, "candidates": len(cands.get(m, [])), "collected": strata.get(m, 0)}
               for m in ("fixation", "saccade", "smooth_pursuit")])

    print(f"\n[done] windows={len(all_windows)} frames={len(frow)} "
          f"{'(dry-run, nothing copied)' if args.dry_run else ''}")
    print(f"       out={os.path.abspath(out)}  (frame/ event/ label/ + manifests)")
    if args.dry_run:
        print("       review meta/strata_summary.csv & manifest_windows.csv, then re-run without --dry-run")


if __name__ == "__main__":
    main()
