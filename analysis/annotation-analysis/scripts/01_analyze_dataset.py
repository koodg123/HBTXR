"""
01_analyze_dataset.py — EV-Eye structure + per-session frame/event statistics.

Outputs:
  <out>/01_dataset_structure.json   full inventory
  <out>/01_session_stats.csv        one row per (user,eye,session)
  <out>/01_summary.json             totals + presence of processed_data folders

Usage:
  python 01_analyze_dataset.py --root E:/DATASET/eveye --out ../results
  # add --exact to count every event (slow on 2.7B events); default is fast estimate
  # --max-users N to limit scope for a quick smoke test
"""
import os, csv, argparse, sys
import evlib as ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--exact", action="store_true", help="exact event count (full pass)")
    ap.add_argument("--sample-events", type=int, default=200000,
                    help="rows to scan per events.txt when not --exact")
    ap.add_argument("--max-users", type=int, default=0)
    args = ap.parse_args()

    ev.ensure_dir(args.out)
    davis = ev.davis_root(args.root)
    if not os.path.isdir(davis):
        print(f"[!] Data_davis not found at {davis}", file=sys.stderr)
        sys.exit(1)

    users = ev.list_users(davis)
    if args.max_users:
        users = users[: args.max_users]
    print(f"[i] {len(users)} users under {davis}")

    rows = []
    totals = {"frames": 0, "events_est": 0, "sessions": 0,
              "sessions_with_csv": 0, "labelled_frames": 0}
    structure = {"users": {}}

    for u in users:
        structure["users"][u] = {}
        for eye in ev.EYES:
            base = ev.p(davis, u, eye)
            if not os.path.isdir(base):
                continue
            structure["users"][u][eye] = {}
            for s in ev.SESSIONS:
                sp = ev.p(base, s)
                if not os.path.isdir(sp):
                    continue
                frames_dir = ev.p(sp, "frames")
                events_txt = ev.p(sp, "events", "events.txt")
                ffiles = ev.frame_files(frames_dir)
                nframes = len(ffiles)
                # fps from timestamps.txt if present
                ts = ev.read_timestamps_txt(ev.p(frames_dir, "timestamps.txt"))
                fps = None
                if len(ts) >= 2:
                    dt = (ts[-1][1] - ts[0][1]) / 1e6 / (len(ts) - 1)
                    fps = round(1.0 / dt, 2) if dt > 0 else None
                # events
                if args.exact:
                    es = ev.events_summary(events_txt)
                    ecount = es["count"] if es else 0
                    erate = es["rate_hz"] if es else None
                else:
                    es = ev.events_summary(events_txt, sample_cap=args.sample_events)
                    ecount = ev.estimate_event_count_by_size(events_txt) or 0
                    erate = None
                    if es and es["duration_s"] and es["count"]:
                        # rate from the sampled span (events/sec) — robust to size estimate
                        erate = es["rate_hz"]
                csv_path = ev.p(sp, f"{u}.csv")
                has_csv = os.path.isfile(csv_path)
                via_labelled = 0
                if has_csv:
                    via_labelled = sum(1 for r in ev.parse_via_csv(csv_path)
                                       if r["region_count"] > 0)
                row = {
                    "user": u, "eye": eye, "session": s,
                    "session_type": ev.SESSION_MEANING.get(s, "?"),
                    "n_frames": nframes, "fps": fps,
                    "events_count_or_est": ecount,
                    "events_exact": bool(args.exact),
                    "event_rate_hz": erate,
                    "has_via_csv": has_csv,
                    "via_labelled_frames": via_labelled,
                }
                rows.append(row)
                structure["users"][u][eye][s] = row
                totals["frames"] += nframes
                totals["events_est"] += ecount
                totals["sessions"] += 1
                totals["sessions_with_csv"] += int(has_csv)
                totals["labelled_frames"] += via_labelled
        print(f"  [{u}] done")

    # processed_data presence checks
    presence = {
        "Data_davis_labelled_with_mask": os.path.isdir(ev.mask_root(args.root)),
        "Data_tobii": os.path.isdir(ev.tobii_root(args.root)),
        "Data_davis_predict": os.path.isdir(ev.predict_root(args.root)),
        "Frame_event_pupil_track_result": os.path.isdir(ev.track_root(args.root)),
        "Pixel_error_evaluation": os.path.isdir(
            ev.p(args.root, "processed_data", "Pixel_error_evaluation")),
        "Pre-trained_models": os.path.isdir(
            ev.p(args.root, "processed_data", "Pre-trained_models"))
            or os.path.isdir(ev.p(args.root, "processed_data", "Pre-trained")),
    }

    # write
    with open(ev.p(args.out, "01_session_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["user"])
        w.writeheader()
        w.writerows(rows)
    ev.dump_json(structure, ev.p(args.out, "01_dataset_structure.json"))
    ev.dump_json({"users": len(users), "totals": totals,
                  "processed_data_present": presence},
                 ev.p(args.out, "01_summary.json"))

    print("\n=== SUMMARY ===")
    print(f" users={len(users)} sessions={totals['sessions']} "
          f"frames={totals['frames']:,} events(est)={totals['events_est']:,}")
    print(f" sessions_with_via_csv={totals['sessions_with_csv']} "
          f"via_labelled_frames={totals['labelled_frames']:,}")
    print(" processed_data_present:")
    for k, v in presence.items():
        print(f"   {'OK ' if v else 'NO '} {k}")
    print(f"\n[i] wrote results to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
