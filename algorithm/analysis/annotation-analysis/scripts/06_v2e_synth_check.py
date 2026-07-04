"""
06_v2e_synth_check.py — (optional) evaluate the v2e simulator (#2) by synthesizing
events from a session's APS frames and comparing the statistics of the synthetic
event stream against the REAL DAVIS events.txt of the same session.

What it does:
  1. Builds a grayscale .avi from frames/*.png (uses the embedded ns timestamps to
     set the frame rate) via OpenCV.
  2. Calls v2e CLI (must be installed/importable; pass --v2e to point at v2e.py or
     rely on the `v2e` console script) to produce a DVS text file (t x y p).
  3. Compares synthetic vs real: event rate (Hz), ON/OFF ratio, spatial extent,
     events-per-frame — a fidelity sanity check for using v2e as a label-transfer tool.

This script is intentionally conservative: if v2e or cv2 is missing it explains how
to proceed instead of failing hard. It does NOT attempt label transfer; it only
quantifies distributional similarity.

Usage:
  python 06_v2e_synth_check.py --root E:/DATASET/eveye --out ../results \
      --user user1 --eye right --session session_1_0_2 \
      --v2e /path/to/third/v2e/v2e.py --max-frames 200
"""
import os, csv, argparse, subprocess, sys, tempfile, shutil
import evlib as ev


def build_avi(frames, avi_path, fps):
    import cv2
    if not frames:
        return False
    img0 = cv2.imread(frames[0], cv2.IMREAD_GRAYSCALE)
    h, w = img0.shape[:2]
    vw = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h), isColor=False)
    if not vw.isOpened():
        vw = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h), isColor=False)
    for fp in frames:
        im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        vw.write(im)
    vw.release()
    return os.path.isfile(avi_path)


def real_event_stats(events_txt, max_events=0):
    cap = max_events if max_events else None
    return ev.events_summary(events_txt, sample_cap=cap)


def synth_event_stats(dvs_text):
    n = on = off = 0
    t0 = t1 = None
    xmax = ymax = 0
    with open(dvs_text, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                t = float(parts[0]); x = int(float(parts[1]))
                y = int(float(parts[2])); pol = int(float(parts[3]))
            except ValueError:
                continue
            if t0 is None:
                t0 = t
            t1 = t; n += 1
            on += int(pol == 1); off += int(pol != 1)
            xmax = max(xmax, x); ymax = max(ymax, y)
    if n == 0:
        return None
    dur = (t1 - t0) if (t0 is not None and t1 is not None) else 0
    return {"count": n, "duration_s": round(dur, 6),
            "rate_hz": round(n / dur, 1) if dur > 0 else None,
            "pol_on": on, "pol_off": off,
            "on_off_ratio": round(on / off, 4) if off else None,
            "x_max": xmax, "y_max": ymax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--user", default="user1")
    ap.add_argument("--eye", default="right", choices=ev.EYES)
    ap.add_argument("--session", default="session_1_0_2")
    ap.add_argument("--v2e", default="", help="path to v2e.py (else uses `v2e` console script)")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--pos-thres", default="0.2")
    ap.add_argument("--neg-thres", default="0.2")
    ap.add_argument("--cutoff-hz", default="30")
    args = ap.parse_args()
    ev.ensure_dir(args.out)

    sp = ev.p(ev.davis_root(args.root), args.user, args.eye, args.session)
    frames_dir = ev.p(sp, "frames")
    events_txt = ev.p(sp, "events", "events.txt")
    if not os.path.isdir(frames_dir):
        print(f"[!] frames dir not found: {frames_dir}", file=sys.stderr); sys.exit(1)

    frames = ev.frame_files(frames_dir)[: args.max_frames]
    ts = ev.read_timestamps_txt(ev.p(frames_dir, "timestamps.txt"))
    fps = 25.0
    if len(ts) >= 2:
        dt = (ts[-1][1] - ts[0][1]) / 1e6 / (len(ts) - 1)
        if dt > 0:
            fps = round(1.0 / dt, 3)
    print(f"[i] {len(frames)} frames @ ~{fps} fps")

    try:
        import cv2  # noqa
    except ImportError:
        print("[!] OpenCV not installed: pip install opencv-python", file=sys.stderr); sys.exit(2)

    work = tempfile.mkdtemp(prefix="v2e_check_")
    avi = ev.p(work, "src.avi")
    v2e_out = ev.p(work, "v2e_out")
    try:
        if not build_avi(frames, avi, fps):
            print("[!] failed to build avi", file=sys.stderr); sys.exit(3)
        # locate v2e
        if args.v2e and os.path.isfile(args.v2e):
            base_cmd = [sys.executable, args.v2e]
        elif shutil.which("v2e"):
            base_cmd = ["v2e"]
        else:
            print("[!] v2e not found. Install it (pip install -e third/v2e) or pass --v2e "
                  "path/to/v2e.py. Skipping synthesis; only REAL stats will be written.",
                  file=sys.stderr)
            base_cmd = None

        synth = None
        if base_cmd:
            cmd = base_cmd + [
                "-i", avi, "--overwrite", "--output_folder", v2e_out,
                "--dvs_text", "v2e_events.txt", "--skip_video_output",
                "--no_preview", "--disable_slomo",
                "--pos_thres", args.pos_thres, "--neg_thres", args.neg_thres,
                "--cutoff_hz", args.cutoff_hz, "--input_frame_rate", str(fps),
                "--dvs346",
            ]
            print("[run]", " ".join(cmd))
            r = subprocess.run(cmd)
            dvs_text = ev.p(v2e_out, "v2e_events.txt")
            if r.returncode == 0 and os.path.isfile(dvs_text):
                synth = synth_event_stats(dvs_text)
            else:
                print(f"[!] v2e returned {r.returncode} or no output", file=sys.stderr)

        real = real_event_stats(events_txt)
        result = {"session": f"{args.user}/{args.eye}/{args.session}",
                  "fps": fps, "n_frames_used": len(frames),
                  "real": real, "synthetic": synth}
        if real and synth and real.get("rate_hz") and synth.get("rate_hz"):
            result["rate_ratio_synth_over_real"] = round(synth["rate_hz"] / real["rate_hz"], 3)
            if real.get("on_off_ratio") and synth.get("on_off_ratio"):
                result["onoff_ratio_diff"] = round(synth["on_off_ratio"] - real["on_off_ratio"], 3)
        ev.dump_json(result, ev.p(args.out, "06_v2e_vs_real.json"))
        print("\n=== v2e vs REAL ===")
        print(result)
        print(f"[i] wrote {ev.p(args.out, '06_v2e_vs_real.json')}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    import json  # noqa
    main()
