"""
09_inject_pred.py — inject HBTXR pupil-center predictions (y_pred) over collected samples.

Uses FACET's HBTXR inference helpers on OUR collected event windows. For each target
frame it builds the causal event frame from the last --n-events events <= frame ts,
runs the model, decodes the center in the model grid, and maps to sensor pixels
(346x260) so it is directly comparable to y_orig / y_unet / y_gsam2.

Writes:  samples/label/{key}/pred.json = {"mode","pred_centers":[{idx,ts,valid,cx,cy,score}]}

RUN IN THE FACET ENV (torch + EvEye + the HBTXR deps), with a trained HBTXR .ckpt.

Example (subject-independent test users 1-10 were used to collect; use the matching ckpt):
  python 09_inject_pred.py --out ../samples \
    --facet-root ../../../references/codebase/software/FACET \
    --config ../../../references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet.yaml \
    --ckpt /path/to/HBTXR_best.ckpt --img-size 256 --patch-size 4 \
    --n-events 5000 --mode event --anchors-only
"""
import os, csv, glob, json, argparse, sys
import numpy as np


def load_facet(a):
    sys.path.insert(0, os.path.abspath(a.facet_root))
    import yaml, torch, albumentations as A
    from EvEye.model.model_factory import make_model
    # NOTE: Predict.pre_process hardcodes 256x256 (the img256/full_unet model). For the
    # img64 (subject-independent) ckpt the DeiT PatchEmbed *requires* exactly img_size x
    # img_size input (it raises otherwise), so we resize to a.img_size to match training.
    from EvEye.model.DavisEyeEllipse.HBTXR.Predict import event_to_frame, post_process

    def pre_process(event_frame):
        aug = A.Compose([A.Resize(a.img_size, a.img_size)])(image=event_frame)["image"]
        x = (aug / 255.0).astype("float32")
        x = np.moveaxis(x, -1, 0)[None]        # (H,W,2) -> (1, 2, img_size, img_size)
        return torch.from_numpy(x)

    with open(a.config, "r") as f:
        cfg = yaml.safe_load(f)
    model = make_model(cfg["model"])
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[i] {len(missing)} missing keys (ok if buffers), e.g. {missing[:3]}")
    model.eval().to(a.device)
    return model, cfg, torch, (event_to_frame, pre_process, post_process)


def seg_from_npz(npz, t_target, n_events):
    """Structured event array (fields t,x,y,p) of the last n_events with t<=t_target."""
    t = npz["t"]
    hi = int(np.searchsorted(t, t_target, side="right"))
    lo = max(0, hi - n_events)
    n = hi - lo
    if n <= 0:
        return None
    seg = np.zeros(n, dtype=[("t", "<i8"), ("x", "<i2"), ("y", "<i2"), ("p", "i1")])
    seg["t"] = t[lo:hi]
    seg["x"] = npz["x"][lo:hi]
    seg["y"] = npz["y"][lo:hi]
    seg["p"] = npz["p"][lo:hi]
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../samples")
    ap.add_argument("--facet-root", default="../../../references/codebase/software/FACET")
    ap.add_argument("--config", required=True, help="HBTXR yaml (for make_model)")
    ap.add_argument("--ckpt", required=True, help="trained HBTXR .ckpt (state_dict)")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--sensor-w", type=int, default=346)
    ap.add_argument("--sensor-h", type=int, default=260)
    ap.add_argument("--n-events", type=int, default=5000)
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--mode", default="event", help="label tag (event/hybrid/frame)")
    ap.add_argument("--anchors-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    G = args.img_size // args.patch_size          # output grid (e.g. 256/4 = 64)
    sx, sy = args.sensor_w / G, args.sensor_h / G  # grid -> sensor px scale
    print(f"[i] grid={G}x{G}, scale x*{sx:.3f} y*{sy:.3f}; loading model ...")
    model, cfg, torch, (event_to_frame, pre_process, post_process) = load_facet(args)

    # frame ts + role per (key, idx)
    frames_meta = {}
    mf = os.path.join(args.out, "manifest_frames.csv")
    with open(mf, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            frames_meta[(r["key"], int(r["frame_index"]))] = (
                int(r["frame_ts_us"]), r["role"])

    keys = sorted(os.listdir(os.path.join(args.out, "label")))
    done = nvalid = ntotal = 0
    for key in keys:
        ldir = os.path.join(args.out, "label", key)
        outp = os.path.join(ldir, "pred.json")
        if os.path.isfile(outp) and not args.overwrite:
            continue
        ev_file = os.path.join(args.out, "event", key + ".npz")
        if not os.path.isfile(ev_file):
            continue
        npz = np.load(ev_file)
        frames = sorted(glob.glob(os.path.join(args.out, "frame", key, "*.png")))
        rows = []
        for fp in frames:
            base = os.path.basename(fp)
            try:
                idx = int(base.split("_")[0])
            except ValueError:
                continue
            meta = frames_meta.get((key, idx))
            if meta is None:
                continue
            ts, role = meta
            if args.anchors_only and role != "anchor":
                continue
            seg = seg_from_npz(npz, ts, args.n_events)
            if seg is None or len(seg) < 10:
                rows.append({"idx": idx, "ts": ts, "valid": False, "reason": "too_few_events"})
                ntotal += 1
                continue
            frame = event_to_frame(seg, sensor_size=(args.sensor_w, args.sensor_h, 2),
                                   events_interpolation="causal_linear", weight=1)
            inp = pre_process(frame).to(args.device)
            with torch.no_grad():
                out = model(inp)
            dets = post_process(out)
            xs = float(dets["xs"][0, 0]); ys = float(dets["ys"][0, 0])
            score = float(dets["scores"][0, 0]) if "scores" in dets else 1.0
            valid = score >= args.score_thr
            rows.append({"idx": idx, "ts": ts, "valid": bool(valid),
                         "cx": round(xs * sx, 3), "cy": round(ys * sy, 3),
                         "grid_x": round(xs, 3), "grid_y": round(ys, 3),
                         "score": round(score, 4)})
            ntotal += 1
            nvalid += int(valid)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump({"mode": args.mode, "ckpt": os.path.basename(args.ckpt),
                       "grid": G, "pred_centers": rows}, f, ensure_ascii=False, indent=2)
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(keys)} windows  (valid {nvalid}/{ntotal})")

    print(f"\n[done] windows={done} frames={ntotal} valid={nvalid} "
          f"({100*nvalid/max(1,ntotal):.1f}%)")
    print("  -> samples/label/*/pred.json ready for 10_eval")


if __name__ == "__main__":
    main()
