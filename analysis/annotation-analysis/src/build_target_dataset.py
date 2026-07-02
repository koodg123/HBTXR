"""build_target_dataset.py — build Dataset_full_gsam2_subject_independent cache from an
Eye-ROI crop of the EV-Eye data, in the FACET MemmapCacheStructedEvents format.

Per frame it writes: EVENT (crop, per-frame slice) + APS (crop PNG) + GSAM2 ELLIPSE
(crop, from mask fit) + GSAM2 MASK (crop, packed-bit) + QC + original human GT/timestamp.
Cache layout mirrors DeanDataset_full_unet_subject_independent:
  cached_data/  events_batch_{N}.memmap [t,x,y,p]<i8 (merged) + events_indices_{N}.npy [start,end]/frame + info
  cached_ellipse/ ellipse_records.npy [t,x,y,a,b,ang] + ellipses_batch_{N}.memmap + indices [i,i+1] + info + ellipse_qc.npy
  cached_mask/  masks_batch_{N}.memmap (n,packed) uint8 + mask_info_{N}.txt   (np.packbits of 240x160)
  cached_aps/   {key}/{idx6}_{ts}.png (crop 240x160)
  labels_original/ human_ellipse.npy, frame_index.npy
+ manifest.json, crop_boxes.json, qc_summary.json

PoC: reads the audit set (samples/, GSAM2 already computed). Run in .venv-gsam2.
"""
import os, sys, glob, json, math, argparse
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S
import motion_label as ML

FRAME = f"{S.AA}/samples/frame"
EVENT = f"{S.AA}/samples/event"
MOT_DT = np.dtype([("t", "<i8"), ("motion", "u1"), ("velocity", "<f4"), ("blink", "?"), ("conf", "u1")])
EV_DT = np.dtype([("t", "<i8"), ("x", "<i8"), ("y", "<i8"), ("p", "<i8")])
EL_DT = np.dtype([("t", "<i8"), ("x", "<f8"), ("y", "<f8"), ("a", "<f8"), ("b", "<f8"), ("ang", "<f8")])
QC_DT = np.dtype([("t", "<i8"), ("valid", "?"), ("mislabel", "?"), ("blink", "?"),
                  ("det", "<f4"), ("area", "<f4"), ("source", "u1")])
HU_DT = np.dtype([("t", "<i8"), ("nx", "<f8"), ("ny", "<f8"), ("a", "<f8"), ("b", "<f8"), ("ang", "<f8"),
                  ("cx", "<f8"), ("cy", "<f8"), ("subject", "u1"), ("eye", "u1"), ("session", "u2"), ("idx", "<i8")])
FI_DT = np.dtype([("idx", "<i8"), ("t", "<i8"), ("subject", "u1"), ("eye", "u1"),
                  ("session", "u2"), ("box_id", "<i2"), ("key", "U80")])


def read_gray(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return im if im is not None else None


def write_memmap(data, base):
    mm = np.memmap(f"{base}.memmap", dtype=data.dtype, mode="w+", shape=data.shape)
    mm[:] = data[:]; mm.flush()
    open(f"{base}_info.txt", "w").write(f"Data shape: {data.shape}\nData dtype: {data.dtype}\n")


def ellipse_from_mask(mcrop):
    m = (mcrop > 0).astype(np.uint8)
    if m.sum() < 10:
        return None
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == i).astype(np.uint8)
    area = float(stats[i, cv2.CC_STAT_AREA])
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts and len(cnts[0]) >= 5:
        (cx, cy), (A, B), ang = cv2.fitEllipse(cnts[0])
    else:
        cc = cent[i]; cx, cy = float(cc[0]), float(cc[1]); A = B = 2 * math.sqrt(area / math.pi); ang = 0.0
    return cx, cy, float(A), float(B), float(ang), area, comp


def session_code(eye, session):
    return (0 if eye == "left" else 1) * 1000 + int(session.replace("_", ""))  # e.g. left 1_0_2 -> 102


def build(box, out_root, split="train", batch_size=5000, limit=0):
    x0, y0, x1, y1 = box; W, H = x1 - x0, y1 - y0
    base = f"{out_root}/{split}"
    for d in ("cached_data", "cached_ellipse", "cached_mask", "cached_aps", "labels_original"):
        os.makedirs(f"{base}/{d}", exist_ok=True)
    keys = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{S.LAB}/*/gt.json"))
    if limit:
        keys = keys[:limit]

    ev_frames, ell_recs, qc_recs, mask_packed, human_recs, fi_recs, mot_recs = [], [], [], [], [], [], []
    box_ids = {}
    PACK = int(np.ceil(H * W / 8))
    for key in keys:
        p = S.parse_key(key)
        subj, eye, sess = p["user"], p["eye"], p["session"]
        box_ids.setdefault(key, len(box_ids))
        gt = json.load(open(f"{S.LAB}/{key}/gt.json")); ai = gt["anchor_idx"]
        gsx = {c["idx"]: c for c in json.load(open(f"{S.LAB}/{key}/gsam2.json"))["gsam2_centers"]}
        unet = {c["idx"]: c for c in json.load(open(f"{S.LAB}/{key}/unet_dense.json"))["unet_centers"]}
        mot = ML.label_window(key)
        conf = 0 if sess == "1_0_1" else 1
        ev = np.load(f"{EVENT}/{key}.npz")
        et, ex, ey, ep = ev["t"], ev["x"], ev["y"], ev["p"]
        order = np.argsort(et); et, ex, ey, ep = et[order], ex[order], ey[order], ep[order]
        ts_lo = int(ev["ts_lo"])
        frames = sorted(glob.glob(f"{FRAME}/{key}/*.png"))
        f_ts = [int(os.path.basename(f)[:-4].split("_")[1]) for f in frames]
        bounds = [ts_lo] + f_ts                                        # per-frame event slice (prev, cur]
        for fi, fp in enumerate(frames):
            stem = os.path.basename(fp)[:-4]; idx = int(stem.split("_")[0]); ts = int(stem.split("_")[1])
            im = read_gray(fp)
            crop = im[y0:y1, x0:x1]
            kd = f"{base}/cached_aps/{key}"; os.makedirs(kd, exist_ok=True)
            cv2.imwrite(f"{kd}/{idx:06d}_{ts}.png", crop)
            # events for this frame -> crop + shift
            lo, hi = np.searchsorted(et, bounds[fi], "right"), np.searchsorted(et, bounds[fi + 1], "right")
            sx, sy, st, sp = ex[lo:hi], ey[lo:hi], et[lo:hi], ep[lo:hi]
            m = (sx >= x0) & (sx < x1) & (sy >= y0) & (sy < y1)
            fa = np.zeros(int(m.sum()), EV_DT)
            fa["t"] = st[m]; fa["x"] = sx[m] - x0; fa["y"] = sy[m] - y0; fa["p"] = sp[m]
            ev_frames.append(fa)
            # gsam2 mask -> ellipse (crop) + packed mask
            gc = gsx.get(idx)
            mp = glob.glob(f"{S.LAB}/{key}/gsam2_masks/{idx:06d}_*_mask.png")
            fit = None
            if mp:
                mfull = read_gray(mp[0])
                if mfull is not None:
                    fit = ellipse_from_mask(mfull[y0:y1, x0:x1])
            valid = fit is not None and gc is not None and gc.get("valid")
            mislabel = bool(gc and gc.get("mislabel", False))
            if fit:
                cx, cy, A, B, ang, area, comp = fit
                ell_recs.append((ts, cx, cy, A, B, ang))
                mask_packed.append(np.packbits(comp.reshape(-1)))
            else:                                                       # fallback: unet center, empty mask
                uc = unet.get(idx)
                if uc and uc.get("valid") and x0 <= uc["cx"] < x1 and y0 <= uc["cy"] < y1:
                    ell_recs.append((ts, uc["cx"] - x0, uc["cy"] - y0, 0.0, 0.0, 0.0)); src = 1
                else:
                    ell_recs.append((ts, -1.0, -1.0, 0.0, 0.0, 0.0)); src = 1
                mask_packed.append(np.zeros(PACK, np.uint8))
            src = 0 if fit else 1
            qc_recs.append((ts, bool(valid) and not mislabel, mislabel, False,
                            float(gc["det_score"]) if gc and "det_score" in gc else 0.0,
                            float(fit[5]) if fit else 0.0, src))
            if idx == ai:
                cx0, cy0, ra, rb, th = gt["ellipse_cx_cy_rx_ry_theta"]
                human_recs.append((ts, cx0, cy0, 2 * ra, 2 * rb, th, cx0 - x0, cy0 - y0,
                                   subj, 0 if eye == "left" else 1, int(sess.replace("_", "")), idx))
            fi_recs.append((idx, ts, subj, 0 if eye == "left" else 1, int(sess.replace("_", "")), box_ids[key], key))
            mr = mot.get(idx, {})
            mot_recs.append((ts, mr.get("motion", 0), mr.get("velocity") if mr.get("velocity") is not None else np.nan,
                             bool(mr.get("blink", False)), conf))

    n = len(ev_frames)
    # ---- write batched caches ----
    def get_indices(arrs):
        idx = np.zeros((len(arrs), 2), np.int32); c = 0
        for j, a in enumerate(arrs):
            e = c + (a.shape[0] if a.shape else 1); idx[j] = [c, e]; c = e
        return idx
    ell_arr = np.array(ell_recs, EL_DT)
    mask_arr = np.stack(mask_packed).astype(np.uint8)
    np.save(f"{base}/cached_ellipse/ellipse_records.npy", ell_arr)
    np.save(f"{base}/cached_ellipse/ellipse_qc.npy", np.array(qc_recs, QC_DT))
    for b, s in enumerate(range(0, n, batch_size)):
        sl = slice(s, min(s + batch_size, n))
        merged = np.concatenate(ev_frames[sl]) if any(a.shape[0] for a in ev_frames[sl]) else np.zeros(0, EV_DT)
        write_memmap(merged, f"{base}/cached_data/events_batch_{b}")
        np.save(f"{base}/cached_data/events_indices_{b}.npy", get_indices(ev_frames[sl]))
        write_memmap(ell_arr[sl], f"{base}/cached_ellipse/ellipses_batch_{b}")
        np.save(f"{base}/cached_ellipse/ellipses_indices_{b}.npy",
                np.stack([np.arange(sl.stop - sl.start), np.arange(1, sl.stop - sl.start + 1)], 1).astype(np.int32))
        write_memmap(mask_arr[sl], f"{base}/cached_mask/masks_batch_{b}")
    np.save(f"{base}/labels_original/human_ellipse.npy", np.array(human_recs, HU_DT))
    np.save(f"{base}/labels_original/frame_index.npy", np.array(fi_recs, FI_DT))
    np.save(f"{base}/labels_original/motion_labels.npy", np.array(mot_recs, MOT_DT))
    mc = np.array([r[1] for r in mot_recs])
    motion = {ML.NAME[k]: int((mc == k).sum()) for k in range(4)}
    return dict(split=split, frames=n, batches=b + 1, events=int(sum(a.shape[0] for a in ev_frames)),
                human=len(human_recs), valid=int(sum(q[1] for q in qc_recs)),
                mislabel=int(sum(q[2] for q in qc_recs)), motion=motion, box_ids={k: v for k, v in box_ids.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{S.AA}/datasets/Dataset_full_gsam2_poc")
    ap.add_argument("--box", default="53,28,293,188")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    box = tuple(int(v) for v in a.box.split(","))
    W, H = box[2] - box[0], box[3] - box[1]
    meta = build(box, a.out, a.split, limit=a.limit)
    manifest = dict(source="EV-Eye Data_davis + events + GSAM2(ROI+geom) labels; Eye-ROI crop (marker-excluded)",
                    label_source="gsam2_roi_geom", modalities=["events", "aps"], labels=["ellipse", "mask", "motion"],
                    crop_box=list(box), resolution=[W, H], native=[S.W, S.H],
                    split_policy="leak_free_32_36_48 (PoC: audit users 1-10 as train)",
                    counts={meta["split"]: meta["frames"]}, note="Phase-0 PoC from audit set")
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    json.dump({k: list(box) for k in meta["box_ids"]}, open(f"{a.out}/crop_boxes.json", "w"), indent=1)
    json.dump(dict(frames=meta["frames"], valid=meta["valid"], mislabel=meta["mislabel"],
                   valid_frac=round(meta["valid"] / max(meta["frames"], 1), 4)),
              open(f"{a.out}/qc_summary.json", "w"), indent=1)
    print(f"[build] split={meta['split']} frames={meta['frames']} events={meta['events']:,} "
          f"batches={meta['batches']} human_GT={meta['human']} valid={meta['valid']} mislabel={meta['mislabel']}")
    print(f"[build] motion: {meta['motion']}")
    print(f"[build] resolution={W}x{H} box={box} -> {a.out}")


if __name__ == "__main__":
    main()
