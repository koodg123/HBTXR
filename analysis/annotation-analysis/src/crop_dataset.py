"""crop_dataset.py — build a marker-excluded Eye-ROI dataset (APS frames + Event stream),
cropped to a data-driven box so the bottom calibration-marker pillars are removed.

Design (validated in docs/12 + measured envelope):
- Pupil center envelope (all valid GSAM2/U-Net centers) defines the eye ROI; markers sit in
  the bottom band (cy>=189). The crop box = center-envelope + margin, with the BOTTOM capped
  below the marker line -> keeps every pupil CENTER, drops the marker pillars.
- APS: crop the grayscale frame. Events: spatial filter (x,y in box) + shift. Labels: subtract
  crop offset (radii/theta unchanged). Native crop resolution (max real resolution; DAVIS346
  is the ceiling). Both modalities share the SAME box/grid so they stay aligned.

Output: datasets/eye_crop_{W}x{H}/{key}/{aps/*.png, events.npz, labels.json} + meta.json + _qc/.
Run in .venv-gsam2 (cv2+numpy). Marker-free by construction; no per-frame GSAM2 needed.
"""
import os, sys, glob, json, math, argparse
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S

FRAME = f"{S.AA}/samples/frame"
EVENT = f"{S.AA}/samples/event"


def read_gray(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        from PIL import Image
        im = np.array(Image.open(path).convert("L"))
    return im


def collect_centers():
    """all valid pupil centers (gsam2 non-mislabel preferred, unet fallback) + equiv radii."""
    cs, rs = [], []
    for gj in sorted(glob.glob(f"{S.LAB}/*/gsam2.json")):
        key = os.path.basename(os.path.dirname(gj))
        g = {c["idx"]: c for c in json.load(open(gj))["gsam2_centers"]}
        u = {c["idx"]: c for c in json.load(open(f"{S.LAB}/{key}/unet_dense.json"))["unet_centers"]}
        for i in set(g) | set(u):
            gc, uc = g.get(i), u.get(i)
            if gc and gc.get("valid") and not gc.get("mislabel", False):
                cs.append((gc["cx"], gc["cy"]))
                if gc.get("area"):
                    rs.append((gc["area"] / math.pi) ** 0.5)
            elif uc and uc.get("valid"):
                cs.append((uc["cx"], uc["cy"]))
                if uc.get("area"):
                    rs.append((uc["area"] / math.pi) ** 0.5)
    return cs, rs


def compute_box(cs, rs, margin=8.0, marker_cap=188, snap=16):
    cx = np.array([c[0] for c in cs]); cy = np.array([c[1] for c in cs])
    r = float(np.percentile(rs, 95)) if rs else 18.0
    m = r + margin                                        # cover pupil disc + buffer
    x0d, x1d = cx.min() - m, cx.max() + m
    y0d, y1d = cy.min() - m, min(cy.max() + m, marker_cap)

    def snap_axis(a, b, lo, hi, cap_hi):
        w = int(math.ceil((b - a) / snap) * snap)
        b2 = min(b, cap_hi); a2 = b2 - w                  # anchor to bottom (respect marker cap)
        if a2 < lo:
            a2 = lo; b2 = lo + w
        if b2 > hi:
            b2 = hi; a2 = hi - w
        return int(round(max(a2, lo))), int(round(min(b2, hi)))
    x0, x1 = snap_axis(x0d, x1d, 0, S.W, S.W)
    y0, y1 = snap_axis(y0d, y1d, 0, S.H, marker_cap)
    return (x0, y0, x1, y1), dict(radius_p95=round(r, 2),
                                  center_env=dict(x=[float(cx.min()), float(cx.max())],
                                                  y=[float(cy.min()), float(cy.max())]))


def _load_src(key):
    def cc(fn, fld):
        try:
            return {c["idx"]: c for c in json.load(open(f"{S.LAB}/{key}/{fn}"))[fld]}
        except (FileNotFoundError, KeyError):
            return {}
    return (cc("unet_dense.json", "unet_centers"), cc("gsam2.json", "gsam2_centers"),
            cc("pred.json", "pred_centers"))


def process(box, out_root, limit=0, qc_n=6):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    os.makedirs(out_root, exist_ok=True)
    qc_dir = f"{out_root}/_qc"; os.makedirs(qc_dir, exist_ok=True)
    keys = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{S.LAB}/*/gt.json"))
    if limit:
        keys = keys[:limit]
    n_fr = n_ev_win = 0
    clip_center = 0
    marker_px_before = marker_px_after = 0
    ev_in = ev_out = 0
    for ki, key in enumerate(keys):
        gt = json.load(open(f"{S.LAB}/{key}/gt.json"))
        ai = gt["anchor_idx"]
        unet, gsam2, pred = _load_src(key)
        kd = f"{out_root}/{key}"; os.makedirs(f"{kd}/aps", exist_ok=True)
        # --- APS frames ---
        labels = dict(key=key, box=[x0, y0, x1, y1], orig=[S.W, S.H], crop=[W, H],
                      anchor_idx=ai, frames={})
        for fp in sorted(glob.glob(f"{FRAME}/{key}/*.png")):
            stem = os.path.basename(fp)[:-4]
            idx = int(stem.split("_")[0]); ts = stem.split("_")[1]
            im = read_gray(fp)
            marker_px_before += int((im[189:, :] > 0).sum() > 0)   # any content in marker band
            crop = im[y0:y1, x0:x1]
            cv2.imwrite(f"{kd}/aps/{idx:06d}_{ts}.png", crop)
            n_fr += 1
            fr = {"ts": ts}
            if idx == ai:
                cx, cy, rx, ry, th = gt["ellipse_cx_cy_rx_ry_theta"]
                inb = x0 <= cx < x1 and y0 <= cy < y1
                clip_center += int(not inb)
                fr["gt"] = [round(cx - x0, 3), round(cy - y0, 3), rx, ry, th]
            for name, d in (("unet", unet), ("gsam2", gsam2), ("pred", pred)):
                c = d.get(idx)
                ok = c and c.get("valid") and (name != "gsam2" or not c.get("mislabel", False)) and "cx" in c
                if ok and x0 <= c["cx"] < x1 and y0 <= c["cy"] < y1:
                    fr[name] = [round(c["cx"] - x0, 3), round(c["cy"] - y0, 3)]
            labels["frames"][str(idx)] = fr
        json.dump(labels, open(f"{kd}/labels.json", "w"))
        # --- Events (spatial crop + shift) ---
        ep = f"{EVENT}/{key}.npz"
        if os.path.exists(ep):
            d = np.load(ep)
            xx, yy = d["x"], d["y"]
            m = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
            ev_in += len(xx); ev_out += int(m.sum())
            np.savez_compressed(f"{kd}/events.npz",
                                t=d["t"][m], x=(xx[m] - x0).astype(np.int16),
                                y=(yy[m] - y0).astype(np.int16), p=d["p"][m],
                                ts_lo=d["ts_lo"], ts_hi=d["ts_hi"], box=np.array([x0, y0, x1, y1]))
            n_ev_win += 1
        # --- QC overlay (a few) ---
        if ki < qc_n:
            afp = glob.glob(f"{FRAME}/{key}/{ai:06d}_*.png")
            if afp:
                crop = cv2.cvtColor(read_gray(afp[0])[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
                cx, cy = gt["ellipse_cx_cy_rx_ry_theta"][:2]
                cv2.circle(crop, (int(cx - x0), int(cy - y0)), 3, (0, 255, 0), -1)
                cv2.imwrite(f"{qc_dir}/{key}.png", cv2.resize(crop, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST))
    meta = dict(box=[x0, y0, x1, y1], crop_resolution=[W, H], orig=[S.W, S.H],
                n_windows=len(keys), n_aps_frames=n_fr, n_event_windows=n_ev_win,
                centers_clipped=clip_center, frames_with_marker_band_content=marker_px_before,
                event_kept_frac=round(ev_out / max(ev_in, 1), 4),
                note="markers excluded by capping box bottom below cy=189; native crop resolution")
    json.dump(meta, open(f"{out_root}/meta.json", "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--box", default=None, help="x0,y0,x1,y1 (override)")
    ap.add_argument("--marker-cap", type=int, default=188)
    ap.add_argument("--margin", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cs, rs = collect_centers()
    if a.box:
        box = tuple(int(v) for v in a.box.split(",")); info = {}
    else:
        box, info = compute_box(cs, rs, margin=a.margin, marker_cap=a.marker_cap)
    W, H = box[2] - box[0], box[3] - box[1]
    out = a.out or f"{S.AA}/datasets/eye_crop_{W}x{H}"
    print(f"[crop] centers n={len(cs)} envelope={info.get('center_env')} r_p95={info.get('radius_p95')}")
    print(f"[crop] BOX={box}  crop_resolution={W}x{H}  marker_cap={a.marker_cap}")
    meta = process(box, out, limit=a.limit)
    print(f"[crop] windows={meta['n_windows']} aps_frames={meta['n_aps_frames']} event_windows={meta['n_event_windows']}")
    print(f"[crop] centers_clipped={meta['centers_clipped']}  event_kept_frac={meta['event_kept_frac']}")
    print(f"[crop] -> {out}  (aps/ + events.npz + labels.json per window; _qc/ overlays; meta.json)")


if __name__ == "__main__":
    main()
