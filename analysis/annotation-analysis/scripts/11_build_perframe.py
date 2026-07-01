"""
11_build_perframe.py — reshape the window-level `samples/label/*` into a per-frame
(per-image) directory tree under `samples/perframe/`.

For every frame listed in manifest_frames.csv it creates:

  samples/perframe/{key}/window.json                 # window meta (once per key)
  samples/perframe/{key}/{stem}/                      # stem = frame file name w/o .png
    ├─ frame.png            # copy (or --link symlink) of the APS frame
    ├─ meta.json            # {key, idx, ts, role, motion, user, eye, session}
    ├─ gt.json              # human GT ellipse            (anchor only)
    ├─ gt_bbox.json         # AABB from the GT ellipse    (anchor only)
    ├─ unet/  center.json  mask.png  bbox.json           # U-Net (all frames)
    ├─ gsam2/ center.json  mask.png  bbox.json           # GSAM2 (needs 08 --save-masks)
    └─ pred/  center.json                                # HBTXR (only if pred.json exists)
  samples/perframe/index.csv                          # one row per frame + presence flags

Sources (window level): label/{key}/{gt,unet_dense,gsam2,pred}.json,
label/{key}/unet_masks/{stem}_mask.gif, label/{key}/gsam2_masks/{stem}_mask.png.

Masks are written as 0/255 binary PNG (346x260). bbox = {"xyxy":[x0,y0,x1,y1],...}.
This script is pure post-processing (no GPU); safe to re-run (idempotent per frame).
"""
import os, csv, json, glob, math, shutil, argparse
import numpy as np
from PIL import Image
import evlib as ev

BBOX_KEYS = ("x0", "y0", "x1", "y1")


def ellipse_aabb(cx, cy, rx, ry, theta):
    """Axis-aligned bbox of a rotated ellipse (theta in radians, rx/ry semi-axes)."""
    ct, st = math.cos(theta), math.sin(theta)
    dx = math.hypot(rx * ct, ry * st)
    dy = math.hypot(rx * st, ry * ct)
    return [round(cx - dx, 2), round(cy - dy, 2), round(cx + dx, 2), round(cy + dy, 2)]


def mask_bbox(mask):
    """Tight bbox [x0,y0,x1,y1] of nonzero pixels, or None if empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def load_json(path):
    return json.load(open(path)) if os.path.isfile(path) else None


def by_idx(dct, listkey):
    """{idx: entry} from a {listkey:[{idx,...}]} json (or {} )."""
    if not dct:
        return {}
    return {e["idx"]: e for e in dct.get(listkey, []) if "idx" in e}


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../samples", help="samples root (has frame/, label/, manifests)")
    ap.add_argument("--perframe", default=None, help="output dir (default: <out>/perframe)")
    ap.add_argument("--link", action="store_true", help="symlink frame.png instead of copying")
    ap.add_argument("--limit", type=int, default=0, help="process only first N frames (debug)")
    ap.add_argument("--overwrite", action="store_true", help="rewrite frame dirs that already exist")
    a = ap.parse_args()
    out = a.out
    pf = a.perframe or ev.p(out, "perframe")
    os.makedirs(pf, exist_ok=True)

    # window meta (for window.json)
    winmeta = {}
    with open(ev.p(out, "manifest_windows.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            winmeta[r["key"]] = r

    frames = list(csv.DictReader(open(ev.p(out, "manifest_frames.csv"), encoding="utf-8")))
    if a.limit:
        frames = frames[: a.limit]
    print(f"[i] {len(frames)} frames -> {pf}")

    # per-key caches (avoid re-reading label jsons per frame)
    cache = {}

    def get_key_labels(key):
        if key not in cache:
            ld = ev.p(out, "label", key)
            cache[key] = {
                "gt": load_json(ev.p(ld, "gt.json")),
                "unet": by_idx(load_json(ev.p(ld, "unet_dense.json")), "unet_centers"),
                "gsam2": by_idx(load_json(ev.p(ld, "gsam2.json")), "gsam2_centers"),
                "pred": by_idx(load_json(ev.p(ld, "pred.json")), "pred_centers"),
                "ld": ld,
            }
        return cache[key]

    idx_rows = []
    seen_win = set()
    n = 0
    for r in frames:
        key = r["key"]
        idx = int(r["frame_index"])
        stem = os.path.splitext(os.path.basename(r["dst_frame"]))[0]
        role = r["role"]
        lab = get_key_labels(key)

        # window.json once
        if key not in seen_win:
            seen_win.add(key)
            os.makedirs(ev.p(pf, key), exist_ok=True)
            if key in winmeta:
                write_json(winmeta[key], ev.p(pf, key, "window.json"))

        fdir = ev.p(pf, key, stem)
        if os.path.isdir(fdir) and not a.overwrite:
            pass  # still (re)compute presence flags below cheaply
        os.makedirs(fdir, exist_ok=True)

        # frame.png (copy or symlink)
        srcpng = ev.p(out, "frame", key, os.path.basename(r["dst_frame"]))
        dstpng = ev.p(fdir, "frame.png")
        if not os.path.exists(dstpng) or a.overwrite:
            if os.path.islink(dstpng) or os.path.exists(dstpng):
                os.remove(dstpng)
            if a.link:
                os.symlink(os.path.abspath(srcpng), dstpng)
            elif os.path.isfile(srcpng):
                shutil.copy2(srcpng, dstpng)

        # meta.json
        write_json({"key": key, "idx": idx, "ts": int(r["frame_ts_us"]), "role": role,
                    "motion": r["motion"], "user": r["user"], "eye": r["eye"],
                    "session": r["session"]}, ev.p(fdir, "meta.json"))

        # gt (anchor only)
        has_gt = 0
        if role == "anchor" and lab["gt"]:
            gt = lab["gt"]
            write_json(gt, ev.p(fdir, "gt.json"))
            e = gt.get("ellipse_cx_cy_rx_ry_theta")
            if e:
                cx, cy, rx, ry, th = e
                write_json({"xyxy": ellipse_aabb(cx, cy, rx, ry, th), "source": "gt_ellipse"},
                           ev.p(fdir, "gt_bbox.json"))
            has_gt = 1

        # U-Net (all frames)
        has_unet = has_unet_mask = 0
        u = lab["unet"].get(idx)
        if u and u.get("valid"):
            os.makedirs(ev.p(fdir, "unet"), exist_ok=True)
            write_json({k: u.get(k) for k in ("cx", "cy", "area", "valid")},
                       ev.p(fdir, "unet", "center.json"))
            has_unet = 1
            gif = ev.p(lab["ld"], "unet_masks", stem + "_mask.gif")
            if os.path.isfile(gif):
                m = (np.array(Image.open(gif).convert("L")) > 0).astype("uint8")
                Image.fromarray(m * 255).save(ev.p(fdir, "unet", "mask.png"))
                bb = mask_bbox(m)
                if bb:
                    write_json({"xyxy": bb, "source": "unet_mask"}, ev.p(fdir, "unet", "bbox.json"))
                has_unet_mask = 1

        # GSAM2 (needs full 08 + --save-masks)
        has_gsam2 = has_gsam2_mask = 0
        g = lab["gsam2"].get(idx)
        if g and g.get("valid"):
            os.makedirs(ev.p(fdir, "gsam2"), exist_ok=True)
            write_json({k: g.get(k) for k in ("cx", "cy", "area", "det_score", "sam_score",
                                              "repeats", "valid", "mislabel", "mislabel_reason",
                                              "e_unet")},
                       ev.p(fdir, "gsam2", "center.json"))
            gbb = {"xyxy": [round(float(v), 2) for v in g["box"]], "source": "gdino_detection"} \
                if g.get("box") else {}
            has_gsam2 = 1
            gpng = ev.p(lab["ld"], "gsam2_masks", stem + "_mask.png")
            if os.path.isfile(gpng):
                shutil.copy2(gpng, ev.p(fdir, "gsam2", "mask.png"))
                m = (np.array(Image.open(gpng).convert("L")) > 0).astype("uint8")
                mb = mask_bbox(m)
                if mb:
                    gbb["mask_xyxy"] = mb
                has_gsam2_mask = 1
            if gbb:
                write_json(gbb, ev.p(fdir, "gsam2", "bbox.json"))

        # pred (only if present)
        has_pred = 0
        p = lab["pred"].get(idx)
        if p and p.get("valid"):
            os.makedirs(ev.p(fdir, "pred"), exist_ok=True)
            write_json({k: p.get(k) for k in ("cx", "cy", "grid_x", "grid_y", "score", "valid")},
                       ev.p(fdir, "pred", "center.json"))
            has_pred = 1

        idx_rows.append({"key": key, "stem": stem, "idx": idx, "ts": r["frame_ts_us"],
                         "role": role, "motion": r["motion"], "user": r["user"], "eye": r["eye"],
                         "session": r["session"], "has_gt": has_gt, "has_unet": has_unet,
                         "has_unet_mask": has_unet_mask, "has_gsam2": has_gsam2,
                         "has_gsam2_mask": has_gsam2_mask, "has_pred": has_pred})
        n += 1
        if n % 500 == 0:
            print(f"  {n}/{len(frames)} frames")

    # index.csv
    with open(ev.p(pf, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(idx_rows[0].keys()))
        w.writeheader(); w.writerows(idx_rows)

    # summary
    def s(col):
        return sum(row[col] for row in idx_rows)
    print(f"\n[done] frames={n}  windows={len(seen_win)}")
    print(f"  has_gt={s('has_gt')}  has_unet={s('has_unet')}  has_unet_mask={s('has_unet_mask')}  "
          f"has_gsam2={s('has_gsam2')}  has_gsam2_mask={s('has_gsam2_mask')}  has_pred={s('has_pred')}")
    print(f"  -> {pf}/  (+ index.csv)")


if __name__ == "__main__":
    main()
