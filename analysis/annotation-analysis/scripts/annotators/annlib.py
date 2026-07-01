"""annlib — shared eval harness for the third/ annotator tools (EllSeg/RITnet/
DeepVOG/Edge-Guided/SAM3-I/YOLOE) run as INDEPENDENT audit sources for EV-Eye
pupil centers.

Mirrors the 08 contract: frame(346x260) -> pupil binary mask -> ellipse-fit center,
compared to the human GT anchor. Every tool produces a pupil-class binary mask (at
whatever resolution it runs); annlib resizes it back to 346x260 and fits ONE uniform
ellipse for an apples-to-apples center/radius, so differences reflect the detector,
not the fit. Per-tool runners only supply load_fn + detect_fn and call run_tool().

Metric per anchor: err = ||center - GT||_2 (px, 346x260). radius_ratio = r_equiv /
GT r_equiv (>~1.5 flags iris/over-seg; center is blind to it — see docs/14).
"""
import os, sys, glob, json, math, csv, time, argparse
import numpy as np
import cv2

# Restore numpy aliases removed in numpy 2.0 that the 2019-era repos (EllSeg/Edge-Guided/
# RITnet) still use at import & forward time. Non-invasive: patches numpy for THIS process
# only, so we never edit third/ code. Imported before any repo import (runners import annlib
# first, repos in load_fn later).
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    for _a, _t in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(np, _a):
            setattr(np, _a, _t)
    if not hasattr(np, "in1d"):      # removed in numpy 2.0 -> isin (Edge-Guided postproc)
        np.in1d = np.isin

AA = "/home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis"
LAB = f"{AA}/samples/label"
FRAME = f"{AA}/samples/frame"
RESULTS = f"{AA}/results/annotators"
W, H = 346, 260                                   # frame is 346 wide x 260 tall (landscape)

# ------- GT / motion -------
_MOTION = None
def motion_of(key):
    global _MOTION
    if _MOTION is None:
        _MOTION = {}
        mp = f"{AA}/samples/manifest_windows.csv"
        if os.path.exists(mp):
            for r in csv.DictReader(open(mp)):
                _MOTION[r["key"]] = r["motion"]
    if key in _MOTION:
        return _MOTION[key]
    return "smooth_pursuit" if key.startswith("smooth_pursuit") else key.split("_", 1)[0]


def load_gt(key):
    gt = json.load(open(f"{LAB}/{key}/gt.json"))
    cx, cy, rx, ry, th = gt["ellipse_cx_cy_rx_ry_theta"]
    return dict(key=key, anchor_idx=gt["anchor_idx"], cx=float(cx), cy=float(cy),
                rx=float(rx), ry=float(ry), theta=float(th),
                r_equiv=math.sqrt(max(rx * ry, 1e-6)),
                good=bool(gt.get("quality", {}).get("good", True)),
                motion=motion_of(key))


def anchor_frame(key, anchor_idx):
    fr = sorted(glob.glob(f"{FRAME}/{key}/{anchor_idx:06d}_*.png"))
    return fr[0] if fr else None


def iter_anchors(good_only=False):
    for gj in sorted(glob.glob(f"{LAB}/*/gt.json")):
        key = os.path.basename(os.path.dirname(gj))
        gt = load_gt(key)
        if good_only and not gt["good"]:
            continue
        fp = anchor_frame(key, gt["anchor_idx"])
        if fp:
            yield key, fp, gt


def read_gray(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:                                # fallback via PIL if cv2 codec missing
        from PIL import Image
        im = np.array(Image.open(path).convert("L"))
    return im                                     # (H,W) = (260,346) uint8


# ------- mask -> center (uniform for all tools) -------
def mask_to_center(mask):
    """mask: HxW binary at 346x260. -> dict(cx,cy,r_equiv,area) or None. Largest CC,
    cv2.fitEllipse when >=5 contour pts else centroid."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() < 5:
        return None
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == i).astype(np.uint8)
    area = float(stats[i, cv2.CC_STAT_AREA])
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts and len(cnts[0]) >= 5:
        (ex, ey), (MA, ma), _ = cv2.fitEllipse(cnts[0])
        cx, cy = float(ex), float(ey)
        r_equiv = math.sqrt(max(MA * ma, 1e-6)) / 2.0
    else:
        cx, cy = float(cent[i][0]), float(cent[i][1])
        r_equiv = math.sqrt(area / math.pi)
    return dict(cx=cx, cy=cy, r_equiv=r_equiv, area=area)


def resize_mask(mask, w=W, h=H):
    m = np.asarray(mask).astype(np.uint8)
    if m.shape[:2] == (h, w):
        return m
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def center_from_maskres(mask):
    """mask at ANY resolution -> resize to 346x260 -> uniform ellipse-fit."""
    if mask is None:
        return None
    return mask_to_center(resize_mask(mask))


# ------- records / csv -------
COLS = ["key", "motion", "good", "valid", "cx", "cy", "r_equiv", "area", "score",
        "gt_cx", "gt_cy", "gt_r", "err", "radius_ratio"]


def make_record(key, gt, det, score=None):
    rec = dict(key=key, motion=gt["motion"], good=int(gt["good"]),
               gt_cx=round(gt["cx"], 3), gt_cy=round(gt["cy"], 3), gt_r=round(gt["r_equiv"], 3),
               score=("" if score is None else round(float(score), 4)))
    if det is None:
        rec.update(valid=0, cx="", cy="", r_equiv="", area="", err="", radius_ratio="")
    else:
        err = math.hypot(det["cx"] - gt["cx"], det["cy"] - gt["cy"])
        rr = det["r_equiv"] / gt["r_equiv"] if gt["r_equiv"] > 0 else ""
        rec.update(valid=1, cx=round(det["cx"], 3), cy=round(det["cy"], 3),
                   r_equiv=round(det["r_equiv"], 3), area=round(det["area"], 1),
                   err=round(err, 3), radius_ratio=(round(rr, 3) if rr != "" else ""))
    return rec


def write_csv(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        for r in records:
            wr.writerow({k: r.get(k, "") for k in COLS})


def five(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return (float("nan"),) * 5 + (0,)
    return (v.mean(), np.median(v), np.percentile(v, 95), np.percentile(v, 99),
            v.std(ddof=1) if len(v) > 1 else 0.0, len(v))


# ------- generic runner -------
def run_tool(name, load_fn, detect_fn, extra_args=None):
    """load_fn(device)->model ; detect_fn(model,img_gray,device)->(mask|None, score|None).
    mask is a pupil binary mask at ANY resolution (resized to 346x260 here)."""
    ap = argparse.ArgumentParser(description=f"annotator runner: {name}")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--limit", type=int, default=0, help="0=all anchors")
    ap.add_argument("--good-only", action="store_true", help="only quality.good anchors")
    ap.add_argument("--keys", default="", help="comma-separated keys (smoke test)")
    ap.add_argument("--out", default="", help="output csv (default results/annotators/<name>.csv)")
    ap.add_argument("--save-masks", action="store_true")
    if extra_args:
        extra_args(ap)
    a = ap.parse_args()
    out = a.out or f"{RESULTS}/{name}.csv"

    t_load = time.time()
    model = load_fn(a.device, a) if _accepts2(load_fn) else load_fn(a.device)
    print(f"[{name}] model loaded on {a.device} in {time.time()-t_load:.1f}s", flush=True)

    keyset = set(k for k in a.keys.split(",") if k)
    items = list(iter_anchors(a.good_only))
    if keyset:
        items = [it for it in items if it[0] in keyset]
    if a.limit:
        items = items[:a.limit]

    recs = []
    n_ok = 0
    t0 = time.time()
    for j, (key, fp, gt) in enumerate(items):
        img = read_gray(fp)
        try:
            mask, score = detect_fn(model, img, a.device)
        except Exception as e:
            if j < 3:
                print(f"[{name}] detect error on {key}: {type(e).__name__}: {e}", flush=True)
            mask, score = None, None
        det = center_from_maskres(mask)
        recs.append(make_record(key, gt, det, score))
        n_ok += int(det is not None)
        if a.save_masks and mask is not None:
            md = f"{RESULTS}/masks/{name}/{key}"
            os.makedirs(md, exist_ok=True)
            cv2.imwrite(f"{md}/{gt['anchor_idx']:06d}.png", resize_mask(mask) * 255)
        if (j + 1) % 50 == 0:
            print(f"[{name}] {j+1}/{len(items)}  valid={n_ok}  {(time.time()-t0)/(j+1):.2f}s/frame", flush=True)

    write_csv(out, recs)
    # summary
    errs = [r["err"] for r in recs if r["valid"] == 1]
    rr = [r["radius_ratio"] for r in recs if r["valid"] == 1 and r["radius_ratio"] != ""]
    m, md, p95, p99, sd, n = five(errs)
    print("=" * 64)
    print(f"[{name}] anchors={len(recs)} valid={n_ok} ({100*n_ok/max(len(recs),1):.1f}%)  "
          f"elapsed={time.time()-t0:.1f}s")
    print(f"[{name}] center err(px): mean={m:.3f} median={md:.3f} p95={p95:.3f} p99={p99:.3f} std={sd:.3f} n={n}")
    if rr:
        rr = np.asarray(rr, float)
        print(f"[{name}] radius_ratio: median={np.median(rr):.2f} p95={np.percentile(rr,95):.2f} "
              f"| iris-suspect(ratio>1.5): {int((rr>1.5).sum())}/{len(rr)}")
    print(f"[{name}] -> {out}")
    return out


def _accepts2(fn):
    import inspect
    try:
        return len(inspect.signature(fn).parameters) >= 2
    except (ValueError, TypeError):
        return False
