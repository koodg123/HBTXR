"""io_schema.py (E0) — shared loaders, constants, and schema report for the
Annotation Precision / Label Noise pipeline (HBTXR reviewer response).

Decisions baked in (user-confirmed):
- Samples: existing users 1-10 (NB: these are HBTXR TRAIN subjects; split is
  train 1-32 / val 33-36 / test 37-48 -> STEP5 E_orig is optimistic, caveated).
- Frame: everything in native 346x260 px, Euclidean distance. 64x64<->346x260 is
  ANISOTROPIC (x*5.406, y*4.0625) -> never a single-factor rescale.
- y_pred used ONLY in corrected_error (STEP5); never in precision/uncertainty.
- Mask (Data_davis_labelled_with_mask) is DERIVED from the ellipse via cv2.ellipse
  (verified: axis-corrected IoU~0.95 + constant (-1,-1) px offset) -> La.3 = a
  RASTERIZATION floor, and human_mask is NOT an independent source.
"""
import os, re, glob, json, csv, math
import numpy as np

AA = "/home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis"
LAB = f"{AA}/samples/label"
RES = f"{AA}/results"
OUT = f"{AA}/results/precision"           # scalars json + summaries
FIG = f"{AA}/fig"
TAB = f"{AA}/tables"
RAW = "/mnt/e/DATASET/eveye/raw_data"
MASKROOT = f"{RAW}/Data_davis_labelled_with_mask"
ANNOT_CSV = f"{RES}/annotators"
for d in (OUT, FIG, TAB):
    os.makedirs(d, exist_ok=True)

W, H = 346, 260                            # native frame (x=346, y=260)
SX, SY = 346.0 / 64.0, 260.0 / 64.0        # 5.406, 4.0625  (64x64 -> 346x260)
GEO = math.sqrt(SX * SY)                    # 4.686 isotropic-approx scale
REPORTED_0181_64 = 0.1812                   # reported precision, in 64x64 vs U-Net dense
MOTIONS = ["fixation", "saccade", "smooth_pursuit"]
ANNOT = ["ellseg", "ritnet", "edge_guided", "deepvog", "yoloe"]
TRAIN_SUBJ, VAL_SUBJ, TEST_SUBJ = set(range(1, 33)), set(range(33, 37)), set(range(37, 49))

KEY_RE = re.compile(r"^(fixation|saccade|smooth_pursuit)_user(\d+)_(left|right)_session_([0-9_]+?)_w\d+_a\d+$")


def parse_key(key):
    m = KEY_RE.match(key)
    if not m:
        return None
    return dict(motion=m.group(1), user=int(m.group(2)), eye=m.group(3), session=m.group(4))


def five(v):
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return dict(mean=float("nan"), median=float("nan"), p95=float("nan"),
                    p99=float("nan"), min=float("nan"), max=float("nan"), std=float("nan"), n=0)
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                min=float(v.min()), max=float(v.max()),
                std=float(v.std(ddof=1) if len(v) > 1 else 0.0), n=int(len(v)))


# ---------- per-anchor label sources (all 346x260) ----------
def _centers(path, field):
    try:
        return {c["idx"]: c for c in json.load(open(path))[field]}
    except (FileNotFoundError, KeyError):
        return {}


def load_anchor(key):
    """Return dict of sources at the anchor (None if missing/invalid). y_pred kept
    separate for STEP5 only. gsam2 excludes mislabel."""
    gt = json.load(open(f"{LAB}/{key}/gt.json"))
    ai = gt["anchor_idx"]
    cx, cy, rx, ry, th = gt["ellipse_cx_cy_rx_ry_theta"]
    p = parse_key(key)
    u = _centers(f"{LAB}/{key}/unet_dense.json", "unet_centers").get(ai)
    g = _centers(f"{LAB}/{key}/gsam2.json", "gsam2_centers").get(ai)
    pr = _centers(f"{LAB}/{key}/pred.json", "pred_centers").get(ai)
    rep = g.get("repeats") if (g and g.get("valid")) else None
    return dict(
        key=key, subject=p["user"], eye=p["eye"], session=p["session"], motion=p["motion"],
        anchor_idx=ai, human=(float(cx), float(cy)),
        rx=float(rx), ry=float(ry), theta=float(th), r_equiv=math.sqrt(max(rx * ry, 1e-6)),
        unet=((u["cx"], u["cy"]) if (u and u.get("valid")) else None),
        unet_area=(u.get("area") if u else None),
        gsam2=((g["cx"], g["cy"]) if (g and g.get("valid") and not g.get("mislabel", False)) else None),
        gsam2_area=(g.get("area") if g else None),
        gsam2_valid=bool(g and g.get("valid")),
        repeats=([tuple(p2) for p2 in rep] if rep else None),
        pred=((pr["cx"], pr["cy"]) if (pr and pr.get("valid") and "cx" in pr) else None),
        pred_present=bool(pr is not None))


def anchors():
    for gj in sorted(glob.glob(f"{LAB}/*/gt.json")):
        key = os.path.basename(os.path.dirname(gj))
        if parse_key(key):
            yield load_anchor(key)


def load_annotator_centers():
    """{tool: {key:(cx,cy)}} in 346x260 from results/annotators/<tool>.csv."""
    out = {a: {} for a in ANNOT}
    for a in ANNOT:
        p = f"{ANNOT_CSV}/{a}.csv"
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if str(r.get("valid")) == "1" and r.get("cx") not in ("", None):
                out[a][r["key"]] = (float(r["cx"]), float(r["cy"]))
    return out


# ---------- dense per-frame centers (for La.2 F2F jitter) ----------
def dense_centers(key, source):
    """Ordered [(idx,cx,cy)] over all frames of a window for source in
    {unet,gsam2,pred}; gsam2 skips mislabel/invalid."""
    if source == "unet":
        d = _centers(f"{LAB}/{key}/unet_dense.json", "unet_centers")
        ok = lambda c: c.get("valid")
    elif source == "gsam2":
        d = _centers(f"{LAB}/{key}/gsam2.json", "gsam2_centers")
        ok = lambda c: c.get("valid") and not c.get("mislabel", False)
    else:
        d = _centers(f"{LAB}/{key}/pred.json", "pred_centers")
        ok = lambda c: c.get("valid") and "cx" in c
    return [(i, d[i]["cx"], d[i]["cy"]) for i in sorted(d) if ok(d[i])]


# ---------- mask h5 <-> ellipse (La.3 rasterization floor) ----------
def _via_ellipses(csv_path):
    out = []
    for r in csv.DictReader(open(csv_path)):
        s = r.get("region_shape_attributes", "")
        if "ellipse" in s:
            d = json.loads(s)
            out.append((d["cx"], d["cy"], d.get("rx"), d.get("ry"), d.get("theta", 0)))
    return out


def mask_vs_ellipse(user, eye, session):
    """Matched (per annotated frame) mask-centroid vs ellipse-center distance (px,
    346x260) + rasterized-ellipse<->mask IoU. Returns list of (dist, iou). Mask h5 is
    stored x-major (346,260); transpose to (H,W)=(260,346) image convention."""
    import h5py, cv2
    h5p = f"{MASKROOT}/{eye}/user{user}_session_{session}.h5"
    csvp = f"{RAW}/Data_davis/user{user}/{eye}/session_{session}/user_{user}.csv"
    if not (os.path.exists(h5p) and os.path.exists(csvp)):
        return []
    lab = h5py.File(h5p, "r")["label"][:]                      # (346,260,N)
    ell = _via_ellipses(csvp)
    n = min(len(ell), lab.shape[2])
    out = []
    for k in range(n):
        cx, cy, rx, ry, th = ell[k]
        if rx is None:
            continue
        m = (lab[:, :, k].T > 0).astype(np.uint8)             # -> (260,346) image conv
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        dist = math.hypot(xs.mean() - cx, ys.mean() - cy)     # mask centroid vs ellipse center
        ras = np.zeros((H, W), np.uint8)
        cv2.ellipse(ras, (int(round(cx)), int(round(cy))),
                    (max(int(round(rx)), 1), max(int(round(ry)), 1)), math.degrees(th), 0, 360, 1, -1)
        inter = (m & ras).sum(); uni = (m | ras).sum()
        out.append((dist, float(inter / uni) if uni else 0.0))
    return out


def to_64(cx, cy):
    """346x260 -> 64x64 (anisotropic)."""
    return cx / SX, cy / SY


def write_e0_report():
    lines = [
        "# E0 — Schema discovery report (data + code cross-verified)\n",
        "## Assets / label types",
        "| asset | format / frame | nature | evidence |",
        "|---|---|---|---|",
        "| Data_davis (ellipse) | VIA CSV region_shape_attributes(cx,cy,rx,ry,theta), 346x260 | **PRIMARY human annotation** | evaluate_hbtxr_val_motion.py:379 |",
        "| labelled_with_mask | HDF5 data/label (346x260xN, binary) | **DERIVED from ellipse (cv2.ellipse rasterize)** | ev_eye_dataset_utils.py:103; data: axis-corrected IoU~0.95, constant (-1,-1)px offset |",
        "| Data_davis_predict (U-Net) | mask gif | trained on DERIVED masks -> NOT independent of human | EV-Eye/train.py:92 |",
        "| gsam2.repeats | 5 pts/anchor, 346x260 | box-jitter/TTA repeatability | present |",
        "",
        "## Key decisions",
        "- **Frame = 346x260** (anisotropic to 64x64: x*%.3f, y*%.3f). Reported **0.1812 = 64x64 vs U-Net dense**; converted only for the budget figure." % (SX, SY),
        "- **Samples = users 1-10 = HBTXR TRAIN subjects** (test=37-48) -> STEP5 E_orig is optimistic (caveated), per user decision.",
        "- **La.3 = rasterization floor** (mask derived); human_mask NOT used as an independent source.",
        "- **La.2-human SKIP** (human labels sparse) -> improved GSAM2 as repeatability proxy.",
        "- **3CH {human, gsam2, unet}** (independence risk accepted); U-Net~derived-human so expect possible SHARED-BIAS -> bracket fallback. Optional independent triple {human, gsam2, ellseg}.",
        "- y_pred used ONLY in STEP5.",
    ]
    os.makedirs(RES, exist_ok=True)
    open(f"{RES}/e0_schema_report.md", "w").write("\n".join(lines) + "\n")
    return f"{RES}/e0_schema_report.md"


if __name__ == "__main__":
    n = sum(1 for _ in anchors())
    ac = load_annotator_centers()
    print(f"[io_schema] anchors={n}  annotator sources={{" +
          ", ".join(f'{a}:{len(ac[a])}' for a in ANNOT) + "}")
    print(f"[io_schema] wrote {write_e0_report()}")
    # quick mask matcher sanity
    mv = mask_vs_ellipse(10, "left", "1_0_2")
    if mv:
        import numpy as _np
        dd = _np.array([d for d, _ in mv]); ii = _np.array([i for _, i in mv])
        print(f"[io_schema] mask_vs_ellipse user10/left/1_0_2: n={len(mv)} dist med={_np.median(dd):.2f} IoU med={_np.median(ii):.3f}")
