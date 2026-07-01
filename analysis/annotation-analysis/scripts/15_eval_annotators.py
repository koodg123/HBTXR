"""15_eval_annotators.py — aggregate the third/ annotator runners into ONE comparison.

Reads results/annotators/<tool>.csv (produced by scripts/annotators/run_*.py) plus the
existing U-Net / GSAM2 baselines (from samples/label/*/{unet_dense,gsam2}.json), and
compares each source's pupil-center accuracy vs the human GT anchor:
  center err (px, 346x260): mean / median / p95 / p99 / std, overall + per motion,
  valid rate, and radius_ratio iris-suspect count (center is blind to iris confusion).
Primary table = quality.good anchors. Writes results/annotators/summary.md.

  ../.venv-gsam2/bin/python 15_eval_annotators.py            # after running run_*.py
"""
import os, csv, glob, json, math
import numpy as np

AA = "/home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis"
LAB = f"{AA}/samples/label"
RES = f"{AA}/results/annotators"
MOTIONS = ["fixation", "saccade", "smooth_pursuit"]


def five(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return (float("nan"),) * 5 + (0,)
    return (v.mean(), np.median(v), np.percentile(v, 95), np.percentile(v, 99),
            v.std(ddof=1) if len(v) > 1 else 0.0, len(v))


_MOTION = {}
mp = f"{AA}/samples/manifest_windows.csv"
if os.path.exists(mp):
    for r in csv.DictReader(open(mp)):
        _MOTION[r["key"]] = r["motion"]


def motion_of(key):
    return _MOTION.get(key, "smooth_pursuit" if key.startswith("smooth_pursuit") else key.split("_", 1)[0])


def load_gt():
    gts = {}
    for gj in sorted(glob.glob(f"{LAB}/*/gt.json")):
        key = os.path.basename(os.path.dirname(gj))
        g = json.load(open(gj))
        cx, cy, rx, ry, _ = g["ellipse_cx_cy_rx_ry_theta"]
        gts[key] = dict(idx=g["anchor_idx"], cx=float(cx), cy=float(cy),
                        r=math.sqrt(max(rx * ry, 1e-6)),
                        good=bool(g.get("quality", {}).get("good", True)),
                        motion=motion_of(key))
    return gts


def rec(key, gt, cx, cy, r=None, valid=True):
    if not valid or cx is None:
        return dict(key=key, motion=gt["motion"], good=gt["good"], valid=0,
                    err=None, radius_ratio=None)
    err = math.hypot(cx - gt["cx"], cy - gt["cy"])
    rr = (r / gt["r"]) if (r and gt["r"] > 0) else None
    return dict(key=key, motion=gt["motion"], good=gt["good"], valid=1, err=err, radius_ratio=rr)


# ---- baselines from label jsons ----
def baseline(gts, kind):
    out = []
    for key, gt in gts.items():
        ai = gt["idx"]
        try:
            if kind == "unet":
                d = {c["idx"]: c for c in json.load(open(f"{LAB}/{key}/unet_dense.json"))["unet_centers"]}
                c = d.get(ai)
                ok = bool(c and c.get("valid"))
                out.append(rec(key, gt, c["cx"] if ok else None, c["cy"] if ok else None, valid=ok))
            elif kind == "gsam2":
                d = {c["idx"]: c for c in json.load(open(f"{LAB}/{key}/gsam2.json"))["gsam2_centers"]}
                c = d.get(ai)
                ok = bool(c and c.get("valid") and not c.get("mislabel", False))
                out.append(rec(key, gt, c["cx"] if ok else None, c["cy"] if ok else None, valid=ok))
        except (FileNotFoundError, KeyError, TypeError):
            out.append(rec(key, gt, None, None, valid=False))
    return out


def load_tool(path, gts):
    out = []
    seen = set()
    for r in csv.DictReader(open(path)):
        key = r["key"]
        seen.add(key)
        gt = gts.get(key)
        if gt is None:
            continue
        valid = r.get("valid") in ("1", 1, "True", True)
        err = float(r["err"]) if valid and r.get("err") not in ("", None) else None
        rr = float(r["radius_ratio"]) if r.get("radius_ratio") not in ("", None) else None
        out.append(dict(key=key, motion=gt["motion"], good=gt["good"],
                        valid=1 if valid and err is not None else 0, err=err, radius_ratio=rr))
    return out


def stat_block(records, good_only=True):
    rows = [r for r in records if (r["good"] or not good_only)]
    n_tot = len(rows)
    valid = [r for r in rows if r["valid"]]
    errs = [r["err"] for r in valid]
    rr = [r["radius_ratio"] for r in valid if r["radius_ratio"] is not None]
    m, md, p95, p99, sd, n = five(errs)
    iris = int(np.sum(np.asarray(rr) > 1.5)) if rr else 0
    per = {}
    for mo in MOTIONS:
        e = [r["err"] for r in valid if r["motion"] == mo]
        per[mo] = five(e)
    return dict(n_tot=n_tot, n_valid=len(valid), vr=100 * len(valid) / max(n_tot, 1),
                mean=m, median=md, p95=p95, p99=p99, std=sd, n=n,
                iris=iris, n_rr=len(rr), per=per)


def main():
    gts = load_gt()
    sources = {}
    sources["U-Net (EV-Eye)"] = baseline(gts, "unet")
    sources["GSAM2 (audit)"] = baseline(gts, "gsam2")
    for p in sorted(glob.glob(f"{RES}/*.csv")):
        name = os.path.splitext(os.path.basename(p))[0]
        if name == "summary":
            continue
        sources[name] = load_tool(p, gts)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    ngood = sum(1 for g in gts.values() if g["good"])
    out("# Annotator accuracy vs human GT (quality.good anchors)")
    out(f"\nGT anchors: {len(gts)} total, {ngood} good. Center err in px @346x260. "
        f"radius_ratio>1.5 = iris/over-seg suspect (center blind to it).\n")
    hdr = f"| {'source':22s} | valid% | mean | median | p95 | p99 | std | n | iris? |"
    out(hdr)
    out("|" + "-" * 24 + "|" + "|".join(["-" * 7] * 8) + "|")
    order = list(sources.keys())
    rows_stat = {k: stat_block(v) for k, v in sources.items()}
    # sort tools (non-baseline) by median asc, keep baselines first
    base = ["U-Net (EV-Eye)", "GSAM2 (audit)"]
    tools = sorted([k for k in order if k not in base],
                   key=lambda k: (math.isnan(rows_stat[k]["median"]), rows_stat[k]["median"]))
    for k in base + tools:
        s = rows_stat[k]
        out(f"| {k:22s} | {s['vr']:5.1f} | {s['mean']:.3f} | {s['median']:.3f} | "
            f"{s['p95']:.3f} | {s['p99']:.3f} | {s['std']:.3f} | {s['n']:4d} | "
            f"{s['iris']}/{s['n_rr']} |")

    out("\n## Per-motion median / p95 (px)\n")
    out(f"| {'source':22s} | " + " | ".join(f"{m[:8]} med/p95" for m in MOTIONS) + " |")
    out("|" + "-" * 24 + "|" + "|".join(["-" * 16] * 3) + "|")
    for k in base + tools:
        s = rows_stat[k]
        cells = []
        for mo in MOTIONS:
            f = s["per"][mo]
            cells.append(f"{f[1]:.2f}/{f[2]:.2f} (n{f[5]})")
        out(f"| {k:22s} | " + " | ".join(cells) + " |")

    out("\nNotes: mask->uniform ellipse-fit for ALL sources (fair center/radius). "
        "GSAM2 baseline excludes mislabel-flagged. Tools trained off-domain (EllSeg/RITnet/"
        "Edge-Guided/DeepVOG on other IR sets; YOLOE/SAM3 RGB) — cross-dataset audit, not tuned.")

    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n-> {RES}/summary.md")


if __name__ == "__main__":
    main()
