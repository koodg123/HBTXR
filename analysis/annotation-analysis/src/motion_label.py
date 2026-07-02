"""motion_label.py — per-frame motion class {fixation, saccade, smooth_pursuit, blink}.

Hybrid (data-driven, see docs/16 analysis):
- BLINK  : GSAM2 invalid | mislabel(eyelid-slit over-seg / tiny) | area < BLINK_FRAC * session-median.
- SMOOTH : session in {2_0_1, 2_0_2} (task-defined; smooth's frame velocity ~= fixation so NOT
           velocity-separable). Optional catch-up SACCADE sub-split when v >= VSAC.
- 1_0_2  : I-VT on per-frame center velocity (px/frame, U-Net dense) -> v>=VSAC SACCADE else FIXATION.
- 1_0_1  : task UNVERIFIED (no human GT); default I-VT like 1_0_2 (flag conf=low).
Motion is a DERIVED label (no human motion GT; EV-Eye Tobii/stimulus unused).

codes: 0 fixation, 1 saccade, 2 smooth_pursuit, 3 blink.  Run standalone to validate on audit set.
"""
import os, sys, glob, json, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S

FIX, SAC, SMO, BLK = 0, 1, 2, 3
NAME = {0: "fixation", 1: "saccade", 2: "smooth", 3: "blink"}
VSAC = 6.0          # px/frame saccade threshold (I-VT; matches 07)
BLINK_FRAC = 0.35   # area < frac*session-median -> collapse/blink
SMOOTH_SESSIONS = {"2_0_1", "2_0_2"}


def label_window(key):
    """-> {idx: dict(motion, velocity, blink, source)}."""
    p = S.parse_key(key)
    sess = p["session"]
    smooth_task = sess in SMOOTH_SESSIONS
    def _load(path, field):
        try:
            return {c["idx"]: c for c in json.load(open(path))[field]}
        except (FileNotFoundError, KeyError):
            return {}
    u = _load(f"{S.LAB}/{key}/unet_dense.json", "unet_centers")   # empty for 33-48 (no U-Net predict)
    g = _load(f"{S.LAB}/{key}/gsam2.json", "gsam2_centers")
    idxs = sorted(set(u) | set(g))
    areas = [g[i]["area"] for i in idxs if g.get(i) and g[i].get("valid") and not g[i].get("mislabel") and g[i].get("area")]
    med_area = float(np.median(areas)) if areas else 0.0

    out, prev = {}, None
    for i in idxs:
        uc, gc = u.get(i), g.get(i)
        # velocity from GSAM2 (dense, primary label source -> works for all subjects incl. no-UNet 33-48;
        # GSAM2 F2F jitter ~0.21px in fixation so I-VT is clean), U-Net fallback only where GSAM2 invalid.
        v = None
        vc = None
        if gc and gc.get("valid") and not gc.get("mislabel"):
            vc = (gc["cx"], gc["cy"])
        elif uc and uc.get("valid"):
            vc = (uc["cx"], uc["cy"])
        if vc is not None:
            if prev and i - prev[0] == 1:
                v = ((vc[0] - prev[1]) ** 2 + (vc[1] - prev[2]) ** 2) ** 0.5
            prev = (i, vc[0], vc[1])
        else:
            prev = None
        # blink
        g_invalid = not (gc and gc.get("valid"))
        g_mis = bool(gc and gc.get("mislabel", False))
        g_small = bool(gc and gc.get("valid") and gc.get("area") and med_area and gc["area"] < BLINK_FRAC * med_area)
        blink = g_invalid or g_mis or g_small
        # class
        if blink:
            m = BLK
        elif smooth_task:
            m = SAC if (v is not None and v >= VSAC) else SMO
        else:                                   # 1_0_2 (and 1_0_1 default)
            m = SAC if (v is not None and v >= VSAC) else FIX
        out[i] = dict(motion=m, velocity=(round(v, 3) if v is not None else None),
                      blink=blink, source=("gsam2" if not blink else "blink"))
    return out


def main():
    motion = {r["key"]: r["motion"] for r in csv.DictReader(open(f"{S.SAMPLES}/manifest_windows.csv"))}
    from collections import Counter, defaultdict
    per_win_frames = defaultdict(Counter)   # window-motion -> per-frame class counts
    tot = Counter()
    for gj in sorted(glob.glob(f"{S.LAB}/*/gsam2.json")):
        key = os.path.basename(os.path.dirname(gj))
        wm = motion.get(key, "?")
        for i, r in label_window(key).items():
            per_win_frames[wm][r["motion"]] += 1
            tot[r["motion"]] += 1
    print("=== per-frame motion counts by WINDOW motion (검증: 대각선 우세 기대) ===")
    print(f"  {'window\\frame':16s} " + " ".join(f"{NAME[c]:>9s}" for c in (0, 1, 2, 3)))
    for wm in ("fixation", "saccade", "smooth_pursuit"):
        c = per_win_frames[wm]
        n = sum(c.values())
        print(f"  {wm:16s} " + " ".join(f"{100*c[k]/n:8.1f}%" for k in (0, 1, 2, 3)) + f"   (n{n})")
    print(f"\n=== 전체 per-frame 분포 ===  " + " ".join(f"{NAME[k]}:{tot[k]}({100*tot[k]/sum(tot.values()):.1f}%)" for k in (0, 1, 2, 3)))


if __name__ == "__main__":
    main()
