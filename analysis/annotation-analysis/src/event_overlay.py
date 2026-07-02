"""event_overlay.py — visualize the cropped Event stream as an overlay.
3 panels per window: APS crop | Event frame (red=+pol, blue=-pol) | APS+Event overlay,
with the GT pupil center (green). Confirms events are marker-free and eye-aligned.
Run in .venv-gsam2. -> datasets/eye_crop_240x160/_qc_event/*.png
"""
import os, sys, glob, json
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S

DS = f"{S.AA}/datasets/eye_crop_240x160"


def render(key, mode="window", tbin_us=40000, scale=2):
    kd = f"{DS}/{key}"
    lab = json.load(open(f"{kd}/labels.json"))
    ai = str(lab["anchor_idx"])
    ats = int(lab["frames"][ai]["ts"])
    gt = lab["frames"][ai].get("gt")
    ap = glob.glob(f"{kd}/aps/{int(ai):06d}_*.png")[0]
    aps = cv2.imread(ap, 0)
    H, W = aps.shape
    apsb = cv2.cvtColor(aps, cv2.COLOR_GRAY2BGR)

    d = np.load(f"{kd}/events.npz")
    t, x, y, p = d["t"], d["x"], d["y"], d["p"]
    m = np.ones(len(t), bool) if mode == "window" else (np.abs(t - ats) <= tbin_us)
    xe, ye, pe = x[m].astype(int), y[m].astype(int), p[m].astype(int)
    pos = pe == 1

    ev = np.full((H, W, 3), 30, np.uint8)                       # dark bg
    ev[ye[pos], xe[pos]] = (60, 60, 255)                        # + polarity: red
    ev[ye[~pos], xe[~pos]] = (255, 120, 60)                     # - polarity: blue
    ovl = apsb.copy()
    ovl[ye[pos], xe[pos]] = (60, 60, 255)
    ovl[ye[~pos], xe[~pos]] = (255, 120, 60)

    if gt:
        gx, gy = int(round(gt[0])), int(round(gt[1]))
        for img in (apsb, ev, ovl):
            cv2.circle(img, (gx, gy), 3, (0, 255, 0), -1)

    def lbl(img, txt):
        cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    lbl(apsb, "APS")
    lbl(ev, f"EVENT n={len(xe)}")
    lbl(ovl, "APS+EVENT")
    comp = np.hstack([apsb, np.full((H, 2, 3), 200, np.uint8), ev,
                      np.full((H, 2, 3), 200, np.uint8), ovl])
    return cv2.resize(comp, (comp.shape[1] * scale, comp.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def main():
    out = f"{DS}/_qc_event"
    os.makedirs(out, exist_ok=True)
    keys = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{DS}/*/labels.json"))
    picked = []
    for mo in ("fixation", "saccade", "smooth_pursuit"):
        k = next((x for x in keys if x.startswith(mo)), None)
        if k:
            picked.append(k)
    for k in picked:
        cv2.imwrite(f"{out}/{k}.png", render(k))
        print(f"[event_overlay] {k}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
