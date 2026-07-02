"""render_all_masks.py — GSAM2 mask overlay for ALL frames + events, per window.

Each window -> one PNG: (top) montage grid of every frame (crop APS + GSAM2 mask cyan +
pupil ellipse + motion-colored border), (bottom) event overlay (accumulated events, pol-colored,
+ anchor GSAM2 mask contour). Motion border: fixation=green, saccade=red, smooth=blue, blink=yellow;
mislabel=magenta X. -> datasets/eye_crop_240x160/_qc_masks/{key}.png
"""
import os, sys, glob, json, math, argparse
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S
import motion_label as ML

FRAME = f"{S.AA}/samples/frame"
EVENT = f"{S.AA}/samples/event"
BOX = (53, 28, 293, 188)
MCOL = {0: (0, 200, 0), 1: (0, 0, 230), 2: (230, 120, 0), 3: (0, 220, 220)}  # fix/sac/smooth/blink (BGR)


def read_gray(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return im


def panel(imc, mask, cx, cy, mo, mis):
    b = cv2.cvtColor(imc, cv2.COLOR_GRAY2BGR)
    if mask is not None and mask.any():
        b[mask] = (b[mask] * 0.45 + np.array([200, 200, 0]) * 0.55).astype(np.uint8)  # cyan mask
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(b, cnts, -1, (60, 60, 255), 1)                                # red contour
    if cx is not None:
        cv2.drawMarker(b, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
    b = cv2.copyMakeBorder(b, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=MCOL.get(mo, (128, 128, 128)))
    if mis:
        cv2.drawMarker(b, (14, 14), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    return b


def render(key):
    x0, y0, x1, y1 = BOX; W, H = x1 - x0, y1 - y0
    gt = json.load(open(f"{S.LAB}/{key}/gt.json")); ai = gt["anchor_idx"]
    gs = {c["idx"]: c for c in json.load(open(f"{S.LAB}/{key}/gsam2.json"))["gsam2_centers"]}
    mot = ML.label_window(key)
    frames = sorted(glob.glob(f"{FRAME}/{key}/*.png"))
    panels = []
    for fp in frames:
        idx = int(os.path.basename(fp)[:-4].split("_")[0])
        imc = read_gray(fp)[y0:y1, x0:x1]
        mp = glob.glob(f"{S.LAB}/{key}/gsam2_masks/{idx:06d}_*_mask.png")
        mask = (read_gray(mp[0])[y0:y1, x0:x1] > 0) if mp else None
        gc = gs.get(idx)
        cx = (gc["cx"] - x0) if (gc and gc.get("valid")) else None
        cy = (gc["cy"] - y0) if (gc and gc.get("valid")) else None
        mr = mot.get(idx, {})
        panels.append(panel(imc, mask, cx, cy, mr.get("motion", 0), gc.get("mislabel", False) if gc else False))
    # montage grid
    ncol = min(len(panels), 8); ph, pw = panels[0].shape[:2]
    rows = []
    for r in range(0, len(panels), ncol):
        row = panels[r:r + ncol]
        while len(row) < ncol:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    # event panel (accumulated + anchor mask contour)
    ev = np.load(f"{EVENT}/{key}.npz"); ex, ey, ep = ev["x"], ev["y"], ev["p"]
    m = (ex >= x0) & (ex < x1) & (ey >= y0) & (ey < y1)
    evimg = np.full((H, W, 3), 30, np.uint8)
    xe, ye, pe = (ex[m] - x0).astype(int), (ey[m] - y0).astype(int), ep[m]
    evimg[ye[pe == 1], xe[pe == 1]] = (60, 60, 255); evimg[ye[pe == 0], xe[pe == 0]] = (255, 120, 60)
    amp = glob.glob(f"{S.LAB}/{key}/gsam2_masks/{ai:06d}_*_mask.png")
    if amp:
        am = (read_gray(amp[0])[y0:y1, x0:x1] > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(am, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(evimg, cnts, -1, (0, 255, 0), 1)
    evpanel = cv2.copyMakeBorder(evimg, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(80, 80, 80))
    evpanel = cv2.resize(evpanel, (grid.shape[1], int(evpanel.shape[0] * grid.shape[1] / evpanel.shape[1])),
                         interpolation=cv2.INTER_NEAREST)
    out = np.vstack([grid, evpanel])
    title = np.zeros((22, out.shape[1], 3), np.uint8)
    cv2.putText(title, f"{key}  frames={len(panels)}  (fix=grn sac=red smooth=blu blink=yel, X=mislabel; bottom=EVENT+mask)",
                (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([title, out])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=f"{S.AA}/datasets/eye_crop_240x160/_qc_masks")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keys = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{S.LAB}/*/gt.json"))
    if a.limit:
        keys = keys[:a.limit]
    for i, key in enumerate(keys):
        cv2.imwrite(f"{a.out}/{key}.jpg", render(key), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(keys)}", flush=True)
    print(f"[render_all_masks] {len(keys)} windows -> {a.out}")


if __name__ == "__main__":
    main()
