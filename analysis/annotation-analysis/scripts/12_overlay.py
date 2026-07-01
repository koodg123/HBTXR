"""
12_overlay.py — render QA overlays from the per-frame tree (samples/perframe).

For selected anchor frames it draws, on the APS frame (upscaled xS):
  - GT (human)  : green ellipse + center + bbox        (anchor only)
  - U-Net       : blue  mask contour + center + bbox
  - GSAM2       : red   mask contour + center + bbox (xyxy=GDINO box)
with a caption listing per-source centers and GSAM2/U-Net error vs GT.

Selection: per motion, the N lowest-error anchors ("good") + the K highest-error
anchors ("mis-detect"), so both clean cases and GSAM2 failures are visible.

Output PNGs -> --overlay-out (default: <samples>/../overlay).
Pure post-processing (cv2 only, no GPU).
"""
import os, csv, json, argparse, math
import numpy as np
import cv2

GREEN, BLUE, RED, MAGENTA, WHITE = (0, 200, 0), (255, 160, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255)


def load(p):
    return json.load(open(p)) if os.path.isfile(p) else None


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) if a and b else float("nan")


def draw_contour(img, mask_path, color, S):
    if not os.path.isfile(mask_path):
        return
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return
    m = cv2.resize(m, (m.shape[1] * S, m.shape[0] * S), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours((m > 127).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, 2)


def draw_bbox(img, bbox_path, color, S, key="xyxy"):
    b = load(bbox_path)
    if not b or key not in b:
        return
    x0, y0, x1, y1 = [v * S for v in b[key]]
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 1)


def draw_center(img, c, color, S):
    if not c:
        return
    p = (int(c[0] * S), int(c[1] * S))
    cv2.circle(img, p, 5, WHITE, -1)
    cv2.circle(img, p, 4, color, -1)


def render(fdir, S):
    fp = os.path.join(fdir, "frame.png")
    base = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if base is None:
        return None, None
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (img.shape[1] * S, img.shape[0] * S), interpolation=cv2.INTER_NEAREST)

    gt = load(os.path.join(fdir, "gt.json"))
    gtc = gt["ellipse_cx_cy_rx_ry_theta"][:2] if gt else None
    u = load(os.path.join(fdir, "unet", "center.json"))
    uc = [u["cx"], u["cy"]] if u else None
    g = load(os.path.join(fdir, "gsam2", "center.json"))
    gc = [g["cx"], g["cy"]] if g else None
    p = load(os.path.join(fdir, "pred", "center.json"))
    pc = [p["cx"], p["cy"]] if p else None

    draw_contour(img, os.path.join(fdir, "unet", "mask.png"), BLUE, S)
    draw_contour(img, os.path.join(fdir, "gsam2", "mask.png"), RED, S)
    draw_bbox(img, os.path.join(fdir, "unet", "bbox.json"), BLUE, S)
    draw_bbox(img, os.path.join(fdir, "gsam2", "bbox.json"), RED, S)
    draw_bbox(img, os.path.join(fdir, "gt_bbox.json"), GREEN, S)
    if gt:  # GT ellipse
        cx, cy, rx, ry, th = gt["ellipse_cx_cy_rx_ry_theta"]
        cv2.ellipse(img, (int(cx * S), int(cy * S)), (int(rx * S), int(ry * S)),
                    math.degrees(th), 0, 360, GREEN, 2)
    draw_center(img, uc, BLUE, S)
    draw_center(img, gc, RED, S)
    draw_center(img, pc, MAGENTA, S)
    draw_center(img, gtc, GREEN, S)

    cap = [f"GT(green) uNet(blue) GSAM2(red) pred(magenta)  {os.path.basename(fdir)}"]
    if gtc:
        cap.append(f"err vs GT: uNet={dist(uc,gtc):.2f}  GSAM2={dist(gc,gtc):.2f}  pred={dist(pc,gtc):.2f} px")
    bar = np.zeros((22 * len(cap) + 8, img.shape[1], 3), np.uint8)
    for i, t in enumerate(cap):
        cv2.putText(bar, t, (6, 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    out = np.vstack([bar, img])
    return out, dist(gc, gtc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perframe", default="../samples/perframe")
    ap.add_argument("--overlay-out", default="../overlay")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--n-good", type=int, default=3, help="lowest-err anchors per motion")
    ap.add_argument("--n-bad", type=int, default=6, help="highest-err anchors overall (mis-detect)")
    a = ap.parse_args()
    os.makedirs(a.overlay_out, exist_ok=True)

    idx = list(csv.DictReader(open(os.path.join(a.perframe, "index.csv"))))
    anchors = [r for r in idx if r["role"] == "anchor" and r["has_gt"] == "1"
               and r["has_gsam2"] == "1" and r["has_unet"] == "1"]
    # compute gsam2-vs-gt err for ranking
    scored = []
    for r in anchors:
        fdir = os.path.join(a.perframe, r["key"], r["stem"])
        gt = load(os.path.join(fdir, "gt.json"))
        g = load(os.path.join(fdir, "gsam2", "center.json"))
        if not gt or not g:
            continue
        e = dist([g["cx"], g["cy"]], gt["ellipse_cx_cy_rx_ry_theta"][:2])
        scored.append((e, r, fdir))

    pick = {}
    for m in ["fixation", "saccade", "smooth_pursuit"]:
        good = sorted([s for s in scored if s[1]["motion"] == m], key=lambda s: s[0])[: a.n_good]
        for e, r, fdir in good:
            pick[fdir] = ("good", e, r)
    for e, r, fdir in sorted(scored, key=lambda s: -s[0])[: a.n_bad]:
        pick[fdir] = ("bad", e, r)

    print(f"[i] rendering {len(pick)} overlays -> {a.overlay_out}")
    n = 0
    for fdir, (tag, e, r) in sorted(pick.items(), key=lambda kv: kv[1][0]):
        img, _ = render(fdir, a.scale)
        if img is None:
            continue
        name = f"{tag}_{r['motion']}_{e:06.2f}px_{r['key']}__{r['stem']}.png"
        cv2.imwrite(os.path.join(a.overlay_out, name), img)
        n += 1
    print(f"[done] {n} overlays written to {a.overlay_out}/")
    print("  naming: {good|bad}_{motion}_{gsam2_err}px_{key}__{stem}.png")


if __name__ == "__main__":
    main()
