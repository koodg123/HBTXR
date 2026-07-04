"""Eye-closed / blink detection for EV-Eye near-eye grayscale frames.

A *Blink* frame = no clear dark-pupil blob (the eyelid occludes the pupil).
This is a DIFFERENT kind of label from the velocity-based motion phases
(Fixation/Saccade/Smooth): those describe how an *open* eye moves; Blink is an
*eye-closed / pupil-absent* state, so it is decided from pupil PRESENCE, not speed.

Design
------
- Mirrors the exact dark-pupil preprocessing of `pupil_detect.detect_center`
  (Gaussian blur -> darkest-3% threshold -> open/close -> contour score), but
  returns rich features so the blink decision is transparent and tunable.
- Calibration: EV-Eye GT-annotated frames are guaranteed eye-OPEN (a pupil
  ellipse was annotated), so their pupil-score distribution defines the
  "eye-open floor". `calibrate_tau()` sets the blink threshold below that floor.
- Decision: a frame is Blink if no qualifying pupil contour is found, OR the
  pupil score falls below the calibrated threshold `tau`. A short temporal
  hysteresis (`smooth_blink_runs`) removes 1-frame flicker and merges tiny gaps,
  because physiological blinks last ~100-400 ms (~3-10 frames at 25 Hz).

This detector is intentionally lightweight (cv2 only, no torch) so it runs on a
CPU/WSL box. For maximum fidelity you can instead drive the blink decision from
the EV-Eye per-user U-Net masks (see `pupil_features_from_mask`): an
empty/near-empty predicted pupil mask is a blink. Both paths feed the same
`is_blink` rule.
"""
import numpy as np
import cv2

# Preprocessing constants -- keep identical to pupil_detect.detect_center
DARK_PCT = 3            # pupil is among the darkest ~3% of pixels
MIN_AREA = 30           # ignore specks
MAX_AREA_FRAC = 0.25    # ignore blobs covering >25% of the image (eyelid shadow)


def pupil_features(img_gray):
    """Return dict of pupil-presence features for one grayscale frame.

    keys: has_pupil(bool), score(float), area(float), circ(float),
          dark_frac(float), cx, cy
    `score` == circularity * sqrt(area), the same quantity pupil_detect uses as
    its detection confidence (so it is directly comparable to the cached
    `det_conf` column in velocity_at_gt.csv).
    """
    if img_gray is None:
        return dict(has_pupil=False, score=0.0, area=0.0, circ=0.0,
                    dark_frac=0.0, cx=np.nan, cy=np.nan)
    g = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thr = np.percentile(g, DARK_PCT)
    _, m = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY_INV)
    dark_frac = float((m > 0).mean())
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score, best_area, best_circ = None, -1.0, 0.0, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_AREA or a > MAX_AREA_FRAC * img_gray.size:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = 4 * np.pi * a / (peri * peri)
        score = circ * np.sqrt(a)
        if score > best_score:
            best_score, best_area, best_circ, best = score, a, circ, c
    if best is None:
        return dict(has_pupil=False, score=0.0, area=0.0, circ=0.0,
                    dark_frac=dark_frac, cx=np.nan, cy=np.nan)
    M = cv2.moments(best)
    if M["m00"] == 0:
        return dict(has_pupil=False, score=0.0, area=best_area, circ=best_circ,
                    dark_frac=dark_frac, cx=np.nan, cy=np.nan)
    return dict(has_pupil=True, score=float(best_score), area=float(best_area),
                circ=float(best_circ), dark_frac=dark_frac,
                cx=M["m10"] / M["m00"], cy=M["m01"] / M["m00"])


def pupil_features_from_mask(mask, min_area=20):
    """Optional high-fidelity path: derive blink features from a predicted pupil
    mask (e.g. EV-Eye U-Net output). An empty/near-empty mask => blink.
    `mask` is a HxW array (bool or 0/255)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    area = float(m.sum())
    if area < min_area:
        return dict(has_pupil=False, score=0.0, area=area, circ=0.0,
                    dark_frac=0.0, cx=np.nan, cy=np.nan)
    ys, xs = np.nonzero(m)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = cv2.arcLength(max(cnts, key=cv2.contourArea), True) if cnts else 0.0
    circ = (4 * np.pi * area / (peri * peri)) if peri else 0.0
    return dict(has_pupil=True, score=float(circ * np.sqrt(area)), area=area,
                circ=float(circ), dark_frac=0.0, cx=float(xs.mean()), cy=float(ys.mean()))


def is_blink(feat, tau):
    """Per-frame blink decision from features and a calibrated score threshold."""
    return (not feat["has_pupil"]) or (feat["score"] < tau)


def calibrate_tau(open_scores, frac=0.5, floor=2.0):
    """Derive the blink score threshold from a sample of guaranteed eye-OPEN
    scores (e.g. the `det_conf` of GT frames). tau = frac * 1st-percentile of
    open scores, clamped to at least `floor`. Frames scoring below tau are far
    outside the eye-open distribution and treated as blink candidates.
    """
    open_scores = np.asarray([s for s in open_scores if s is not None and s > 0], float)
    if open_scores.size == 0:
        return floor
    return float(max(floor, frac * np.percentile(open_scores, 1)))


def smooth_blink_runs(blink_bool, min_len=2, max_gap=1):
    """Temporal hysteresis over a per-frame blink boolean array (frame order):
    drop isolated blink runs shorter than `min_len`, then fill open-eye gaps of
    length <= `max_gap` that sit between two blink runs (a single mis-detected
    open frame inside a blink). Returns a cleaned boolean array."""
    b = np.asarray(blink_bool, bool).copy()
    n = len(b)
    if n == 0:
        return b
    # fill short gaps between blink runs
    i = 0
    while i < n:
        if not b[i]:
            j = i
            while j < n and not b[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                b[i:j] = True
            i = j
        else:
            i += 1
    # drop too-short blink runs
    i = 0
    while i < n:
        if b[i]:
            j = i
            while j < n and b[j]:
                j += 1
            if (j - i) < min_len:
                b[i:j] = False
            i = j
        else:
            i += 1
    return b


if __name__ == "__main__":
    # tau calibration demo against the cached GT detector scores (eye-open floor)
    import os, sys, pandas as pd
    cache = sys.argv[1] if len(sys.argv) > 1 else "cache/velocity_at_gt.csv"
    if os.path.exists(cache):
        d = pd.read_csv(cache)
        tau = calibrate_tau(d["det_conf"].tolist())
        print(f"GT eye-open score: p1={d.det_conf.quantile(.01):.2f} "
              f"p5={d.det_conf.quantile(.05):.2f} p50={d.det_conf.median():.2f}")
        print(f"calibrated blink tau = {tau:.2f}  "
              f"(GT frames below tau: {(d.det_conf < tau).mean()*100:.2f}% -> "
              f"~0 expected, since GT frames are eye-open)")
    else:
        print("no cache found; pass path to velocity_at_gt.csv to calibrate tau")
