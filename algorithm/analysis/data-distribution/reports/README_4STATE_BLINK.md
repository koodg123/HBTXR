# EV-Eye 4-state classification with **Blink** — run guide (WSL/GPU)

Standard eye-movement taxonomy: **Fixation / Saccade / Smooth pursuit / Blink**.
This replaces the velocity-only "Reset" 4th class with the physiologically
standard **Blink** state.

Two new files (drop both into `codes/motion_type/src/`, next to `common.py`):

| file | role |
|---|---|
| `blink_detect.py` | per-frame eye-closed/blink detector (cv2; optional U-Net-mask path) |
| `build_4state_blink.py` | full-stream pipeline → per-frame labels + per-subject table + figure |

---

## Why this must run on the raw stream (and on WSL)

Blink is **not** a velocity phase — it is an *eye-closed / pupil-absent* state,
so it is decided from pupil **presence**, not speed. The EV-Eye GT annotations
(9,011 frames) only label frames where a pupil is visible, so **blink frames are
absent from GT by construction**. They exist only in the full near-eye frame
stream (~1.49M frames under `raw_data/Data_davis`), which is not on the cloud
sandbox. Hence: run on the WSL/GPU box with `EVEYE_ROOT` pointing at the dataset.

## The 4 states (per frame)

1. **Blink** — no pupil present (eyelid occludes it).
2. **Saccade** — eye open and `speed > v_sacc` (standard I-VT, any session). The
   former "Reset" (high-speed frames in smooth sessions = return/catch-up
   saccades) correctly falls here.
3. **Fixation** — eye open, `speed ≤ v_sacc`, saccade-session (101/201).
4. **Smooth** — eye open, `speed ≤ v_sacc`, smooth-session (102/202).

## How Blink is detected (and calibrated)

`blink_detect.pupil_features()` mirrors the existing `pupil_detect` dark-pupil
pipeline (blur → darkest-3% threshold → open/close → best circular contour) and
returns a `score = circularity·√area`, directly comparable to the cached
`det_conf`. A frame is **Blink** if no qualifying pupil contour is found **or**
`score < tau`. A short temporal hysteresis (`smooth_blink_runs`) drops 1-frame
flicker and fills 1-frame gaps (real blinks last ~3–10 frames at 25 Hz).

`tau` is auto-calibrated from the GT (guaranteed eye-open) score distribution:
`tau = 0.5 × p1(eye-open scores)`. On this dataset that is **tau ≈ 4.10**, and
**0.00 %** of eye-open GT frames fall below it (verified) — i.e. the threshold
does not mislabel open eyes as blinks. Override with `--blink-tau` if needed.

`v_sacc` defaults to the **same 493 px/s** used by the validated 3-state pipeline
(90th pct of saccade-session speed). Override with `--v-sacc`.

## Run

```bash
cd codes/motion_type/src
export EVEYE_ROOT=/mnt/e/DATASET/eveye        # parent of raw_data/ and processed_data/

# smoke test (1 user, cap frames) — confirms paths/threshold before the big run
EVEYE_ROOT=$EVEYE_ROOT python3 build_4state_blink.py --users 1 --max-frames 2000

# full run, all 48 users, every frame (resumable; checkpoints every 20 sessions)
EVEYE_ROOT=$EVEYE_ROOT python3 build_4state_blink.py

# faster first pass (every 2nd frame). Keep stride ≤ 2 so brief blinks survive.
EVEYE_ROOT=$EVEYE_ROOT python3 build_4state_blink.py --stride 2 --time-budget 7200
```

Dependencies: `numpy pandas opencv-python` (+ `matplotlib` for the figure). No
torch/GPU needed for the cv2 path. The run is **resumable** — re-running skips
sessions already in the per-frame cache.

### Optional high-fidelity blink source (U-Net masks)
If you prefer EV-Eye's own pupil masks over the cv2 detector, feed predicted
masks to `blink_detect.pupil_features_from_mask(mask)` (empty mask ⇒ blink) and
swap it in inside `classify_session`. Needs the per-user U-Net
(`processed_data/Pre-trained_models`) + torch/GPU.

## Outputs (written under `codes/motion_type/`)

| path | contents |
|---|---|
| `cache/frames_4state_blink.csv` | per-frame: user, eye, code, frame_idx, ts, cx, cy, score, speed_pxps, **state** |
| `reports/tbl_4state_blink_per_subject.csv` | Subject × {Fixation, Saccade, Smooth, Blink} (+ Total, + All row) |
| `reports/fig_4state_blink_per_subject.png` | per-subject stacked share figure |

## Validation already done (in the sandbox, on the GT cache)

- **Syntax**: both files compile.
- **Blink threshold**: `tau = 4.10`; 0.00 % of eye-open GT frames fall below it.
- **Temporal hysteresis**: `[0,1,0,0,1,1,1,0,1,0] → [0,0,0,0,1,1,1,1,1,0]` (drops
  isolated blinks, fills 1-frame gaps).
- **I-VT assignment**: the exact eye-open rule used here reproduces the validated
  3-state counts on the GT frames — **Fixation 3868 / Saccade 813 / Smooth 4301**.
  So Fixation/Saccade/Smooth are unchanged; the pipeline only **adds** Blink from
  the full stream.

The blink-detection half cannot be number-validated here (no raw frames in the
sandbox); run the smoke test on WSL and eyeball a few flagged frames first.

## Caveats / tuning

- **Threshold sensitivity**: `tau` (blink) and `v_sacc` (saccade) are both
  tunable; sanity-check `tau` by viewing frames flagged Blink vs open.
- **cv2 vs U-Net**: the cv2 detector can be fooled by heavy lashes/shadow; the
  U-Net-mask path is more robust if you have the GPU budget.
- **Stride**: blinks are brief — `--stride > 2` risks undercounting Blink.
- **Half-blinks**: partial lid closure is borderline; the score threshold treats
  it by pupil visibility, not lid position.
- Frame stream is ~25 Hz; speeds > ~2000 px/s are physiologically saccade-range.
