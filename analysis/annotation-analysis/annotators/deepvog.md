# DeepVOG — Pupil Segmentation / Gaze-Estimation Annotator Review

Analysis of the vendored repo at `HBTXR/third/DeepVOG` as a **potential independent
pupil-mask / pupil-center audit source** for the EV-Eye pipeline, to sit alongside
the GroundingDINO+SAM2 ("GSAM2") audit and the EV-Eye U-Net pseudo-labels.

- **Upstream**: https://github.com/pydsgz/DeepVOG (Yiu et al.)
- **Vendored version**: `1.1.4` (`setup.py:10`, `RELEASE.md:2`; released 31-07-2019)
- **License**: GNU GPLv3 (`LICENSE`, `setup.py:17`, `README.md:175`) — **copyleft; relevant if any DeepVOG code is redistributed or linked into our tooling.**
- **Scope note**: what follows distinguishes *what the repo actually contains* from *unclear / missing / mismatched* items.

---

## 1. What it is

- **Purpose**: "a framework for pupil segmentation and gaze estimation based on a fully convolutional neural network … offline gaze estimation of eye-tracking video clips" (`README.md:5`).
- **Method / architecture**: a **U-Net-style fully-convolutional encoder–decoder** built in **Keras** (`deepvog/model/DeepVOG_model.py`).
  - `DeepVOG_net()` (`DeepVOG_model.py:70`): 4 encoding blocks (filters 16→32→64→128, each conv + BatchNorm + ReLU then a stride-2 downsample conv) and 5 decoding blocks (`Conv2DTranspose` upsampling + **skip `Concatenate`** from the matching encoder stage), i.e. the classic U-Net long skip connections (`DeepVOG_model.py:44`, comments cite U-Net / V-Net).
  - **Output head** (`DeepVOG_model.py:98`): `Conv2D(filters=3, 1x1)` + **softmax** → a 3-channel per-pixel map. Per the code comment (`DeepVOG_model.py:68`) channel 0 = non-pupil, **channel 1 = pupil probability**, channel 2 = trivial/zeros. Downstream everything uses `Y[:,:,1]` (`inferer.py:192,232`).
  - Fixed input size **240×320×3** with a large **10×10** conv kernel at load time (`DeepVOG_model.py:107`: `DeepVOG_net(input_shape=(240,320,3), filter_size=(10,10))`).
- **Paper / year**: Yiu YH et al., "DeepVOG: Open-source Pupil Segmentation and Gaze Estimation in Neuroscience using Deep Learning", *Journal of Neuroscience Methods*, vol. 324, **2019** (`README.md:11`, `citations.bib`). Peer-reviewed, open access.
- **Two-stage pipeline** (this is the important structural point):
  1. **CNN pupil segmentation** → pupil probability heatmap (per frame).
  2. **Geometric / model-based post-processing** (pure NumPy, no learning): heatmap → largest connected component → **least-squares ellipse fit** → 3D unprojection → RANSAC **3D eyeball-sphere model** → gaze angles. This lives in `deepvog/eyefitter.py`, `draw_ellipse.py`, `unprojection.py`, `intersection.py`, `ellipses.py`, `CheckEllipse.py`.

---

## 2. Eye outputs (segmentation + geometry)

DeepVOG can emit, per frame:

| Output | Produced? | Where | Format |
|---|---|---|---|
| **Pupil probability heatmap** | **Yes** | `model.predict()` → `Y[:,:,1]`, float [0,1], **240×320** | in-memory np array; only *rendered* to video via `-m/--heatmap` (`visualisation.py:120`) |
| **Binary pupil mask** | **Derivable, not saved by default** | `isolate_islands()` thresholds heatmap at 0.5 and keeps the **largest** connected component (`draw_ellipse.py:10`) | 0/1 np array, 240×320. **The CLI never writes this to disk** — it is an internal intermediate. |
| **Pupil ellipse** | **Yes** | least-squares fit (`draw_ellipse.py:37` via `ellipses.LSqEllipse`) | `(center[x,y], w, h, radian)` in the 240×320 frame |
| **Pupil center (2D)** | **Yes** | ellipse center | `pupil2D_x, pupil2D_y` columns in results CSV (`visualisation.py:132`, `documentation.md:51`) |
| **Ellipse/segmentation confidence** | **Yes** | mean pupil prob inside the fitted ellipse (`CheckEllipse.py:8`) | `confidence` column |
| **3D eyeball model** | **Yes** | RANSAC sphere fit over many frames (`eyefitter.py:131,153`) | `eye_centre` (3×1) + `aver_eye_radius`, saved as JSON (`inferer.py:165`) |
| **Gaze angles (yaw/pitch)** | **Yes** | 3D unprojection + consistent-pupil selection (`eyefitter.py:206`) | `gaze_x, gaze_y` (deg) + `consistence` flag |

**Full results CSV schema**: `frame, pupil2D_x, pupil2D_y, gaze_x, gaze_y, confidence, consistence` (`visualisation.py:138`, `documentation.md:59`).

**Key takeaways for us:**
- DeepVOG **does produce a pupil mask internally**, but the shipped CLI's on-disk deliverables are (a) the **CSV** (center + gaze + confidence) and (b) an optional **visualization video**. To get an actual **mask array** or an ellipse per frame we must call the Python API and slice `Y[:,:,1]` (or reuse `isolate_islands` / `fit_ellipse`) ourselves — the CLI won't dump masks.
- `--no_gaze` mode (`__main__.py:72`, `documentation.md:24`) gives **pupil-only** output (center + confidence, gaze columns = NaN) **without needing an eyeball model**. This is the mode most relevant to us.
- The pupil **center comes from the ellipse fit of the segmentation**, i.e. exactly the same "mask → ellipse-fit center" recipe our GSAM2 runner already uses (`scripts/08_run_gsam2.py` fits an ellipse to the SAM2 mask for `y_gsam2`). So DeepVOG's center is directly comparable to GSAM2's and to the EV-Eye ellipse-center convention.

---

## 3. Input assumptions vs EV-Eye DAVIS346

- **Modality**: near-eye monochrome eye videos from a **head-mounted camera**, single eye filling the frame (`README.md:150-153`). This is a **near-IR/grayscale eye-tracking domain — the same family as EV-Eye's DAVIS346 APS frames** (no natural-image domain gap, unlike GSAM2's GroundingDINO).
- **Color handling**: input is converted to grayscale (`rgb2gray`) then tiled into 3 identical channels (`inferer.py:342-345`). Our frames are already grayscale (PNG mode `L`), so this is a no-op conversion — **DeepVOG natively expects the kind of image we have.**
- **Resolution / aspect ratio**: network is **fixed at 240×320** (aspect 0.75). Any other size is `skimage.resize`d to 240×320 (`inferer.py:344`). **Our frames are 346×260** (verified: `samples/frame/.../*.png` → PIL size `(346,260)`, mode `L`), aspect **260/346 ≈ 0.751** — essentially exactly 0.75, so the resize to 240×320 is near-isotropic and introduces negligible distortion. The docs explicitly warn that off-0.75 aspect ratios hurt accuracy (`__main__.py:20`); we are fine.
- **Pupil scale**: our pupil is ~40–60 px in a 346×260 frame (~0.12–0.17 of width). After resize to 320 wide that is ~37–55 px — comfortably within what a U-Net trained on head-mounted eye video segments. No obvious scale mismatch.
- **Input plumbing mismatch**: the CLI/`inferer` are **video-only** — they open the source with `skvideo.io.FFmpegReader` and iterate `nextFrame()` (`inferer.py:299-303`). **Our samples are per-frame PNG folders** (`samples/frame/<clip>/<frame>.png`), not `.mp4`. So the file-level entry points don't ingest our data as-is (see §4/§6).

---

## 4. How to run inference end-to-end

### 4.1 Entry points
- **CLI**: `python -m deepvog ...` (`deepvog/__main__.py`). Three mutually-exclusive modes:
  - `--fit VIDEO MODEL.json` → fit 3D eyeball model (`__main__.py:59`).
  - `--infer VIDEO MODEL.json RESULTS.csv` → per-frame center + gaze (`__main__.py:60`).
  - `--table LIST.csv` → batch (`__main__.py:61`).
  - Pupil-only, **no eyeball model needed**: add `--no_gaze` to `--infer` (`__main__.py:72`, `documentation.md:24`).
  - Useful flags: `-b/--batchsize`, `-g/--gpu`, `-vs/--vidshape`, `-s/--sensor`, `-f/--flen`, `-v/--visualize`, `-m/--heatmap`.
  - Example (README `demo`): `python -m deepvog --infer ./demo.mp4 ./model.json ./out.csv -b 32 -v ./vis.mp4 -m`.
- **Python API** (more flexible, what we'd actually use):
  ```python
  import deepvog
  model = deepvog.load_DeepVOG()                                   # __init__.py:3, DeepVOG_model.py:105
  inferer = deepvog.gaze_inferer(model, flen, video_shape, sensor) # inferer.py:14
  inferer.process("clip.mp4", mode="Infer", output_record_path="out.csv")
  ```
  To get **masks/ellipses directly** (bypassing video I/O), the reusable primitives are `model.predict(batch)` → `Y[:,:,1]`, then `deepvog.draw_ellipse.isolate_islands()` / `fit_ellipse()` and `SingleEyeFitter.unproject_single_observation()`.
- **Smoke test**: `deepvog/model/test_if_model_work.py` loads the model, predicts on `model/test_image.png`, and saves `test_prediction.png` (channel 1). Good minimal sanity check that weights load and segment.

### 4.2 Pretrained weights — **present in-repo (major plus)**
- `deepvog/model/DeepVOG_weights.h5` — **exists locally, ~99 MB, valid HDF5** (verified: `file` → "Hierarchical Data Format (version 5)"; 98,964,040 bytes). Loaded by `load_DeepVOG()` via `model.load_weights(...)` (`DeepVOG_model.py:108`).
- **No download step required** — weights are checked in and packaged (`setup.py:28` `package_data` includes `model/*.h5`). This is a decisive advantage over methods whose weights must be fetched.
- Corresponds to the **2D pupil-segmentation model** described in the paper. (The eyelid `mask=` argument in `fit_ellipse` is stubbed for a separate unreleased "DeepVOG-3D" — comments at `draw_ellipse.py:69,98`; **not** in this repo.)

### 4.3 Dependencies / framework / GPU
- Framework: **TensorFlow 1.x + Keras** (`setup.py:36-37`: `tensorflow-gpu>=1.12.0`, `keras>=2.2.4`; imports are top-level `from keras...`, `DeepVOG_model.py:3-7`). **This is the standalone Keras + TF1 stack — a legacy combination.**
- Other deps: `numpy`, `scikit-video` (needs **FFmpeg**), `scikit-image`, plus `matplotlib` used in `draw_ellipse.py`. Python **3.5–3.7** (`setup.py:32`).
- GPU: `tensorflow-gpu` is the declared dependency and the code sets `CUDA_VISIBLE_DEVICES` (`jobman.py:22`); a Docker image `yyhhoi/deepvog:v1.1.4` is offered (`README.md:69`). CPU-only TF would work for inference but the packaging assumes GPU.
- **Environment risk**: TF1/Keras + Python ≤3.7 does **not** coexist with our modern stack. Our repo runs Python 3.12 (`.venv`, `__pycache__/*.cpython-312.pyc`) and PyTorch-based GSAM2 (`.venv-gsam2`). DeepVOG needs its **own isolated legacy env** (separate venv/conda or the vendor Docker image), exactly as GSAM2 already has a dedicated `.venv-gsam2`.

---

## 5. Applicability as an independent audit source (vs GSAM2)

**Can it produce a per-frame pupil mask/center on our frames?** **Yes**, subject to (a) feeding PNGs (not video) and (b) standing up a legacy TF1 env. The model, weights, and mask→ellipse→center path are all present and domain-appropriate.

**Independence** — this is the strong argument for including it:
- DeepVOG is **methodologically orthogonal** to both existing sources. It is a **domain-specific eye U-Net** (trained on head-mounted eye video), whereas GSAM2 is a **natural-image open-vocabulary detector+segmenter** (GroundingDINO text prompt "pupil." + SAM2) and the EV-Eye U-Net is trained **on EV-Eye's own manual ellipse labels**. Three independent priors ⇒ a genuinely independent third opinion for triangulating label quality.
- Because it is trained on IR/grayscale eye imagery, it has **no natural-image domain gap** — the concern that dominates GSAM2 on near-IR frames (the GSAM2 runner even documents that GroundingDINO's top box is often the whole eye, needing a `max_frac` filter; `scripts/08_run_gsam2.py`). DeepVOG sidesteps that class of failure.

**Strengths**
- Weights shipped in-repo; no external fetch.
- Purpose-built for exactly this modality (monocular near-eye grayscale, single eye).
- Emits a **calibration-free confidence** per frame (mean pupil-prob inside the ellipse, `CheckEllipse.py`) → directly usable to weight/filter its audit votes, analogous to how we treat EV-Eye quality flags.
- Center recipe (mask → LSq ellipse → center) is **identical in spirit** to our GSAM2 center, so metrics (pixel-error vs EV-Eye ellipse center) are apples-to-apples.
- `--no_gaze` avoids the whole 3D-eyeball-fit machinery — we only need segmentation.

**Concerns / caveats**
- **Training-domain mismatch (unquantified)**: trained on the authors' clinical VOG rig, **not** DAVIS346. IR wavelength, glint pattern, contrast, and especially **DVS/APS sensor characteristics** differ. Expected to generalize better than GSAM2, but **must be empirically validated** on our frames before trusting it — treat as a hypothesis, not a given.
- **Assumes a single centered eye** (`README.md:150`); fine for EV-Eye close-ups, but frames with heavy lid closure / blink / extreme gaze may yield no ellipse (code returns NaN center — `inferer.py:287`), i.e. **coverage gaps** on exactly the hard frames (saccade/blink) we care about.
- **Legacy framework** (TF1.x/Keras, Py≤3.7): install friction, no active maintenance, Apple-Silicon/newer-CUDA unfriendly. Isolated env or Docker mandatory.
- **Code maturity**: research-grade. Video-only I/O; a latent `np.int` deprecation in RANSAC (`eyefitter.py:144`, removed in modern NumPy — would break under a too-new NumPy, reinforcing the pinned legacy env); the batching loop in `inferer.process` is intricate and frame-order-sensitive.
- **License is GPLv3** — running it out-of-process to produce label JSON is fine, but **do not import/vendor DeepVOG code into our (non-GPL) tooling**; keep it behind a subprocess/CLI boundary.
- **Mask not persisted by the CLI** — to store masks (for IoU vs EV-Eye masks) we must use the Python API, not `--infer`.

---

## 6. Integration effort: **Medium**

Wiring DeepVOG as a per-frame pupil-center (+ optional mask) producer over our samples, mirroring the GSAM2 audit (`scripts/08_run_gsam2.py` → `samples/label/{key}/gsam2.json`).

**Why not Low:**
- **Separate legacy runtime.** TF1.x + Keras + Py≤3.7 cannot live in `.venv` (3.12) or `.venv-gsam2` (PyTorch). Needs its own conda/venv or the vendor Docker image — an env-provisioning task, not just a script. (Precedent exists: GSAM2 already has a dedicated env, so the pattern is established.)
- **Input adapter needed.** `inferer` ingests **video via FFmpegReader**; our data is **per-frame PNG folders** (`samples/frame/<clip>/*.png`). Two options: (a) `ffmpeg`/`skvideo` each clip's PNGs → a temp `.mp4` then `--infer --no_gaze`; or (b) **bypass `inferer` entirely** and write a thin loop: read PNGs → resize to 240×320 → `model.predict` → `Y[:,:,1]` → `isolate_islands`/`fit_ellipse` → center (+ mask). Option (b) is cleaner, avoids video round-trips, and lets us persist masks; it reuses only `deepvog.load_DeepVOG`, `draw_ellipse.py`, and (optionally) `SingleEyeFitter.unproject_single_observation`.
- **Coordinate bookkeeping.** Model runs at 240×320; centers/masks must be scaled back to the native **346×260** grid to compare with EV-Eye labels, GSAM2, and the U-Net. Straightforward but must be exact (our other tools work in 346×260; cf. `11_build_perframe.py`).

**Why not High:**
- Weights are already local — **zero model-acquisition work**.
- Reusable primitives already exist (`load_DeepVOG`, `isolate_islands`, `fit_ellipse`, `computeEllipseConfidence`); the mask→ellipse→center path is done for us.
- The target output contract is already defined by the GSAM2 runner — we emit a parallel `samples/label/{key}/deepvog.json` (`{frame, cx, cy, w, h, angle, confidence}` in 346×260 coords), so `05_eval_annotation_quality.py` can consume it exactly like `gsam2.json` / `unet_dense.json`.

**Concrete plan (est. ~1 focused day, most of it env setup):**
1. Provision a legacy env (conda `python=3.7 tensorflow-gpu=1.15 keras=2.2.4 scikit-image scikit-video`, or `docker run yyhhoi/deepvog:v1.1.4`).
2. Add `scripts/NN_run_deepvog.py`: iterate `samples/frame/<key>/*.png` → grayscale (already `L`) → resize 240×320 → `model.predict` (batched) → `Y[:,:,1]` → `isolate_islands`+`fit_ellipse` → center/ellipse/confidence → rescale to 346×260 → write `samples/label/{key}/deepvog.json` (optionally also save the 0/1 mask for IoU vs EV-Eye masks).
3. Feed into the existing eval (`05_eval_annotation_quality.py`) as a **third independent audit vote** (pixel-error vs EV-Eye ellipse center; agreement/disagreement with GSAM2 and the U-Net; per-frame confidence gating).

---

## 7. Bottom line

DeepVOG is a **2019 U-Net pupil segmenter + geometric ellipse/3D-gaze post-processor** (Keras/TF1, GPLv3) built for **exactly our modality** (monocular near-eye grayscale, single centered eye). It **ships its own ~99 MB pretrained weights in-repo**, produces a **pupil probability map → binary mask → least-squares ellipse → 2D center + confidence** (and optional 3D gaze we don't need), and its `--no_gaze` mode gives calibration-free pupil-only output. Our **346×260 grayscale frames are a near-perfect fit** (aspect ≈0.75; grayscale native; pupil scale in range). The blockers are **operational, not conceptual**: video-only I/O vs our PNG folders, and a **legacy TF1/Py≤3.7 env** that must be isolated (as GSAM2 already is). As an **independent audit source it is arguably better-motivated than GSAM2** for near-IR eyes (no natural-image domain gap), with the main open question being **empirical generalization from the authors' rig to DAVIS346**, which must be measured before trusting its votes. Integration is **Medium** — mostly env provisioning plus a thin PNG→predict→ellipse adapter emitting a `deepvog.json` parallel to `gsam2.json`.
