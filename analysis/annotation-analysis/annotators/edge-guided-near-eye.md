# Edge-Guided Near-Eye Image Analysis for HMDs — Annotator Analysis

Analysis of the repository `third/Edge-Guided-Near-Eye-Image-Analysis-for-Head-Mounted-Displays`
(GitHub: `zhaoyuhsin/Edge-Guided-Near-Eye-Image-Analysis-for-Head-Mounted-Displays`) as a potential
**independent pupil/eye segmentation audit source** for our EV-Eye near-IR MASK annotation pipeline,
to compare against our GroundingDINO+SAM2 ("GSAM2") audit and the EV-Eye U-Net.

Repo root (paths below are relative to it):
`/home/user/project/PRJXR-HBTXR/HBTXR/third/Edge-Guided-Near-Eye-Image-Analysis-for-Head-Mounted-Displays/`

---

## 1. What it is (purpose, method, paper, license)

**Purpose.** Purpose-built **near-eye eye-tracking front-end for AR/VR Head-Mounted Displays under
near-infrared (NIR) illumination**. It performs iris/pupil **segmentation** and **ellipse fitting**
on near-eye camera frames — exactly the HMD eye-tracking setting, so it is a strong domain match on
paper for EV-Eye near-IR close-ups.

**Paper / year.** PyTorch implementation of the **ISMAR 2021** conference paper
*"Edge-Guided Near-Eye Image Analysis for Head Mounted Displays"*, Zhimin Wang, Yuxin Zhao, Yunfei
Liu, Feng Lu (IEEE ISMAR 2021, pp. 11–20, doi `10.1109/ISMAR52148.2021.00015`). See `README.md`
lines 7–16, 83–97. The codebase is explicitly a **modification of the EllSeg framework**
(`RSKothari/EllSeg`) — the model file headers still credit `@author: rakshit` (Rakshit Kothari),
e.g. `models/RITnet_v2.py:4`, `test.py`, `evaluate.py`. So it inherits the EllSeg / RITnet lineage
(DenseNet2D encoder-decoder + ellipse regression) and adds an edge branch.

**Method — how "edge-guided" works.** Two-stage pipeline:
1. **Edge Extraction Network (BDCN).** A **BDCN** (Bi-Directional Cascade Network) edge detector,
   VGG16-based (`bdcn_new.py:65` `class BDCN`, backbone `vgg16_c.VGG16_C` at `bdcn_new.py:7`),
   predicts a "clean" edge map that (per the paper) contains only eyelid + iris/pupil contours,
   suppressing NIR reflections/other edges. It is fed a 3-channel input by **replicating the
   grayscale channel 3×** (`utils.py:649` and `evaluate.py:106`:
   `edge_model(torch.cat((img, img, img), dim=1))[-1]`), and the network returns a list of sigmoid
   side-outputs; the **fused map (last element `[-1]`)** is used as the edge (`bdcn_new.py:189-191`).
2. **Edge-Guided Segmentation & Fitting Network (ESF-Net).** A `DenseNet2D`
   (`models/RITnet_v2.py:203`) — a RITnet/EllSeg-style dense encoder-decoder. With
   `add_edge:1` (the shipped "ours" config `configs/baseline_edge.yaml`), the **same encoder is run
   twice** — once on the image, once on the edge map — and the two bottleneck feature stacks are
   **concatenated** (`RITnet_v2.py:283-287`, `feature_channels 153 → 306`) before the decoder and the
   ellipse-regression head. The repo also contains ablation variants for how edges enter the net:
   `only_edge` (feed edge only), `input_concat` (concat edge as a 2nd input channel), plus an
   AdaIN/style branch (`add_seg`) and an edge-threshold option — see `configs/` (`baseline.yaml`,
   `baseline_edge.yaml`, `baseline_only_edge.yaml`, `baseline_input_concat.yaml`,
   `baseline_adain*.yaml`, `baseline_edge_thres.yaml`) and the flags consumed in
   `RITnet_v2.py:273-308`.

**License.** **No license file** is present (`ls LICENSE* COPYING*` → none). The README grants no
usage terms beyond a citation request ("If you only use our code base, please cite…",
`README.md:83`). Treat as **all-rights-reserved academic code** — usable for internal
research/audit, but redistribution/derivative status is unclear. It also carries the upstream EllSeg
lineage, whose own license would need checking if code is reused.

---

## 2. Eye outputs (what it produces, exact format)

For each eye frame the model returns a **3-class semantic segmentation map + two fitted ellipses**.

- **Segmentation MASK — YES.** 3 classes: **0 = background, 1 = iris, 2 = pupil** (argmax over the
  3-channel decoder output). See `utils.py:65` `get_predictions` (argmax of `[B,3,H,W]`),
  `models/RITnet_v2.py:225` (`out_c=3`), and the color overlay in `helperfunctions.py:521-536`
  (`seg_map==1`→iris green `[120,183,53]`, `seg_map==2`→pupil yellow `[36,231,253]`). So a **pupil
  binary mask is directly available as `seg_map == 2`**, and an **iris mask as `seg_map == 1`**.
  This is the RITnet/EllSeg convention: background + iris + pupil. **Note: no separate sclera
  class** (sclera falls into background); this differs from OpenEDS-style 4-class models.
- **Ellipse fit — YES (this is the headline output).** For **both pupil and iris** the model emits
  ellipse parameters `[cx, cy, a, b, angle]` in pixel coords. In `evaluate.py:138-151` the network's
  normalized ellipse params (`elPred[5:10]` pupil, `elPred[0:5]` iris) are de-normalized via the
  `my_ellipse(...).transform(H)` matrix, then **refined by a local IoU search against the seg mask**
  (`search_proper_parameter_iou_for_our_data`, `utils.py:450-486`).
- **Center — YES.** The pupil center is the ellipse `(cx, cy)`. `evaluate.py:254` explicitly collects
  `app_center[i].append((pupil_ellipse[0], pupil_ellipse[1]))` per frame and pickles them to
  `our_data_test/app_centers.pkl`. There are actually **two center estimates**: a
  segmentation-center-of-mass (`elPred`, `test.py:100`) and a latent/regressed center (`elOut`,
  `test.py:97`); the video path uses the ellipse center.
- **Gaze — NO direct 3D gaze from this code.** The paper *demonstrates* AR-HMD gaze applications,
  but the released inference (`evaluate.py`) outputs **per-frame masks + pupil/iris ellipses +
  centers only**; no gaze vector is written.

**Concrete artifacts written by `evaluate.py`** (per input `.avi`):
- `<name>_result_<method>.mp4` — overlay video (mask colors + ellipses).
- `<name>_pred2_<method>.npy` — dict `{frame_index: (iris_ellipse, pupil_ellipse)}` (`evaluate.py:273,302`).
- `our_data_test/app_centers.pkl` — per-frame pupil centers (`evaluate.py:290-293`).
- Segmentation maps themselves are computed in-memory; there is a `--save_maps` flag
  (`evaluate.py:40`) but **no code path actually writes raw mask PNGs** in `evaluate_ellseg_per_video`
  — masks are only rendered into the overlay. Extracting per-frame mask arrays requires a small code
  addition (they exist as the `seg_map` numpy array).

---

## 3. Input assumptions & fit for EV-Eye DAVIS346

- **Modality.** Near-eye **NIR/grayscale** — exactly our domain. Frames are converted to grayscale
  (`evaluate.py:245` `cv2.cvtColor(..., COLOR_BGR2GRAY)`) then z-score normalized
  (`evaluate.py:102` `(img - img.mean())/img.std()`). The BDCN edge weights are noted as
  **trained on a real (not synthetic) dataset** (`evaluate.py:320`), which is favorable for real
  DAVIS346 frames.
- **Resolution / preprocessing.** The network expects **320×240 (W×H) per eye**. `preprocess_frame`
  (`evaluate.py:69-104`, comment line 68 "Input frames must be resized to 320X240") resizes by
  matching **width to 320** then vertically pads/crops to height 240. Our **DAVIS346 frames are
  346×260**, so they will be scaled to width 320 → ~320×240.4 then padded/cropped to 240 — a mild,
  near-isotropic rescale that should preserve the ~40–60 px pupil disc reasonably (pupil becomes
  ~37–55 px). Ellipses are rescaled back to original resolution afterward (`rescale_to_original`,
  `evaluate.py:169-192`), so outputs can be expressed in native 346×260 coords.
- **Two-eye assumption (IMPORTANT mismatch).** `evaluate.py:242-243` **hard-codes a stereo layout**:
  it splits every frame into a left half `[:, 0:320]` and right half `[:, 320:640]` and runs both,
  i.e. it assumes a **640-px-wide two-eye video**. EV-Eye DAVIS346 frames are **single-eye 346-wide**.
  As-is this splitter would mis-crop our frames (a 346-wide frame yields one 320-slice + a 26-px
  slice). **This is a small but mandatory code change** for our single-eye data (drop the loop, treat
  the whole frame as one eye).
- **Input container.** `evaluate.py` only ingests **`.avi` video** via `Path(...).rglob('*.avi')`
  (`evaluate.py:375`) and OpenCV `VideoCapture`. Our per-frame image samples would need to be
  packaged as a video or the loader adapted to read image folders (an inline single-image path is
  present but commented out at `evaluate.py:324-338`).

**Fit verdict:** modality and purpose are an excellent match; the frictions are the hard-coded
two-eye split, the video-only loader, and the fixed 320×240 target — all mechanical, not
fundamental.

---

## 4. How to run inference end-to-end

**Two entry points:**

**(A) `evaluate.py` — "run on your own eye videos" (the relevant path for us).**
`README.md:68-81`. Example: `python3 evaluate.py --path2data videos`.
- Loads **BDCN edge net** from `gen_00000016.pt` (`evaluate.py:319-323`).
- Loads **segmentation/fitting net** from `baseline_edge_16.pkl` with `configs/baseline_edge.yaml`
  (`evaluate.py:357-359`, `load_from_file` in `pytorchtools.py`).
- Iterates `*.avi` in `--path2data`, writes overlay mp4 + `_pred2_*.npy` ellipse dicts.
- **Caveat:** the loop assumes stereo (see §3) and it writes to `our_data_test/app_centers.pkl`
  (`evaluate.py:290`) — **`our_data_test/` does not exist in the repo** and is not auto-created, so
  the run will crash at the final pickle unless the dir is created (minor). `edge_out`/`vid_out`
  release lines are also active while some `edge_out.write` lines are commented — small cleanup
  needed.

**(B) `test.py` — benchmark on the paper's datasets.**
`README.md:44-66`. Example:
`python3 test.py --curObj LPW --path2data datasets --loadfile baseline_edge_16.pkl --setting configs/baseline_edge.yaml`.
- **Requires preprocessed `.h5` dataset files** under `datasets/Datasets/TEyeD-h5-Edges`
  (`test.py:274`) **and** pickled curriculum/condition objects `baseline/cond_<curObj>.pkl`
  (`test.py:271`). **Neither `datasets/` nor `baseline/` is in the repo** (confirmed absent). The
  datasets/objects must be generated (`dataset_generation/` scripts) or downloaded from the Google
  Drive link in `README.md:31`. Not needed for auditing our own frames.

**Pretrained weights — PRESENT IN-REPO (key advantage):**
- `gen_00000016.pt` — **63 MB**, BDCN edge model (loaded as `state_dict['a']`). Present.
- `baseline_edge_16.pkl` — **13 MB**, the edge-guided seg+fit model ("ours"). Present.
So the full "ours" inference stack ships with weights — **no external download required** to run
`evaluate.py`. (The comparison baselines DeepVOG/RITnet/plain-EllSeg weight paths in
`evaluate.py:344-353` are empty/commented, but we don't need those.)

**Key deps / framework / version:**
- **PyTorch** (README states 1.4.0; `requirements.txt` conda spec pins `pytorch=1.2.0` + `cudatoolkit=9.2`,
  `torchvision=0.4.0`), **Python 3.6–3.7**, numpy 1.16–1.19, **OpenCV 3.4.2**, scikit-image, scipy,
  h5py, deepdish, pyyaml, tqdm, matplotlib (`requirements.txt`, `README.md:20-25`). These are **old
  pins** (2021-era, CUDA 9.2); running on modern GPUs/CUDA will likely need a newer torch and minor
  API touch-ups (e.g. `yaml.load` without `Loader` at `test.py:21`/`evaluate.py:310`).
- **GPU:** code is **CUDA-hard-wired** in the shipped path — `evaluate.py:319-323` calls `.cuda()`
  unconditionally on the edge net, and `torch.device("cuda")` is the default. There is an
  `--eval_on_cpu` flag for the seg model (`evaluate.py:50,200-203,371`), but the **BDCN edge net is
  forced to CUDA regardless**, so a GPU is effectively required without a small edit. Models are
  small (13 MB + 63 MB), so any modest GPU suffices; batch is 1 frame at a time in the video path.

**Training datasets (context, from `README.md:52` + `dataset_generation/`):**
**TEyeD** (with sub-sources **LPW, NVGaze, Fuhl**), **OpenEDS**, and **RITEyes** (synthetic) — all
standard NIR near-eye eye-tracking datasets. Extractors: `dataset_generation/Extract_TEyeD_*_histo.py`,
`ExtractOpenEDS_seg_histo.py`, `ExtractRITEyes_general.py`. **EV-Eye is not among the training sets**,
so applying to EV-Eye is a cross-dataset / domain-transfer use (see §5).

---

## 5. Applicability as an independent audit source (like GSAM2)

**Yes — it can serve as an independent per-frame pupil producer**, and a *methodologically
independent* one:
- **Independence from GSAM2 and EV-Eye U-Net.** This is a **different model family and different
  supervision signal**: a BDCN edge detector + RITnet/EllSeg dense seg/fit net trained on
  TEyeD/OpenEDS/RITEyes. That is entirely distinct from (a) GSAM2's open-vocab detection+promptable
  SAM2 masks and (b) the EV-Eye U-Net. Agreement across all three would be strong evidence; this
  method's disagreements are informative because its inductive biases (edge contours, ellipse
  geometry) differ.
- **Directly yields what an audit needs:** a **pupil binary mask** (`seg_map == 2`), a **pupil
  ellipse** `[cx,cy,a,b,θ]`, and a **pupil center** — plus an iris mask/ellipse for free. Ellipse
  center + axes give clean center/size checks against our masks, and the mask gives IoU against our
  GSAM2/U-Net masks.

**Strengths.**
- **Purpose-built for near-eye NIR HMD** — the exact modality of EV-Eye; far better domain prior
  than a generic segmenter.
- **Pretrained weights ship in-repo** (both stages) — no missing/downloadable-weights blocker for
  the "ours" configuration.
- Outputs an **explicit pupil ellipse** (center, axes, angle), which is more informative than a raw
  blob and matches how pupil GT is often parameterized; includes a mask→ellipse IoU refinement step.
- Runs cheaply (small models, per-frame).

**Concerns.**
- **Domain gap on EV-Eye specifics.** Trained on TEyeD/OpenEDS/RITEyes, **not EV-Eye**; EV-Eye is
  DAVIS346 (event-camera APS) grayscale with its own contrast/noise/reflection characteristics.
  Cross-dataset degradation is plausible and **must be spot-checked** before trusting it as an
  auditor. It expects an iris to be visible/segmentable; very dark or low-contrast irises could hurt
  the iris channel (though the **pupil** channel is what we care about).
- **Hard-coded stereo split + `.avi`-only loader** (`evaluate.py:242-243,375`) — needs adaptation
  for single-eye per-frame data (mechanical).
- **Code maturity is "research script" level.** Lots of commented-out debug code, a missing
  `our_data_test/` dir that breaks the final pickle, an active `edge_out.release()` with commented
  writes, `yaml.load` without a Loader, **no license file**, and 2021-era dependency pins
  (torch 1.2/CUDA 9.2). Expect to modernize the environment and patch a few lines to run on current
  hardware.
- **Not a raw-mask exporter out of the box.** `--save_maps` exists but is not wired to write mask
  files; per-frame masks are in-memory (`seg_map`) and need a few lines to dump as PNG/npz.
- **CUDA required** in practice (edge net forced to GPU).

**Net:** a credible, purpose-built, weights-included **independent auditor for pupil mask + center +
ellipse**, provided we (1) sanity-check accuracy on a handful of EV-Eye frames and (2) make small
loader/output edits.

---

## 6. Integration effort: **LOW–MEDIUM**

To wire it as a **per-frame pupil producer over our EV-Eye samples**:

**Low-effort parts (weights + core forward already work):**
- Pretrained weights are present; the forward path `evaluate_ellseg_on_image`
  (`evaluate.py:112-166`) already returns `(edge_map, seg_map, pupil_ellipse, iris_ellipse)` — i.e.
  **the pupil mask/center/ellipse we want are one function call away.** We can import
  `preprocess_frame`, `evaluate_ellseg_on_image`, `rescale_to_original` and call them per image.

**The required edits (why it's not "trivial/low"):**
1. **Single-eye loader.** Replace the two-eye split (`evaluate.py:242-243`) and the `*.avi`
   `rglob` (`evaluate.py:375`) with a loop over our per-frame image files (a single-image code stub
   already exists, commented, at `evaluate.py:324-338`). ~30–60 min.
2. **Mask/center export.** Add a few lines to dump `seg_map==2` (pupil) and `seg_map==1` (iris) and
   the ellipse dict per frame in our schema (the arrays already exist in-memory). Create
   `our_data_test/` or remove that pickle. ~30 min.
3. **Environment.** Stand up an old-ish PyTorch env (or port to a modern torch — the model code is
   plain conv/BN, so a newer torch should load the state dicts with minor tweaks: `yaml.load` Loader
   arg, `np.int` deprecation at `RITnet_v2.py:22-23`, CUDA/CPU device handling). This is the biggest
   variable — clean on 2021 pins, some friction on modern CUDA. ~1–3 h.
4. **Optional sanity/calibration.** Run on a small labeled EV-Eye subset, compare pupil center/mask
   vs our GSAM2 + U-Net to confirm it's trustworthy and to characterize any systematic bias from the
   320×240 rescale.

**Overall:** **LOW–MEDIUM.** Core inference + weights are ready; the effort is a small loader/output
shim plus environment modernization, not any modeling work. Compared to GSAM2 (large foundation
models, prompt engineering), this is lighter to run but needs the near-eye-specific plumbing above
and an EV-Eye accuracy spot-check before it's relied upon as an auditor.

---

## Repo-contains vs unclear/missing (quick reference)

| Item | Status |
|---|---|
| Edge-guided seg+fit weights `baseline_edge_16.pkl` (13 MB) | **Present in-repo** |
| BDCN edge weights `gen_00000016.pt` (63 MB) | **Present in-repo** |
| `evaluate.py` (own-video inference) + configs | **Present** |
| Example input `videos/example1.avi` | **Present** |
| Pupil/iris **mask** (3-class) + **ellipse** + **center** outputs | **Present** (mask array in-memory; raw-mask file export not wired) |
| Baseline weights (DeepVOG/RITnet/plain EllSeg) | **Missing/empty paths** (not needed for "ours") |
| `datasets/` + `baseline/cond_*.pkl` for `test.py` benchmarking | **Missing** (generate or download) |
| `our_data_test/` output dir used by `evaluate.py` | **Missing** (must create) |
| **License** | **Missing** (citation-only; treat as academic all-rights-reserved) |
| Gaze-vector output | **Not in released inference** |
| Training code (`train.py` present, but README says "coming soon") | Script present; training data not shipped |
| EV-Eye in training set | **No** (TEyeD/OpenEDS/RITEyes only) → cross-dataset use |
