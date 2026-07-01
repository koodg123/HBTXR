# EllSeg as an Independent Pupil/Iris Audit Source

Analysis of the vendored `EllSeg` repository at `HBTXR/third/EllSeg` for use as an
**independent audit / label producer** on EV-Eye DAVIS346 near-IR eye frames,
alongside our GroundingDINO+SAM2 ("GSAM2") audit and the EV-Eye U-Net pseudo-labels.

> Scope note: this reviews only the code and assets actually present in
> `HBTXR/third/EllSeg`. Where the repo is ambiguous or a file is missing, that is
> called out explicitly.

---

## 1. What it is

- **Purpose.** EllSeg is a CNN framework for **robust pupil/iris tracking** in
  video-oculography. Its core idea: instead of segmenting only the *visible* pupil/iris
  pixels and then fitting an ellipse (which breaks under eyelid/eyelash/camera
  occlusion), train the network to **directly segment the entire elliptical structure**
  (the full pupil disc and full iris disc, including occluded parts), and additionally
  **regress the ellipse parameters** directly. See `README.md:10-14` (Abstract) and the
  loss design in `loss.py` / `models/RITnet_v3.py`.
- **Architecture.** A U-Net-style **DenseNet encoder-decoder** the authors nickname
  **DenseElNet**, implemented as `DenseNet2D` in `models/RITnet_v3.py`. It has:
  - an encoder (`DenseNet_encoder`, dense down-blocks with avg-pool transitions),
  - a decoder (`DenseNet_decoder`) producing a **3-class segmentation map**
    (`out_c=3`, `models/RITnet_v3.py:177`),
  - a parallel **ellipse regression head** (`regressionModule`, `utils.py:616-667`)
    that outputs 10 numbers: pupil `(cx,cy,a,b,θ)` + iris `(cx,cy,a,b,θ)` in
    normalized coordinates.
  - Instance norm, LeakyReLU, single grayscale input channel (`in_c=1`).
- **RITnet lineage: yes.** The model files are literally named `models/RITnet_v1..v7.py`
  and the code/CLI refers to the model as `ritnet_v3` (`modelSummary.py:11,21`;
  inference hard-codes `model_dict['ritnet_v3']` at `evaluate_ellseg.py:281`). EllSeg is
  the same RIT (Rochester Institute of Technology) eye-segmentation line as RITnet, and
  a PyTorch **DeepVOG** re-implementation is also bundled (`models/deepvog_pytorch.py`)
  as a comparison baseline.
- **Paper + year.** Kothari, Chaudhary, Bailey, Pelz, Diaz, *"EllSeg: An Ellipse
  Segmentation Framework for Robust Gaze Tracking"*, arXiv:2007.09600, **2020**
  (published IEEE TVCG / IEEE VR 2021). BibTeX in `README.md:120-127`.
- **License.** **MIT** (`License.md`, © 2021 the five authors) — permissive, fine for
  research use and internal audit tooling.
- **Repo maturity note.** The README top line (`README.md:3`) says *"PLEASE USE OTHER
  INTERNAL REPOSITORY FOUND HERE"* pointing to a Bitbucket mirror. So this GitHub copy is
  a somewhat frozen public release; several README features are marked "Coming soon!"
  (image-mode inference, Pupil Labs integration, `--save_maps`).

---

## 2. Eye outputs (what it actually produces per frame)

EllSeg produces **all three** things we care about. Per frame, the inference core
`evaluate_ellseg_on_image()` (`evaluate_ellseg.py:98-172`) returns
`(seg_map, latent, pupil_ellipse, iris_ellipse)`:

1. **Segmentation MASK** — `seg_map`, a per-pixel class map with **3 classes**:
   - `0` = background (sclera/skin),
   - `1` = **iris**,
   - `2` = **pupil**.

   (Class meanings confirmed by the overlay code: `helperfunctions.py:521-522`
   `loc_iris = seg_map==1`, `loc_pupil = seg_map==2`.) Because of the EllSeg training
   objective, the pupil/iris regions are the **full elliptical discs** — so a pupil
   **binary mask** is directly available as `seg_map==2`, and it is intended to be
   valid even when the real pupil is partially occluded.

2. **ELLIPSE parameters** for pupil and iris — each a 5-vector
   **`[cx, cy, a, b, θ]`** = center-x, center-y, semi-axis a, semi-axis b, orientation
   (radians). Order confirmed by the draw call
   `draw.ellipse_perimeter(int(e[1]) /*row=cy*/, int(e[0]) /*col=cx*/, int(e[3]) /*b*/,
   int(e[2]) /*a*/, orientation=e[4])` at `helperfunctions.py:549-553`.
   The `--ellseg_ellipses` flag (`evaluate_ellseg.py:46`) selects **how** ellipses are
   obtained:
   - `--ellseg_ellipses=1`: **network-predicted ellipse** — center from the segmentation
     center-of-mass, axes+angle from the regression head, un-normalized via a homography
     `H` (`evaluate_ellseg.py:116-132`). *This is the occlusion-robust EllSeg mode.*
   - `--ellseg_ellipses=0`: **ElliFit on the mask** — classic ellipse fit to the mask
     boundary points, optionally with RANSAC outlier rejection (`--skip_ransac`)
     (`evaluate_ellseg.py:134-166`, `ElliFit`/`ransac` in `helperfunctions.py:209-310`).
   - `--ellseg_ellipses=-1` (**default**): returns `-1`-filled ellipses (mask only, no
     ellipse) — so **you must pass `--ellseg_ellipses=1` (or `0`) to get ellipses**.

3. **Pupil center** (and iris center) — available directly as the ellipse center
   `(cx, cy)`. In network mode the center is the softmax center-of-mass of the pupil
   channel (`get_seg2ptLoss`, `loss.py:16-37`), which is a sub-pixel estimate.

4. **Latent vector** — a bottleneck embedding is also returned (not needed for us).

**Per-video artifacts** (`evaluate_ellseg_per_video`, `evaluate_ellseg.py:197-272`):
an overlay video `*_ellseg.mp4` and a NumPy dict `*_pred.npy` mapping
`frame_index -> {'pupil': [cx,cy,a,b,θ], 'iris': [cx,cy,a,b,θ]}`. Missing/blank frames
are stored as `-1*np.ones(5,)` sentinels. (Mask H5 export `--save_maps` is stubbed —
"coming soon", `README.md:46`.)

**Net for us:** EllSeg is a *superset* of what GSAM2 gives us — it yields a pupil binary
mask **and** an occlusion-robust pupil ellipse (center/axes/angle) in one pass, which is
exactly the "ellipse notable even under occlusion" property called out in the goal.

---

## 3. Input assumptions and fit for EV-Eye DAVIS346

- **Grayscale / IR near-eye images: yes, native.** Input channel is 1
  (`DenseNet_encoder(in_c=1 ...)`). The video path forcibly converts frames to grayscale
  (`cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`, `evaluate_ellseg.py:233`). This is a
  purpose-built near-eye eye-tracking model — **no natural-image domain gap** the way
  GSAM2 has.
- **Expected resolution: 320×240 (W×H).** `preprocess_frame` resizes by **matching
  width** to 320 with Lanczos, then vertically pads/crops to height 240
  (`evaluate_ellseg.py:59-95`, called with `(240, 320)` at line 241). Height-alignment is
  **not** implemented (`sys.exit(...)` at line 91) — only width-alignment is supported.
- **Preprocessing: per-frame z-score.** `img = (img - img.mean())/img.std()`
  (`evaluate_ellseg.py:93`) — no fixed IR normalization constants, so it adapts to each
  frame's intensity. A dark-frame guard skips frames with `max()<20`
  (`evaluate_ellseg.py:235`).
- **Fit for EV-Eye DAVIS346 (346×260 grayscale near-eye).** Good geometric match:
  - 346×260 → resize width 346→320 (scale ≈0.925), height 260→240.5 → pads/crops the
    ~19 px difference. The built-in `rescale_to_original`
    (`evaluate_ellseg.py:175-194`) maps ellipses/masks **back to native 346×260**, so we
    can compare against EV-Eye/GSAM2 in original pixel coordinates.
  - **Concern — small pupil.** Our pupil is ~40–60 px in 346×260; after the 0.925 scale
    it is ~37–55 px, still well-resolved for a model trained on 320×240 near-eye imagery.
    Fine.
  - **Concern — full-frame vs eye-crop.** EllSeg's training sets are tight near-eye
    crops where the iris/pupil fill much of the frame. If EV-Eye APS frames are similarly
    framed (they are close-ups), fine; if there is a lot of surrounding face/background,
    accuracy may drop and an eye-crop pre-step would help. Worth a visual check on a few
    samples.

---

## 4. How to run inference end-to-end

### Entry point
- **Video mode (the working path): `evaluate_ellseg.py`.**
  ```
  python evaluate_ellseg.py --path2data=${DIR_OF_EYE_VIDEOS} \
         --loadfile=./weights/all.git_ok \
         --ellseg_ellipses=1        # 1 = EllSeg ellipse; 0 = ElliFit-on-mask; -1 = mask only
         [--eval_on_cpu=1]          # if no GPU
  ```
  It recursively globs `*.mp4` under `--path2data` (`evaluate_ellseg.py:288-292`) and
  writes `*_ellseg.mp4` + `*_pred.npy` next to each video. `--vid_ext` is documented but
  the glob is hard-coded to `.mp4`, so non-mp4 needs a tiny edit or a re-container step.
- **Image-folder mode: NOT provided.** README says *"Try it out on your eye images!
  Coming soon!"* (`README.md:52-53`). For our per-frame samples we would either
  (a) assemble frames into an mp4, or (b) reuse the three exposed building blocks
  (`preprocess_frame`, `evaluate_ellseg_on_image`, `rescale_to_original`) in a ~30-line
  loop over image files. This is the main integration cost (see §6).
- `test.py` / `train.py` / `runLocal.sh` are for the paper's dataset-based
  train/evaluate pipeline (need pre-built H5 "curriculum" objects under
  `curObjects/baseline`); **not** relevant to running on our raw frames.

### Pretrained weights — **present in-repo (key finding)**
- `weights/*.git_ok` are the actual **PyTorch checkpoints**, just renamed with a
  `.git_ok` suffix (the repo's `.gitignore` excludes `*.pt`, so they were committed under
  a non-ignored name). Verified: each is ~10.4 MB and the binary header is a Python
  pickle containing `state_dict` with keys like `enc.head.conv1.weight` — i.e. a
  `DenseNet2D`/`ritnet_v3` checkpoint (legacy pre-1.6 non-zip pickle, consistent with the
  pinned `pytorch=1.2.0`). They are real weights, **not** Git-LFS pointer stubs (repo has
  no `.gitattributes`/LFS).
- Available checkpoints (`ls weights/`): **`all.git_ok`** (trained on all datasets, the
  recommended one and the default in `evaluate_ellseg.py:38` and `args.py:43`), plus
  per-dataset `openeds`, `nvgaze`, `riteyes`, `LPW`, `Fuhl`, `pupilnet`.
- Loading: `netDict = torch.load(args.loadfile); model.load_state_dict(
  netDict['state_dict'], strict=True)` (`evaluate_ellseg.py:279-282`). `strict=True`
  means the checkpoint matches `ritnet_v3` exactly — no surgery needed.
- **So weights are NOT missing** — unlike many repos, this one is self-contained and can
  run offline today (given the right Python env).

### Training domains (affects domain match — §5)
Trained on **OpenEDS, NVGaze, RITEyes, LPW, Fuhl (ElSe+ExCuSe), PupilNet**, plus an
"all" combination (`README.md:16-24`). Relevant mix for us:
- **LPW, Fuhl, PupilNet** are **real IR head-mounted eye-camera** datasets → closest to
  EV-Eye's real near-IR APS frames.
- **NVGaze / RITEyes** are synthetic; **OpenEDS** is real but a specific HMD sensor.
- The **`all`** checkpoint blends real+synthetic and is the intended deployment model
  (`README.md:24`, `README.md:49`).

### Dependencies / framework / GPU
- From `requirements.txt` (a conda `--file` export, linux-64): **Python 3.6**,
  **PyTorch 1.2.0 + torchvision 0.4.0 + CUDA 9.2/cuDNN 7.6**, `opencv 3.4.2`, `numpy`,
  `scipy`, `scikit-image`, `scikit-learn`, `h5py`, `deepdish`, `matplotlib`, `tqdm`.
  Runtime scripts also import `kornia` (`create_meshgrid` in `loss.py`) — **not listed in
  `requirements.txt`**, so it must be added manually.
- **This env has no PyTorch installed** (import fails), so a dedicated env is required
  before running. The pinned stack is old (2019-era CUDA 9.2); on modern GPUs a newer
  PyTorch usually loads these legacy pickles fine, but that is untested here.
- **GPU:** default path uses `.cuda()` (`evaluate_ellseg.py:284-285`). **CPU is
  supported** via `--eval_on_cpu=1` (`evaluate_ellseg.py:42,201-204`) — the model is tiny
  (~2.6M params, 10 MB), so CPU inference on our sample set is feasible if slower.

---

## 5. Applicability as an independent audit source (like GSAM2)

**Yes — EllSeg is a strong independent auditor.** It is *methodologically independent*
from both of our existing sources: different training data, different architecture
(dense U-Net + ellipse regression vs. GSAM2's open-vocabulary detector + SAM2 vs. the
EV-Eye U-Net). Agreement across all three is meaningful cross-validation; disagreement
flags suspect frames.

**Strengths**
- **Purpose-built near-eye IR ellipse model** — no natural-image domain gap (unlike
  GroundingDINO+SAM2, which are trained on natural photos).
- **Directly outputs what we audit**: pupil binary mask (`seg_map==2`), pupil **ellipse**
  `[cx,cy,a,b,θ]`, and sub-pixel pupil center — one model, all three targets.
- **Occlusion-robust by design** (the headline EllSeg property): it segments the *full*
  pupil/iris disc and regresses the ellipse even when eyelids/lashes hide part of the
  pupil, whereas mask-then-fit methods (incl. GSAM2 → ellipse, or ElliFit on a partial
  mask) bias toward the visible arc. Good for a *disagreement detector* on hard frames.
- **Weights included + MIT license + built-in native-resolution rescale** → low external
  dependency risk, results directly comparable in 346×260 pixel space.
- **Deterministic, fast, batchable** — small model, no prompt engineering, unlike the
  GSAM2 prompt/threshold sensitivity.

**Concerns**
- **Training-domain match is only *partial*.** DAVIS346 APS is a specific event-camera
  APS sensor; EllSeg never saw it. The real-IR subsets (LPW/Fuhl/PupilNet) reduce risk,
  but expect some systematic offset vs. EV-Eye's own labels. Treat EllSeg as an
  *independent estimate*, not ground truth; **spot-check on a handful of EV-Eye frames**
  before trusting it at scale.
- **APS artifacts.** DAVIS346 APS frames can be low-contrast / noisier than
  conventional IR eye cameras; the per-frame z-score helps, but heavy noise could hurt
  the mask edges (and thus ElliFit mode more than network-ellipse mode).
- **Code maturity / env fragility.** Python 3.6 + Torch 1.2 + CUDA 9.2 pins are stale;
  `kornia` missing from requirements; image-mode inference is unimplemented; README
  points elsewhere as the "real" repo. All surmountable, but it is research code.
- **Full-frame framing assumption** (see §3): needs a visual check that EV-Eye APS crops
  resemble EllSeg's tight near-eye framing.
- **Iris.** EllSeg also gives an iris ellipse/mask "for free"; if we only care about
  pupil, iris output is a bonus we can ignore (or use as an extra sanity signal:
  pupil must sit inside iris).

**Recommended usage.** Run with `--ellseg_ellipses=1` (network ellipse, occlusion-robust)
as the primary audit signal; optionally also `--ellseg_ellipses=0` (ElliFit-on-mask) as
a *second internal* estimate — divergence between EllSeg's two modes is itself an
occlusion/quality flag. Compare EllSeg pupil center + ellipse + mask against GSAM2 and
the EV-Eye U-Net per frame; frames where all three agree are high-confidence, where
EllSeg disagrees with the U-Net pseudo-label are prime audit candidates.

---

## 6. Integration effort: **LOW–MEDIUM**

| Sub-task | Effort | Why |
|---|---|---|
| Get weights | **trivial** | Already in-repo (`weights/all.git_ok`), MIT, load with `strict=True`. |
| Build a runnable env | **medium** | Old pins (Py3.6 / Torch1.2 / CUDA9.2) + missing `kornia`. Easiest: a fresh conda env, likely with a *newer* Torch that still loads the legacy pickle; needs one validation run. This env currently has **no torch at all**. |
| Run on our frames | **low–medium** | No image-folder mode. Either (a) pack our per-sample frames into an mp4 and use `evaluate_ellseg.py` as-is, or (b) write a ~30-line driver that reuses `preprocess_frame` → `evaluate_ellseg_on_image` → `rescale_to_original` over an image list and dumps per-frame `{pupil,iris}` ellipses + `seg_map==2` masks. Option (b) is cleaner for a per-frame audit and avoids video re-encoding artifacts. |
| Wire outputs into our audit | **low** | Output is already `{frame: {'pupil':[cx,cy,a,b,θ], 'iris':[...]}}` in native resolution; add mask export (currently stubbed) by saving `seg_map` (or `seg_map==2`) in the same loop — one line. Maps 1:1 onto how we consume GSAM2 pupil masks/ellipses. |

**Bottom line.** Effort is **low-to-medium**, dominated by (1) standing up a compatible
PyTorch environment and (2) writing a small image-list inference wrapper (since only
video mode ships). Everything conceptually hard — a trained, occlusion-robust near-IR
pupil/iris ellipse+mask model — is already present and permissively licensed. This makes
EllSeg an attractive, genuinely independent third audit source next to GSAM2 and the
EV-Eye U-Net pseudo-labels.

---

### Key file references
- Inference entry point + preprocessing + output packing: `HBTXR/third/EllSeg/evaluate_ellseg.py`
- Model (DenseElNet / `ritnet_v3`, 3-class seg + ellipse head): `HBTXR/third/EllSeg/models/RITnet_v3.py`
- Ellipse math, ElliFit, RANSAC, overlay/class meanings, `getValidPoints`: `HBTXR/third/EllSeg/helperfunctions.py`
- Regression head + `get_predictions`: `HBTXR/third/EllSeg/utils.py` (`:64`, `:616`)
- Center-of-mass / seg-to-point loss (pupil center): `HBTXR/third/EllSeg/loss.py:16`
- Pretrained checkpoints (in-repo, ~10.4 MB each): `HBTXR/third/EllSeg/weights/*.git_ok` (`all` = default)
- Deps (conda export): `HBTXR/third/EllSeg/requirements.txt` · License: `HBTXR/third/EllSeg/License.md` (MIT)
- Usage / training domains / "coming soon" gaps: `HBTXR/third/EllSeg/README.md`
