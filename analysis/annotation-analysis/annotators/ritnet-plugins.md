# RITnet (Pupil Labs Core Plugins) — Annotator / Audit-Source Analysis

Analysis of `third/Pupil-Labs-Core-RITnet-Plugins` for use as an **independent pupil segmentation / audit-label source** on the EV-Eye dataset (DAVIS346 grayscale near-eye close-ups; pupil ~40–60 px dark disc in a 346×260 frame), alongside our GroundingDINO+SAM2 "GSAM2" audit and the EV-Eye U-Net.

Repo: `PerForm-Lab-RIT/Pupil-Labs-Core-RITnet-Plugins` (HEAD `44e106b`, branch `master`). Local path:
`/home/user/project/PRJXR-HBTXR/HBTXR/third/Pupil-Labs-Core-RITnet-Plugins`

---

## 1. What it is

This repo is a set of **Pupil Labs Core pupil-detector plugins** that swap Pupil Core's classic 2D detector for **deep semantic segmentation networks** — the original **RITnet** and its successors **EllSeg** / **EllSeg-v2**. It is NOT the upstream RITnet repo; it is an integration layer authored by the PerForm Lab (RIT), with the original RITnet source vendored under `ritnet/` and EllSeg under `ritnet/Ellseg/` and `ritnet/Ellseg_v2/`.

- **RITnet method / architecture**: A compact **DenseNet-style encoder–decoder (U-Net-shaped)** for dense per-pixel eye-part segmentation. Defined in `ritnet/densenet.py` (`DenseNet2D`): 5 dense down-blocks + 4 dense up-blocks with skip concatenation, `1×1` output conv to **4 classes**, LeakyReLU, dropout, `AvgPool` downsampling, bilinear upsampling. It is deliberately tiny — **248,900 parameters** (`ritnet/README.md` model summary) — designed for real-time inference. The header of `densenet.py` cites `ShusilDangi/DenseUNet-K` and calls it "a simplified version of DenseNet with U-NET architecture." So yes: **RITnet ≈ DenseNet-based U-Net**.
- **Output classes (RITnet)**: 4-class semantic mask — **background / sclera / iris / pupil** (`out_channels=4` in `DenseNet2D`; class index 3/pupil, 2/iris used downstream).
- **Paper + year**: RITnet, Chaudhary et al., **ICCVW 2019** — "RITnet: real-time semantic segmentation of the eye for gaze tracking." BibTeX in `ritnet/README.md`. EllSeg (Kothari et al., 2021) and EllSeg-v2 are the related follow-ups vendored here.
- **License**: **MIT** (`ritnet/License.md`, © 2019 Chaudhary, Kothari, Acharya, Dangi, Nair, Bailey, Kanan, Diaz, Pelz). EllSeg subdir has its own `ritnet/Ellseg/License.md`. Permissive — fine for internal research/audit use. The top-level repo has **no LICENSE file** of its own.
- **Relationship to Pupil Core**: Pupil Labs Core is the open-source head-mounted **gaze tracker**; it detects the pupil per eye-camera frame and fits a 3D eye model. Each plugin here subclasses Pupil Core's `Detector2DPlugin` and replaces the pupil detection step with a RITnet/EllSeg forward pass, then (usually) re-uses Pupil Core's own ellipse-fit on the network's pupil mask. The plugins **cannot be imported without Pupil Core on the path** — see §4.

**Four plugins** are provided (`plugins/`, selected at launch via `plugin_selector.py` using a `--plugin=...` CLI flag):

| Plugin file | `--plugin` key | Model | Weights file (in-repo) |
|---|---|---|---|
| `detector_2d_ritnet_pupil_plugin.py` | `ritnetpupil` | RITnet DenseNet 4-ch | `ritnet/ritnet_pupil.pkl` |
| `detector_2d_ritnet_bestmodel_plugin.py` | `bestmodel` | RITnet DenseNet 4-ch | `ritnet/best_model.pkl` |
| `detector_2d_ritnet_ellseg_pupil_plugin.py` | `ellseg` | EllSeg (RITnet_v2 enc/dec + ellipse regressor) | `ritnet/Ellseg/weights/all.git_ok` |
| `detector_2d_ritnet_ellsegv2_pupil_plugin.py` | `ellseg_v2` | EllSeg-v2 (`DenseElNet`) | `ritnet/Ellseg_v2/pretrained/pretrained.git_ok` |

---

## 2. Eye outputs

**Both a mask and a pupil ellipse/center are available** — this is the most useful property for us.

- **Semantic mask**: RITnet emits a 4-class argmax map (`get_predictions()` in `ritnet/utils.py` = `output.cpu().max(1)` → H×W index map). EllSeg emits a 3-class map (bg/iris/pupil; `RITnet_v2` decoder `out_c=3`). Helpers to turn this into a normalized/visual mask: `get_mask_from_PIL_image` / `get_mask_from_cv2_image` in `ritnet/image.py`. **The pupil class is directly extractable** (pupil = index 3 in vanilla RITnet, index 2 in EllSeg).
- **Pupil ellipse / center**: Two routes.
  1. **RANSAC ElliFit on the pupil mask** — `get_pupil_parameters()` (`ritnet/helperfunctions.py:547`) collects pupil pixels and fits an ellipse via `ransac(..., ElliFit, ...)`, returning `[cx, cy, major_r, minor_r, angle]`. Wrapped by `get_pupil_ellipse_from_PIL_image` in `ritnet/image.py`.
  2. **Direct ellipse regression** — EllSeg models have an `elReg` head that regresses pupil+iris ellipse params directly (`model.elReg(...)` in `image.py` / `ritnet/Ellseg/evaluate_ellseg.py`).
- **In the plugin flow**, the default (`customellipse=False`) path actually feeds the network's pupil mask back into **Pupil Core's own C++ ellipse detector** (`super().detect(framedup)` with `framedup.gray = mask`) to get the final ellipse — see `detector_2d_ritnet_pupil_plugin.py:147-168`. The EllSeg plugin also computes a bespoke **confidence** metric (support-pixel ratio + segmentation entropy) and can **save masks + overlays to disk** (`saveMaskAsImage`, `--save-masks`).

So the pupil **mask, center (cx,cy), axes, angle, diameter, and a confidence** are all obtainable. Center is what we'd primarily compare to GSAM2 / U-Net; the mask lets us compute IoU too.

---

## 3. Input assumptions

- **Modality**: near-eye **grayscale/IR** eye-camera images — exactly the RITnet/EllSeg training domain (OpenEDS, LPW, Fuhl, NVGaze, RITEyes, PupilNet). This is a **close domain match to EV-Eye's DAVIS346 near-eye near-IR frames**. Plugins consume `frame.gray` and everything runs on single-channel L images.
- **Preprocessing (vanilla RITnet)** — `process_PIL_image` in `ritnet/image.py` and `ritnet/dataset.py`: (1) **gamma correction γ=0.8** via LUT; (2) **CLAHE** `clipLimit=1.5, tileGridSize=(8,8)`; (3) **normalize mean 0.5 / std 0.5**. Confirmed identical in the paper's `README`.
- **Preprocessing (EllSeg)** — `preprocess_frame` in `ritnet/Ellseg/evaluate_ellseg.py:61`: **per-image standardization** `(img-mean)/std` (no CLAHE/gamma), then resize/pad — see resolution note.
- **Resolution — important nuance**:
  - **EllSeg / EllSeg-v2** explicitly resize to **320×240** (op_shape `(240,320)`), preserving aspect by matching width (`INTER_LANCZOS4`) then **vertically padding or cropping** to 240, tracking a `scale_shift` so ellipses can be mapped back to original coords (`rescale_to_original`). This handles **arbitrary input sizes cleanly, including EV-Eye 346×260** (346→320 width scale ≈0.925, height 260→240 after pad/crop). Good fit.
  - **Vanilla RITnet (DenseNet) path does NOT resize** — `process_PIL_image` runs `transforms.ToTensor` on the native frame and feeds it straight to the net. The `DenseNet2D` is fully convolutional so it will *run* at 346×260 (dims divisible enough for 4× avg-pool: 260/16≈16.25 → interpolate handles it), but it was **trained at ~640×400 / 320×240-class eye crops**, so a raw 346×260 near-eye frame may be **out of the trained scale regime** unless resized. The README model summary shows input tensors at 640×400. This is a **preprocessing-match risk** for the vanilla models specifically.
- **Note on `.pkl` (vanilla) input size**: the README summary is for a 640×400 model, but `ritnet_pupil.pkl`/`best_model.pkl` are 4-ch DenseNet and the code applies no fixed resize — you would want to **resize EV-Eye frames to the trained eye-crop scale** for best results, or prefer the EllSeg path which normalizes scale internally.

---

## 4. How to run

- **As shipped: requires the full Pupil Labs Core app.** Every plugin does `sys.path.append('.../pupil_src/shared_modules/pupil_detector_plugins')` and imports `visualizer_2d`, `pupil_detectors.DetectorBase`, `pupil_detector_plugins.*`, `methods.normalize`, `pyglui.ui` (top of each `plugins/*.py`). The documented workflow (`README.md`): install Pupil Core, drop this repo into `player_settings/plugins/`, fill an autorun CSV, run `autorun/run.sh <csv> <pupil_folder>` which launches Pupil Player with `--plugin=<key>`. GUI/`pyglui`/OpenGL are in the loop for the plugin path.
- **BUT the RITnet/EllSeg inference core is cleanly separable.** The heavy lifting lives in `ritnet/image.py`, `ritnet/models.py`, `ritnet/densenet.py`, `ritnet/utils.py`, `ritnet/helperfunctions.py`, and `ritnet/Ellseg/evaluate_ellseg.py` — **none of which import Pupil Core**. `ritnet/image.py::init_model()` + `get_mask_from_PIL_image()` / `get_pupil_ellipse_from_PIL_image()` are a self-contained mask→ellipse pipeline. For EllSeg: `preprocess_frame` → `evaluate_ellseg_on_image_GD(tensor, model)` → `rescale_to_original` returns `(seg_map, pupil_ellipse, iris_ellipse, seg_out)` with **no Pupil Core dependency**.
- **Standalone entry points already in-repo**: `ritnet/test.py` (batch test over a dataset dir), `ritnet/video.py` (runs a model over a video with overlays/metrics — imports only `ritnet/*` + `Ellseg/*`), `ritnet/Ellseg/evaluate_ellseg.py` (video/folder evaluation with `--loadfile`). These prove the net can run **without Pupil Core**.
- **Pretrained weights — PRESENT IN-REPO (not missing, no download needed, no LFS):**
  - `ritnet/best_model.pkl` (~1.02 MB), `ritnet/ritnet_pupil.pkl` (~1.02 MB), `ritnet/ritnet_400400.pkl` (~1.01 MB) — real PyTorch `state_dict` pickles (~249k params ≈ 1 MB, consistent).
  - `ritnet/Ellseg/weights/{all,Fuhl,LPW,nvgaze,openeds,pupilnet,riteyes}.git_ok` (~10.45 MB each) — **real PyTorch checkpoints** (verified: pickle magic + `state_dict` with `enc.head.conv1.weight` keys), despite the odd `.git_ok` extension. `all.git_ok` is the recommended all-datasets model and is what the EllSeg plugin loads.
  - `ritnet/Ellseg_v2/pretrained/pretrained.git_ok` (~12 MB) — EllSeg-v2 `DenseElNet` checkpoint.
  - Loaded via `torch.load(...)` then `model.load_state_dict(netDict['state_dict'], strict=True)` (EllSeg) or `model.load_state_dict(torch.load(path))` (vanilla).
- **Framework / deps**: **PyTorch + torchvision**, OpenCV (`cv2`), numpy, scikit-image (`skimage.measure`), scipy (`entropy`, `binary_closing`), matplotlib, PIL, tqdm. `ritnet/requirements.txt` is minimal/loose (lists `torch`, `cv2`, `PIL`, `os` as pip names — not directly pip-installable as written; treat as indicative). `ritnet/environment.yml` and `ritnet/Ellseg/requirements.txt` are fuller. No pinned single version; code style (Python-3.6 `.pyc`, `torch.load` without `weights_only`) suggests an **older PyTorch (≈1.x)** era; should still load on modern PyTorch with minor care.
- **GPU**: Plugins **hard-code `torch.device("cuda")`** (`USEGPU=True`, `model.cuda()`) — see `detector_2d_ritnet_pupil_plugin.py:35-43`, `ellseg` plugin `model.cuda()`. Standalone helpers accept `useGpu`/CPU (`init_model(devicestr=...)`, EllSeg `--eval_on_cpu`), so **CPU inference is possible** for the core, but the net is tiny so a GPU makes batch runs trivial and is recommended.

---

## 5. Applicability to our task (independent audit source)

**Yes — a standalone per-frame RITnet/EllSeg pupil mask + center can be extracted as an independent audit source, analogous to GSAM2.** The inference core is Pupil-Core-free and there are existing standalone scripts to model our harness on.

**Strengths**
- **Purpose-built for exactly this modality**: near-eye grayscale/IR pupil+iris segmentation — much closer to EV-Eye than a general open-vocabulary detector like GSAM2. Trained on multiple eye-tracking datasets (OpenEDS/LPW/Fuhl/NVGaze/RITEyes/PupilNet via the `all` EllSeg model).
- **Genuinely independent** from both our GSAM2 audit and the EV-Eye U-Net (different architecture lineage, different training data, different author group) → good triangulation value.
- **Produces mask AND ellipse/center AND a confidence** → supports both center-distance and mask-IoU comparisons, plus a built-in reject signal (EllSeg entropy/support-ratio confidence).
- **Weights are in-repo and permissively (MIT) licensed** — no download friction, no LFS, no missing-checkpoint problem.
- **Tiny + fast** (~249k params vanilla; EllSeg ~10 MB) → cheap to run over all EV-Eye frames.
- **EllSeg path resizes/pads to 320×240 internally and maps ellipses back to original coordinates**, so it directly ingests 346×260 EV-Eye frames without us reimplementing scale handling.

**Concerns**
- **Pupil Core coupling in the *plugins*** (not the core net): the four `plugins/*.py` are unusable without Pupil Core + pyglui + OpenGL, contain Windows-isms (hard-coded `\\` path separators, `source_path.rindex("\\")`), hard-coded CUDA, and per-eye flip logic tied to `g_pool.eye_id`. **Do not use the plugin classes**; lift the `ritnet/` core instead.
- **Preprocessing match (vanilla RITnet)**: the DenseNet path applies gamma+CLAHE+normalize but **no resize** — feeding native 346×260 may be off the trained scale. Prefer EllSeg (auto-resize) or add an explicit resize to the trained eye-crop scale for the `.pkl` models.
- **Code maturity**: research-grade. `ritnet/image.py` has leftover `plt.imshow` calls inside inference, commented `pdb`, mixed `.cpu().numpy()` vs `.numpy()` (a couple of `.numpy()` on possibly-GPU tensors in `get_pupil_ellipse_from_cv2_image` would need a `.cpu()`), and loose requirements. Expect light debugging when lifting it out.
- **`cv2.findContours` return-arity** in a few helpers (`get_area_perimiters_from_mask`, EllSeg plugin `_, contours, _`) assumes **OpenCV 3.x** (3-tuple). On OpenCV 4.x this must be adjusted to a 2-tuple. Minor but real.
- **3-class vs 4-class** difference between EllSeg (bg/iris/pupil) and vanilla RITnet (bg/sclera/iris/pupil): pupil is cleanly isolable in both, but index bookkeeping differs — mind the `channels`/index constants (`model_channel_dict` in `ritnet/models.py`).
- **DAVIS346 specifics**: EV-Eye frames are event-camera APS grayscale; contrast/illumination differ from the training sets. Expect a **domain gap** — RITnet/EllSeg is a strong *independent prior*, not ground truth. Validate on a labeled EV-Eye subset before trusting it as an audit oracle.

---

## 6. Integration effort

**Effort: Low–Medium (call it Medium)** to wire a standalone per-frame **pupil-center + pupil-mask producer** over our EV-Eye samples.

**Why it's not High**: the segmentation core **can be fully lifted out of the plugin** — it does not depend on Pupil Core. A minimal driver is essentially:

- **EllSeg (recommended, ~Low)**: `import` from `ritnet/Ellseg/` → build `model_dict['ritnet_v2']` (`ritnet/Ellseg/modelSummary.py`) → `torch.load('ritnet/Ellseg/weights/all.git_ok')['state_dict']` → for each frame: `preprocess_frame(gray, (240,320), align_width=True)` → `evaluate_ellseg_on_image_GD(tensor, model)` → `rescale_to_original(...)` → read `pupil_ellipse` (center = `[0],[1]`) and `seg_map==2` (pupil mask). Internal resize + coordinate mapping already handle 346×260. This is close to copy-adapt from the existing `evaluate_ellseg.py` and the plugin's `RITPupilDetector.detect`.
- **Vanilla RITnet (~Low–Med)**: `init_model('cuda'/'cpu', 'ritnet_pupil.pkl')` (`ritnet/image.py`) → `get_mask_from_PIL_image(frame, model, channels=4)` for the mask and `get_pupil_ellipse_from_PIL_image(frame, model)` for `[cx,cy,a,b,angle]`. Add an explicit resize to the trained scale for best accuracy.

**Why it's not trivial (the Medium part)**:
- Need to **relocate/import the `ritnet/` package** (it uses bare `from models import ...`, `from dataset import ...`, `sys.path` hacks) into our env — set `sys.path` or repackage.
- **Dependency/version pinning**: unpinned reqs; must nail down a working PyTorch + OpenCV combo. Fix the **OpenCV-4 `findContours` arity** and a couple of `.cpu()` omissions if we hit the ellipse helper.
- **Strip GUI/debug side-effects** (`plt.imshow`, `cv2.imshow`, disk-writes) from the copied inference functions.
- **Batching harness** over our sample manifest + output schema alignment to match how GSAM2 / U-Net results are stored (center x/y in original pixel coords, optional mask, confidence).
- **Validation pass** on a labeled EV-Eye subset to characterize the domain gap before using it as an audit signal.

Net: budget roughly a day to stand up an EllSeg-based standalone pupil-center/mask extractor over our frames; the weights and a near-drop-in inference path are already in the repo.

---

## Key file references

- Architecture (RITnet): `ritnet/densenet.py` (`DenseNet2D`, 4-class, 248,900 params)
- Model registry: `ritnet/models.py` (`model_dict`, `model_channel_dict`)
- Inference core (Pupil-Core-free): `ritnet/image.py` (`init_model`, `get_mask_from_PIL_image`, `get_pupil_ellipse_from_PIL_image`, `process_PIL_image`)
- Preprocessing / dataset: `ritnet/dataset.py` (gamma 0.8, CLAHE 1.5/(8,8), normalize 0.5/0.5)
- Mask→ellipse (RANSAC ElliFit): `ritnet/helperfunctions.py:547` (`get_pupil_parameters`), `ritnet/utils.py:186` (`get_predictions`)
- EllSeg inference: `ritnet/Ellseg/evaluate_ellseg.py` (`preprocess_frame` L61, `evaluate_ellseg_on_image_GD` L100, `rescale_to_original` L232); model `ritnet/Ellseg/models/RITnet_v2.py`, registry `ritnet/Ellseg/modelSummary.py`
- Weights: `ritnet/*.pkl`; `ritnet/Ellseg/weights/all.git_ok` (+per-dataset); `ritnet/Ellseg_v2/pretrained/pretrained.git_ok`
- Plugins (Pupil-Core-coupled, do NOT lift): `plugins/detector_2d_ritnet_pupil_plugin.py`, `..._bestmodel_plugin.py`, `..._ellseg_pupil_plugin.py`, `..._ellsegv2_pupil_plugin.py`; selector `plugin_selector.py`
- Standalone runners to model on: `ritnet/test.py`, `ritnet/video.py`, `ritnet/Ellseg/evaluate_ellseg.py`
- License: `ritnet/License.md` (MIT); paper: `ritnet/README.md` (Chaudhary et al., ICCVW 2019)
