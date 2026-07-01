# X-AnyLabeling — MASK/Segmentation Model Survey for Near-IR Pupil Annotation

**Scope:** Enumerate and assess the segmentation ("MASK") models bundled in the vendored X-AnyLabeling checkout, and judge which could serve as an *independent* pupil-mask annotator / audit source for grayscale near-IR eye images (EV-Eye / DAVIS346, 346×260, pupil ~40–60 px dark disc), analogous to our current GroundingDINO+SAM2 pipeline.

**Repo analyzed:** `/home/user/project/PRJXR-HBTXR/HBTXR/third/X-AnyLabeling`
**Version:** `4.0.0-beta.11` (`anylabeling/app_info.py`), Python 3.11+, PyQt6.
**License:** GPL-3.0 (`LICENSE` = GNU GPL v3; `README.md` "License" section states GPL-3.0). *Note:* the README badge image says "LGPL v3", but the actual `LICENSE` file and prose are GPL-3.0 — treat this as **GPL-3.0** (copyleft; matters if we redistribute).
**Upstream:** CVHub520/X-AnyLabeling (fork/descendant of vietanhdev/AnyLabeling + LabelMe lineage).

---

## 1. What X-AnyLabeling is

A desktop **GUI auto-labeling tool** ("Advanced Auto Labeling Solution") for multi-modal CV annotation. It wraps an AI inference engine (ONNX Runtime / TensorRT / OpenCV-DNN backends) behind a LabelMe-style Qt UI, so a human can prompt a model (click a point, drag a box, or type a text phrase) and get shapes back. It handles classification, detection, **segmentation**, pose, tracking, OCR, VQA, grounding, matting, depth, etc.

- **Entry point:** `xanylabeling = "anylabeling.app:main"` (`pyproject.toml` `[project.scripts]`); main module `anylabeling/app.py`. Launch: `xanylabeling` (GUI), or `python -m anylabeling.app` from source.
- **Install:** `uv pip install --pre "x-anylabeling-cvhub[cpu]"` (or `[gpu]`, `[gpu-cu11]`); from source `uv pip install -e ".[cpu]"` (see `docs/en/get_started.md`).
- **Auto-labeling panel:** open with `Ctrl+A` / the `AI` button; pick a model from a dropdown; models with a `http(s)` weight URL auto-download on first use.
- **Optional remote inference:** a separate `X-AnyLabeling-Server` (PyTorch backend) can host heavier models (notably full SAM3 with visual prompting); the client talks to it via the `Remote-Server` model entry.

### Model registry mechanics (how "the model list" actually works)
- Master registry: **`anylabeling/configs/models.yaml`** — a flat list of `{model_name, config_file: ":/<name>.yaml"}` entries (loaded at startup by `ModelManager.load_model_configs`, `anylabeling/services/auto_labeling/model_manager.py:68`). The `:/` prefix resolves to `anylabeling/configs/auto_labeling/<name>.yaml`.
- Per-model YAML configs live in **`anylabeling/configs/auto_labeling/*.yaml`** (191 configs tracked in this checkout). Each has a `type:` (dispatches to a Python model class), a `display_name:`, and weight URLs/paths.
- Each `type:` maps to an implementation in **`anylabeling/services/auto_labeling/*.py`** (e.g. `type: segment_anything` → `segment_anything.py`). Model behavior flags (which models accept marks/text/conf, which can batch, etc.) are declared as lists in **`anylabeling/services/auto_labeling/__init__.py`**.

> **Actual vs. unclear:** `docs/en/model_zoo.md` lists a *superset* of the upstream project (e.g. `medsam_vit_b.yaml`, `yolov8n_efficientvit_sam_l0_vit_h.yaml`, `hyper-yolon-seg.yaml` are referenced in the doc **but the config YAMLs are not present** in this checkout's `auto_labeling/` dir and not in `models.yaml`). What is actually runnable here = what's in `models.yaml` + `auto_labeling/*.yaml`. The doc is aspirational/upstream.

---

## 2 & 3. MASK / segmentation-capable models (present in this checkout)

Two broad families produce masks:
- **(A) Promptable "Segment Anything" models** — interactive; you give point/box (or text) prompts and get a mask per object. These are the true "annotator" tools.
- **(B) Closed-vocabulary instance/semantic segmenters** — YOLO-seg / RF-DETR-seg / Hyper-YOLO-seg; run automatically but only over their fixed (COCO) class list; useless for "pupil" unless retrained.

All mask-producing models emit either a **polygon** or a **rectangle** (default: polygon). Masks are converted to polygons in `post_process()` via `cv2.findContours` + `cv2.approxPolyDP` (an adjustable "mask fineness" epsilon slider). Polygons are the interchange primitive that feeds all exporters (see §4).

### (A) Segment-Anything family (interactive / promptable) — the relevant ones

| Model (display_name) | `type:` | Config YAML | Prompt(s) | Output | Weights |
| --- | --- | --- | --- | --- | --- |
| Segment Anything (ViT-B/L/H, ±quant) | `segment_anything` | `segment_anything_vit_{b,l,h}[_quant].yaml` | point, box | polygon/rect | auto-download (GitHub release ONNX) |
| **Segment Anything 2.1** (Tiny/Small/Base/Large) | `segment_anything_2` | `sam2_hiera_{tiny,small,base,large}.yaml` | point, box | polygon/rect | auto-download |
| SAM 2 Video (Tiny…Large) | `segment_anything_2_video` | `sam2_hiera_*_video.yaml` | point, box + **text** (batch), tracks across frames | polygon/rect | auto-download |
| **Segment Anything 3 (ViT-H)** | `segment_anything_3` | `sam3_vit_h.yaml` | **text** (client-side ONNX); text+box/point via server | polygon/rect | auto-download (6 files incl. `.onnx.data`, ~5 GB total) |
| **SAM-HQ** (ViT-B/L/H, ±quant) | `sam_hq` | `sam_hq_vit_{b,l,h}[_quant].yaml` | point, box | polygon/rect | auto-download |
| **MobileSAM** (ViT-H decoder) | `segment_anything` | `mobile_sam_vit_h.yaml` | point, box | polygon/rect | auto-download (~27 MB enc) |
| **EfficientViT-SAM** (l0, l1) | `efficientvit_sam` | `efficientvit_sam_l{0,1}_vit_h.yaml` | point, box | polygon/rect | auto-download |
| **EdgeSAM** | `edge_sam` | `edge_sam.yaml` | point, box | polygon/rect | auto-download (~21+18 MB) |
| EdgeSAM + CN-CLIP | `edge_sam` | `edge_sam_with_chinese_clip.yaml` | point, box, **Chinese text** (CLIP) | polygon/rect | auto-download (adds CN-CLIP onnx + extra_file) |
| **SAM-Med2D** (ViT-B, 256px) | `sam_med2d` | `sam_med2d_vit_b.yaml` | point, box | polygon/rect | auto-download (~1 GB enc) |

### (A′) Grounded / text-driven SAM combos (closest to our GroundingDINO+SAM2 pipeline)

| Model | `type:` | Config YAML | Prompt | Output | Weights |
| --- | --- | --- | --- | --- | --- |
| **GroundingSAM2** (GroundingDINO-SwinT + SAM 2.1-Large) | `grounding_sam2` | `groundingdino_swint_sam2_large.yaml` | **text** phrase (+ optional box/point marks) | polygon/rect | auto-download (GDINO-SwinT-quant + SAM2.1-L enc/dec) |
| GroundingSAM (GroundingDINO-SwinB fused + HQ-SAM ViT-L) | `grounding_sam` | `groundingdino_swinb_attn_fuse_sam_hq_vit_l_quant.yaml` | **text** phrase | polygon/rect | auto-download |
| Open Vision (BERT + SAM 2.1-Large) | `open_vision` | `open_vision.yaml` | **text** (counting/grounding) | polygon/rect | mixed: SAM2 ONNX auto-download, but needs `bert-base-uncased` **fetched manually** (HF path in config) + a `.pth` |
| YOLOv8s-SAM2 | `yolov8_sam2` | `yolov8s_sam2_hiera_base.yaml` | automatic (YOLOv8 boxes → SAM2 masks), COCO classes only | polygon | auto-download |
| YOLOv5s-MobileSAM | `yolov5_sam` | `yolov5s_mobile_sam_vit_h.yaml` | automatic (YOLOv5 boxes → MobileSAM), COCO only | polygon | auto-download |
| GECO (SAM-HQ ViT-H) | `geco` | `geco_sam_hq_vit_h.yaml` | exemplar box (few-shot counting) → masks | polygon | auto-download (+`.bin` encoder data) |
| SAM 3 (Video) | `segment_anything_3` (video ex.) | `examples/interactive_video_object_segmentation/sam3` | text / visual, tracked | polygon | server or ONNX |

### (B) Closed-vocabulary instance/semantic segmenters (mask output, but fixed classes)

`type: yolov8_seg` / `yolov5_seg` / `yolo11_seg` / `yolo26_seg` / `rfdetr_seg` / `hyper_yolo_seg`; configs `yolov8{n,s,m,l,x}_seg.yaml`, `yolov5s_seg.yaml`, `yolo11s_seg.yaml`, `yolo26s_seg.yaml`, `hyper_yolos_seg.yaml`, `rfdetr_seg_preview.yaml` (+ `*_seg_botsort`/`*_seg_bytetrack`/`*_seg_tracktrack` tracking variants). All are **COCO-trained** (80 classes; no "eye"/"pupil"). Output: polygons + boxes automatically. **Not usable for pupil without training a custom head** — but see §5 for the training path.

### (C) Matting (foreground alpha, not object masks)
RMBG v1.4 / v2.0 (`type: rmbg`, `rmbg_v14.yaml`, `rmbg_v20[_quant].yaml`) — background removal / salient-foreground alpha matte. Not a pupil segmenter (would grab the whole eye/face region), but noted for completeness since it is "MASK-like".

### Prompt / capability declarations (source of truth: `anylabeling/services/auto_labeling/__init__.py`)
- **Point/box interactive ("marks") models** (`_AUTO_LABELING_MARKS_MODELS`): `segment_anything`, `segment_anything_2`, `segment_anything_3`, `segment_anything_2_video`, `sam_med2d`, `sam_hq`, `yolov5_sam`, `efficientvit_sam`, `grounding_sam`, `grounding_sam2`, `open_vision`, `edge_sam`, `florence2`, `geco`, `yoloe`.
- **Text-prompt + batch-capable ("one-click over a folder")** (`_BATCH_PROCESSING_TEXT_PROMPT_MODELS`): `grounding_dino`, `grounding_sam`, `grounding_sam2`, `segment_anything_3`, `yoloe`, `remote_server`.
- **Interactive-only, cannot batch** (`_BATCH_PROCESSING_INVALID_MODELS`): `segment_anything`, `segment_anything_2`, `sam_med2d`, `sam_hq`, `efficientvit_sam`, `edge_sam`, `open_vision`, `geco` — i.e. plain SAMs require a human prompt per image and have no folder-wide auto mode.
- **Mask-fineness (polygon epsilon) slider models** (`_AUTO_LABELING_MASK_FINENESS_MODELS`): all the SAM family + `grounding_sam(2)` + `rfdetr_seg`.

---

## 4. Mask annotation workflow & export formats

### Workflow (interactive SAM)
1. Open image/folder in the GUI; open auto-label panel (`Ctrl+A`).
2. Select a SAM-family model (weights auto-download on first pick).
3. Prompt: `+Point`/`-Point` (positive/negative clicks) or `+Rect`/`-Rect` (box), or type a phrase for grounded/SAM3 models. SAM2/SAM-HQ etc. re-run on each new mark.
4. The predicted mask is converted to a **polygon** (fineness slider = `approxPolyDP` epsilon); choose `Polygon` or `Rectangle` output mode. Confirm/label the shape (`f` to finish an object).
5. Repeat per object; edit vertices (eraser tool, intent-aware vertex selection, per-shape lock all exist as of the 2026-06 changelog).
6. Save → produces LabelMe-style per-image JSON (the native "xlabel" format holding `shapes[]` with `points`, `shape_type`, `label`).

### Export / import formats (`anylabeling/views/labeling/label_converter.py`, also CLI `xanylabeling convert`)
Segmentation-relevant exporters (from native polygon JSON):
- **COCO** instance seg — `custom_to_coco` (`label_converter.py:1489`; writes `segmentation` polygons + bbox + area).
- **YOLO-seg** (polygon txt) — `custom_to_yolo` (`:1209`, seg mode).
- **VOC** — `custom_to_voc` (`:1407`; primarily boxes).
- **Binary/semantic MASK PNG** — `custom_to_mask` (`:1770`): rasterizes polygon shapes to a **grayscale class-indexed PNG** or **RGB color PNG** via `cv2.fillPoly`, using a label→color mapping table. *Only `shape_type == "polygon"` shapes are rendered* — so SAM polygon output flows straight to mask PNGs. This is the direct route to per-frame pupil mask PNGs.
- Also: DOTA, MOT/MOTS, and the reverse importers (`coco_to_custom`, `mask_to_custom`, `yolo_to_custom`, `voc_to_custom`).
- README feature list confirms import/export: `COCO, VOC, YOLO, DOTA, MOT, MASK, PPOCR, MMGD, VLM-R1, ShareGPT`.

### Weight acquisition
`Model.get_model_abs_path` (`anylabeling/services/auto_labeling/model.py:213`): if a config field is an `http(s)` URL, it downloads (with retry + integrity check) into `~/xanylabeling_data/models/<model_name>/`. **All SAM/SAM2/SAM3/SAM-HQ/MobileSAM/EfficientViT-SAM/EdgeSAM/SAM-Med2D/GroundingSAM(2) weights are `http(s)` GitHub-release ONNX → auto-download.** Exceptions needing manual fetch: `open_vision` (`bert-base-uncased` local path + `.pth`), and any server-hosted PyTorch model (needs X-AnyLabeling-Server). Mirrors: GitHub releases + Baidu + (some) HuggingFace/ModelScope (`docs/en/model_zoo.md`).

---

## 5. Usability as an INDEPENDENT pupil-mask annotator / audit source

**Goal recap:** produce (or cross-check) pupil masks on grayscale near-IR DAVIS346 crops, ideally with the same "text prompt → box → mask" ergonomics as our GroundingDINO+SAM2 pipeline (`third/Grounded-SAM-2`), and export to mask PNG / COCO for auditing.

### Ranked candidates

1. **GroundingSAM2** (`grounding_sam2`, `groundingdino_swint_sam2_large.yaml`) — **best direct analog to our existing pipeline.** It *is* GroundingDINO-SwinT + SAM 2.1-Large, text-prompted (`box_threshold: 0.3`, `text_threshold: 0.25`), batchable over a folder, mask/polygon output, all weights auto-download. As an **independent audit source it is only weakly independent** from our GroundingDINO+SAM2 pipeline (same two model families, though different weights: quantized SwinT ONNX + ONNX-exported SAM2.1-L vs. our PyTorch stack). Good for a fast second opinion / regression check; not a truly orthogonal method.

2. **SAM 3 (ViT-H)** (`segment_anything_3`, `sam3_vit_h.yaml`) — text-grounded ("segment all instances of a concept"), batchable, polygon output, auto-download. **Genuinely more independent** (newer architecture than our stack). Caveats: client-side ONNX is **text-prompt-only** (no box/point) and **slow** (full ViT-H encoder on CPU/GPU, several sec/image; ~5 GB weights) per `examples/grounding/sam3/README.md`; box/point visual prompting needs the PyTorch **X-AnyLabeling-Server**. Overlaps with our `third/SAM3-I` — so as an "independent annotator" it duplicates something we already have, just inside a GUI.

3. **Plain SAM 2.1 / SAM-HQ / MobileSAM / EfficientViT-SAM / EdgeSAM** (interactive point/box) — **strongest domain-gap robustness for the pupil specifically.** A single positive click on the dark pupil disc, or a tight box, is a very easy prompt for SAM; these are class-agnostic and don't rely on any text/semantic prior, so grayscale-IR domain gap is minimal (SAM segments by contrast/edges, and the pupil is a high-contrast dark blob). **SAM-HQ** is attractive for crisp small-object boundaries (relevant at 40–60 px). **Downside:** they are `_BATCH_PROCESSING_INVALID_MODELS` — **no folder-wide auto mode**; every frame needs a human click. That makes them excellent for a **manual audit / gold-standard spot-check on a sample of frames**, but not for bulk auto-annotation of a video.

4. **SAM-Med2D** (`sam_med2d`) — SAM fine-tuned on medical 2D images; also interactive point/box only, 256×256 input. Its medical-imaging (often grayscale) fine-tuning *might* transfer slightly better to IR eye texture than vanilla SAM, but there's **no evidence in-repo** and it's still per-frame interactive. Speculative; would need a quick empirical test.

5. **YOLOv8-SAM2 / YOLOv5-MobileSAM / YOLO*-seg / RF-DETR-seg** — **not usable out-of-the-box** (COCO classes, no pupil/eye). Only relevant if we train a custom pupil detector/segmenter — which X-AnyLabeling *does* support via its Ultralytics auto-training panel (`examples/training/ultralytics`) and custom-model loading (`docs/en/custom_model.md`). That's a separate workstream, not an off-the-shelf annotator.

6. **RMBG matting** — not appropriate (foreground alpha of whole scene, not the pupil).

### Concerns specific to grayscale near-IR pupil
- **Domain gap (text-grounded models):** GroundingDINO / SAM3 / YOLOE text encoders were trained on natural RGB. "pupil" / "eye pupil" as a phrase on a low-contrast grayscale IR close-up may under-fire or grab the whole iris/eye. Prompt engineering ("dark circle", "black disc", "pupil") + lowering `box_threshold`/`conf_threshold` (both are runtime-adjustable in the GUI/config) will likely be needed. This is the same risk we already manage in our GroundingDINO+SAM2 pipeline.
- **Domain gap (interactive SAMs):** minimal. Contrast-based, class-agnostic; a dark pupil on a lighter sclera/skin is close to an ideal SAM target. Main risk is over-segmentation into the iris if the IR contrast between pupil and dark iris is low — mitigated by negative clicks or a tight box.
- **Small object / resolution:** 40–60 px pupil in 346×260. SAM encoders internally resize to 1024; tiny masks can get ragged. SAM-HQ (HQ token) and the mask-fineness slider help. `sam_med2d` runs at 256px (closer to native), which may actually suit these small crops.
- **Grayscale input:** all these ONNX models expect 3-channel; the tool loads images as RGB, so a grayscale IR frame becomes a replicated-3-channel RGB automatically — fine, no code change needed.
- **Text-prompt availability:** confirmed for `grounding_sam2`, `grounding_sam`, `segment_anything_3`, `yoloe`, `edge_sam_with_chinese_clip` (Chinese only) — see `__init__.py` lists and per-model `preprocess(..., text_prompt)` (`grounding_sam2.py:166`, `segment_anything_3.py:143` splits on `.`/`,`). Plain SAM/SAM2/SAM-HQ/MobileSAM/EfficientViT/EdgeSAM/SAM-Med2D have **no text path** — point/box only.
- **Independence for auditing:** For a *truly independent* mask audit vs. our GroundingDINO+SAM2 outputs, the best choices in-repo are **(a) an interactive SAM-HQ / SAM2 with human box/click** (different, human-in-the-loop signal), or **(b) a custom-trained YOLO*-seg pupil model** (fully orthogonal, automatic). GroundingSAM2/SAM3 give speed but share model DNA with what we're auditing.

---

## 6. How to run + key file paths

**Run (GUI):**
```bash
# from source, in this checkout
uv pip install -e ".[cpu]"     # or [gpu] / [gpu-cu11]
xanylabeling                    # == anylabeling.app:main ; or: python -m anylabeling.app
# open image/folder → Ctrl+A → pick model → prompt → Save → convert/export
```
CLI conversion (headless mask/COCO export) is available: `xanylabeling convert --task <task> ...` (see `docs/en/cli.md`).

**Key paths (all under `/home/user/project/PRJXR-HBTXR/HBTXR/third/X-AnyLabeling`):**
- Master model registry: `anylabeling/configs/models.yaml`
- Per-model configs: `anylabeling/configs/auto_labeling/*.yaml`
  - SAM: `segment_anything_vit_{b,l,h}[_quant].yaml`, `mobile_sam_vit_h.yaml`
  - SAM2: `sam2_hiera_{tiny,small,base,large}[_video].yaml`
  - SAM3: `sam3_vit_h.yaml`
  - SAM-HQ: `sam_hq_vit_{b,l,h}[_quant].yaml`; SAM-Med2D: `sam_med2d_vit_b.yaml`
  - EfficientViT-SAM: `efficientvit_sam_l{0,1}_vit_h.yaml`; EdgeSAM: `edge_sam.yaml`, `edge_sam_with_chinese_clip.yaml`
  - Grounded combos: `groundingdino_swint_sam2_large.yaml`, `groundingdino_swinb_attn_fuse_sam_hq_vit_l_quant.yaml`, `open_vision.yaml`
  - Closed-vocab seg: `yolov8{n,s,m,l,x}_seg.yaml`, `yolo11s_seg.yaml`, `yolo26s_seg.yaml`, `hyper_yolos_seg.yaml`, `rfdetr_seg_preview.yaml`
- Model implementations: `anylabeling/services/auto_labeling/{segment_anything,segment_anything_2,segment_anything_3,sam_hq,sam_med2d,edge_sam,efficientvit_sam,grounding_sam,grounding_sam2,rmbg}.py`
- Capability/flag lists (marks/text/batch/fineness): `anylabeling/services/auto_labeling/__init__.py`
- Registry loader / weight download: `anylabeling/services/auto_labeling/model_manager.py` (`load_model_configs`, ~L68), `anylabeling/services/auto_labeling/model.py` (`get_model_abs_path`, L213)
- Export/convert: `anylabeling/views/labeling/label_converter.py` (`custom_to_mask` L1770, `custom_to_coco` L1489, `custom_to_yolo` L1209, `custom_to_voc` L1407)
- Entry point: `anylabeling/app.py` (`main`), declared in `pyproject.toml` `[project.scripts]`
- Docs: `README.md` (model library table), `docs/en/model_zoo.md` (weights/links — **superset of what's shipped**), `docs/en/get_started.md`, `docs/en/user_guide.md`, `docs/en/custom_model.md`, `examples/segmentation/`, `examples/grounding/sam3/README.md`

---

## Summary table (segmentation models actually shipped in this checkout)

| Model | Type | Prompt | Auto-batch? | Text? | Weights | Pupil-IR verdict |
| --- | --- | --- | --- | --- | --- | --- |
| SAM (ViT-B/L/H ±q) | `segment_anything` | point/box | No | No | auto | Good manual spot-check; class-agnostic, low domain gap |
| MobileSAM | `segment_anything` | point/box | No | No | auto | Lightweight manual audit |
| SAM 2.1 (T/S/B/L) | `segment_anything_2` | point/box | No | No | auto | Good interactive; strong masks |
| SAM 2 Video | `segment_anything_2_video` | point/box (+text batch) | Yes(video) | Partial | auto | Video-track pupil across frames (promising) |
| **SAM 3 (ViT-H)** | `segment_anything_3` | text (client) / +box(server) | Yes | **Yes** | auto (~5 GB) | Independent-ish; text-only+slow on client; overlaps our SAM3-I |
| **SAM-HQ (B/L/H ±q)** | `sam_hq` | point/box | No | No | auto | Best boundary quality for tiny pupil; manual |
| SAM-Med2D (B) | `sam_med2d` | point/box | No | No | auto (~1 GB) | Medical-tuned; maybe better IR transfer (unverified) |
| EfficientViT-SAM (l0/l1) | `efficientvit_sam` | point/box | No | No | auto | Fast interactive alt |
| EdgeSAM | `edge_sam` | point/box | No | No | auto | Very fast interactive |
| EdgeSAM+CN-CLIP | `edge_sam` | point/box/**zh-text** | No | zh only | auto | Text is Chinese-only; niche |
| **GroundingSAM2** | `grounding_sam2` | **text**(+marks) | **Yes** | **Yes** | auto | Closest analog to our pipeline; weak independence |
| GroundingSAM (HQ) | `grounding_sam` | **text** | Yes | Yes | auto | Text→HQ-SAM mask; SwinB heavier |
| Open Vision | `open_vision` | text | No | Yes | partial (BERT manual) | Counting-oriented; setup friction |
| YOLOv8-SAM2 | `yolov8_sam2` | auto (COCO) | Yes | No | auto | COCO only — no pupil |
| YOLO*-seg / RF-DETR-seg / Hyper-YOLO-seg | `*_seg` | auto (COCO) | Yes | No | auto | COCO only; needs custom training for pupil |
| RMBG 1.4/2.0 | `rmbg` | auto | Yes | No | auto | Matting, not pupil |

**Bottom line:** For *bulk auto* pupil masking with our existing ergonomics, **GroundingSAM2** (and **SAM 3** for a fresher-but-overlapping second opinion) are the shipped, text-promptable, auto-download options — but both share model DNA with (GroundingSAM2) or duplicate (SAM3) our current stack, so they are convenient rather than strongly *independent* audit sources. For a *genuinely independent, high-quality* pupil mask on a **sample of frames**, the interactive **SAM-HQ / SAM 2.1** (single click or tight box) is the most robust to the grayscale-IR domain gap and the best gold-standard/audit tool — at the cost of one human interaction per frame (no folder-wide auto mode). A **custom-trained YOLO-seg** (via the tool's Ultralytics training path) is the only shipped route to a fully automatic, orthogonal pupil segmenter. All mask output → polygons → exportable to **COCO / YOLO-seg / grayscale or RGB mask PNG** via `label_converter.py`, matching our audit needs.
