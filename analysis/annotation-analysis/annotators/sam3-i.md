# SAM3-I: Segment Anything with Instructions — Audit-Source Assessment

**Scope.** Assessment of the local repo at `third/SAM3-I` as a candidate *independent* auto-annotator / audit source for near-IR **pupil** mask annotation on the EV-Eye dataset (DAVIS346 grayscale near-eye close-ups; pupil = dark disc), to compare against our current GroundingDINO+SAM2 ("GSAM2") pipeline.

**Bottom line up front.** SAM3-I is **instruction (free-text) tuned segmentation built directly on Meta's SAM 3** via lightweight adapters. It is a *research training/eval codebase*, not a turnkey annotator: there is **no single-image demo**, the **fine-tuned checkpoint is an external Google-Drive download (not in-repo)**, and the **base SAM 3 weights are auto-pulled from the gated HuggingFace repo `facebook/sam3`**. It could *in principle* be prompted with `"the black pupil"`, but there is a large domain gap (trained purely on RGB COCO/ReasonSeg referring/reasoning data), unresolved licensing (no LICENSE file present), and non-trivial setup. As an audit source it is a **higher-effort, lower-certainty** option than GSAM2 today.

---

## 1. What SAM3-I actually is

### 1.1 Identity and origin
- The repo is the **"official repository for the paper *SAM3-I: Segment Anything with Instructions*"** (`README.md:1-3`), authored by Li Jingjing et al., cited as ACL 2026 (`README.md:196-203`). Git remote: `https://github.com/debby-0527/SAM3-I.git` (an academic third-party repo, **not** a Meta/`facebookresearch` repo).
- It is **built on top of Meta SAM 3**, not a from-scratch model. The packaged library is literally named `sam3` and describes itself as "SAM3 (Segment Anything Model 3) implementation" (`sam3/pyproject.toml:6-8`); source-file headers read `Copyright (c) Meta Platforms, Inc.` throughout, and `pyproject` URLs point to `github.com/facebookresearch/sam3` (`sam3/pyproject.toml:82-84`). SAM3-I = **SAM 3 backbone (frozen) + instruction-tuning adapters**.

### 1.2 The "with Instructions" concept — how it differs from SAM / SAM2 / SAM 3
The vanilla SAM family and their prompt types, for context:
- **SAM (SAM 1):** promptable with **geometric prompts** — points / boxes / masks — on a single image, class-agnostic.
- **SAM 2:** SAM 1 + memory for **video** tracking; still geometric prompts.
- **SAM 3:** adds **"promptable concept segmentation"** — short **noun-phrase / concept text prompts** and exemplars ("segment all instances of *X*"), returning *all* matching instances.

**SAM3-I's contribution** is to extend SAM 3 from short *concept* prompts to full **natural-language *instructions*** — referring expressions and, crucially, **reasoning expressions that do not name the target**. The code defines a three-tier text taxonomy used everywhere (`scripts/inference.py:146-177`, model forward `sam3/sam3/model/sam3_image.py:608-732`):

| Tier | Field | Nature | Example (from `predictions.json`) |
|---|---|---|---|
| **concept** | `concept` | bare noun / category (SAM-3-style) | `"helmet"` |
| **simple** | `simple_query` | referring expression that **contains** the target term | `"Can you identify the two helmets worn by the riders on the black and brown horses?"` |
| **complex** | `complex_query` | reasoning expression that **omits** the target term (functional/contextual) | `"Locate the head protection devices on the riders of the black and brown horses."` |

So relative to SAM 3, SAM3-I = **SAM 3 + instruction understanding (referring + reasoning segmentation)**. The instruction is *free text*; it is **not** in-context image exemplars, and at the instruction level it is **not** points/boxes.

### 1.3 How it is implemented (architecture)
Built entirely in `sam3/sam3/model_builder.py`. The model (`build_sam3_image_model`) is standard SAM 3:
- **Vision backbone:** `ViT` — img_size 1008, patch 14, embed_dim **1024, depth 32, 16 heads**, RoPE + windowed attention (i.e., a ViT-L / "Perception-Encoder"-class trunk) (`model_builder.py:163-172`).
- **Text encoder:** CLIP-style BPE tokenizer (`bpe_simple_vocab_16e6.txt.gz`, present in-repo at `sam3/assets/`) + a **VE text transformer, width 1024, 24 layers, 16 heads** (`model_builder.py:259-264`, `text_encoder_ve_ada.py:395-433`).
- **Fusion + decoder:** DETR-style `TransformerEncoderFusion` (6 layers) + `TransformerDecoder` with **200 object queries**, box-refine, presence token (`model_builder.py:192-218`).
- **Mask head:** MaskFormer-style `UniversalSegmentationHead` + `PixelDecoder` (3 upsampling stages) (`model_builder.py:229-239`).
- **The "-I" part:** `MultiHeadMLPAdapter` bottleneck modules (MHA + down/up projection, **zero-initialized** so they start as identity) are inserted into the text encoder's residual blocks and the transformer (`text_encoder_ve_ada.py:133-200`). A `text_type` integer selects the adapter path per instruction tier: `1`=bypass (concept), `2`=adapter-1 (simple), `3/4`=serial adapters (complex) (`text_encoder_ve_ada.py:75-99`; wired in `sam3_image.py:615-732`). **The SAM 3 backbone is frozen; only adapters + a few projection heads train** (`model_builder.py:598-613`, whitelist = `adapter`, `simple_query_proj`, `complex_query_proj`, `concept_proj`). LoRA is an alternative path (`model_builder.py:753-758`).

Training is **3 staged runs** (`README.md:121-146`): Stage 1-1 simple queries → Stage 1-2 complex queries → Stage 3 joint (adds KL / InfoNCE / margin losses, `sam3i_3_all.yaml`), each initialized from the previous, all starting from SAM 3 base.

---

## 2. Capabilities (inputs / outputs)

- **Input:** one **RGB image** + one **text instruction** (concept / simple / complex). The inference dataset opens every image as `Image.open(...).convert("RGB")` (`scripts/inference.py:186`) at **1008×1008** square resize, normalized with mean/std = 0.5 (`inference.py:76-77,104-111`).
- **Output:** a **set of instance masks with scores** (open-vocabulary detection→segmentation). Each prediction is a COCO-RLE mask + a float score (`inference.py:249-262`); post-processing thresholds at `detection_threshold` (default 0.5). Because it inherits SAM 3's *all-instances* behavior, a single instruction can return **multiple masks** (the eval supports 1-to-1, 1-to-N, and 1-to-All modes, `scripts/evaluate.py`, `scripts/eval.sh:141-150`).
- **Zero-shot vs trained:** it is **open-vocabulary / language-driven** (no fixed class list), but its instruction-following ability is **learned from the training datasets**: RefCOCO/RefCOCO+/RefCOCOg, gRefCOCO, Ref-ZOM, ReasonSeg, MMR, and the paper's own **HMPL-Instruct** (1to1/1toN/1toAll) — all **RGB natural images** over **COCO 2014/2017** (`README.md:60-118`, `base.yaml:585-596`). No near-IR, medical, or eye imagery anywhere in the training mix.
- **Prompt modalities NOT used in the provided pipeline:** although the codebase *inherits* SAM 1/SAM 2 interactive point/box + video tracker machinery (`model_builder.py:283-341,721-725`; `sam1_task_predictor.py`), the shipped image inference path builds datapoints with **`objects=[]`** and `enable_inst_interactivity=False` (`inference.py:189-198`, `model_builder.py:648`). So out-of-the-box SAM3-I here is **text-prompt only** — you cannot feed it a click/box seed without writing new code.

---

## 3. How to run inference end-to-end

### 3.1 Entry points (cite paths)
- **Unified launcher:** `run.sh` — `install` / `train` / `eval` / `inference` (`run.sh:22-47`).
- **Batch inference:** `scripts/inference.py` — multi-GPU, COCO-JSON-driven (`--json_path`, `--image_root`, `--output_path`, `--checkpoint_path`, `--categories {concept,simple,complex}`, `--batch_size`, `--gpus`, `--detection_threshold`) (`scripts/inference.py:385-402`).
- **One-click eval (reproduce paper):** `scripts/eval.sh` (requires `CHECKPOINT`, `DATASET_JSON_ROOT`, `IMAGE_ROOT_BASE`) → runs inference + `scripts/evaluate.py` (gIoU / mIoU / P@0.50 / COCO AP).
- **Agentic mode (optional):** `sam3/sam3/agent/` implements an **LLM-driven** loop (chain-of-thought "iterative checking") that calls an **OpenAI-compatible** endpoint (`agent/client_llm.py:1-104`, needs `api_key`/`base_url`) plus a SAM3 service — extra infra, not needed for basic use.

> **Important:** There is **no simple "one image + one string" demo script and no example notebook**. The only supported inference path is the COCO-style-JSON batch runner (`scripts/inference.py`), which expects an annotation JSON whose `images[*].text_inst_input` carries the `concept`/`simple_query`/`complex_query` fields (`inference.py:143-177`). To annotate arbitrary EV-Eye frames you would have to **synthesize such a JSON yourself** (or write a small wrapper around `build_sam3_image_model` + `PostProcessImage`).

### 3.2 Required weights — availability
| Weight | Role | In repo? | How obtained |
|---|---|---|---|
| **SAM 3 base** (`sam3.pt` + `config.json`) | frozen backbone | **No** | Auto-download from HuggingFace **`facebook/sam3`** via `hf_hub_download` (`model_builder.py:73-75, 620-622`). This is a **gated Meta model** — requires HF login + accepting Meta's license; download can fail without access. |
| **SAM3-I Stage-3 fine-tuned checkpoint** | the instruction adapters (the actual "SAM3-I") | **No** | External **Google Drive** link only (`README.md:42-44`). Must be downloaded manually and passed as `--checkpoint_path`. |
| BPE vocab (`bpe_simple_vocab_16e6.txt.gz`) | tokenizer | **Yes** | `sam3/assets/` (present) |

There are **no `.pt`/`.safetensors` files in the repo**. Without both downloads the model cannot run with instruction skills: with *no* checkpoint it loads only base SAM 3 from HF and zero-inits the adapters (→ behaves like un-tuned SAM 3, `model_builder.py:537-561`).

A sample prediction file `predictions.json` **is** included (2830 items on COCO-2017 helmet/spoon/shoe queries) — useful only to see the **output format**, not as weights.

### 3.3 Dependencies / hardware
- Install: `bash run.sh install` → `pip install -e ".[dev,train]"` + `pycocotools tqdm pillow einops` (`run.sh:23-29`). Core deps: `torch/torchvision` (implied), `timm>=1.0.17`, `numpy`, `einops`, `ftfy`, `regex`, `iopath`, `huggingface_hub`; train extras add `hydra-core`, `submitit`, `fvcore`, `fairscale`, etc. (`sam3/pyproject.toml:27-80`).
- **GPU:** CUDA required in practice — inference uses `torch.autocast("cuda", bfloat16)`, per-GPU process spawning, and TF32 (`inference.py:84-89,313`). Defaults assume **8 GPUs**, batch 32 (`eval.sh:77-78`). A ViT-L trunk at **1008×1008** is heavy; realistically needs a **modern ≥16–24 GB GPU** (single-GPU is possible via `--gpus 0` but VRAM for 1008² + 200 queries is non-trivial; no official VRAM figure is stated in the repo).

---

## 4. Applicability to near-IR pupil annotation

### 4.1 Could it segment the pupil from `"the black pupil"`?
Mechanically, **yes** — you can pass a `concept` (`"pupil"`) or a `simple_query` (`"the black pupil"`) and it will emit instance masks + scores. The task (one dark, roughly elliptical, high-contrast blob) is *geometrically* easy for any SAM-class mask head, and "pupil"/"iris" are common English nouns the CLIP-style text encoder knows.

**But the practical risk is high:**
1. **Domain gap (grayscale near-IR eye close-ups):** every training image is **RGB COCO/ReasonSeg natural scenes**; input is force-converted to RGB and normalized as if RGB (`inference.py:186,104-111`). DAVIS346 near-IR eye crops (low-texture, IR reflections/glints, sometimes the pupil merges with dark lashes/shadow) are **far outside** the training distribution. Referring/reasoning tuning was optimized to disambiguate *among many everyday objects*, a skill that is largely irrelevant when the scene is a single eye — and the language priors may even *mislead* (e.g., latching onto a dark eyelash region or a glint-bounded circle).
2. **"pupil" vs "iris" vs "eyeball" confusion:** because the concept vocabulary is inherited from SAM 3's broad training, there is no guarantee it separates *pupil* (inner dark disc) from *iris* or the whole eye reliably in IR. Corneal/IR **glints inside the pupil** (bright specular dots) may split or hole the mask.
3. **Multi-instance behavior:** SAM 3 lineage returns *all* instances of a concept; for a single-pupil crop this is usually fine (top-1 or union), but scoring/threshold tuning would be needed (the pipeline exposes `--keep_top1` and `detection_threshold`, `inference.py:370-375`).
4. **No geometric-prompt fallback in the shipped path:** GSAM2's strength is text→box (GroundingDINO) → **box-seeded SAM2** for a crisp mask. SAM3-I's shipped path has **no box/point seeding** (Sec. 2). You lose the "geometry-anchored" safety net; the text alone must localize the pupil.

### 4.2 As an independent audit source vs GroundingDINO+SAM2 (GSAM2)
| Dimension | GSAM2 (current) | SAM3-I |
|---|---|---|
| Independence from GSAM2 | — | **High** — different detector (SAM 3 open-vocab, no GroundingDINO), different mask head, different training data. Genuinely independent → good for cross-checking. |
| Prompt | text→box→mask (geometry-anchored) | **text-only** instruction (no box seed in shipped code) |
| Domain fit to IR eye | already in use / tuned by you | **worse** — pure RGB natural-image tuning, larger domain gap |
| Turnkey for single images | (your pipeline) | **No** — must build COCO-style JSON or write a wrapper |
| Weights ready | (yours) | **No** — 2 external downloads, one **gated** (HF `facebook/sam3`), one Google-Drive |
| Setup / compute cost | (yours) | **Higher** — ViT-L @1008², multi-GPU defaults, heavy deps |
| Code maturity | — | research-grade: extensive commented-out code, debug branches, `.bak` eval script, no unit demo |

**Verdict for auditing:** SAM3-I's *independence* is its one strong selling point — as a second opinion it does not share GroundingDINO or SAM2 with your pipeline. However, its **domain gap (RGB→near-IR), text-only prompting, missing/gated weights, and lack of a single-image entry point** make it a **costly, uncertain** audit source right now. A cheaper independent audit source would be **plain SAM 3** (the same backbone without the RGB-referring adapters) or a lightweight IR-tuned classical/threshold pupil detector. If you do pursue SAM3-I, a smarter integration would be to bypass its text path and use the *inherited* SAM2/point-prompt machinery with a centroid seed — but that requires custom code the repo does not provide.

### 4.3 Concrete concerns / unknowns
- **Weights availability & access:** both required artifacts are off-repo; `facebook/sam3` is a **gated** download (HF auth + Meta license acceptance) that may block CI/automation.
- **No reported near-IR / grayscale numbers** anywhere; all metrics target RefCOCO/HMPL/ReasonSeg RGB benchmarks (`eval.sh:141-150`).
- **Code completeness:** the image inference works, but it is clearly a stripped research release — dead/commented code in `sam3_image.py` and `text_encoder_ve_ada.py`, a `.eval_old.sh.bak`, agent mode needing an external LLM endpoint. Expect to debug.

---

## 5. Lineage & license

- **Lineage:** direct descendant of **Meta's Segment Anything family → SAM 3**. Package = `sam3` "Segment Anything Model 3 implementation" (`pyproject.toml:6-8`); base weights = HF `facebook/sam3` (`model_builder.py:73`); all headers `Copyright (c) Meta Platforms, Inc.`; upstream URL `facebookresearch/sam3` (`pyproject.toml:82-84`). SAM3-I is the **academic instruction-tuning layer** (adapters/LoRA) on top of that, from the ACL-2026 paper (`README.md:196-203`).
- **License — UNCLEAR / effectively unresolved.** `pyproject.toml` declares `license = {file = "LICENSE"}` and classifier "OSI Approved :: MIT License" (`pyproject.toml:11,18`), **but there is NO `LICENSE` (or `NOTICE`) file anywhere in the repo** (verified: `find … -iname 'license*'` → none; not in `SOURCES.txt` either). Meanwhile every source file is stamped `Copyright (c) Meta Platforms, Inc. All Rights Reserved`. **Net:** the SAM3-I code's license text is missing, and the *base SAM 3 weights carry Meta's own (separate) model license/gating*. **Treat licensing as unresolved and verify before any use/redistribution** — do not assume MIT just because a classifier says so.

---

## Summary table

| Question | Answer |
|---|---|
| What is it? | Meta **SAM 3** + free-text **instruction** tuning via zero-init **adapters/LoRA** (SAM 3 frozen). ACL-2026 academic repo. |
| Prompt type | **Free text**: concept / referring / reasoning. Not exemplars; not points/boxes in shipped path. |
| Output | COCO-RLE instance masks + scores; supports 1-to-1 / 1-to-N / 1-to-All. |
| Runs how? | `scripts/inference.py` (COCO-JSON batch, multi-GPU, CUDA/bf16). No single-image demo/notebook. |
| Weights | **Not in repo.** Base = gated HF `facebook/sam3`; fine-tuned = Google-Drive. BPE vocab is in-repo. |
| Fit to IR pupil | Mechanically possible from `"the black pupil"`, but **large RGB→near-IR domain gap**, no geometry seed, uncertain pupil/iris disambiguation. |
| vs GSAM2 (audit) | **Independent** (good) but **higher effort, lower certainty, missing weights**; plain SAM 3 or IR-tuned detector likely cheaper. |
| License | **Unclear** — declares MIT but **no LICENSE file present**; Meta copyright headers + gated base weights. Verify before use. |
