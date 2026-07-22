# HBTXR Algorithm

Reimplementation of the HBTXR event-camera pupil/gaze model (IEEE JETCAS), in a
**flat, functional layout**. It is **not pip-installed**: build a venv with `uv`
and run everything from this `algorithm/` root, where the subdirectories are
top-level import packages (`models`, `common`, `utils`, `dataset`, `engine`).

## Layout

| Dir | Role |
|---|---|
| `models/` | the models. Shared parts — `backbones/` (ViT + patch embed), `blocks/` (transformer block), `heads/` (bbox · ellipse · mask · reliability · roi + factory), `geometry/` (state `(x,y,a,b,θ)`, codec `g()` / residual). Modality assemblies — `frame/`, `event/` (direct detectors), `hybrid/` (shared backbone: search + track + FSM scheduler), `mask/` (standalone UNet). |
| `common/` | training math shared across models: `losses/`, `optim/`, `schedulers/`. |
| `utils/` | leaf helpers: `paths`, `io`, `state6`, `metrics`, `cache`, … |
| `dataset/` | `loaders/` (Davis eye datasets), `hbtxr/` (HBTXR sample pipeline), `preprocess/`, `annotation/`. |
| `engine/` | run lifecycle: `data/` (sample→model adapter), `train/`, `eval/`, `infer/`, `predict/`, `distill/`, `compress/`, `runspec/`, `model_factory`, `tools/`. |
| `configs/` | `modality/*` model fragments, `base/*` shared data+training, `experiment/*` composed runs. |
| `scripts/` | `hbtxr.py` unified CLI dispatcher. |
| `tests/` | pytest suite (torch tests skip if torch is absent). |
| `docs/`, `analysis/` | durable docs and result packages. |

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

## Run (from the `algorithm/` root)

```bash
# train / eval / infer / distill / prune — via the dispatcher …
python scripts/hbtxr.py train -c configs/experiment/frame_hbtxr.yaml
python scripts/hbtxr.py eval  -c configs/experiment/frame_hbtxr.yaml --ckpt runs/frame_hbtxr/train/final.pt
python scripts/hbtxr.py infer -c configs/experiment/frame_hbtxr.yaml -o predictions.csv

# … or the module entrypoints directly
python -m engine.train.entrypoint   -c configs/experiment/hybrid_hbtxr.yaml
python -m engine.distill.entrypoint -c configs/experiment/frame_distill.yaml
python -m engine.compress.entrypoint -c configs/experiment/frame_prune.yaml --ckpt runs/frame_hbtxr/train/final.pt
```

Configs compose via a top-level `include:` (deep-merged, local keys win), so an
experiment file layers a modality model fragment over shared `base/data.yaml` and
`base/training.yaml`. See `docs/ARCHITECTURE.md` for the model and training-pipeline
overview.

## Test

```bash
python -m pytest            # run from algorithm/ (conftest pins the import root)
```

See also the paper-faithful design notes in `docs/ARCHITECTURE.md`.
