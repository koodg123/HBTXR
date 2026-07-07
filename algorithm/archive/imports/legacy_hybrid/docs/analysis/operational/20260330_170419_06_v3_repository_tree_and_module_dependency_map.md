# HBTXR_v3_0 저장소 트리 및 모듈 의존성 맵

최종 갱신: 2026-03-30 KST

Role: `저장소 아키텍트`, `코드베이스 분석가`, `파이프라인 통합 엔지니어`

## 요약

- 이 문서는 현재 checkout된 실제 `HBTXR_v3_0` 저장소 구조를 기준으로 정리한다.
- 목적은 active execution surface와 archive / compatibility surface를 분리하는 것이다.
- 두 번째 목적은 `src/hbtxr`를 의존성 순서로 읽을 수 있게 만드는 것이다.

## 범위

포함:

- `configs/`
- `docs/`
- `exps/`
- `scripts/`
- `src/hbtxr/`
- `tests/`
- top-level metadata 파일

제외:

- `.git/`
- `.venv/`
- 생성된 cache 폴더
- `src/hbtxr_v3_0.egg-info/`

## 현재 트리

```text
HBTXR_v3_0/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- uv.lock
|-- .gitignore
|
|-- configs/
|   |-- base.yaml
|   |-- mode1_stage1.yaml
|   |-- mode1_stage2.yaml
|   |-- mode2_stage1.yaml
|   |-- mode2_stage2.yaml
|   |-- paths.example.json
|   `-- paths/
|
|-- docs/
|   |-- chat/
|   |-- exps/
|   |-- others/
|   |-- plan/
|   `-- prj/
|
|-- exps/
|   |-- README.md
|   |-- configs/
|   |   |-- groundedsam/
|   |   |-- mode0_*.yaml
|   |   |-- mode2_*_optimizer_pool.yaml
|   |   |-- mode2_stage2_lazy_2000fps.yaml
|   |   |-- stage1_train_a.yaml
|   |   `-- stage2_hybrid_v3.yaml
|   `-- scripts/
|       `-- preview / render / review / mode0 / target_fps / tsgss helpers
|
|-- scripts/
|   |-- _bootstrap.py
|   |-- _config.py
|   |-- _viz.py
|   |-- annotate_groundedsam_ev_eye.py
|   |-- build_groundedsam_dataset.py
|   |-- canonicalize_hbtxr.py
|   |-- eval_hbtxr.py
|   |-- infer_hbtxr.py
|   |-- prepare_ev_eye.py
|   |-- train_hbtxr.py
|   |-- visualize_dataset.py
|   |-- visualize_inference_results.py
|   |-- visualize_runtime.py
|   |-- build_dataset_groundedsam.sh
|   |-- run_build_manifests.sh
|   |-- run_canonicalize.sh
|   |-- run_eval.sh
|   |-- run_groundedsam_annotation.sh
|   |-- run_infer.sh
|   |-- run_mode2_end_to_end.sh
|   |-- run_prepare.sh
|   |-- run_prepare_and_train.sh
|   |-- run_prepare_and_train_mode2.sh
|   |-- run_relocate.sh
|   |-- run_sync_packages.sh
|   |-- run_train.sh
|   `-- run_vis.sh
|
|-- src/
|   `-- hbtxr/
|       |-- __init__.py
|       |-- loss_common.py
|       |-- loss_distillation.py
|       |-- loss_primitives.py
|       |-- loss_stage.py
|       |-- optim/
|       |-- data/
|       |-- models/
|       |-- preprocess/
|       |-- runtime/
|       |-- training/
|       `-- utils/
|
|-- tests/
|   |-- conftest.py
|   |-- test_config_v3.py
|   |-- test_dataset_v3.py
|   |-- test_import_graph_v3.py
|   |-- test_loss_catalog_v3.py
|   |-- test_model_v3.py
|   |-- test_optimizer_pool_v3.py
|   |-- test_preprocess_v3.py
|   |-- test_pretrained_loader_v3.py
|   |-- test_runtime_e2e_v3.py
|   |-- test_train_pipeline_v3.py
|   `-- test_trainer_console_v3.py
|
|-- legacy/
|-- ref_cods/
`-- runs/
```

## 표면 분류

### Active execution surface

- `configs/`
  - official runtime preset과 path config
- `exps/`
  - experimental preset, preview, render, ablation, mode0, target-FPS helper
- `scripts/`
  - 공식 CLI 및 shell entrypoint
- `src/hbtxr/`
  - 실제 importable 구현
- `tests/`
  - contract / regression 검증

### Knowledge surface

- `docs/chat/`
- `docs/exps/`
- `docs/plan/`
- `docs/prj/`
- `docs/others/`

### Historical / reference surface

- `legacy/`
  - v1 / v2 시절 config, script, 문서, retired helper 모듈
- `ref_cods/`
  - HW / SW reference snapshot

## `src/hbtxr` 의존성 맵

### 패키지 수준 뷰

```text
utils
|-- paths
|-- io              -> utils.paths
|-- cache
`-- state6

preprocess
|-- path_utils           -> utils.paths
|-- io_utils             -> utils.io
|-- protocol
|-- annotation_groundedsam
|   `-- -> utils.io + utils.paths + utils.state6
|-- groundedsam_build
|   `-- -> annotation_groundedsam + io_utils + utils.io
|-- canonicalize
|   `-- -> annotation_groundedsam + io_utils + path_utils
|        + progress + protocol + session_package + utils.state6
|-- build_manifests
|   `-- -> path_utils + utils.io + utils.paths
|-- target_fps_build
|   `-- -> io_utils + utils.io + utils.paths + utils.state6
|-- target_fps_canonical
|   `-- -> path_utils + utils.io + utils.paths
`-- relocate_dataset
    `-- -> io_utils + path_utils

data
|-- loader           -> dataset
`-- dataset          -> utils.cache + utils.io + utils.paths + utils.state6

models
|-- backbone         -> blocks
|-- heads            -> patch_embeddings
|-- hybrid_tracker
|   `-- -> adapters + backbone + controller + heads + patch_embeddings
`-- preload.pretrained_loader

optim
|-- registry         -> common + per-optimizer files + modifiers
|-- pool             -> registry
`-- modifiers

training
|-- losses facade    -> loss_common + loss_primitives + loss_stage + loss_distillation
`-- trainer
    `-- -> data.loader + models.hybrid_tracker
         + models.preload.pretrained_loader + training.losses + optim.registry

runtime
|-- similarity
`-- tracker          -> runtime.similarity
```

### 통합 뷰

```text
raw EV-Eye
  -> preprocess.groundedsam_build / canonicalize / target_fps_build
  -> preprocess.build_manifests / target_fps_canonical
  -> data.dataset / data.loader
  -> models.hybrid_tracker
  -> training.trainer
  -> runtime.tracker
```

## 구조 해석

- `utils`
  - 최하단 기반 계층이다.
  - path, JSONL I/O, cache helper, `state6` 변환 로직이 여기 있다.
- `preprocess`
  - offline artifact 생성 계층이다.
  - annotation, canonical package, manifest, target-FPS store를 만든다.
  - `data`, `models`, `training`을 직접 import하지 않는다.
- `data`
  - canonical + manifest 산출물을 소비하는 계층이다.
  - preprocessing 함수를 직접 호출하지 않는다.
- `models`
  - 내부 응집도가 높은 계층이다.
  - `hybrid_tracker.py`가 patch embedding, backbone, heads, controller를 묶는 집결점이다.
- `optim`
  - 현재는 독립적인 optimizer registry 및 benchmark surface다.
  - trainer가 이 계층을 통해 optimizer를 resolve한다.
- `training`
  - orchestration 허브다.
  - `trainer.py`에서 data, model, optimizer, pretrained loading, loss, checkpoint, device logic이 만난다.
- `runtime`
  - 의도적으로 얇다.
  - runtime scheduler와 wrapper를 제공하지만 model construction 자체를 소유하지 않는다.

## 실무 권장 읽기 순서

1. [base.yaml](../../configs/base.yaml)
2. [scripts/_config.py](../../scripts/_config.py)
3. [train_hbtxr.py](../../scripts/train_hbtxr.py)
4. `src/hbtxr/preprocess/`
5. `src/hbtxr/data/`
6. [hybrid_tracker.py](../../src/hbtxr/models/hybrid_tracker.py)
7. [trainer.py](../../src/hbtxr/training/trainer.py)
8. [tracker.py](../../src/hbtxr/runtime/tracker.py)

## 메모

- 저장소는 의도적으로 active code와 reference / archive layer를 함께 갖는다.
- 구현 작업은 기본적으로 `configs/`, `scripts/`, `src/hbtxr/`, `tests/`를 기준으로 해야 한다.
- `legacy/`, `ref_cods/`는 기본 수정 대상이 아니라 참고 배경으로 취급하는 것이 맞다.
- `docs/chat`, `docs/jetcas`는 별도 역할을 유지하고, 나머지 문서는 현재 canonical 구조에 따라 읽는 것이 맞다.
