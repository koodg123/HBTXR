# HBTXR_v3_0 학습·평가·배포 운영 가이드

최종 갱신: 2026-04-09 KST

## 목적

이 문서는 현재 통합된 `HBTXR_v3_0` 기준으로 아래 내용을 한 번에 정리한 운영용 가이드다.

- 전체 프로젝트 구조
- 전체 파이프라인
- 학습 및 평가 방법
- 핵심 메트릭
- 함수 호출 스택
- 배포 및 실행 스크립트

심화 문서는 아래를 함께 본다.

- [20260408_233000_v3_project_structure_and_pipeline_analysis.md](20260408_233000_v3_project_structure_and_pipeline_analysis.md)
- [20260408_233200_train_hbtxr_actual_execution_flow.md](20260408_233200_train_hbtxr_actual_execution_flow.md)
- [20260408_233400_dataset_model_loss_detailed_call_stack.md](20260408_233400_dataset_model_loss_detailed_call_stack.md)

## 1. 전체 프로젝트 구조

현재 루트 구조는 아래처럼 이해하면 된다.

```text
HBTXR_v3_0/
  configs/
  docs/
  exps/
  legacy/
  packages/
  scripts/
  src/hbtxr/
  tests/
```

### 1.1 공식 운영 표면

- `configs/`
  - 공식 preset
  - `base.yaml`
  - `mode1_stage1.yaml`
  - `mode1_stage2.yaml`
  - `mode2_stage1.yaml`
  - `mode2_stage2.yaml`
- `scripts/`
  - 공식 CLI와 shell wrapper
- `src/hbtxr/`
  - 실제 구현
- `tests/`
  - 회귀, snapshot, CI guard

### 1.2 실험 표면

- `exps/scripts/`
  - one-off 실행, preview, review, render, TSGSS, debug helper
- `exps/configs/`
  - mode0, optimizer pool, lazy mode2, Grounded-SAM option JSON

### 1.3 외부 패키지 허브

- `packages/`
  - annotation
    - `Grounded-Segment-Anything`
    - `ultralytics`
    - `Grounded-SAM-2`
  - frame interpolation
    - `timelens`
    - `TimeLens-XL`
  - event generation
    - `v2e`
  - inactive optional reference
    - `Swift-Eye`

운영 기준은 [20260408_235500_external_package_operating_guide.md](20260408_235500_external_package_operating_guide.md)를 본다.

### 1.4 핵심 코드 계층

- `src/hbtxr/preprocess/`
  - annotation, canonicalization, target-FPS, manifest build
- `src/hbtxr/data/`
  - dataset, loader, ROI/event/target/sample assembly
- `src/hbtxr/models/`
  - patch frontend, backbone, branch heads, tracker runtime
- `src/hbtxr/loss/`
  - primitive, assigner, stage1/stage2, metrics
- `src/hbtxr/training/`
  - model factory, session, checkpoint, step runner, export
- `src/hbtxr/runtime/`
  - online tracker wrapper와 similarity

## 2. 전체 파이프라인

공식 파이프라인은 아래 순서다.

```text
external package sync
-> dataset relocation
-> Grounded-SAM annotation
-> canonicalization
-> manifest generation
-> dataloader inspection
-> stage1 training
-> stage2 training
-> evaluation / inference / visualization
-> export
```

### 2.1 external package sync

- Python: [sync_packages.py](../../scripts/sync_packages.py)
- Shell: [run_sync_packages.sh](../../scripts/run_sync_packages.sh)
- 목적:
  - `packages/Grounded-Segment-Anything`
  - `packages/timelens`
  - `packages/ultralytics`
  - `packages/Grounded-SAM-2`
  - `packages/TimeLens-XL`
  - `packages/v2e`
  - `packages/Swift-Eye`
  동기화

### 2.2 dataset relocation

- Python: [relocate_ev_eye_dataset.py](../../scripts/relocate_ev_eye_dataset.py)
- Shell: [run_relocate.sh](../../scripts/run_relocate.sh)
- 목적:
  - canonical / manifest 내부 경로 rewrite
  - symlink 복구

### 2.3 Grounded-SAM annotation

- Python: [annotate_groundedsam_ev_eye.py](../../scripts/annotate_groundedsam_ev_eye.py)
- Shell: [run_groundedsam_annotation.sh](../../scripts/run_groundedsam_annotation.sh)
- 핵심 모듈:
  - [annotation_backends.py](../../src/hbtxr/preprocess/annotation_backends.py)
  - [groundedsam_build.py](../../src/hbtxr/preprocess/groundedsam_build.py)
  - [groundedsam_pipeline.py](../../src/hbtxr/preprocess/groundedsam_pipeline.py)

### 2.4 canonicalization

- Python: [canonicalize_hbtxr.py](../../scripts/canonicalize_hbtxr.py)
- Shell: [run_canonicalize.sh](../../scripts/run_canonicalize.sh)
- 핵심 모듈:
  - [canonicalize.py](../../src/hbtxr/preprocess/canonicalize.py)
  - [canonicalize_pipeline.py](../../src/hbtxr/preprocess/canonicalize_pipeline.py)

### 2.5 manifest generation

- Python: [build_mode_manifests.py](../../scripts/build_mode_manifests.py)
- Shell: [run_build_manifests.sh](../../scripts/run_build_manifests.sh)
- 핵심 모듈:
  - [build_manifests.py](../../src/hbtxr/preprocess/build_manifests.py)

### 2.6 dataset / dataloader

- Python: [check_dataloader.py](../../scripts/check_dataloader.py)
- Shell: [run_dataloader.sh](../../scripts/run_dataloader.sh)
- 핵심 모듈:
  - [loader.py](../../src/hbtxr/data/loader.py)
  - [dataset.py](../../src/hbtxr/data/dataset.py)
  - [pipeline.py](../../src/hbtxr/data/pipeline.py)

### 2.7 학습

- Python: [train_hbtxr.py](../../scripts/train_hbtxr.py)
- Shell: [run_train.sh](../../scripts/run_train.sh)
- shortcut:
  - [run_prepare_and_train.sh](../../scripts/run_prepare_and_train.sh)
  - [run_prepare_and_train_mode2.sh](../../scripts/run_prepare_and_train_mode2.sh)
  - [run_mode2_end_to_end.sh](../../scripts/run_mode2_end_to_end.sh)

### 2.8 평가 / 추론 / 시각화 / export

- 평가:
  - [eval_hbtxr.py](../../scripts/eval_hbtxr.py)
  - [run_eval.sh](../../scripts/run_eval.sh)
- 추론:
  - [infer_hbtxr.py](../../scripts/infer_hbtxr.py)
  - [run_infer.sh](../../scripts/run_infer.sh)
- 시각화:
  - [visualize_dataset.py](../../scripts/visualize_dataset.py)
  - [visualize_inference_results.py](../../scripts/visualize_inference_results.py)
  - [visualize_runtime.py](../../scripts/visualize_runtime.py)
  - [run_vis.sh](../../scripts/run_vis.sh)
- export:
  - [export_hbtxr.py](../../scripts/export_hbtxr.py)
  - [run_export.sh](../../scripts/run_export.sh)

## 3. 학습 방법

## 3.1 stage1

- 목적:
  - search / eye / mask foundation training
- 기본 best metric:
  - `metric_search_p10_pct`
- 대표 checkpoint:
  - `best_search_p10.pt`
  - `best_search_p5.pt`

## 3.2 stage2

- 목적:
  - search + event + track hybrid training
- 보통 stage1 checkpoint를 초기화 가중치로 사용
- 기본 best metric:
  - `metric_track_p10_pct`
- 대표 checkpoint:
  - `best_track_p10.pt`
  - `best_track_p5.pt`

## 3.3 주요 학습 흐름

```text
load_config
-> apply_config_overrides
-> resolve_run_contract
-> train()
   -> make_loader()
   -> build_model()
   -> build_optimizer()
   -> create_teacher_model() optional
   -> TrainingSession.run()
      -> train epoch
      -> val epoch
      -> best metric checkpoint save
      -> early stop check
      -> export optional
```

## 4. 평가 방법과 메트릭

평가는 [eval_hbtxr.py](../../scripts/eval_hbtxr.py)에서 model forward 후 [metrics.py](../../src/hbtxr/loss/metrics.py)의 `compute_metrics()`를 사용해 계산한다.

### 4.1 sample weight와 유효성

metrics 계산은 아래 정보를 함께 사용한다.

- `annotation_quality`
- `closed_eye_flag`
- `mask_valid`
- `valid_track`

즉 단순 평균이 아니라 quality와 geometry validity를 반영한 평균이다.

### 4.2 search 메트릭

- `metric_search_center_px`
  - search state 중심 오차 평균 픽셀
- `metric_search_p10_pct`
  - 중심 오차가 10px 이하인 비율
- `metric_search_p5_pct`
  - 중심 오차가 5px 이하인 비율

### 4.3 event 메트릭

- `metric_event_center_px`
- `metric_event_p10_pct`

### 4.4 track 메트릭

- `metric_track_center_px`
- `metric_track_p10_pct`
- `metric_track_p5_pct`
- `metric_track_quality_mean`
  - `track/pupil[:, 7]` sigmoid 평균

### 4.5 best metric 정책

- stage1 기본:
  - `metric_search_p10_pct`
- stage2 기본:
  - `metric_track_p10_pct`
- `trainer.py`는 이 값을 기준으로 best checkpoint를 저장한다.

## 5. 함수 호출 스택

## 5.1 train_hbtxr.py 기준 상위 호출 스택

```text
scripts/train_hbtxr.py
-> load_config()
-> apply_config_overrides()
-> resolve_run_contract()
-> resolve_training_entry()
-> write_run_artifacts()
-> hbtxr.training.trainer.train()
```

## 5.2 dataset -> model -> loss 상세 호출 스택

```text
manifest row
-> build_dataset_kwargs()
-> Mode1Dataset / Mode2Dataset
-> EVEyeHBTXRDataset.__getitem__()
   -> DatasetPipeline.build_sample()
      -> SessionFrameEventReader.read_sample_assets()
      -> AdaptiveRoiResolver.resolve_effective_roi()
      -> SpatialTransformResolver.build()
      -> SessionFrameEventReader.load_mask()
      -> EventInputBuilder.build()
      -> AnnotationTargetBuilder.build()
      -> SampleAssembler.assemble()
-> collate_samples()
-> move_to_device()
-> HBTXRTracker.forward_train()
   -> TrackerTokenEncoder
   -> SearchBranch.forward()
   -> EventStateBranch.forward()
   -> TrackStateBranch.forward()
-> compute_stage1_losses() or compute_stage2_losses()
-> compute_metrics()
```

## 5.3 runtime 추적 스택

```text
RuntimeHBTXRTracker.step()
-> model.forward_train()
-> ellipse_similarity()
-> TrackSearchSchedulerFSM.step()
-> runtime state 유지
```

## 6. 배포 및 실행 스크립트

현재 공식 shell entry는 아래처럼 이해하면 된다.

### 6.1 단일 작업 wrapper

- [run_train.sh](../../scripts/run_train.sh)
- [run_eval.sh](../../scripts/run_eval.sh)
- [run_infer.sh](../../scripts/run_infer.sh)
- [run_vis.sh](../../scripts/run_vis.sh)
- [run_export.sh](../../scripts/run_export.sh)
- [run_dataloader.sh](../../scripts/run_dataloader.sh)

이 wrapper들은 대부분 [hbtxr_mode_pipeline.sh](../../scripts/hbtxr_mode_pipeline.sh) 또는 대응 Python entry를 호출하는 얇은 entrypoint다.

### 6.2 통합 실행 wrapper

- [run_prepare_and_train.sh](../../scripts/run_prepare_and_train.sh)
  - canonical + manifests + stage1 + stage2
- [run_prepare_and_train_mode2.sh](../../scripts/run_prepare_and_train_mode2.sh)
  - mode2 preset shortcut
- [run_mode2_end_to_end.sh](../../scripts/run_mode2_end_to_end.sh)
  - annotation + mode2 canonical + manifests + stage1 + stage2

### 6.3 통합 dispatcher

- [hbtxr_mode_pipeline.sh](../../scripts/hbtxr_mode_pipeline.sh)

지원 action:

- `train`
- `eval`
- `infer`
- `vis`
- `dataloader`
- `export`

즉 운영 관점에서는 아래 두 레벨로 보면 된다.

- low-level Python canonical surface
- high-level shell wrapper surface

## 7. 권장 읽기 순서

1. [README.md](../../README.md)
2. [20260408_233000_v3_project_structure_and_pipeline_analysis.md](20260408_233000_v3_project_structure_and_pipeline_analysis.md)
3. [20260408_233200_train_hbtxr_actual_execution_flow.md](20260408_233200_train_hbtxr_actual_execution_flow.md)
4. [20260408_233400_dataset_model_loss_detailed_call_stack.md](20260408_233400_dataset_model_loss_detailed_call_stack.md)
5. [20260408_190000_tracker_dataset_output_contracts.md](20260408_190000_tracker_dataset_output_contracts.md)
