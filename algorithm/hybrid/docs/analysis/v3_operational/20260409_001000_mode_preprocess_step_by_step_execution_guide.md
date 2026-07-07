# Mode별 Pre-process 단계별 실행 가이드

최종 갱신: 2026-04-09 KST

## 목적

이 문서는 `HBTXR_v3_0`의 pre-process 파이프라인을 `mode0`, `mode1`, `mode2 materialized`, `mode2 lazy target-FPS` 기준으로 단계별로 정리한 운영 가이드다.

다루는 범위:

- 어떤 순서로 실행해야 하는지
- 어떤 스크립트를 써야 하는지
- 단계별 핵심 옵션이 무엇인지
- 각 단계가 어떤 산출물을 만드는지
- 공식 실행면과 experimental 실행면이 어떻게 나뉘는지

함께 보면 좋은 문서:

- [README.md](../../README.md)
- [20260330_170419_03_groundedsam_dataset_pipeline.md](20260330_170419_03_groundedsam_dataset_pipeline.md)
- [20260330_170419_09_v3_mode_data_pipeline_contracts.md](20260330_170419_09_v3_mode_data_pipeline_contracts.md)
- [20260330_170419_10_v3_yaml_config_reference.md](20260330_170419_10_v3_yaml_config_reference.md)
- [20260408_235500_external_package_operating_guide.md](20260408_235500_external_package_operating_guide.md)

## 1. 공통 전제

### 1.1 공식 실행면과 experimental 실행면

- 공식 pre-process 실행면
  - [run_sync_packages.sh](../../scripts/run_sync_packages.sh)
  - [run_groundedsam_annotation.sh](../../scripts/run_groundedsam_annotation.sh)
  - [run_canonicalize.sh](../../scripts/run_canonicalize.sh)
  - [run_build_manifests.sh](../../scripts/run_build_manifests.sh)
  - [run_prepare.sh](../../scripts/run_prepare.sh)
  - [run_dataloader.sh](../../scripts/run_dataloader.sh)
- experimental pre-process 실행면
  - [run_mode0_prepare.sh](../../exps/scripts/run_mode0_prepare.sh)
  - [run_build_target_fps_dataset.sh](../../exps/scripts/run_build_target_fps_dataset.sh)
  - [run_build_target_fps_canonical.sh](../../exps/scripts/run_build_target_fps_canonical.sh)
  - [run_build_target_fps_manifests.sh](../../exps/scripts/run_build_target_fps_manifests.sh)

### 1.2 공통 입력

- raw dataset
  - `raw_root`
- annotation export root
  - `annotation_root`
- canonical workspace
  - `canonical_root`
- manifests root
  - `manifests_root`
- external package roots
  - `groundedsam_root`
  - `ultralytics_root`
  - `groundedsam2_root`
  - `timelens_root`
  - `timelens_xl_root`
  - `v2e_root`

권장 경로 설정 파일:

- [configs/paths.example.json](../../configs/paths.example.json)
- 보통 실제 실행은 `configs/paths/ev_eye_groundedsam_paths.json`를 사용

### 1.3 공통 Step 0: external package sync

annotation, interpolation, event-generation backend를 쓸 예정이면 먼저 external package hub를 동기화한다.

```bash
sh scripts/run_sync_packages.sh
```

annotation backend만 먼저 받고 싶으면:

```bash
PYTHONPATH=src .venv/bin/python scripts/sync_packages.py --category annotation
```

핵심 옵션:

- `--category annotation|frame_interpolation|event_generation|reference|inactive_optional`
- `--update-existing`
- `--dry-run`
- `--print-manifest`

대표 산출물:

- `packages/Grounded-Segment-Anything`
- `packages/ultralytics`
- `packages/Grounded-SAM-2`
- `packages/timelens`
- `packages/TimeLens-XL`
- `packages/v2e`

## 2. 공통 pre-process 단계 구조

mode에 따라 달라지지만 기본 골격은 아래 네 단계다.

```text
annotation
-> canonicalization
-> manifest generation
-> dataloader check
```

shortcut script:

```bash
sh scripts/run_prepare.sh ...
```

이는 내부적으로 아래 둘을 순서대로 수행한다.

1. [prepare_ev_eye.py](../../scripts/prepare_ev_eye.py) 내부에서 `canonicalize_dataset()`
2. 같은 스크립트 안에서 `build_manifests()`

즉 `run_prepare.sh`는 canonical + manifest를 한 번에 수행하는 shortcut이다.

## 3. Mode0 step-by-step

## 3.1 언제 쓰나

- `mode0`는 raw CSV supervision 기반 실험 경로다.
- Grounded-SAM annotation을 전제하지 않는다.
- 기본 canonical/manifest 이름은 `canonical0`, `manifest0`이다.

## 3.2 Step 1: annotation 단계 생략

`mode0`는 `annotation_mode=manual_csv`가 강제된다.

즉 아래는 하지 않는다.

- `run_groundedsam_annotation.sh`
- `annotate_groundedsam_ev_eye.py`

대신 raw session 내부 CSV를 canonicalization이 직접 읽는다.

## 3.3 Step 2: canonical + manifest shortcut

가장 간단한 실행:

```bash
sh exps/scripts/run_mode0_prepare.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json
```

이 wrapper는 내부적으로 아래 기본값을 강제한다.

- `--annotation-mode manual_csv`
- `--data-mode mode0`
- `--canonical-name canonical0`
- `--manifest-name manifest0`
- `--frame-source original`

## 3.4 Step 2를 low-level로 분리 실행

canonical만 먼저 실행:

```bash
sh scripts/run_canonicalize.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-mode manual_csv \
  --data-mode mode0 \
  --canonical-name canonical0 \
  --frame-source original \
  --link-mode symlink
```

manifest만 따로 실행:

```bash
sh scripts/run_build_manifests.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --data-mode mode0 \
  --canonical-name canonical0 \
  --manifest-name manifest0 \
  --frame-source original \
  --event-policy fixed_count \
  --event-count-target 5000
```

## 3.5 Step 3: dataloader 확인

```bash
sh scripts/run_dataloader.sh \
  --config exps/configs/mode0_stage1.yaml \
  --mode mode0 \
  --stage stage1 \
  --split train
```

## 3.6 mode0에서 자주 조정하는 옵션

canonicalize 쪽:

- `--link-mode symlink|copy|skip`
- `--overwrite-links`
- `--strict-layout`
- `--include-nonstandard-sessions`
- `--disable-raw-ellipse-blink-heuristic`

manifest 쪽:

- `--split-scheme exgaze_with_val|exgaze|random`
- `--resize-policy facet_square_direct|letterbox_square|sensor_full_square`
- `--event-policy fixed_count|time_bin|interval_all`
- `--time-bin-us`
- `--event-count-target`
- `--accumulation plain|causal_linear|fast_causal_linear`

## 3.7 mode0 산출물

- `workspace/canonical0/sessions/...`
- `workspace/canonical0/indexes/sessions.jsonl`
- `manifests/manifest0/train_manifest.jsonl`
- `manifests/manifest0/val_manifest.jsonl`
- `manifests/manifest0/test_manifest.jsonl`

## 4. Mode1 step-by-step

## 4.1 언제 쓰나

- original frame + event voxel baseline
- 기본 canonical/manifest 이름은 `canonical1`, `manifest1`
- annotation source는 보통 `groundedsam` backend를 권장

## 4.2 Step 1: annotation backend 선택

annotation backend 선택지:

- `groundedsam`
- `ultralytics_sam3`
- `groundedsam2`

기본 backend는 `groundedsam`이다.

## 4.3 Step 2: annotation export 생성

기본 Grounded-SAM:

```bash
sh scripts/run_groundedsam_annotation.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-backend groundedsam \
  --groundingdino-checkpoint /path/to/groundingdino_swint_ogc.pth \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --device cuda:0
```

Ultralytics SAM3:

```bash
PYTHONPATH=src .venv/bin/python scripts/annotate_groundedsam_ev_eye.py \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-backend ultralytics_sam3 \
  --ultralytics-sam3-checkpoint /path/to/sam3_b.pt \
  --device cuda:0
```

Grounded-SAM-2:

```bash
PYTHONPATH=src .venv/bin/python scripts/annotate_groundedsam_ev_eye.py \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-backend groundedsam2 \
  --groundedsam2-config /path/to/sam2.1_hiera_l.yaml \
  --groundedsam2-checkpoint /path/to/sam2.1_hiera_large.pt \
  --groundedsam2-groundingdino-checkpoint /path/to/groundingdino_swint_ogc.pth \
  --device cuda:0
```

annotation 쪽 핵심 옵션:

- `--annotation-backend`
- `--device`
- `--classes`
- `--box-threshold`
- `--text-threshold`
- `--nms-threshold`
- `--min-mask-area`
- `--frame-step`
- `--max-frames-per-session`
- `--max-sessions`
- `--overwrite`

## 4.4 Step 3: canonicalize mode1

```bash
sh scripts/run_canonicalize.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-mode groundedsam \
  --annotation-backend groundedsam \
  --data-mode mode1 \
  --canonical-name canonical1 \
  --frame-source original \
  --link-mode symlink
```

핵심 옵션:

- `--annotation-mode auto|manual_csv|groundedsam`
- `--annotation-backend groundedsam|ultralytics_sam3|groundedsam2`
- `--data-mode mode1`
- `--canonical-name canonical1`
- `--frame-source original`
- `--link-mode`
- `--overwrite-links`
- `--strict-layout`

`mode1`에서는 frame interpolation 옵션을 쓰지 않는다.

## 4.5 Step 4: build manifest1

```bash
sh scripts/run_build_manifests.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --data-mode mode1 \
  --canonical-name canonical1 \
  --manifest-name manifest1 \
  --frame-source original \
  --event-policy fixed_count \
  --event-count-target 5000 \
  --resize-policy facet_square_direct
```

핵심 옵션:

- `--split-scheme`
- `--resize-policy`
- `--input-width`
- `--input-height`
- `--event-policy`
- `--time-bin-us`
- `--event-count-target`
- `--accumulation`

## 4.6 Step 5: shortcut으로 한 번에 수행

```bash
sh scripts/run_prepare.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-mode groundedsam \
  --annotation-backend groundedsam \
  --data-mode mode1 \
  --canonical-name canonical1 \
  --manifest-name manifest1 \
  --frame-source original
```

주의:

- `run_prepare.sh`는 annotation export를 생성하지 않는다.
- 즉 mode1 기본 운영 순서는 보통
  1. `run_groundedsam_annotation.sh`
  2. `run_prepare.sh`
  순서다.

## 4.7 Step 6: dataloader 확인

```bash
sh scripts/run_dataloader.sh \
  --config configs/mode1_stage1.yaml \
  --mode mode1 \
  --stage stage1 \
  --split train
```

## 4.8 mode1 산출물

- `workspace/groundedsam_annotations/...`
- `workspace/canonical1/sessions/...`
- `workspace/canonical1/indexes/sessions.jsonl`
- `manifests/manifest1/*.jsonl`

## 5. Mode2 materialized step-by-step

## 5.1 언제 쓰나

- interpolated frame를 canonical row에 직접 materialize하는 `mode2`
- 기본 canonical/manifest 이름은 `canonical2`, `manifest2`
- 대표적으로 `mode2_stage1.yaml`, `mode2_stage2.yaml`가 이 경로를 사용

## 5.2 Step 1: annotation export 생성

annotation 단계는 `mode1`과 동일하다.

가장 흔한 실행:

```bash
sh scripts/run_groundedsam_annotation.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-backend groundedsam \
  --groundingdino-checkpoint /path/to/groundingdino_swint_ogc.pth \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --device cuda:0
```

## 5.3 Step 2: canonicalize mode2 with interpolated frames

linear blend:

```bash
sh scripts/run_canonicalize.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-mode groundedsam \
  --annotation-backend groundedsam \
  --data-mode mode2 \
  --canonical-name canonical2 \
  --frame-source interpolated \
  --interp-backend linear_blend \
  --interp-target-fps 60
```

TimeLens:

```bash
sh scripts/run_canonicalize.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-mode groundedsam \
  --annotation-backend groundedsam \
  --data-mode mode2 \
  --canonical-name canonical2 \
  --frame-source interpolated \
  --interp-backend timelens \
  --interp-target-fps 60 \
  --interp-timelens-checkpoint /path/to/attention.bin \
  --interp-timelens-device cuda:0
```

TimeLens-XL:

```bash
sh scripts/run_canonicalize.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-mode groundedsam \
  --annotation-backend groundedsam \
  --data-mode mode2 \
  --canonical-name canonical2 \
  --frame-source interpolated \
  --interp-backend timelens_xl \
  --interp-target-fps 60 \
  --interp-timelens-xl-checkpoint /path/to/checkpoint.bin \
  --interp-timelens-xl-device cuda:0
```

핵심 interpolation 옵션:

- `--interp-backend linear_blend|timelens|timelens_xl`
- `--interp-target-fps`
- `--interp-fixed-insert`
- `--interp-count-policy round|floor|ceil`
- `--interp-max-insert`
- `--interp-timelens-checkpoint`
- `--interp-timelens-device`
- `--interp-timelens-xl-checkpoint`
- `--interp-timelens-xl-device`
- `--synthetic-overlap-policy`

중요:

- 이 경로는 interpolated frame와 provenance를 canonical에 기록한다.
- synthetic event를 `v2e`로 이 단계에서 별도 materialize하지는 않는다.
- materialized mode2의 event 해석은 이후 manifest + dataset 단계에서 `raw_window` 또는 `source_pair_average` 전략으로 처리된다.

## 5.4 Step 3: build manifest2

```bash
sh scripts/run_build_manifests.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --data-mode mode2 \
  --canonical-name canonical2 \
  --manifest-name manifest2 \
  --frame-source interpolated \
  --event-policy fixed_count \
  --event-count-target 5000 \
  --synthetic-overlap-policy reuse_event_window
```

mode2에서 manifest row는 아래 metadata를 더 많이 가진다.

- `synthetic_frame_flag`
- `interp_source_pair`
- `interp_alpha`
- `interp_model`
- `synthetic_event_window`
- `interpolation_ref`

## 5.5 Step 4: shortcut으로 한 번에 수행

```bash
sh scripts/run_prepare.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-mode groundedsam \
  --annotation-backend groundedsam \
  --data-mode mode2 \
  --canonical-name canonical2 \
  --manifest-name manifest2 \
  --frame-source interpolated \
  --interp-backend timelens \
  --interp-target-fps 60
```

## 5.6 Step 5: dataloader 확인

```bash
sh scripts/run_dataloader.sh \
  --config configs/mode2_stage1.yaml \
  --mode mode2 \
  --stage stage1 \
  --split train
```

또는 stage2 preset:

```bash
sh scripts/run_dataloader.sh \
  --config configs/mode2_stage2.yaml \
  --mode mode2 \
  --stage stage2 \
  --split train
```

## 5.7 mode2 materialized 산출물

- `workspace/canonical2/sessions/.../frames/*.png`
- `workspace/canonical2/sessions/.../labels/interpolation_index.json`
- `workspace/canonical2/indexes/sessions.jsonl`
- `manifests/manifest2/*.jsonl`

## 6. Mode2 lazy target-FPS step-by-step

## 6.1 언제 쓰나

- 매우 높은 target FPS 실험
- `mode2`를 `frame_source=target_fps_grid`로 쓰는 경로
- canonical row가 source pair / session store를 참조하고, dataset가 lazy하게 frame/event를 재구성

현재 이 경로는 official core라기보다 `exps/scripts/` 아래의 advanced experimental pipeline이다.

## 6.2 전체 순서

```text
annotation export
-> target-fps dataset build
-> target-fps canonical bridge
-> target-fps manifest build
-> lazy dataloader / train
```

## 6.3 Step 1: annotation export 준비

`target_fps_build`는 annotation_root의 annotation store를 참조할 수 있으므로, 보통 먼저 annotation export를 준비한다.

예:

```bash
sh scripts/run_groundedsam_annotation.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --annotation-backend groundedsam \
  --groundingdino-checkpoint /path/to/groundingdino_swint_ogc.pth \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --device cuda:0
```

## 6.4 Step 2: target-fps session store build

planning만 먼저:

```bash
sh exps/scripts/run_build_target_fps_dataset.sh \
  --target-fps 2000 \
  --interpolation-backend timelens
```

실제 materialize:

```bash
sh exps/scripts/run_build_target_fps_dataset.sh \
  --target-fps 2000 \
  --execute \
  --interpolation-backend timelens \
  --timelens-checkpoint /path/to/attention.bin \
  --timelens-device cuda:0 \
  --frame-storage-mode lazy_source_frames
```

TimeLens-XL + v2e 예시:

```bash
PYTHONPATH=src .venv/bin/python exps/scripts/build_target_fps_dataset.py \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --target-fps 2000 \
  --execute \
  --interpolation-backend timelens_xl \
  --timelens-xl-checkpoint /path/to/checkpoint.bin \
  --timelens-xl-device cuda:0 \
  --event-generation-backend v2e \
  --v2e-device cuda:0 \
  --frame-storage-mode materialized_target_frames
```

핵심 옵션:

- `--target-root`
- `--target-fps`
- `--user-id`
- `--eye left|right|both`
- `--session-code 101,102`
- `--include-nonstandard-sessions`
- `--max-sessions`
- `--execute`
- `--interpolation-backend linear_blend|timelens|timelens_xl`
- `--timelens-checkpoint`
- `--timelens-device`
- `--timelens-xl-checkpoint`
- `--timelens-xl-device`
- `--event-generation-backend none|v2e`
- `--v2e-device`
- `--session-store-format auto|h5|npz`
- `--frame-storage-mode materialized_target_frames|lazy_source_frames`
- `--overwrite`

중요 제약:

- `frame_storage_mode=lazy_source_frames`일 때는 `event_generation_backend=none`만 허용된다.
- `v2e`는 interpolated frame 이후 synthetic event를 생성하는 backend다.
- 즉 `v2e`는 frame interpolation backend가 아니라 별도 단계다.

## 6.5 Step 3: target-fps canonical bridge

```bash
sh exps/scripts/run_build_target_fps_canonical.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --target-root ./workspace/target_data \
  --target-fps 2000 \
  --data-mode mode2 \
  --canonical-name canonical2 \
  --frame-source target_fps_grid
```

핵심 옵션:

- `--target-root`
- `--target-fps`
- `--data-mode mode1|mode2`
- `--canonical-name`
- `--frame-source`
- `--overwrite`
- `--link-mode symlink|copy|skip`

이 단계의 역할:

- target-fps session store를 canonical row로 bridge
- `frame_path=None`일 수 있는 lazy row 생성
- `session_store_path`, `session_store_layout`, `event_index_range` 메타데이터 정규화

## 6.6 Step 4: target-fps manifest build

```bash
sh exps/scripts/run_build_target_fps_manifests.sh \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --data-mode mode2 \
  --canonical-name canonical2 \
  --manifest-name manifest2 \
  --frame-source target_fps_grid \
  --event-policy interval_all \
  --resize-policy facet_square_direct
```

이 단계는 결국 [build_manifests.py](../../src/hbtxr/preprocess/build_manifests.py)를 사용하므로 핵심 옵션은 일반 manifest build와 같다.

## 6.7 Step 5: dataloader / 학습 진입

lazy target-FPS는 보통 아래 experimental preset과 결합한다.

- `exps/configs/mode2_stage2_lazy_2000fps.yaml`

예:

```bash
sh scripts/run_dataloader.sh \
  --config exps/configs/mode2_stage2_lazy_2000fps.yaml \
  --mode mode2 \
  --stage stage2 \
  --split train
```

## 6.8 mode2 lazy target-FPS 산출물

- `workspace/target_data/fps_<tag>/sessions/...`
- target-fps session store (`.h5` 또는 `.npz`)
- `workspace/target_data/fps_<tag>/indexes/sessions.jsonl`
- `workspace/canonical2/...` bridge된 canonical row
- `manifests/manifest2/*.jsonl` with `frame_source=target_fps_grid`

## 7. 빠른 선택 가이드

### 7.1 가장 보수적인 baseline

- `mode1`
- `groundedsam`
- `canonical1`
- `manifest1`

### 7.2 interpolated frame를 쓰되 구조를 단순하게 유지하고 싶을 때

- `mode2 materialized`
- `interp-backend=linear_blend` 또는 `timelens`
- `canonical2`
- `manifest2`

### 7.3 매우 높은 FPS와 대규모 session store 실험

- `mode2 lazy target-FPS`
- `build_target_fps_dataset`
- `build_target_fps_canonical`
- `build_target_fps_manifests`

### 7.4 raw CSV supervision 기반 실험

- `mode0`
- `annotation-mode=manual_csv`
- `canonical0`
- `manifest0`

## 8. 핵심 정리

- `mode0`
  - annotation 단계 없이 raw CSV 기반으로 바로 canonicalize + manifest
- `mode1`
  - annotation -> canonical1 -> manifest1
- `mode2 materialized`
  - annotation -> interpolated canonical2 -> manifest2
- `mode2 lazy target-FPS`
  - annotation -> target-fps dataset -> canonical bridge -> manifest2

즉, `mode`가 바뀌면 단순 preset만 달라지는 것이 아니라 pre-process 단계의 source, canonical 이름, frame source, 그리고 중간 산출물 구조까지 함께 달라진다.
