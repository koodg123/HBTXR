# dataset -> model -> loss 상세 호출 스택

최종 갱신: 2026-04-08 KST

Role: `데이터-모델 인터페이스 분석가`, `호출 스택 문서 작성자`

## 목적

이 문서는 학습 중 한 샘플이 manifest row에서 시작해 dataset을 거쳐 model 출력과 loss 계산으로 이어지는 흐름을 함수 단위까지 자세히 풀어쓴다.

분석 기준 파일:

- [src/hbtxr/data/loader.py](../../src/hbtxr/data/loader.py)
- [src/hbtxr/data/dataset.py](../../src/hbtxr/data/dataset.py)
- [src/hbtxr/data/session_reader.py](../../src/hbtxr/data/session_reader.py)
- [src/hbtxr/data/event_builder.py](../../src/hbtxr/data/event_builder.py)
- [src/hbtxr/data/components.py](../../src/hbtxr/data/components.py)
- [src/hbtxr/models/hybrid_tracker.py](../../src/hbtxr/models/hybrid_tracker.py)
- [src/hbtxr/models/tracker/*.py](../../src/hbtxr/models/tracker)
- [src/hbtxr/loss/stage1.py](../../src/hbtxr/loss/stage1.py)
- [src/hbtxr/loss/stage2.py](../../src/hbtxr/loss/stage2.py)
- [src/hbtxr/loss/metrics.py](../../src/hbtxr/loss/metrics.py)

## 1. 전체 개요

```text
train_manifest.jsonl row
-> build_dataset_kwargs()
-> Mode1Dataset / Mode2Dataset
-> EVEyeHBTXRDataset.__getitem__()
   -> reader
   -> roi
   -> transform
   -> event_builder
   -> target_builder
   -> sample_assembler
-> collate_samples()
-> move_to_device()
-> HBTXRTracker.forward_train()
   -> forward_search()
   -> forward_event()
   -> forward_track()
-> compute_stage1_losses() or compute_stage2_losses()
-> compute_metrics()
-> loss_total
```

## 2. loader 생성 스택

학습 시작 시 loader는 아래 순서로 만들어진다.

### 2.1 make_loader_by_mode()

`make_loader_by_mode(manifest_path, cfg, shuffle)`

역할:

1. `build_dataset_kwargs(data_cfg)`
2. `resolve_data_mode(data_cfg)`
3. dataset class 선택
   - `Mode0Dataset`
   - `Mode1Dataset`
   - `Mode2Dataset`
4. `DataLoader(...)` 생성

### 2.2 build_dataset_kwargs()

이 함수는 YAML에서 실제 dataset constructor 인자로 들어갈 값을 materialize한다.

주요 해석 항목:

- `input_size`
- `resize_policy`
- `canonical_root`
- `cache_root`
- `use_cache`
- `per_channel_normalize`
- `event_builder`
- `data_mode`
- `canonical_name`
- `manifest_name`
- `frame_source`
- `mode2_execution`
- `component_cfg`

즉 manifest를 어떻게 읽을지, 어떤 dataset subcomponent를 쓸지 여기서 확정된다.

## 3. dataset 초기화 스택

### 3.1 EVEyeHBTXRDataset.__init__()

dataset 초기화 때 아래가 순서대로 일어난다.

1. manifest row 전체 로드
2. `component_cfg` 해석
3. component variant별 class resolve
4. 아래 component instance 생성

- `SessionFrameEventReader`
- `SpatialTransformResolver`
- `EventInputBuilder`
- `AdaptiveRoiResolver`
- `AnnotationTargetBuilder`
- `SampleAssembler`

즉 현재 dataset는 하나의 비대한 클래스가 아니라 “orchestration facade + 6개 component” 구조다.

### 3.2 component selector

YAML에서 아래처럼 variant를 바꿀 수 있다.

```yaml
data:
  components:
    reader: { variant: split_v1 }
    transform: { variant: split_v1 }
    event_builder: { variant: split_v1 }
    roi: { variant: split_v1 }
    targets: { variant: split_v1 }
    assembler: { variant: split_v1 }
```

현재 `default`와 `split_v1`는 같은 구현을 가리킨다.

## 4. __getitem__() 상세 호출 스택

실제 sample assembly는 아래 한 줄에서 시작된다.

```python
row = self.rows[index]
```

이후 순서는 고정되어 있다.

### 4.1 reader.read_sample_assets(row)

내부 호출:

1. `load_annotation(row["annotation_ref"])`
2. `load_annotation(prev_annotation_ref)` if exists
3. `load_frame_sensor(row)`
4. `sensor_size_wh` 계산
5. `end_timestamp_us` 계산

결과:

- `LoadedSampleAssets`
  - `annotation`
  - `prev_annotation`
  - `frame_sensor`
  - `resolved_frame_path`
  - `resolved_store_path`
  - `sensor_size_wh`
  - `end_timestamp_us`

#### 4.1.1 frame load 분기

`load_frame_sensor()`는 세 경로 중 하나를 탄다.

- `frame_path` 직접 load
- session store의 materialized target frame load
- lazy source-frame synthesis

### 4.2 roi_resolver.resolve_effective_roi_xywh(...)

이 단계에서 최종 ROI가 결정된다.

내부 분기:

- `crop_policy == manifest_roi`
  - manifest row의 ROI 사용
- `crop_policy == density_adaptive`
  - event density를 보고 adaptive ROI 계산

필요하면 내부에서

- `collect_event_points_for_adaptive_crop()`
- `resolve_event_generation_strategy()`
- `resolve_source_pair_timestamps()`

까지 호출된다.

즉 ROI는 단순 bbox field 복사가 아니라 event 생성 전략과 연결되어 있다.

### 4.3 transform_resolver.build(...)

입력:

- row
- `sensor_size_wh`
- `roi_xywh`

출력:

- `SpatialTransform`

이 transform은 이후 frame, mask, event, bbox, ellipse 모두에 공통 적용된다.

### 4.4 reader.load_mask(...)

mask는 다음 우선순위로 생성된다.

1. `pupil_mask_path` 또는 `mask_path`
2. ellipse rasterization
3. bbox rasterization
4. 없으면 zero mask

### 4.5 frame transform

```text
frame_sensor
-> apply_frame(transform)
-> float32 / 255.0
```

결과는 `[1, H, W]` grayscale tensor용 이미지가 된다.

### 4.6 event_input_builder.build(...)

이 단계가 dataset 내부에서 가장 복잡한 부분이다.

#### 4.6.1 generation strategy resolve

`resolve_event_generation_strategy(row)`가 먼저 실행된다.

가능한 전략:

- `raw_window`
- `source_pair_average`
- `target_fps_session_store`

#### 4.6.2 event window resolve

`resolve_event_window(row, end_timestamp_us=...)`

여기서 manifest row의

- `event_window`
- `synthetic_event_window`
- policy
- time_bin_us
- event_count_target

를 합쳐 최종 effective window가 만들어진다.

#### 4.6.3 실제 event voxel build

전략별 내부 경로:

- `target_fps_session_store`
  - `build_event_input_from_session_store()`
- `source_pair_average`
  - `resolve_source_pair_timestamps()`
  - `resolve_source_pair_scope()`
  - left/right event 생성
  - alpha 또는 mean weighted average
- `raw_window`
  - `build_transformed_event_from_window()`

최종적으로는 모두 아래 형태로 통합된다.

- transformed event tensor `[2, H, W]`
- `selected_count`
- event meta

### 4.7 target_builder.build(...)

이 단계에서 supervision target이 만들어진다.

내부 계산:

1. `state6_from_annotation(annotation)`
2. `normalize_uv(...)`
3. `xywht_from_state6(...)`
4. `transform.ellipse(...)`
5. `xywht_to_xyabuv(...)`
6. `pupil_region_target` 계산
7. `eye_target_box` 계산
8. `annotation_quality`, `closed_eye_flag`, `mask_valid`, `valid_track` 계산
9. `legacy_track_delta_to_uv(prev_state, cur_state)`
10. `pupil_track_target` 생성

결과는 `TargetBundle`이다.

### 4.8 sample_assembler.assemble(...)

마지막 단계에서 실제 sample dict가 만들어진다.

주요 출력 key:

- `frame`
- `event`
- `mask_target`
- `eye_target`
- `pupil_region_target`
- `prev_state`
- `cur_state`
- `pupil_search_target`
- `pupil_track_target`
- `constraint_center`
- `annotation_quality`
- `similarity_target`
- `event_density`
- `closed_eye_flag`
- `mask_valid`
- `valid_track`
- `aux_target`
- `meta`

즉 dataset 단계의 최종 ABI는 여기서 확정된다.

## 5. collate와 device 이동

### 5.1 collate_samples()

`DataLoader`는 `collate_samples()`를 사용한다.

동작:

- tensor는 `torch.stack`
- `sample_id`, `meta`는 list 유지
- 기타는 list 유지

### 5.2 move_to_device()

학습 직전 `trainer.py`에서 recursive tensor move가 일어난다.

- tensor만 `device`로 이동
- nested dict도 재귀 이동
- `meta`, `sample_id`는 그대로 유지

## 6. model forward 상세 호출 스택

학습 중 model entry는 `HBTXRTracker.forward_train()`이다.

### 6.1 forward_train()

입력 batch에서 아래를 읽는다.

- `frame`
- `event`
- `prev_state`
- optional `cached_frame`
- optional `patch_cache`

그리고 아래 세 branch를 순서대로 실행한다.

1. `forward_search(frame)`
2. `forward_event(event)`
3. `forward_track(event, prev_state, cached_frame, patch_cache)`

### 6.2 forward_search() 스택

```text
forward_search()
-> encode_frame()
   -> patch_frontend.forward_search()
   -> encode_tokens()
      -> frame_adapter()
      -> apply_width_mask()
      -> backbone()
-> SearchBranch.forward()
   -> eye_head
   -> search_head
   -> search_bbox_aux_head / search_obb_aux_head
   -> roi_bbox_head optional
   -> mask_head optional
   -> SearchMaskGuidanceRefiner.refine() optional
   -> aux_head optional
```

주요 출력:

- `search/eye`
- `search/pupil`
- `search/state`
- `search/pupil_bbox`
- `search/pupil_obb`
- `search/mask_logits`
- `search/aux`

### 6.3 forward_event() 스택

```text
forward_event()
-> encode_event()
   -> patch_frontend.forward_event()
   -> encode_tokens()
      -> event_adapter()
      -> backbone()
-> EventStateBranch.forward()
   -> event_head
   -> event_bbox_aux_head optional
   -> event_obb_aux_head optional
   -> aux_head optional
```

주요 출력:

- `event/pupil`
- `event/state`
- `event/pupil_bbox`
- `event/pupil_obb`
- `event/aux`

### 6.4 forward_track() 스택

```text
forward_track()
-> encoder.encode_track()
   -> patch_frontend.forward_track()
   -> event_adapter()
   -> backbone()
-> TrackStateBranch.forward()
   -> prev_state_encoder(prev_state)
   -> apply_width_mask(prev_feat)
   -> concat(pooled, prev_feat)
   -> apply_track_mask(fused)
   -> track_head(fused)
   -> TrackStateCodec.decode(prev_state, track_logits)
```

주요 출력:

- `track/fused`
- `track/pupil`
- `track/state`

### 6.5 head 생성 경로

model init 시 실제 head 종류는 `TrackerHeadFactory.build()`가 결정한다.

예:

- eye
  - `EyeRegionHead`
  - `DenseEyeRegionHead`
  - `YOLO26EyeRegionHead`
  - `YOLOPointEyeRegionHead`
  - `YOLODetectEyeRegionHead`
  - `SOTCenterPredictor`
  - `SOTCornerPredictor`
- mask
  - `SearchMaskHead`
  - `ProtoMaskHead`
- track
  - `PupilTrackHead`

즉 YAML head option은 결국 factory를 거쳐 실제 class 인스턴스로 materialize된다.

## 7. loss 호출 스택

학습 중 `_forward_once()`에서 아래가 수행된다.

```text
outputs = model(batch)
-> compute_stage1_losses() or compute_stage2_losses()
-> compute_distillation_losses() optional
-> compute_regularization_ssl_losses() optional
-> pruning regularizer optional
-> compute_metrics()
```

### 7.1 stage1 loss 스택

`compute_stage1_losses(batch, outputs, loss_cfg, active_head=...)`

내부 순서:

1. `resolve_sample_masks(batch)`
2. `resolve_search_state(outputs)`
3. `pupil_branch_losses(...)`
4. `pupil_bbox_aux_losses(...)` optional
5. `pupil_obb_aux_losses(...)` optional
6. `compute_eye_logs(...)`
7. `compute_mask_losses(...)`
8. `compute_aux_loss(...)`
9. `constraint_center_loss(...)` optional
10. `sum_loss_logs(...)`

핵심 포인트:

- stage1은 기본적으로 search/eye/mask 중심이다
- event/track loss는 기본 경로에서 비활성이다

### 7.2 stage2 loss 스택

`compute_stage2_losses(batch, outputs, loss_cfg, active_head=...)`

내부 순서:

1. `resolve_sample_masks(batch)`
2. `resolve_search_state(outputs)`
3. `resolve_event_state(outputs)`
4. `resolve_track_state(batch, outputs)`
5. search branch loss
6. event branch loss
7. track branch loss
8. eye/mask/aux loss optional
9. consistency loss
10. constraint center loss
11. `sum_loss_logs(...)`

핵심 포인트:

- stage2는 search/event/track을 함께 본다
- `loss_consistency`가 stage1과 가장 큰 차이 중 하나다

### 7.3 metrics 스택

`compute_metrics(batch, outputs)`

내부 계산:

- `metric_search_center_px`
- `metric_search_p10_pct`
- `metric_search_p5_pct`
- `metric_event_center_px`
- `metric_event_p10_pct`
- `metric_track_center_px`
- `metric_track_p10_pct`
- `metric_track_p5_pct`
- `metric_track_quality_mean`

즉 loss와 metrics는 분리되어 있지만, 둘 다 같은 output contract를 소비한다.

## 8. output key 생산자 맵

| 출력 key | 생산 위치 |
|---|---|
| `search/eye` | `SearchBranch.forward()` |
| `search/pupil` | `SearchBranch.forward()` |
| `search/state` | `SearchBranch.state_from_branch()` |
| `search/mask_logits` | `mask_head` -> `SearchBranch.forward()` |
| `event/pupil` | `EventStateBranch.forward()` |
| `event/state` | `EventStateBranch.forward()` |
| `track/pupil` | `TrackStateBranch.forward()` |
| `track/state` | `TrackStateCodec.decode()` |
| `pruning/active_dim` | 각 branch |
| `pruning/active_width` | `HBTXRTracker.forward_train()` |

## 9. 이 호출 스택에서 디버깅하기 좋은 지점

문제 위치를 좁힐 때는 아래 순서가 좋다.

1. manifest row
2. dataset `meta`
3. `check_dataloader.py` 결과
4. `outputs.keys()`와 snapshot contract
5. `compute_stage1_losses()` 또는 `compute_stage2_losses()` 입력 key 존재 여부
6. `history.jsonl`의 metric/loss 패턴

## 10. 요약

한 샘플의 실제 호출 흐름은 다음 한 줄로 요약할 수 있다.

```text
manifest row
-> reader + roi + transform + event builder + target builder + assembler
-> search/event/track forward
-> stage1/stage2 loss bundle
-> metrics
-> loss_total
```

즉 이 프로젝트의 중심 ABI는 세 군데다.

- dataset sample contract
- tracker output contract
- stage loss contract

그래서 구조를 더 분리하더라도 이 세 ABI를 유지하면 시스템 전체는 안정적으로 리팩터링할 수 있다.
