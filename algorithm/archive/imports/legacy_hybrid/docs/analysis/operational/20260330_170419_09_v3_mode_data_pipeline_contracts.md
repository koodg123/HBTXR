# HBTXR_v3_0 Mode 데이터 파이프라인 계약

최종 갱신: 2026-03-28 KST

Role: `데이터 파이프라인 분석가`, `이벤트 비전 연구자`, `XR 트래킹 아키텍트`

## 요약

이 문서는 현재 데이터 계약을 다음 세 경로 기준으로 정리한다.

- `mode1`
- `mode2` materialized interpolation
- `mode2` lazy target-FPS execution

초점은 데이터 측면만이다.

- dataset build
- canonical surface
- manifest surface
- dataset / dataloader 동작
- mode-stage preset의 실제 연결 방식
- 각 경로가 생성하는 산출물
- 대표 raw JSON / JSONL 구조

주 코드 기준:

- [canonicalize.py](../../src/hbtxr/preprocess/canonicalize.py)
- [build_manifests.py](../../src/hbtxr/preprocess/build_manifests.py)
- [target_fps_build.py](../../src/hbtxr/preprocess/target_fps_build.py)
- [target_fps_canonical.py](../../src/hbtxr/preprocess/target_fps_canonical.py)
- [dataset.py](../../src/hbtxr/data/dataset.py)
- [loader.py](../../src/hbtxr/data/loader.py)

## 분석 기준

중요 진입점:

- loader mode 기본값
  - `src/hbtxr/data/loader.py:35`
- manifest mode 기본값
  - `src/hbtxr/preprocess/build_manifests.py:15`
- mode2 interpolation canonical path
  - `src/hbtxr/preprocess/canonicalize.py:115`
- target-FPS build
  - `src/hbtxr/preprocess/target_fps_build.py:1346`
- target-FPS canonical bridge
  - `src/hbtxr/preprocess/target_fps_canonical.py:164`
- dataset event / frame loading
  - `src/hbtxr/data/dataset.py:640`
  - `src/hbtxr/data/dataset.py:692`
  - `src/hbtxr/data/dataset.py:843`

## 1. 현재 active mode 목록

| Data Path | Canonical | Manifest | Frame Source | Dataset Class | Typical Use |
|---|---|---|---|---|---|
| `mode1` | `canonical1` | `manifest1` | `original` | `Mode1Dataset` | original-frame baseline |
| `mode2` materialized | `canonical2` | `manifest2` | `interpolated` | `Mode2Dataset` | synthetic / interpolated frame path |
| `mode2` lazy target-FPS | `canonical2` | `manifest2` | `target_fps_grid` | `Mode2Dataset` | 고율 Stage2를 위한 lazy path |

## 2. Loader 기본값

loader는 `mode`를 아래 기본 조합으로 묶는다.

| Mode | Default `canonical_name` | Default `manifest_name` | Default `frame_source` | Dataset Class |
|---|---|---|---|---|
| `mode1` | `canonical1` | `manifest1` | `original` | `Mode1Dataset` |
| `mode2` | `canonical2` | `manifest2` | `interpolated` | `Mode2Dataset` |

중요 메모:

- `mode2_execution`은 선언형 mode flag로 dataset에 전달된다.
- 실제 lazy frame 동작은 row / store metadata의 `session_store_layout`이 최종 스위치 역할을 한다.

## 3. 데이터 경로별 end-to-end build 흐름

| Data Path | Build Entry | Canonical Build | Manifest Build | Dataset Load Behavior |
|---|---|---|---|---|
| `mode1` | raw EV-Eye + annotation source | `canonicalize.py` with `data_mode=mode1`, `canonical_name=canonical1`, `frame_source=original` | `build_manifests.py` with `data_mode=mode1`, `manifest1` 생성 | `frame_path` 직접 load, manifest `event_window`로 event input 생성 |
| `mode2` materialized | raw EV-Eye + interpolation backend | `canonicalize.py` with `data_mode=mode2`, `canonical_name=canonical2`, `frame_source=interpolated` | `build_manifests.py` with `data_mode=mode2`, `manifest2` 생성 | interpolated `frame_path`를 load하고 `raw_window` 또는 `source_pair_average`로 event 생성 |
| `mode2` lazy target-FPS | raw EV-Eye -> target-FPS session store -> canonical bridge | `target_fps_build.py` 후 `target_fps_canonical.py`, 여전히 `canonical2`를 목표로 함 | target-FPS canonical row 기반 `manifest2` 생성 | `frame_path`가 없고 `session_store_layout=lazy_source_frames`면 source pair에서 on-the-fly frame 합성, event는 session store `event_index_range` 사용 |

## 4. Canonical 계약 차이

### 4.1 `mode1`

| 항목 | 계약 |
|---|---|
| `data_mode` | `mode1` |
| `canonical_name` | `canonical1` |
| `frame_source` | `original` |
| frame file | original sensor-aligned frame를 링크 또는 복사 |
| interpolation metadata | 필요 없음 |
| synthetic flag | 사용 안 함 |

### 4.2 `mode2` materialized

| 항목 | 계약 |
|---|---|
| `data_mode` | `mode2` |
| `canonical_name` | `canonical2` |
| `frame_source` | `interpolated` |
| frame file | synthetic interpolated frame를 이미지 파일로 materialize |
| interpolation metadata | `interpolation_index.json` 및 `interp_alpha`, `interp_model`, `interp_target_fps` 등 row field에 저장 |
| synthetic flag | synthetic row는 `synthetic_frame_flag=True` |

### 4.3 `mode2` lazy target-FPS

| 항목 | 계약 |
|---|---|
| `data_mode` | `mode2` |
| `canonical_name` | `canonical2` |
| `frame_source` | `target_fps_grid` |
| frame file | canonical row의 `frame_path=None` |
| session store | canonical row가 target-FPS session store를 가리킴 |
| layout marker | `session_store_layout`이 `materialized_target_frames` 또는 `lazy_source_frames`를 기록 |
| timing metadata | row가 `sample_timestamp_us`, source pair info, `event_index_range`를 포함 |

## 5. Manifest row 차이

모든 manifest row는 broad structure는 공유하지만, mode별 필드가 달라진다.

| Row Field | `mode1` | `mode2` materialized | `mode2` lazy target-FPS |
|---|---|---|---|
| `data_mode` | `mode1` | `mode2` | `mode2` |
| `canonical_name` | `canonical1` | `canonical2` | `canonical2` |
| `manifest_name` | `manifest1` | `manifest2` | `manifest2` |
| `frame_source` | `original` | `interpolated` | `target_fps_grid` |
| `frame_path` | 존재 | 보통 존재 | 보통 없음 |
| `sample_timestamp_us` | raw/current frame timestamp | synthetic 또는 재사용 timestamp | target-FPS sample timestamp |
| `prev_sample_timestamp_us` | 이전 annotation timestamp | 이전 synthetic/raw sample timestamp | 이전 target-FPS sample timestamp |
| `event_window` | manifest event policy 계약 | manifest event policy 계약 | 여전히 존재하지만 session-store `event_index_range`가 더 강함 |
| `synthetic_frame_flag` | 없음 또는 false | synthetic row는 true | interpolated target-FPS row는 true |
| `interp_alpha` | 사용 안 함 | 존재 | interpolated row에 존재 |
| `source_pair_index` | 사용 안 함 | 있을 수 있음 | 존재 |
| `session_store_layout` | optional | optional | 중요한 runtime switch |
| `synthetic_event_window` | 사용 안 함 | 존재 | 보통 `event_index_range` provenance 포함 |

## 6. Dataset event 생성 경로

dataset은 row마다 event 생성 전략을 선택한다.

| Strategy | Trigger | 의미 | Typical Path |
|---|---|---|---|
| `raw_window` | non-synthetic row 기본값, 많은 `mode1` row | manifest event window로 raw event slice | `mode1`, 단순 `mode2-stage1` |
| `source_pair_average` | `mode2` synthetic row + session-store event index가 지배적이지 않을 때 | source pair anchor 또는 pair-local interval로 synthetic event 표현 생성 | materialized `mode2-stage2` |
| `target_fps_session_store` | row에 `session_store_path`, `event_index_range` 존재 | target-FPS session store에서 preindexed event interval 읽고 on-the-fly voxelize | lazy `mode2-stage2` |

## 7. Dataset frame load 경로

| Frame Load Path | Trigger | 동작 |
|---|---|---|
| direct frame load | row에 `frame_path` 존재 | canonical image 직접 열기 |
| materialized target frame load | session store에 `frame_images` 존재 | materialized target-FPS frame stack 읽기 |
| lazy source-frame synthesis | `session_store_layout=lazy_source_frames` 또는 target frame stack 부재 | source frame pair를 읽고 현재 frame을 on-the-fly 합성 |

## 8. Mode / Stage preset 매핑

| Preset | Data Path | Canonical | Manifest | Frame Source | Event Path |
|---|---|---|---|---|---|
| `mode1_stage1` | `mode1` | `canonical1` | `manifest1` | `original` | 표준 manifest event window, 보통 `fixed_count` |
| `mode1_stage2` | `mode1` | `canonical1` | `manifest1` | `original` | `prev_sample_timestamp_us`와 `sample_timestamp_us` 사이 `interval_all` |
| `mode2_stage1` | `mode2` materialized | `canonical2` | `manifest2` | `interpolated` | synthetic event builder 기본은 `raw_window` |
| `mode2_stage2` | `mode2` materialized | `canonical2` | `manifest2` | `interpolated` | `source_pair_average + pair_local + alpha` |
| `exps/configs/mode2_stage2_lazy_2000fps.yaml` | `mode2` lazy target-FPS experimental preset | `canonical2` | `manifest2` | `target_fps_grid` | `target_fps_session_store`, 보통 `interval_all` 의미와 결합 |

## 9. 세 데이터 경로를 읽는 방법

### 9.1 `mode1`

가장 깨끗한 original-frame baseline이다.

- original canonical surface
- original frame file
- original manifest row
- dataset가 raw event window에서 직접 event tensor 생성

주 사용처:

- `mode1-stage1`
- `mode1-stage2`

### 9.2 `mode2` materialized

고전적인 interpolated-frame 경로다.

- canonicalization이 synthetic frame를 materialize
- manifest row가 synthetic frame file을 직접 가리킴
- dataset는 `raw_window` 또는 `source_pair_average`로 synthetic event tensor를 생성 가능

주 사용처:

- `mode2-stage1`
- 중간 규모 `mode2-stage2`

### 9.3 `mode2` lazy target-FPS

규모 대응용 경로다.

- target-FPS build가 schedule + session store를 생성
- canonical bridge가 store를 참조하는 row를 작성
- manifest row는 sample 설명만 담고 가볍게 유지
- dataset는 source frame pair에서 lazy하게 frame을 합성
- event는 preindexed event range를 기준으로 생성

주 사용처:

- 대규모 target-FPS `mode2-stage2`
- 특히 `2000 FPS` 급 실험

## 10. 실무적 해석

현재 데이터 구조는 layered contract로 이해하는 것이 가장 정확하다.

- `canonical`
  - source-of-truth annotated frame / session unit 정의
- `manifest`
  - 실험 시점 sampling과 event window policy 정의
- `dataset`
  - row field를 보고 frame / event를 실제 tensor로 materialize

중요 포인트:

- `mode1`, `mode2`는 단순 프레임 소스 차이만이 아니라 canonical / manifest / runtime load behavior 전체를 함께 바꾼다.
- 특히 `mode2`는 materialized와 lazy target-FPS가 겉보기 preset은 비슷해도 실제 데이터 경로는 크게 다르다.

## 11. 코드 참조

- [canonicalize.py](../../src/hbtxr/preprocess/canonicalize.py)
- [build_manifests.py](../../src/hbtxr/preprocess/build_manifests.py)
- [target_fps_build.py](../../src/hbtxr/preprocess/target_fps_build.py)
- [target_fps_canonical.py](../../src/hbtxr/preprocess/target_fps_canonical.py)
- [dataset.py](../../src/hbtxr/data/dataset.py)
- [loader.py](../../src/hbtxr/data/loader.py)

## 12. 데이터 경로별 생성 산출물

### 12.1 `mode1` canonicalization 산출물

대표 산출물:

- `sessions/.../frames/*.png`
- `sessions/.../events/events.npz`
- `sessions/.../labels/frame_annotations.jsonl`
- `sessions/.../labels/frame_index.jsonl`
- `sessions/.../labels/session_package.json`
- `indexes/sessions.jsonl`
- `indexes/summary.json`

### 12.2 `mode2` materialized canonicalization 산출물

대표 산출물:

- materialized interpolated frame file
- `interpolation_index.json`
- synthetic row metadata
- `canonical2`용 `frame_annotations.jsonl`, `frame_index.jsonl`, `session_package.json`

### 12.3 `mode1` / `mode2` manifest 산출물

대표 산출물:

- `manifest1/train.jsonl`, `val.jsonl`, `test.jsonl`
- `manifest2/train.jsonl`, `val.jsonl`, `test.jsonl`
- split summary / stats JSON

### 12.4 target-FPS build 산출물

| Scope | Artifact | Format | 의미 |
|---|---|---|---|
| target-FPS root | `indexes/sessions.jsonl` | JSONL | planned target-FPS session job |
| target-FPS root | `indexes/skipped_sessions.jsonl` | JSONL | skipped job |
| target-FPS root | `indexes/annotation_failures.jsonl` | JSONL | aggregated annotation failure row |
| target-FPS root | `build_summary.json` | JSON | plan / materialization summary |
| per session | `session.h5` or `session_arrays.npz` | H5 / NPZ | target-FPS session store |
| per session | `frame_labels.jsonl` | JSONL | target-FPS frame label row |
| per session | `annotation_failures.jsonl` | JSONL | per-session annotation failure |
| per session | `session_meta.json` | JSON | target-FPS session metadata |

### 12.5 target-FPS canonical bridge 산출물

| Scope | Artifact | Format | 의미 |
|---|---|---|---|
| session | `labels/frame_annotations.jsonl` | JSONL | target-FPS frame label을 canonical row로 bridge한 결과 |
| session | `labels/frame_index.jsonl` | JSONL | target-FPS frame index |
| session | `labels/session_package.json` | JSON | session store를 가리키는 canonical session package |
| session | `meta.json` | JSON | target-FPS canonical session meta |
| indexes | `sessions.jsonl` | JSONL | canonicalized target-FPS session |
| indexes | `canonical_skipped_sessions.jsonl` | JSONL | skipped target-FPS canonical session |
| indexes | `canonical_annotation_failures.jsonl` | JSONL | canonical bridge annotation failure |
| indexes | `canonical_summary.json` | JSON | target-FPS canonical summary |

## 13. 대표 JSON / JSONL 구조

아래 block은 축약본이지만 현재 코드의 실제 field 이름을 사용한다.

### 13.1 `mode1` `session_package.json`

```json
{
  "session_key": "user01/left/session_1_0_1",
  "user_id": 1,
  "subject_id": 1,
  "eye": "left",
  "session_code": "1_0_1",
  "data_mode": "mode1",
  "canonical_name": "canonical1",
  "frame_source": "original",
  "sensor_size_wh": [346, 260],
  "eye_region_xywh": [0, 0, 346, 260],
  "events_npz": "sessions/user01/left/session_1_0_1/events/events.npz",
  "frame_index_path": "sessions/user01/left/session_1_0_1/labels/frame_index.jsonl",
  "annotation_store_path": "sessions/user01/left/session_1_0_1/labels/frame_annotations.jsonl",
  "split_policy": "subject_independent_exgaze_with_val",
  "session_motion_regime_prior": {},
  "n_frames": 4980,
  "n_labelled_frames": 4980
}
```

### 13.2 `mode2` `interpolation_index.json`

```json
{
  "enabled": true,
  "model": "linear_blend",
  "backend": "linear_blend",
  "interp_alpha": 0.5,
  "interp_target_fps": 2000.0,
  "interp_fixed_insert": null,
  "interp_count_policy": "round",
  "interp_max_insert": 255,
  "synthetic_overlap_policy": "reuse_event_window",
  "n_source_annotations": 4980,
  "n_synthetic_frames": 397921
}
```

### 13.3 manifest row 예시

```json
{
  "sample_id": "user01__left__session_1_0_1__000123_1659091000500",
  "split": "train",
  "subject_id": 1,
  "data_mode": "mode2",
  "canonical_name": "canonical2",
  "manifest_name": "manifest2",
  "frame_source": "interpolated",
  "sample_timestamp_us": 1659091000500,
  "prev_sample_timestamp_us": 1659091000000,
  "frame_path": "sessions/user01/left/session_1_0_1/frames/000123_1659091000500.png",
  "synthetic_frame_flag": true,
  "interp_alpha": 0.125
}
```

### 13.4 target-FPS `session_meta.json`

```json
{
  "stage": "materialized_session",
  "session_key": "user01/left/session_1_0_1",
  "target_fps": 2000.0,
  "target_step_us": 500,
  "session_store_format": "h5",
  "frame_storage_mode": "lazy_source_frames",
  "n_source_frames": 4980,
  "n_target_frames": 397921,
  "interpolation_backend": "linear_blend"
}
```

### 13.5 target-FPS `frame_labels.jsonl` row

```json
{
  "session_key": "user01/left/session_1_0_1",
  "target_fps": 2000.0,
  "frame_store_index": 123,
  "event_index_range": [90123, 90587],
  "event_count": 464,
  "target_timestamp_us": 1659091000500,
  "source_kind": "interpolated",
  "source_pair_index": [122, 123],
  "interp_alpha": 0.125
}
```

### 13.6 target-FPS canonical `frame_annotations.jsonl` row

```json
{
  "ann_id": "user01__left__session_1_0_1__000123_1659091000500",
  "frame_filename": "000123_1659091000500.png",
  "frame_index": 123,
  "sample_timestamp_us": 1659091000500,
  "data_mode": "mode2",
  "canonical_name": "canonical2",
  "frame_source": "target_fps_grid",
  "annotation_status": "annotated_complete",
  "session_store_path": "sessions/user01/left/session_1_0_1/session.h5",
  "session_store_layout": "lazy_source_frames",
  "frame_path": null,
  "synthetic_frame_flag": true,
  "interp_model": "linear_blend",
  "interp_target_fps": 2000.0
}
```

## 14. 실무 읽기 규칙

산출물을 읽을 때 가장 빠른 mental model은 다음이다.

- `canonicalize.py`
  - canonical session artifact 생성
- `build_manifests.py`
  - 실험 sampling row 생성
- `target_fps_build.py`
  - target-FPS session store와 frame-label schedule 생성
- `target_fps_canonical.py`
  - target-FPS store를 canonical style row로 bridge

row가 다음을 가지면:

- `frame_path`
  - frame pixel이 직접 materialize된 경우가 많다.
- `session_store_path` + `event_index_range`
  - event selection은 session-store 기반이다.
- `session_store_layout=lazy_source_frames`
  - frame pixel은 source frame pair에서 lazy하게 합성된다.
