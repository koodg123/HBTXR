# 외부 패키지 운영 가이드

최종 갱신: 2026-04-09 KST

## 목적

이 문서는 `HBTXR_v3_0`에서 외부 저장소를 어떻게 통합하고 선택하는지 정리한 운영 가이드다.

핵심 원칙은 아래와 같다.

- `Third/`가 통합된 heavyweight runtime의 공식 vendor tree다.
- `HBTXR_v3_0/packages`는 upstream registry 및 fallback hub다.
- `/home/kjm26/project/PRJXR/HBTXR_MERGE/packages`는 migration source로만 취급한다.
- 외부 코드는 `src/`로 넣지 않고 `Third/`에 repo root 형태로 vendor한다.
- HBTXR는 manifest, path resolution, backend selector, wrapper 문서를 내부에서 관리한다.

## 1. 현재 통합 대상

### 1.1 Annotation backends

- `groundedsam`
  - repo: `Grounded-Segment-Anything`
  - dir: `packages/Grounded-Segment-Anything`
  - 기본 annotation backend
- `ultralytics_sam3`
  - repo: `ultralytics`
  - dir: `packages/ultralytics`
  - `SAM3` 기반 annotation backend
- `groundedsam2`
  - repo: `Grounded-SAM-2`
  - dir: `packages/Grounded-SAM-2`
  - 병렬 annotation backend

### 1.2 Frame interpolation backends

- `timelens`
  - repo: `timelens`
  - dir: `packages/timelens`
  - 기본 frame interpolation backend
- `timelens_xl`
  - repo: `TimeLens-XL`
  - dir: `Third/FI`
  - 확장 interpolation backend

### 1.3 Event generation backend

- `v2e`
  - repo: `v2e`
  - dir: `Third/EI`
  - synthetic event generation backend

`v2e`는 `timelens` 대체물이 아니라, interpolated frame 이후 synthetic event를 만드는 별도 backend로 취급한다.

## 2. 기본 selector 정책

- annotation backend 기본값: `groundedsam`
- frame interpolation backend 기본값: `timelens`
- event generation backend 기본값: `none`

지원 selector:

- annotation
  - `groundedsam`
  - `ultralytics_sam3`
  - `groundedsam2`
- frame interpolation
  - `linear_blend`
  - `timelens`
  - `timelens_xl`
- event generation
  - `none`
  - `v2e`

## 3. 경로 해석

공통 경로 해석은 [path_utils.py](../../src/hbtxr/preprocess/path_utils.py)에서 수행한다.

자동 탐지 대상:

- `Third/FI`
- `Third/EI`
- `packages/Grounded-Segment-Anything`
- `packages/ultralytics`
- `packages/Grounded-SAM-2`
- `packages/timelens`
- `packages/TimeLens-XL`
- `packages/v2e`

주요 paths config field:

- `groundedsam_root`
- `ultralytics_root`
- `groundedsam2_root`
- `timelens_root`
- `timelens_xl_root`
- `v2e_root`

예시는 [configs/paths.example.json](../../configs/paths.example.json)을 본다.

## 4. 실제 사용 위치

### 4.1 Annotation

- CLI
  - [annotate_groundedsam_ev_eye.py](../../scripts/annotate_groundedsam_ev_eye.py)
  - [build_groundedsam_dataset.py](../../scripts/build_groundedsam_dataset.py)
- runtime façade
  - [annotation_backends.py](../../src/hbtxr/preprocess/annotation_backends.py)
  - [groundedsam_build.py](../../src/hbtxr/preprocess/groundedsam_build.py)

### 4.2 Interpolation

- CLI
  - [prepare_ev_eye.py](../../scripts/prepare_ev_eye.py)
  - [canonicalize_hbtxr.py](../../scripts/canonicalize_hbtxr.py)
- runtime façade
  - [interpolation.py](../../src/hbtxr/preprocess/interpolation.py)

### 4.3 Event generation

- runtime façade
  - [event_generation.py](../../src/hbtxr/preprocess/event_generation.py)
  - [target_fps_build.py](../../src/hbtxr/preprocess/target_fps_build.py)

## 5. Sync 운영

기본 sync:

```bash
sh scripts/run_sync_packages.sh
```

annotation category만 sync:

```bash
PYTHONPATH=src .venv/bin/python scripts/sync_packages.py --category annotation
```

manifest 확인:

```bash
PYTHONPATH=src .venv/bin/python scripts/sync_packages.py --dry-run --print-manifest
```

manifest source는 [packages/repositories.json](../../packages/repositories.json)이다.

## 6. 운영 판단 기준

- main pipeline에서 바로 쓰는 패키지는 selector와 path resolver에 연결한다.
- 실험 후보 패키지도 manifest에는 올리되, 기본 selector는 보수적으로 둔다.
- 외부 repo 구조가 달라도 HBTXR 쪽 façade가 공통 산출물 contract를 맞춘다.
- root `/packages`는 바로 제거하지 않고 migration source로 유지한다.
