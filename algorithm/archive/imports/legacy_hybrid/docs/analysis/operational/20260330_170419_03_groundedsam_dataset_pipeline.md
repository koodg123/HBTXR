# Grounded-SAM 데이터셋 파이프라인

최종 갱신: 2026-03-26 KST

Role: `데이터셋 파이프라인 아키텍트`, `Grounded-SAM 통합 엔지니어`, `XR 비전 연구 엔지니어`

## 요약

- 이 문서는 `HBTXR_v3_0`의 Grounded-SAM 기반 데이터셋 생성 경로를 정리한다.
- 파이프라인의 핵심 원칙은 외부 raw dataset과 외부 Grounded-SAM repository를 read-only로 유지하는 것이다.
- 생성되는 annotation, canonical, manifest, preview, run artifact는 모두 프로젝트 workspace 안에 기록된다.
- 현재 문서는 실제 script entrypoint, `groundedsam_root`의 의미, annotation export 산출물 구조까지 포함한다.

## 외부 read-only 입력

- Raw EV-Eye dataset
  - `E:/WSL/Shared/dataset/Eye/EV_Eye/raw_data/Data_davis`
- Grounded-SAM repository
  - `E:/WSL/Shared/ETRI_SYNC/HBTXR/annotation_tools/Grounded-Segment-Anything-main`

이 경로들은 보통 아래를 통해 주입된다.

- `configs/paths/ev_eye_groundedsam_paths.json`
- `--paths-config`
- `path_utils.py`가 해석하는 환경 변수

## 쓰기 가능한 출력

- Grounded-SAM annotation
  - `workspace/groundedsam_annotations`
- canonical dataset
  - `workspace/canonical`
- manifest
  - `manifests`
- overlay preview
  - `workspace/previews`
- experiment output
  - `runs/experiments`

## 스크립트 진입 표면

### 1. Annotation-only 경로

```text
run_groundedsam_annotation.sh
  -> annotate_groundedsam_ev_eye.py
  -> resolve_paths()
  -> annotate_dataset_with_groundedsam()
  -> workspace/groundedsam_annotations
```

역할:

- shell wrapper가 가능하면 기본 `--paths-config`를 주입
- Python CLI가 `annotation_root`, `groundedsam_root`를 검증
- `groundedsam_build.py`가 Grounded-SAM을 로드하고 raw session을 순회하며 annotation store와 mask를 기록

### 2. Full build 경로

```text
build_dataset_groundedsam.sh
  -> build_groundedsam_dataset.py
  -> annotate_dataset_with_groundedsam()
  -> canonicalize_dataset(annotation_mode="groundedsam")
  -> build_manifests()
  -> manifests + canonical + summary json
```

역할:

- 필요 시 annotation export 수행
- Grounded-SAM 전용 canonicalization 경로 사용
- downstream train / eval / infer용 manifest 생성

## 파이프라인 단계

```text
raw_data/Data_davis
  -> Grounded-SAM annotation export
  -> workspace/groundedsam_annotations
  -> canonicalize (annotation_mode=groundedsam)
  -> workspace/canonical
  -> build_manifests
  -> manifests/*.jsonl
  -> overlay preview / train / eval / infer
```

## 주요 스크립트

- [run_groundedsam_annotation.sh](../../scripts/run_groundedsam_annotation.sh)
- [annotate_groundedsam_ev_eye.py](../../scripts/annotate_groundedsam_ev_eye.py)
- [build_dataset_groundedsam.sh](../../scripts/build_dataset_groundedsam.sh)
- [build_groundedsam_dataset.py](../../scripts/build_groundedsam_dataset.py)
- [run_canonicalize.sh](../../scripts/run_canonicalize.sh)
- [run_build_manifests.sh](../../scripts/run_build_manifests.sh)
- [overlay_preview.py](../../exps/scripts/overlay_preview.py)
- [run_overlay_preview.sh](../../exps/scripts/run_overlay_preview.sh)
- [run_train.sh](../../scripts/run_train.sh)
- [run_eval.sh](../../scripts/run_eval.sh)
- [run_infer.sh](../../scripts/run_infer.sh)

메모:

- annotation / canonical / manifest / train / eval / infer는 공식 main surface에 남아 있다.
- overlay / preview 계열 helper는 `exps/scripts/` experimental surface로 이동했다.

## `groundedsam_root`의 의미

`groundedsam_root`는 출력 디렉토리가 아니다.

정확한 의미:

- 외부 Grounded-SAM repository root
- `GroundingDINO` Python module을 import하는 기준 경로
- 아래 파일들의 기본 탐색 기준
  - `GroundingDINO/GroundingDINO_SwinT_OGC.py`
  - `groundingdino_swint_ogc.pth`
  - `sam_vit_h_4b8939.pth`

운영적으로 `groundedsam_build.py`는:

- `groundedsam_root` 존재 여부를 검증
- `groundedsam_root`, `groundedsam_root/GroundingDINO`를 `sys.path`에 추가
- 다음을 import
  - `groundingdino.util.inference.Model`
  - `segment_anything.SamPredictor`
  - `segment_anything.sam_model_registry`

즉 `groundedsam_root`는 프로젝트 내부 상태가 아니라 외부 runtime dependency root다.

## Annotation export 레이아웃

annotation export는 session 단위로 정리된다.

```text
workspace/groundedsam_annotations/
  annotation_summary.json
  annotation_sessions.jsonl
  sessions/
    userXX/
      left|right/
        session_<code>/
          frame_annotations.jsonl
          session_summary.json
          masks/
            *.png
```

[annotation_groundedsam.py](../../src/hbtxr/preprocess/annotation_groundedsam.py)가 정의하는 저장 row 계약:

- `ann_id`
- `frame_filename`
- `timestamp_us`
- `eye_region_bbox_xywh_sensor`
- `pupil_mask_path`
- `pupil_region_bbox_xywh_sensor`
- `pupil_ellipse_xywht_sensor`
- `ellipse_sensor_xywht`
- `state_xyabuv`
- `annotation_source`
- `annotation_quality`
- `closed_eye_flag`
- `blink_candidate_score`
- `blink_candidate_reasons`
- `blink_candidate_source`
- `mask_valid`

export 중에 [groundedsam_build.py](../../src/hbtxr/preprocess/groundedsam_build.py)는 세션 로컬 metadata도 같이 기록한다.

- `frame_idx`
- `session_key`
- `user_id`
- `subject_id`
- `eye`
- `session_code`
- `gsam_class_name`
- `gsam_box_xyxy`
- `gsam_box_confidence`
- `gsam_mask_score`

또한 `session_summary.json`, `annotation_sessions.jsonl`은 세션 수준 `blink_candidate_report`도 포함한다.

해석:

- Grounded-SAM 저장소는 더 이상 geometry-only export가 아니다.
- 세션 annotation이 끝난 뒤 export된 ellipse sequence에 대해 세션 로컬 blink heuristic을 다시 적용한다.
- 이 blink field는 annotation store 자체에 다시 써 넣기 때문에 downstream canonicalization이 직접 소비할 수 있다.
- `--overwrite` 없이 재사용하는 기존 store도 blink metadata가 없으면 in-place backfill된다.

## Canonicalization 모드

canonicalizer는 다음을 지원한다.

- `annotation_mode=auto`
  - Grounded-SAM export가 있으면 우선 사용, 없으면 CSV fallback
- `annotation_mode=manual_csv`
  - raw CSV만 사용
- `annotation_mode=groundedsam`
  - 각 세션에 Grounded-SAM export가 반드시 있어야 함

full Grounded-SAM dataset build 경로에서는 [build_groundedsam_dataset.py](../../scripts/build_groundedsam_dataset.py)가 항상 아래 형태를 호출한다.

- `canonicalize_dataset(..., annotation_mode="groundedsam", annotation_root=paths.annotation_root)`

즉 `workspace/groundedsam_annotations` 아래 annotation store가 canonical supervision의 source-of-truth가 된다.

store에 이미 Grounded-SAM blink metadata가 있으면 canonicalization은 이를 그대로 보존한다.  
older store처럼 blink field가 없으면, 저장된 ellipse sequence를 기준으로 동일 metadata를 메모리에서 다시 계산한 뒤 row를 정규화한다.

## 상세 모듈 흐름

```text
annotate_groundedsam_ev_eye.py
  -> preprocess.path_utils.resolve_paths
  -> preprocess.groundedsam_build.annotate_dataset_with_groundedsam
      -> _GroundedSamRuntime
      -> io_utils.collect_users / discover_session_layout / collect_frame_records
      -> annotation_groundedsam.GroundedSamAnnotation
      -> annotation_groundedsam.save_mask
      -> utils.io.write_json / write_jsonl

build_groundedsam_dataset.py
  -> preprocess.groundedsam_build.annotate_dataset_with_groundedsam
  -> preprocess.canonicalize.canonicalize_dataset
  -> preprocess.build_manifests.build_manifests
```

해석:

- `annotation_groundedsam.py`
  - schema와 저장 helper 정의
- `groundedsam_build.py`
  - 외부 repository와 연결되는 실제 runtime bridge
- `canonicalize.py`
  - export된 annotation store를 canonical session package로 변환
- `build_manifests.py`
  - canonical row를 train / val / test manifest row로 변환

## Overlay preview

overlay preview 단계는 다음을 수행한다.

- canonical `sessions.jsonl`을 읽음
- 임의 session 선택
- session당 annotation row 하나 선택
- 다음 overlay 렌더링
  - frame
  - mask overlay
  - eye region box
  - pupil box
  - pupil ellipse

## Runtime 제약

- Grounded-SAM checkpoint는 외부 의존성이므로, 없으면 사용자가 직접 제공해야 한다.
- annotation-only wrapper와 full build wrapper 모두 유효한 `paths-config` 또는 그에 준하는 env / CLI path resolution에 의존한다.
- `build_groundedsam_dataset.py`는 `--skip-annotation`으로 기존 annotation store를 재사용할 수 있다.
- 이 문서는 구현된 script path를 기록한 것이며, 정확도 보고서가 아니다.
- 실제 성능은 checkpoint 유무와 환경 구성에 크게 의존한다.

## Focused blink 검증 스냅샷

store-level blink 경로는 helper-level synthetic row뿐 아니라 작은 실제 inference subset에서도 점검했다.

검증 설정:

- runtime: 저장소 로컬 `.venv`, `cuda:0`, 구성된 Grounded-SAM checkpoint
- subset
  - `user01/left/session101`: frame `005004` ~ `005012`
  - `user01/left/session102`: frame `000309` ~ `000311`, `001409` ~ `001411`

관찰 결과:

- `session101`
  - `9` annotated frame
  - `3` blink-flagged frame
    - `005008_1657710986457638.png`
    - `005009_1657710986497638.png`
    - `005012_1657710986617638.png`
- `session102`
  - `6` annotated frame
  - `2` blink-flagged frame
    - `000311_1657711096857726.png`
    - `001411_1657711140857761.png`

해석:

- `session101`
  - 예전 bbox-only `rejected_large_box(...)` 사례 중 일부만 blink heuristic이 선택했다.
  - 즉 새 분류기가 `bbox reject => blink`보다 더 선택적으로 동작한다는 의미다.
- `session102`
  - Grounded-SAM blink flag는 예전 raw-CSV blink candidate보다 대략 한 frame 뒤에 걸렸다.
  - 두 heuristic은 같은 짧은 blink interval을 가리키지만, source 간 작은 phase shift가 있다.

한계:

- 이것은 focused runtime check이지 full-session 또는 full-dataset 정확도 연구가 아니다.

## Runtime compatibility 메모

현재 `HBTXR_v3_0`는 외부 Grounded-SAM repository 자체를 수정하지 않고도 환경 drift를 견디기 위해 [groundedsam_build.py](../../src/hbtxr/preprocess/groundedsam_build.py)에 작은 compatibility layer를 둔다.

현재 처리 중인 사례:

- 중첩 `segment_anything/segment_anything` package layout
  - 일부 fork에서 `from segment_anything import SamPredictor`가 깨지지 않도록 보정
- `transformers 5.x` BERT helper 변경
  - `get_head_mask` 호환
  - legacy `get_extended_attention_mask(attention_mask, input_shape, device)` 호환
- `groundingdino.util.inference.Model`에 `device` 명시 전달
- `_C` custom-op 부재 fallback
  - GroundingDINO C++/CUDA deformable-attention extension이 없으면 crash 대신 `multi_scale_deformable_attn_pytorch(...)` 경로로 fallback

의미:

- GroundingDINO extension을 다시 빌드하지 않아도 annotation을 진행할 수 있다.
- 단 compiled custom-op 경로보다 느릴 수 있다.
- 최종 runtime 확인을 위해서는 사용자 환경에서 실제 CUDA end-to-end 실행이 여전히 필요하다.
