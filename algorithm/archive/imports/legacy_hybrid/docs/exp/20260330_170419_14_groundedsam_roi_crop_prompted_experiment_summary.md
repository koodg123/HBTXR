# Grounded-SAM ROI Crop Prompted 실험 요약

이 문서는 2026-03-28에 수행한 Grounded-SAM ROI-crop 실험, 저장된 옵션 세트, 산출물 루트, 그리고 현재 권장 operating point를 한데 정리한다.

## 범위

- WSL GPU 환경에서 수행한 raw EV-Eye runtime 실험
- 주요 초점
  - Stage1 eye ROI 품질
  - crop-local pupil detection 품질
  - CSV 누락 또는 close-eye 유사 프레임에서의 동작
- 주요 평가 구간
  - 집중 평가: `user05/left/session_101`, `n=16`
  - 공식 `user17/user18` 16-session subset
  - 더 넓은 partial all-session preview batch: `80/388` sessions

## 산출물 루트

- 옵션 설정
  - `configs/groundedsam/`
- 이 sweep의 보관 로그
  - `dataset_session_samples_1/logs/`
- 이 sweep의 보관 preview
  - `workspace_session_samples_1/`

주의:

- 일부 생성 JSON row는 여전히 `/workspace/...`, `/dataset/...` 같은 legacy runtime 문자열을 포함한다.
- 문서화에 사용한 tracked 복사본은 아래 경로 기준이다.
  - `dataset_session_samples_1/`
  - `workspace_session_samples_1/`

## 실험 진행 순서

### 1. Raw-CSV overlay sanity pass

- raw CSV annotation을 사용해 raw-sample overlay를 생성했다.
- 목적
  - Grounded-SAM 동작을 바꾸기 전에 기본 pupil ellipse / bbox rendering 경로가 정상인지 확인

대표 산출물:

- `workspace_session_samples_1/raw_csv_overlay_preview_user05_left/`

### 2. CSV-missing 고정 5단계 Grounded-SAM 경로 재현

- CSV가 없는 세션에 대해 아래 고정 순서를 재현했다.
  1. `eye_region_mask`
  2. `eye_region_bbox`
  3. `eye_region_bbox`로 crop ROI 생성
  4. ROI-local `pupil_mask`
  5. `pupil_bbox`, `pupil_state` 도출
- 집중 평가 대상
  - `user05/left/session_101`, `n=16`

결과:

- `16`개 프레임 모두 runtime은 성공했다.
- 하지만 eye stage가 프레임 대부분을 덮고, pupil stage도 crop ROI 대부분을 먹는 경우가 많아 품질은 낮았다.

대표 산출물:

- `workspace_session_samples_1/groundedsam_no_csv_preview_user05_left_session101/contact_sheet.png`

### 3. bbox-only eye ROI 복구 경로 튜닝

- eye stage를 예전 문서의 bbox-only 경로와 near-full-frame rejection 조합으로 교체했다.
- 핵심 eye-stage 옵션
  - `box_threshold=0.25`
  - `text_threshold=0.25`
  - `nms_threshold=0.80`
  - `max_box_area_ratio=0.85`
  - `border_margin_px=3`
  - `max_border_touches=3`

`user05/left/session_101`, `n=16` 기준 결과:

- near-full-frame eye box
  - `14/16 -> 0/16`
- 남은 oversized pupil case
  - 패널 `1, 2, 6, 8, 14`

대표 산출물:

- note
  - `dataset_session_samples_1/logs/2026-03-28_groundedsam_user05_left_session101_tuned_notes.md`
- preview
  - `workspace_session_samples_1/groundedsam_tuned_preview_user05_left_session101/contact_sheet.png`

### 4. geometry-only safety rule screening

- 튜닝된 `session_101` 샘플에 대해 geometry-only reject rule 여러 개를 점검했다.
- oversized pupil failure는 geometry만으로도 상당히 잘 분리됐다.

유효했던 후보 규칙:

- `crop_mask_area_ratio_max=0.30`
- `crop_bbox_fill_max=0.85`

집중 평가 결과:

- 두 규칙 모두 동일한 5개 oversized panel을 거부했다.
  - `1, 2, 6, 8, 14`

대표 산출물:

- `dataset_session_samples_1/logs/2026-03-28_roi_crop_pupil_reinfer_rule_candidates.json`
- `workspace_session_samples_1/roi_crop_pupil_reinfer_filtered_user05_left_session101_crop_mask_area_ratio/contact_sheet_pairs_clean.png`
- `workspace_session_samples_1/roi_crop_pupil_reinfer_filtered_user05_left_session101_crop_bbox_fill_max/contact_sheet_pairs_clean.png`

### 5. prompted crop-local pupil recovery

- 튜닝된 eye ROI stage는 유지하고, crop-local pupil prompt만 교체했다.
- 사용한 pupil class
  - `small dark circular pupil`
  - `black pupil`
  - `pupil`
- 사용한 threshold
  - `box_threshold=0.30`
  - `text_threshold=0.30`

결과:

- 지나치게 큰 dark-mask failure가 줄어들었다.
- focused session에서 geometry warning과 visually plausible 사례가 더 잘 분리됐다.

### 6. 공식 `user17 / user18` subset 비교

- 공식 16-session subset에서 후보 옵션 조합을 비교했다.
- 이 비교는 “session 하나에만 맞춘 튜닝”이 아니라 공식 slice에서도 재현되는지 보는 용도였다.

해석:

- tuned eye ROI + prompted crop-local pupil 경로가 좁은 단일 세션을 넘어 보다 넓은 subset에서도 재현성이 있었다.

## Partial all-session preview batch

- runner
  - `scripts/run_roi_crop_prompted_preview_batch.py`
- archived partial summary
  - `dataset_session_samples_1/logs/2026-03-28_roi_crop_prompted_preview_all_sessions_partial_80_summary.json`
- archived partial markdown
  - `dataset_session_samples_1/logs/2026-03-28_roi_crop_prompted_preview_all_sessions_partial_80_summary.md`

Partial 결과:

- `80/388` sessions 처리
- `1280` sampled frames
- `1254` completed
- `12` eye-failed
- `14` pupil-failed
- `35` fail-like
- predicted classes
  - `small dark circular pupil=1238`
  - `black pupil=16`

해석:

- prompted ROI-crop 경로는 단일 집중 세션을 넘어 더 넓은 세션군에도 꽤 잘 일반화됐다.
- eye failure는 일부 문제 세션에 집중됐고, 이것이 더 강한 `v2` eye prompt 도입의 근거가 됐다.

## `v2` pupil_failed review

- review files
  - `workspace_session_samples_1/pupil_failed_v2_review/summary.json`
  - `workspace_session_samples_1/pupil_failed_v2_review/summary_with_v5.json`
  - `workspace_session_samples_1/pupil_failed_v2_review/pupil_failed_contact_sheet.png`
  - `workspace_session_samples_1/pupil_failed_v2_review/v5_rescued_contact_sheet.png`
- archived review JSON
  - `dataset_session_samples_1/logs/2026-03-28_pupil_failed_v2_close_eye_review_with_v5.json`

관찰 결과:

- `v2`의 `pupil_failed` 프레임 `17/17`은 positive CSV pupil supervision이 없었다.
- 세부 분해
  - `14` frame: `region_count=0`
  - `3` frame: CSV row 없음
- `16/17`은 해당 세션 샘플의 median보다 더 낮은 eye-height/width ratio를 보였다.
- `v5` fallback은 `2/17`만 rescue했다.
- rescue된 두 frame도 모두 `csv_max_region_count=0`이었다.

해석:

- 남아 있는 `v2` `pupil_failed`는 강한 외부 supervision이 없는 한 close-eye / near-close-eye 후보로 취급하는 것이 맞다.
- 따라서 fallback은 기본 비활성 상태를 유지하는 쪽이 타당하다.

## 현재 권장안

### 기본값으로 사용할 것

- [x] eye stage
  - `v2` stronger eye prompt
- [x] pupil stage
  - `v1/v2` 기반 strict prompted crop-local stage
- [x] post-filter
  - `crop_mask_area_ratio_max=0.30`
  - `crop_bbox_fill_max=0.85`
- [x] fallback
  - 기본 비활성

### 선택적으로만 유지할 것

- [x] `v5` gated fallback
  - 향후 close-eye / blink 정책이 명시적으로 허용할 때만 rescue 경로로 검토

### 기본값으로는 비추천

- [x] `v3`
  - fail-like dark mask가 너무 많음
- [x] `v4`
  - fail-like dark mask가 너무 많음

## 진행 체크리스트

- [x] raw-CSV overlay sanity check 완료
- [x] CSV-missing 5-step Grounded-SAM 경로 재현 및 검토 완료
- [x] bbox-only eye ROI 안정화 검증 완료
- [x] geometry-only reject rule 검토 완료
- [x] focused `session_101`에서 prompted crop-local pupil prompt 검증 완료
- [x] 공식 `user17/user18` subset 비교 완료
- [x] 재사용 가능한 option family `v1` ~ `v5` 저장 완료
- [x] partial broader all-session preview batch 인프라 추가
- [x] `v2` `pupil_failed` close-eye review 완료
- [x] 현재 기본 operating point를 `v2`로 고정
- [~] broader rerun 상태
  - partial `80/388` sweep 완료
  - full all-session `v2` rerun은 당시 미완료
- [ ] production target-FPS label path에 close-eye 정책 기본 반영

## 최종 All-48 / 4-Session `v2` baseline 완료 결과

- 권장 `v2` operating point로 full broader run 완료
  - `48 users`
  - `4 official sessions`
  - `2 eyes`
  - `4 sampled frames per eye-session`
- 최종 sampled frame 수
  - `1536`
- 최종 결과
  - `completed=1506`
  - `eye_failed=0`
  - `pupil_failed=30`
  - `fail_like=28`

Final grouped review 결과:

- `csv_positive=12`
- `no_raw_csv_supervision=1524`
- `blink_labeled=1143`

직접 검토 결론:

- 대부분의 `pupil_failed`는 blink / near-close 예제에서의 보수적 miss로 해석할 수 있다.
- 일부 `pupil_failed`는 사실상 tiny stage-1 eye ROI escape다.
- 대부분의 `fail_like`는 실제 geometry violation이다.
- 드물지만 open-eye hard false positive도 존재하며, 이것이 현재 geometry warning 규칙을 유지해야 하는 이유다.

최종 review 참고 문서:

- 전용 failure review
  - [15_all48_v2_baseline_failure_review.md](20260330_170419_15_all48_v2_baseline_failure_review.md)
- 다른 서버용 handoff
  - [16_all48_v2_handoff.md](20260330_170419_16_all48_v2_handoff.md)

## 갱신된 진행 체크리스트

- [x] full all-user broader `v2` preview rerun 완료
- [x] final grouped review outputs 생성 완료
- [x] final failure review 완료 및 문서화
- [x] doc-local review figure를 committed resources로 반영
- [x] 현재 기본 operating point는 여전히 `v2`
- [~] production integration은 아직 미완료
  - blink / close-eye 정책이 production target-FPS label path에 아직 연결되지 않음
- [ ] minimum eye-ROI sanity gate를 production preprocessing에 통합
