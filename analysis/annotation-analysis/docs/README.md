# docs/ — Annotation Precision & Label Noise 리벗 작업 문서

리뷰어 코멘트("annotation precision / label noise / label uncertainty를 보고하라") 대응 작업의 전체 기록.
생성/갱신: 2026-07-01.

## 문서 목록
| 파일 | 내용 |
|---|---|
| `00_goal_and_context.md` | 목표·배경(리뷰어 코멘트, 0.1812 px 문제, 데이터셋 실체) |
| `01_status_and_roadmap.md` | 현재 진행상황 + 향후 계획(파이프라인 08~10, 단계별 상태) |
| `02_decision_log.md` | 핵심 결정사항(왜 그렇게 정했는지) |
| `03_worklog.md` | 세션 로그 + 대화 흐름(질문·답·조치 시간순) |
| `04_samples_dataset.md` | **samples/ 데이터 문서**(구조·스키마·라벨 구성·통계·좌표계·provenance) |
| `05_environment_and_gotchas.md` | 실행 환경 제약(bash/UNC/WSL) + 발견된 버그·주의점 |
| `06_analysis_design.md` | 평가 설계(3표·E_i/U_i/Spearman·bootstrap·리벗 문구) |
| `07_gsam2_audit_results.md` | **08 GSAM2 실행 결과**(env 레시피·프롬프트 수정·483 valid 100%·label-noise 0.75px·오검출 tail) |
| `08_perframe_dataset.md` | **per-frame 트리**(samples/perframe: 프레임별 gt/unet/gsam2 center·mask·bbox) + 오버레이(overlay/) + annotator 도구 서베이(annotators/) |
| `09_hbtxr_pred_results.md` | **09 HBTXR 예측 결과**(img64 subject-indep ckpt, E_orig median 5.70px vs sparse 사람 GT; 0.1812는 dense(U-Net) 라벨 기준 결과 — 별개 reference, 누수 단정 안 함) |
| `10_gsam2_mislabel_analysis.md` | **GSAM2 mislabel 분석**(비율 ~3%, 원인=반가림 동공→하단 마커 오검출, 해결=U-Net 교차검증 게이트 적용) |
| `11_eveye_dataset_provenance.md` | **EV-Eye 데이터셋 provenance**(U-Net=EV-Eye 확인, data/label 개수, 세션 커버리지 프레임6/predict4/사람3, predict 개수 불일치·user30 중복 규명, 1:1 대응 1,436,776) |
> 평가 결과 수치는 `../results/`: `label_noise_gt_unet.md`, `label_noise_gt_unet_gsam2.md`(GT-vs-U-Net/GSAM2 mean/median/p95/p99/std, 전체+모션별) + `plots/`.

## 한 줄 요약
> HBTXR가 보고한 **0.1812 px**는 **U-Net dense 의사라벨(DeanDataset_full_unet) 기준**으로 측정된 값. 정직한 비교 대상 GT는 **사람 VIA/마스크 키프레임(~9,011, DavisWithMaskDataset_labelled_subset)**. 이를 GT로 고정하고, U-Net·Grounded-SAM2는 audit로만 써서 **annotation precision / label noise / label uncertainty**를 분리 보고하고, **원래 사람 GT 기준으로 primary accuracy를 재계산**해 dense-label 기준(0.1812)과 나란히 비교한다(누수로 단정하지 않음 — 사용자 정정 2026-07-01).

## 관련 위치
- 분석 보고서: `../01_v2e_codebase_analysis.md`, `../02_eveye_dataset_analysis.md`, `../03_annotation_tool_evaluation.md`
- 수집 계획: `../samples/COLLECTION_PLAN_v2.md`
- 스크립트: `../scripts/` (01~14 + evlib + 07b + run_all; 08 GSAM2, 09 HBTXR, 10 label-noise, 11 perframe, 12 overlay, 13/14 mislabel)
- 가중치: `../weights/` (gsam2/ SAM2.1+GDINO · hbtxr/ · EllSeg·RITnet·DeepVOG·Edge-Guided + ANNOTATOR_WEIGHTS.md)
- 산출: `../samples/`(frame·event·label·**perframe**/ + manifests) · `../overlay/`(오버레이) · `../results/`(label_noise_*.md + plots/) · `../annotators/`(도구 서베이)
