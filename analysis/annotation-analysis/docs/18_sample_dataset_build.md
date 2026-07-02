# 18 · Sample subject-independent 데이터셋 구축

2026-07-03. 개선 GSAM2 라벨 기반 **sample-scale subject-independent** 데이터셋 `Dataset_sample_gsam2_subject_independent` 구축. 설계배경 [16](16_target_dataset_plan.md), 진행 [17](17_progress_synthesis.md).

## 목표
audit 표본(users 1–10, 4,669f)을 **train**으로, subject-independent **val(33–36)/test(37–48)** 를 신규 수집·라벨링해 3-split 구성. 파이프라인/모델 검증용(표본 규모; 전체는 `Dataset_full_...`).

## Split (leak-free, 70/10/20)
| split | subj | frames | anchors | GSAM2 valid | mislabel | motion(fix/sac/smo/blk) |
|---|---|---|---|---|---|---|
| train | 1–10 | 4,669 | 483 | ~99.0% | 4 | 2746/223/1651/49 |
| val | 33–36 | 667 | 69 | 99.3% | 2 | 385/32/232/18 |
| test | 37–48 | 1,334 | 138 | 99.7% | 0 | 773/60/478/23 |

- windowing: anchor 기준, K=11(fix/smooth)·7(sac) → `anchor/motion × 29 = frames`. val 23/mo·test 46/mo. **leak-free 확인**(교집합 공집합).

## 파이프라인 (val/test 신규; train은 기존 samples/ 재빌드)
1. `07_collect_samples --test-users .. --no-balance --nf/--nsm/--nsac N` → `samples_{val,test}/`
2. `08_run_gsam2 --roi 25,10,325,195 --min-box 18 --geom-select --repeats 0 --save-masks` (label config)
3. `14_flag_gsam2_mislabel` (U-Net xcheck>15px OR det<0.35&area<400)
4. `build_target_dataset --split {val,test} --box 53,28,293,188` (env `AA_SAMPLES=samples_{val,test}`)

## 코드 변경 (split-aware; train 하위호환)
- `io_schema`: `SAMPLES = os.environ.get("AA_SAMPLES") or AA/samples` → `LAB` 파생.
- `build_target_dataset`·`render_all_masks`: `FRAME/EVENT = S.SAMPLES`; unet 로드 tolerant.
- `motion_label`: **속도를 GSAM2 중심 기반**(U-Net 없는 33–48도 saccade 검출; fixation F2F ~0.21px로 clean), unet 로드 tolerant.

## 결과 (검증)
- **subject-independent GSAM2 vs human GT @anchor**: val median **0.712px**, test **0.767px** → 개선 GSAM2가 미학습 subject(33–48)에도 일반화(학습 subject 0.77px와 동급).
- un-crop 정확(`max|Δ|=0.000`), count 일관성(mot=ellipse=qc=aps), mask packbits(4800B/f).
- **mislabel 처리**: motion에서는 blink(코드3)로 흡수(mislabel⊂blink), QC에서는 `mislabel` flag 별도 보존. ※ QC.blink 필드는 미채움 → blink 판정은 `motion_labels` 사용.

## 산출물
- `/mnt/e/DATASET/eveye/target_data/Dataset_sample_gsam2_subject_independent` (389M): `{train,val,test}/{cached_data,cached_aps,cached_ellipse,cached_mask,labels_original}` + `README.md`·`manifest.json`·`crop_boxes.json`·`qc_summary.json` + `_overlays/{split}/*.jpg`(motion별 2개).
- 소스 트리 `samples_{val,test}/`(재생성/오버레이용, gitignore).

## 다음
- HBTXR-imgsz64 훈련 리허설(로더 분기 + 240×160→64×64 anisotropic remap). val/test가 진짜 held-out이라 "GSAM2 라벨이 모델을 개선하나"를 표본 규모 내에서 측정 가능.
