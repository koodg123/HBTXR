# 17 · 진행상황 종합 (Progress Synthesis)

2026-07-02. HBTXR(JETCAS 리비전) annotation-analysis 전체 작업의 종합 — 진행상황·핵심발견·결정·계획.
브랜치 `annotation`. 상세는 각 docs 참조.

## 1. 목표
EV-Eye 동공 라벨의 **정밀도/노이즈/불확실성**을 리뷰어 응답용으로 규명하고(재훈련·신규주석 없이), 이를
바탕으로 **개선 GSAM2 라벨 + Eye-ROI crop(마커 제외) + APS·Event + Mask·Ellipse·Motion 라벨**의 새
학습 데이터셋(`Dataset_full_gsam2_subject_independent`)을 구축.

## 2. 완료 워크스트림 (커밋됨)
| # | 워크스트림 | 핵심 결과 | 문서 |
|---|---|---|---|
| 1 | 파이프라인 07→10 (GSAM2 audit·HBTXR pred·label-noise) | GSAM2-vs-GT 0.77px, U-Net 1.58px | docs/07·09·10 |
| 2 | GSAM2 harness + 마커/홍채 수정 | mislabel **161→4**(0.09%), iris **0** | docs/12·13·14 |
| 3 | 독립 annotator audit 5종 | EllSeg 0.62·RITnet 0.59·Edge 0.72·DeepVOG 0.91·YOLOE 1.40px, 홍채 0 | docs/15 |
| 4 | 데이터셋 provenance | 사람GT sessions 3·predict 4·frames 6, user30 중복 규명 | docs/11 |
| 5 | Precision/Label-noise(3CH) + 10_eval | σ_human·budget·리벗 | docs/10·`results/precision*` |

## 3. 핵심 발견 (Findings)
1. **0.1812px는 dense(U-Net) 라벨·64×64 프레임 수치**(누수 아님). 사람 GT 대비 정직한 오차 **median 5.70px(346)/1.17px(64)** — 단 **samples 1-10=학습 subject라 낙관**(test=37-48).
2. **σ_human ≈ 0.55px(346)/0.12px(64)** (three-cornered-hat, 독립 triple·inter-method RMS 3중 확증). 보고 0.1812는 이 **라벨노이즈 floor에 위치**.
3. **mask = 타원 rasterize 파생**(cv2.ellipse, IoU~0.95+상수 offset). **U-Net은 이 파생 mask로 학습→사람GT와 비독립**.
4. **개선 GSAM2 > U-Net 라벨러**: 사람GT 0.77 vs 1.75px, IoU 0.914 vs 0.853, 계통offset 없음, 홍채 0.
5. **홍채혼동은 ROI+geom로 해결**(pupil-select 불필요). **4 mislabel = 전부 blink(눈감김) 프레임** → QC가 정상 배제.
6. **event엔 정적 마커가 원래 거의 없음**(~0.5%): crop 이득은 APS=마커제거, Event=zoom/re-centering.
7. **Motion**: smooth는 **속도로 분리 불가**(median 0.14=fixation)→**세션 태스크(2_0_1/2_0_2) 정의**; fixation/saccade=I-VT; blink=검출실패/면적붕괴.
8. **eye/marker 공간분리**: 동공 center cy≤177 vs 마커 cy≥189 → crop으로 배제 가능(centers_clipped 0).

## 4. 결정 로그 (사용자)
- 평가 프레임 **346×260**(64×64 병기) · **samples 1-10 유지**(STEP5만 낙관 캐비앳) · 3CH 독립성 무시 · human-repeatability=개선 GSAM2.
- 데이터셋: **APS+Event 둘 다**, **Mask+Ellipse 둘 다**, **원본 ts/사람GT 보존**, **Motion(fix/smooth/sac/blink)**.
- 해상도: **저장 native 240×160**, 모델입력 128×96/144×96.

## 5. 현재 워크스트림 — Target Dataset (docs/16)
### 완료 (미커밋 → 이번 커밋 포함)
- **crop 파이프라인** `crop_dataset.py`(240×160, 마커제거, centers_clipped 0, event 60%) + `event_overlay.py`.
- **타깃 빌더** `build_target_dataset.py` — FACET memmap 호환 캐시: `cached_data`(event)·`cached_aps`·`cached_ellipse`(+qc)·`cached_mask`(packed-bit)·`labels_original`(human_ellipse·frame_index·motion_labels) + manifest/crop_boxes/qc.
- **motion 라벨러** `motion_label.py`(I-VT+세션+blink, 검증 대각선우세: fix98.6%·smooth93.2%·sac버스트10.6%).
- **전 프레임/이벤트 GSAM2 mask 오버레이** `render_all_masks.py`(483 윈도우 몽타주+event → `_qc_masks/`).
- **Phase 0 PoC**(audit 4669f): valid 98.95%, 로드백·정합 검증(ellipse↔human GT 0.16px·un-crop 정확·mask/ellipse/human 오버레이 일치).

### 라벨/모달리티 현황 (완비·검증)
| Ellipse | Mask | Motion(4-class) | 원본 사람GT/ts | QC(valid/mislabel/blink) | APS | Event |
|---|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## 6. 계획 / 남은 것
| Phase | 내용 | 상태 |
|---|---|---|
| 0 | PoC + 포맷 검증 | ✅ |
| 1 | val(33-36,123k) 구축 → 소규모 학습 검증 | ⛏ |
| 2 | test(37-48,366k) + 사람GT | ⛏ |
| 3 | **train(1-32,969k) 전면 = /mnt/e 전 프레임 GSAM2 라벨링** | ⛏ (유일 대량 GPU) |
| — | FACET 로더 APS/mask 분기 · per-session crop box · blink 정교화 · event 프레임별 time-bin | ⛏/⏸ |

**결론**: 설계·빌더·5종 라벨·양 모달리티·시각화·PoC 검증 **완비**. 남은 것은 **/mnt/e 전 프레임 GSAM2 대량 라벨링(Phase 3)**뿐.

## 7. 산출물 맵
- **코드**: `src/` (io_schema·align·repeatability·reproducibility·accuracy_view·corrected_error·frame64·figures·tables_out·run_all / crop_dataset·event_overlay·build_target_dataset·motion_label·render_all_masks) + `scripts/`(01–15·10b).
- **문서**: `docs/00–17`, `results/precision_full_report{,_ko}.md`, `results/packages/`(INTERPRETATION 등), 타깃 README(`/mnt/e/.../Dataset_full_gsam2_subject_independent/`).
- **결과**: `results/`(label_noise·shape·eval·precision) + `datasets/eye_crop_240x160/`·`Dataset_full_gsam2_poc/`(벌크는 gitignore).
- **환경**: `.venv-gsam2`(torch2.5.1+h5py/scipy/cv2/skimage/sklearn/ultralytics), `.venv-deepvog`(TF2.15).

## 8. 커밋 이력 (branch annotation)
- `adfd2be` 파이프라인+iris/marker+annotator audit
- `4914dbf` precision/label-noise(3CH)+10_eval+packages
- `8af93a2` eye-ROI crop 파이프라인
- (이번) target 빌더+motion+mask오버레이+docs/16·17
