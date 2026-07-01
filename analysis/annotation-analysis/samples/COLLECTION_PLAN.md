# 샘플 수집 계획 (Sample Collection Plan)

리뷰어 코멘트(annotation precision / label noise / label uncertainty) 대응을 위한 **audit 표본 수집** 설계.
출력 위치: `.../HBTXR/analysis/annotation-analysis/samples/`

## 확정된 전제 (Locked assumptions)
- **Test GT = original EV-Eye VIA 동공 타원 주석으로 동결.** U-Net / Grounded-SAM2는 **audit 도구**로만 사용(GT 교체 금지).
- **현재 0.1812 px는 Grounded-SAM dense 라벨 대비로 측정됨 → self-consistency(누수)**. 따라서 primary accuracy는 **original EV-Eye GT가 있는 keyframe에서만** 재계산한다(dense 아님).
- **사람 재주석 불가** → 사람 inter/intra-annotator precision은 측정 불가. 대신 **자동 precision proxy** 2종으로 대체하고 한계를 명시한다(아래 §4 Tier B).

---

## 0. 샘플 → 보고 테이블 매핑 (왜 모으는가)

| Tier | 표본 | 채우는 리뷰어 테이블 |
|---|---|---|
| **A** | original GT가 있는 keyframe (모션·피험자 층화) | Label Noise(orig vs unet/gsam2), Label Uncertainty `U_i`·`E_i`, per-subject/per-motion |
| **B** | **고정 응시 구간의 연속 프레임(F2F) 윈도우** | Annotation **Precision proxy** = 참 운동≈0 구간의 center jitter(audit별) |
| **C** | saccade / smooth-pursuit 연속 윈도우 | 모션 의존 uncertainty, error가 noise floor에 근접하는 구간 |

핵심 통찰: 사람이 없어도 **고정(fixation) 구간에서는 동공이 거의 정지** → 그 구간의 프레임 간 center 변동이 곧 그 annotator의 **정밀도 proxy**가 된다. 이것이 "state별 F2F 시퀀스 균일 표집"의 과학적 근거다.

---

## 1. 소스/감사 정의 (디스크 경로)

```
y_orig  : original EV-Eye VIA 타원 (GT, frozen)
          E:\DATASET\eveye\raw_data\Data_davis\userN\{eye}\{session}\user_N.csv
          (+ 파생 마스크: raw_data\Data_davis_labelled_with_mask\{eye}\userN_session_X.h5)
y_unet  : EV-Eye U-Net 예측 마스크 → ellipse-fit center (audit, 디스크에 이미 존재)
          E:\DATASET\eveye\processed_data\Data_davis_predict\userN\userN\{eye}\{session}\predict\{idx}_{ts}_mask.gif
y_gsam2 : Grounded-SAM2 예측 마스크 → ellipse-fit center (audit, 본 수집 후 별도 실행으로 생성)
y_pred  : HBTXR 모델 예측 center (사용자 제공/실행; pred/ 에 채움)
frames  : E:\DATASET\eveye\raw_data\Data_davis\userN\{eye}\{session}\frames\{idx}_{ts}.png (+ timestamps.txt)
events  : ...\{session}\events\events.txt (event-mode E_i용, 옵션)
```

> **center 추출 단일화(필수)**: 모든 소스에 **동일 ellipse-fit center 정의**를 적용한다. `y_orig`는 이미 타원중심(cx,cy)이라 재적합 불필요하지만, U-Net/GSAM2 자유형 마스크는 `mask→contour→fitEllipse→center`로 통일한다(centroid와 혼용 금지). 형상차(타원 vs 자유형)는 bias이지 noise가 아님.

---

## 2. 표집 모집단 & test-split 주의

- 모집단 = **paper의 test split에 속한** 피험자/시퀀스 중 **`y_orig`가 있는 프레임**(`region_count>0`).
- `session_1_0_1`은 **GT 없음 → 제외**. GT는 `session_1_0_2`(saccade+fixation), `2_0_1`·`2_0_2`(smooth pursuit)에만.
- ⚠ **subject-independent 유지**: audit 표본은 반드시 test 피험자에서만 뽑는다(학습 피험자 혼입 금지). → `config`에 `test_users` 리스트를 입력받는다(미지정 시 전체에서 뽑되 manifest에 split 플래그를 남겨 후처리 가능).

---

## 3. 층화 설계 (Stratification)

**축**: `subject(user) × eye(left/right) × motion_bin`.

**motion_bin 유도** (GT가 희소하므로 **U-Net dense center 시계열의 속도**로 산출):
```
v_t = || c_t - c_{t-1} ||2   (U-Net per-frame center, px/frame)
fixation : 윈도우 내 v_t 가 v_fix 이하로 지속        (기본 v_fix = 0.5 px/frame)
saccade  : 윈도우 내 peak v_t 가 v_sac 이상           (기본 v_sac = 5 px/frame)
smooth   : session_2_0_1/2_0_2 (설계상 추종) + 중간 속도 지속
blink/occlusion : U-Net 마스크 면적 < a_min 또는 타원 부적합 → 별도/제외
```
- 임계값은 데이터로 **자동 보정**(예: 전체 v_t 분포의 하위 20% = fixation, 상위 5% = saccade) 옵션 제공.
- blink는 `blink_flag`로 표시하고 precision/`E_i`/`U_i`에서 **동일 기준으로 제외**(별도 bin 보고 가능).

---

## 4. 3개 Tier 표본 (기본 예산, 조정 가능)

### Tier A — GT-anchored keyframes (가장 중요)
- 목적: Label Noise, `U_i`, 재계산 `E_i`, per-subject/per-motion.
- 규모(기본): **≈1,500 프레임**, motion 3종 균형(각 ≈500), `≤40 keyframe/ (subject×eye)`, test 피험자에 고르게.
- 조건: `region_count>0` (y_orig 존재), `blink_flag=False`, U-Net 유효.

### Tier B — fixation F2F 윈도우 (precision proxy, no-human 대체)
- 목적: **Annotation Precision proxy** = 고정 구간 center jitter(audit별 F2F std).
- 규모(기본): **16 subject × 2 eye × 2 window × 11 frame ≈ 700 프레임**.
- 윈도우: `session_1_0_2`에서 v_t가 v_fix 이하로 ≥11프레임 지속하는 구간을 탐지, 그 연속 프레임 전체 수집(GT 불필요).

### Tier C — saccade / smooth-pursuit 윈도우
- 목적: 모션 의존 uncertainty, error≈noise floor 구간.
- 규모(기본): **12 subject × {saccade, smooth} × 2 window × 11 frame ≈ 530 프레임**.
- 윈도우: saccade=`1_0_2`의 고속 구간, smooth=`2_0_1/2_0_2` 추종 구간.

**총합 ≈ 2,700 프레임** (Grounded-SAM2 실행 부담 고려한 기본값; `--scale`로 비례 확대/축소).

> **Precision 테이블 처리**: "Human A/B", "intra-annotator" 행은 **N/A(사람 불가)** 로 명시하고, ① Tier B 고정 F2F jitter(U-Net, GSAM2), ② GSAM2 perturbation(같은 프레임, prompt jitter/TTA) 두 proxy로 대체. 리벗에 "automated proxy, not human precision" 한계를 명기.

---

## 5. 샘플별 수집 내용 & 정렬 키

**canonical key**: `u{ID}_{eye}_{session}_{idx6}`  예) `u1_left_s1_0_2_000123`

**정렬(조인) 규칙** — 4개 모달리티를 frame **index**로 결합:
```
frame PNG  : {idx}_{ts}.png            → idx, ts 파싱
timestamps : frames/timestamps.txt     → idx→ts 확인
y_orig     : user_N.csv                 → ⚠ csv filename의 ts는 frame 폴더와 미세 상이 → "index"로 매칭
y_unet     : predict/{idx}_{ts}_mask.gif→ idx로 매칭 (userN\userN 이중경로 주의)
mask h5    : Data_davis_labelled_with_mask\*.h5 → 내부 인덱싱 스키마는 수집 시 점검(02 스크립트 layout 활용)
events     : events.txt → keyframe ts ±W ms 윈도우(옵션, event-mode E_i)
```

**각 샘플에 저장**: 원본 프레임 PNG(복사), `y_orig` 타원(json), `y_unet` center(+옵션 마스크), `y_gsam2`/`y_pred`는 placeholder(후속 실행이 채움), 옵션 event 윈도우(npz), 메타(모션/품질플래그/blink/tier/window_id).

---

## 6. 디렉터리 레이아웃 & manifest 스키마

```
samples/
├─ COLLECTION_PLAN.md          # 본 문서
├─ manifest.csv                # 프레임 1행 (모든 메타 + center 컬럼)
├─ manifest.json               # 풍부한 per-sample 레코드 + provenance
├─ frames/   {key}.png         # 원본 프레임 복사
├─ gt/       {key}.json        # y_orig 타원 (cx,cy,rx,ry,theta,quality)
├─ unet/     {key}.json        # y_unet center (+ 옵션 mask)
├─ gsam2/    (후속 실행이 채움) # y_gsam2 mask+center, perturbation 반복본
├─ pred/     (사용자 모델이 채움)# y_pred center (frame/event/hybrid 모드별)
├─ events/   {key}.npz         # 옵션: keyframe 주변 event 윈도우
└─ meta/
   ├─ strata_summary.csv       # (subject,eye,motion) bin별 카운트
   ├─ selection_config.json    # seed, 임계값, test_users, 예산
   └─ rejections.csv           # 제외 프레임 + 사유(no GT/blink/invalid ellipse/unet fail)
```

**manifest.csv 컬럼**:
```
key, tier(A/B/C), user, eye, session, motion_bin, window_id,
frame_index, frame_ts_us, split(test/other/unknown), blink_flag,
has_gt, gt_cx, gt_cy, gt_rx, gt_ry, gt_theta, q_good, q_frontal, q_illum,
unet_valid, unet_cx, unet_cy, unet_area,
gsam2_valid, gsam2_cx, gsam2_cy,           # 후속 채움
pred_cx, pred_cy, pred_mode,                # 후속 채움
path_frame, path_gt, path_unet, path_events
```

---

## 7. 선택 알고리즘 (결정론적·재현가능)

1. `selection_config`(seed=20260630, 임계값, test_users, 예산) 고정.
2. 모집단 스캔: GT keyframe 목록 + U-Net dense center로 motion_bin·blink 산출.
3. **층화 균등 표집**: (subject×eye×motion) 셀에서 라운드로빈, subject당 캡 적용(특정 피험자 지배 방지).
4. Tier B/C: 모션 조건 만족 **연속 윈도우** 탐지 후 윈도우 단위로 수집.
5. 동일 seed로 항상 같은 표본 재현. 전 과정 `meta/`에 기록(provenance).

---

## 8. 품질 게이트 / 제외 (투명성)
- 제외 사유 코드: `no_gt`, `blink`, `invalid_ellipse`, `unet_invalid`, `out_of_split`.
- 모두 `meta/rejections.csv`에 카운트 → 리벗의 **valid-rate** 보고에 사용.
- ⚠ Grounded-SAM2는 일반 모델이라 근적외·깜빡임·속눈썹에서 실패 가능 → valid-rate가 낮으면 해당 audit 비중 하향(보고에 명시).

---

## 9. 기본 파라미터 (조정 노브)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `seed` | 20260630 | 재현성 |
| `tierA_total` | 1500 | GT keyframe 총수(모션 균형) |
| `per_subject_eye_cap` | 40 | 피험자 지배 방지 |
| `window_len K` | 11 | F2F 윈도우 길이(홀수) |
| `tierB_subjects/windows` | 16 / 2 | 고정 precision proxy 표본 |
| `tierC_subjects/windows` | 12 / 2 | saccade·smooth 표본 |
| `v_fix / v_sac` | 0.5 / 5 px·f⁻¹ | 모션 임계(자동보정 옵션) |
| `a_min` | (자동, 면적분포 하위 X%) | blink 판정 |
| `event_window_W` | ±10 ms | event-mode 윈도우(옵션) |
| `test_users` | (입력) | subject-independent 보장 |

---

## 10. 이 단계 산출물 & 다음 단계
- **산출물**: 위 레이아웃의 `samples/` (frames/gt/unet 채움 + manifest + meta).
- **다음**: ① `gsam2/` 채우는 Grounded-SAM2 실행 스크립트(+perturbation), ② `pred/`에 HBTXR 예측 주입, ③ precision/noise/uncertainty 3표·`E_i`/`U_i`/Spearman·CDF/scatter 계산(cluster-bootstrap CI), ④ 리벗 단락 초안.
- 수집기는 `scripts/07_collect_samples.py`로 구현 예정(기존 `evlib.py` 재사용; `--root`, `--out samples`, `--test-users`, `--scale`).

> 확인 필요: (a) **test split 피험자 목록**(subject-independent 보장용), (b) **기본 예산(~2,700 프레임)** 수용 여부 또는 `--scale` 값.
