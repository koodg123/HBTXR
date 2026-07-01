# 00 · 목표 & 배경

## 리뷰어 코멘트 (대응 대상)
> "It would also help to report annotation precision and label noise.
> If the label uncertainty is comparable to or larger than the reported error, this should be discussed."

## 목표
1. **Annotation Precision / Label Noise / Label Uncertainty를 각각 다른 목적의 수치로 분리 보고**한다(같은 말처럼 쓰지 않는다).
2. 보고 정확도(**0.1812 px**)를 **원래 EV-Eye 사람 라벨 기준으로 재계산**하고, 그 오차를 **라벨 불확실성과 비교**한다.
3. `E_i ≲ U_i`(모델오차 ≤ 라벨 불확실성)이면 명시하고 논의한다 = "label-noise-limited accuracy".

## 핵심 배경 (왜 이게 문제인가)
- HBTXR/EPNet는 **`DeanDataset_full_unet`** 로 학습됨 — 이는 사람 라벨(마스크)로 학습한 U-Net을 **전체 Data_davis 프레임에 돌려 만든 dense 의사라벨**(mask→ellipse).
- **0.1812 px는 그 U-Net(dense) 의사라벨 기준으로 측정**된 값(= 모델이 dense 라벨과 얼마나 일치하는지). ⚠️ **누수로 단정하지 않음**(사용자 정정 2026-07-01) — 사람 GT와는 다른 reference일 뿐.
- 동공 반경 ~20px에서 0.18px(픽셀의 1%)는 사람 주석 정밀도(보통 ~1–3px)보다 한 자릿수 작음 → dense 라벨(모델 학습 기준)과 사람 GT는 **서로 다른 reference**라 이런 차이가 날 수 있음.

## 데이터셋 실체 (확인된 사실)
| 라벨셋 | 프레임 커버리지 | 출처 | 역할 |
|---|---|---|---|
| VIA 타원(`user_N.csv`) / 마스크(`Data_davis_labelled_with_mask/*.h5`) / `DavisWithMaskDataset_labelled_subset` | **키프레임만(sparse, ~9,011)** | 사람 | **진실(frozen test GT)** |
| `DeanDataset_full_unet` | **전 프레임(dense)** | U-Net 의사라벨 | 0.1812의 기준 라벨(dense) → audit |
| `Data_davis_predict/*.gif` (E: 사본) | 전 프레임(dense) | EV-Eye 공식 U-Net 예측 | audit(y_unet proxy) |

- 근거: `user_1.csv` 실측 — 60프레임 중 `000010`만 타원, 나머지 `region_count=0`(빈 값). EV-Eye README/프로젝트 문서와 일치.
- EV-Eye: 48명 × 양안 × 4세션(1_0_1·1_0_2=saccade+fixation, 2_0_1·2_0_2=smooth pursuit), 마스크는 **1_0_1 제외** 후3세션에만.

## 최종 산출 목표
- **Corrected primary accuracy**(원래 GT 기준, frame/event/hybrid)
- **Annotation Precision 표** / **Label Noise 표** / **Label Uncertainty 표**
- **E_i vs U_i** 관계(Spearman, `E≤U`/`E≤2U` 비율) + CDF/scatter
- **리벗 문구**: "0.1812는 U-Net 의사라벨 대비였고, 9,011 사람 라벨로 재계산했으며, 오차는 라벨 노이즈 수준이다."
