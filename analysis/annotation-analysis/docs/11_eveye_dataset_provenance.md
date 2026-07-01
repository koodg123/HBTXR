# 11 · EV-Eye 데이터셋 provenance & 개수 (실측)

2026-07-01. 이 세션에서 `/mnt/e`(원본) 접근 가능(과거 Cowork 세션과 달리) → 직접 카운트/검증. **원본은 미수정**(읽기만).

## 1. 데이터셋 3종 — data / label 개수
| 데이터셋 | 경로 | data(프레임) | label | 라벨 성격 | 세션 | 포맷 |
|---|---|---|---|---|---|---|
| **Data_davis** | `raw_data/Data_davis` | **1,506,387** | **9,012** | 사람 VIA **타원** | 프레임 6 / 라벨 3 | png + `user_N.csv` |
| **Data_davis_labelled_with_mask** | `raw_data/...` | **9,011** | **9,011** | 사람 **마스크**(타원→래스터) | 3 | h5 288개(`data`+`label`) |
| **Data_davis_predict** | `processed_data/...` | 1,622,918(raw) / **1,436,776**(정상) | 동일 | EV-Eye **U-Net 예측**(dense) | 4 | `*_mask.gif` |

- Data_davis CSV: 총 1,008,316행(3세션 프레임당 1행) 중 **실제 타원 9,012개**(나머지 `region_count=0`). 세션별 라벨 2,324/4,306/2,382(1_0_2/2_0_1/2_0_2).
- labelled_with_mask: `data`(프레임)·`label`(마스크) 각 shape `(346,260,K)`. 합 9,011(left 4,530/right 4,481; 세션 2,323/4,306/2,382). data=label 완전 일치.
- **사람 라벨 정합**: CSV 타원 9,012 ≈ mask 9,011(1_0_2에서 1개 차이 — 타원 1개가 마스크 미생성).

## 2. U-Net = EV-Eye 것 (확인)
- EV-Eye 코드베이스에 U-Net(`references/codebase/software/EV-Eye/unet/unet_model.py`), README에 `train.py`(학습)·`predict.py`(→`Data_davis_predict`). "U-Net pupil segmentation benchmark".
- 0.1812의 기준 `DeanDataset_full_unet`도 EV-Eye U-Net: `EvEye/utils/scripts/build_full_dean_dataset_with_unet.py`가 `from EvEye.model.DavisEyeEllipse.UNet.UNet import UNet` + trained checkpoint로 전 프레임 추론.
- **내 y_unet = 공식 `Data_davis_predict`**(= EV-Eye U-Net 예측). 학습머신 `DeanDataset_full_unet`(0.1812 생성)의 **proxy**(같은 U-Net, 다른 run/checkpoint → 마스크가 완전 동일하진 않을 수 있음, docs/05 G1).

## 3. 세션 커버리지 (중요)
| 항목 | 세션 수 | 세션 |
|---|---|---|
| Data_davis **프레임** | **6** | 1_0_1, 1_0_2, 2_0_1, 2_0_2, 3_0_1, 3_0_2 |
| **U-Net predict** 라벨 | **4** | 1_0_1, 1_0_2, 2_0_1, 2_0_2 |
| **사람 GT** 라벨 | **3** | 1_0_2, 2_0_1, 2_0_2 |
- U-Net은 사람 라벨 없는 **1_0_1까지 예측**(일부 user 미완성). session_3는 미추론. 사람 라벨은 1_0_1·session_3 없음.
- **내 분석(y_orig/y_unet)은 사람 라벨 3세션만** 사용(anchor는 사람 GT 필요).

## 4. Data_davis_predict 개수 불일치 규명 (⚠️ 데이터 gotcha)
raw predict(1,622,918) > Data_davis 프레임(1,506,387), 차이 +116,531 — **3원인으로 정확 재구성**:
1. **`user30` 손상 (+186,142)**: `user30/user30/user{1..6}/session_1/{eye}/session_X/events/predict/*.gif` — user1~6 predict의 **중첩 중복 복사본**(.rar 오추출 아티팩트). user30 정상본(25,779)과 별개.
2. **session_1_0_1 predict 미완성 (−53,772)**: 정상 predict 1_0_1=428,464 vs DD 482,236. 일부 user의 1_0_1 예측 누락(user1·user2 등은 완전).
3. **session_3 미추론 (−15,839)**: DD엔 3_0_1/3_0_2 있으나 predict 없음.
```
1,622,918 = 정상 1,436,776 + user30중복 186,142
차이 +116,531 = +186,142 − 53,772 − 15,839  ✓ (프레임 단위 정확 일치)
```

## 5. predict 라벨 ↔ 프레임 1:1 대응 = **1,436,776**
파일명(`{idx}_{ts}_mask.gif` ↔ `{idx}_{ts}.png`) 수준 검증 → orphan predict = 0(predict ⊆ DD).
| 세션 | 1:1 대응 | DD 프레임 | 커버리지 |
|---|---|---|---|
| 1_0_2 / 2_0_1 / 2_0_2 | 264,251 / 480,760 / 263,301 | 동일 | **100% 정확 일치** |
| 1_0_1 | 428,464 | 482,236 | 부분(일부 user 누락) |
| **합** | **1,436,776** | | |
- Data_davis 프레임 중 predict 있는 것 = 1,436,776 / 없는 것 = 69,611(=53,772 1_0_1 + 15,839 session_3).
- user30 중복 186,142는 이미 커버된 프레임의 **중복**이라 1:1에 미포함.

## 6. 내 분석에 미치는 영향: 없음
- y_unet은 **users 1–10, 세션 1_0_2/2_0_1/2_0_2의 정상 경로**에서 추출 — 이 3세션은 DD와 **정확 1:1**.
- 손상 user30(test-users 1–10에 없음)·미완성 1_0_1(라벨 없어 미사용)·session_3(미사용)은 **내가 쓴 데이터 아님**.

## 7. 처리 방침 (사용자 결정 2026-07-01)
- user30 중복본·1_0_1 미완성은 **분석에서 제외만**, **원본은 미삭제**. (Claude는 `/mnt/e`에 읽기만 수행, 삭제·수정 없음.)
