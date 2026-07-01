# 09 · HBTXR 예측 주입 결과 (y_pred)

실행: 2026-07-01, Claude 직접. 산출: `samples/label/*/pred.json` + perframe `pred/`. 스크립트 `scripts/09_inject_pred.py`.

## 1. 모델·설정
- **ckpt**: `weights/hbtxr/hbtxr-imgsz64-epoch66-pe0.5401.ckpt` (**subject-independent, img64**; D12 준수 — test users 1–10 제외 학습).
- **config**: `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml` — `type: HBTXR, img_size:64, patch_size:4, input_channels:2`. → **출력 grid G = 64/4 = 16**, 스케일 `x*346/16=21.625, y*260/16=16.25`.
- **env**: `.venv-gsam2`(torch 2.5.1) + albumentations·lightning·thop·natsort. FACET는 `references/codebase/software/FACET`(sys.path).

## 2. 핵심 수정 (정확성)
- **입력 크기**: `Predict.pre_process`가 **256×256 하드코딩**(img256/full_unet용)인데, img64 모델의 DeiT `PatchEmbed`는 입력이 정확히 `img_size×img_size`가 아니면 **에러를 raise**. → 09의 `load_facet`에 **`a.img_size`(64) 크기 resize pre_process**를 넣어 교체(학습 전처리와 일치).
- **interpolation**: config는 `causal_linear_ori`지만 `to_frame_stack_numpy`가 미지원(ValueError) → **유효 모드는 `causal_linear`**(09 기본값). 
- 스모크(6 anchor)로 pred가 GT 근처(median ~4px) 착지 확인 후 전 실행.

## 3. 실행 결과 (전 4,669 프레임)
- valid **4,186/4,669 (89.7%)**. 무효 483 = 대부분 **각 윈도우 첫 프레임**(선행 이벤트 <10, `too_few_events`). anchor는 전부 유효.
- **커맨드**:
```bash
cd scripts
.venv-gsam2/bin/python 09_inject_pred.py --out ../samples \
  --facet-root ../../../references/codebase/software/FACET \
  --config ../../../references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml \
  --ckpt ../weights/hbtxr/hbtxr-imgsz64-epoch66-pe0.5401.ckpt \
  --img-size 64 --patch-size 4 --n-events 5000 --mode event
```

## 4. Corrected primary accuracy — `E_orig = ‖y_pred − y_orig‖` (anchor, n=483)
| 구간 | median | mean | p90 | p95 | max |
|---|---|---|---|---|---|
| **ALL** | **5.70** | 11.43 | 24.59 | 56.46 | 101.67 |
| fixation | 5.78 | 11.53 | | | |
| saccade | 4.40 | 7.96 | | | |
| smooth_pursuit | 7.45 | 14.81 | | | |
(단위 px, 346×260)

## 5. 해석 (관측 사실 — 프레이밍 중립)
> **사용자 정정(2026-07-01): 0.1812를 "누수(leakage) 확정"으로 단정하지 않는다.** 0.1812는 **dense(U-Net) 라벨을 사용해 얻은 결과**라는 사실만 기술한다.
1. **서로 다른 기준 라벨셋**: 보고된 **0.1812px는 dense(U-Net) 라벨 기준**의 정확도이고, 여기서 잰 **5.70px(median)는 sparse 사람 GT 기준**이다. 같은 모델을 서로 다른 reference로 평가한 값이므로 두 수치의 직접 비교(≈30배 차)는 해석에 주의(누수라 단정하지 않음).
2. **사람 GT 기준 맥락**: 5.70px는 label-noise floor(GSAM2 vs 사람 GT, median **0.75px**)보다 ~7배 크다 → 사람 GT 기준으로 보면 모델 잔차가 라벨 불확실성 범위를 넘음(즉 사람 GT 기준에서는 label-noise-limited 아님). 이는 dense-label 기준 0.1812와는 별개의 측정.
3. **모드 주의**: 이 수치는 **event 모드·subject-independent img64 ckpt** 기준. 0.1812 헤드라인이 다른 config(예: full_unet img256, subject-dependent)였다면 그 ckpt로도 재계산해 함께 보고 필요(→ 미결).

## 6. 산출물
- `samples/label/{key}/pred.json` = `{mode, ckpt, grid, pred_centers:[{idx,ts,valid,cx,cy,grid_x,grid_y,score}]}`.
- perframe `pred/center.json` 채워짐(`has_pred=4,186`). 오버레이(`overlay/`)에 pred=magenta 추가.
- **다음(10_eval)**: `y_orig`/`y_unet`/`y_gsam2`/`y_pred` 4소스로 3표(precision/noise/uncertainty) + `E_i/U_i`/Spearman + CDF/scatter + subject-level cluster-bootstrap CI. 데이터는 이제 전부 준비됨.

## 7. 미결
- 0.1812 원 헤드라인의 정확한 config/mode 확인 → 그 ckpt로도 E_orig 재계산(비교 완결).
- GSAM2 오검출 ~3% 게이트, 추가 audit 소스(EllSeg/RITnet, `weights/`에 수집됨) 투입 여부.
