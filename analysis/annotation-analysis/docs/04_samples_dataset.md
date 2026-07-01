# 04 · samples/ 데이터 문서

`../samples/`에 수집된 audit 표본의 구조·스키마·통계·좌표계·provenance. 생성 스크립트: `07_collect_samples.py`(+`07b_reslice_events.py`).

## 1. 개요
- **단위**: GT-앵커 연속 윈도우. 각 윈도우 = 사람 GT 키프레임 1장(중앙 anchor) + 양옆 연속 프레임(K-1장, 예측/audit만).
- **규모**: **483 윈도우**(fixation 161 / saccade 161 / smooth_pursuit 161), 프레임 총 **≈4,669**(fixation·smooth K=11, saccade K=7 → 161×29).
- **피험자**: users 1–10 (subject-independent test용), 양안, 사람 GT가 있는 세션(1_0_2, 2_0_1, 2_0_2).
- **좌표계**: 모든 center는 원본 APS **346×260 px**(DAVIS346).

## 2. 디렉터리 구조
```
samples/
├─ COLLECTION_PLAN.md / COLLECTION_PLAN_v2.md   # 설계 문서
├─ manifest_frames.csv                          # 프레임 1행
├─ manifest_windows.csv                         # 윈도우 1행
├─ frame/{key}/{idx6}_{ts}.png                  # 원본 APS 프레임(anchor+이웃)
├─ event/{key}.npz                              # 윈도우 시간범위의 전 이벤트
├─ label/{key}/
│  ├─ gt.json            # anchor 사람 GT 타원 (진실)
│  ├─ unet_dense.json    # 전 프레임 U-Net center (audit, 07에서 생성)
│  ├─ gsam2.json         # 전/anchor 프레임 Grounded-SAM2 center (08에서 생성)
│  ├─ pred.json          # HBTXR 예측 center (09에서 생성)
│  ├─ provenance.json    # 원본 경로 + 설정
│  └─ unet_masks/        # (옵션 --copy-masks) U-Net 마스크 gif 복사본
└─ meta/
   ├─ strata_summary.csv       # 모션별 candidates/collected
   ├─ rejections.csv           # 제외 프레임 + 사유
   └─ selection_config.json    # seed·임계·split·파라미터
```

## 3. key 명명 규칙
`{motion}_{user}_{eye}_{session}_w{win:03d}_a{anchor_idx:06d}`
예: `fixation_user1_left_session_1_0_2_w000_a000010`

## 4. 라벨 구성 매트릭스 (프레임별 어떤 값이 있나)
| 프레임 | y_orig(사람 GT) | y_unet(예측) | y_gsam2(audit) | y_pred(모델) |
|---|---|---|---|---|
| **anchor**(중앙) | ✅ (gt.json) | ✅ | ✅(08) | ✅(09) |
| **이웃** | ❌ | ✅ | ✅(08, anchors-only면 ❌) | ✅(09, anchors-only면 ❌) |
> 진실은 anchor에만. 이웃은 dense 예측/audit. `--anchors-only`로 08/09를 돌리면 gsam2/pred는 anchor만 채워짐.

## 5. 파일 스키마

### 5.1 manifest_windows.csv
`key, motion, user, eye, session, K, anchor_index, anchor_ts_us, ts_lo, ts_hi, n_frames, n_events, q_good, q_frontal, q_illum, event_file, label_dir`
- `ts_lo/ts_hi`: 윈도우 첫/끝 프레임의 µs UNIX 타임스탬프(이벤트 슬라이스 범위)
- `n_events`: 윈도우 내 이벤트 수(07b로 채워짐; 0이면 재슬라이스 필요)
- `q_*`: anchor 프레임의 VIA 품질 플래그

### 5.2 manifest_frames.csv
`key, motion, role(anchor/neighbor), user, eye, session, frame_index, frame_ts_us, has_gt, gt_cx, gt_cy, gt_rx, gt_ry, gt_theta, unet_valid, unet_cx, unet_cy, unet_area, src_frame, dst_frame`
- `has_gt`=True는 role=anchor에서만. `gt_*`는 VIA 타원(346×260). `unet_*`는 U-Net center/면적.

### 5.3 label/{key}/gt.json (사람 GT, anchor)
```json
{"anchor_idx": 10, "anchor_ts": 1657711084817717,
 "ellipse_cx_cy_rx_ry_theta": [168, 138, 21.15, 19, 2.636],
 "quality": {"good": true, "frontal": true, "good_illumination": true},
 "source_csv": "E:\\...\\Data_davis\\user1\\left\\session_1_0_2\\user_1.csv"}
```

### 5.4 label/{key}/unet_dense.json (U-Net, 전 프레임)
```json
{"unet_centers": [{"idx": 5, "ts": 1657...,"valid": true, "cx": 167.2, "cy": 139.1, "area": 612, "src_gif": ".../predict/000005_..._mask.gif"}, ...]}
```

### 5.5 label/{key}/gsam2.json (08 생성)
```json
{"prompt": "pupil.", "gsam2_centers": [{"idx": 10, "ts": 1657...,"valid": true,
  "cx": 168.4, "cy": 137.8, "area": 604, "det_score": 0.41, "sam_score": 0.97,
  "box": [147,117,190,159], "repeats": [[168.1,137.9],[168.7,137.6], ...]}]}
```
- `repeats`: 박스 지터/TTA로 얻은 반복 center(정밀도 proxy용 산포).

### 5.6 label/{key}/pred.json (09 생성)
```json
{"mode": "event", "ckpt": "HBTXR_si.ckpt", "grid": 64,
 "pred_centers": [{"idx": 10, "ts": 1657...,"valid": true,
   "cx": 169.0, "cy": 138.5, "grid_x": 31.2, "grid_y": 34.1, "score": 0.88}]}
```
- `cx/cy`: 346×260 px(비교용). `grid_x/grid_y`: 모델 출력 그리드(원시).

### 5.7 event/{key}.npz
- 키: `t`(int64, µs UNIX), `x`(int16), `y`(int16), `p`(int8, 0/1), `ts_lo`, `ts_hi`.
- `t` 오름차순. 윈도우 `[ts_lo, ts_hi]` 범위의 전 이벤트.

### 5.8 label/{key}/provenance.json
원본 경로 기록: `frames_dir, events_txt, via_csv, predict_dir, mask_h5` + `ts_lo/ts_hi` + `collected_at`.

## 6. 모션 정의 (분류 기준)
| 모션 | 정의 | 세션 | K |
|---|---|---|---|
| fixation | U-Net center **median 속도 < 1.2 px/frame** | 1_0_2 | 11 |
| saccade | U-Net center **max 속도 ≥ 6 px/frame** | 1_0_2 | 7 |
| smooth_pursuit | 설계상 추종 | 2_0_1 / 2_0_2 | 11 |
| (blink) | 마스크 면적 붕괴 프레임 비율 > 0.34 → **제외** | — | — |

## 7. 통계 (strata_summary.csv)
| 모션 | candidates | collected |
|---|---:|---:|
| fixation | 306 | 161 |
| saccade | 161 | 161 |
| smooth_pursuit | 1474 | 161 |
- 균형 기준 = saccade 가용량(161). n_events 예시(07b 후): 7,945 / 8,910 / 11,712 …

## 8. 좌표계 & 정합
- 전부 **346×260 px**. VIA 타원중심=y_orig. U-Net/GSAM2=마스크 ellipse-fit center. pred=grid×(346/G, 260/G), G=img_size/patch_size.
- 시계: 프레임·이벤트 모두 µs UNIX(동일 DAVIS 시계). Tobii(별도, 미사용).

## 9. 로드 예시 (파이썬)
```python
import json, csv, numpy as np, os
OUT = "../samples"
# 윈도우 목록
wins = list(csv.DictReader(open(f"{OUT}/manifest_windows.csv")))
key = wins[0]["key"]
gt   = json.load(open(f"{OUT}/label/{key}/gt.json"))
unet = json.load(open(f"{OUT}/label/{key}/unet_dense.json"))["unet_centers"]
ev   = np.load(f"{OUT}/event/{key}.npz")          # ev["t"], ev["x"], ev["y"], ev["p"]
# anchor 사람 GT center
ocx, ocy = gt["ellipse_cx_cy_rx_ry_theta"][:2]
```

## 10. 재생성 방법
```bash
# 전체 재수집(깨끗한 out에서)
python 07_collect_samples.py --root /mnt/e/DATASET/eveye --out ../samples \
    --test-users 1,2,3,4,5,6,7,8,9,10 --copy-masks
# 이벤트만 재슬라이스(수집 후)
python 07b_reslice_events.py --root /mnt/e/DATASET/eveye --out ../samples
```
