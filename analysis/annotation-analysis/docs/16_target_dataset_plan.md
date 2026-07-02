# 16 · Dataset_full_gsam2_subject_independent — 구축 계획

2026-07-02. 대상: `/mnt/e/DATASET/eveye/target_data/Dataset_full_gsam2_subject_independent`
(이미 빈 디렉토리 생성됨). 목표: **GSAM2 라벨 + Eye-ROI crop(마커 제외) + APS·Event 양 모달리티**의
subject-independent 전체 학습 데이터셋을, 기존 `DeanDataset_full_unet_subject_independent`와 **동일 캐시
포맷**으로 구축(HBTXR/FACET 로더 호환).

## 1. 개요 / 기존과의 차이
| 축 | 기존 (full_unet) | **신규 (full_gsam2)** |
|---|---|---|
| 라벨 소스 | U-Net dense 의사라벨 | **개선 GSAM2**(ROI+geom, 사람GT 0.77px·IoU0.914·홍채0) |
| 좌표/해상도 | native 346×260 | **Eye-crop 240×160**(마커 제외, docs/12·crop_dataset.py) |
| 모달리티 | Event only | **Event + APS 둘 다** |
| split | train1-32/val33-36/test37-48 | 동일(leak-free) |
| 캐시 포맷 | memmap 배치 | **동일**(드롭인 호환) + `cached_aps/` 추가 |

## 2. 구성 (split·규모)
`manifest.json` 기준(기존과 동일 subject split):
- **train**: subjects 1–32 — **968,873 프레임**
- **val**: 33–36 — **122,776**
- **test**: 37–48 — **366,171**
- 합계 **1,457,820 프레임 / 384 세션**(48subj × 4session × 2eye). 원본: `Data_davis`(APS) + 원본 event 스트림 + GSAM2 라벨.

## 3. 디렉토리 구조
```
Dataset_full_gsam2_subject_independent/
├── README.md                # 데이터셋 상세 설명(구성·포맷·좌표·타임스탬프·로드·provenance)
├── manifest.json            # split·counts·provenance·label_source·crop·resolution·modalities
├── progress_state.json      # 재개용 진행상태(세션별 done/failed)
├── crop_boxes.json          # 세션별 crop box [x0,y0,x1,y1] (un-crop·eval 매핑용)
├── qc_summary.json          # 세션별 valid율·mislabel·blink·fallback 집계
├── train/  (val/ test/ 동일)
│   ├── cached_data/         # EVENT (crop 좌표): events_batch_{N}.memmap + _info_{N}.txt + _indices_{N}.npy
│   ├── cached_aps/          # APS (crop 좌표) [신규]: aps_batch_{N}.* (또는 PNG) + timestamps
│   ├── cached_ellipse/      # GSAM2 ELLIPSE 라벨(crop 좌표): ellipse_records.npy[t,x,y,a,b,ang]
│   │                        #   + ellipses_batch_{N}.memmap/_info/_indices + ellipse_qc.npy[신규]
│   ├── cached_mask/         # GSAM2 MASK 라벨(crop 좌표) [신규]: mask_records.npy(RLE) 또는
│   │                        #   masks_batch_{N}.memmap(packed-bit) + _indices; (옵션 PNG)
│   └── labels_original/     # 원본 EV-Eye 라벨·타임스탬프 보존 [신규]
│                            #   human_ellipse.npy  (사람GT: ts, native x,y,a,b,ang, crop x,y, subj/eye/session/idx)
│                            #   unet_ellipse.npy   (원본 U-Net dense center: ts, native x,y) [옵션]
│                            #   frame_index.npy    (프레임별: idx, ts_orig, subj/eye/session, crop_box_id, src_path)
```

## 4. 데이터 저장 포맷
### 4.1 Event (`cached_data/`, 기존과 동일 dtype)
- 구조체 memmap: **dtype `[('t','<i8'),('x','<i8'),('y','<i8'),('p','<i8')]`**, 배치당 ~25M event(기존 배치0=24.97M).
- **crop 적용**: 원본 event를 세션 crop box로 공간 필터(x∈[x0,x1),y∈[y0,y1)) + **좌표 shift(x-x0,y-y0)** → 240×160.
- `events_indices_{N}.npy`: 프레임→event 구간 매핑(기존과 동일). `_info_{N}.txt`: shape/dtype.

### 4.2 APS (`cached_aps/`, 신규 모달리티)
- crop 프레임 **240×160 uint8**. 두 옵션:
  - **(A·권장) 압축 PNG**: `cached_aps/{session_id}/{idx6}_{ts}.png` — 디스크 ~5–7GB(전체), APS는 dense·압축 잘됨.
  - **(B·속도) memmap 스택**: `aps_batch_{N}.memmap` (uint8, [n,160,240]) + `aps_ts_{N}.npy` — I/O 빠름·~56GB(전체).
- 권장: 저장은 **native crop 240×160**(해상도 상한). 학습 입력은 로더에서 128×96/144×96 리사이즈(docs 해상도 분석).

### 4.3 해상도
- **native crop 240×160**로 저장(DAVIS346 subset = 실해상도 상한). APS·Event **동일 crop box/grid**로 정렬.
- crop box: **세션별**(A2, GSAM2/U-Net center로 그 세션 눈 중심에 240×160 배치) 권장 — subject/헤드셋 편차 강인. `crop_boxes.json`에 저장(un-crop·eval용). 전역 고정도 가능(현 audit box [53,28,293,188]).

## 5. 라벨 저장 포맷 — **Mask + Ellipse 둘 다 + 원본 보존**
GSAM2 라벨을 **두 형태로 모두 저장**: (A) segmentation용 **Mask**, (B) center/regression용 **Ellipse**.
둘 다 **crop 좌표(240×160)**, 동일 프레임 인덱스로 정렬(같은 `t`).

### 5.1 GSAM2 Ellipse (`cached_ellipse/`, 기존 dtype 드롭인)
- **dtype `[('t','<i8'),('x','<f8'),('y','<f8'),('a','<f8'),('b','<f8'),('ang','<f8')]`** (기존과 동일).
  `x,y`=동공중심(crop), `a,b`=타원 **축 지름**(GSAM2 mask ellipse-fit), `ang`=degree. `ellipse_records.npy` + 배치.

### 5.2 GSAM2 Mask (`cached_mask/`, 신규 — 원시 분할)
- GSAM2 **동공 이진 mask**(crop 240×160), ellipse-fit 이전의 원시 분할. 프레임 인덱스는 ellipse와 동일 정렬.
- 저장 옵션(택1):
  - **(권장) RLE records**: `mask_records.npy` dtype `[('t','<i8'),('h','<i2'),('w','<i2'),('rle','O')]`
    (COCO식 run-length) — 동공 mask는 소면적이라 **~수백 B/frame**(전체 ~1GB).
  - **(속도) packed-bit memmap**: `masks_batch_{N}.memmap`(1bit/px, 240×160/8=4800 B/frame, ~7GB) + `_indices`.
  - **(검수) PNG**: `cached_mask/{session}/{idx6}_{ts}.png`(0/255) — inspectable, ~2GB.
- 용도: mask=segmentation supervision(EllSeg류)·IoU 평가, ellipse=center regression(HBTXR). 둘 다 있으면 다목적.

### 5.3 QC 플래그 (`ellipse_qc.npy`, mask/ellipse 공용)
- dtype `[('t','<i8'),('valid','?'),('mislabel','?'),('blink','?'),('det','<f4'),('area','<f4'),('source','u1')]`
  (source: 0=gsam2, 1=unet-fallback). → 학습 시 `valid & ~mislabel` 필터, blink 제외(G6). mask/ellipse 동일 적용.

### 5.4 원본 EV-Eye 라벨·타임스탬프 보존 (`labels_original/`, 신규)
- **모든 레코드가 원본 DAVIS µs 타임스탬프 `t`를 그대로 보존**(event/aps/ellipse/mask 공통 키). 원본 raw로 역추적 가능.
- **`human_ellipse.npy`**: 원본 사람 GT(Data_davis VIA) — dtype
  `[('t','<i8'),('nx','<f8'),('ny','<f8'),('a','<f8'),('b','<f8'),('ang','<f8'),('cx','<f8'),('cy','<f8'),('subject','u1'),('eye','u1'),('session','u1'),('idx','<i8')]`
  (nx,ny=native 346×260, cx,cy=crop 좌표). 존재하는 keyframe(~9011)만. → **정직 평가·provenance**.
- **`unet_ellipse.npy`**(옵션): 원본 U-Net dense center(native) + ts — 기존 라벨과의 대조용.
- **`frame_index.npy`**: 프레임별 `[('idx','<i8'),('t','<i8'),('subject','u1'),('eye','u1'),('session','u1'),('crop_box_id','<i2'),('src_path','O')]`
  — 원본 파일 경로·타임스탬프·crop box 매핑(un-crop·재현·감사).
- **정직성**: train/val supervision=GSAM2(mask+ellipse), **test 정확도 평가=`human_ellipse`(native)** ↔ un-crop pred 비교(precision 분석과 일관).

## 6. 라벨링 방법 (GSAM2 at scale)
- 개선 harness(`08_run_gsam2.py --roi <session_box> --geom-select`, repeats/TTA **OFF**=속도) → mask → ellipse-fit(x,y,a,b,ang).
- **QC 게이트**: mislabel(det<thr·area범위밖·radius_ratio>1.5 홍채·U-Net 교차검증>15px), blink(면적 붕괴), 무검출.
  실패 프레임 → **U-Net fallback**(학습 라벨엔 독립성 무관, 커버리지 우선; `source=1` 표시) 또는 `valid=False`.
- **컴퓨트**(핵심): train ~969k 프레임. 단일패스 ~0.4s/frame ≈ 4–5일(단일 GPU). 가속: **SAM2 video-propagation**
  (세션당 GDINO seed→전파, 5–10배)·batching·multi-GPU. FACET 학습과 GPU 공유 → 전용창/멀티GPU 필요.

## 7. 빌드 파이프라인 (단계·재개 가능)
```
S1 crop box: 세션별 U-Net/GSAM2 center envelope → 240×160 box → crop_boxes.json
S2 label   : 세션 프레임에 GSAM2(crop ROI) → ellipse(crop좌표) + QC 플래그
S3 crop    : event 공간필터+shift, APS crop (240×160)
S4 cache   : event/aps/ellipse를 배치 memmap + indices + info + records + manifest/progress/qc
             (기존 MemmapCacheStructedEvents.py 포맷 준수)
```
- `progress_state.json`으로 세션 단위 재개. 각 단계 [done/skip/UNVERIFIED] 로그.

## 8. 빌드 순서 (phased — 큰 커밋 전 검증)
1. **Phase 0**: 1 세션 PoC → 캐시 생성 → **FACET 로더로 로드 확인**(dtype/indices/APS 확장) + QC 육안.
2. **Phase 1**: **val(33-36, 122k)** 구축 → HBTXR 소규모 학습으로 파이프라인 검증.
3. **Phase 2**: **test(37-48, 366k)** + `ellipse_human` → 정직 평가 기준선.
4. **Phase 3**: **train(1-32, 969k)** 전면(가속 적용) → 본 학습.

## 9. 호환성 / 검증
- `cached_data`(event)·`cached_ellipse`는 기존 dtype/구조 동일 → **FACET 로더 드롭인**.
- `cached_aps`는 **신규** → FACET dataset에 APS 로드 분기 추가 필요(또는 event-only 학습 시 무시).
- 검증: (a) 배치 1개 memmap 로드 shape/dtype 일치, (b) indices로 프레임↔event 정렬 정확, (c) crop 좌표 라벨이
  APS/Event와 정렬(overlay), (d) manifest counts == 실제 레코드 수.

## 10. 한계 / 주의
- **컴퓨트**: GSAM2 969k 프레임 라벨링이 최대 비용(가속 필수). crop 자체는 즉시(per-frame GSAM2 불필요).
- **좌표 프레임**: 세션별 crop box → 라벨/event/APS 모두 crop 좌표. eval은 box로 un-crop 후 native 사람GT와 비교.
- **APS 크기**: memmap ~56GB vs PNG ~7GB → 디스크·속도 trade-off(권장 PNG).
- **모델 입력 해상도**: 저장 240×160, 학습 128×96/144×96(동공 ~16–20px).
- **정직성**: train/val 라벨=GSAM2(supervision), test 정확도 평가=사람GT(native)로 — precision 분석(§docs/10,precision_report)과 일관.
