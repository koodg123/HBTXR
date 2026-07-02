# Eye-ROI Crop Dataset (marker-excluded) — 240×160

캘리브레이션 **마커 기둥을 제거**한 Eye-ROI crop 데이터셋 (**APS 프레임 + Event 스트림**).
개선 GSAM2/U-Net 중심 envelope으로 crop box를 산정, 하단 마커 밴드를 잘라 눈만 남김.

## 구성
```
eye_crop_240x160/
├── meta.json                     # box, 해상도, 통계
├── README.md
├── _qc/*.png                     # 육안검증 오버레이(2x, GT 중심 초록)
└── {key}/                        # window별 (483개)
    ├── aps/{idx6}_{ts}.png       # 크롭 APS 프레임 240×160 (grayscale)
    ├── events.npz                # 크롭 이벤트: t,x,y,p (+ts_lo/hi, box), 좌표 shift됨
    └── labels.json               # 프레임별 중심(crop 좌표): gt(anchor)/unet/gsam2/pred
```

## crop box (데이터 기반)
- **BOX = [x0,y0,x1,y1] = [53, 28, 293, 188]** (원본 346×260 기준) → **crop 240×160**.
- 산정: 동공 center envelope x[85,265] y[61,176] + 반경 p95(20.4) + 여유, **하단 marker_cap=188**(마커 cy≥189 배제).
- 검증: **centers_clipped=0**(전 동공 중심 보존) · **frames_with_marker_band_content=4669**(전 프레임이 마커밴드 콘텐츠 보유→crop으로 제거) · **event_kept_frac=0.60**(주변부 eyebrow/side/marker 제거, 눈 이벤트 유지).

## 해상도 권고
- **저장 = native crop 240×160** (DAVIS346 subset = 최대 실해상도; 그 이상 upscale은 정보 없음).
- **모델 입력 권고 = 128×96 또는 144×96** (동공 지름 native ~30–40px → ~16–20px, 현행 64×64-from-full의 2–3배; 종횡비 왜곡 최소). 64×64는 zoom 이득 낭비, 256+는 event sparsity상 비권장.
- **Event는 ≤128×96** (sparse), **APS는 native 240×160까지** 가능. 두 모달리티 동일 box/grid로 정렬됨.

## 라벨 (crop 좌표)
`labels.json.frames[idx]`: `gt`=[cx,cy,rx,ry,theta](anchor만), `unet`/`gsam2`/`pred`=[cx,cy]. 모두 **crop 좌표**(원본−[x0,y0]). 반경·θ 불변(resize 없음). event 좌표도 동일 shift.

## 생성 / 스케일업
```bash
# 이 PoC (audit 483 window, samples/ 기반)
.venv-gsam2/bin/python src/crop_dataset.py --out datasets/eye_crop_240x160
# 옵션: --box x0,y0,x1,y1 (수동) / --marker-cap 188 / --margin 8
```
**전체 확장(train subject 1-32, /mnt/e)**: I/O 소스만 교체(`Data_davis` APS + 원본 event 스트림 + 라벨) → 동일 crop 로직. box는 전역 재검증(전 48 subject U-Net center envelope) 또는 per-session GSAM2 센터링 권장. crop 자체는 per-frame GSAM2 불필요(즉시).

## 한계
- **Event 모달리티엔 마커가 이미 거의 없음**(정적→이벤트 ~0.5%); crop의 실이득은 event에선 **눈 zoom/re-centering**, APS에선 **마커 제거**.
- 극단 시선 대형 동공의 disc 하단이 marker_cap(188)에서 미세 clip 가능(중심은 보존). 필요시 `--marker-cap` 상향.
- PoC는 users 1-10(audit). 전체는 스케일업 필요.
