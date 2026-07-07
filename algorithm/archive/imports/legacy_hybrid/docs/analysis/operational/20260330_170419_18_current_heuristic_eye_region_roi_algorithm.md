# 현재 Heuristic Eye Region ROI 알고리즘

최종 갱신: 2026-03-26 KST

이 문서는 현재 `HBTXR_v3_0` 코드 안에 이미 존재하는 heuristic `Eye Region ROI` 계산 로직을 설명한다.

중요한 점은, 현재 구현은 `raw frame`만 보고 `eye ROI`를 찾는 detector가 아니다.  
기존 ellipse annotation을 바탕으로 세션 단위의 안정적인 ROI를 만드는 규칙 기반 알고리즘이다.

## 위치

- 핵심 구현
  - [io_utils.py](../../src/hbtxr/preprocess/io_utils.py)
  - `derive_eye_region()`
- 주요 호출부
  - [canonicalize.py](../../src/hbtxr/preprocess/canonicalize.py)
  - [raw_overlay_preview_manual_csv.py](../../scripts/raw_overlay_preview_manual_csv.py)

## 입력

입력은 raw frame 자체가 아니라 `EllipseAnnotation` 목록이다.

각 annotation은 대략 다음 정보를 가진다.

- `frame_filename`
- `frame_idx`
- `timestamp_us`
- `ellipse_xywht = [cx, cy, w, h, theta]`

즉 현재 heuristic ROI는 pupil ellipse supervision이 있을 때 계산된다.

## 출력

출력은 `EyeRegion(x, y, w, h)`이다.

특징:

- sensor 좌표계 기준
- 세션 단위 공통 ROI
- 정수 bbox

## 알고리즘 요약

`derive_eye_region()`은 아래 순서로 동작한다.

### 1. Empty-annotation fallback

annotation이 비어 있으면 전체 sensor bbox를 반환한다.

```text
EyeRegion(0, 0, sensor_w, sensor_h)
```

현재 기본 sensor는 코드상 `346 x 240`이다.

### 2. Per-ellipse padded box 생성

각 ellipse마다 padding이 포함된 bbox를 만든다.

입력 ellipse:

```text
[cx, cy, w, h, theta]
```

기본 padding box:

```text
x0 = cx - w/2 - margin_px
y0 = cy - h/2 - margin_px
x1 = cx + w/2 + margin_px
y1 = cy + h/2 + margin_px
```

현재 기본값:

- `margin_px = 24`

즉 ellipse를 그대로 쓰지 않고 사방으로 24px 확장한다.

현재 코드는 비대칭 margin override도 지원한다.

- `margin_left_px`
- `margin_right_px`
- `margin_top_px`
- `margin_bottom_px`

즉 다음처럼 per-side margin을 줄 수 있다.

```text
x0 = cx - w/2 - margin_left_px
y0 = cy - h/2 - margin_top_px
x1 = cx + w/2 + margin_right_px
y1 = cy + h/2 + margin_bottom_px
```

이 확장은 specular highlight나 반사광 때문에 ROI가 과도하게 넓어지는 현상을 조절하기 위한 장치다.

### 3. Session envelope 계산

모든 ellipse의 padded box를 합쳐 outer envelope를 만든다.

```text
x0 = min(all x0)
y0 = min(all y0)
x1 = max(all x1)
y1 = max(all y1)
```

이후 sensor boundary 기준 1차 clamp를 적용한다.

### 4. Target aspect 강제

그다음 ROI 종횡비를 `target_aspect_wh`에 맞춘다.

현재 기본값:

- `target_aspect_wh = 256 / 160 = 1.6`

동작 방식:

- 현재 bbox가 너무 좁으면 가로를 늘린다.
- 너무 넓으면 세로를 늘린다.

즉 canonical crop과 downstream input 비율을 강제로 맞추는 단계다.

### 5. Boundary re-clamp

aspect를 맞추는 과정에서 sensor 밖으로 나간 부분을 다시 안으로 밀어 넣는다.

- `x0 < 0`이면 오른쪽으로 이동
- `y0 < 0`이면 아래로 이동
- `x1 > sensor_w`이면 왼쪽으로 이동
- `y1 > sensor_h`이면 위로 이동

마지막으로 다시 `[0, sensor_w]`, `[0, sensor_h]` 범위로 clamp한다.

### 6. Integer ROI 변환

최종 bbox를 반올림해서 정수 `EyeRegion`으로 만든다.

```text
EyeRegion(
  x = round(x0),
  y = round(y0),
  w = round(x1 - x0),
  h = round(y1 - y0),
)
```

## 핵심 성격

현재 heuristic eye ROI는 다음 성격을 가진다.

- detector가 아니다.
- pupil ellipse annotation 기반의 session-level ROI envelope다.
- 안정적인 crop window를 만들기 위한 preprocessing 규칙이다.

즉 현재 eye ROI는 “frame마다 새로 탐지하는 eye box”가 아니라 “세션 전체를 안전하게 덮는 공통 crop 영역”에 가깝다.

## 장점

- 구현이 단순하고 안정적이다.
- session 단위 crop 일관성이 높다.
- downstream canonical crop 크기와 aspect를 쉽게 맞출 수 있다.

## 한계

- raw frame만으로는 계산되지 않는다.
- pupil ellipse supervision이 있어야 한다.
- close-eye, annotation noise, reflection 패턴을 직접 이해하는 detector는 아니다.

## 실무적 해석

이 heuristic ROI는 다음 상황에서 유용하다.

- canonical dataset 생성
- raw overlay sanity check
- baseline eye crop 영역 확보

반면 다음 상황에는 한계가 있다.

- CSV가 없거나 pupil annotation이 없는 세션
- frame-by-frame eye detection이 필요한 경우
- Grounded-SAM 또는 detector 기반 ROI와 직접 경쟁하는 경우

즉 현재 heuristic ROI는 “학습용 정답 기반 crop 생성기”로 이해하는 것이 가장 정확하다.
