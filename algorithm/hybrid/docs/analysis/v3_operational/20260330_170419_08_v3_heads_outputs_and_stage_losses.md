# HBTXR_v3_0 Head, Output, Stage Loss 매핑

최종 갱신: 2026-03-28 KST

Role: `모델 아키텍처 분석가`, `학습 손실 설계 엔지니어`, `이벤트 비전 연구자`

## 요약

이 문서는 현재 `HBTXR_v3_0`의 active head 구조, output ABI, supervised loss 매핑, stage별 학습 범위를 정리한다.

핵심 구분:

- `head`는 [heads.py](../../src/hbtxr/models/heads.py)에 정의된 실제 prediction module이다.
- `state` output은 종종 head output에서 decode된 파생 tensor이다.
- `Stage1`은 `search + eye + mask (+ optional search aux)`를 학습한다.
- `Stage2`는 `search + event + track (+ optional event aux)`를 학습한다.

주 코드 기준:

- [heads.py](../../src/hbtxr/models/heads.py)
- [hybrid_tracker.py](../../src/hbtxr/models/hybrid_tracker.py)
- [losses.py](../../src/hbtxr/training/losses.py)

## 분석 기준

이 문서는 주로 다음 지점을 기준으로 작성됐다.

- head 정의
  - `src/hbtxr/models/heads.py:20`
  - `src/hbtxr/models/heads.py:29`
  - `src/hbtxr/models/heads.py:38`
  - `src/hbtxr/models/heads.py:47`
  - `src/hbtxr/models/heads.py:56`
  - `src/hbtxr/models/heads.py:65`
- model output wiring
  - `src/hbtxr/models/hybrid_tracker.py:264`
  - `src/hbtxr/models/hybrid_tracker.py:277`
  - `src/hbtxr/models/hybrid_tracker.py:314`
- stage loss
  - `src/hbtxr/training/losses.py:534`
  - `src/hbtxr/training/losses.py:553`

## 1. Active head 목록

| Head Module | Output Key | Output Shape | 의미 |
|---|---|---:|---|
| `EyeRegionHead` | `search/eye` | `[B, 5]` | eye ROI `(cx, cy, w, h, conf)` |
| `PupilSearchHead` | `search/pupil` | `[B, 7]` | frame-search pupil `(x, y, a, b, u, v, conf)` |
| `EventSearchHead` | `event/pupil` | `[B, 7]` | event-only pupil `(x, y, a, b, u, v, conf)` |
| `PupilTrackHead` | `track/pupil` | `[B, 8]` | tracking residual `(dx, dy, dloga, dlogb, du, dv, conf, quality)` |
| `SearchMaskHead` | `search/mask_logits` | `[B, 1, 256, 256]` | dense search mask logits |
| `AuxStateHead` | `search/aux`, `event/aux` | `[B, C]` | auxiliary classification logits, 기본 `C=5` |

## 2. 파생 output

아래 항목은 standalone head는 아니지만, 실제 loss와 metric이 직접 사용하므로 중요하다.

| 파생 Output | Shape | Derived From | 의미 |
|---|---|---:|---|
| `search/state` | `[B, 6]` | `search/pupil[..., :6]` | absolute state `(x, y, a, b, u, v)` |
| `event/state` | `[B, 6]` | `event/pupil[..., :6]` | absolute state `(x, y, a, b, u, v)` |
| `track/state` | `[B, 6]` | `decode_track_state(prev_state, track/pupil)` | residual tracking에서 decode한 현재 state |
| `search/pooled` | `[B, D]` | search encoder | pooled search feature |
| `event/pooled` | `[B, D]` | event encoder | pooled event feature |
| `track/fused` | `[B, 2D]` | event pooled + prev-state feature | fused tracking feature |

## 3. head와 연결된 target ABI

| Target Key | Shape | 사용처 |
|---|---|---:|
| `eye_target` | `[B, 5]` | `search/eye` |
| `pupil_search_target` | `[B, 7]` | `search/pupil`, `event/pupil` |
| `pupil_track_target` | `[B, 8]` | `track/pupil` |
| `cur_state` | `[B, 6]` | `search/state`, `event/state`, `track/state` geometry term |
| `mask_target` | `[B, 1, 256, 256]` | `search/mask_logits` |
| `aux_target` | `[B]` | `search/aux`, `event/aux` |
| `constraint_center` | `[B, 2]` | `search/state`, `track/state` constraint term |

## 4. Head-to-loss 매핑

| Output | Main Target | Supervised Loss Terms | 실제 형태 |
|---|---|---|---|
| `search/eye` | `eye_target` | `loss_eye` | bbox `0:4`에 `SmoothL1` + conf `4`에 `BCEWithLogits`, 하나의 loss로 집계 |
| `search/pupil` | `pupil_search_target`, `cur_state` | `loss_search_xy`, `loss_search_ab`, `loss_search_trig`, `loss_search_geo`, `loss_search_conf` | `SmoothL1`, `SmoothL1`, trig loss, `ellipse_gwd_loss`, `BCEWithLogits` |
| `event/pupil` | `pupil_search_target`, `cur_state` | `loss_event_xy`, `loss_event_ab`, `loss_event_trig`, `loss_event_geo`, `loss_event_conf` | search branch와 동일한 구조 |
| `track/pupil` | `pupil_track_target`, `cur_state` | `loss_track_xy`, `loss_track_ab`, `loss_track_trig`, `loss_track_geo`, `loss_track_conf`, `loss_track_quality` | residual `SmoothL1`, residual `SmoothL1`, residual `SmoothL1`, `ellipse_gwd_loss(track_state, cur_state)`, `BCEWithLogits`, `BCEWithLogits` |
| `search/mask_logits` | `mask_target` | `loss_mask` | BCE + Dice 집계 loss |
| `search/aux` | `aux_target` | `loss_aux` | `CrossEntropy` |
| `event/aux` | `aux_target` | `loss_aux` | `CrossEntropy` |

## 5. 추가 non-head loss

| Loss | 참조 Output | 의미 |
|---|---|---|
| `loss_constraint_center` | Stage1의 `search/state`, Stage2의 `track/state` | 제공된 prior center 주변으로 pupil center를 제한 |
| `loss_consistency` | `search/state`, `track/state` | Stage2에서 search와 track state 예측을 정렬 |

## 6. Stage1 학습 범위

`Stage1`은 frame 기반 search foundation 단계다. search 계열 output만 supervision한다.

| Stage | 학습 대상 Head / Output | Loss Terms |
|---|---|---|
| `Stage1` | `search/eye`, `search/pupil`, `search/mask_logits`, optional `search/aux` | `loss_eye`, `loss_mask`, `loss_constraint_center`, optional `loss_aux`, `loss_search_xy`, `loss_search_ab`, `loss_search_trig`, `loss_search_geo`, `loss_search_conf` |

중요 메모:

- `event/pupil`은 `Stage1`에서 supervision하지 않는다.
- `track/pupil`도 `Stage1`에서 supervision하지 않는다.
- `search/state`는 search geometry term을 통해 간접적으로 supervision된다.

## 7. Stage2 학습 범위

`Stage2`는 hybrid tracking 단계다. search branch를 유지하면서 event와 track supervision을 추가한다.

| Stage | 학습 대상 Head / Output | Loss Terms |
|---|---|---|
| `Stage2` | `search/pupil`, `event/pupil`, `track/pupil`, optional `event/aux` | `loss_constraint_center`, `loss_consistency`, optional `loss_aux`, `loss_search_xy`, `loss_search_ab`, `loss_search_trig`, `loss_search_geo`, `loss_search_conf`, `loss_event_xy`, `loss_event_ab`, `loss_event_trig`, `loss_event_geo`, `loss_event_conf`, `loss_track_xy`, `loss_track_ab`, `loss_track_trig`, `loss_track_geo`, `loss_track_conf`, `loss_track_quality` |

중요 메모:

- `search/eye`는 `Stage2`에서 직접 supervision하지 않는다.
- `search/mask_logits`도 `Stage2`에서 직접 supervision하지 않는다.
- `Stage2`에서 optional aux term은 `search/aux`가 아니라 `event/aux`를 사용한다.

## 8. Stage별 head 매트릭스

| Head | Stage1 | Stage2 | 메모 |
|---|---|---|---|
| `eye_head` | trained | not directly trained | `search/eye`를 통해서만 사용 |
| `search_head` | trained | trained | 두 stage 모두의 핵심 branch |
| `event_head` | not trained | trained | Stage2 전용 supervised branch |
| `track_head` | not trained | trained | Stage2 전용 residual tracking branch |
| `mask_head` | trained | not directly trained | Stage1 search foundation supervision |
| `aux_head` | optional via `search/aux` | optional via `event/aux` | 같은 module이지만 branch output은 다름 |

## 9. 실무적 해석

현재 stage 분리는 다음처럼 이해하는 것이 가장 정확하다.

- `Stage1`
  - frame 경로만으로 eye region, pupil state, search mask를 찾는 foundation을 학습
- `Stage2`
  - search를 유지한 채 event-only search와 residual tracking을 추가 학습

즉 `Stage2`는 tracking-only 단계가 아니다. 다음을 동시에 학습하는 hybrid stage다.

- search branch 유지
- event branch supervision 추가
- track branch supervision 추가
- search-track consistency 명시

## 10. 코드 참조

- head inventory
  - [heads.py](../../src/hbtxr/models/heads.py)
- search / event / track output wiring
  - [hybrid_tracker.py](../../src/hbtxr/models/hybrid_tracker.py)
- stage-specific supervised loss
  - [losses.py](../../src/hbtxr/training/losses.py)
