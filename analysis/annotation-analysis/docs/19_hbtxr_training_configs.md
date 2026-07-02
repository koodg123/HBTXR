# 19 · HBTXR 훈련 config 가이드 (patch_size / grid / frame·event)

2026-07-03. `Dataset_sample_gsam2_subject_independent` 기반 HBTXR 훈련 config 설계 규칙과 준비된 config 목록. 데이터셋 [18](18_sample_dataset_build.md), 로더 분기(use_cached_aps/events)는 FACET `DavisEyeEllipseFrameDataset`.

## 1. 핵심 규칙 — `patch_size = down_ratio = 4`
- **모델 출력 grid** = `img_size // patch_size` (`HBTXR.py: self.output_size = img_size // patch_size`, `img_size % patch_size == 0` 필수).
- **라벨 grid** = `img_size // down_ratio`, **`down_ratio = 4` 하드코딩**(로더 `__getitem__`).
- 둘이 같아야 loss 계산 성립 → **`patch_size = down_ratio = 4`** (img_size 무관).
  - 다른 patch를 쓰려면 로더 `down_ratio`도 같이 바꿔야 함(항상 `patch_size == down_ratio`).

## 2. 조합 표 (patch = down_ratio 유지)
| img | patch | down_ratio | tokens (img/patch)² | 출력 grid G | 비고 |
|---|---|---|---|---|---|
| 64 | **4** | 4 | 16²=256 | 16 | frame(현재)·event |
| **128** | **4** | 4 | 32²=**1024** | **32** | ⭐ 128 기본(drop-in), sub-pixel↑, tokens 4×·attn ~16× |
| 128 | 8 | 8(로더수정) | 256 | 16 | 경량 대안(down_ratio도 8로) — drop-in 아님 |
| 128 | 16 | 16 | 64 | 8 | 동공엔 너무 coarse |

## 3. Frame vs Event
- **patch_size는 동일**(grid 규칙) → frame·event 모두 64→patch4(G16), 128→patch4(G32).
- **다른 점은 `input_channels`(입력 표현)**:
  - **Frame(APS)**: grayscale **1채널**, 로더 `use_cached_aps`(cached_aps PNG).
  - **Event**: **2채널**(극성 pos/neg), 로더 `use_cached_events`(cached_data events→2ch 프레임). 참조 event 표현은 `DavisEyeEllipseDataset`(`to_frame_stack_numpy` causal_linear, sensor 346×260). 우리 데이터셋은 **crop 240×160 좌표**라 sensor=(240,160,2).
- 이벤트 희소성 튜닝 시 patch8+down_ratio8 실험 가능(토큰당 이벤트↑).

## 4. 준비된 config (FACET `configs/`)
| config | 입력 | img/patch | input_ch | 로더 | 상태 |
|---|---|---|---|---|---|
| `..._frame_gsam2crop_si_img64_patch4.yaml` | APS | 64/4→G16 | 1 | use_cached_aps | ✅ 훈련중(bjirk04dr) |
| `..._frame_gsam2crop_si_img128_patch4.yaml` | APS | 128/4→G32 | 1 | use_cached_aps | ✅ 준비 |
| `..._event_gsam2crop_si_img64_patch4.yaml` | Event | 64/4→G16 | 2 | use_cached_events | ✅ 준비 |
| `..._event_gsam2crop_si_img128_patch4.yaml` | Event | 128/4→G32 | 2 | use_cached_events | ✅ 준비 |

## 5. 주의
- **img_size 변경 → DeiT `pos_embed` 크기 변경** → 다른 img_size ckpt 로드 불가, **from scratch**(또는 pos_embed 보간).
- 128은 tokens 4배 → GPU/시간 부담↑ (attention ~16×). 부족하면 patch8+down_ratio8.
- 실행: `PYTHONPATH=FACET FACET_DEVICES=0 .venv/bin/python tools/train.py -c <config>.yaml`.

## 6. 훈련 결과 (Dataset_sample, subject-independent val 33–36)
| config | best val_mean_distance | train_loss | p1_acc | best ckpt |
|---|---|---|---|---|
| **frame img64** (G16) | **0.156** @ep84 | 5.24→**1.71**(수렴) | 0.99 | `epoch=84-...0.1558` |
| **frame img128** (G32) | 0.678 @ep74 | 9.72→**3.80**(미수렴) | 0.80 | `epoch=74-...0.6777` |
| event img64/128 | (미실행) | 초기 loss 16.1/17.7 | — | — |

⚠️ **img64 vs img128 직접 비교 불가**: `cal_mean_distance`는 **출력 grid 단위**(img64=16, img128=32) → 같은 물리오차도 img128 숫자 2배. **프레임 비율 정규화**: img64 **0.98%**(0.156/16) vs img128 **2.1%**(0.678/32). p1_acc도 img128의 "1 grid"가 절반크기(더 엄격). 게다가 **img128 미수렴**(4× tokens→수렴 2–4배 느림, loss 아직 하락) → 공정비교엔 200–300ep+LR/warmup 필요.
- **실용**: img64가 빠른 수렴·고정확도·4× 저렴 → 표본 규모 baseline 적합. 미학습 subject(33–36)서 프레임비 0.98% = 우수.

## 7. Event n-events 누적 설계 (현재 per-frame → dense화)
**실측(train)**: per-frame events median **132**(fix194·sac6018·smooth49·blink458), per-window median **6279**(82% ≥1000·55% ≥5000), frame dt **40ms(~25Hz)**. 참조 event 모델 = 연속스트림 **fixed-count 5000/샘플**.
**제약**: 우리 데이터는 윈도우 저장 → 로더 누적은 **윈도우 총량 상한**(saccade는 충분, smooth/fix 윈도우는 총량 부족).

| 옵션 | 방식 | 밀도 | 시간국소성 | 인과 |
|---|---|---|---|---|
| A | per-frame(현재) | 낮음(132) | 최고 | ✅ |
| **B** | fixed-count N backward(윈도우내 cap) | 높음(≤N) | 가변 | ✅ |
| **C** | fixed-time ΔT backward | 가변 | 고정 | ✅ |
| D | whole-window→i | 높음(후반) | 낮음 | ✅ |
| E | centered ±N/2 | 높음 | 중간 | ❌ 미래사용 |

**권장**:
1. **1차(로더)**: `event_accum="count"` **N≈2500**(또는 `time` ΔT≈80ms) **backward-within-window**(동일 key 내). 신호 프레임 5–40× dense↑, 시간국소성 유지.
2. **2차(재수집)**: smooth/fix까지 dense 필요 시 build에서 per-frame 컨텍스트를 **연속스트림 last-N** 저장(`07 --event-pad-us` 확대 후 재빌드 → 윈도우 상한 제거).
3. **병행**: **hybrid(APS+event)** 검토 — event-only는 fixation 본질적 약점(정적=이벤트 적음, E_orig event gross-fail 25% 원인).

구현 스케치: `__getitem__` event 분기에서 `event_accum`에 따라 `_ev_backward(index, n=N|dt=ΔT)` — `frame_index` key 동일 내 `ev_idx` 역방향 concat.

## 커밋 이력 (branch annotation)
`f49f999b7` dataset · `c1691c32e` frame pipeline · `ba0267097` event loader+configs · `40ab5737f` docs/E_orig · (이번) 훈련결과+n-events 설계.
