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
