# 세션 03: 학습 대 추론 파이프라인 표

날짜: 2026-03-27

## 목적

FACET의 tensor pipeline을 학습 관점과 추론 관점으로 나눠 단계별 데이터 계약을 비교하기 쉽게 정리한다.

## 학습 파이프라인

| 단계 | 객체 | shape / 구조 | 의미 |
|---|---|---|---|
| Raw source | `events.txt` | `[N_events]`, fields=`(t,x,y,p)` | 전체 event stream |
| Raw source | `ellipses.txt` | `[N_frames]`, fields=`(t,x,y,a,b,ang)` | timestamp별 ellipse 1개 |
| Offline sample build | fixed-count slice | `[5000]` | ellipse timestamp 직전 event |
| Cached sample | event memmap segment | `[5000]` | index 기반 로딩 |
| Cached sample | ellipse record | scalar structured record | GT ellipse 1개 |
| Dataset transform | frame stack | `[1, 2, 260, 346]` | single time-bin event image |
| Dataset transform | squeezed | `[2, 260, 346]` | time 차원 제거 |
| Dataset transform | HWC | `[260, 346, 2]` | image-like layout |
| Dataset transform | resized | `[256, 256, 2]` | 학습 해상도 |
| Dataset output | `input` | `[2, 256, 256]` | 최종 sample 입력 |
| Dataset output | targets | mixed tensors | `hm`, `ab`, `trig`, `reg`, `mask` 등 |
| Default collate | batch input | `[B, 2, 256, 256]` | 모델 입력 batch |
| Model forward | backbone/FPN | feature pyramid | ellipse detection backbone |
| Model output | heads | `hm/ab/trig/reg/mask` on `[B,*,64,64]` | 학습 supervision 공간 |

## 추론 파이프라인

| 단계 | 객체 | shape / 구조 | 의미 |
|---|---|---|---|
| Raw source | `events.txt` 또는 cached segment | `[N_events]` 또는 `[5000]` | 전체 stream 또는 사전 생성 slice |
| Slice | fixed-count segment | `[5000]` | `predict_txt()` 기준 chunk |
| Preprocess | frame stack | `[1, 2, 260, 346]` | resize 전 event image |
| Preprocess | squeezed | `[2, 260, 346]` | time-bin 축 제거 |
| Preprocess | resized tensor | `[2, 256, 256]` | 추론 입력 |
| Batch add | model input | `[1, 2, 256, 256]` | single-sample batch |
| Model output | heads | `hm/ab/trig/reg/mask` | dense prediction map |
| Post-process | gathered ellipse | `[1, 5]` | 64x64 출력 grid에서 top ellipse |
| Restore | final ellipse | `((x,y),(a,b),ang)` | sensor scale 복원 |

## 짧은 비교

- 학습 경로는 `input`과 dense supervision tensor를 함께 생성합니다.
- 추론 경로는 `input`만 만들고, 이후 post-processing으로 ellipse 1개를 복원합니다.
- 두 경로 모두 같은 event-image 계약을 공유합니다.
  - 표준 FACET 입력: `[2, 256, 256]`
  - 표준 FACET batch 입력: `[B, 2, 256, 256]`

## 공개 코드와 논문 설명 사이 메모

- 논문은 fixed-count event accumulation과 fast causal event volume을 사용한다고 설명합니다.
- 공개 코드는 fixed-count slicing을 명확하게 구현합니다.
- accumulation 구현은 causal accumulation + clipping 쪽이 눈에 더 잘 보입니다.
- 따라서 큰 방법론은 일치하지만, 논문 수준의 세부 구현을 공개 저장소만으로 100% 재현했다고 보기는 어렵습니다.
