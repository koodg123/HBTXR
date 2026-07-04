# 심층 분석 계획 — S-PyTorch 단계 (ViT 양자화 · XR 시선추적 모델)

> 목적: HW 가속기 47편에 이어, **PyTorch 알고리즘 repo**(ViT-Quantization 48 + XR-Eye-Tracking 모델 17)를 `SRC_CASE_MODULE_GUIDE.md`와 동급 깊이의 **S-PyTorch 변형 MODULE_GUIDE**로 작성.
> 전제(기확정): 출력 `REF/Analysis/<cat>/<repo>/MODULE_GUIDE.md`(기존 요약 .md 유지·링크) · 한국어 · third-party 제외 · 전 모듈 정량.
> 핵심 차이: HW의 "MAC lanes/scalar MACs/loop trips/memory(bit)" → PyTorch의 "**FLOPs·params·activation memory·비트폭/관측기(observer)**"로 치환.

---

## 0. 성공 기준 (HW 가이드와 동일 골격, 지표만 치환)
- [ ] 머리말: 대표 케이스(대표 모델·레이어) 선정+근거 · S-PyTorch 수치 규약 · 운영 경로(학습/PTQ↔체크포인트↔평가) · 모델/데이터셋/정확도 기준
- [ ] 모듈(=`nn.Module`/클래스/핵심 함수)마다 6요소: ①역할+상위/하위 ②Mermaid(텐서 shape 흐름) ③forward call stack ④대표 코드 위치 ⑤실제 코드 블록(quantize/observer/attention/scale·zp, 파일:라인) ⑥연산·수치표현 분해 + **정량(FLOPs/params/activation mem/비트폭)**
- [ ] 모듈 한눈표 + 학습·평가 파이프라인(재현 명령) + 우리 프로젝트(FPGA ViT 가속) 시사점
- [ ] 모든 수치/구조에 파일:라인 근거. 미실행/부재는 "확인 불가", 추론은 "추정".

---

## 1. S-PyTorch 변형 템플릿

```markdown
# <repo> 모듈 통합 가이드 (S-PyTorch)
## 0. 머리말
 - 대표 케이스: <대표 모델/블록>(예: DeiT-S block, quantizer+attention)
 - 수치 규약: FLOPs(MACs) / params / activation memory(=shape×bit) / 비트폭(W/A) / observer
 - 운영 경로: (학습 or PTQ 캘리브레이션) → 체크포인트 → 평가(top-1 / p-error 등)
 - 모델/데이터셋/정확도: <backbone>, <ImageNet/3ET+ 등>, <보고 정확도>
## 1. Repo/Layer 개요: 디렉토리 맵(모델 정의 / 양자화 모듈 / 학습·평가 / config) + 외부 프레임워크(timm/DeiT/mmcv/mmrotate 등) 제외 + forward 진입점
## 2..N. 모듈별 6요소 (nn.Module/클래스/핵심 함수 1개 = 1섹션)
## N+1. 모듈 한눈표 (모듈 | 역할 | 양자화 방식 | params/FLOPs)
## N+2. 학습/평가 파이프라인 + 재현 명령
## N+3. 우리 프로젝트(FPGA ViT 가속/XR) 시사점 + FPGA 친화도 평가
```

### 정량 도출법 (정적·분석적, bash 불가 환경)
| 지표 | 도출 방법 |
|---|---|
| params | 모듈 차원(레이어 in/out, dim·depth·heads)에서 분석적 계산 또는 config 근거 |
| FLOPs/MACs | 표준식 × config — Linear: in·out, Attention: ~2·S²·d, Conv: H·W·Cout·Cin·K² (대표 레이어 산출 후 환원) |
| activation memory | 텐서 shape × 비트폭 |
| 비트폭/observer | 코드 직접: W/A bits, per-tensor/channel, symmetric/asym, MinMax/MSE/percentile/Hessian |
| 정확도/속도 | README·논문·로그 인용. 학습/평가 **실행 불가 → 없으면 "확인 불가"** |

---

## 2. 대상 티어링 (총 65 repo)

### ViT-Quantization (48)
- **Tier A — FPGA 직결(정수전용·곱셈기-free·비선형 HW화)**: I-ViT, FQ-ViT, RepQ-ViT, integer-only-transformer, Q-HyViT, P2-ViT, ShiftAddViT, PTQ4ViT, NoisyQuant, outlier-free-transformers, AdaLog, APHQ-ViT, mixed-non-linear-quantization, int-flashattention, INT8-Flash-Attention-FMHA-Quantization, qflash (16)
- **Tier B — PTQ 정밀화·백본·일반**: Castling-ViT, RepViT, Next-ViT, EdgeVisionTransformer, psaq-vit, Q-ViT-DeiT, Bi-ViT, FIMA-Q, OFQ, transformer-quantization, UQ-ViT, AdaTSQ, Mix-Quant, CLAMP-ViT, Quantformer, AHCPTQ, postcalibration4quantization, M3ViT (18)
- **Tier C — 확장 도메인(VLM/Diffusion/SAM/Detector)**: MBQ, qvlm, QuantVLA, Q-DiT, Q-VDiT, ViDiT-Q, ptq4sam, SAQ-SAM, Q-DETR, PQV-Mobile, FPQVAR, mimiq (12) + Castling/q-hyvit 중복정리

### XR-Eye-Tracking 모델 (17)
- **우선(저지연·스트리밍·HW 이식 후보)**: cb-convlstm-eyetracking(3ET), EventMamba, gg_ssms, ais2024(TENNs-Eye), ais2025(TDTracker), EX-Gaze, retina(SNN), E-Track
- **다음(세그/타원·하이브리드·후처리)**: FACET, EV-Eye, EllSeg, RITnet, Swift-Eye, eyegraph, event_based_gaze_tracking, EyeLoRiN
- ESDA: HW 가이드 기존 → XR 관점은 알고리즘 cross-ref 1편

---

## 3. repo별 워크플로우 (7단계)
1. 경계 확정(자체 vs 외부 프레임워크) + forward 진입점 식별
2. 대표 케이스 선정(대표 backbone/block, 또는 quantizer+attention)
3. forward call stack 추출(모델→블록→양자화/observer)
4. 핵심 코드 블록 수집(quantize/scale·zp/attention/비선형, 파일:라인)
5. 연산·수치표현 분해(양자화 방식·비트폭·observer·비선형 처리)
6. 정량 산출(FLOPs/params/activation mem/비트폭, §1식)
7. 다이어그램 + 조립 + 자체검증(§0 체크리스트)

---

## 4. 실행 단계 (세션 한도 대응: 배치 3개·멱등·INDEX 갱신)
- **Pilot**: I-ViT(정수전용 ViT) + cb-convlstm-eyetracking(3ET) → 포맷 확정·승인
- **Phase S1**: Tier A ViT-Quant 16
- **Phase S2**: XR 시선추적 모델 17
- **Phase S3**: Tier B ViT-Quant 18
- **Phase S4**: Tier C ViT-Quant 12
- **Phase SV**: 검증 + INDEX §8에 S-PyTorch 현황표 추가

---

## 5. 품질 게이트 / 검증
- repo별 §0 체크리스트 self-check
- 배치당 1편 표본 정독(코드블록 라인 정합·FLOPs/params 산식 정합·call stack 정확성)
- `_DEEP_VERIFICATION.md`에 [repo | MODULE_GUIDE | 정량충족 | 결손] 누적

---

## 6. 리스크 & 대응
| 리스크 | 대응 |
|---|---|
| bash 불가 → FLOPs/정확도 미실행 | 분석적 산출 + README/논문 인용, 미확보는 "확인 불가" |
| 외부 프레임워크 위 커스텀(timm/mmrotate 등) | 커스텀 부분만 분석, 원본은 외부로 표기 |
| 대형 repo(M3ViT/MBQ/QuantVLA) | 핵심 모듈(양자화·attention) 집중, 주변은 환원 |
| 도메인 상이(Diffusion/VLA/SAM) | 변형 템플릿 유지, "ViT와 차이"를 시사점에 명시 |
| 세션 한도 | 3개/배치·멱등(기존 skip) |

---

## 7. 공수 & 권장 1차
- ~65편, 3/배치 → 약 22배치(세션 한도로 수 차례 분할).
- **권장 파일럿: I-ViT + cb-convlstm-eyetracking** → 두 변형(양자화/시선추적)의 포맷을 한 번에 확정한 뒤 확산.

> 승인 시: 파일럿 2편 작성 → 검수 → Phase S1부터 자동 확산(배치 3개, INDEX 갱신).
