# HG-PIPE 이식 로드맵 (Porting & Extension Roadmap)

> 목적: **HG-PIPE를 기준선**으로, REF에서 심층 분석한 112편(HW 47 + S-PyTorch 65) MODULE_GUIDE의 기법을 **(Track A) 동일 모델 효율 극대화** + **(Track B) XR 이벤트 시선추적 확장** 두 갈래로 이식하는 단계별 계획.
> 근거: 각 항목에 출처 `MODULE_GUIDE.md` 표기. 미검증 추정은 "추정", 합성/실측 미확보는 "확인 필요"로 명시. 합성 PPA 수치는 본 분석 환경에서 미실행이므로 모두 **이식 후 합성 검증 필요**.
> 작성: 2026-06 · 위치: `REF/Analysis/HG-PIPE_PORTING_ROADMAP.md`

---

## 0. HG-PIPE Baseline 진단

출처: [`Transformer-Accel/HG-PIPE/MODULE_GUIDE.md`](Transformer-Accel/HG-PIPE/MODULE_GUIDE.md)

**구조 요약**
- 타깃 **Versal VCK190**, **DeiT-Tiny**(embed 192 / heads 3 / depth 12 / seq 196+CLS), **3-bit 가중치**, **425MHz·7118 FPS·71.05%** (`README.md` 근거).
- 26개 레이어(PatchEmbed + ATTN×12 + MLP×12 + Head)를 각각 독립 HLS 커널로 합성하고, **SpinalHDL `BlockSequence`가 FIFO+핸드셰이크로 체인** → 레이어 단위 coarse + 레이어 내부 dataflow fine의 **2계층 파이프라인**.
- **전 레이어 가중치·중간 텐서 온칩 상주**. matmul은 output-stationary(`matmul.h`, `bind_op impl=dsp`), attention은 Q를 `wb`에 적재 후 K/V를 dynamic weight SRAM에 적재·replay.
- 비선형은 전부 **LUT 기반**(`softmax.h` exp+reciprocal 이중 LUT, `gelu.h`/`quant.h` 단일 LUT, `layernorm.h` rsqrt LUT).
- `step0`가 `ATTN.cpp.template`/`MLP.cpp.template`를 레이어별 case로 생성, `step1`이 target-aware HLS 인스턴스 생성.

**강점**: 전 레이어 온칩 + 전계층 파이프라인 → 초고처리량/저지연, 3-bit + LUT 비선형으로 자원 절감, 비트정합 C-sim(`check_stream`) 내장.

**제약(=이식 동기)**
1. **온칩 상주 전제 → DeiT-Tiny급만 수용**. 더 큰 ViT/긴 시퀀스 불가(자원 한계).
2. **이미지 분류 전용** — 입력=패치, 출력=classifier. 이벤트 입력·시선추적 헤드 없음.
3. **DSP 효율 여지** — output-stationary MAC가 3-bit 가중치를 DSP당 1곱으로만 쓰면 DSP가 비효율적일 수 있음(패킹 미적용 추정 → 합성 리포트로 확인 필요).
4. **attention 3-pass softmax** — 버퍼·패스 수 부담(긴 시퀀스에서 악화).
5. **VCK190 고정** — 에지(ZCU102/Ultra96)·저전력 타깃 이식 시 자원 재맞춤 필요.

---

## 1. 이식 전략 개요

두 트랙을 병행하되, **Track A의 저위험 효율 기법을 먼저 확보**해 자원 여유를 만든 뒤 **Track B(XR 전환)**에 그 여유를 투입한다. HG-PIPE의 `step0/step1` 코드생성 골격과 `BlockSequence` 체인은 **두 트랙 모두에서 재사용**한다(레이어 커널만 교체/추가).

```
[Phase 0] Baseline 재현·계측(PPA/정확도/지연)
   │
   ├── Track A (효율) ──► A1 DSP packing → A2 비선형 정교화 → A3 attention 1-pass → A4 비트폭 탐색
   │                         │ (자원·처리량 확보)
   └── Track B (XR) ──► B1 이벤트 front-end → B2 회귀 head → B3 시간/희소 → B4 에지 보드 → B5 측정
                             ▲
                 (Track A 효율 이득을 XR 가속기에 흡수)
```

---

## 2. Track A — 동일 모델(DeiT-Tiny) 효율 극대화

> HG-PIPE 구조·정확도는 유지하고 **처리량/면적/전력**을 개선. 레이어 커널 내부만 수정 → 파이프라인 골격 불변.

| ID | 이식 기법 | 출처 MODULE_GUIDE | HG-PIPE 통합 지점 | 기대효과 | 난이도 | 리스크 |
|---|---|---|---|---|---|---|
| A1 | **DSP packing**(저비트 다중 MAC/DSP) | `CNN-Accel/Uint-Packing-master`(DSPopt3 4-세그먼트), `ViT-Accelerator/TATAA`(INT8↔bf16 변형 PE) | `matmul.h` MAC(`bind_op impl=dsp`) | 3-bit 가중치를 1 DSP에 다중 곱 → MAC 밀도↑(동일 DSP로 처리량↑ 또는 DSP↓) | 중 | carry/guard 비트 보정, 누산 오버플로 |
| A2 | **비선형 정교화/축소** | `ViT-Quantization/FQ-ViT`(Log-Int-Softmax+PTF), `I-ViT`(ShiftGELU/Shiftmax), `mixed-non-linear-quantization`(함수별 비교표) | `softmax.h`/`gelu.h`/`layernorm.h` LUT | LUT 크기↓ 또는 저비트에서 정확도↑. 시프트 근사로 reciprocal LUT 제거 가능 | 저~중 | 정확도 회귀(재캘리브 필요) |
| A3 | **1-pass online softmax**(곱셈기 없는 exp) | `ViT-Accelerator/AURA-FlashAttention-AISC-Accelerator`(ExpMul + 1-pass + 트리리덕션) | `softmax.h`(현 3-pass) + `attn.h` dynamic matmul | 패스 3→1, raw/exp 버퍼 축소 → attention 지연·BRAM↓ | 중~상 | attn.h 재구성, 수치안정 검증 |
| A4 | **비트폭/표현 탐색**(2-bit·APoT·reparam) | `ViT-Quantization/AdaLog`(로그밑→시프트+LUT), `P2-ViT`(PoT 시프트), `RepQ-ViT`(채널→레이어 reparam) | `quant.h` + 가중치 export(`step0` 통계) | 3→2bit 또는 PoT로 추가 자원↓, reparam으로 outlier 흡수 | 중 | 정확도 하한, QAT 재학습 가능성 |
| A5 | **체계적 DSE 자동화** | `Others/AGNA-FCCM2023`(MIGP DSE+codegen), `Others/HiSpMV`(automation_tool), `Transformer-Accel/TeraFly`(toml→HLS codegen) | `step0/step1` + `SRC_CASE_PARALLELISM_TUNING` 노브(CIP/COP/TP) | 보드 자원 적합 자동 탐색(수동 튜닝 대체) | 중 | DSE 모델 정합성 |

**권장 착수 순서(저위험→고효과)**: A1 → A2 → A4 → A3 → A5.
**핵심**: A1(DSP packing)은 HG-PIPE의 가장 직접적 처리량/면적 레버. 3-bit 가중치라 패킹 이득이 특히 큼(추정).

---

## 3. Track B — XR 이벤트 시선추적 확장

> HG-PIPE의 전계층 파이프라인·코드생성 골격을 **이벤트 기반 시선추적 가속기**로 재목적화. 핵심 질문은 "어떤 시선추적 모델이 HG-PIPE 피드포워드 파이프라인에 가장 잘 매핑되는가".

### 3.1 모델 선택 (파이프라인 재사용성 기준)

| 후보 모델 | 출처 | HG-PIPE 매핑성 | 비고 |
|---|---|---|---|
| **ViT/Transformer 시선추적**(BRAT, EX-Gaze MobileViTv2 선형attn) | `Papers/BRAT-CVPRW25`, `XR/EX-Gaze` | ★★★ 최상 — ATTN/MLP 커널 그대로 재사용, head만 교체 | 가장 적은 구조변경. 1순위 |
| **TENNs-Eye**(인과 Conv3d 스트리밍) | `XR/ais2024`(TENNs-Eye) | ★★★ 높음 — 인과 Conv3d는 FIFO 스트리밍이라 파이프라인 친화 | 시간축 스트리밍에 적합 |
| **3ET / CB-ConvLSTM**(delta sparsity) | `XR/cb-convlstm-eyetracking`, `Papers/3ET` | ★★ 중 — 시간 재귀(상태 보유) 필요, delta sparsity로 연산↓ | 경량(0.42M), 재귀 상태 관리 추가 |
| **ESDA**(이벤트 sparse CNN) | `CNN-Accel/ESDA` | ★ 낮음(구조 상이) — sparse dataflow는 HG-PIPE dense 파이프라인과 패러다임 다름 | 단, 이벤트 희소성 활용은 최고. 별도 dataflow 필요 |
| **retina**(IAF SNN) | `XR/retina` | ★ 낮음 — 뉴로모픽(Speck)용, FPGA dense와 상이 | mW급 저전력 참고용 |

**권장**: **1차로 ViT 기반 시선추적**(HG-PIPE 커널 최대 재사용) → 2차로 **TENNs-Eye식 인과 스트리밍**(저지연·시간모델) 또는 **3ET delta sparsity**(연산 절감) 결합.

### 3.2 구성요소 이식 매핑

| ID | 구성요소 | 이식 기법/출처 | HG-PIPE 통합 지점 | 비고 |
|---|---|---|---|---|
| B1 | **이벤트 입력 front-end** | 이벤트 표현(voxel/time-bin frame) — `Papers/3ET`(time-bin), `XR/ESDA`(token+mask), `XR/event_based_gaze_tracking`(9B BHHI 포맷) | `patch_embed.h` 대체(이벤트→토큰/프레임 임베딩) | 이벤트 슬라이싱·텐서화 모듈 신설 |
| B2 | **출력 head**(분류→회귀) | 좌표회귀/heatmap/타원 — `XR/FACET`(타원+GWD), `XR/E-Track`(세그+타원피팅), `Papers/TDTracker`(SimDR heatmap) | `head.h`(classifier) 대체 | 가장 간단: (x,y) 좌표 회귀 |
| B3 | **시간 모델/희소성** | 인과 Conv3d(`ais2024 TENNs-Eye`), delta sparsity(`3ET`), 선형 attention(`ViTALiTy`/`Castling-ViT`) | `attn.h`/`mlp.h` 또는 신규 temporal 커널 | 저지연·연산절감. SSM(EventMamba)은 selective scan HW화 난이도↑ |
| B4 | **에지 보드 포팅** | 자원 재맞춤 — Track A(A1/A4) + `SRC_CASE_PARALLELISM_TUNING` 노브 | `step1` target 설정(ZCU102/Ultra96) | VCK190→에지 시 자원 적합 핵심 |
| B5 | **on-device 측정 하네스** | `CNN-Accel/SEE`(ZCU102 PYNQ 지연 + INA226 18레일 전력 동시측정) | board flow(step3/step4) | 지연·전력 실측 표준화 |
| B6 | **데이터셋/메트릭** | `eyetracking-dataset-benchmark-harness` 스킬, EV-Eye/3ET+ | 학습·평가 SW | p-error/거리, p10 정확도 |

### 3.3 후처리(선택, 무재학습)
- `XR/EyeLoRiN`(median 필터 + ROI 옵티컬플로우 정렬) / `Papers/GazeRefine`(motion-aware) → 출력 jitter 억제. 모델 변경 없이 추론단 추가.

---

## 4. 통합 단계 계획 (Phase)

| Phase | 내용 | 산출물 | 의존성 |
|---|---|---|---|
| **P0** | HG-PIPE baseline 재현·계측(step1 합성, PPA/정확도/지연 기준선) | baseline PPA 표 | — |
| **P1** | Track A: A1 DSP packing + A2 비선형 정교화 (DeiT-Tiny 유지) | 처리량/면적 개선판 + 비트정합 검증 | P0 |
| **P2** | Track A: A4 비트폭 탐색 + A3 1-pass softmax | 저자원/저지연판 | P1 |
| **P3** | Track B: B1 이벤트 front-end + B2 회귀 head를 **ViT 시선추적 모델**에 적용(HG-PIPE 커널 재사용) | 이벤트 시선추적 v1(피드포워드) | P1(효율 흡수) |
| **P4** | Track B: B3 시간/희소(TENNs-Eye 인과 Conv3d 또는 3ET delta) + B4 에지 보드 포팅 | 저지연 스트리밍 시선추적 가속기 | P3 |
| **P5** | Track A 효율 이득을 XR 가속기에 통합 + B5 on-device 측정 + A5 DSE | 최종 XR 가속기 + 실측 PPA | P2,P4 |

---

## 5. 검증 방법론
- **기능 비트정합**: HG-PIPE 내장 `check_stream`(C-sim) + SW 골든(PyTorch) ↔ HLS/RTL 비트 비교. 각 이식 커널마다 적용.
- **합성 PPA**: `step1` 재합성 → LUT/FF/DSP/BRAM/URAM·달성 주파수. baseline 대비 회귀 추적(`fpga-resource-power-reporting` 스킬 양식).
- **정확도**: Track A=ImageNet Top-1, Track B=3ET+/EV-Eye p-error·p10. 이식 전후 비교.
- **지연/처리량**: `latency-throughput-characterization` 양식(평균 + p95/p99 + 지속 이벤트율). end-to-end 경계는 `e2e-latency-boundary-accounting`로 센서~결과 vs 코어 구분.
- **on-device**: SEE 하네스로 ZCU102 실측 지연+전력.

---

## 6. 리스크 레지스터
| 리스크 | 영향 | 완화 |
|---|---|---|
| 온칩 상주 전제와 이벤트 모델 메모리 패턴 불일치 | Track B 구조 충돌 | 1차는 **dense 이벤트-프레임 + ViT**로(파이프라인 유지), sparse(ESDA)는 후순위 |
| DSP packing carry/guard 오류 | 정확도 손상 | Uint-Packing의 보정식(`(p>>1)+(p&1)`) 그대로 차용 + 비트정합 회귀 |
| 3-bit/2-bit 회귀 정확도(특히 좌표회귀) | 시선추적 정밀도 저하 | QAT 재학습(I-ViT/FQ-ViT 레시피) + 비선형은 mixed-non-linear 비교표로 선택 | 
| VCK190→에지 자원 초과 | 합성 실패 | Track A 효율 선확보 + 병렬도 노브(CIP/COP/TP) 축소 |
| 합성 PPA 미확보(본 분석 환경 한계) | 효과 정량 불확실 | 각 Phase 종료 시 실합성 필수, 본 로드맵 수치는 "추정" |

---

## 7. 즉시 착수 권장 (Quick wins)
1. **A1 DSP packing**(`matmul.h`): 3-bit 가중치 → 가장 큰 처리량/면적 레버. `Uint-Packing-master/MODULE_GUIDE.md`의 비트배치 그대로 시작.
2. **A2 Log-Int-Softmax/ShiftGELU**(`softmax.h`/`gelu.h`): reciprocal/대형 LUT 축소. `FQ-ViT`/`I-ViT` 가이드 참조.
3. **B-탐색 스파이크**: HG-PIPE ATTN/MLP 커널을 그대로 두고 `patch_embed.h`→이벤트 임베딩, `head.h`→(x,y) 회귀로 바꾼 **최소변경 이벤트 시선추적 PoC**(P3 선행).

> 참고 문서: 전체 카탈로그·교차분석은 [`INDEX.md`](INDEX.md) §8(HW)·§9(S-PyTorch), HG-PIPE 상세는 [`Transformer-Accel/HG-PIPE/MODULE_GUIDE.md`](Transformer-Accel/HG-PIPE/MODULE_GUIDE.md).
