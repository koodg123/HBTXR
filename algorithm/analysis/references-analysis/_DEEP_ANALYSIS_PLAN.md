# 코드베이스 심층 분석 계획 — "SRC_CASE_MODULE_GUIDE 수준" 확산

> 목표: HG-PIPE `SRC_CASE_MODULE_GUIDE.md`와 **동일 수준의 모듈 단위 통합 가이드**를 REF의 각 코드베이스에 대해 생성.
> 결정사항: ①범위=HW 가속기 우선 ②비-HW=유형별 변형 템플릿 ③정량=전 모듈 완전 정량화 ④출력=`REF/Analysis/<cat>/<repo>/MODULE_GUIDE.md`(기존 요약 .md 유지·링크).
> 작성일: 2026-06 · 표기: 근거 없는 수치는 "확인 불가", 추론은 "추정".

---

## 0. 성공 기준 (Acceptance Criteria)

각 `MODULE_GUIDE.md`는 원본 가이드의 다음 요소를 **모두** 충족해야 "이 수준"으로 인정한다.

- [ ] 문서 머리말: 대표 케이스 선정 + 수치 표기 규약 + 운영 경로(소스↔생성물↔플로우) 명시
- [ ] 모듈마다 6요소: ①역할/상위·하위 ②Mermaid 데이터플로우 ③function call stack ④대표 코드 위치(정확 경로) ⑤대표 코드 블록(실제 스니펫+해설) ⑥마이크로아키텍처(stage 분해)+메모리/재사용+**정량 수치**
- [ ] 정량: `MAC lanes`, `scalar MACs`, `loop trips`, `memory size(payload bit)`를 대표 케이스 기준으로 산출(또는 변형 지표)
- [ ] 말미: 모듈 한눈 요약 표 + 읽기 순서 + 코드 추적 순서 + 병목 후보
- [ ] 모든 수치/구조에 파일·라인 또는 파라미터 근거. 도출 불가 시 "확인 불가" 명시

---

## 1. 산출물 표준 (템플릿 사양)

### 1.1 MODULE_GUIDE.md 골격 (HW-HLS 기준 원본 포맷)

```markdown
# <repo> src/case 모듈 통합 가이드
## 0. 문서 머리말
- 대표 케이스: <top1>, <top2> ...
- 수치 표기 규약: MAC lanes / scalar MACs / loop trips / memory size(payload bit)
- 운영 경로: 소스(<path>) ↔ concrete case(<path>) ↔ 생성 스크립트 ↔ 인스턴스 산출물
- 모델/타깃: <model dims>, <board>, <freq/perf if any>
## 1. Repo / Case Layer  (라이브러리 vs 인스턴스 vs refs vs codegen)
## 2..N. 모듈별 섹션 (모듈 1개 = 1섹션, 아래 6요소 고정)
   - 역할 + 상위/하위
   - ```mermaid flowchart``` 데이터플로우
   - Function call stack (case top → src 모듈 → 서브함수)
   - 대표 코드 위치 (파일 경로)
   - 대표 코드 블록 (실제 스니펫 + 1줄 해설) ×2~4
   - 마이크로아키텍처: Stage 분해(M0/M1/...) + 메모리·재사용 + 대표 수치/병목
## N+1. 모듈 한눈 요약 (표)
## N+2. 읽기 순서 / 코드 추적 순서
## N+3. 병목 후보 & 다음 읽기(병렬도/자원)
```

### 1.2 정량 수치 — 정의와 도출법 (전 모듈 완전 정량화)

| 지표 | 정의 | HW(HLS/RTL) 도출법 | 비고 |
|---|---|---|---|
| MAC lanes | 1 cycle 병렬 곱셈 lane | 최내곽 `#pragma HLS unroll` 차원 곱(예: `TP*COP*CIP`) | 코드의 unroll/array partition에서 정적 도출 |
| scalar MACs | 전체 스칼라 곱셈 수 | 수학적 shape 곱(예: `T*CO*CI`) | 텐서 shape에서 해석적 산출 |
| loop trips | II=1 scheduler loop 반복 | tile 차원 곱(예: `TT*COT*CIT`) | 루프 bound에서 도출 |
| memory size | payload bit | 배열 크기×bitwidth(예: `weight_arr=CO*CI*Wb`) | FIFO/BRAM/URAM 구현 오버헤드는 별도 표기 |

- **대표 케이스 환원**: 원본처럼 대표 top(예: 가장 큰 layer, 1개 PE, 1개 head)만 정밀 산출하고, 동형 인스턴스는 "topology 동일, bitwidth/FIFO/LUT만 상이"로 환원 — 단, 환원 시 각 인스턴스의 파라미터 표를 함께 제시해 "전 모듈 정량화" 요건을 만족.
- **bash 불가** 환경이므로 합성 리포트가 동봉된 경우만 LUT/DSP/BRAM/주파수 인용, 없으면 "확인 불가". scalar MACs/loop trips/MAC lanes/memory는 **정적 코드 분석만으로 도출 가능**(원본 가이드와 동일 방식).

### 1.3 템플릿 변형 (Variant)

| 변형 | 대상 | 핵심 치환 |
|---|---|---|
| **H-HLS** | Vitis/HLS C++ 커널 (대부분 HW repo) | 원본 포맷 그대로 |
| **H-RTL** | Verilog/SystemVerilog/SpinalHDL/Chisel (AURA, TATAA, efficient-transformer-accel, MobileVit, XJTU 등) | "case/top"→top module, "src 함수"→submodule; call stack→module 인스턴스 계층; 수치→generate/parameter 기반 MAC array·파이프 depth |
| **H-SIM** | 사이클 시뮬레이터(ViTCoD), codegen 프레임워크(TeraFly/FlexCNN/AGNA) | 모듈→파이썬 모델/패스; 수치→시뮬레이터가 모델링하는 cycle/utilization 식 |
| **S-PyTorch** | ViT 양자화·시선추적 DL repo | 모듈→`nn.Module`/클래스; 데이터플로우→tensor shape; call stack→forward 호출 그래프; 수치→**FLOPs / params / activation memory**, 양자화 비트폭·observer; 대표 코드 블록→양자화/attention 핵심 함수 |
| (제외) | 논문 PDF | 심층 가이드 비대상 — 기존 `Papers/*.md` 유지 |

---

## 2. 대상 선정 & 티어링

> HW 우선. S-PyTorch는 HW 완료 후 후속 단계. 논문 제외.

### Tier 0 — 레퍼런스 (작업 불필요)
- **HG-PIPE**: 원본 가이드 존재 → 모든 산출물의 품질 기준선·표현 레퍼런스.

### Tier 1 — Full Deep (소스 완비 + HG-PIPE 관련도 높음, 최우선)
- HLS 커널: `TATAA`, `FlexLLM`, `FPGA_Friendly_SpinQuant`, `Edge-MoE`, `Diff-DiT`, `trans-fat`, `hls-fpga-accelerators`, `Transformer-Accelerator-Based-on-FPGA`, `Transformer_dataflow`, `ViT-Accelerator(서브디렉토리)`, `ViT-Accelerator-on-FPGA-with-INT8-quantization`, `HLS-Acceleration-of-LLaMA2`, `submission`, `HLSTransformation`
- RTL/SpinalHDL: `AURA-FlashAttention(ASIC)`, `efficient-transformer-accelerator`, `MobileVit-AI-Hardware-Accelerator`, `llama-fpga`, `ternaryLLM(HW부분)`
- Systolic GEMM: `ViT-FPGA-TPU`, `acap-gemm-sa`, `TMMA`, `SJTU_microe-master`
- 이벤트/희소/CNN HW: `ESDA`, `SEE`, `Uint-Packing`, `AnyPackingNet`, `HiSparse`, `HiSpMV`, `HiSpMM`, `DPACS`, `MSD-FCCM23`, `AGNA-FCCM2023`, `REMOT-FPGA-22`, `ViM-Q-FCCM-2026`, `XJTU-Tripler`
- DAC-SDC UltraNet 계열: `dac_sdc_2020/2021/2022/2023`, `SkrSkr`, `SkyNet-ZCU104`, `yolo-fpga-accelerator`, `yolov2_xilinx_fpga`, `Kria-YOLOv4-Tiny`(안티패턴 대조)

### Tier 2 — Medium (교육용/부분/소형) → 핵심 모듈만 정밀, 나머지 환원
- `transformer-hls-thesis`, `vit-tiny-accelerator`, `Tiny-GPT-on-Vortex`(toy MLP), `FlexCNN`, `tataa_tvm_dev`(자체 패스만)

### Tier 3 — Blocked / Best-effort (소스 결손 → "확인 불가" 명시)
- 소스 부재(빌드 산출물만): `LLM_FPGA`, `LUT-LLM`, `TeraFly`, `flightllm_test_demo`
- 빈/부분 repo: `TinyTransformer`(빈), `ViTALiTy`(HW 부재), `Trio-ViT`(알고리즘만), `ViTCoD`(시뮬레이터 only)
- → 가능한 만큼(리포트/디렉토리 규약/시뮬레이터 식) 작성 + 결손 명시. 정량은 도출 가능 항목만.

### S-Tier — 변형(S-PyTorch), HW 완료 후 후속
- ViT-Quantization 48종, XR-Eye-Tracking 모델 17종 → FLOPs/params/양자화 중심 변형 가이드.

---

## 3. repo별 작업 워크플로우 (8단계)

1. **인벤토리/경계 확정**: 자체 소스 vs third-party 재확인, 진입점(top/case/forward) 식별.
2. **대표 케이스 선정**: 가장 큰 layer·1개 PE·1개 head·1개 top 등 "환원 가능한 대표 단위" 결정(원본의 PATCH_EMBED/ATTN0/MLP0/HEAD에 해당).
3. **Function call stack 추출**: top→모듈→서브함수(또는 module→submodule, forward→sub-module) 호출 경로 작성.
4. **대표 코드 블록 수집**: 각 모듈 핵심 동작 스니펫 2~4개 + 1줄 해설(실제 라인 인용).
5. **Stage 분해(마이크로아키텍처)**: 모듈 내부를 M0/M1/… 파이프 stage로 분해, 메모리/재사용 패턴 기술.
6. **정량 도출**: §1.2 식으로 MAC lanes/scalar MACs/loop trips/memory 계산(파라미터 표 포함). 합성 수치는 리포트 있을 때만.
7. **다이어그램 생성**: 모듈별 Mermaid flowchart(+ 필요시 블록 타이밍).
8. **조립 & 자체검증**: §0 체크리스트로 self-check, 누락/확인불가 표기.

---

## 4. 출력 구조 & 네이밍

```
REF/Analysis/<category>/<repo>/
├── MODULE_GUIDE.md          # 본 심층 가이드 (필수)
├── (옵션) PARALLELISM.md     # 병렬도/자원 튜닝 노브 (원본 자매문서 대응, Tier1 HW만)
└── (옵션) CALL_STACK.md      # 호출 스택 상세 (대형 repo만 분리)
기존 REF/Analysis/<category>/<repo>.md  → 1차 요약으로 유지, MODULE_GUIDE 상단에서 상호 링크
INDEX.md → 각 repo 행에 MODULE_GUIDE 링크 컬럼 추가
```

---

## 5. 실행 단계 (Phase) — 세션 한도 대응

> 직전 작업에서 대규모 병렬 에이전트가 세션 한도에 자주 걸렸음. 따라서 **소배치(2~3 repo/배치)** + **멱등(이미 있으면 skip)** + **단계별 사용자 확인**으로 진행.

- **Phase P (파일럿, 1 repo)**: Tier1 대표 1개로 템플릿·정량식·다이어그램 스타일 확정 → 사용자 승인. 권장 후보: `TATAA`(HLS+RTL 혼합·HG-PIPE와 연산 변형 직접 연관) 또는 `ESDA`(이벤트 sparse, XR 연계).
- **Phase 1 (Tier1 HLS)**: 14개 HLS 커널 repo, 배치당 2~3개.
- **Phase 2 (Tier1 RTL/Systolic)**: RTL·SpinalHDL·systolic GEMM repo.
- **Phase 3 (Tier1 이벤트/희소/CNN + DAC-SDC)**: 나머지 Tier1.
- **Phase 4 (Tier2)**: 교육용/부분.
- **Phase 5 (Tier3)**: best-effort + 결손 명시.
- **Phase 6 (S-PyTorch, 선택)**: 양자화·시선추적 모델 변형 가이드.
- **Phase V (검증/INDEX 갱신)**: 각 Phase 종료 시 커버리지·품질 점검, INDEX에 링크 반영.

각 배치는 분석 에이전트가 파일을 직접 저장하고 짧은 확인만 반환(컨텍스트 절약). 배치 간 멱등 Glob로 진행상황 추적.

---

## 6. 품질 게이트 & 검증

- repo별 self-check: §0 체크리스트 통과 여부 표기.
- Phase별 표본검사: 배치당 1개를 정독해 (a) 코드 블록이 실제 라인과 일치, (b) 정량식이 shape/pragma와 정합, (c) call stack 정확성 확인.
- 누락 추적: `_DEEP_VERIFICATION.md`에 [repo | MODULE_GUIDE 유무 | 정량 충족 | 결손사유] 누적.

---

## 7. 리스크 & 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| bash 불가(UNC) | 합성/FLOPs 실행 불가 | 정량은 정적 도출, 합성수치는 리포트 인용·없으면 확인불가 |
| 세션 한도 | 대규모 배치 중단 | 소배치+멱등, 진행상황 디스크 기준 추적 |
| 소스 부재 repo | 라인근거 불가 | Tier3로 분류, 결손 명시, 도출 가능 항목만 |
| 대표 케이스 오선정 | 환원 왜곡 | 파일럿에서 기준 확정, 각 repo 대표 선정 근거 기록 |
| RTL/HLS 혼재 | 포맷 불일치 | H-RTL/H-HLS 변형 분리 적용 |
| 정량 과신 | 잘못된 MAC/메모리 | shape·unroll 근거 라인 명시, 불확실은 추정 표기 |

---

## 8. 공수 추정 (대략)

- Tier1 ≈ 35 repo, Tier2 ≈ 5, Tier3 ≈ 8 → HW 합 ≈ 48 MODULE_GUIDE.
- repo당 가이드 1편 = 중간 규모 1 분석 세션 분량(대형 repo는 2). 배치 2~3/턴 기준 다수 배치 필요 → **세션 한도 때문에 여러 차례 나눠 진행**이 현실적.
- S-PyTorch(선택) 65 repo는 별도 대규모 단계.

---

## 9. 권장 1차 실행

1. **파일럿 1개**(`TATAA` 또는 `ESDA`)로 `MODULE_GUIDE.md`를 만들어 포맷·정량 깊이를 사용자와 합의.
2. 승인되면 Phase 1부터 소배치로 확산.
3. 각 Phase 종료 시 INDEX/검증 갱신.

> 다음 액션: 파일럿 대상(기본=TATAA)을 지정해 주시면 즉시 1편을 작성해 "이 수준"이 맞는지 검수받겠습니다.
