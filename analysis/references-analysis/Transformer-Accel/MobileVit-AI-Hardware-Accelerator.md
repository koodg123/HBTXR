# MobileViT AI Hardware Accelerator — 정밀 분석

> 분석 대상: `REF/Transformer-Accel/MobileVit-AI-Hardware-Accelerator`
> 작성 기준: 실제 소스(RTL SystemVerilog + Python)를 Read로 직접 확인, 파일:라인 근거 표기
> 표기 규칙: **[확실]** = 코드에 명시, **[추정]** = 코드/문서 정황 근거, **[확인 불가]** = 소스 부재

---

## 1. 개요

- **무엇인가**: MobileViT(경량 CNN+Transformer 하이브리드 이미지 분류망) 추론을 위한 FPGA 타깃 AI 하드웨어 가속기. SystemVerilog RTL 데이터패스 + Python 모델링/골든모델로 구성된 **학생/졸업연구 프로젝트**(README.md L5, L49-50, L80; COMPLETE_DATA_FLOW_GUIDE.md L6-14).
- **한 줄 요약**: 재구성형(LEGO) weight-stationary 16×16 시스톨릭 어레이 4타일 + 디스크립터 기반 글로벌 컨트롤러 + AGU(im2col 주소생성) + 핑퐁 SRAM + 후처리(BatchNorm/Swish/LayerNorm) 데이터패스를 AXI로 묶은 단일 코어 가속기.
- **MobileViT 특성 반영**: 패키지 `layer_type_t`에 Conv3×3/Conv1×1/DWConv3×3/MV2 블록/Q·K·V matmul/QKᵀ/Attn×V/Global Pool/FC/MViT 블록까지 12종 레이어 타입이 enum으로 정의됨(accelerator_common_pkg.sv L64-77). 즉 CNN(MobileNetV2 stem)과 Transformer(self-attention)를 한 코어가 모두 처리하도록 설계된 의도가 명확.
- **출처/타깃**:
  - AGU 계열 모듈 헤더에 "STMICROELECTRONICS / AI Accelerators Hands-on HW Design - Jul.2025 / author: Mahmoud Abdo" 명시(AGU.sv L1-9, ConvOffsetsAGU.sv L1-11) → AGU는 ST 교육과정 산출물을 통합한 것 **[확실]**.
  - 타깃 디바이스: Xilinx Zynq UltraScale+ FPGA, Vivado 합성, ModelSim/Vivado Sim/Verilator 시뮬, SystemVerilog IEEE 1800-2017(README.md L56-61).
  - 설계 목표(미검증): 400 MHz, ~100 GOPS(INT8), 160 KB on-chip SRAM, <5W(README.md L37-45). **[추정]**(합성/측정 전 hand-calc).

---

## 2. 디렉토리 구조 (자체 RTL + Python + 문서)

```
MobileVit-AI-Hardware-Accelerator/
├── README.md                              # 프로젝트 개요/타깃/상태
├── IMPLEMENTATION_GUIDE.md                # 아키텍처 다이어그램, 레지스터맵, 디스크립터 포맷, 데이터플로우 8단계
├── Include/                               # 패키지(타입/파라미터)
│   ├── accelerator_pkg.sv                 # 통합 import 패키지
│   ├── accelerator_common_pkg.sv         # ★ 폭/버퍼/SA 파라미터, 모든 enum, descriptor_t, 플래그
│   ├── accelerator_matmul_pkg.sv         # matrix_id_t, 타일 FSM/phase enum
│   ├── accelerator_norm_pkg.sv           # LayerNorm FSM enum, NORM_SCALE_FACTOR, SQRT_ITERATIONS
│   └── accelerator_activation_pkg.sv     # Swish 상수(3, 6)
├── RTL/
│   ├── mobilevit_accelerator_top.sv      # ★ 최상위 통합(AXI Slave/Master, 모든 서브모듈 인스턴스)
│   ├── Control/global_controller.sv      # ★ 19-state 글로벌 FSM(DMA/AGU/SA/PP/핑퐁 오케스트레이션)
│   ├── DMA/dma_wrapper.sv                # AXI4 MM2S/S2MM DMA(64bit AXI ↔ 32bit mem)
│   ├── AGU/AGU.sv                        # 주소생성 상위(타일생성+오프셋생성)
│   ├── AGU/Components/ConvTileIndicesGenerator.sv  # 타일 인덱스 iterator (※ 본 분석 미독)
│   ├── AGU/Components/ConvOffsetsAGU.sv  # ★ Conv/DWConv/PW/Matmul 주소 함수(HWC, im2col, 대각DW)
│   ├── Memory Subsystem/memory_subsystem.sv  # ★ 16뱅크 SRAM(ActA/ActB/Wgt/PSum), 핑퐁, 누산
│   ├── Compute/sa_compute_unit.sv        # ★ Lego SA 래퍼 + 누산(accumulation)
│   ├── Compute/post_processing_pipeline.sv  # ★ BN→Swish→LN 파이프(현재 대부분 bypass)
│   ├── Lego SA/SA NxN/                   # 단일 NxN 시스톨릭 어레이(독립형)
│   │   ├── PE.sv                         # ★ weight-stationary MAC 셀
│   │   ├── SA_NxN.sv                     # ★ NxN PE 메시
│   │   ├── SA_CU.sv                      # 단일 SA 제어 FSM(IDLE→LOAD_W→FEED_A→DRAIN→OUTPUT)
│   │   ├── SA_NxN_top.sv                 # CU+TRSRL+SA+TRSDL 통합(자립형)
│   │   ├── TRSRL.sv                      # ★ 삼각 시프트 입력 스큐(활성화)
│   │   └── TRSDL.sv                      # ★ 삼각 시프트 출력 디스큐(psum)
│   ├── Lego SA/Lego SA/                  # ★ 재구성형 LEGO 어레이(4타일)
│   │   ├── Lego_SA.sv                    # ★ 4×(16×16) 타일 + 공유 CU + 라우팅/출력 mux
│   │   ├── Lego_CU.sv                    # ★ 단일 공유 제어 FSM
│   │   ├── L_SA_NxN_top.sv               # 제어 없는 타일(TRSRL+SA+TRSDL)
│   │   ├── PE.sv / SA_NxN.sv / TRSRL.sv / TRSDL.sv  # SA NxN과 동일 계열
│   ├── layer_normalization/             # ★ LayerNorm 데이터패스(파이프라인 분해)
│   │   ├── layer_normalization_top.sv   # SUM→MEAN→sub/mul→SUM2→VAR→1/√→NORM 연결
│   │   ├── ELEMENTS_SUM.sv / ELEMENTS_SUM_2.sv  # 재귀 adder tree
│   │   ├── MEAN.sv / VARI.sv            # 합 >> log2(N) (산술 우시프트 = 나눗셈)
│   │   ├── array_multiplier_subtractor.sv  # (x-mean), (x-mean)^2
│   │   ├── standard_deviation_inv.sv    # ★ 1/√var (root_1/root_2/root_final)
│   │   ├── root_1.sv                    # 지수 추출(MSB 위치 k, 가수 m)
│   │   ├── root_2.sv                    # ★ 256엔트리 1/√ LUT 초기값 y0
│   │   ├── root_final.sv               # √2 보정상수 + 시프트로 최종 1/√
│   │   └── NORMALIZATION_OUT.sv         # (x-mean)*std_dev_inv, Q포맷 시프트
│   ├── Batch Norm/Batch_Normalization.sv  # ★ y = A·x + B (γ/β 320채널 ROM), Q8.8
│   ├── Batch Norm/Batch_Norm_row.sv     # 행 단위 변형
│   ├── Swish/swish.sv                   # ★ hard-swish ≈ x·ReLU6(x+3)/6 (시프트-합 근사)
│   ├── Swish/swisharray.sv              # N병렬 swish 래퍼
│   └── Integration/SA_Batch.sv (+ tb)   # SA + BatchNorm 통합 데모
├── Python modelling/
│   ├── MobileVit Pytorch Model/MVT.py + Mapping_PretraindModel_MVT.py  # PyTorch MobileViT(MV2/LocalRep/Transformer)
│   ├── MobileVit NO API Model/V2,V3/*.py  # no-API(순수 numpy) 모델 + 가중치 매핑
│   ├── layer_normalization_golden_model/golden_model.py  # ★ float 골든
│   ├── layer_normalization_golden_model/layer_normalization_approximation.py  # ★ fixed-point RTL 미러
│   └── Batch Norm/generate_bn_from_txt.py  # BN 파라미터 생성
├── Documentation/  # COMPLETE_DATA_FLOW_GUIDE, READING_ORDER, PRESENTATION, EXECUTIVE_SUMMARY,
│                   # OPERATIONS_TABLE_AND_HARDWARE_FLOW, MEMORY_BANKING_ARCHITECTURE, DOCUMENTATION_GUIDE
└── Testbench/      # Swish/BatchNorm/LayerNorm/Lego SA/SA NxN/top 테스트벤치
```

**제외(이름만 언급)**: `.git`, `Python modelling/Apple Pretrained Model/*.bin·*.h5`(사전학습 가중치), `*.xlsx`(문서 표), `*.txt`(시뮬 입출력 덤프: normalized_out/sa_out/rtl_bn_out 등), `cases.m`/`wave.do`(MATLAB/ModelSim 보조), `__pycache__`, 비트스트림/합성산출물.

---

## 3. 핵심 모듈 정밀 분석

### 3.1 패키지 (Include/) — 파라미터·타입 기반

**accelerator_common_pkg.sv** — 전 모듈 공유 상수/타입.
- 데이터 폭: `ACT_WIDTH=8`, `WEIGHT_WIDTH=8`(INT8), `PSUM_WIDTH=32`(INT32), `DATA_WIDTH=32`(mem), `ADDR_WIDTH=32`, `IDX_WIDTH=16`(L11-16). → **INT8 MAC, INT32 누산**이 기본 수치 포맷 **[확실]**.
- 버퍼: `ACTBUF_DEPTH=8192`(32KB), `WGTBUF_DEPTH=8192`(32KB), `PSUMBUF_DEPTH=16384`(64KB), `NUM_BANKS=4`(L21-24). ※ 주의: 실제 `memory_subsystem.sv`는 `NUM_BANKS=16`을 자체 파라미터로 사용(L1-7) → 패키지 상수와 불일치 **[확실, 불일치]**.
- SA: `SA_SIZE=64`(LEGO 최대), `SA_ROWS=16`, `SA_COLS=16`(L29-31).
- enum: `op_mode_t`{REGULAR_CONV, POINTWISE, DEPTHWISE, MATMUL}(L54-59), `layer_type_t` 12종(L64-77), `buffer_id_t`{ActA, ActB, Wgt, PSum}(L82-87), `sa_type_t`{16X64, 32X32, 64X16, 16X16}(L92-97).
- `descriptor_t`: 256bit 패킹 구조체 — dram_addr[31:0], sram_addr[15:0], length, stride, reserved, tile_h/tile_w/c_in, flags[7:0](L102-113). 플래그 비트: IS_WEIGHT/IS_LAST_TILE/ENABLE_BN/ENABLE_SWISH/ENABLE_LN/TRANSPOSE/ACCUMULATE/WRITEBACK(L116-123).
- AXI: `AXI_DATA_WIDTH=64`(L134) → DMA가 64bit AXI를 32bit mem으로 변환.

**accelerator_norm_pkg.sv**: `NORM_SCALE_FACTOR=128`, `SQRT_ITERATIONS=4`(Newton-Raphson 의도), LayerNorm FSM enum(LN_IDLE/LOAD/SUM1/MEAN/SUM2/VARI/NORM/DONE)(L13-28). ※ 실제 `layer_normalization_top.sv`는 이 FSM enum을 사용하지 않고 valid 핸드셰이크 체인으로 구현됨(아래 3.6) **[확실, FSM enum 미사용]**.

**accelerator_activation_pkg.sv**: `SWISH_CONST_THREE=3`, `SWISH_CONST_SIX=6`(hard-swish 상수)(L13-14).

**accelerator_matmul_pkg.sv**: `matrix_id_t`{A,B,C,INVALID}, 타일 컨트롤/AGU FSM enum, phase enum(L13-45).

---

### 3.2 시스톨릭 어레이 — PE / SA_NxN / TRSRL / TRSDL

이 가속기의 연산 핵심. **weight-stationary** 방식.

**PE.sv (Processing Element)** — 단일 MAC 셀.
- 포트: 활성화 좌→우(`in_act`/`out_act`), psum 상→하(`in_psum`/`out_psum`), 가중치 로드 방향 2개(`w_in_down`/`w_in_left` 입력, `w_out_up`/`w_out_right` 출력), 제어(`load_w`, `transpose_en`)(L31-55).
- 동작:
  - 가중치 래치: `load_w && !transpose_en` → `W_reg<=w_in_down`(아래에서 위로 전파), `load_w && transpose_en` → `W_reg<=w_in_left`(오른→왼)(L76-83). → **전치(Kᵀ) 지원이 하드웨어 가중치 로딩 방향 전환으로 구현됨** — attention QKᵀ에 직접 대응 **[확실]**.
  - MAC: `mac_mul = in_act * W_reg`(2*DATA_W 폭, 오버플로 방지), `mac_res = mac_mul + in_psum`(L65-69). `!load_w`일 때 `act_reg<=in_act`(우측 전달), `psum_reg<=mac_res`(하향 전달)(L89-98).
  - 가중치 전달: normal이면 `w_out_up=W_reg`, transpose면 `w_out_right=W_reg`, 비활성 방향은 0(L107-108).

**SA_NxN.sv (NxN 메시)** — 기본 16×16 PE 그리드.
- 경계: `act_in[k]`가 행 k 좌측 진입, `psum_sig[0][k]=0`(상단 0 주입), `weight_D_sig[N][k]=weight_in[k]`(하단 경계, normal), `weight_L_sig[k][N]=weight_in[k]`(우측 경계, transpose)(L78-85).
- PE 배열: `PE[i][j]`가 act 좌→우, psum 상→하, 가중치 하→상(normal)/우→좌(transpose)로 연결(L93-114). 출력은 하단 행 `psum_out[j]=psum_sig[N][j]`(L116-118).
- N_SIZE 사이클 가중치 로드 후 `PE[row][col]=W[row][col]` 보장(헤더 L16-25).

**TRSRL.sv (입력 스큐)** — 삼각 시프트레지스터.
- lane k를 k 사이클 지연(lane 0=직결, lane k=k개 레지스터). 총 레지스터 = N(N-1)/2(L46-52). 1-based flat 배열 인덱싱(base = k(k-1)/2)(L66-100). → weight-stationary 대각 wavefront에 맞춰 활성화가 줄지어 진입하도록 정렬 **[확실]**.

**TRSDL.sv (출력 디스큐)** — TRSRL과 구조 동일, psum에 적용(기본 DATAWIDTH=32). 컬럼 j가 j 사이클 늦게 완료되는 것을 보정해 N개 결과가 동시 출력되게 정렬(L1-36, L60-105). `SA_NxN_top`/`L_SA_NxN_top`은 TRSDL 전후로 컬럼 순서를 mirror(`psum[N-1-i]`)하여 가장 지연 큰 컬럼을 레지스터 가장 많은 lane에 매핑(SA_NxN_top.sv L124-130).

**SA_CU.sv (단일 SA 제어)**: FSM IDLE→LOAD_W→FEED_A→DRAIN→OUTPUT. LOAD_W/FEED_A는 `valid_in`으로 stall 가능(카운터 freeze), DRAIN(N-1)/OUTPUT(N)은 자율 진행. `load_w=(state==LOAD_W)&&valid_in`, `valid_out=(state==OUTPUT)`(L94-162). 단일 16×16 자립형(`SA_NxN_top`)에서 사용.

---

### 3.3 LEGO 재구성형 시스톨릭 어레이 — Lego_SA / Lego_CU / L_SA_NxN_top

이 프로젝트의 **차별 포인트**. 4개의 16×16 타일(RU/LU/RD/LD)을 런타임에 3가지 논리 형상으로 재구성(Lego_SA.sv L1-72).

**3가지 형상(lego_type)** (Lego_SA.sv L19-58):
| TYPE | 논리 형상 | A | W | C(출력) | 출력 결합 | 활성 결과 수 |
|------|----------|---|---|---------|-----------|-------------|
| 0 (16X64) | 16행×64열 | 16×16 | 16×64 | 16×64 | 4타일 단순 concat `{RU|LU|RD|LD}` | 64 |
| 1 (32X32) | 32행×32열 | 16×32 | 32×32 | 16×32 | 좌쌍합 RU+RD, 우쌍합 LU+LD | 32 |
| 2 (64X16) | 64행×16열 | 16×64 | 64×16 | 16×16 | 4타일 전부 element-wise 합 | 16 |

- **활성화 라우팅**(L217-254): TYPE0=전 타일 broadcast, TYPE1=상/하 절반 분할, TYPE2=4개 독립 슬라이스. `!load_w_cu && valid_in`일 때만 구동.
- **가중치 라우팅**(L187-199): 4등분 고정 배선, `load_w_cu`일 때만 슬롯에 적재(caller가 lego_type에 맞춰 weight 행을 배치하는 책임 — 헤더 L159-185에 cycle별 배치 규칙 명시).
- **출력 adder tree**(L130-138): `psum_add_left=RU+RD`, `psum_add_right=LU+LD`, `psum_add_all=RU+RD+LU+LD` 사전 계산 후 출력 mux에서 lego_type별 선택(L270-297).
- **타일 인스턴스**(L307-365): `L_SA_NxN_top` 4개, 모두 단일 `load_w_cu`로 동기 제어.

**Lego_CU.sv (공유 제어)**: 단일 FSM이 4타일을 동시 제어. 헤더에 "타일별 SA_CU를 쓰면 start 펄스가 1 valid 사이클을 소모해 영구 위상 오프셋이 생기므로, 외부 단일 CU가 load_w를 직접 구동해 내부 FSM과 오프셋을 제거"라고 설계 의도 명시(L9-16). 위상: LOAD_W(N_TILE)→FEED_A(N_TILE)→DRAIN(N_TILE-1)→OUTPUT(N_TILE). LOAD_W/FEED_A는 valid_in stall 가능, DRAIN/OUTPUT 자율(L106-181). → **재구성형 SA의 위상 정합 문제를 단일 컨트롤러로 해결한 점이 핵심 설계 결정** **[확실]**.

**L_SA_NxN_top.sv**: 제어 없는 타일(TRSRL+SA_NxN+TRSDL), `load_w`만 외부 입력(L39-115). `SA_NxN_top`과 달리 자체 CU 없음.

> **[확실, 통합 결함]** `sa_compute_unit.sv`(L51-65)는 `Lego_Systolic_Array #(...)` 모듈을 포트 `valid_in/act_in/weight_in/TYPE_Lego/load_w/transpose_en/psum_out/valid_out`으로 인스턴스화하지만, 실제 RTL 모듈명은 `Lego_SA`이고 포트는 `lego_type`(`TYPE_Lego` 아님)이며 `y_input_size`를 요구한다(Lego_SA.sv L74-97). 즉 **top→sa_compute_unit→Lego_SA 인스턴스화는 현재 이름/포트 불일치로 그대로는 elaborate 불가**. 통합이 미완(IMPLEMENTATION_GUIDE.md L425 "Post-Processing Partial", README.md L49 "design/verification phase")인 정황과 일치 **[추정→확실 근거]**.

---

### 3.4 SA Compute Unit (Compute/sa_compute_unit.sv)

- 역할: LEGO SA를 감싸 **다중 타일 누산(accumulation)** 제공(C_in>16일 때 부분합 합산)(헤더 L2-9).
- 누산 로직(L72-83): `accum_en && sa_valid_raw`이면 `psum_out[i]=sa_psum_raw[i]+psum_in[i]`, 아니면 `=sa_psum_raw[i]`. SA_SIZE=64 전부에 generate.
- 인터페이스: act/weight/psum_in 각 64개, psum_out 64개, valid/done(L29-39). done은 단순히 valid 통과(L88-93, 향후 사이클 카운터 보강 TODO 주석).

---

### 3.5 Post-Processing Pipeline (Compute/post_processing_pipeline.sv)

- 의도: PSum 16엘리먼트/사이클을 **BN → Swish → LN** 3단 파이프로 처리(헤더 L2-3, L25-30).
- **현 상태: 대부분 bypass(미완)** **[확실]**.
  - Stage1 BN(L62-93): `bn_enable`여도 `bn_out[i]<=data_in[i]`(주석 "Bypass for now", 실제 `batch_norm.sv` 인스턴스 미연결).
  - Stage2 Swish(L95-136): `swish_enable`여도 부호/오버플로 분기만 있고 전부 pass-through(주석 "In real design, instantiate swish.sv").
  - Stage3 LN(L138-169): bypass(주석 "Layer norm would require mean/variance...For now bypass").
- 즉 BN/Swish/LN **실제 연산 모듈들은 따로 존재하지만(아래 3.6~3.8) 파이프에 연결돼 있지 않다**. top에서도 BN 파라미터를 상수(mean=0,var=1,gamma=1,beta=0)로 묶어 사실상 항등(mobilevit_accelerator_top.sv L582-587).

---

### 3.6 LayerNorm 데이터패스 (RTL/layer_normalization/)

가장 완성도 높은 자체 연산 데이터패스. `layer_normalization_top.sv`가 valid 핸드셰이크로 7스테이지 체인(L1-124):

```
activation_in[0:31]
  → ELEMENTS_SUM(재귀 adder tree) → sum_1_out
  → MEAN: mean = sum_1 >>> log2(EMBED_DIM)         # 산술우시프트=÷N (MEAN.sv L12)
  → array_multiplier_subtractor: sub=x-mean, mul=(x-mean)^2
  → ELEMENTS_SUM_2(2*DATA_WIDTH) → sum_2_out
  → VARI: vari = sum_2 >>> log2(EMBED_DIM)          # 분산 (VARI.sv L10)
  → standard_deviation_inv: 1/√vari
  → NORMALIZATION_OUT: out = (x-mean) * std_dev_inv
```
기본 파라미터: `DATA_WIDTH=32, EMBED_DIM=32, K_WIDTH=6, M_WIDTH=17`(layer_normalization_top.sv L1-6).

**ELEMENTS_SUM.sv**: `$clog2(EMBED_DIM)` 스테이지 재귀 파이프라인 adder tree. 각 스테이지 폭+1bit로 확장, EMBED_DIM=1에서 base case(L13-63). → log2(N) 사이클 합산 **[확실]**.

**MEAN.sv / VARI.sv**: 나눗셈을 산술 우시프트로 구현(EMBED_DIM이 2의 거듭제곱 전제)(MEAN.sv L12, VARI.sv L10). → 면적/지연 절감, EMBED_DIM=2^n 제약 **[확실, 한계]**.

**standard_deviation_inv.sv (1/√var)** — 3단 서브모듈(L23-67):
1. **root_1.sv**: 분산을 `vari = 2^k · m` 형태로 분해. MSB 위치를 카운터로 탐색(`counter`를 2*DATA_WIDTH-1부터 감소시키며 `vari[counter]` 첫 1 검출, L28-104), `k=shifter=(counter-DATA_WIDTH/2)`, `m=vari>>shifter`(L86-94). → **범위 축소(range reduction)** **[확실]**.
2. **root_2.sv**: `m[15:8]`(8bit)을 인덱스로 **256엔트리 LUT**에서 1/√ 초기값 `y0_1`(Q2.14, 1/√1.0~1/√2.0 구간 테이블) 읽음(L16-289). 예: idx 0x00→0x3FF0(≈1/√1.002), 0xFF→0x2D47(≈1/√1.998). default 0x4000(=1.0). → **LUT 기반 초기 추정** **[확실]**.
3. **root_final.sv**: 지수 보정. k가 홀수면 √2 보정상수 `0x2D41`(Q2.14≈1/√2) 곱, 짝수면 단순 시프트. `shifter=k>>1`(짝수)/`(k-1)>>1`(홀수), `std_dev_inv = (y0>>shifter)*const`(홀) / `(y0>>shifter)`(짝)(L36-52). 출력 Q포맷: 홀수 k→Q4.28, 짝수 k→Q18.14(주석 L12).

> 즉 표준 부동소수 rsqrt의 "지수 반감 + 가수 LUT" 기법을 고정소수점으로 구현. `accelerator_norm_pkg.SQRT_ITERATIONS=4`는 Newton-Raphson 의도였으나 실제는 **LUT+시프트 단발**(NR 반복 코드 없음) **[확실, 패키지 의도와 구현 상이]**.

**NORMALIZATION_OUT.sv**: `(x-mean)*std_dev_inv`를 32×병렬 곱(generate, L39-44) 후 k 패리티에 따라 시프트(홀수: `[28+W-1:28]`, 짝수: `[14+W-1:14]`)로 Q24.8 복원(L24-37). → **fixed-point Q포맷 동적 정렬** **[확실]**.

---

### 3.7 BatchNorm (RTL/Batch Norm/Batch_Normalization.sv)

- 어파인 정규화 `y = A·x + B`(A=γ, B=β). 추론 시 mean/var를 γ',β'로 folding한 형태로 해석 **[추정]**.
- 포맷: Q8.8(Data_Width=16, FRAC_BITS=8), N=32 엘리먼트/행(L1-5).
- 동작(L46-64): `mult_result = A[i+base]·x[i]`(Q16.16), B를 sign-extend·Q16.16 정렬 후 가산, `y = sum[Data_Width+FRAC_BITS-1:FRAC_BITS]`로 Q8.8 절단.
- **채널 ROM**: A/B는 `[0:319]` 320엔트리(L11-12), `channel_base` 카운터가 N=32씩 진행해 320에서 wrap(L30-42). → 최대 320채널 BN 파라미터를 내장(하드코딩 입력) **[확실]**. MobileViT XXS 채널 수에 맞춤 **[추정]**.

---

### 3.8 Swish/SiLU 활성 (RTL/Swish/)

- **swish.sv**: hard-swish 근사. `relu6(x+3)` 계산 후 `product = x·relu6_val`, 이를 6으로 나누는 대신 **1/6 ≈ Σ 2^-3,-5,-7,-9,-11,-13** 시프트-합으로 근사(L25-49). 즉 `y ≈ x·ReLU6(x+3)·(1/6)`. Q포맷 WIDTH=16/FRACT_BITS=8(L1-4). → DSP 없는 나눗셈 회피 **[확실]**.
  - ※ PyTorch 모델은 `nn.SiLU()`(=x·sigmoid(x))를 사용(MVT.py L26,33). 즉 **RTL은 hard-swish 근사로 SiLU를 대체** — 정확도 trade-off **[확실, 근사]**.
- **swisharray.sv**: scalar swish를 N(기본4)병렬 인스턴스화 + 입력/출력 레지스터 + valid 파이프(L1-70). 단, 본 모듈은 post_processing_pipeline에 연결되지 않음(3.5 참조).

---

### 3.9 AGU (RTL/AGU/) — 주소 생성

ST 교육과정 산출물(헤더 L1-9). 2서브모듈 구조(AGU.sv L62-157): `ConvTileIndicesGenerator`(타일 인덱스 iterator, 핸드셰이크/all_tiles_done) + `ConvOffsetsAGU`(타일 내 주소 스트림).

**ConvOffsetsAGU.sv** — 4모드 통합 주소 함수:
- op_mode: REGULAR(00)/POINTWISE(01)/DEPTHWISE(10)/MATRIX_MUL(11)(L58-61).
- 파생치(L100-122): `out_H/out_W = ((act + 2·pad - K_eff)/stride)+1`, `K_block = ker_H·ker_W`.
- **레이아웃 HWC**: `input_offset(h,w,c) = baseA + (h·act_W + w)·act_CIN + c`(L143-150). → on-the-fly im2col 주소 생성 **[확실]**.
- A 주소(compute_input_addr, L168-236): MATMUL은 `baseA+i·K+k`; DWConv/Conv/PW는 출력좌표→입력좌표 역산 후 패딩 경계 체크(out-of-bound시 NULL_ADDR). PW는 kh=kw=0.
- B 주소(compute_kernel_addr, L240-287): MATMUL `baseB+k·N+j`; DWConv는 **대각 읽기(diagonal reading)** — 가상 대각행렬에서 채널별 블록만 유효, 나머지 NULL(L259-277). 이는 DW conv를 시스톨릭 matmul로 매핑하기 위한 sparsity 트릭 **[확실]**.
- C 주소(L291-304): 전 모드 `baseC+i·N+j`.
- FSM IDLE→GEN_A→GEN_B→GEN_C→TILE_DONE, `read_req`로 게이팅·stall(L342-453). 출력은 1사이클 레지스터(valid_out_reg).

---

### 3.10 메모리 서브시스템 (RTL/Memory Subsystem/memory_subsystem.sv)

- **16뱅크 SRAM**(파라미터 NUM_BANKS=16, BANK_WIDTH=32, 뱅크당 4×8bit), 4버퍼: ActBufA/ActBufB/WgtBuf/PSumBuf(L1-7, L73-76). 뱅크당 32bit×4 → 한 사이클 64×8bit 공급.
- **DMA write**: `dma_waddr[3:0]`=뱅크선택, `[19:4]`=뱅크내 워드(인터리빙)(L88-104).
- **AGU read(SA 타입별 활성 뱅크 수)**: sa_type 00→16뱅크(64엘리먼트), 01→8뱅크(32), 10→4뱅크(16)(L116-149). → LEGO 형상에 맞춰 메모리 대역폭 가변 **[확실]**. 비활성 뱅크는 0 출력.
- **폭 변환**: 16×32bit → 64×8bit 언팩(L156-167). 1사이클 read latency를 valid 파이프로 정합(L176-186).
- **핑퐁**: `ping_pong_sel ? ActBufB : ActBufA`(L130).
- **PSum 누산 RMW**: `accum_mode`면 `PSumBuf[..]<=PSumBuf[..]+psum_wdata[k]`, 아니면 fresh write; `clear_psum`이면 전체 0클리어(L205-230). 16워드 단위 bank/addr 매핑(`(addr+k)%16`, `/16`).
- DMA read(writeback)는 ActA/ActB/PSum에서 32bit 공급(L235-252).

> **[확인 불가/한계]** `clear_psum`이 전 PSumBuf(16뱅크×4096워드)를 1사이클에 0클리어하는 기술(L207-213)은 합성 시 거대 reset fanout — 실제 BRAM에 매핑 불가, 추가 FSM 필요. 학생 프로젝트 단순화 **[확실, 합성성 우려]**.

---

### 3.11 DMA Wrapper (RTL/DMA/dma_wrapper.sv)

- Xilinx AXI DMA(MM2S/S2MM, 64bit AXI) 인터페이스(헤더 L1-12). FSM: IDLE→READ_AR→READ_DATA→/WRITE_AW→WRITE_DATA→WRITE_RESP→DONE/ERROR(L99-208).
- 버스트: INCR, arsize/awsize=3'b011(64bit=8B), BURST_LEN=16beat(=128B)(L304-321). length≥burst*8이면 arlen=15, 아니면 length/8-1(L237-255).
- **64↔32 변환**: read 시 64bit AXI 1beat을 2×32bit로 분할 write(word_count로 lower/upper 선택, L330-341); write 시 32bit mem을 64bit로 `{mem_rdata,mem_rdata}` 복제(L323) — ※ 상위/하위 동일 복제이므로 실제 32bit 데이터만 의미 **[확실, 단순화]**.

---

### 3.12 Global Controller (RTL/Control/global_controller.sv) — 메인 FSM

19상태 FSM(L109-129): IDLE → FETCH_DESCRIPTOR → LOAD_WEIGHTS → WAIT_WEIGHT_DMA → WEIGHT_LOAD_SA → CLEAR_PSUM → LOAD_ACT_PING → WAIT_ACT_PING_DMA → AGU_SETUP_PING → COMPUTE_PING → (LOAD_ACT_PONG → … → COMPUTE_PONG ↔ 핑퐁) → APPLY_POST_PROCESS → WRITEBACK → WAIT_WRITEBACK → LAYER_DONE → (다음 descriptor or IDLE) / ERROR_STATE.

- **디스크립터 구동**: IDLE에서 `start && desc_valid`로 진입(L167-171). FETCH에서 descriptor 래치, 가중치 미적재면 LOAD_WEIGHTS(L173-180).
- **가중치 로드**: DMA로 WgtBuf 적재(L346-351) → WEIGHT_LOAD_SA에서 32사이클 SA PE 로딩(L194-198, weight_load_cycles≥32).
- **핑퐁 오케스트레이션**: COMPUTE_PING 중 다음 타일을 ActBufB로 DMA(LOAD_ACT_PONG) → COMPUTE_PONG, 교대(L225-267). `mem_ping_pong_sel=ping_active`, `mem_accum_mode=!is_first_tile`(첫 타일 후 누산)(L325-326).
- **후처리 인에이블**: APPLY_POST_PROCESS에서 descriptor 플래그로 `pp_bn/swish/ln_enable` 구동(L402-406).
- **상태/IRQ**: `busy=(state!=IDLE)`, `done=irq=(state==LAYER_DONE)`(L332-335).

> **[확실, 미완 다수]**:
> - `total_tiles<=1` 하드코딩(L448, 주석 "Will be computed") → 멀티타일 루프 실질 미동작.
> - AGU 설정 전부 상수(`op_mode=OP_REGULAR_CONV`, ker=3×3, out_chs=16, pad=1, TM/TN/TK=16, baseB=0, baseC=0x10000)(L505-522, 주석 "TODO derive from descriptor") → **레이어별 동적 구성 미구현**. 즉 컨트롤러는 단일 3×3 conv 시나리오만 사실상 구동.
> - `sa_type=SA_TYPE_16X64` 고정(L525).

---

### 3.13 Top (RTL/mobilevit_accelerator_top.sv)

- AXI4-Lite Slave(레지스터맵: 0x00 CONTROL/0x04 STATUS/0x10~0x2C DESC_DATA[8]/0x30 DESC_PUSH/0x34 TILE_CNT/0x38 CYCLE_CNT, L206-271) + AXI4 Master(DMA).
- 디스크립터 포매팅(L283-294): reg_desc_data 8워드를 256bit descriptor_t로 패킹.
- 인스턴스: global_controller, dma_wrapper, AGU, memory_subsystem, sa_compute_unit, post_processing_pipeline(L304-587).
- **MVP 단순화**(L525-537): 메모리는 16×32bit psum 제공하나 SA는 64개 기대 → 첫 16개만 사용, 16~63은 0패딩. SA 출력 64개 중 첫 16개만 후처리로 전달(L563).
- BN 파라미터 상수 결선(mean=0/var=1/gamma=1/beta=0)으로 후처리 항등(L582-587).

---

## 4. 데이터플로우 (MobileViT 추론 의도)

IMPLEMENTATION_GUIDE.md L115-176, COMPLETE_DATA_FLOW_GUIDE.md L54-71 기준(이상적·미검증):

1. **CPU 설정**: DRAM에 입력영상(256×256×3 @0x8000_0000)·가중치·BN파라미터 적재 → 디스크립터(256bit) 작성 → DESC_PUSH → CONTROL.start.
2. **가중치 로드**: 컨트롤러→DMA→WgtBuf → SA PE에 32사이클 적재.
3. **활성화 로드(핑퐁)**: 첫 타일 ActBufA, 동시에 PSumBuf clear.
4. **연산(타일0)**: AGU가 A/B/C 주소 스트림 생성(HWC im2col) → memory_subsystem이 64×8bit act/wgt 공급 → LEGO SA가 `Out = Act × Weight` → psum을 PSumBuf write(누산 모드는 RMW).
5. **다음 타일 오버랩**: SA가 타일0 연산 중 DMA가 타일1을 ActBufB 적재 → 교대.
6. **C_in 누산**: c_in>16이면 부분합을 PSumBuf에 누산(sa_compute_unit + memory_subsystem RMW).
7. **후처리**: PSum → BN(`y=γx+β`) → Swish(hard-swish) → LayerNorm(SUM→MEAN→VAR→1/√→norm). ※ **현재 pipeline은 bypass — 개별 모듈은 standalone만 검증** **[확실]**.
8. **Writeback**: WRITEBACK 플래그면 DMA로 PSumBuf→DRAM, IRQ 발생.

**MobileViT 매핑(설계 의도)**: Conv stem/MV2(PW expand→DWConv→PW project, SiLU)는 op_mode REGULAR/POINTWISE/DEPTHWISE + BN + Swish로, MobileViT 블록의 self-attention은 Q/K/V matmul(MATMUL) + transpose_en(Kᵀ) + LayerNorm으로 매핑(layer_type_t L64-77, PE transpose L79-83). 타일링은 AGU의 TM/TN/TK + ConvTileIndicesGenerator로 처리하도록 의도.

---

## 5. HW/SW 매핑 (Python 골든모델 ↔ RTL)

**LayerNorm — golden_model.py(float) ↔ layer_normalization_approximation.py(fixed) ↔ RTL**:
| 단계 | golden_model.py | approximation.py | RTL |
|------|-----------------|------------------|-----|
| 합 | `np.sum`(L11) | `np.sum`(L11) | ELEMENTS_SUM(adder tree) |
| 평균 | `//embed_dim`(L12) | `>>5`(L13) | MEAN `>>>log2(N)` (MEAN.sv L12) |
| 차/제곱 | `x-mean`,`s**2`(L16-17) | 동일(L16-21) | array_multiplier_subtractor |
| 분산 | `sum_2//N`(L23) | `sum_2>>5`(L22) | VARI `>>>log2(N)` (VARI.sv L10) |
| 1/√ | `1/np.sqrt`(L29-30) | `k=bit_length-17`, LUT `y0`, `*0x2D41`(L31-57) | root_1(k추출)/root_2(LUT)/root_final(√2보정) |
| 정규화 | `sub*std_inv`(L35) | `>>28`(홀)/`>>14`(짝)(L65-72) | NORMALIZATION_OUT(L24-37) |

→ approximation.py는 **RTL을 비트정확하게 미러링**(`k=v.bit_length()-17`이 root_1의 MSB탐색, `0x2D41`이 root_final 상수, Q24.8 시프트 패리티가 NORMALIZATION_OUT과 일치)(layer_normalization_approximation.py L31-72 ↔ root_1.sv/root_final.sv/NORMALIZATION_OUT.sv). **[확실, 정확 매핑]**

**고정소수점 포맷 요약**:
- SA: INT8 입력 × INT8 가중치 → INT32 psum(common_pkg L14-16).
- BatchNorm: Q8.8(중간 Q16.16)(Batch_Normalization.sv L49-62).
- Swish: Q8.8(WIDTH16/FRACT8)(swish.sv L1-4).
- LayerNorm: 입력 Q24.8(32bit), std_dev_inv Q4.28(홀)/Q18.14(짝)(approximation.py L8, root_final.sv L12).
- → **모듈별 Q포맷이 제각각**(INT8 vs Q8.8 vs Q24.8). 통합 시 스케일 정합이 필요하나 미구현 **[확실, 한계]**.

**MobileViT 모델**: PyTorch(MVT.py: MV2/LocalRepresentation/TransformerEncoder, SiLU 사용) + no-API numpy 모델(V2/V3) + 사전학습 가중치 매핑 스크립트. RTL과의 end-to-end 비트정확 매핑은 LayerNorm에만 존재하고 전체 망 수준 골든은 부재 **[확인 불가]**.

---

## 6. 빌드·실행

- **시뮬레이션**: ModelSim / Vivado Simulator / Verilator(README.md L59). 테스트벤치 존재: Swish, BatchNorm, LayerNorm(layer_norm_tb1/2), Lego SA(Lego_SA_tb, SA_NxN_top_tb, PE_tb, TRSDL/TRSRL_tb), Integration(SA_Batch_tb), top(mobilevit_accel_tb.sv). AGU.sv는 `$dumpfile("agu_sim.vcd")` 내장(L159-162) → iverilog/Verilator 파형 의도.
- **합성**: Xilinx Vivado, 타깃 Zynq UltraScale+(README.md L58-61). Vivado 프로젝트는 미작성(IMPLEMENTATION_GUIDE.md L429 "Vivado Project TODO").
- **소프트웨어 흐름**: CPU가 AXI Slave로 디스크립터/CONTROL 작성, STATUS 폴링 또는 IRQ(IMPLEMENTATION_GUIDE.md L232-290 의사코드).
- **Python**: 골든모델은 `python golden_model.py` 단독 실행 가능(`__main__` 포함, golden_model.py L40-48; approximation.py L83-89).

---

## 7. 의존성

- **HW**: SystemVerilog IEEE 1800-2017, Xilinx AXI DMA IP(axi_dma v7.1, Direct Register Mode — dma_wrapper.sv L7-8), Vivado(합성), ModelSim/Verilator/iverilog(시뮬).
- **SW**: Python + numpy(golden/approximation), PyTorch + einops(MVT.py L1-13), TensorFlow/h5(Apple 사전학습 가중치 변환 — 추정, *.h5 제외 대상).
- **외부 IP 의존**: DMA가 Xilinx AXI DMA IP 전제 → 합성 시 Vivado IP 카탈로그 필요 **[확실]**.

---

## 8. 강점·한계

**강점**:
- **재구성형(LEGO) SA**: 4×16×16 타일을 16×64/32×32/64×16로 런타임 재구성 — 레이어별 행렬 형상(MViT의 다양한 M/K/N)에 dataflow를 맞춰 PE 활용률 향상 의도(Lego_SA.sv L19-58). 메모리 뱅크 수도 형상에 연동(memory_subsystem.sv L116-127).
- **단일 공유 CU로 위상 정합 해결**: 타일별 FSM의 start-pulse 오프셋 문제를 명시적으로 인지·제거(Lego_CU.sv L9-16) — 잘 정리된 설계 결정.
- **전치(Kᵀ) 하드웨어 지원**: PE 가중치 로드 방향 전환만으로 attention QKᵀ 대응(PE.sv L79-83, 107-108).
- **AGU 일반성**: REGULAR/PW/DW/MATMUL 4모드 + HWC im2col + 대각 DW 매핑을 단일 모듈에 통합(ConvOffsetsAGU.sv).
- **LayerNorm rsqrt가 DSP-free**: MSB탐색+256LUT+√2상수+시프트로 1/√ 구현, Python 비트정확 골든 보유.
- **Swish/BN/MEAN/VARI의 나눗셈 회피**: 시프트-합 1/6, 산술우시프트 ÷N 등 FPGA 친화적.

**한계(미완·근사)**:
- **통합 미완**: `sa_compute_unit`이 존재하지 않는 모듈명/포트(`Lego_Systolic_Array`/`TYPE_Lego`)로 인스턴스 → top elaborate 불가(3.3). post_processing_pipeline 전 스테이지 bypass(3.5). global_controller `total_tiles<=1`·AGU 상수 하드코딩(3.12). → **현재 코드 그대로는 end-to-end MobileViT 추론 불가** **[확실]**.
- **하드코딩**: BN γ/β 320채널 ROM 입력(특정 모델 의존), 컨트롤러의 ker3×3/out16/pad1 고정.
- **근사 정확도**: hard-swish가 SiLU 대체, rsqrt가 단발 LUT(Newton-Raphson 미적용 — 패키지 SQRT_ITERATIONS=4와 불일치).
- **합성성 우려**: `clear_psum` 1사이클 전체 0클리어(memory_subsystem.sv L207-213), DMA 64bit 복제(L323).
- **Q포맷 비통일**: INT8/Q8.8/Q24.8 혼재로 모듈 간 스케일 정합 미해결.
- **성능 수치 전부 미검증**: 400MHz/100GOPS/<5W는 hand-calc(README.md L45, COMPLETE_DATA_FLOW_GUIDE.md L6-14).
- **NUM_BANKS 불일치**: 패키지 4 vs memory_subsystem 16.

---

## 9. 우리 프로젝트 시사점 (HG-PIPE 계열 고처리량 ViT/Transformer FPGA + XR 시선추적 — 추정)

> 우리 프로젝트 성격은 외부 단서 기반 **추정**이며, 본 repo의 어떤 부분이 직접 재사용 가능한지를 코드 근거로 정리.

1. **재구성형 systolic의 재사용성**: LEGO 4타일 재구성(16×64/32×32/64×16)은 ViT의 가변 행렬 형상(attention의 head_dim·seq_len, MLP의 hidden_dim)에 PE를 맞추는 데 유효한 패턴. 단, **단일 공유 CU 방식(Lego_CU.sv L9-16)을 채택해 위상 오프셋을 피한 점**이 핵심 교훈. HG-PIPE류의 **완전 파이프라인/스트리밍** 목표와는 결이 다름(여기는 weight-stationary + LOAD/FEED/DRAIN 단발 배치) — 우리가 고처리량을 노린다면 weight 재로딩 오버헤드(매 matmul마다 N사이클 LOAD_W)가 병목이 될 수 있으므로, weight 더블버퍼링/스트리밍 가중치 구조로 개선 필요.
2. **LayerNorm/Swish 데이터패스**: rsqrt(MSB탐색+LUT+√2보정)와 hard-swish(시프트-합 1/6)는 **DSP-free·저면적**이라 XR용 저전력 엣지 추론에 직접 이식 가치 높음. 특히 LayerNorm은 **Python 비트정확 골든(approximation.py)이 동봉**돼 검증 비용 절감. 다만 단발 LUT 정밀도가 ViT 정확도에 충분한지 검증 필요(우리 정확도 예산에 맞춰 NR 1회 추가 검토).
3. **AGU의 im2col on-the-fly 주소생성**: HWC 레이아웃 + 4모드 통합 + 대각 DW 매핑은 conv 위주 backbone(MobileViT stem)에 유용. XR 시선추적의 입력 전처리(작은 해상도 conv)에 적용 가능.
4. **디스크립터 구동 + AXI 인터페이스**: SoC 통합(ARM/RISC-V + AXI4-Lite/Master) 골격은 그대로 참고 가능. 단 본 repo의 컨트롤러는 멀티레이어 자동 시퀀싱이 미완이라, 우리는 레이어 시퀀서/디스크립터 컴파일러(Python tiler, IMPLEMENTATION_GUIDE.md L428 TODO)를 추가 구현해야 함.
5. **반면교사**: 통합 단계(모듈명/포트 정합, 파이프 연결, Q포맷 통일, 멀티타일 루프)가 가장 취약 — 우리 프로젝트에서는 **블록 단위 검증된 IP를 top에 연결할 때 elaborate/시뮬 회귀를 CI로 강제**하는 것이 중요(이 repo가 그 부재로 통합 불가 상태).

---

## 10. 근거/한계 표기 요약

| 항목 | 판정 | 근거 |
|------|------|------|
| INT8 MAC / INT32 psum | 확실 | accelerator_common_pkg.sv L14-16 |
| weight-stationary + transpose(Kᵀ) | 확실 | PE.sv L79-83, 107-108 |
| LEGO 3형상 재구성 | 확실 | Lego_SA.sv L19-58, L217-297 |
| 단일 공유 CU 설계의도 | 확실 | Lego_CU.sv L9-16 |
| LayerNorm rsqrt = MSB+LUT+√2 | 확실 | root_1/root_2/root_final.sv |
| Python approx ↔ RTL 비트정확 | 확실 | layer_normalization_approximation.py L31-72 |
| hard-swish가 SiLU 대체(근사) | 확실 | swish.sv L25-49 ↔ MVT.py L26 |
| post_processing 전부 bypass | 확실 | post_processing_pipeline.sv L62-169 |
| sa_compute_unit 인스턴스 이름/포트 불일치 | 확실 | sa_compute_unit.sv L51-65 vs Lego_SA.sv L74-97 |
| global_controller total_tiles=1·AGU 상수 하드코딩 | 확실 | global_controller.sv L448, L505-525 |
| NUM_BANKS 패키지(4) vs mem(16) 불일치 | 확실 | common_pkg L24 vs memory_subsystem.sv L1 |
| SQRT_ITERATIONS=4(NR) vs 실제 LUT단발 | 확실 | norm_pkg L14 vs standard_deviation_inv.sv |
| 400MHz/100GOPS/<5W | 추정(미검증) | README.md L45, DATA_FLOW_GUIDE L6-14 |
| AGU 출처 = ST 교육과정(Mahmoud Abdo) | 확실 | AGU.sv L1-9 |
| BN 320채널 = XXS 대응 | 추정 | Batch_Normalization.sv L11-12, L30-42 |
| 전체 망 end-to-end 골든 | 확인 불가 | 해당 소스 부재 |
| 우리 프로젝트 성격(HG-PIPE+XR) | 추정 | 외부 단서(과업 지시) |

---
*분석 종료. 본 repo는 "잘 설계된 블록 IP들(LEGO SA, LayerNorm, BN, Swish, AGU) + 미완의 통합층(top/controller/post-pipe)" 구조이며, 블록 단위 재사용 가치는 높으나 그대로는 MobileViT end-to-end 추론이 불가능한 상태.*
