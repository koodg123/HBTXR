# Diff-DiT 코드베이스 정밀 분석

> 대상: `REF/Transformer-Accel/Diff-DiT`
> 논문: Tang et al., *"Diff-DiT: Temporal Differential Accelerator for Low-bit Diffusion Transformers on FPGA"*, ICCAD 2025 (README.md:62-69)
> 근거 표기 규칙: **확인** = 실제 코드 라인 인용(`파일:라인`), **추정** = 코드/문맥 기반 합리적 추론, **확인 불가** = 코드만으로 단정 불가
> 분석 방식: 자체 소스 6개 파일(TOP.cpp 2479L, config.h, TOP.h, common.h, AMA_rtl.cpp, Adder_rtl.cpp, test.cpp) 전량 Read 기반

---

## 1. 개요

Diff-DiT는 **Diffusion Transformer(DiT)의 FPGA 가속기**를 **Vitis HLS C++(High-Level Synthesis)** 로 기술한 알고리즘-하드웨어 협동설계(algorithm-hardware co-design) 프로젝트다. 순수 SW 모델/논문 PyTorch 구현이 아니라, **HBM 기반 FPGA 가속기 커널(HLS C 합성 대상)** 이다 (확인: `TOP.cpp:2422-2462` `#pragma HLS INTERFACE m_axi ... bundle=HBM_PORT_*`, `#pragma HLS DATAFLOW`).

핵심 아이디어는 논문 제목대로 **시간차분(Temporal Differential) 연산 + 저비트(Low-bit) 양자화**다 (확인: README.md:1-3). Diffusion 추론은 여러 timestep을 반복하는데, 인접 timestep의 activation이 유사하다는 점을 이용한다. 이전 timestep의 activation(`ACT_PRE_*`)을 저장해 두고, 현재 timestep과의 **차분(delta)** 만 계산·양자화하여 비트폭을 줄인다 (확인: `TOP.cpp:459` `quantize_fp16_to_int3(out_val - act_pre_val, ...)`).

비트폭 설계가 이를 명확히 보여준다 (확인: config.h:16-22):
- `BIT_ACT_FP = 8` : 일반(full) 경로 activation = **INT8**
- `BIT_ACT_DIFF = 3` : 차분(diff) 경로 activation = **INT3** (3비트)
- `BIT_WEIGHT = 4` : weight = **INT4**
- `BIT_FP = 16` / `BIT_CLASS = 16` : FP16(half) 중간연산 및 class condition

즉 **두 가지 동작 경로**가 한 데이터패스에 공존한다: ① **일반 경로(INT8 act × INT4 weight)**, ② **차분 경로(INT3 diff-act × INT4 weight)**. 이 둘을 13개의 `mode`(0~12)로 전환한다 (확인: README.md:43-58, `ap_uint<4> mode` 전 함수에 전파).

DiT 블록 한 개는 README.md:43-58 모드 표에 따라:
- `mode 0~6` = 일반(full) 경로: HC(class condition)+LN+QKV-Proj, Proj1, GELU MLP, Residual, QK·SoftMax, S·V attention
- `mode 7~12` = 차분(diff) 경로: 위와 동일하나 Diff-Proj / Diff-GELU / Diff-SoftMax / Diff-SV (INT3 차분 연산)

**아키텍처 핵심**: 32개 systolic array tile(`NUM_SA_TILE=32`, 각 16×16 PE)로 구성된 output-stationary(OS) MAC 어레이 + DSP-packing(1 DSP에 2~6 MAC) + SiLU/LayerNorm/SoftMax/GELU를 PWL(piecewise-linear)로 근사하는 SFU(Special Function Unit) + 차분 SoftMax/GELU를 위한 mask 저장/재로드 메커니즘.

---

## 2. 디렉토리 구조

### 2-1. 자체 소스 (분석 대상, 전량 Read 완료)

```
Diff-DiT/
├── README.md                  코드 구조·모드 표·인용정보 (확인: 70L)
├── LICENSE
└── src/
    ├── config.h    (183L)     비트폭·PE 크기·DiT 텐서 shape 매크로 (확인)
    ├── common.h    (6L)       ap_int/hls_math/hls_stream include만 (확인)
    ├── TOP.h       (52L)      TOP 커널 함수 시그니처 선언 (확인)
    ├── TOP.cpp     (2479L)    ★ 핵심: 전체 가속기 데이터플로우 (확인)
    ├── AMA_rtl.cpp (60L)      Add-Multiply-Add DSP wrapper의 C 모델 (확인)
    ├── Adder_rtl.cpp (58L)    분할 가산기(packed adder)의 C 모델 (확인)
    ├── test.cpp    (482L)     C-sim 테스트벤치(main): 모드별 shape 계산·파일 I/O (확인)
    └── test.h      (1L 내용)  (실제로는 비어있고 test.cpp 내부에 read/write 템플릿 존재) (추정)
```

> 주의: README.md:37-38은 `test.cpp`/`test.h`를 "Test implementation/header"로 명시하나, 실제 테스트 I/O 템플릿(`read_input_from_file`/`write_output_to_file`)과 `main()`은 모두 `test.cpp`에 있다 (확인: test.cpp:10-481). `test.h` 파일은 본 분석 시점 내용이 사실상 비어 있음 (확인 불가: 별도 헤더로서의 역할 미확인).

### 2-2. 제외(생성물/vendor) — 이름만 언급

- `.git/` : git 메타데이터(pack, hooks, refs) — 분석 제외
- `assets/challenges_contributions.jpg`, `assets/speedup.jpg` : 논문 figure(생성물 이미지) — 분석 제외
- 빌드 스크립트(.tcl/.cfg/.sh/.py 등) **없음** (확인: Glob `**/*.{tcl,cfg,mk,sh,py,json,ini,xdc}` → "No files found")
- 체크포인트/데이터셋/.dat 입력 파일 **저장소에 없음** (test.cpp가 `activation_256bit_1M.dat` 등 외부 파일을 기대하나 repo에는 미포함) (확인: test.cpp:373-375)

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

전체 호출 계층은 다음과 같다 (확인: TOP.cpp의 호출관계):

```
TOP (HW 최상위, m_axi/HBM 인터페이스)              TOP.cpp:2375-2478
 ├─ HalfCondition  (class/timestep condition)      TOP.cpp:331-346
 │   ├─ get_scale_param                            TOP.cpp:306-328
 │   └─ condition_dataflow                         TOP.cpp:250-303
 │       ├─ silu                                   TOP.cpp:79-102
 │       └─ PE_wrapper_no_dsp_packing (systolic)   TOP.cpp:105-248
 └─ matmul → matmul_kernel (#pragma HLS DATAFLOW)  TOP.cpp:2314-2372 / 2216-2311
     ├─ load_sfu_mask                              TOP.cpp:714-747
     ├─ load_quant_factor (×2: quant/dequant)      TOP.cpp:1055-1188
     ├─ input_from_axi                             TOP.cpp:1805-1859
     │   ├─ load_activation → input_sfu(LN)        TOP.cpp:1511-1626 / 1029-1052
     │   │   └─ layernorm_dataflow                 TOP.cpp:470-512
     │   │       ├─ ln_statistics                  TOP.cpp:349-392
     │   │       ├─ ln_norm                        TOP.cpp:395-421
     │   │       └─ ln_scale_shift (adaLN)         TOP.cpp:424-467
     │   ├─ load_weight                            TOP.cpp:1628-1661
     │   └─ load_attn_q/k/s/v                      TOP.cpp:1664-1802
     ├─ PE_input  (tile별 데이터 재배열)            TOP.cpp:1862-1957
     ├─ PE_tile → PE_array<16,16> ×32 (OS MAC)     TOP.cpp:1960-1981 / 1358-1508
     │   ├─ AMA_rtl (DSP pack MAC)                 AMA_rtl.cpp:36-59
     │   └─ Adder_rtl (분할 가산)                  Adder_rtl.cpp:17-57
     ├─ PE_output (결과 unpacking)                 TOP.cpp:1984-2054
     ├─ sfu (#pragma HLS DATAFLOW)                 TOP.cpp:2057-2166
     │   ├─ GELU / diff_GELU                       TOP.cpp:802-891
     │   ├─ SoftMax / diff_softmax                 TOP.cpp:515-711
     │   ├─ Scale / diff_Scale (residual+scale)    TOP.cpp:894-990
     │   └─ Skip_LN / Straight                     TOP.cpp:993-1026
     ├─ store_sfu_mask                             TOP.cpp:750-799
     └─ output_to_axi                              TOP.cpp:2169-2213
```

### 3-1. 양자화/역양자화 프리미티브 — `TOP.cpp:14-76`

가속기의 데이터 표현 변환 기본 함수들이다.
- `apuint16_to_half` / `half_to_apuint16` (TOP.cpp:14-24): FP16(`half`)과 raw 16비트 비트패턴(`ap_uint<16>`)을 `memcpy`로 상호 변환. activation/scale은 HBM에서 INT 컨테이너로 운반되므로 매 연산에서 비트 재해석이 필요.
- `clamp_to_int3` (TOP.cpp:27-35): [-4, 3]로 포화 → **3비트 signed** 범위 (확인).
- `clamp_to_int8` (TOP.cpp:38-46): [-128, 127] 포화.
- `quantize_fp16_to_int3` / `_int8` (TOP.cpp:49-69): `(input - zero_point) / scale` 후 clamp. **affine 양자화**.
- `dequantize_int3_to_fp16` / `_int8` (TOP.cpp:57-76): `int * scale + zero_point`.

근거: 차분 경로는 INT3, 일반 경로는 INT8을 일관되게 사용. 양자화 파라미터(scale/zero)는 HBM의 `QUANT_FACTOR`/`DEQUANT_FACTOR`에서 BRAM으로 로드(load_quant_factor) 후 packed형태로 SFU 전반에 전달.

### 3-2. HalfCondition — class/timestep condition + SiLU + class systolic — `TOP.cpp:79-346`

DiT의 **adaLN-Zero conditioning**(class label·timestep embedding으로부터 LayerNorm의 scale γ / shift β / residual gate α를 생성)을 담당하는 유닛으로 추정된다 (추정; 근거는 아래).

- `silu(in, out)` (TOP.cpp:79-102): SiLU 활성 `x / (1 + exp(-x))`를 FP16으로 계산 (확인: TOP.cpp:94 `tmp_out_val[k] = tmp_in_val[k] / (1+(half)hls::expf((float)(-tmp_in_val[k])))`). `CLASS_SIMD=2` 폭으로 병렬. SiLU는 DiT timestep MLP에서 표준적으로 사용 → conditioning 경로임을 뒷받침.
- `PE_wrapper_no_dsp_packing` (TOP.cpp:105-248): class condition 전용 **output-stationary systolic array** (`CLASS_SIMD=2` × `CLASS_PE=32`). FP16 곱셈누산을 직접 수행(`PE_P_reg[i][j] = dsp_port_C + dsp_port_A * dsp_port_B`, TOP.cpp:216). DSP packing 미적용(이름 그대로) — class condition은 데이터량이 작아 packing 불필요 (추정). A는 우측 전달, W는 하향 전달(OS 전형), `out_flag`/`rep4add` 타이밍으로 anti-diagonal 시점에 결과 추출 (확인: TOP.cpp:219-230).
- `condition_dataflow` (TOP.cpp:250-303): `#pragma HLS DATAFLOW`로 feed_a → silu → feed_w → PE_wrapper → store_to_bram을 파이프라인. 출력은 전역 `bram_class_out[2][...][...]` (TOP.cpp:4)에 ping-pong 저장 (확인: TOP.cpp:299).
- `get_scale_param` (TOP.cpp:306-328): `bram_class_out`에서 **β(shift), γ(scale), α(gate)** 를 분리 추출. mode 0/2/7/9에서 β·γ, mode 1/3/8/10에서 α (확인: TOP.cpp:316-325). → 이것이 adaLN의 6개 modulation 파라미터 중 일부에 해당함을 강하게 시사 (추정).
- `HalfCondition` (TOP.cpp:331-346): ping-pong 버퍼 선택 후 condition을 계산하거나 이전 timestep 값을 재사용. 전역 ping-pong은 timestep 간 condition 재사용(차분의 기반)과 연결 (추정).

### 3-3. LayerNorm 데이터플로우 (adaLN) — `TOP.cpp:349-512`

DiT의 핵심 정규화. 3-stage DATAFLOW로 분리 (확인: TOP.cpp:507-509).
- `ln_statistics` (TOP.cpp:349-392): INT8/INT3을 dequant 후 채널(`IMG_SHAPE_3_2=1152`) 방향 1-pass로 평균/분산 누적, `half_sqrt`로 std 계산 (확인: TOP.cpp:384). pack 폭은 일반=`HBM_DATA_PACK_A=128`, 차분=`HBM_DATA_PACK_A_DIFF=256` 결정(TOP.cpp:497).
- `ln_norm` (TOP.cpp:395-421): `(x - mean) / std` 정규화, std=0 보호(0.001) (확인: TOP.cpp:412-415).
- `ln_scale_shift` (TOP.cpp:424-467): **adaLN 핵심**. `out = x * γ + β` (γ=`bram_gamma`, β=`bram_beta`, TOP.cpp:450). 그 후 mode 0~5는 INT8 재양자화, mode 7~12(차분)는 `(out_val - act_pre_val)`의 **차분을 INT3로 양자화**하며 동시에 `act_pre`를 갱신 출력(TOP.cpp:457-464). → adaLN과 차분-양자화가 같은 stage에서 융합됨 (확인).

이것이 Diff-DiT의 알고리즘 본질: **LayerNorm 직후 출력의 timestep 간 차분만 INT3로 보내** PE array 트래픽·연산비트를 1/2.67로 압축 (8bit→3bit) (확인적 추론: BIT_ACT_FP=8 vs BIT_ACT_DIFF=3).

### 3-4. SoftMax / Diff-SoftMax — `TOP.cpp:515-711`

Attention의 softmax. 일반/차분 두 구현.
- `sfm_max` (TOP.cpp:515-549): row 최대값 (수치안정화) — dequant 후 max 추적.
- `sfm_exp` (TOP.cpp:552-600): `exp(x - max)`를 `hls::expf`로 계산 + **PWL slope index mask** 산출. `tmp(=x-max)`를 8구간으로 양자화한 3비트 `mask`를 `fifo_sfm_mask`에 저장(TOP.cpp:577-584), exp 합도 `fifo_sfm_sum`에 저장(TOP.cpp:598). 이 mask/sum이 **다음 timestep의 Diff-SoftMax 재사용 데이터**.
- `sfm_norm` (TOP.cpp:603-631): `exp/sum` 후 INT8 재양자화.
- `SoftMax` (TOP.cpp:634-660): 위 3개를 row(`S_SHAPE_4_3=256`) 단위 DATAFLOW로 묶음.
- `diff_softmax` (TOP.cpp:663-711): **차분 버전**. 저장된 `sfm_slope[8]`(TOP.cpp:7) PWL 계수와 이전 timestep mask/sum을 읽어, INT3 delta_x에 대해 `exp ≈ x * slope`로 **선형 근사**하고 `exp/sum`을 다시 INT3로 출력 (확인: TOP.cpp:692-704). 즉 차분 경로에서는 exp를 직접 계산하지 않고 **이전 timestep의 구간 정보(mask)를 재사용한 선형근사**로 대체 → 연산 절감.

### 3-5. SFU mask 저장/재로드 — `TOP.cpp:714-799`

차분 연산의 timestep 간 상태전달 메커니즘 (Diff-DiT 특유).
- `load_sfu_mask` (TOP.cpp:714-747): mode 9(diff-GELU)는 gelu mask, mode 11(diff-softmax)은 softmax sum+mask를 HBM `MASK_0..3`에서 읽어 FIFO로 (확인).
- `store_sfu_mask` (TOP.cpp:750-799): mode 2(GELU)는 gelu mask, mode 4(SoftMax)는 softmax sum/mask를 HBM에 기록 → **다음 timestep diff 단계가 재사용** (확인). HBM 4채널에 비트 슬라이싱하여 packed 저장.

근거: 일반 경로(mode 2/4)에서 **PWL 구간 인덱스(mask)와 정규화 분모(sum)를 한 번 계산해 저장**하고, 차분 경로(mode 9/11)에서 **그대로 재로드**하여 비선형 함수 재계산을 회피하는 것이 핵심 절감 트릭 (확인적 추론).

### 3-6. GELU / Diff-GELU — `TOP.cpp:802-891`

MLP의 활성화.
- `GELU` (TOP.cpp:802-852): 16구간 PWL 근사. `index = 2.5*x + 8` (구간 결정), `gelu_slope[16]`·`gelu_shift[16]`(TOP.cpp:8-11) 적용 `out = slope*x + shift` (확인: TOP.cpp:837-842). 동시에 구간 index를 4비트 mask로 저장(TOP.cpp:849).
- `diff_GELU` (TOP.cpp:855-891): 저장된 mask로 동일 slope를 선택, INT3 delta에 `out = slope * x`만 적용(shift 생략 — 차분이므로 상수항 소거) (확인: TOP.cpp:881-886). 차분의 선형성을 이용한 정확한 단순화.

### 3-7. Scale / Diff-Scale (residual + adaLN gate) — `TOP.cpp:894-990`

- `Scale` (TOP.cpp:894-942): `out = x * α + residual` (α=`bram_alpha`=adaLN gate, residual=`fifo_act_res`, mode 3에서만 잔차 적용) (확인: TOP.cpp:932). adaLN-Zero의 residual-gating에 해당 (추정).
- `diff_Scale` (TOP.cpp:945-990): 차분 버전, INT3, `out = x*1 + act_res` (mode 10에서 잔차). gate scale이 1로 단순화된 점은 차분 경로 특성 (확인: TOP.cpp:981).

### 3-8. DSP packing MAC — `AMA_rtl.cpp`, `TOP.cpp:1191-1341`

가속기 면적효율의 핵심. Xilinx DSP48의 `(A+D)*B+C` 구조를 활용해 **1개 DSP에 여러 저비트 곱셈을 packing**한다.
- `DSP_Wrapper` (TOP.cpp:1191-1200): `(A+D)*B+C` 그대로 (확인). FPGA DSP48E2의 pre-adder + MAC.
- `pack2_corr` (TOP.cpp:1230-1234) / `pack6_corr` (TOP.cpp:1203-1227): packing 시 발생하는 **부호 보정항(correction term)** 을 C포트로 주입. INT8×INT4 일반경로는 **2-packing**(한 DSP에 2 MAC), INT3×INT4 차분경로는 **6-packing**(한 DSP에 6 MAC) (확인적 추론: 함수명 pack2/pack6 + bit slicing 구조).
- `AMA_rtl` (AMA_rtl.cpp:36-59): 위를 통합한 **Add-Multiply-Add**의 합성용 C 모델. `is_diff`로 일반(8b)/차분(3b) 분기. 일반: A=8b, D=8b<<18(상위 8b를 18비트 위로 올려 packing), B=8b. 차분: A=3b, D=3b<<24, B=4b×3 슬롯(`4*5`비트, 각 슬롯 사이 zero-padding)으로 **6개 INT3×INT4를 한 27×24 곱에 인코딩** (확인: AMA_rtl.cpp:42-53). 이는 `TOP.cpp:1237-1282`의 `AMA_Wrapper`와 동일 로직의 별도 합성단위 (추정: RTL black-box 대체용 C 모델).
- `Adder_rtl` (Adder_rtl.cpp:17-57) / `TOP.cpp:1300-1341 Adder`: 48비트 누산기를 **6개의 8비트 full-adder(`FA_8bits_Wrapper`)** 로 분할 구현 (확인: Adder_rtl.cpp:29-53). 일반 경로는 캐리체인을 2-슬롯 구조(case 0/3에서 carry 차단)로, 차분 경로는 6슬롯 모두 carry 차단(packed 독립 누산) (확인: Adder_rtl.cpp:31-51). `#pragma HLS BIND_OP impl=fabric`(TOP.cpp:1294)로 LUT 가산기 강제 — DSP는 곱셈에 전용.

핵심: 이 packing이 INT3 차분경로의 **6배 MAC 밀도**를 만들어 speedup의 주 원천 (확인적 추론).

### 3-9. 메인 systolic array — `PE_array<NUM_PE_ROW,NUM_PE_COL>` `TOP.cpp:1358-1508`

- 템플릿화된 **16×16 output-stationary systolic array** (`SIZE_SA_PE=16`, TOP.cpp:1972). A는 행 방향 우측 전달(TOP.cpp:1479-1481), B(weight)는 열 방향 하향 전달(TOP.cpp:1483-1485). 각 PE는 `AMA_rtl` + `Adder_rtl`로 누산(TOP.cpp:1475-1476).
- `out_flag`/`rep4add`/`rep4out` 카운터로 anti-diagonal 결과추출 타이밍 제어(TOP.cpp:1413-1497). `rep_count = NumLines + ROW + COL - 2`로 systolic fill/drain 고려 (확인: TOP.cpp:1401).
- `is_diff` 플래그(mode 7~12)로 동일 하드웨어가 INT8/INT3 모두 처리 (확인: TOP.cpp:1474). → **하나의 reconfigurable PE가 일반/차분 양쪽 지원**.
- `PE_tile` (TOP.cpp:1960-1981): `PE_array<16,16>`를 `NUM_SA_TILE=32`개(=`SA_TILE_ROW 4 × SA_TILE_COL 8`, config.h:26-28) UNROLL 인스턴스화 → 총 32×256 = **8192 PE** (추정 산술).

### 3-10. 데이터 재배열 PE_input / PE_output — `TOP.cpp:1862-2054`

- `PE_input` (TOP.cpp:1862-1957): HBM에서 온 packed A/B를 mode별로 32 tile에 분배. mode에 따라 tile 좌표(b,i,j) 매핑이 다름(일반 4×8, 차분 8×4, SV 12 등) (확인: TOP.cpp:1882-1900). INT3는 부호확장하여 16비트 컨테이너에 packing(TOP.cpp:1930-1947).
- `PE_output` (TOP.cpp:1984-2054): 48비트 PE 결과를 mode별로 unpacking. 차분(7~10,12)은 3슬롯, diff-QK(11)는 2슬롯, 일반은 1슬롯으로 분해하여 SFU로 (확인: TOP.cpp:2013-2049). → packing된 6/2-MAC 결과를 다시 분리하는 역과정.

### 3-11. 최상위 데이터플로우 — `matmul_kernel` `TOP.cpp:2216-2311`, `TOP` `TOP.cpp:2375-2478`

- `matmul_kernel` (TOP.cpp:2216-2311): `#pragma HLS DATAFLOW`로 load_sfu_mask → load_quant_factor(×2) → input_from_axi → PE_input → PE_tile → PE_output → sfu → store_sfu_mask → output_to_axi를 **단일 task-level 파이프라인**으로 연결 (확인: TOP.cpp:2294-2309).
- `TOP` (TOP.cpp:2375-2478): HW 커널 진입점. **34개 HBM m_axi 포트**(MAT_A 8 + MAT_A_PRE 8 + MAT_B 4 + out 8 + MASK 4 + QUANT/DEQUANT 2 + class 3 등) 각각 별도 bundle (확인: TOP.cpp:2422-2460). `HalfCondition` + `matmul`을 DATAFLOW로 실행(TOP.cpp:2468-2469).

근거: `MAT_A_PRE_*`(이전 timestep activation) 전용 8개 포트가 별도로 존재 → 차분 연산을 위한 **이전 timestep 데이터 상주** 설계가 인터페이스 레벨에서 확정됨 (확인: TOP.cpp:2384-2391, 2430-2437).

### 3-12. 테스트벤치 — `test.cpp:1-481`

- `read_input_from_file`/`write_output_to_file` 템플릿(test.cpp:10-73): hex `.dat` 파일 I/O. 비트폭별 우측정렬 truncation.
- `main` (test.cpp:77-481): `mode`(상단 하드코딩, 기본 0) 별로 13가지 텐서 shape(matA/B/O, NWnum 등)를 계산(test.cpp:90-321)하고, HBM 채널 인터리빙(4채널 또는 8채널)으로 activation/weight 분배(test.cpp:385-436) 후 `TOP` 호출(test.cpp:439-446). → **C-simulation 검증용 단일 모드 실행 하네스** (확인). 모드별로 수동으로 `mode` 상수를 바꿔 13패스를 각각 검증하는 구조 (추정).

---

## 4. 데이터 플로우 (end-to-end, DiT 블록 1개 기준)

```
[HBM]                                                   [HBM]
ACT(INT8/INT3) ─┐                                  ┌─> out (INT8) 또는
ACT_PRE       ─┤   load_activation/attn_q/k/s/v    │   bram_act_quant_out(INT3, 차분)
WEIGHT(INT4)  ─┤        │                          │
QUANT/DEQUANT ─┤        ▼                          │
MASK(sum/idx) ─┘   input_sfu = LayerNorm(adaLN)    │
class_cond    ─> HalfCondition→SiLU→PE(FP16)→γ/β/α │
                        │ (INT8 또는 차분INT3 양자화) │
                        ▼                          │
                  PE_input (32 tile 분배, packing) │
                        ▼                          │
                  PE_tile = 32 × (16×16 OS MAC)    │
                   각 PE: AMA_rtl(2/6-pack DSP)    │
                          + Adder_rtl(LUT 6×8b)    │
                        ▼                          │
                  PE_output (48b → 슬롯 unpack)    │
                        ▼                          │
                  sfu: GELU/SoftMax/Scale          │
                       또는 diff_* (mask/sum 재사용)│
                        │                          │
                  store_sfu_mask ──(다음 timestep)─┤
                        ▼                          │
                  output_to_axi ───────────────────┘
```

흐름 근거: `matmul_kernel`의 DATAFLOW 연결(TOP.cpp:2294-2309), adaLN 융합 양자화(TOP.cpp:457-464), mask 저장/재로드(TOP.cpp:750-799 / 714-747).

**차분 루프(시간차분 핵심)**: timestep t에서 일반경로(mode 2/4)가 비선형함수 mask/sum을 HBM에 저장 → timestep t+1에서 차분경로(mode 9/11)가 ACT_PRE(이전 act)와 저장 mask를 재사용해 INT3 delta만 연산 → 비선형 재계산·고비트 트래픽 회피.

---

## 5. HW/SW 매핑

| 계층 | 구현 | 근거 |
|------|------|------|
| **HW 합성 대상(커널)** | `TOP` HLS C → Vitis HLS 합성 → FPGA 비트스트림 | TOP.cpp:2422-2462 `#pragma HLS INTERFACE m_axi`, `#pragma HLS DATAFLOW` |
| **연산 코어** | 16×16 OS systolic ×32 tile = MAC array | TOP.cpp:1358-1508, 1960-1981 |
| **MAC 프리미티브** | DSP48 `(A+D)*B+C` 2/6-packing | AMA_rtl.cpp:36-59, TOP.cpp:1191-1282 |
| **누산기** | LUT-fabric 6×8b 분할 가산 | Adder_rtl.cpp:17-57, TOP.cpp:1294 |
| **비선형(SFU)** | LayerNorm/SoftMax/GELU/SiLU PWL 근사 | TOP.cpp:7-11, 349-891 |
| **외부 메모리** | HBM 다채널(34 m_axi bundle) | TOP.cpp:2422-2460 |
| **온칩 메모리** | BRAM(quant factor, class_out, FIFO depth) | TOP.cpp:492-495, 2285-2292 |
| **데이터 타입** | INT8 act / INT3 diff-act / INT4 weight / FP16 중간 | config.h:16-22 |
| **제어** | 13개 `mode`로 DiT 연산단계 시퀀싱 | README.md:43-58, mode 전파 |
| **SW(호스트) 드라이버** | **저장소에 없음** (test.cpp는 C-sim 벤치만) | Glob 결과 + test.cpp:77 |

요약: 이 repo는 **HW 커널(HLS) + C-sim 테스트벤치**까지만 포함. 실제 FPGA on-board용 host(XRT/OpenCL) 코드, Vitis 빌드 스크립트, .dat 데이터는 **미포함** (확인: Glob "No files found", test.cpp가 외부 .dat 의존).

---

## 6. 빌드·실행

- **빌드 시스템 파일 없음** (Makefile/CMake/tcl/cfg 부재, 확인: Glob). README에도 빌드 절차 미기재 (확인: README.md 전체).
- C-simulation: `test.cpp`의 `main`을 Vitis HLS의 `ap_int`/`hls_*` 헤더와 함께 컴파일하여 실행하는 구조 (추정). 외부 입력 `activation_256bit_1M.dat`, `weight_256bit_1M.dat`, `act_diff_768bit_1M.dat` 필요(test.cpp:373-375) → repo에는 미포함이므로 **그대로는 실행 불가** (확인).
- 합성: `TOP`(TOP.h:4 / TOP.cpp:2375)을 top function으로 지정해 Vitis HLS csynth 수행하는 것이 의도 (추정; tcl 부재로 확인 불가).
- 모드 전환: `test.cpp:78`의 `ap_uint<4> mode = 0;`을 수동 변경하여 13단계 각각 검증 (확인).

---

## 7. 의존성

- **Vitis HLS 라이브러리** (확인: config.h:5-8): `ap_int.h`, `hls_half.h`, `hls_math.h`, `hls_stream.h`. `AP_INT_MAX_W 4096`로 초광폭 비트벡터 사용(config.h:4).
- `hls::expf`(SoftMax/SiLU, TOP.cpp:94,586), `half_sqrt`(LayerNorm, TOP.cpp:384) 등 HLS math 내장.
- 표준 C++ `<fstream>/<sstream>/<iomanip>`(테스트 I/O, test.cpp:1-4).
- 외부 ML 프레임워크(PyTorch 등) 의존 **없음** — 순수 HLS C++ (확인). 양자화/mask는 모두 오프라인 생성된 .dat로 주입되는 형태 (추정).

---

## 8. 강점 · 한계

### 강점 (코드 근거)
- **이중 정밀도 단일 데이터패스**: 동일 PE/DSP/SFU가 INT8(일반)·INT3(차분) 모두 처리, `mode`/`is_diff`로 전환 (확인: TOP.cpp:1474, 1245-1270). 하드웨어 재사용률 극대화.
- **DSP 6-packing**: INT3×INT4를 1 DSP에 6 MAC 인코딩(AMA_rtl.cpp:48-53) → 면적·전력 효율의 직접 원천.
- **시간차분 + 비선형 mask 재사용**: SoftMax/GELU의 PWL 구간 인덱스·분모를 저장·재로드(TOP.cpp:714-799)하여 비선형 재계산 회피. Diffusion의 timestep 유사성을 정확히 활용.
- **adaLN 융합**: LayerNorm scale/shift와 차분-양자화를 한 stage에서 결합(TOP.cpp:457-464) → 중간 트래픽 절감.
- **task-level DATAFLOW** 전면 적용(TOP.cpp:2264, 2462)으로 단계 간 파이프라이닝.

### 한계 (코드 근거)
- **호스트/빌드/데이터 부재**: on-board host, Vitis tcl, .dat 입력 모두 미포함 → 재현·온보드 검증 불가 (확인: Glob, test.cpp:373).
- **모드 수동전환**: 13 mode를 `main`에서 상수로 하나씩 바꿔 검증 — 통합 시퀀서/오케스트레이션 코드 부재 (확인: test.cpp:78). 전체 DiT 블록 자동 실행 흐름은 코드상 미구현 (추정).
- **하드코딩된 shape**: config.h의 텐서 크기(1152, 256, 16 head 등)가 특정 DiT(추정 DiT-XL/2 계열)에 고정 (확인: config.h:120-177). 일반화 어려움.
- **수치 디버그 흔적**: `// for debug` 분기(TOP.cpp:385,587), magic number(0.001 std 보호) 다수 → 연구 프로토타입 성격 (확인).
- **3비트 클램프 비대칭** [-4,3] 등 저비트 포화로 인한 정확도 손실은 코드만으로 정량 평가 불가 (확인 불가).
- **mask/sum HBM 왕복**: 차분의 대가로 비선형 상태를 매 timestep HBM에 저장/로드 — 메모리 대역폭 비용 존재 (확인: store/load_sfu_mask).

---

## 9. 우리 프로젝트(PRJXR-HBTXR) 시사점

우리 프로젝트는 **고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적**으로 추정된다. Diff-DiT는 ViT가 아니라 DiT(생성형)지만, **저비트 Transformer를 FPGA에서 돌리는 공통 토대**가 많아 직접 차용 가치가 있다.

1. **DSP packing 기법(2/6-MAC)** — `AMA_rtl`/`pack6_corr`(AMA_rtl.cpp:10-59)는 INT3×INT4 6-packing의 완성된 레퍼런스. XR용 저비트 ViT에서 DSP-제약 FPGA의 처리량을 끌어올리는 데 그대로 응용 가능 (추정). 단, HG-PIPE는 binarization/저비트를 다르게 다룰 수 있어 packing correction 항만 발췌 권장.
2. **OS systolic array 템플릿** — `PE_array<ROW,COL>`(TOP.cpp:1358)는 mode 가변·diff/full 겸용의 깔끔한 OS 어레이. 우리 ViT의 QKV/Proj/MLP matmul 코어로 재사용 가능한 형태 (추정).
3. **adaLN 융합 양자화 + LayerNorm DATAFLOW**(TOP.cpp:349-512) — ViT의 LayerNorm 가속에 그대로 이식 가능. ln_statistics/norm/scale_shift 3-stage 분리는 II=1 파이프라인 모범.
4. **PWL SFU(SoftMax/GELU/SiLU)** — `sfm_slope`/`gelu_slope`(TOP.cpp:7-11) 기반 구간근사는 XR 실시간성(저지연)에 부합. 시선추적의 attention/MLP 비선형부에 저비용 적용 가능 (추정).
5. **시간차분(temporal diff)의 XR 적용 가능성** — XR 시선추적은 **연속 프레임(video) 입력**이므로 Diffusion timestep ↔ video frame 간 유사성으로 치환 시, 프레임 간 차분(INT3)만 연산하는 동일 전략이 **시선추적 ViT의 전력/지연 절감**에 적용될 잠재성 큼 (추정; 검증 필요). mask/ACT_PRE 재사용 인터페이스(TOP.cpp:2384-2391)가 청사진.
6. **HBM 다채널 인터페이스 설계**(TOP.cpp:2422-2460) — 34 bundle 분리는 대역폭 확보 패턴 참고. 단 XR 엣지 FPGA가 HBM 없는 DDR/온칩 중심이면 그대로는 부적합, 채널수 축소 필요 (추정).
7. **차용 시 주의** — 호스트/빌드/데이터 부재로 **블랙박스 재현 불가**. 기법(packing, diff, PWL)은 알고리즘으로 추출해 우리 코드에 재구현하는 접근이 현실적 (확인적 판단).

---

## 부록: 근거 표기 요약

- **확인(라인 인용)**: 비트폭(config.h:16-22), HW 인터페이스(TOP.cpp:2422-2462), DSP packing(AMA_rtl.cpp:36-59), adaLN/차분양자화(TOP.cpp:424-467), mask 저장·재로드(TOP.cpp:714-799), systolic(TOP.cpp:1358-1508), 모드 표(README.md:43-58), 빌드/데이터 부재(Glob 결과·test.cpp:373).
- **추정**: HalfCondition=adaLN-Zero conditioning, 8192 PE 총수 산술, video-frame 차분 적용, 6-packing 의미, host 자동 오케스트레이션 부재.
- **확인 불가**: 정확도 손실 정량값, 합성 리소스/주파수(보고서 부재), test.h의 헤더 역할, Vitis 합성 절차(tcl 부재).
