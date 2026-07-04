# ViM-Q (FCCM 2026) 정밀 분석

> 분석 대상: `REF/Others/ViM-Q-FCCM-2026`
> 분석 방식: 실제 소스 Read 기반, 라인 근거 명시. third-party(`SW/mamba-1p1p1` 원본·`csrc/*.cu` CUDA 커널·HLS 생성 `*.v`/`*.dat`·`data.zip`)는 이름만 언급하고 분석 대상에서 제외.
> 표기 규칙: 코드 라인으로 확인된 사실은 근거 표기, 불명확한 부분은 "추정"/"확인 불가"로 구분.

---

## 1. 개요

- **목적**: Vision Mamba(ViM, SSM 기반 비전 백본)에 대한 **PTQ(Post-Training Quantization)** 와 이를 위한 **FPGA 가속기**를 한 저장소에서 end-to-end로 제공. SW측은 가중치 4bit(APoT) / 활성 8bit 양자화 + SmoothQuant 류 평활화 + HW 입력 export, HW측은 HLS C++ → SpinalHDL 통합 → Vivado 구현으로 이어지는 풀스택 가속기.
- **한줄요약**: "Vision Mamba를 **APoT(Additive Powers-of-Two) 4bit 가중치 + INT8 활성** 으로 양자화하고, SSM(selective scan)·causal conv1d·linear projection·patch embed를 **shift-add 기반 곱셈기 없는 HLS 커널**로 구현해 SpinalHDL로 데이지체인 연결한 FPGA 가속기".
- **원논문**: FCCM 2026 투고 추정. 저장소명 `ViM-Q-FCCM-2026`, 루트 README L1 "ViM-Q". (정확한 논문 제목/저자는 저장소에 미기재 — 확인 불가)
- **타깃 디바이스**: Xilinx, Vivado/Vitis HLS **2025.2** (`project_config.sh` L16 `XILINX_PATH="/opt/Xilinx/2025.2"`, README L17). 구체적 보드 파트번호는 본 분석에서 읽은 파일에는 미기재(`create_bd.tcl`/`template*.tcl`에 존재 가능 — 추정). 토큰화/AXI 폭 256bit(`common.h` L196).
- **타깃 모델**: ViM-Tiny/Small/Base. 기본 실험은 ViM-Tiny (`SW/run.sh` L40,45가 vim-t만 활성, vim-s/b는 주석). 차원 설정 `common.h` L43–48: `D_MODEL=192`, `D_INNER=384`, `D_STATE=16`, `DT_RANK=16`, `NUM_PATCHES=197`, `NUM_LAYERS=24`(합성 시) / 1(C-sim).

---

## 2. 디렉토리 구조 (자체 소스 / 제외 구분)

### 2.1 자체 핵심 소스 (분석 대상)

```
ViM-Q-FCCM-2026/
├── README.md                       # 전체 빠른 경로
├── project_config.sh               # 경로/툴체인 중앙 설정
│
├── SW/                             # [소프트웨어: PTQ + export]
│   ├── a_generate_act_scale.py     # 활성 스케일(act_scales) 캘리브레이션 수집
│   ├── b_smooth_model.py           # SmoothQuant류 평활화 (in_proj/conv/x_proj/dt_proj/out_proj)
│   ├── c_quantize_model.py         # nn.Linear/Conv1d → QuantLinear/QuantConv1d 치환
│   ├── e_export_model.py           # 양자화 가중치/중간텐서를 HW 입력 bin으로 export
│   ├── main.py / engine.py / datasets.py / losses.py / samplers.py / utils*.py  # DeiT 학습/평가 골격(원본 파생)
│   ├── models_mamba.py             # ViM 모델 정의(원본 ViM 파생 — 외부)
│   ├── run.sh, scripts/{vim-t,vim-s,vim-b}/*.sh
│   └── model/                      # ★ 자체 양자화 모듈 ★
│       ├── quant_linear.py         # QuantLinear (APoT 4bit W + INT8 A)
│       ├── quant_conv1d.py         # QuantConv1d (APoT 5bit W + INT8 A)
│       ├── quant_utils.py          # build_power_value (APoT 기저 생성), round_to_power_of_2
│       ├── mamba_simple_module.py  # 중간텐서 캡처용 MambaModule(export hook)
│       └── ops/{scan_ref.py, selective_scan_interface.py, triton/layernorm.py}
│
└── HW/                             # [하드웨어: HLS + SpinalHDL + Vivado]
    ├── src/                        # ★ HLS 커널 헤더 ★
    │   ├── common.h                # 모델/데이터타입/HW상수(INT4/INT8, AXI 256b)
    │   ├── embed.h                 # patch_embed (Conv 패치 임베딩)
    │   ├── norm_sum.h              # RMSNorm + residual add (4 모드)
    │   ├── linear_block.h          # APoT LUT 기반 INT4 linear (in/x/dt/out proj, head)
    │   ├── conv.h                  # causal conv1d (shift-add INT4)
    │   ├── ssm.h                   # selective scan (prescan+scan 융합 dataflow)
    │   ├── smooth.h                # smooth scale 곱(평활화 HW 적용)
    │   ├── patch_ops.h             # patch flip / CLS 토큰 로드
    │   ├── activation.h            # SiLU/Softplus (LUT 보간)
    │   ├── silu_table.h / softplus_table.h / max.h / utils.h
    ├── case/                       # 모듈별 HLS top 테스트 케이스 (.cpp)
    │   ├── EMBED/CONV/LINEAR_BLOCK/NORM_SUM/SSM/SMOOTH/PATCH_OPS.cpp
    ├── testbench/{data_loader.hpp, tb_utils.hpp}
    ├── SPINAL/src/main/scala/      # ★ SpinalHDL 통합 ★
    │   ├── ViM.scala               # 7개 모듈 데이지체인 통합 + AXI 노출
    │   ├── ViM_ACCELERATOR.scala / ViM_Full_Test.scala
    │   ├── {EMBED,CONV,LINEAR_BLOCK,NORM_SUM,SSM,SMOOTH,PATCH_OPS}.scala  # HLS Verilog BlackBox 래퍼 + 시뮬
    │   └── utils/{Manager, FixedPointTypes, BlackboxAxi(Lite), LatencyTracker, SimUtils}.scala
    ├── step1~5_*.py, run_vivado_*.py, *.tcl   # HLS/Vivado 자동화 플로우
    └── constants.py / hls_status_check.py / util_generate_ssm_data.py
```

### 2.2 제외 목록 (외부/생성물 — 이름만 언급)

- **`SW/mamba-1p1p1/`**: 외부 Mamba 레퍼런스(서브모듈 포크). `.gitmodules`/`AUTHORS`/`setup.py` 존재. `csrc/selective_scan/*.cu`(CUDA selective scan fwd/bwd 커널), `mamba_ssm/`(원본 mamba_simple.py 등). **단, 자체 추가분으로 보이는 `mamba_ssm/ops/selective_scan_interface_quant.py`, `mamba_ssm/modules/mamba_simple_quant.py`는 self-modified(이름에 `_quant`)** — 본 저장소의 양자화 포팅분으로 추정되나, SW 메인 플로우(`SW/model/`)와 중복 구현이며 export/PTQ 경로에서 직접 사용되는 것은 `SW/model/*`이므로 분석 비중은 `SW/model/`에 둠.
- **HLS 생성물**: `HW/SPINAL/src/main/verilog/*/all.v`, `*.dat`(ROM 초기화), `HW/data.zip`(pre-exported HW 입력 + 비트 데이터).
- **`.git/`, `.Xil`, `instances/`, `logs/`, `ip_repo/`, `vivado_reports/`** 등 빌드 산출물.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.A SW측 — 양자화 알고리즘

#### 3.A.1 `SW/model/quant_utils.py` — APoT 기저 생성 (`build_power_value`)

양자화 전체의 수학적 핵심. **APoT(Additive Powers-of-Two)** 양자화 격자를 생성한다.

- `round_to_power_of_2(x)` (L4–9): `2^round(log2(x))` — 활성 스케일을 2의 거듭제곱으로 반올림(시프트로 dequant 가능하게).
- `build_power_value(B, additive)` (L11–82): `B = weight_bits - 1` (부호 제외 크기 비트). 가법적 PoT 기저 `base_a/base_b/base_c`를 비트수에 따라 생성.
  - **B=4 (즉 weight_bits=5, conv용)**: L22–24 `base_a += 2^(-2i-1)`, `base_b += 2^(-2i-2)` (i=0..2). → 각 가중치 = `a + b` 두 PoT 합.
  - **B=3 (weight_bits=4, linear용 — 실제 호출은 `weight_bits=4`라 `B=3`)**: L29–35 혼합 분배.
  - L51–66: 모든 `(a,b,c)` 조합의 합 `values`와 **정수 인덱스 패킹** `int_weight = i | (j<<bits_a) | (k<<(bits_a+bits_b))` 를 B비트로 마스킹. → HW가 디코드할 수 있는 정수 코드워드.
  - L69–73: 중복 제거 후 정렬, `value_scale = 1/max(values)` 로 정규화.
  - 반환: `(values, value_scale, int_weights, base_a/b/c, bits_allocation)`.

이 `int_weights`가 그대로 HW의 nibble(4bit) 코드로 export되며, HW 디코더가 동일 시프트 합으로 복원한다(3.B 참조). 이것이 SW↔HW 정합성의 핵심 계약.

#### 3.A.2 `SW/model/quant_linear.py` — `QuantLinear`

`nn.Linear` 상속. 핵심 파라미터(L46–73): `weight_bits=4`(c_quantize에서 전달), `act_bits=8`, `per_block=True`, `block_size=32`, `act_per_token=True`, `power=True`, `additive=True`.

- **`QuantizeSTE.forward`** (L9–39, power 분기 L13–29): 가중치를 블록 스케일로 나눠 `[-1,1]` clamp(L14) → 부호 분리(L15) → `proj_set`(APoT 격자)과의 최소거리 인덱스 `argmin`(L17–18)로 격자에 투영 → `q_w = proj_set[idx]*sign`. `return_int_weights`면 `int_weights[idx]`에 부호비트(`1<<(weight_bits-1)`) OR(L24–27). backward는 STE(L42–44).
- **`get_weight_scales`** (L96–116): `per_block`이면 `[out, in/block, block]`로 reshape 후 블록별 absmax(L100–102). **power 모드에서는 absmax를 그대로 스케일로 사용**(qmax로 안 나눔, L104) — APoT는 [-1,1] 정규화 후 격자투영이므로.
- **`finalize_calibration`** (L130–171): 스케일 계산→`build_power_value`로 격자 생성(L144)→ 가중치 사전양자화(`weight_quant`, 정수코드 `weight_quant_pot`)→ `weight_dequant = weight_quant * scale`(블록 단위 repeat_interleave, L161–165). PTQ면 원본 `weight` 삭제(L169–170).
- **활성 양자화** (`quantize_activation_absmax` L172–186, `initialize_act_scales` L194–201): per-token absmax / `act_qmax(=127)`. `act_per_token=True`라 토큰(시퀀스 위치)별 스케일. `round_to_power_of_2`로 2^k 반올림(L197).
- **`smooth_x`** (L188–192): `smooth_scales`(b_smooth에서 등록)를 입력에 곱해 활성 분포 평활화.
- **`forward`** (L217–242): smooth → 활성양자화(`q_x`) → `F.linear(q_x, weight_dequant)` → `* act_scales` → bias. (실수 시뮬레이션상 dequant 곱이지만 정수경로는 `weight_quant_pot`+`act` 정수곱과 등가, HW가 이를 정수로 수행).

#### 3.A.3 `SW/model/quant_conv1d.py` — `QuantConv1d`

`nn.Conv1d` 상속. **conv는 `weight_bits=5`(APoT B=4), `act_bits=8`** (c_quantize L135–136). 구조는 QuantLinear와 평행:
- `from_float`(L95–122): 스케일=채널별 absmax(L90), 격자투영으로 `weight_quant`/`weight_quant_pot`/`weight_dequant`(채널스케일 view L119).
- `forward`(L165–195): 활성양자화 후 `weight_dequant * act_scales`를 곱해 `F.conv1d`. depthwise(groups) 처리 위해 `_out_channel_group_idx`로 그룹별 act_scale 매핑(L181–188). ViM의 conv1d는 depthwise causal conv.

#### 3.A.4 `SW/b_smooth_model.py` — SmoothQuant류 평활화

활성 outlier를 가중치로 이전하는 평활화. `scales = act^α / weight^(1-α)`, α=0.5(전형 SmoothQuant).
- `smooth_in_proj`(L32–53): RMSNorm 가중치를 `/scales`, in_proj 가중치를 `*scales`로 — norm-linear 수학적 등가 흡수(L50,53). **bias 없는 RMSNorm 전제**(assert L51).
- `smooth_both_conv`(L55–96): forward/backward conv1d 두 개의 스케일을 `max`/`mean`으로 병합(L85–89) 후 in_proj 일부 채널과 conv 가중치에 분배. ViM의 양방향(bidirectional) 구조 반영(`conv1d`/`conv1d_b`).
- `smooth_dt_x_proj`(L98–130): dt_proj/x_proj 쌍 평활화, x_proj에 `smooth_scales` 버퍼 등록(런타임 입력에 곱하기 위함).
- `smooth_fc`(L7–30): 일반 fc(out_proj)용. `smooth_scales`(역수)를 파라미터로 등록.
- `smooth_vim_model`(L132–177): 블록별로 위 함수들을 in_proj→conv→x/dt_proj(fwd+bwd)→out_proj 순서로 적용, head는 `norm_f`와 평활.

#### 3.A.5 `SW/c_quantize_model.py` — 모델 양자화 치환

- `quantize_vim_model`(L72–180): `models_mamba.Block`을 순회하며 mixer의 8개 레이어(`in_proj, conv1d, x_proj, dt_proj, conv1d_b, x_proj_b, dt_proj_b, out_proj` L87–96)를 `QuantLinear`/`QuantConv1d`로 치환. head도 `quantize_head`면 4bit 양자화(L152–171).
- `initialize_act_scales`(L29–70): 캘리브 샘플(`sample_size=4`)로 forward, `ActivationHook`로 각 양자화 레이어 입력 캡처 후 act_scale 초기화. QuantLinear는 smooth 적용 후 캘리브(L60).

#### 3.A.6 `SW/e_export_model.py` — HW 입력 export

- `reorder_weights_to_blocks`(L26–45): `(out,in)` 가중치를 `16×16` 타일로 재배열(`reshape→transpose(0,2,1,3)→reshape`). **이 16×16 타일 레이아웃이 HW `linear_block.h`의 `load_linear_weights`(3.B.3)와 정확히 일치** — SW/HW 가중치 패킹 계약.
- `save_intermediate_outputs`(L47–): hook으로 각 양자화 레이어 입출력 + **SSM 중간텐서**(delta, B, C, deltaA(lambda), deltaBu, xs, out_z, 양방향 `_b`)를 `ref_*_block/`에 float32 bin으로 저장(L76–120). 이것이 HW SSM 시뮬의 ground-truth(`SSM.scala` L197–223이 읽는 `ssm_float32/*.bin`).

#### 3.A.7 데이터타입 매핑 요약 (SW PTQ 설정)

| 텐서 | bits | 방식 | 근거 |
|---|---|---|---|
| linear 가중치 | 4bit | APoT, per-block(block=32) | c_quantize L105–117, quant_linear L75 |
| conv1d 가중치 | 5bit | APoT, per-channel | c_quantize L135–136 |
| 활성(linear) | 8bit | per-token absmax, 2^k 반올림 | quant_linear L194–201 |
| 활성(conv) | 8bit | per-channel/tensor | quant_conv1d L139–146 |

---

### 3.B HW측 — HLS 커널 (`HW/src/*.h`)

#### 3.B.0 공통 정의 `common.h`

- **데이터타입(L60–88)**: feature map `fm_t = ap_fixed<32,14>`(32bit, 정수 14bit), pixel `ap_fixed<32,10>`, patch/norm 가중치 `ap_fixed<16,1>`. **linear/conv 가중치 = `ap_uint<4>`(INT4 코드워드)** (L77, L84), conv 가중치는 부호(`bool`)+크기(`ap_uint<4>`) 분리(L83–84). 스케일류는 `ap_ufixed<32,6>`/`ap_ufixed<32,14>`.
- **양자화 상수(L196–199)**: `Q_MAX=127, Q_MIN=-128`(INT8 활성), `Q_MAX_FLOAT=1/127`.
- **블록/병렬 상수**: `AXI_XFER_BIT_WIDTH=256`(L196), `FEATURE_BLOCK_SIZE=8`(L202, fm 병렬도), `LINEAR_BLOCK_SIZE=16`, `CONV_BLOCK_SIZE=16`. `wt_wide_t = ap_uint<256>`(L206) — AXI 1워드. `MAGS_PER_WORD=64`(256/4, INT4 nibble 64개/워드, L207).
- **벡터 타입**: `fm_block_t = hls::vector<fm_t,8>`(L210) 가 데이터플로우의 기본 단위. patch별·feature블록별 다차원 typedef(L213–242).
- `FOR_EACH/FOR_BLOCK/FOR_OFFSET` 매크로(L92–135)로 라벨 자동생성 루프, `ceildiv/roundup/roundup_p2/bitcount`(L137–165) 컴파일타임 유틸. `fast_exp`용 `ap_fixed_relu/epsilon/min`(L167–183).

#### 3.B.1 `ssm.h` — Selective Scan (가장 핵심)

ViM의 SSM(상태공간) 연산을 dataflow 파이프라인으로 구현. 입력: `u, delta, z_silu, A, B, C, D`, 출력: scan 결과.

- **`fast_exp_approx`** (L12–26): `exp(x)≈(1+x/32)^32` 제곱연쇄(t2,t4,t8,t16,res)로 음수에서 안정적인 근사 exp. discretization의 `exp(delta*A)` 계산용.
- **`compute_state_update`** (L96–196): **prescan+scan 융합 단일 루프**(논문 셀링포인트로 추정). `pe_state_token[NUM_FEATURE_BLOCKS_SCAN][8]`을 상태로 유지(L109, dim=2 complete partition). UNIFIED_LOOP(L143) II=1:
  - prescan(L172–180): `delta_A = exp(delta*A)`(L177–178), `delta_Bu = (delta*u)*B`(L179).
  - scan(L184–192): `new_token = current*delta_A + delta_Bu`(L188) — 선형 재귀 상태 갱신, `FEATURE_BLOCK_SIZE=8` PE 병렬(unroll factor=4, L185). `dependence ... inter false`(L146)로 II=1 달성.
- **`state_to_xC_stream`** (L199–283): 상태 x와 C의 내적(`x·C`). 8-way 부분곱(L257–261) → **트리 리덕션**(s01..s47, L263–269)으로 타이밍 개선, state_dim 누적(L271).
- **`read_in_u_D_stream`** (L366–399): `u*D` (skip connection 항).
- **`compute_output_on_stream`** (L421–454): `out = (x_C + u*D) * z_silu`(L450) — SiLU 게이트 곱.
- **`compute_ssm_output_impl`** (L477–540): `#pragma HLS dataflow`(L492)로 8개 스테이지(read×5, state_update, xC, uD, output, write)를 스트림(`depth=2`)으로 연결. 깊은 버퍼링 없이 fine-grained 파이프라인.
- A/D는 `load_A/D_buffer`(L32–93)로 256bit 워드에서 32bit씩 언팩.

→ **A는 fixed-point로 저장(이미 `exp` 입력용 실수), B/C/u/delta/z_silu는 fm_t로 스트림**. 즉 SSM 내부는 양자화가 아니라 **fixed-point(ap_fixed<32,14>) 연산** — 양자화는 SSM 앞단의 projection/conv에 집중되고 SSM 자체는 고정소수점.

#### 3.B.2 `conv.h` — Causal Conv1d (shift-add INT4)

depthwise causal conv. **곱셈기 없이 시프트+덧셈으로 MAC**.

- 타입: 입력 활성 `conv_quant_t = ap_int<8>`(L13), 가중치 = 부호(bool)+크기(`ap_uint<4>`).
- `load_conv_weights_magnitudes/signs`(L38–106): 256bit 워드에서 4bit mag / 1bit sign 언팩, `[dim_block][dim_offset][kernel_offset]` 포맷.
- `read_in_stream`(L193–283): **2-pass** — pass1 absmax로 활성 스케일 계산(L223–246, `act_scale = max*Q_MAX_FLOAT`), pass2 스트리밍.
- `window_and_quantize_on_stream`(L285–369): 라인버퍼(`CONV_KERNEL_SIZE-1` 깊이)로 causal window 구성 + 활성 INT8 양자화(round, clamp 127/-128, L339–348).
- **`compute_conv_on_stream`** (L371–446): 핵심 MAC. 각 INT4 크기코드 `mag_val`을 **하위2bit/상위2bit로 분해**(L415–416)하여 각각 시프트량으로 디코드:
  - 하위2bit→`>>8 / <<7 / <<5 / <<3`(L421–426), 상위2bit→`>>8 / <<6 / <<4 / <<2`(L428–433).
  - `accumulator = shifted_a + shifted_b`(L435) → **APoT 두 PoT 합을 두 번의 시프트+1덧셈으로 실현**(SW build_power_value B=4의 `a+b`와 정확히 대응). 부호 적용(L436), `>>8` 정규화(L441).
- `dequantize_on_stream`(L448–497): `quant*weight_scale*act_scale + bias`(L488–489), 이후 **SiLU 적용**(L494).
- `compute_causal_conv_impl`(L543–579): `#pragma HLS dataflow`(L555)로 read→window/quant→MAC→dequant→write 5스테이지.

#### 3.B.3 `linear_block.h` — APoT LUT 기반 INT4 Linear (in/x/dt/out proj + head)

가장 정교한 커널. 16×16 가중치 타일, INT4 APoT, **LUT 기반 곱셈 제거**.

- `load_linear_weights`(L52–102): export의 16×16 재배열(3.A.6)과 일치하게 256bit×4워드=256개 4bit 가중치를 `[block][out16][in16]`로 언팩(L83–99). nibble 위치 `flat>>6, flat&63, <<2`(L93–95).
- **`decode_weights_on_stream`** (L366–444): 입력 타일별로 **APoT LUT 사전계산**. 활성 `x`에 대해 `s4=x<<4, s5,s6,s7`(L420–423) 미리 시프트 후 8개 엔트리 LUT 구성(L425–432): `{0, s7, s6, s4, s5, s7+s5, s6+s5, s4+s5}` — 즉 **가중치 크기코드(3bit)가 LUT 인덱스**가 되어, 곱셈을 "활성 시프트의 사전합 테이블 조회"로 대체. (SW build_power_value B=3 격자의 8개 값과 대응).
- **`compute_mac_on_stream`** (L446–551): 가중치 4bit를 부호(bit3)+크기(bit0–2)로 분해(L505–506), `lut.entries[ic][mag_idx]`로 곱 결과 조회(L508), 부호 적용(L509). 16입력 **adder tree 16→8→4→2→1**(L514–533)로 누적. 16출력채널 병렬(`unroll`, L492). 32엘리먼트(2타일)마다 부분합 전송(L536–549).
- `scale_and_accumulate_on_stream`(L553–600): 부분합 × per-block weight_scale, `>>8` 정규화 누적(L592–593).
- `dequantize_on_stream_linear`(L602–663): `*act_scale + bias`, **flags로 SiLU/Softplus 선택 적용**(L658–659). `FLAG_BIAS/SILU/SOFTPLUS`(L44–46)로 한 커널이 in_proj(SiLU 없음)·dt_proj(Softplus)·x_proj 등 모두 처리.
- `compute_linear_block_impl`(L719–760): dataflow 7스테이지(read→quant→decode_weights(LUT)→MAC→scale→dequant→write).

→ **conv는 "가중치 분해 시프트", linear은 "활성 시프트 LUT 조회"** — 둘 다 APoT의 곱셈없는 실현이나 데이터플로우 방향이 반대(가중치 정적 vs 활성당 LUT 재생성). 설계상 흥미로운 비대칭.

#### 3.B.4 `norm_sum.h` — RMSNorm + Residual Add (4모드)

- 4모드(L15–20): `NORM_SUM_BOTH=norm(a+b)`, `NORM_ONLY`, `ADD_ONLY`, `DIV2_ONLY`. 한 커널이 ViM의 "residual add → RMSNorm" 패턴 전부 커버.
- `compute_norm_on_stream`(L99–159): patch별 2-pass — sum of squares 누적(`*inv_model_dim`, L134) → `rms_inv = 1/sqrt(sum_sq)`(L140, `hls::sqrt`) → `val*rms_inv*weight`(L154). 가중치 `ap_fixed<16,1>`.
- dataflow 4스테이지(L221–232).

#### 3.B.5 `smooth.h` — 평활화 스케일 HW 적용

- `compute_smooth_on_stream`(L86–136): SW b_smooth에서 등록된 `smooth_scales`(역수)를 입력에 elementwise 곱(L129). 즉 **SW가 가중치에 흡수 못한 입력측 평활화(x_proj/out_proj용 smooth_scales)를 HW 런타임에 적용**. `wt_linear_ss_t = ap_ufixed<32,6>`.
- linear_block과 동일한 16블록 reorder 패턴 공유.

#### 3.B.6 `embed.h` — Patch Embedding (Conv)

- `patch_embed_impl`(L17–220): 16×16 패치를 D_MODEL 채널로 투영하는 conv. 가중치/bias/CLS/pos_embed를 BRAM에 캐시(L35–43, cyclic partition factor=32로 채널병렬). CLS 토큰은 `numPatches/2` 위치에 pos_embed만 더해 삽입(L100–119) — ViM의 mid-CLS 토큰. 이미지 패치는 `EMBED_PAR_CHANNELS=32` 채널 병렬 conv(L159–204, bias+pos_embed 초기화 후 3채널×256픽셀 MAC).
- **embed 가중치는 INT4 아님 — `ap_fixed<16,1>`** (L70). 즉 패치 임베딩은 양자화 대상 아님(첫 레이어 정밀도 보존, 일반적 양자화 관행).

#### 3.B.7 `activation.h` — SiLU/Softplus (LUT 보간)

- `silu`(L11–30), `softplus`(L46–65): ReLU + (테이블 보간된 델타). `x_abs`로 ROM 테이블(`SILU_DELTA_TABLE`) 인덱싱 + 선형보간(L24–29). 테이블 범위 밖이면 ReLU(L19). vector 버전(L33–43, L68–78). ROM은 BRAM 바인딩.

---

### 3.C HW측 — SpinalHDL 통합 (`HW/SPINAL/`)

#### 3.C.1 모듈 래퍼 패턴 (`SSM.scala` 대표)

- `SSM_Blackbox`(L11–69): HLS 생성 `src/main/verilog/SSM/all.v`를 BlackBox로 래핑. 8개 AXI4 master(in_u/delta/z_silu/B/C, weights_A/D, out_r) + 2개 AXI-Lite slave(control, control_r). `mapClockDomain`으로 `ap_clk/ap_rst_n` 연결(L67).
- `SSM`(Component, L71–136): BlackBox를 표준 Axi4/AxiLite4로 변환(`connect2std`), **`DaisyChain[ManagerSignals]`** + `Manager`로 제어 시퀀싱(L95–96). `Axi4SpecRenamer`로 표준 신호명 부여.
- `simulate_ssm`(App, L138–427): Verilator 시뮬. `ssm_float32/*.bin`을 읽어(L197–211) `FixedPointTypes.floatToFixed`로 고정소수점 변환(L213–219, fm_t/scan_t), AxiMemorySim에 적재. AXI-Lite로 src/dst 주소·`SCAN_DIM/INNER_DIM/NUM_PATCHES` 설정(L332–359), ap_start 트리거(L376–380), 완료 폴링 후 출력 읽어 `compare_arrays`로 MSE/MAE 검증(L422).

#### 3.C.2 `ViM.scala` — 전체 파이프라인 통합

- `ViM`(Component, L10–): 7개 모듈 1회 인스턴스화(L12–18). **DaisyChain 체인(L95–101)**: `embed → norm_sum → linear → conv → smooth → ssm`. 모든 모듈의 AXI4 메모리 IF를 개별 master로 노출(L26–62, "공유 DRAM에 AXI Interconnect로 연결" 주석 L104), AXI-Lite control을 개별 slave로 노출(L66–90, "CPU에 SmartConnect+주소디코딩" L139).
- 즉 **HW는 모듈별 separate accelerator를 데이지체인 제어로 순차 실행**, 데이터는 공유 DRAM 경유(스트리밍 fused가 아닌 DRAM-중심 layer-by-layer). `ViM_ACCELERATOR.scala`가 Vivado용 top, `ViM_Full_Test.scala`가 단일레이어 end-to-end 시뮬(README L261–281, 224×224 cycle-accurate latency).

#### 3.C.3 `FixedPointTypes.scala`

HLS `common.h` 타입을 Scala로 미러(L25–49): `fm_t=FixedPointType(32,14,signed)` 등 1:1 대응. `floatToFixed`(L57–73, **truncation** L58)/`fixedToFloat`(L75–93)로 시뮬 입출력 변환. SW(float bin) ↔ HW(고정소수점) 경계의 정합성 보증 코드.

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 SW 파이프라인 (PTQ + Export)
```
[원본 ViM 체크포인트(tiny/small/base)]
  → a_generate_act_scale.py : 캘리브 데이터로 레이어별 활성 absmax 수집 → act_scales/*.pt
  → b_smooth_model.py       : SmoothQuant(α=0.5), norm·in_proj·conv·x/dt_proj·out_proj 평활
  → c_quantize_model.py     : Linear→QuantLinear(W4 APoT/A8), Conv1d→QuantConv1d(W5 APoT/A8) 치환
  → (ptq-vim-*.sh)          : ImageNet val로 Acc@1/Acc@5 측정
  → e_export_model.py       : 16×16 타일 재배열 가중치 + SSM 중간텐서 → output/{bin,image,ref}_float32_block/
[output/* → HW/data/]
```

### 4.2 HW 파이프라인 (HLS → RTL → 비트스트림)
```
step1_hls_sim (C-sim) → step2_hls_syn (C합성, .v 생성) → step3_hls_cosim (RTL 검증)
  → step4_print_resource → step5_spinal_flow (Verilog→SPINAL/verilog/all.v, .dat 복사)
  → sbt compile → simulate_vim_full (Verilator, latency_report.csv)
  → generate_vim_accelerator (ViM_ACCELERATOR.v)
  → run_vivado_flow.py (IP 패키징+BD) → run_vivado_implementation.py (impl+bitstream+reports)
```

### 4.3 메모리 계층 / 병렬화 / 데이터타입
- **메모리계층**: 외부 DRAM ↔ AXI4(256bit) ↔ 모듈 BRAM 캐시(가중치/스케일 static 버퍼) ↔ HLS 스트림(`depth=2` FIFO) ↔ 레지스터(array_partition complete). DRAM-중심 layer-by-layer(모듈간 공유 DRAM, `ViM.scala` L104).
- **병렬화**: `FEATURE_BLOCK_SIZE=8`(fm 벡터), `LINEAR/CONV_BLOCK_SIZE=16`(타일), `EMBED_PAR_CHANNELS=32`. 각 dataflow 스테이지 II=1 파이프라인. linear MAC은 16출력×16입력 unroll + adder tree.
- **데이터타입**: 가중치 **INT4(linear)/INT5(conv) APoT**, 활성 **INT8**, 내부 누적 **fixed-point ap_fixed<32,14>**(`fm_t`), 곱은 **시프트+덧셈(곱셈기 free)**. SSM 내부는 fixed-point(양자화 아님). 패치임베딩·RMSNorm 가중치는 `ap_fixed<16,1>`.
- **양자화 정합 계약**: SW `int_weights`(APoT 코드) ⇄ HW nibble 디코더, SW 16×16 reorder ⇄ HW `load_linear_weights`, SW act_scale(2^k 반올림) ⇄ HW `>>` 정규화.

---

## 5. HW/SW 매핑

| ViM 연산 | SW 양자화 모듈 | HW 커널 | 양자화 |
|---|---|---|---|
| patch embed (conv) | (비양자화, embed) | `embed.h` patch_embed_impl | fixed-point(16,1) |
| RMSNorm + residual | (smooth가 흡수) | `norm_sum.h` (4모드) | fixed-point |
| in_proj / x_proj / dt_proj / out_proj / head | `QuantLinear`(W4 APoT/A8) | `linear_block.h`(LUT MAC, flags) | INT4 W + INT8 A |
| smooth scale 적용 | `smooth_x`(런타임 입력 곱) | `smooth.h` | fixed-point scale |
| causal conv1d (fwd/bwd) | `QuantConv1d`(W5 APoT/A8) | `conv.h`(shift-add MAC) | INT5 W + INT8 A |
| selective scan (SSM) | export 중간텐서(delta/B/C/A/D) | `ssm.h`(prescan+scan fused) | fixed-point(32,14) |
| patch flip / CLS 로드 | (모델 구조) | `patch_ops.h` | fixed-point |
| 활성 SiLU/Softplus | (모델) | `activation.h`(LUT 보간) | fixed-point |
| 통합/제어 | — | `ViM.scala` DaisyChain + AXI | — |

**핵심**: SW가 양자화 격자(`build_power_value`)와 가중치 코드(`int_weights`)·레이아웃(16×16)·스케일을 결정하고 bin으로 export → HW 커널이 동일 규약으로 디코드·시프트연산. SSM은 양자화하지 않고 export된 중간텐서를 fixed-point로 처리.

---

## 6. 빌드·실행

- **사전설정**: `project_config.sh` 편집(XILINX_PATH, DATA/Checkpoints 경로). SW env(Python 3.10.13, PyTorch 2.1.1+cu118, GCC11), HW env(Java17, SBT 1.10, Verilator 5.004, Vitis HLS/Vivado 2025.2) — README L16–17.
- **SW**: `cd SW && bash run.sh` (ptq-vim-t.sh + export-model-vim-t.sh). 출력 `SW/output/{bin,image,ref}_float32_block/`.
- **데이터 이동**: `cp -r SW/output/* HW/data/` (또는 빠른 평가용 `unzip data.zip -d data`, 1_run_flow L77–90).
- **HW HLS**: `source setup_env.sh` 후 `python3 step1_hls_sim.py [MODULE]` → step2(syn) → step3(cosim) → step4(resource) → step5(spinal).
- **HW Spinal/Vivado**: `cd SPINAL; sbt compile; sbt "runMain simulate_vim_full"`(latency_report) → `sbt "runMain generate_vim_accelerator"` → `python3 run_vivado_flow.py` → `run_vivado_implementation.py`(`--reports-only` 옵션). 결과 `vivado_reports/post_impl/{utilization,power}_*.rpt`.

---

## 7. 의존성

- **SW**: PyTorch 2.1.1(cu118), timm(`create_model`), einops, numpy, huggingface(ImageNet-1K), DeiT 학습 골격(engine/datasets/samplers/losses는 DeiT 파생). `SW/mamba-1p1p1`(외부 mamba_ssm, causal_conv1d). `vim_requirements.txt` 존재.
- **HW**: Vitis HLS(ap_int/ap_fixed/hls_stream/hls_vector/hls_math/hls_burst_maxi, `common.h` L14–20), SpinalHDL(Scala/SBT, spinal.lib AXI4/AxiLite4/AxiMemorySim), Verilator, Vivado 2025.2. Python(step*.py 자동화).

---

## 8. 강점 / 한계 / 리스크

### 강점
- **풀스택 정합**: SW APoT 격자 ↔ HW 시프트 디코더, SW 16×16 reorder ↔ HW 언팩, SW act 2^k ↔ HW shift — 비트정확 계약이 코드로 명시되어 재현/검증 가능(cosim + Verilator MSE 비교).
- **곱셈기-free 설계**: INT4/INT5 APoT를 conv는 가중치분해 시프트, linear은 활성 LUT 조회로 실현 → DSP 절감(고처리량/저전력 FPGA 친화).
- **SSM 융합 파이프라인**: prescan+scan을 단일 II=1 루프로 융합(`ssm.h` L143), 깊은 버퍼링 제거 — Mamba HW 가속의 핵심 난제(순차 재귀)를 dataflow로 해결.
- **모듈러 + DaisyChain**: 7개 독립 가속기 + 순차제어로 검증/재사용 용이, 한 커널이 다중 역할(norm_sum 4모드, linear flags).

### 한계 / 리스크
- **DRAM-중심 layer-by-layer**: 모듈간 공유 DRAM 경유(`ViM.scala` L104) → 메모리 대역폭 병목 가능, 진정한 레이어융합 스트리밍 아님(추정).
- **SSM 비양자화**: SSM은 fixed-point<32,14>로 유지 → 양자화 이득이 projection/conv에 한정. SSM이 latency/area에서 차지하는 비중은 latency_report 확인 필요(본 분석 미확인).
- **단일레이어 시뮬 위주**: `simulate_vim_full`이 single-layer(README L274), 합성은 NUM_LAYERS=24이나 end-to-end 24레이어 cycle-accurate 검증 범위는 확인 불가.
- **`fast_exp_approx` 근사**: `(1+x/32)^32` 근사의 정확도/포화 영향은 정량 미검증(코드 주석 수준).
- **SW 중복 구현**: `SW/model/*`와 `mamba-1p1p1/.../_quant`에 양자화 로직이 병존 — 유지보수 혼선 리스크(어느 쪽이 canonical인지 export 경로상 `SW/model/`로 추정).
- **보드/주파수/실측 성능**: 구체 파트번호·달성 Fmax·최종 Acc/전력 수치는 본 분석에서 읽은 파일에 없음(tcl/리포트/논문 필요 — 확인 불가).

---

## 9. 우리 프로젝트(HG-PIPE 계열 ViT/Transformer FPGA + XR 시선추적) 관점 시사점

### 9.1 직접 재사용 가능 포인트
1. **APoT 곱셈기-free MAC 패턴** (`linear_block.h` LUT 조회 / `conv.h` 가중치분해 시프트): HG-PIPE의 ViT linear/attention projection에 그대로 이식 가능. 특히 `decode_weights_on_stream`의 "활성당 시프트 LUT 사전계산 → 가중치 코드로 조회"는 INT4 ViT 가속의 DSP 절감 레시피.
2. **SW↔HW 비트정확 계약 설계**: `build_power_value`의 `int_weights` ↔ HW nibble 디코더, 16×16 reorder ↔ `load_linear_weights`, `FixedPointTypes.scala` 미러. 우리 ViT-Quantization↔ViT-Accelerator 사이의 정합 검증 인프라(cosim+Verilator MSE)로 채택 가치 높음.
3. **HLS dataflow + SpinalHDL BlackBox 래핑 + DaisyChain 통합 플로우**: HLS 커널을 SpinalHDL로 묶어 Vivado까지 자동화하는 step1~5 파이프라인은 우리 HW 빌드 자동화의 참고 템플릿.
4. **한 커널 다역할(flags/mode)**: `norm_sum` 4모드, `linear` SiLU/Softplus flags — 우리 가속기의 LayerNorm/GELU/residual 통합 커널 설계에 적용.

### 9.2 XR 시선추적(EventMamba / MambaPupil 계열) 연관 — 핵심
- **본 저장소는 Vision Mamba(SSM) HW 가속의 거의 유일한 풀스택 레퍼런스**. XR 시선추적에서 EventMamba/MambaPupil 등 **Mamba/SSM 기반 모델을 FPGA에 올릴 때 `ssm.h`의 selective scan 융합 파이프라인이 직접적 출발점**이 된다(순차 재귀의 II=1 구현, prescan+scan 융합, x·C 트리리덕션, `(x_C+u*D)*z_silu` 게이트).
- **`fast_exp_approx`(discretization exp)** 와 **SiLU/Softplus LUT**(`activation.h`)는 SSM 공통 빌딩블록 — 시선추적 SSM에 재사용.
- 시선추적은 저지연·저전력이 생명 → **APoT INT4/INT5 + 곱셈기-free** 조합은 XR 엣지(웨어러블) 제약에 부합. 다만 본 설계의 DRAM-중심·layer-by-layer는 초저지연 요구 시 스트리밍 융합으로 개선 필요.
- ViM은 양방향(fwd/bwd conv1d, `conv1d_b`/`x_proj_b`)인데, 이벤트기반 시선추적은 인과(causal) 단방향이 자연스러움 → `conv.h`의 causal 라인버퍼 구조는 오히려 이벤트 스트림에 더 적합(재사용 시 단방향만 취하면 자원 절감).

### 9.3 차별점 인식 (우리 ViT 가속기와 비교)
- HG-PIPE(ViT)는 attention(QKV·softmax) 중심, 본 저장소는 SSM(scan) 중심 — **연산 커널은 다르나 양자화/곱셈기-free/HLS-Spinal 인프라는 공통**. SSM 가속 know-how가 우리에게 없는 보완 자산.

---

## 10. 근거 표기

- **코드 라인 확인(확정)**: 데이터타입/양자화비트(`common.h` L60–88, L196–207; `c_quantize_model.py` L105–136; `quant_linear.py` L75), APoT 격자(`quant_utils.py` L11–82), shift-add MAC(`conv.h` L415–441; `linear_block.h` L417–533), SSM 융합(`ssm.h` L143–195), SmoothQuant(`b_smooth_model.py` L7–177), 16×16 reorder(`e_export_model.py` L26–45), DaisyChain 통합(`ViM.scala` L95–101), 타입미러(`FixedPointTypes.scala` L25–49), 빌드 플로우(README/1_run_flow.md/1_run.md).
- **추정(근거 있으나 미확정)**: 논문 제목/저자(저장소 미기재), 보드 파트번호(tcl 미확인), SSM의 latency 비중, single-layer vs 24-layer 검증 범위, `mamba-1p1p1/_quant` 파일의 canonical 여부, "fused가 아닌 DRAM-중심"(주석 L104 기반 해석).
- **확인 불가**: 달성 Acc@1/Acc@5 수치(ptq-*.txt 미확인), Fmax/전력/리소스 실측치(리포트 미생성·미확인), 정확한 타깃 FPGA 디바이스.
- **제외(분석 안 함, 이름만)**: `SW/mamba-1p1p1` 원본 mamba_ssm·`csrc/*.cu` CUDA 커널, HLS 생성 `*.v`/`*.dat`, `data.zip`, `.git`/빌드산출물.
