# HG-PIPE 정밀 분석

> 분석 대상: `REF/Transformer-Accel/HG-PIPE`
> 본 repo는 본 카테고리 최우선·핵심 분석 대상이며, 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적)의 **기반으로 추정되는** 가속기이다.
> 근거 표기 규칙: 코드/문서에서 직접 확인된 사실은 라인 근거를 명시, 추론은 "추정", 확인 못한 부분은 "확인 불가"로 표기.

---

## 1. 개요

- **목적**: Vision Transformer(ViT/DeiT) 추론을 FPGA에서 **하이브리드 그레인 파이프라인(Hybrid-Grained Pipeline)**으로 가속하는 오픈소스 구현. (`README.md` L9)
- **한줄요약**: 26개 레이어(PatchEmbed + ATTN×12 + MLP×12 + Head)를 각각 독립 HLS 커널로 합성하고, SpinalHDL로 레이어 간을 FIFO+핸드셰이크 체인으로 연결하여 **레이어 단위 코스그레인 + 레이어 내부 dataflow 파인그레인**의 2계층 파이프라인을 구성. 전 레이어 온칩 상주(가중치/중간 텐서) + 3비트 가중치 양자화 + LUT 기반 비선형(softmax/gelu/layernorm/quant) 처리로 초고처리량 달성.
- **원논문**: HG-PIPE: Vision Transformer Acceleration with Hybrid-Grained Pipeline, Q. Guo, J. Wan, S. Xu, M. Li, Y. Wang, **ICCAD 2024**. (`README.md` L233-241)
- **타깃 디바이스**: AMD/Xilinx **Versal VCK190** (`VCK190-bd-base.tcl`, `README.md` L216-221). 목표 동작 주파수 **425MHz**, 보고 성능 **7118 FPS(ImageNet 224×224), 17.8 TOPs, 381 GOPs/W, 정확도 71.05%** (`README.md` L15-17).
- **타깃 모델**: **DeiT-Tiny** (embed_dim=192, heads=3, depth=12, seq_len=196+CLS, patch_dim=768, classes=1000). (`SPINAL/.../BlockCfg.scala` L63-66, `constants.py` L12, `case/MLP.cpp.template` L5-11)

---

## 2. 디렉토리 구조 (자체 소스 트리 + 제외)

```text
HG-PIPE/
├── src/                       # [핵심] HLS 설계 헤더(템플릿 클래스). 모든 연산 모듈 정의
│   ├── common.h               # 공통 타입/상수(SYSTEM_WIDTH=64, RAM 스타일 enum)
│   ├── adapter.h              # 병렬도 변환기(pack/unpk, LCM 기반 non-divisible)
│   ├── matmul.h               # [핵심] 정적/동적 weight matmul 엔진(dataflow)
│   ├── layernorm.h            # LayerNorm(3-pass, rsqrt LUT)
│   ├── softmax.h             # Softmax(3-pass, exp LUT + reciprocal 이중 LUT)
│   ├── gelu.h                 # GeLU(LUT 룩업)
│   ├── quant.h                # 양자화(LUT 룩업, gelu와 동형)
│   ├── head_split.h           # 멀티헤드 split/merge (H=3 하드코딩)
│   ├── reshaper.h             # 텐서 재배열/전치(LCM 버퍼)
│   ├── patch_embed.h          # 패치 임베딩(URAM weight, CLS 삽입, pos-embed=2D bias)
│   ├── head.h                 # 분류 헤드(CLS 선택 → LN → matmul)
│   ├── attn.h                 # [핵심] MHA 블록 전체(dataflow로 11개 서브모듈 결선)
│   ├── mlp.h                  # [핵심] MLP 블록 전체(LN→matmul1→GeLU→matmul2+residual)
│   ├── wrapper.h              # AXI-Stream 래퍼(TLAST 생성)
│   └── utils.h                # clamp/quantize_clamp, check_stream(검증용)
├── case/                      # HLS top 함수 케이스(레이어별)
│   ├── ATTN.cpp.template      # ATTN top 템플릿(데이터타입 치환자 포함)
│   ├── MLP.cpp.template       # MLP top 템플릿
│   ├── ATTN0~11.cpp / MLP0~11.cpp  # step0가 생성한 레이어별 top(자체생성물)
│   ├── PATCH_EMBED.cpp / HEAD.cpp / RESHAPER.cpp / HEAD_SPLIT.cpp  # top
│   ├── SOFTMAX_*.cpp / LAYERNORM_*.cpp / GELU.cpp / QUANT.cpp      # 단위테스트 top
│   ├── ATTN5/7/8/10/11.cpp 등 디버그용 변형
│   └── refs/, refs.7z         # [제외] 골든데이터/가중치(생성물·바이너리)
├── statistics/                # 데이터타입/레인지 통계(type.npy, range.npy) → step0 입력
│   └── print_statistics.py
├── SPINAL/                    # [핵심] SpinalHDL 통합 레이어(레이어간 파이프라인 글루)
│   └── src/main/scala/
│       ├── block/Block.scala          # HLS 레이어를 BlackBox로 래핑
│       ├── block/BlockSequence.scala  # 26블록 + FIFO + Manager 체인 결선(top)
│       ├── block/BlockCfg.scala       # 모델/병렬도/스트림폭 전역 설정
│       ├── block/Fifo.scala           # 레이어간 FIFO
│       ├── block/Axi4StreamSpecRenamer.scala
│       ├── ctrl/Controller.scala      # AxiLite 제어 + TLAST 래핑 FSM
│       ├── ctrl/Manager.scala         # ap_start/ap_ready 핸드셰이크 데이지체인 FSM
│       ├── fc/FC.scala, fc_cfg.scala  # FC(분류기) 보조
│       └── network/Network.scala
│   └── src/test/scala/network/        # 시뮬/Verilog 생성 엔트리(Verilator)
│   ├── to_vivado.py           # 생성 Verilog/메모리초기화 → vivado 폴더 정리
│   └── [제외] .idea/, build/, *.v(BlockSequence.v 등 생성물), latency/*.txt
├── step0~step5*.py            # 엔드투엔드 빌드 플로우(아래 6장)
├── pre_syn_process.py / pst_syn_process.py  # HLS 프로젝트 생성/결과수집
├── constants.py               # FIFO 깊이 탐색 테이블, NUM_BLOCKS=24
├── template.tcl / VCK190-bd-base.tcl        # Vitis HLS / Vivado BD 스크립트
└── notebooks/HG_PIPE.ipynb    # 온보드 테스트(PYNQ류)
```

**제외 항목(이름만 언급)**: `case/refs.7z`·`case/refs/*`(골든데이터/가중치 텍스트), `SPINAL/build/`(sbt 빌드 산출물), `SPINAL/*.v`(SpinalHDL 생성 Verilog `BlockSequence.v`/`BlockSequence_bb.v`), `SPINAL/.idea/`(IDE 설정), `SPINAL/latency/*.txt`(시뮬 결과), `__pycache__`, `*.pyc`, `*__dup1.*`(중복본). `instances/`는 step1이 자동 생성하는 레이어별 Vitis HLS 프로젝트로 현재 미존재(README L40 명시).

---

## 3. 핵심 모듈·파일별 정밀 분석

> HG-PIPE의 HLS 설계는 전부 **C++ 템플릿 클래스**이다. 병렬도·데이터타입·FIFO 깊이를 템플릿 파라미터로 받아 레이어마다 인스턴스화하며, 멤버 함수가 곧 HLS 서브커널(`#pragma HLS dataflow` 내 프로세스)이 된다.

### 3.0 공통 타입·유틸 — `common.h`, `utils.h`

- `common.h`: `SYSTEM_WIDTH=64`, `system_t=ap_int<64>`(L21-22), `axis_t=ap_axiu<64,0,0,0>`(L28), `class_index_t=ap_uint<10>`(=1000 클래스 인덱싱, L25), `case_index_t=ap_uint<18>`(이미지 N, L26). RAM 스타일 enum `BRAM_STYLE=0 / URAM_STYLE=1 / LRAM_STYLE=2`(L34-36), `constexpr log2ce`(L30-32).
- `utils.h`: `clamp(val,min,max)`(L5-10), `quantize_clamp(val,bits,is_signed)`로 부호/비부호 비트폭 포화(L12-23). `#ifndef __SYNTHESIS__` 가드 내 `check_stream<...>`(L28-71)은 dataflow 스트림 중간값을 골든 ref와 비교 후 다시 써넣어(write-back) 단계별 검증을 가능케 함 — **레이어 내부 단계별 디버깅 인프라**.

### 3.1 병렬도 변환기 — `adapter.h` (`class Adapter`)

레이어 경계마다 입력 채널병렬도(CIP)와 출력 채널병렬도(COP)가 다르므로, 스트림 벡터 폭을 변환한다.
- 템플릿: `<__data_t, T, TP, C, CIP, COP>` (L12-18). `TT=T/TP`, LCM 계산 `CP=(CIP*COP)/GCD(CIP,COP)`(L32).
- `unpk()`(L40-73): CIP>COP일 때 한 벡터를 여러 작은 벡터로 분해(UNPK_TRIP=CIP/COP회 시프트, II=1 파이프라인).
- `pack()`(L75-104): COP>CIP일 때 여러 입력을 한 큰 벡터로 결합.
- `non_divisible()`(L106-113): CIP·COP가 서로 배수가 아니면 LCM(CP) 폭으로 pack 후 다시 unpk(2단 dataflow).
- `do_adapt()`(L116-127): 가분/비가분 분기 디스패치. **재사용 가치 매우 높음** — 임의 병렬도 간 결선을 자동화하는 범용 폭 변환기.

### 3.2 행렬곱 엔진 — `matmul.h` (`class Matmul`) **[가장 중요]**

전 레이어 GEMM(QKV 생성/QK/RV/O proj/MLP fc1·fc2)의 공통 엔진. **정적(weight-stationary)**과 **동적(weight-streamed)** 두 변형을 한 클래스로 제공.
- 템플릿(L6-25): `__if_t/__we_t/__bi_t/__mc_t`(입력/가중치/바이어스/누산 타입), `T,TP`(토큰·토큰병렬), `CI,CIP,CIAP`(입력채널·내부병렬·어댑터병렬), `CO,COP,COAP`(출력채널·내부병렬·어댑터병렬), 4종 FIFO 깊이, `WEIGHT_RAM_STYLE`, `USE_DSP`.
- 멤버 가중치: `weight_arr[CO][CI]`, `bias_arr[CO]`(L35-36). 생성자에서 비전치 초기화(L40-55).
- **3단 dataflow 파이프라인 `do_matmul()`**(정적, L171-192):
  1. `adapter_i.do_adapt`: 입력 폭 CIAP→CIP 변환.
  2. `matmul_step1_cache_window`(L57-98): 입력 피처맵 타일을 윈도우버퍼 `wb[TP][CI]`(LUTRAM, L62)에 캐시하고 출력채널타일(COT)만큼 재방출 → 입력 재사용. cot==0에 저장, 이후 재읽기(L71-90).
  3. `matmul_step2_mac`(L100-169): 가중치 `weight_arr`를 COP/CIP 방향으로 cyclic reshape(L102-103)해 **TP×COP×CIP MAC을 완전 언롤**(L139-158). 누산은 `bias_arr`로 초기화(cit==0, L126-134), `#pragma HLS bind_op ... impl=dsp|fabric`로 곱셈을 DSP 또는 LUT로 강제(L146-153). cit==CIT-1에 출력 방출.
  4. `adapter_o.do_adapt`: 출력 폭 COP→COAP 변환.
- **동적 matmul**(L389-421): 가중치를 별도 스트림 `w_stream`으로 받아 `matmul_step1_cache_weight`(L204-264) 또는 `..._transposed`(L266-326)로 캐시 후 `matmul_step2_mac(3-arg)`(L328-386)로 MAC. **QK^T, R·V 같은 동적 텐서곱**에 사용(가중치가 상수가 아니라 K, V 텐서).
- 병렬도 의미: 한 사이클에 **TP 토큰 × COP 출력채널 × CIP 입력채널** 곱을 동시 수행(II=1). MLP fc1은 CIP=12/COP=24, fc2는 CIP=24/COP=12(`MLP.cpp.template` L42-61) → 레이어 내부 매우 깊은 공간 병렬화 = "파인그레인".
- **데이터타입 근거**: 가중치 per-element 타입이 `ap_int<3>`(3비트 부호)로 고정(`step0_case_generation.py` L207-208, L221-222) → **3비트(준 ternary) 가중치**, 곱셈을 DSP 없이 LUT(fabric)로 구현 가능(MLP은 `M1_USE_DSP=false`, `MLP.cpp.template` L48,66).

### 3.3 LayerNorm — `layernorm.h` (`class Layernorm`)

- 템플릿(L4-19): 단계별 중간타입 9종(sum/divc/mu/var/cursor/rsqrt/affine/shift/of) + `ENTRIES_RSQRT` + `T,TP,C,CP`. 스칼라 7종(C_1_m, C_1_s, b, s1, bound, s2, clamp_bits)과 `rsqrt_table[ENTRIES]`, `lnw[C]`, `lnb[C]`(L31-47) 보유.
- **3-pass(state 0/1/2) 단일 루프**(`do_layernorm`, L78-197):
  - state0: 입력을 `buffer[TP][C]`(LUTRAM, L92-95)에 저장하며 합 누산 → 마지막에 `mean = acc*C_1_m >> C_1_s`(고정소수 역수곱, L137-144).
  - state1: `(x-mean)^2` 합산 후 `cursor=(sum+b)>>s1` clamp → `rsqrt_table[cursor]`로 1/sqrt 룩업(L162-170). **나눗셈·제곱근을 LUT로 대체**.
  - state2: `(x-mean)*rsqrt*lnw + lnb >> s2` 후 `quantize_clamp`로 정수 출력(L176-191).
- TP개 토큰을 병렬 처리(mean/sum/st_sqrt 배열을 complete partition, L102-105). II=1.

### 3.4 Softmax — `softmax.h` (`class Softmax`)

- **3-pass**(`do_softmax`, L101-215): state0 최댓값 탐색 + 버퍼(L117-143), state1 `exp(x-max)` LUT(`exp_table`) 누산 + 합의 역수를 **이중 reciprocal LUT**(recip_table_one/two, 범위 분기)로 룩업(L145-185), state2 `exp*recip` 후 분기별 `(val+b3)>>s3` requant(L187-209).
- exp/recip 모두 LUT, 누산기 `acc_val[TP]`. 이중 recip 테이블은 합의 동적범위가 넓어 단일 테이블로 정밀도가 안 나오는 문제를 범위 분할로 해결(L171-181) — **정수 softmax의 정밀도 트릭**.

### 3.5 GeLU / Quant — `gelu.h`, `quant.h` (`class GeLU`, `class Quant`)

- 두 클래스 구조 동일(L50-79): `cursor=(x+b)>>s` clamp → `table[cursor]` 룩업. GELU_ENTRIES=64, 양자화 테이블은 비선형 함수+양자화를 한 LUT에 융합(`do_gelu`/`do_quant`, II=1). LUTRAM(rom_2p) 저장(L52). **모든 활성/양자화가 LUT 한 번**으로 끝나 II=1 유지.

### 3.6 멀티헤드 — `head_split.h` (`class HeadSplit`)

- `do_split`(L28-53)/`do_merge`(L55-82): `CH=C/H`(L23), H=3을 if 분기(`h==0/1/2`)로 3개 스트림 분배/병합. **H=3 하드코딩**(DeiT-Tiny 3헤드 전용) → 헤드 수 변경 시 수정 필요(L42-48, L69-75).

### 3.7 재배열기 — `reshaper.h` (`class Reshaper`)

- `unpack`(L38-58) → `reorder`(L60-110) 2단 dataflow(`do_reshape`, L112-119). 작은 LUTRAM `buffer[TP][C]`(L63-64)에 모았다가, `TRANSPOSED` 플래그로 일반/전치 순서로 재방출(L91-97). attn에서 K는 transpose 없이(QK^T 위해 reshape), V는 transpose(R·V 위해)에 사용.

### 3.8 패치 임베딩 — `patch_embed.h` (`class PatchEmbed`)

- 정적 matmul + 특이점 다수: **가중치를 URAM에 저장**(`impl=URAM latency=3`, L116) — patch_dim 768×192가 커서. **2차원 bias `bias_arr[T][CO]`**(L35)로 **위치 임베딩을 bias에 흡수**(토큰별 bias). **CLS 토큰을 t=0에 삽입**: 첫 토큰 출력을 `cls_arr`로 덮어씀(L174-180). 출력 직전 `(acc+b)>>s` 인플레이스 스케일링(L182-191).
- 주의(버그): include guard가 `__INT_MATMUL_H__`로 matmul.h와 충돌(L1) → 동시 include 시 문제 소지(확인 가능한 코드 결함).

### 3.9 분류 헤드 — `head.h` (`class Head`)

- `select_cls`(L109-132)로 t=0(CLS) 토큰만 추출 → `Layernorm<...,1,1,CI,CIAP>`(시퀀스 1)로 정규화 → `Matmul`(192→1000)로 로짓 생성(`do_head`, L134-153). 출력 클래스 1000.

### 3.10 MHA 블록 — `attn.h` (`class Attn`) **[핵심 통합 모듈]**

ATTN 한 블록 전체를 하나의 `#pragma HLS dataflow`로 구성(`do_attn`, L368-525). 서브모듈 인스턴스(L208-250): `Layernorm lnq`, `Matmul matmul_gen_q/k/v`(QKV 생성), `Matmul matmul_qk_head1..3`(동적 QK^T, 헤드별 3개), `Matmul matmul_rv_head1..3`(동적 R·V), `Matmul matmul_gen_o`(O proj), `Quant quant_q/k/v/a`, `HeadSplit split_q/k/v` 및 `merge_a`, `Softmax softmax_qk_head1..3`, `Reshaper reshape_k/v_head1..3`.

**데이터플로우 결선(L476-524)**:
1. `stream_copy2`로 입력을 main/residual 분기 복제(L481) → residual은 `resi_i_adapter`/`resi_o_adapter`로 폭만 맞춰 통과(L482-483).
2. main: `lnq.do_layernorm`(L485) → `stream_copy3`로 LN 출력 3복제(L487) → Q/K/V matmul(L489-491) → quant_q/k/v(L493-495) → head split(L497-499).
3. K는 `reshape_k`(transpose=false) 후(L501-503) `matmul_qk`(동적, false)로 R=QK^T(L505-507) → `softmax_qk`(L509-511) → RQ.
4. V는 `reshape_v`(transpose=true)(L513-515) → `matmul_rv`(동적, true)로 A=R·V(L517-519).
5. `merge_a`로 3헤드 병합(L521) → `quant_a`(L522) → `matmul_gen_o`(O proj, L523) → `stream_merge`로 residual 합산(`out = main + ((resi*RM+RB)>>RS)`, L357, L524).
- residual 스케일링: `RM/RS/RB` 스칼라(L298-300)로 잔차 재양자화(residual의 스케일이 main과 다르므로 정수 도메인 정렬).
- **헤드별 3-way 복제**가 코드에 펼쳐져 있음(head1/2/3) → 헤드 수 변경 시 광범위 수정 필요. 디버그용 `do_attn(...ref...)` 오버로드(L530-676)는 각 단계 출력을 `check_stream`으로 검증.

### 3.11 MLP 블록 — `mlp.h` (`class MLP`)

`#pragma HLS dataflow` 내 `do_mlp`(L225-263): `stream_copy`로 main/residual 복제(L254) → residual은 adapter 2단 통과(L255-256) → `lnq.do_layernorm`(L257) → `m1.do_matmul`(fc1 192→768)(L258) → `ge.do_gelu`(L259) → `m2.do_matmul`(fc2 768→192)(L260) → `stream_merge`로 residual 합산(L261). m1/m2 모두 **BRAM_STYLE 강제**(L125,157). **residual FIFO만 깊게**(`resi_sm depth=512`, L243) — 잔차가 전체 블록을 우회해 길게 대기하기 때문.

### 3.12 AXI-Stream 래퍼 — `wrapper.h` (`class Wrapper`)

- `do_wrap`(L28-52): 내부 벡터 스트림을 `hls::axis`로 변환, N개 이미지에 대해 마지막 원소에 `last` 어서션(L44). keep/strb=-1(all valid). 가속기 출력 → AXIS DMA 경계.

### 3.13 SpinalHDL 통합 레이어 (레이어간 코스그레인 파이프라인) **[핵심]**

HLS는 **레이어 1개**만 합성한다. 26레이어를 하나의 가속기로 잇는 것이 SpinalHDL의 역할.
- **`BlockCfg.scala`**(전역 설정): block ID `-1..24`(PatchEmbed=-1, ATTN/MLP 교차 0..23, Head=24, L8-52). `SEQ_LEN=196, PATCH_DIM=768, EMBED_DIM=192, CLASSES=1000`(L63-66). 토큰병렬 `TIP=2`(L72), 출력 마지막만 TOP/TOT=1(헤드, L69,73). `DATAWIDTH_Is_ARRAY`(L17-32)로 레이어별 스트림 비트폭(8~15bit)을 명시 → `toNextPowerOfTwo`로 인터페이스폭 정렬(L57-60). AXIS config 생성(`iStreamConfig`/`oStreamConfig`, L90-106).
- **`Block.scala`**: `class BlockBlackBox`(L12-51)가 HLS가 만든 Verilog(`src/main/verilog/<name>/all.v`, L49-50)를 BlackBox로 가져옴. ap_hs(ap_start/continue/done/ready/idle) + AXIS(i/o) + clk/rst_n 포트(L23-43), 이름은 `ATTN0`, `MLP3` 등으로 setDefinitionName(L18-21). `class Block`(L54-83)이 BlackBox를 SpinalHDL `Axi4Stream`+`ApChain`으로 결선.
- **`Manager.scala` / `ApChain`**: `ApChain`(L9-34)은 Xilinx ap_ctrl_chain 핸드셰이크 번들. `class Manager`(L47-89)는 N(이미지수)·T(트리거)를 데이지체인으로 전달(`DaisyChain`, L36-40)하고, FSM(s_idle/s_work)으로 블록의 `ap_start`를 N번 어서션하며 `ap_ready`로 다음 입력 진행(L74-86). **각 블록이 독립 ap_ctrl_chain으로 연속(pipelined) 호출** → 레이어 코스그레인 파이프라인의 핵심.
- **`Controller.scala`**: AxiLite 슬레이브(N_addr/T_addr/ap_rst_n_addr, L22-24)로 호스트가 N·트리거·리셋 제어(L47-52). 마지막 블록 출력에 TLAST를 트립카운트 기반으로 래핑하는 FSM(L62-99). `N_bits=20`(최대 1M 이미지).
- **`BlockSequence.scala`**(top): `Controller` 1개 + `Block`×26 + `Manager`×26 + `Fifo`×25를 인스턴스화(L31-36)하고, manager 신호 데이지체인(L38-39), ap_chain 결선(L40), **블록→FIFO→다음블록** 스트림 체인(L43-47)으로 전체 파이프라인을 구성. `gen_block_sequence_verilog`(L51-55)가 최종 Verilog 생성.
- 시뮬레이션: `src/test/scala/network/simulate_whole_network.scala` 등이 Verilator로 전체망 사이클정확 시뮬(README L136-171), Python 소켓 서버(`launch_spinal_server.scala`)와 연동(README L110-114).

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 2계층(하이브리드 그레인) 파이프라인
- **파인그레인(레이어 내부)**: 각 HLS top(ATTN/MLP/...)이 `#pragma HLS dataflow`로 LN/matmul/quant/softmax/gelu 등 수~수십개 프로세스를 FIFO로 직결, 모든 내부 루프 II=1. 한 사이클에 TP×COP×CIP MAC. → 레이어 내부는 스트리밍 dataflow.
- **코스그레인(레이어 간)**: SpinalHDL이 26개 HLS 블록을 FIFO+ap_ctrl_chain로 직렬 연결. 각 블록이 이전 블록의 출력 스트림을 소비하며 동시에 자기 다음 이미지를 처리 → **전 레이어가 동시에 서로 다른 이미지/토큰을 처리하는 매크로 파이프라인**. 전체 지연 ≈ 가장 느린 레이어(ATTN 약 57625 cycle, README L122-133), throughput ≈ 1 이미지 / (느린레이어 지연).

### 4.2 메모리 계층 / 온칩 상주
- **모든 가중치·중간텐서 온칩 상주**(외부 DRAM 가중치 스트리밍 없음). 가중치: matmul은 BRAM/LUTRAM(`WEIGHT_RAM_STYLE`), PatchEmbed는 URAM(L116). LN/softmax/gelu/quant LUT 및 lnw/lnb는 LUTRAM(rom). 잔차 버퍼는 깊은 FIFO(URAM 깊이, `constants.py` `RESI=URAM2+URAM`≈12288, L126).
- FIFO 깊이는 `constants.py`에서 자동 탐색(URAM4/URAM2/DEEP512/MEDIUM64/SHALLOW2 후보, L15-32)하며 일부는 `DONT_TOUCH`로 고정(잔차 deep, MAC/window shallow, L98-146).

### 4.3 병렬화·양자화·데이터타입
- 병렬화 축: 토큰 TP(=2), 입력채널 CIP, 출력채널 COP. 어댑터가 레이어 경계 폭 변환.
- 양자화: **가중치 3비트(`ap_int<3>`)**, 활성/중간값은 레이어·텐서별로 `statistics/type.npy`에서 결정된 가변 비트폭(8~15bit, `BlockCfg.DATAWIDTH_Is_ARRAY`). 비선형/양자화는 전부 **사전계산 LUT**(exp/recip/rsqrt/gelu/quant). 정수 도메인 고정소수 스케일링(`(x*M+B)>>S`)이 일관 패턴.
- step0가 통계→데이터타입을 케이스 cpp에 주입하므로, **모델 가중치/스케일이 합성시점에 상수로 박힘**(완전 특수화). → 모델 교체 시 재합성 필요.

---

## 5. HW/SW 매핑

| 계층 | 구현 | 역할 |
|---|---|---|
| 알고리즘/통계 | Python(`statistics/`, step0) | 레이어별 양자화 통계→데이터타입 결정, 케이스 cpp 생성 |
| 연산 커널 | HLS C++ 템플릿(`src/*.h`) → Vitis HLS | 레이어별 dataflow 가속기(Verilog) 합성 |
| 레이어간 통합 | SpinalHDL(`SPINAL/`) → Verilog | 26블록 FIFO+ap_chain 파이프라인, AxiLite 제어, TLAST |
| 시스템 통합 | Vivado BD(`VCK190-bd-base.tcl`) | DMA(AXIS) + AxiLite + 가속기 IP, VCK190 |
| 런타임 | Jupyter(`notebooks/`, PYNQ류) | 보드 제어, 이미지 입출력, 검증 |
- 호스트↔가속기: AXIS(데이터 in/out) + AxiLite(N/트리거/리셋). DMA가 입력 토큰을 스트림으로 밀어넣고 로짓을 회수.

## 6. 빌드·실행 (6단계, `README.md` L60-227)

- **step0**(`step0_case_generation.py`): `statistics/type.npy` 로드 → ATTN/MLP 템플릿에 데이터타입 치환 → `case/ATTN0..11.cpp`, `MLP0..11.cpp` 생성(L236-240). 사전에 `refs.7z` 해제 필요.
- **step1**(`step1_hls_flow.py`): 26개 케이스에 대해 `instances/proj_*`를 만들고(`create_subprojects`, `pre_syn_process.py` L16-37), `template.tcl`로 run.tcl 생성(csim+csynth+cosim+syn, L20), Vitis HLS 2023.2를 **멀티스레드 병렬**(max_threads=16) 실행(`run_instances`, L75-119). 64GB 미만 메모리면 max_threads 축소 권고(README L77).
- **step2**(`step2_print_resource.py`): 레이어별 LUT/FF/DSP/BRAM/URAM 자원 출력(README L86-100).
- **step3**(`step3_spinal_flow.py`): 생성 Verilog를 `SPINAL/`로 복사, Verilator로 전 레이어 병렬 시뮬, 레이어별 지연(cycle) 출력. 전체망 시뮬은 `simulate_whole_network.scala`.
- **step4**(`step4_package_ip.py` + `SPINAL/to_vivado.py`): `generate_whole_network_verilog.scala`로 `BlockSequence.v`/`_bb.v` 생성 → `to_vivado.py`로 vivado 폴더 구성 → Vivado에서 AXI4 peripheral로 패키징, axilite→aximm_rtl·i/o_stream→axis_rtl 인터페이스 추론, VCK190 BD에 통합, PDI/BOOT.BIN 생성. 합성 "Flow_PerfOptimized_high", 구현 "Flow_ExploreWithRemap"으로 425MHz 목표(README L221).
- **step5**(`step5_test_on_FPGA.py` + notebooks): 보드 업로드 후 노트북으로 온보드 테스트.
- 요구 환경: Vivado HLS 2020.1+ (권장 2023.2), Python3, IDEA+Scala 2.11.12+Spinal 1.7.1+Verilator 4.228 (README L21-24).

## 7. 의존성
- HW: Vitis HLS(ap_int/ap_axi_sdata/hls_stream/hls_vector), SpinalHDL 1.7.1(Scala 2.11.12, sbt), Verilator 4.228, Vivado(VCK190). 디바이스: Versal VCK190.
- SW: Python3(numpy, colorama, threading, shutil, string.Template). 가중치/골든데이터는 `case/refs.7z`(외부 동봉, 제외 대상).
- **vendor-lock 회피**: README L227 "avoids vendor-specific IP"라 다른 FPGA 플랫폼 이식 용이하다고 주장(추정: AXIS/AxiLite 표준 인터페이스만 사용).

## 8. 강점 / 한계 / 리스크
- **강점**: (1) 하이브리드 그레인 파이프라인으로 7118 FPS@425MHz 초고처리량·381 GOPs/W. (2) 완전 온칩 상주 → 외부 메모리 대역폭 병목 제거. (3) 비선형/양자화 전부 LUT II=1 → softmax/gelu/LN이 throughput 저해 안 함. (4) 3비트 가중치 + LUT-MAC으로 DSP 절약(MLP DSP 미사용). (5) 템플릿 기반 → 병렬도/데이터타입/FIFO 깊이가 파라미터화. (6) HLS(연산)+SpinalHDL(통합)+Python(플로우)의 명확한 3계층 분리, check_stream/Verilator로 검증 인프라 충실.
- **한계/리스크**: (1) **DeiT-Tiny 특수화** 강함 — H=3 헤드가 attn.h/head_split.h에 펼쳐져 하드코딩(헤드/임베드 변경 시 광범위 수정). (2) 모델 가중치·스케일·데이터타입이 **합성시점 상수** → 가중치 갱신마다 전체 재합성(런타임 가중치 로딩 불가; 동적 matmul은 텐서 곱셈용일 뿐 가중치 교체용 아님). (3) 완전 온칩 상주는 큰 모델(DeiT-S/B, LLM)로 확장 시 자원 한계. (4) `patch_embed.h` include guard 충돌(코드 결함). (5) seq_len=196 고정(이미지 224×224, 16×16 패치). (6) 통합 플로우가 Windows 경로(`C:/programs/xilinx`)·수동 Vivado GUI 단계 의존(재현성 부담).

## 9. 우리 프로젝트 관점 시사점 (★핵심)

> 우리 목표: "고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적". HG-PIPE는 본 프로젝트 기반으로 **추정**되므로, 재사용·개조 포인트를 구체화한다.

**(A) 직접 재사용 가능한 자산**
- `matmul.h`의 정적/동적 Matmul 엔진: QKV/QK/RV/proj/FFN 전부 이 한 엔진으로 커버. 시선추적용 경량 ViT(예: 작은 해상도 eye-patch)에 그대로 인스턴스화 가능. 병렬도(TP/CIP/COP)만 우리 자원·지연 목표에 맞게 조정.
- `adapter.h`: 레이어 경계 임의 병렬도 결선 자동화 → 우리가 레이어별 병렬도를 다르게 두어도 결선 자동.
- LUT 기반 `softmax/gelu/layernorm/quant`: 비선형을 throughput 무손실로 처리하는 검증된 패턴. 시선추적의 실시간(저지연) 요구에 적합.
- SpinalHDL 통합 계층(`Block/Manager/Controller/BlockSequence`): **레이어를 블록으로 추가/삭제·재배열**하는 프레임워크. 우리 모델의 레이어 구성이 달라도 BlockCfg 배열만 수정하면 재구성.

**(B) XR 시선추적을 위한 개조 포인트 (추정)**
- 입력 파이프: PatchEmbed의 2D-bias(위치임베딩 흡수)·CLS 삽입 구조를 **eye-image 패치 임베딩**으로 치환. 시선추적은 보통 회귀(gaze 좌표) → `head.h`의 분류 1000-class를 **2~3차원 좌표 회귀**(작은 FC)로 교체.
- 시퀀스/임베드 축소: seq_len=196·embed=192를 시선추적용 소형 ViT로 줄이면 자원 여유 → **더 높은 TP/COP**로 초저지연(수십 µs) 달성 여지(XR 프레임율·motion-to-photon 요구).
- 헤드 하드코딩 일반화: H=3 펼침을 파라미터화(루프/배열화)하면 우리 모델 헤드 수 자유. **우선 개선 항목으로 추정**.
- 양자화: 3비트 가중치가 시선추적 정확도에 충분한지 재평가 필요(좌표 회귀는 분류보다 양자화 민감 가능 — 확인 필요). `statistics`+step0 파이프라인을 우리 모델 통계로 재실행.

**(C) 아키텍처 차용 원칙**
- "전 레이어 온칩 상주 + 2계층 파이프라인"은 **저지연·고FPS** XR에 이상적. 단, 모델이 커지면 온칩 한계 → 시선추적용으로 모델을 작게 유지하는 전략과 잘 맞음(추정).
- 합성시점 가중치 상수화는 **모델 고정** 시나리오(배포된 시선추적기)엔 오히려 장점(최대 효율). 모델 자주 바꾸면 단점.

**(D) 우리 repo와의 관계 (확인 불가/추정)**
- `IMPL_REPOS/HGPIPE`, `HGTXR` 등 다른 작업디렉토리에 본 repo 파생본이 있을 것으로 추정되나, 본 분석 범위(REF/Transformer-Accel/HG-PIPE) 밖이라 교차 비교는 **확인 불가**. INDEX 통합 단계에서 대조 권장.

## 10. 근거 표기
- 사실(라인 근거): 3~6장의 모든 라인 인용은 실제 파일 Read로 확인. 성능수치(7118FPS 등)는 `README.md` 자기보고치(실측 재현은 확인 불가).
- 추정: 9장의 개조 방향, vendor-lock 회피 일반화, 우리 repo 파생 관계.
- 확인 불가: `instances/`·`case/refs/` 내부 수치(생성물/바이너리, 제외), 우리 프로젝트(HGTXR 등)와의 구체적 코드 대응 관계.
