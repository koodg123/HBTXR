# TeraFly (LoopLynx) — 멀티노드 FPGA 기반 LLM 협력 추론 가속기 정밀 분석

> 대상 경로: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Accelerator\TeraFly`
> 분석 방식: Glob/Grep/Read 기반 실제 소스 라인 근거. 생성물(*.xclbin/*.xo/*.bin/*.pyc) 및 입력 데이터(.bin)는 이름만 언급하고 내용 분석에서 제외.

---

## 1. 개요

- **한줄요약**: TeraFly는 OPT-1.3B 같은 디코더형 LLM을 Xilinx Alveo U50lv FPGA에서 **INT8 양자화 + 데이터플로우(스트리밍) + 멀티노드(멀티 CU 링) 협력 추론**으로 가속하고, 토큰 생성을 웹 GUI로 실시간 데모하는 전체 스택이다.
- **목적**: 메모리 대역폭(HBM)과 연산 효율을 극대화하여 호스트 개입을 최소화한 end-to-end LLM 추론을 제공. 동시에 **연구용으로 빠르게 커스터마이즈 가능한 HLS 커널**(템플릿 + Python 코드생성)을 제공하는 것이 README가 명시한 하이라이트다(README.md L11, L17).
- **원논문(추정 아님, README가 명시)**: 두 편의 자체 논문에 기반.
  - **Terafly** (TCAD 2025): *"Terafly: A Multi-Node FPGA Based Accelerator Design for Efficient Cooperative Inference in LLMs"*, Zheng, Chen, Huang, Lou, Zheng (README.md L102–L109).
  - **LoopLynx** (DATE 2025): *"LoopLynx: A Scalable Dataflow Architecture for Efficient LLM Inference"*, Jianing Zheng, Gang Chen (README.md L111–L117).
  - 즉 **TeraFly = 멀티노드 확장판 / LoopLynx = 베이스 데이터플로우 아키텍처**. 커널 이름·심볼이 전부 `loopLynx_*`인 것은 이 계보 때문이다.
- **타깃 디바이스 / 환경**: Alveo **U50lv** (`xilinx-u50lv-gen3x4-xdma-base_2` 셸), XRT 2023.2, Vitis HLS & Vivado 2023.2, Ubuntu 18.04 (README.md L33–L40). 생성 코드 디렉토리명/플랫폼 ID는 `xilinx_u50lv_gen3x4_xdma_2_202010_1` (codegen.py L8).
- **모델**: OPT-1.3B (24 layer, hidden 2048, head 32×64, vocab 50272). 코드 내 OPT-125m 분기 흔적도 존재(codegen.py L5 주석).

---

## 2. 디렉토리 구조 (자체 소스 트리)

```
TeraFly/
├── README.md                     # 프로젝트 개요·빌드·인용 (분석함)
├── codegen.py                    # ★ HLS 코드 생성기 (템플릿 치환)
├── weight_packer.py              # ★ 가중치 INT8 양자화·패킹·메모리 레이아웃
├── OPT-1.3b.toml / OPT-1.3b.json # ★ 모델·병렬도 설정 (동일 내용)
│
├── template/                     # ★ 코드생성 입력이 되는 HLS 템플릿
│   ├── top.cpp / top.h           #   디코더 1레이어 파이프라인 최상위
│   ├── kernels.cpp               #   공통 커널(LN copy, Gelu, Res, Acc, KV writer 등)
│   ├── gemm_quant.cpp            #   INT8 GEMM + requant + 라우터
│   ├── attention.cpp             #   Q·K / softmax / P·V 헤드별 데이터플로우
│   ├── adder_tree_32.cpp / _64.cpp  # 파이프라인 곱셈-가산 트리(INT8 MAC)
│   ├── router.cpp                #   멀티 CU 링 라우터 + write_buffer
│   ├── layerNorm.cpp             #   단일 패스 LayerNorm
│   ├── Makefile / makefile_us_alveo.mk / utils.mk  # 빌드 템플릿
│   └── connectivity.cfg          #   CU↔SLR↔HBM 매핑 템플릿
│
├── OPT-1.3b_optimize/            # ★ codegen.py가 생성한 Vitis 프로젝트(생성물)
│   ├── loopLynx.cpp / loopLynx.h #   (생성된) 통합 커널
│   ├── params.h                  #   (생성된) 상수/typedef 헤더
│   ├── host.cpp                  #   C++ XRT 호스트(벤치마크용, 생성물 계열)
│   ├── data.h                    #   상수 데이터 헤더(생성물)
│   ├── Makefile / *.mk / connectivity.cfg / command.sh / xrt.ini
│   ├── build_dir.hw.*/loopLynx.xclbin  # [제외] 합성 산출물(이름만)
│   └── tokenizer/
│       ├── conver_vocab.py       #   vocab key/value swap 유틸
│       ├── tokenizer_predict*.cpp (original/eigen/generate)  # lambada 벤치 호스트
│       ├── json.hpp (nlohmann, third-party)
│       └── inputs_hardware_full/input_*.bin  # [제외] 입력 데이터
│
└── LLM-demo-gui/
    ├── alveo/                    # ★ 단독 데모용 XRT 호스트(현행 v3)
    │   ├── xrt_py_new.py         #   ★ pyxrt 기반 LoopLynx 클래스
    │   ├── client-v3.py          #   ★ websocket 서버
    │   ├── utils_binding.py      #   XRT 옵션/플랫폼 파서(Xilinx 샘플 유래)
    │   └── tokenizer-1.3b/*.json #   GPT2Tokenizer 자원
    └── llm-gui/
        ├── server/python/        # ★ 동일 스택의 서버 사본 (client.py~v3, xrt_py.py)
        ├── web/                  #   프론트엔드(분석 제외 대상이나 경량)
        │   ├── index.html / script/app.js  # Vue3 + Quasar (CDN 로컬 번들)
        │   ├── cdn/*.js,*.css     #   [제외] vue/quasar 번들
        │   └── img/*             #   [제외] 아이콘
        ├── README.md / message-format.md  # GUI·프로토콜 설명(분석함)
└── .git/                         # [제외]
```

> **중요한 사실(제약 충족)**: 본 repo는 **HLS 소스가 명백히 존재**한다. RTL(.v/.sv)은 **없음**(Glob 0건). 프론트엔드는 React/node_modules가 아니라 **Vue3 + Quasar의 CDN 로컬 번들**이며 별도 node 빌드 산출물이 없으므로 자연히 제외된다. 서버 백엔드는 node.js가 아니라 **순수 Python websocket**이다.

---

## 3. 핵심 모듈·파일별 정밀 분석 (가장 중요)

### 3.1 `codegen.py` — 설정 기반 HLS 코드 생성기 (template → 구체 커널)

코드생성의 본질은 **"플레이스홀더 치환형 메타프로그래밍"**이다. RTL/HLS를 새로 합성하는 것이 아니라, `template/*.cpp`의 `{TOKEN}` 자리들을 설정값으로 채워 Vitis용 구체 소스를 찍어낸다.

- **설정 로드**: `OPT-1.3b.toml`을 `toml.load`로 읽음(codegen.py L11–L12). 동일 내용의 `.json`도 존재(README가 `.json`을 언급). 설정 키: `CU, PROCESSOR, INP_NUM, ROUTE_NUM, INP_PARALLEL, HEAD_PARALLEL, HEAD_NUM, HEAD_LEN, FULL_INP_LEN, FULL_SEQ_NUM, NUM_LAYER`.
- **(1) params.h 생성** (L18–L100): 설정값으로부터 **모든 파생 상수를 산술적으로 계산**해 `const int`로 박아 넣는다.
  - `GCD_PE_RT = lcm(PROCESSOR, ROUTE_NUM)` → `ROUNDS_FROM/ROUNDS_TO`(PE 폭과 라우터 폭의 비율, 데이터 리패킹 라운드 수, L23–L27).
  - `WEIGHT_SIZE/_LAYER/_TOTAL` = `FULL_INP_LEN² / (PROCESSOR·CU·INP_NUM)`에 ×12(QKV3+O1+MLP1×4+MLP2×4) ×NUM_LAYER (L32–L34). → 레이어당/전체 가중치 워드 수.
  - `KV_CACHE_SIZE`, `QKV/O/MLP1/MLP2_ROWS/COLS`, 각종 BIAS 오프셋(L44–L61) = 호스트와 커널이 공유하는 메모리 레이아웃의 단일 진실원천(SSOT).
  - `ATTENTION_CHANNELS = HEAD_PARALLEL·HEAD_LEN/INP_NUM` (L28) = 어텐션 병렬 채널 수.
  - **데이터타입 typedef 동적 생성**(L97–L100): `INP_NUM`이 32냐 64냐에 따라 `io_pack_int8 = ap_uint<8*INP_NUM>`, 곱셈-가산 트리용 `datapack_64=ap_uint<16*64>`, `datapack_32=ap_uint<17*32>` … 처럼 **비트폭이 누적(16→17→18→19→20→21)되도록** 단계별 패킹 타입을 찍어낸다. 이는 가산 트리에서 오버플로 방지용 비트 확장(§3.5)과 정확히 대응.
  - `KV_LOCAL_BIAS_{cu}[]` 배열(L64–L95): CU·KV·head·PE 4중 루프로 KV 캐시의 온칩 기록 오프셋 테이블을 미리 펼쳐 상수 배열로 출력.
- **(2) Makefile / makefile_us_alveo.mk 생성** (L104–L147): `{PLATFORM}` 치환, **CU 개수만큼 `loopLynx_{cu_id}.xo` 빌드 규칙을 반복 생성**(L122–L134). 즉 멀티노드는 "동일 커널을 CU개 인스턴스화"하는 방식.
- **(3)~(6) 연산 커널 부분 치환**:
  - layerNorm(L149–L174): `CU≠1`이면 라우터를 끼우고(`router_ln`), 단일 CU면 바로 write (조건부 코드 삽입).
  - gemm(L176–L220): `{MEM_PORT_REGION}`에 `PROCESSOR`개 `w_addr_{i}` 인자/`weight_loader` 호출을 펼치고, `PROCESSOR≠ROUTE_NUM`이면 `adapter`(리패킹) 삽입, `CU≠1`이면 `router` 삽입.
  - attention(L222–L279): `ATTENTION_CHANNELS`만큼 K/V loader 펼침, `HEAD_LEN/INP_NUM` 값(4/2/1)에 따라 `acc_result` 합산식을 분기 선택(L256–L261), `LOG_FULL_SEQ_LEN = log2(FULL_SEQ_NUM)`를 softmax 고정소수 비트폭에 주입(L271).
  - top(L281–L372): `PROCESSOR`개 `m_axi` 포트와 `#pragma HLS interface ... bundle=gmem{i}` 펼침(L300–L304), **CU 개수만큼 `loopLynx_{CU_ID}` 함수 전체를 복제**(L337–L353).
- **(7) 최종 조립**(L374–L416): kernels + adder_tree(INP_NUM=32/64 분기) + router + attention + ln + gemm + top 문자열을 **하나의 `loopLynx.cpp`로 concat**(L392), `{Mul_Adder_Tree}` 토큰을 `Mul_Adder_Tree_64` 등 실제 함수명으로 최종 치환(L394–L408) 후 `OPT-1.3b/loopLynx.cpp`, `loopLynx.h`로 기록.

> **요지**: codegen.py는 "HW를 합성"하지 않는다. **설정값 → 파생 상수 + 병렬 인스턴스 펼치기 + 조건부 라우터/어댑터 삽입**으로 Vitis HLS가 합성할 C++ 소스를 자동 생성하는 **얇은 템플릿 엔진**이다. DSE(Design Space Exploration)는 toml의 `PROCESSOR/INP_NUM/CU/ROUTE_NUM`을 바꿔 재생성하는 식으로 수행된다고 추정.

### 3.2 `weight_packer.py` — INT8 양자화·순환 슬라이싱·메모리 레이아웃

호스트가 FPGA HBM에 올릴 **바이너리 가중치를 만드는 오프라인 스크립트**. 입력은 사전 양자화된 `opt-1.3b.npz`(q/k/v/out/fc1/fc2 weight·bias + `*_a` 스케일).

- **상수 일치**: `LAYER_NUM=24, CU=2, PROCESSOR=8, INP_NUM=64, EMBEDDING_SIZE=2048` (L5–L9) — toml과 동일.
- **순환 슬라이싱 `cyclic_slice`** (L38–L46): 가중치 행렬을 `[CU, PROCESSOR, rows/PROCESSOR, cols]`로 재배열. 행을 **CU로 분할 후 PROCESSOR 개 PE에 라운드로빈(`i + j*PROCESSOR`)**으로 인터리브 → 각 PE가 자기 HBM 뱅크에서 연속 읽기를 하도록 물리 레이아웃을 맞춘다. 이것이 §3.4 `weight_loader`의 순차 접근 패턴과 직결.
- **레이어 패킹** (L48–L59): QKV(이미 concat) + O + FC1 + FC2를 PE축으로 이어붙여 `layer_weights[LAYER, CU, PROCESSOR, EMBED²·12/CU/PROCESSOR]` 구성.
- **INT8 직렬화** (L61–L65, L83–L88): `astype(np.int8)`로 바이트화하여 **`w{cu}_addr_{processor}.bin`** 파일(CU×PROCESSOR = 2×8 = 16개)로 출력. → xrt_py_new.py가 PE별 HBM BO에 그대로 적재.
- **혼합 정밀 패킹** `gen_int8_half_data_pack_bin`(L67–L81): INT8 본체 뒤에 **half를 `round(x·2^6).astype(uint16)`** 고정소수(Q?.6 형태)로 붙이는 변형도 정의(실사용은 const_data 경로). → 스케일/바이어스는 FP, 가중치는 INT8의 하이브리드.
- **bias / LayerNorm / 스케일(alpha) 패킹** (L91–L161):
  - bias = qkv+out+fc1+fc2를 layer·CU별로 concat (L91–L96).
  - LN weight/bias = attn_ln + fc_ln concat (L98–L106).
  - **스케일 팩터 alpha**: `q_a/k_a/v_a` 등을 `sample_granularity=CU`만큼 repeat하고, fc1은 추가로 ×4 repeat(FC1이 4배 폭이므로) (L136–L147). 이들을 `const_data_{cu}.bin` (FP32)으로 출력(L152–L155).
  - `attn_alpha`(QK 스케일), `ctx_alpha`(PV 스케일)는 **C++ `const float [] = {...}` 텍스트 헤더**(`attn_alpha.txt`, `ctx_alpha.txt`)로 생성(L119–L128, L160–L161) → attention.cpp가 `#include`로 직접 흡수(attention.cpp L405–L406).

> **양자화 요약**: 선형/어텐션 행렬곱은 **INT8×INT8 → INT32 누적**, 이후 **FP32 스케일(alpha) 곱 + FP bias 가산 → 클램프 → 다음 단계용 INT8 재양자화**(requant). LayerNorm/softmax/Gelu 경계의 FP는 유지. weight_packer는 이 정밀 분할에 맞춰 "INT8 가중치 / FP 스케일·바이어스"를 분리 직렬화한다.

### 3.3 `template/top.cpp` — 디코더 1레이어 데이터플로우 최상위 (`loopLynx_{CU_ID}`)

생성 후 커널의 심장. 커널 시그니처: `(i_seq, host_addr, load_weight_layer, w_addr_0..PROCESSOR-1, stream_previous, stream_next)` (L1–L6).

- **인터페이스**: `host_addr`는 `m_axi bundle=gmem8`(L8), 각 `w_addr_{i}`는 `gmem{i}`(생성된 `{MEM_PRAGMA_REGION}`). `stream_previous/next`는 CU 간 링 연결용 AXIS 스트림.
- **온칩 버퍼/저장소 바인딩**(L13–L33): `input/res_buffer`는 cyclic partition(factor INP_PARALLEL/INP_NUM), `KV_buffer`는 `[2·ATTENTION_CHANNELS][INP_LEN/ATTENTION_CHANNELS]`, `bias_onboard/ln_*_onboard`는 **URAM**, `linear_alpha`는 **BRAM**에 `bind_storage`. → 가중치 스케일·LN 파라미터는 칩 내부 상주.
- **가중치/상수 적재 모드**(L35–L65): `load_weight_layer >= 0`이면 해당 레이어의 bias·LN을 `host_addr`에서 URAM으로 복사(II=1 파이프라인)하고 즉시 return. 이 분기를 **호스트가 레이어 수만큼 미리 호출**해 온칩 상수를 채운다(xrt_py_new.py L94–L104과 대응).
- **레이어 본체**(L69–L115): `NUM_LAYER` 루프 안에서 한 레이어의 OPT 디코더 연산을 순차 데이터플로우로 수행:
  1. `Fused_Res_LN_copy` — pre-LN(어텐션 LN) + 잔차 복사 (L80).
  2. `GEMM_QUANT(... QKV_ROWS, QKV_COLS ...)` — QKV 투영(INT8 GEMM) (L83).
  3. `{WRITE_KV_ONCHIP}/{WRITE_KV_OFFCHIP}` — KV를 온칩 버퍼 및 HBM(KV 캐시)에 기록 (L86–L87).
  4. `q_writer` → `attention_wrapper_new` — 멀티헤드 어텐션 (L89–L93).
  5. `GEMM_QUANT(... O ...)` — 어텐션 출력 투영, `isFloat=true` (L96–L97).
  6. `Fused_Res_LN_copy`(fc LN) → `GEMM_QUANT(MLP1, fuseActivation=true)` → `Gelu_layer` → `GEMM_QUANT(MLP2)` → `Acc_layer`(잔차합) (L100–L108).
  7. `res_buffer → input_buffer` 복사(다음 레이어 입력) (L110–L114).
- **출력**: `output_writer_memory(res_buffer, host_addr)` — 최종 hidden state를 host_addr로 되돌림(L117). 디코딩 토큰의 logit/샘플링은 **호스트가** 수행(§3.6).

> 즉 FPGA 커널은 **"임베딩 입력 → 24개 디코더 레이어 → 마지막 hidden state"**까지 담당하고, final LayerNorm + lm_head 행렬곱 + 샘플링은 Python 호스트가 처리하는 **HW/SW 분담 구조**.

### 3.4 `template/gemm_quant.cpp` — INT8 GEMM + requant + 라우팅

`#pragma HLS DATAFLOW`로 다음 스테이지를 동시 가동(L135):
- `weight_loader`(PROCESSOR개) — 각 PE의 HBM 포트에서 `io_pack_int8` 워드를 II=1로 스트리밍(L37–L49). 순환 슬라이싱된 레이아웃 덕에 연속 주소 접근.
- `stream_copy` — 입력 활성을 INP_NUM폭으로 패킹해 PROCESSOR개 디스패처에 브로드캐스트(L51–L74).
- `{Mul_Adder_Tree}` + `accumulate_manager` — PE별 곱셈-가산 트리로 부분합, COLS만큼 누적해 행 결과(INT32) 산출(L164–L169, L1–L16).
- `acc_result_merger`/`adapter`(조건부) — PROCESSOR개 INT32를 `proc_pack_int`로 패킹, PE폭≠라우터폭이면 리패킹(L171, L18–L35).
- **`requant`**(L77–L118) — 핵심 양자화 단계: `acc(INT32)·linear_alpha[scale] + bias_onboard[]` → FP. `isFloat`면 그대로(O/MLP2 출력), 아니면 [-128,127] 클램프(다음 INT8 입력). FP 비트는 `converter_t` union으로 패킹해 라우터 워드(`router_pack_float`)에 실어 보냄.
- `{WEATHER_USE_ROUTER}` — `CU≠1`이면 `router(...)`로 부분합을 링에 흘리고 `write_buffer`, 단일이면 바로 `write_buffer`(L201–L205 생성 측).

### 3.5 `template/adder_tree_64.cpp` (및 _32) — INT8 MAC 곱셈-가산 트리

- 1단계: `in_0×in_1` 64개 INT8 곱(`ap_int<16> c = a*b`)을 `#pragma HLS BIND_OP impl=dsp`로 **DSP에 강제 매핑**, II=1 파이프라인(_32: L23–L36).
- 이후 **이진 트리로 절반씩 가산**하며 단계마다 비트폭을 +1 확장(`datapack_32→16→8→4→2`, 16→17→18→…). 이는 §3.1의 typedef 누적 비트폭과 일대일 대응 → 누적 오버플로 없이 INT 정확도 보존.
- `INP_NUM`(=64) 분기로 트리 단수를 결정. codegen이 `{Mul_Adder_Tree}`를 `Mul_Adder_Tree_64`로 치환.

### 3.6 `template/attention.cpp` — 멀티헤드 어텐션 데이터플로우

전 과정이 `#pragma HLS DATAFLOW`로 헤드 병렬 + 스트리밍(L403). 흐름(L408–L444):
- `loader_new`(K/V) + `q_loader_new` — KV 캐시(HBM)와 온칩 Q를 스트림으로 로드. KV 인덱스 = `layer·KV_CACHE_SIZE + head·FULL_SEQ_NUM + row`(L183–L193) → seq 진행에 따라 누적.
- `attention_layer_new` → `Attention_head_wise` — Q를 seq_id+1회 복제(`value_generator`)하고 K와 곱셈-가산 트리로 **Q·Kᵀ 점수** 산출(L26–L38, L12–L24).
- `merge_sub_result` + `{ACC_RESULT_SELECT}` — HEAD_LEN/INP_NUM 채널 부분합을 헤드 점수로 합산(L40–L67).
- `requant_attn` — 점수×`attn_alpha[layer]`로 FP 환산(L69–L84).
- `softmax_new` — **수치안정 softmax**: max 추출 → `exp(x-max)+1e-6` 합산(고정소수 `ap_fixed<32+LOG_SEQ, LOG_SEQ>`로 누적, L92) → 역수 곱 → **×127 후 INT8 재양자화**해 출력(L86–L176). 4-패스 스트리밍(max/exp·sum/inv/normalize)으로 BRAM bypass FIFO 사용.
- `context_layer_new`/`Mac_32` — softmax(INT8)와 V(INT8)의 **P·V 행렬곱**(L196–L228, L322–L333).
- `requant_ctx` — ×`ctx_alpha[layer]` 후 클램프 → FP 패킹(L264–L301).
- `packet_merger` → `{WEATHER_USE_ROUTER_ATTENTION}`(router_attention → write_buffer_int) — 헤드 결과를 링으로 모아 INT8 출력 버퍼에 기록(L303–L320, L444).

### 3.7 `template/router.cpp` — 멀티 CU 링 라우터 (멀티노드 핵심)

TeraFly의 "멀티노드 협력"을 실현하는 **시프트-링(rotate) 통신**. `router`/`router_ln`/`router_attention` 모두 동일 패턴(L139–L184 등):
- 각 CU는 자기 몫(부분합)을 8워드 블록 단위로 버퍼링 후, `CU`회 반복하며 `stream_next.write` → (delay) → `stream_previous.read`로 **이웃 CU에서 받은 데이터로 버퍼를 갱신**. 즉 링을 한 바퀴 돌며 모든 CU의 부분 결과를 합쳐 완전한 활성 벡터를 재구성한다.
- `write_buffer`(L84–L137)는 `(device+1)·STRIDE + cu·STRIDE) % FULL_LEN`로 **CU별 출력 위치를 회전 배치**하고 `fuseActivation`(ReLU류) 옵션 적용. `write_buffer_int`는 INT8로 round 저장(L187–L214).
- 물리 연결은 `connectivity.cfg`에서 `stream_connect=loopLynx_0_1.stream_next:loopLynx_1_1.stream_previous:64` 양방향(§4)으로 실현.

### 3.8 `template/layerNorm.cpp` & `template/kernels.cpp` — 정규화·잔차·보조 커널

- **layerNorm**(L2–L65): 단일 패스로 mean/var 누적(`var = E[x²]-E[x]²`, eps=1e-5), `1/sqrt`로 정규화 후 URAM 상주 `ln_weight/ln_bias` 적용. CU별 `LN_INNER_BIAS=device·INP_LEN` 오프셋.
- **kernels.cpp**: `write_kv_buffer_onchip`/`kv_writer_parallel`(KV INT8 round 저장, L4–L40), `Gelu_layer`(여기선 FP→INT8 round만; 실제 GELU 근사는 상류 단계로 추정, L42–L54), `Res_layer`/`Acc_layer`(잔차 합, L55–L89), `input_loader_memory`/`output_writer_memory`/`q_writer` 등 입출력 보조(L92–L120+).

### 3.9 `LLM-demo-gui/alveo/xrt_py_new.py` — pyxrt 기반 호스트 (★ XRT 제어 흐름)

`class LoopLynx`가 FPGA 제어 전체를 캡슐화.
- **디바이스/xclbin 로드**(L57–L62, L75–L78): `pyxrt.device(0)` → `pyxrt.xclbin(BIN)` → `device.load_xclbin`. `BIN`은 `build_dir.hw.*/loopLynx.xclbin`(L24, 산출물).
- **커널 핸들**(L80–L81): `pyxrt.kernel(d, uuid, "loopLynx_0"/"loopLynx_1", shared)` — CU당 1개(L41).
- **버퍼 객체(BO) 할당**(L84–L88): `inp_addr_bo`(=`INP_LEN·13·4`B, group_id(1)), 그리고 **CU×PROCESSOR(=2×8)개 `w_addr_bo`**(크기 `WEIGHT_SIZE+KV_CACHE_SIZE`, group_id(3+processor)) → HBM 뱅크에 대응.
- **상수/가중치 적재**(L91–L110):
  - `const_data_{cu}.bin`을 레이어별로 잘라 `inp_addr_bo`에 write → `sync(BO_TO_DEVICE)` → **`kernel(..., layer, ...)` 호출로 온칩 URAM에 적재**(top.cpp의 `load_weight_layer>=0` 분기, L94–L99). 마지막에 alpha 적재(L100–L104).
  - `w{cu}_addr_{processor}.bin`(weight_packer 산출)을 각 PE BO에 write→sync(L105–L110).
- **임베딩 행렬·pos·final LN**을 numpy로 메모리 적재(L112–L115).
- **추론 1스텝 `inference`**(L117–L183): 호스트에서 `embedding = lm_head[input_id] + pos_embed[token]`(L118) 계산 → 모든 CU BO에 write/sync → **`run = kernel(token, inp_addr_bo, -1, w_addr_0..7, 0, 0)`** 비동기 실행 후 `run.wait()`(L126–L129; `-1`은 추론 모드). 결과 hidden state를 `BO_FROM_DEVICE` sync 후 `frombuffer`로 회수(L133–L134).
  - **호스트 측 후처리**: final LayerNorm(L135–L138) → `lm_head.T` 행렬곱으로 logits(L139) → **temperature 스케일(0.9) + repetition penalty(1.1) + top-k(k=5) 샘플링**(L143–L158). EOS(id 2) 처리(L166), 특수 토큰 필터(L172–L175).
- **세션 루프**(L196–L233): `process_prompt`(프롬프트 토큰 prefill) → `process_token`(autoregressive) → `print_latency`(토큰당 ms, python wrapper 포함). 단독 CLI 진입점(L239–L242).

> 멀티 CU 실행은 "for cu: kernel(...) 비동기 시작 → for cu: wait"의 **fork-join**이며, CU 간 부분합 교환은 HW 링(stream_connect)이 담당하므로 호스트는 동기화만 한다.

### 3.10 `client-v3.py` / `utils_binding.py` / 서버 사본

- **client-v3.py**(websocket 서버, L1–L97): `xrt_py_new`의 `LoopLynx()`를 1회 생성(L6), `websockets.serve(..., "localhost", 10088)`(L88). 수신 input → `loopLynx.process_prompt` → `responseStart` → 토큰 루프에서 `valid`한 단어마다 `response` 전송 → `responseEnd(latency)`(L52–L65). message-format.md의 ack 핸드셰이크 그대로 구현(`wait_ack`, L9–L18).
- **utils_binding.py**: Xilinx XRT 파이썬 샘플 유래(Apache-2.0). `Options`(bitstream/platform/cu_index 파싱)와 `parsePlatform`(DDR/HBM 주소 세그먼트 카운트)(L20–L105). 실제 v3 경로에서는 보조적.
- **`server/python/`**: `xrt_py.py`(클래스 미사용 절차형 버전, BASE 경로만 다름, L22–L25)와 `client.py~client-v3.py` 사본. 동일 스택의 GUI 통합 버전.

### 3.11 토크나이저 / 벤치마크 호스트

- `conver_vocab.py`: vocab.json의 **key/value를 스왑**(token→id를 id→token으로)해 디코딩용 역사전 생성(L3–L13). 경로가 작성자 로컬(`/home/zjnyly/...`) 하드코딩 → 일회성 유틸.
- `tokenizer/tokenizer_predict_*.cpp`: lambada 벤치마크용 **C++ XRT 호스트**(original/eigen/generate 변형). README는 `tokenizer_predict_eigen.cpp`로 패킹 데이터 적재를 검증하라고 안내(README.md L76). nlohmann `json.hpp`는 third-party.

---

## 4. 데이터플로우 / 실행 흐름

**오프라인(코드·데이터 준비)**
1. `OPT-1.3b.toml` 설정 → `codegen.py` → `OPT-1.3b/`(=OPT-1.3b_optimize 계열)에 `params.h`, `loopLynx.cpp/.h`, `Makefile`, `connectivity.cfg` 생성.
2. `opt-1.3b.npz`(사전 양자화) → `weight_packer.py` → `w{cu}_addr_{p}.bin`(INT8 가중치 16개) + `const_data_{cu}.bin`(FP bias/LN/alpha) + `attn_alpha.txt/ctx_alpha.txt`(C++ 헤더).

**합성**
3. `make run` → Vitis `v++`가 CU개 `.xo` 빌드·링크 → `loopLynx.xclbin`(산출물) + Alveo 프로그래밍.

**런타임(추론)**
4. 호스트(xrt_py_new.py): xclbin 로드 → CU 커널 핸들 → BO 할당(inp + 16개 weight) → const/weight를 sync로 HBM 적재, 레이어별 kernel 호출로 URAM에 bias/LN/alpha 상주.
5. 토큰마다: 호스트가 `embedding = token_embed + pos_embed`(FP32) 작성 → BO write/sync → **CU 커널 fork-join 실행**.
   - 커널 내부(레이어×24): Pre-LN → **INT8 GEMM(QKV)** → KV 캐시(온칩+HBM) → **INT8 어텐션(QKᵀ→softmax(FP)→INT8→PV)** → INT8 GEMM(O) → 잔차+LN → INT8 GEMM(MLP1)+GELU → INT8 GEMM(MLP2) → 잔차. 단계 경계마다 **requant(INT32→FP scale·bias→INT8 클램프)**. CU 간 부분합은 **링 라우터**로 교환.
6. 호스트가 마지막 hidden state 회수 → final LN + lm_head logits + temperature/penalty/top-k 샘플링 → 다음 토큰 id.
7. (GUI) client-v3가 토큰 단어를 websocket으로 web/index.html(Vue3+Quasar)에 스트리밍, ack 핸드셰이크로 흐름 제어, latency 표시.

**데이터타입/메모리 계층**
- 데이터타입: 가중치/활성 행렬곱 = **INT8**, 누적 = **INT32**, 스케일·bias·LN·임베딩·logits = **FP32**, softmax 합산 = **ap_fixed**, 패킹 통신 = `ap_uint<...>`(union으로 FP↔int 비트 전송).
- 메모리: **HBM**(가중치 PE별 뱅크 + KV 캐시, connectivity로 HBM[0..7]/[16..23]) → **URAM**(bias/LN onboard) + **BRAM**(linear_alpha, 깊은 FIFO) → **SRL FIFO**(스트림) → **DSP**(MAC). 호스트↔커널 입출력은 PLRAM(`host_addr`).

---

## 5. HW/SW 매핑

| 레이어 | 구성요소 | 역할 |
| :-- | :-- | :-- |
| 설정 | `OPT-1.3b.toml/.json` | 병렬도·모델 차원 SSOT |
| 코드생성(SW) | `codegen.py` + `template/*` | 설정 → Vitis HLS C++ 자동 생성 |
| 데이터준비(SW) | `weight_packer.py` | INT8 가중치/FP 스케일 패킹·순환 슬라이싱 |
| HW(HLS) | `loopLynx.cpp`(생성) ← top/gemm_quant/attention/adder_tree/router/layerNorm/kernels | INT8 데이터플로우 디코더, 멀티 CU 링 |
| 매핑 | `connectivity.cfg` | CU↔SLR↔HBM 뱅크, stream_connect 링 |
| 호스트(SW) | `xrt_py_new.py`(pyxrt) / `host.cpp`,`tokenizer_predict_*.cpp`(C++) | xclbin 로드·BO·실행·샘플링 |
| 서비스(SW) | `client-v3.py`(websocket) | 추론 ↔ GUI 브리지 |
| GUI(SW) | `web/index.html`,`app.js`(Vue3+Quasar) | 실시간 채팅 데모, fpga/gpu/cpu 아바타 |
| 디바이스 | Alveo U50lv (2 CU = SLR0+SLR1) | 멀티노드 협력 추론 |

`connectivity.cfg`(OPT-1.3b_optimize) 근거: `slr=loopLynx_0_1:SLR0`, `slr=loopLynx_1_1:SLR1`(L3–L4), 양방향 `stream_connect ...:64`(L7–L8), CU0=HBM[0..7]/CU1=HBM[16..23](L11–L29). 즉 **"멀티노드"는 단일 U50lv 내 2-SLR 2-CU 링**으로 구현(보드 간 통신은 본 repo 코드 범위 밖, 확인 불가).

---

## 6. 빌드·실행 방법 (README 근거)

1. **가중치 준비**: 사전 패킹된 OPT-1.3B 가중치 다운로드(README L62–L64) 또는 `weight_packer.py` 실행.
2. **(선택) 코드 재생성**: toml 수정 후 `python codegen.py` → `OPT-1.3b/` 갱신.
3. **합성·프로그래밍**: `cd OPT-1.3b_optimize/ && make run` → xclbin 생성 + Alveo 프로그래밍(README L69–L72).
4. **벤치마크**: `cd tokenizer/ && sh ./command.sh` (lambada; `tokenizer_predict_eigen.cpp`로 패킹 적재 검증) (README L74–L81).
5. **웹 데모**: `cd LLM-demo-gui/alveo && (python==3.6) python client-v3.py` → 브라우저로 `LLM-demo-gui/llm-gui/web/index.html` 열기(README L83–L94). 서버는 ws://localhost:10088.

---

## 7. 의존성

- **HW/툴체인**: Vitis HLS & Vivado 2023.2, XRT 2023.2, Alveo U50lv 셸 `xilinx-u50lv-gen3x4-xdma-base_2`, Ubuntu 18.04 (README L33–L40).
- **HLS 라이브러리**: `ap_int/ap_fixed/ap_axi_sdata`, `hls_stream`, `hls_math`, `hls_half`(params.h include 블록, codegen L82–L84).
- **Python 호스트**: `pyxrt`(`/opt/xilinx/xrt/python`), `numpy`, `transformers.GPT2Tokenizer`, `websockets`, `asyncio`, `toml`(codegen). **python==3.6** 권장(README L86; `__pycache__/*.cpython-36.pyc`로 교차 확인).
- **C++ 벤치**: nlohmann `json.hpp`, Eigen(파일명상, eigen 변형), CMake/Make.
- **프론트엔드**: Vue3 global build + Quasar UMD(`web/cdn/*`), Roboto. **node_modules/React 없음**(CDN 로컬 번들).

---

## 8. 강점 / 한계 / 리스크

**강점**
- 설정 1파일(toml) 변경으로 병렬도(PROCESSOR/INP_NUM/CU/ROUTE_NUM/HEAD_PARALLEL)를 바꿔 **HLS 소스를 자동 재생성** → 빠른 DSE·이식.
- 전 파이프라인 INT8 + 단계별 requant로 DSP 효율·HBM 대역폭 절약. 곱셈을 DSP에 명시 매핑.
- KV 캐시 온칩/HBM 이원화, URAM 상주 스케일/LN, SRL FIFO 스트리밍으로 **호스트 개입 최소화 end-to-end 레이어**.
- 멀티 CU 링(SLR0↔SLR1)으로 단일 칩 내 협력 추론 + 깔끔한 websocket 데모.

**한계 / 리스크**
- codegen은 **단순 문자열 치환**이라 토큰 누락/순서·타입 불일치 시 합성 단계에서야 실패(정적 검증 부재). 주석 처리된 분기(예: top.cpp L324–L329에서 `ring_connection_str`을 조건 무시하고 재대입)로 **단일 CU 경로 미검증 가능성**.
- OPT-1.3B/U50lv·2CU에 **하드코딩 강함**(weight_packer 상수, kernel_names, HBM 뱅크). 다른 모델/디바이스 이식 시 다수 수정 필요.
- 정확도는 **외부에서 사전 양자화된 npz**에 의존(양자화 학습 코드는 repo 외, 확인 불가). `Gelu_layer`가 본문에선 round만 수행 → 실제 GELU 위치는 추정.
- python==3.6, XRT 2023.2 등 환경 고정성이 강함. 가중치는 baidu pan 외부 의존.
- 멀티"노드"의 실제 다중 보드/네트워크 코드는 본 repo에 없음(단일 칩 2-SLR로 시연). 보드 간 확장성은 논문 영역, 코드 확인 불가.

---

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적) 관점 시사점

1. **codegen 기반 HW 생성 재사용**: TeraFly의 "toml → params.h(파생 상수) + 병렬 인스턴스 펼치기 + 조건부 라우터/어댑터 삽입" 패턴은, HG-PIPE형 ViT 가속기에서 **패치/헤드/타일 병렬도와 비트폭을 설정으로 스윕**하는 DSE 프레임으로 거의 그대로 차용 가능. 특히 §3.1의 `lcm/GCD`로 PE폭↔라우터폭 리패킹 라운드를 자동 산출하고, **비트폭 누적 typedef를 동적 생성**하는 기법은 가산 트리 오버플로 안전성을 코드생성 단계에서 보장하는 좋은 레시피.
2. **XRT Python 호스트 패턴**: `xrt_py_new.py`의 *device→xclbin→kernel→BO(group_id별 HBM)→sync→비동기 run→wait→frombuffer* 흐름과 "**상수 적재 모드 kernel 호출(load_weight_layer≥0)로 온칩 파라미터 prefill**" 트릭은, ViT 가속기에서 가중치/포지셔널 임베딩을 1회 적재 후 프레임마다 활성만 갱신하는 **실시간 시선추적 추론 루프**의 표준 호스트 골격으로 적합. fork-join 멀티 CU 동기화도 그대로 응용.
3. **가중치 패킹 재사용**: weight_packer의 **순환 슬라이싱(PE 라운드로빈) + PE별 HBM 뱅크 분리 + INT8 본체/FP 스케일 분리 직렬화**는 ViT의 QKV/MLP 가중치에 동일하게 적용 가능. 스케일을 C++ `const float[]` 헤더로 뽑아 `#include`로 흡수하는 방식은 스케일이 적은 ViT(레이어 수 적음)에서 합성 최적화에 유리.
4. **단계별 requant + 수치안정 softmax**: §3.4 requant(INT32·FP scale + bias → 클램프 → INT8)와 §3.6 max-shift softmax(고정소수 합산, ×127 재양자화)는 ViT의 어텐션·MLP 경계에 바로 이식 가능한 검증된 정밀 분할. XR용 저지연 요구에 맞춰 softmax를 streaming 4-패스로 유지하는 점도 참고.
5. **멀티 CU 링 라우터 아이디어**: 토큰 차원을 CU로 분할하고 부분합을 시프트-링으로 합치는 §3.7 구조는, **시선추적+장면이해를 두 개의 SLR(또는 두 가속기)에 분산**시키고 중간 특징을 AXIS 링으로 교환하는 멀티태스크 XR 파이프라인 설계의 출발점이 될 수 있음.
6. **실시간 데모 GUI 아이디어**: message-format.md의 **ack 기반 토큰 스트리밍 프로토콜**(responseStart/response/responseEnd + acknowledge)과 fpga/gpu/cpu **아바타로 추론 디바이스를 시각 구분**하는 UX, websocket+Vue3(CDN 로컬 번들로 빌드 의존성 0)는, XR 시선추적 가속기의 **라이브 데모(예: 시선 히트맵/예측 좌표를 토큰 스트림처럼 push)**에 그대로 재현 가능. 토큰당 latency를 GUI에 표시하는 것도 데모 설득력에 유효.

---

## 10. 근거 / 한계 표기

- **확인된 사실(코드 라인 직접 근거)**: 디렉토리 구조, codegen 치환 로직, weight_packer 양자화·슬라이싱, top/gemm/attention/router/layerNorm/adder_tree HLS 데이터플로우, xrt_py_new pyxrt 호스트 흐름, connectivity.cfg의 2-CU/SLR/HBM 매핑, websocket 프로토콜, toml/json 설정값 — 모두 본문에 파일·라인 명시.
- **HLS/RTL 존재 여부**: **HLS C++ 명백히 존재**(template/ + 생성된 loopLynx.cpp). **RTL(.v/.sv) 없음**(Glob 0건). 프론트엔드는 React/node_modules가 아니라 Vue3+Quasar CDN 번들, 백엔드는 순수 Python(node.js 없음).
- **추정(코드만으론 단정 불가)**:
  - "멀티노드"가 단일 U50lv 2-SLR/2-CU 링으로 시연되며, **물리적 다중 보드/네트워크 통신 코드는 본 repo 범위 밖**(논문 영역) — *추정/확인 불가*.
  - 실제 GELU 비선형의 정확한 위치(kernels.cpp `Gelu_layer`는 round만 수행) — *추정*.
  - 양자화(스케일 산출) 학습 파이프라인은 `opt-1.3b.npz`로 외부 제공, repo 내 없음 — *확인 불가*.
  - DSE를 toml 재생성으로 수행한다는 워크플로우 — README·codegen 구조 기반 *추정*.
- **제외 처리**: `*.xclbin/*.link.xclbin/*.ltx/*.info`(합성 산출물), `inputs_hardware_full/*.bin`·`w*_addr_*.bin`·`const_data_*.bin`(데이터), `*.pyc`, `web/cdn/*`(프론트 번들), `.git/`는 이름만 언급하고 내용 분석에서 제외.
