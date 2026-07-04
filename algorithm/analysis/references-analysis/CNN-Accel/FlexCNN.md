# FlexCNN 정밀 분석

> 분석 대상: `REF/CNN-Accel/FlexCNN` (UCLA-VAST)
> 분석 방식: 실제 소스 Read 기반, 라인 근거 표기. 외부 포함물(3rdparty/darknet/bitstream/.Xil 등)·중복본(`* copy`, `auto_compile_old`, `HLS copy`, `TAPA_1 copy`, 모델전용 `*_inst_gen.py`)은 제외하고 FlexCNN 자체 분만 분석.
> 표기 규칙: 라인 근거 없는 추론은 "추정", 코드로 확인 불가한 것은 "확인 불가"로 명시.

---

## 1. 개요

- **목적**: CNN을 FPGA에서 실행하는 **엔드투엔드 가속기 합성 프레임워크**. ONNX 모델을 입력받아 → 그래프 추출 → DSE(설계공간탐색) → 명령어(instruction) 생성 → HLS/TAPA C++ 디자인 생성까지 자동화한다. (README.md:8-12)
- **한줄요약**: "ONNX CNN → 명령어 기반 재구성형(reconfigurable) HLS 가속기"를 자동 컴파일하며, 그 핵심 연산기로 **PolySA/AutoSA 계열 자동생성 systolic array**를 사용하는 프레임워크.
- **원논문**:
  - Basalama, Sohrabizadeh, Wang, Guo, Cong. **"FlexCNN: An End-to-End Framework for Composing CNN Accelerators on FPGA"**, ACM TRETS 2022. (README.md:4)
  - Sohrabizadeh, Wang, Cong. **"End-to-End Optimization of Deep Learning Applications"**, FPGA 2020. (README.md:6)
- **타깃 디바이스**: Xilinx Alveo **U250/U280** Data Center 카드, Vitis 2021.2, TAPA HLS. (README.md:28-37) 보드 스펙은 `auto_compile/data/boards/*.json`에 U200/U250/U280/U50 정의. (Glob 결과)
- **검증된 네트워크**: U-Net, E-Net(ENet), VGG16. (README.md:12)
- **특징**: ① 명령어 기반(layer_config) — 비트스트림 1개로 여러 레이어/네트워크 실행. ② 다양한 conv 타입(NConv/DepthWise/Dilated/Transposed) 지원. ③ float32 및 ap_fixed<16,8>/<8,4> 양자화 선택. ④ TensorFlow custom op로 통합 가능(libsacc).

---

## 2. 디렉토리 구조 (자체 소스 + 제외)

### 분석 포함 (FlexCNN 자체 핵심)
```
FlexCNN/
├─ README.md, generate_design.sh, env.sh        # 엔트리/환경
├─ HLS_Codes/                                    # ★ 손수 작성한 레퍼런스 HLS 커널(데이터플로우 백본)
│  ├─ kernel.cpp (5695줄)                        # cin_load/weight_load/depth_conv/conv/relu/add/pool/upsample/cout_write/engine/top_kernel
│  ├─ util.h                                     # 데이터타입·매크로·stencil/maxpool/upsample 템플릿
│  ├─ params.h, cnn_sw.cpp, tb_pose.cpp          # 파라미터/소프트웨어 레퍼런스/테스트벤치
│  ├─ hls_script.tcl, design_prepare.sh
│  └─ systolic_array_kernel/                     # SA 코드젠(레거시 위치)
│     ├─ codegen.py, code_template.py, desp_gen.py, cnn_features.json
├─ auto_compile/                                 # ★ 엔드투엔드 컴파일러
│  ├─ run.sh, graph_translation/{extract_info,mapper}.py
│  ├─ dse/{dse,latency_est,resource_est,utils}.py
│  ├─ inst_generation/inst_gen.py
│  ├─ design_generation/{generate.py, set_headers_mem_*.py,
│  │     HLS/, TAPA_1/, TAPA_2/}                 # 코드 타깃(HLS=Vitis, TAPA=P&R 최적)
│  │     └─ */systolic_array_kernel/{codegen,code_template,desp_gen,h_to_json}.py
│  ├─ data_generation/{generate_data,generate_weights,generate_biases}.py
│  └─ data/{boards/*.json, arch_connectivity/*.json,
│           pre_dse_architectures/*.json}
├─ libsacc/                                      # ★ TensorFlow custom op 통합 라이브러리(SW 호스트)
│  ├─ src/libsacc.cpp, inc/sacc.hpp, inc/sacc_*.h
│  └─ config/, data/ (가중치/명령어/비트스트림 배치)
├─ SDx_project/                                  # SDAccel 호스트 프로젝트(host.cpp, hw_kernel.cpp)
└─ tf_DSA/                                       # TensorFlow + tf-pose(OpenPose) 통합 데모(외부 비중 큼)
```

### 제외(이유)
- `auto_compile/.../* copy`, `HLS copy`, `TAPA_1 copy`, `generate_data copy*.py` — 작업용 중복본. (Glob 결과 다수)
- `auto_compile/auto_compile_old/` — 구버전 protobuf 기반 파이프라인.
- `inst_generation/{VGG16,UNet,ENet}_inst_gen.py`, `inst_gen_old.py`, `ENet_inst_gen_16.py` — 모델 전용 변형(정식 일반본 `inst_gen.py`만 분석).
- `tf_DSA/tf_pose/`, `data/*.dat`, `*.xclbin`, `*.exe`, `*.pyc`, `.git/` — 외부 프레임워크/산출물/바이너리.
- `SDx_project/src/xcl2.*` — Xilinx 제공 OpenCL 유틸.

---

## 3. 핵심 모듈·파일별 정밀 분석

FlexCNN에는 **두 갈래의 HLS 백본**이 공존한다.
1. **수작업 데이터플로우 커널** (`HLS_Codes/kernel.cpp`) — 모든 레이어 타입을 FIFO로 연결한 single `#pragma HLS DATAFLOW` 파이프라인. conv 부분만 `kernel()`(SA)로 위임.
2. **자동생성 systolic array** (`*/systolic_array_kernel/` 의 Python 코드젠 + 생성된 `2DPE/2DDataFeed/...`) — PolySA/AutoSA 계열로, conv를 systolic array로 구현.

### 3.1 HLS 데이터플로우 백본 — `HLS_Codes/kernel.cpp`

#### `top_kernel` (extern "C") — AXI 인터페이스 & 레이어 루프 (kernel.cpp:5646-5694)
- 포트: `global_cin / global_prev_cin / global_cout`(bus_t0=512b), `global_weight`(bus_t1), `global_bias`(bus_t2), `layer_config`(bus_t3=32b). 각각 `m_axi ... bundle=gmem1/2/3/gcontrol` + `s_axilite bundle=control`. (5655-5668)
- `layer_config[0]`=레이어 수, 이후 `5 + CONFIG_PARAMS*layer_id` 위치에서 레이어별 config를 memcpy해 가져옴. (5672-5685)
- **명령어 기반 실행**: `while(layer_id < layer_num)` 루프가 레이어마다 `engine(...)`을 호출. `config[26-1]`(=cur_layer_batch)와 `config[..+26-1]`(=nxt_layer_batch)로 **layer batching**(여러 레이어를 한 engine 호출에 묶음) 지원. (5680-5691)
- 의미: **하나의 비트스트림 + 명령어 스트림**으로 임의 깊이의 CNN을 실행 → 재구성형(overlay) 아키텍처.

#### `engine` — 데이터플로우 파이프라인 정의 (kernel.cpp:5344-5643)
- `#pragma HLS DATAFLOW`(5352)로 모든 모듈을 FIFO 연결. 데이터 FIFO와 **config FIFO(ConfigInst)**를 모듈마다 분리해, 명령어가 파이프라인을 따라 전파됨. (5358-5486)
- 파이프라인 연결 순서 (5491-5638):
  `cin_load → cin_load_prev → weight_load → depth_conv → relu6 → conv → relu → add → upsample → merge_upsample → cout_write`
- 활성화/비활성 모듈은 주석 처리(inter_load, pool, inter_write는 주석; 필요시 켤 수 있다고 README형 주석 명시 5340-5342).
- BN용 gamma/beta는 별도 FIFO(`fifo_beta_depth/conv`, `fifo_gamma_depth/conv`)로 weight_load→relu6/relu에 직접 전달. (5449-5456)
- 명명 규칙 주석: 데이터타입 `<module>+Data+<port>+Type`, FIFO `fifo+<module>+<port>`. (5354-5356)

#### `cin_load` — 입력 특징맵 로드 + 명령어 디코드 (kernel.cpp:419-685)
- 입력: `global_cin`(DRAM), `config[CONFIG_PARAMS]`. 출력: `fifo_cin`, `fifo_config_out`.
- **온칩 더블버퍼**: `cin_burst_buf_ping/pong`을 **URAM**에 할당(`XPM_MEMORY uram`, 427-430). 크기 `IN_NUM_T*(IN_H_T+K_T-1)*(IN_W_T+K_T-1)/BUS_PACK_FACTOR0`.
- **명령어 디코드(핵심)**: 32워드 config를 5개 inst 그룹으로 해석. inst0=HW사이즈, inst1=실제사이즈, inst2=DRAM오프셋+필터/스트라이드, inst3=`LAYER_EN`(모듈 enable 비트)+타일사이즈, inst4=SA 카운터. (516-568) — util.h:64-68의 비트필드 정의와 일치.
- `LAYER_EN` 비트맵: bit0=CONV_1ST, 1=DEPTH_CONV, 2=CONV, 3=RELU, 4=RELU6, 5=POOL, 6=UP_SAMPLE, 7=BIAS, 8=INTER_LOAD, 9=INTER_WRITE, 10=BATCH_NORM, 11=LOAD_PREV_CIN, 12=BATCH_NORM_DEPTH. (548-560)
- **두 가지 데이터 레이아웃**(`cin_load_ddr_read`, 27-80):
  - 전체 패치가 온칩에 들어가면 통째로 memcpy(46-48).
  - 그렇지 않으면 `change_layout && FILTER_S==1`이면 `[num_tile][Th][Tw][Tn]` 타일 레이아웃(54-64), 아니면 패딩 레이아웃 `[h][Tw+F-1][Tn]`을 행 단위로 burst memcpy(65-77). 이 layout 결정이 DSE의 `IN_DATA_LAYOUT`(1 vs 2)과 짝.
- **double buffering**: `task_cnt%2`로 ping/pong 교대, 한쪽 DRAM read 중 다른쪽을 FIFO write(`cin_load_fifo_write`)로 오버랩. (610-633, 679-683)
- `cin_load_fifo_write` (86-247): 버퍼를 `[Th][Tw][Tn]` 순회하며 `DATA_SEL_FACTOR0`(=BUS/SIMD)에 따라 512b 버스 워드에서 SIMD_LANE(8개) 단위로 언팩해 FIFO에 write. (101-215)

#### `cin_load_prev` — 잔차(residual) 입력 로드 (kernel.cpp:692-911)
- `add` 모듈에서 더할 이전 레이어 출력을 DRAM(`global_prev_cin`)에서 로드. `LOAD_PREV_CIN`(LAYER_EN[11]) 켜졌을 때만 동작. (845-859) → MobileNetV2 residual bottleneck 등 지원. (`add` 주석 3258-3259)
- 별도 URAM 더블버퍼 `prev_cin_burst_buf_ping/pong`(700-703). config FIFO를 그대로 통과시켜 downstream에 전달(727-736).

#### `weight_load` 및 하위 write 함수들 (kernel.cpp:918-1951)
- `weight_load_depth_conv_weight_write` (918~): depth_conv용 가중치를 `[Tn/LANE][LANE][F][F]` 레이아웃으로 FIFO에 write, `DATA_SEL_FACTOR1`로 버스 언팩. (967-1065+)
- `weight_load_conv_weight_write` (1116~): point/일반 conv용 가중치.
- `weight_load_bias_write` (1313~), `weight_load_depth_norm_write` (1489~): BN의 gamma/beta(bias)를 conv/depth 경로로 분배.
- `weight_load` (1667~): 위 write들을 묶어 depth_conv weight, conv weight, gamma/beta(depth·conv)를 각 FIFO로 디스패치.

#### `depth_conv` — DepthWise Conv (stencil) (kernel.cpp:2173-2382)
- `DEPTH_CONV_EN==0`이면 입력을 그대로 통과(bypass, 2279-2306). `==1`이면 가중치 로드 후 `stencil_w1`/`stencil_w3` 호출. (2308-2352)
- 가중치 버퍼 `weight_buf[IN_NUM_T/LANE][LANE][K_T][K_T]`를 dim2/3/4 complete 파티션. (2267-2270)
- **필터에 따른 타일 확장**: F=3이면 `IN_H_T+2, IN_W_T+2`로 호출 — stencil halo 반영. (2346-2350)

#### stencil 템플릿 — `util.h`
- **`stencil_w3`** (util.h:178-460): 3×3 depthwise conv를 **line buffer 기반 stencil**으로 II=1 파이프라인. `line_buf1/2/3`(행 버퍼)을 시프트하며 9개 곱(383-391)→가산 트리(394-401)로 누적. stride/타일 경계는 `col_skip/row_skip/col_strip_skip/row_strip_skip`로 출력 마스킹(408-449). `T_UNROLL`(=DEPTH_CONV_LANE=8)만큼 채널 병렬. **`assert(layer_in_w_t==26||50||98)`**(211) — 동적 타일링이 일부 width값만 합성 지원(하드코딩 mux 235-250).
- **`stencil_w1`** (util.h:468-696): 1×1 depthwise. 단일 곱(641-642).
- **`upsample_w2`** (util.h:704-1038): bilinear 2× 업샘플. 짝수행 `0.5/0.25` 가중 평균(912-913), 홀수행 별도 line buffer(`line_buf_inp`)로 처리(920-921). 출력 2개 FIFO(even/odd 행 분리)로 write. (982, 1023)
- **`maxpool_w2`** (util.h:1046-1278): 2×2 max pooling. `max()` 매크로(155)로 4개 비교(1224-1226). `max_en` 꺼지면 패스스루.

#### `conv` — 일반 Conv → systolic array 위임 (kernel.cpp:3045-3254)
- `CONV_EN==0`이면 maxpool/upsample 경로용 패스스루(3116-3238). `==1`이면 **`kernel(fifo_cin, fifo_weight, fifo_cout, fifo_config_in, fifo_config_out)` 호출**(3248) — 이것이 자동생성 systolic array의 진입점(util.h:159-165에 선언).
- 주석(3244-3247): "Calls systolic array. 직접 구현으로 교체 가능. 단순 검증용 naive `kernel`도 파일에 주석으로 존재".
- 주석된 `conv_core`(2388-2419)는 6중 루프 naive conv 레퍼런스(`cout += cin[...]*weight[...]`).

#### `add` — 잔차 가산 (kernel.cpp:3261~)
- `fifo_cin`(prev) + `fifo_conv`(현재 레이어 결과)를 element-wise 합산. residual block 지원.

#### `relu` / `relu6` — 활성화 + Batch Normalization (kernel.cpp:2744(relu6), 3537(relu))
- **둘 다 ReLU/ReLU6/BN/bias를 모두 내장**, 차이는 파이프라인 내 위치뿐(relu6는 depth_conv 뒤, relu는 conv 뒤). (engine 주석 5542-5544)
- `relu`: `beta_buf/gamma_buf[OUT_NUM_T/LANE][LANE]`를 dim2 complete 파티션(3629-3632). `en = RELU_EN||BIAS_EN||RELU6_EN||BATCH_NORM_EN`(3644). BN/bias가 없으면 gamma/beta FIFO를 읽지 않음(`norm_conv_en` 가드, 3670-3673). 입력 채널 타일 마지막(`in_num_iter+IN_NUM_T>=IN_NUM`)에 BN/활성화 적용(3694~) — 누적 완료 시점에만 후처리.
- 계산식(추정, 일반적 BN+ReLU): `out = relu(gamma*x + beta)`. 코드 구조상 gamma/beta가 conv 채널별로 buf에 로드되어 적용됨(3629-3636).

#### `pool` / `upsample` / `merge_upsample` (kernel.cpp:3832 / 3998 / 4172)
- `pool`(3832~): `maxpool_w2` 래핑(현재 engine에서는 주석 처리, add→upsample 직결 5588-5606).
- `upsample`(3998~): `upsample_w2` 호출, 출력 2개(even/odd 행) FIFO.
- `merge_upsample`(4172~): 2개 FIFO를 인터리브해 하나의 출력 스트림으로 병합. (ENet의 디코더 업샘플 경로)

#### `cout_write` — 결과 DRAM 기록 (kernel.cpp:4630 / 4852 / 4967)
- `cout_write_fifo_read`(4630~): FIFO에서 결과 수집. `cout_write_ddr_write`(4852~): 512b 버스로 패킹해 `global_cout`에 burst write. `cout_write`(4967~): 둘을 묶음 + config 통과.

### 3.2 자동생성 systolic array — Python 코드젠 (`*/systolic_array_kernel/`)

이 부분이 FlexCNN의 conv 가속 핵심이며, **PolySA/AutoSA(Jie Wang) 계열** 코드젠이다(생성 파일 헤더 "automatically generated by PolySA CodeGen, Author: Jie Wang", 2DPE_U1.cpp:1-5).

#### `codegen.py` — 코드 생성 드라이버 (codegen.py:11-141)
- `run(input_vsa, mode)`: `desp_gen.run()`로 design descriptor(`output/design_desp.json`) 생성(99) → 그로부터 6개 파일 생성:
  - `tb_app.cpp`(테스트벤치), `top.cpp`(top), `common_header_U{id}.h`, `2DPE_U{id}.cpp`(PE), `2DDataFeed_U{id}.cpp`(피더), `2DDataCollect/DataFeedCollect`(콜렉터/로더). (112-130)
- 각 generate 함수는 `code_template`(tpl)의 코드 조각을 이어붙임. PE는 `PE_MAC + op_transfer + compute + res_transfer + kernel` 순. (44-53)

#### `desp_gen.py` — Virtual Systolic Array(VSA) descriptor 생성 (desp_gen.py)
- **다중 애플리케이션 지원**: `cnn_pass`(conv, 14~), `mm_pass`(matrix-matrix, 323~), `mv_pass`(matrix-vector, 566~), `nw_pass`(Needleman-Wunsch, 769~)를 모두 보유 → 이 코드젠은 **CNN 전용이 아니라 범용 systolic array 합성기**. FlexCNN은 `cnn_pass`를 사용.
- `cnn_pass` 산출물(SA 실행 카운터·버퍼·인덱싱 코드):
  - `LOCAL_REG_NUM = OUT_IMG_H_T * ROW_IL_FACTOR * COL_IL_FACTOR`(16) — 각 PE의 출력 누산 레지스터 수(output-stationary 증거).
  - `LOCAL_ACCUM_NUM = IN_NUM_T * K * K / SIMD_FACTOR`(18) — 한 출력 타일을 위한 reduction 길이.
  - `MAC_STAT = 'sum += op0_u[i] * op1_u[i];'`(23) — PE 내부 MAC 식.
  - `DFC_BUF_SIZE`(cin/weight/cout 피더 버퍼, 28-32), `DF_FEED_COUNTER`(피드용 6중 카운터 c0~c5: OUT_IMG_H_T, ROW_IL, COL_IL, K, K, IN_NUM_T/SIMD)(47-95), `DC_COLLECT_COUNTER`(수집용 4중)(98-130).
  - `DF_FEED_ADDR_CAL_CODE` / `DC_COLLECT_ADDR_CAL_CODE`(168-177): cin/weight를 PE 어레이로 공급하고 cout을 회수하는 BRAM 주소식.
  - `HEAD_CODE`(180~321): A(cin)/B(weight)/C(cout) 각각에 대해 DRAM↔온칩 더블레벨 타일 루프(memcpy + II=1 피드 루프) 생성. `cal_width`(7-12)로 각 루프 카운터 비트폭 자동 결정.
- 즉 desp_gen은 **타일 크기·IL factor·SIMD로 매개화된 메모리/제어 코드를 수식으로 합성**한다.

#### 생성된 PE — `2DPE_U1.cpp` (예시, TAPA_2)
- **`U1_PE_MAC`** (2DPE_U1.cpp:9-51): SIMD MAC. op0/op1(각 128b=float×4)을 SIMD_FACTOR(4)개로 언팩(26-34) → 4-way 곱(38-41) → 가산 트리(43-48) → `init`이면 0에서, 아니면 `*op2`에서 누적(36,50). **output-stationary**(부분합이 PE 로컬 `op2`에 머묾).
- **`U1_op0_transfer` / `U1_op1_transfer`** (53-771): op0(cin)을 어레이를 따라 전달(`fifo0_out`)하면서 로컬 복제(`fifo0_local`)에도 흘림 — op0는 **Down 방향**, op1(weight)은 **Right 방향**(design_desp `OP_CHANNEL_DIR=["D","R"]`, design_desp.json:91-94). `_last` 변형은 어레이 끝 PE용(out 없이 local만). config(레이어 파라미터 24개)를 FIFO로 통과시키며 task/accum/reg 카운터로 데이터 페어 수 제어(181-203).
- **`U1_compute`** (773-961): op0_local·op1_local을 읽어 `U1_PE_MAC` 호출. **가변 conv 지원 핵심**: `LAYER_LOCAL_ACCUM_NUM_ARR[4]`를 `in_ch_factor*KH[i]*KW[i]`로 구성(851-857)하고 `K_NUM`개의 sub-kernel을 순회(927-947) → transposed conv를 여러 sub-kernel로 분해해 동일 어레이로 처리(inst_gen의 conv_type 인코딩과 짝). `init`는 `new_pair && la_counter==0 && i<K_NUM`(926), 결과는 `la_counter==ACCUM-1 && last`일 때 `fifo2_local`로 방출(927-929).
- **`U1_res_transfer`** (979~): 결과(op2)를 **Down 방향**으로 회수(`RES_CHANNEL_DIR=["D"]`, design_desp.json:95-97), `pe_row_id/pe_col_id`로 위치 인식.
- 즉 구조: **2D output-stationary systolic array** (행=OUT_NUM 분할 SA_ROWS, 열=OUT_IMG_W 분할 SA_COLS, 각 PE는 SIMD_FACTOR 폭 reduction). design_desp 예시: SA_ROWS=4, SA_COLS=6, SIMD_FACTOR=4. (design_desp.json:89-90,113)

#### `design_desp.json` (생성 descriptor 예시) (design_desp.json)
- conv를 `out_num/out_img_h/out_img_w/in_num/p/q` 6중 루프로 표현, 앞 4개를 타일링(TILE_FACTOR), p/q(=K)는 타일링 안 함(20-87).
- `OP_CHANNEL_DIR=["D","R"]`, `RES_CHANNEL_DIR=["D"]`, `OP_PE_SIMD_WIDTH=[128,128]`(=float×4×32b), `OP_ENGINE_NUM=[6,4]`(=SA_COLS,SA_ROWS 피더 수). (91-97,160-170)
- `SW_KERNEL_CODE`: 검증용 naive conv 레퍼런스(276-293).

### 3.3 엔드투엔드 컴파일러 — `auto_compile/`
*(서브분석가 정밀 조사 결과를 라인근거와 함께 통합)*

#### 엔트리 — `generate_design.sh` + `auto_compile/run.sh`
- `generate_design.sh`: `echo 13 | run.sh ENet ENet 32 u250 4 TAPA_1` 식 호출. `echo 13`은 DSE의 `input('design id')`에 파이프로 주입. (README.md:57, generate_design.sh)
- `run.sh` 인자: `$1=모델, $2=prefix, $3=dw(데이터폭), $4=board, $5=mem_type(0~4, BRAM/URAM 배치), $6=code(HLS|TAPA_1), $7/$8=옵션`. 6단계 순차: ① graph translation ② DSE ③ inst generation ④ design generation ⑤ weights/biases ⑥ data(dw>16일 때만). design_name=`{code}_{prefix}_{ROWS}_{COLS}_{SIMD}_MEM_{mem}_{dw}`.

#### `graph_translation/extract_info.py` (클래스 `PBPredictor`)
- ONNX→networkx 변환(`onnxToNetworkx`, 124-146)에서 노드 `visited`를 출력 엣지 수로 초기화 → **fan-out 추적**.
- **op→conv타입 매핑(get_conv_info, 299-370)**: `group==1 & dilation==1`→**NConv**, `group==1 & dilation>1`→**DConv(dilated)**, `group>1`→**DWConv**, `ConvTranspose`→**TConv**(in/out 채널 swap). 반환: kernel_h/w, in/out_ch, stride, tstride, dilation, bias_en.
- **레이어 융합(get_insts, 495-715)**: conv를 main op로 잡고 `arch.json`의 모듈 순서대로 전(BN/Pad)·후(Relu/Pool/Add) op를 한 inst로 묶음. `visited` 감소로 중복 소비 방지, fan-out(>1)에서 융합 중단(614-615).
- **enable 비트(get_enabled_modules, 717-738)**: op 시퀀스를 modules_order와 매칭한 0/1 벡터.
- 출력: `pre_dse_models/{model}.csv` (각 행=inst dict: en, in/out_num, downsample/upsample factor, main/secondary input, ops).

#### `graph_translation/mapper.py` (클래스 `Mapper`) — 별도 도구
- ONNX로부터 pre_dse arch(모듈 순서/연결)를 자동 탐색. op→모듈 매핑: `Transpose`→DRAM, `ConvTranspose`→Conv 통합, `GlobalAveragePool/MatMul/Softmax/Squeeze`는 unsupported로 스킵(405-481). descendant 시퀀스 빈도(Counter)로 자주 쓰는 모듈 순열을 path 후보화, double-input 모듈 기준 path1/path2/common 분할 후 bundle 수 최소 아키텍처 선택(`algorithm`, 281-383).

#### `dse/` — Design Space Exploration
- **`dse.py`**:
  - `get_layers_configs`(18-309): csv→hw config. `out=ceil(in*up/down)`, `MACS=in_ch*out_ch*kh*kw*out_h*out_w`(112), padding=`dilated_kernel - stride`(119-120). **`IN_DATA_LAYOUT`=2(no-pad 타일) if `(filter-stride==0)`||`(filter==tstride)` else 1(패딩)**(170) — kernel.cpp의 두 레이아웃과 직결. 레이어 간 타일 의존성(networkx connected_components)으로 IN/OUT_NUM_T·IN_W_T 공유 제약 구축(204-308).
  - 데이터 정밀도(414-437): dw=32→float; 16→`ap_fixed<16,8>`; 8→`ap_fixed<8,4>`. K_T=16.
  - 설계공간(445-486): 타일 후보 + lane∈{2..64} + SA_ROWS(OUT_NUM_T 약수) × SA_COLS(IN_W_T 약수) × SA_SIMD=LANE 곱집합. 멀티프로세스 sweep(497-508, CPU 90%).
  - Pareto 필터(513-573): latency 정렬 → 동일 latency·`DIM_SUM=ROWS+COLS+SIMD` 동일 그룹 내 BRAM 최소만 유지.
  - 디자인 선택(`input()`, 620-679)→`post_dse_architectures/{model}_arch.json` 저장, 마지막 줄에 `ROWS_COLS_SIMD` 출력(run.sh가 design_name에 사용).
  - `param_sweep`(730-824): pruning `IN_W_T%SA_COLS==0 && IN_NUM_T%SA_SIMD==0 && IN_NUM_T%SA_ROWS==0`(759), `DSP > DSP_THRES*보드DSP`면 제외(762), FRE=220MHz 고정(772).
- **`latency_est.py`** (analytical, 사이클):
  - DRAM 실효 대역폭 `eff = port_width*burst/(250+burst)`(20-24).
  - **conv 처리량**: `compute_phase = IN_NUM_T*OUT_NUM_T*OUT_H_T*OUT_W_T*K*K/(SA_ROWS*SA_COLS*SA_SIMD)`(conv_est:117).
  - `layer_latency = extra(파이프 채움/비움) + max(cin,weight,conv,cout)*total_iter`(layer_latency_est:217-219).
  - dynamic tiling 그리디 탐색(model_latency_est:247-407): 의존성 제약 하에서 타일별 layer_latency 최소 선택, 버퍼 초과 거름. 후보 비면 inf.
- **`resource_est.py`**:
  - DSP/MAC: float=5, ap_fixed<16,8>=1, ap_fixed<8,4>=0.25(26-31). **`DSP = SA_ROWS*SA_COLS*SA_SIMD * DSP_per_MAC`**(35-36).
  - BRAM18K: cin_load(×2 더블버퍼) + weight + point_conv 5버퍼 + cout_write(×2) 합(40-57). SDP BRAM 모델(`BRAM_SDP_predict_HLS`, 3-10).
- **`utils.py`**: dse_report.txt/detailed.csv 리포팅, ideal/opt latency·DSP/BRAM 사용률·SA 형상 기록.

#### `inst_generation/inst_gen.py` — 명령어 생성
- **`get_en_value`(110-164)**: en 비트마스크 합성 — `en = conv<<2 + relu<<3 + pool<<5 + upsample<<6 + bias<<7 + bn<<10 + prev<<11 + concat<<16 + add<<17`(163). op_type별 비트 set(143-162). (kernel.cpp LAYER_EN 디코드와 정확히 대응)
- **`get_SA_insts`(13-39)**: SA 실행 카운터. `task_num1=ceil(in/in_t)*ceil(out/out_t)*ceil(in_h/h_t)*ceil(in_w/w_t)`, `local_accum_num=in_t/simd*K*K`, `local_reg_num=(h_t/stride)*(w_t/cols/stride)*(out_t/rows)`, `row_il=out_t/rows`, `col_il=w_t/cols/stride`. (desp_gen `cnn_pass`·2DPE compute 카운터와 일치)
- **`calculate_addresses`(167-288)**: inst 그래프로 DRAM 주소 자동 할당. 출력주소=입력주소+`in_num_hw*in_h_hw*in_w_hw`+패딩오프셋.
- **`run`(290-590)**: weight/bias 포인터 누적, HW 차원 라운드업(`in_num_hw=max(ceil(in/simd)*simd, in_t)`), **conv_type 인코딩**(NConv=0, TConv=1, K_NUM=tstride² sub-kernel로 분해, k_h/k_w 4-워드)(501-534). inst당 7줄(53정수 + k_h/k_w)로 직렬화 → `insts/{design_name}_instructions.dat`.
- 이 7줄을 `h_to_json.py:get_mem_params`가 53필드 instDict로 복원 → 곧 HLS 커널의 CONFIG_PARAMS(32+워드) 실체.

#### `design_generation/generate.py`
- `h_to_json.py` 실행(design_params + mem_type + insts → `cnn_features.json` + `design/src/params.h`) → `design_prepare.sh` → `design/*`를 PRJ 경로의 `{design_name}/`로 복사. (8-43)
- `h_to_json.py`: `cnn_features.json`(AutoSA/TAPA 코드젠 입력: PARAMETERS, ITERATORS+TILE, SA_ROWS/COLS, SIMD_FACTOR, ROW/COL_IL_FACTOR)와 `params.h`(MAX 타일, 버퍼 크기, SIMD_LANE, LAYER_NUM, data_t typedef, `bus_t=tapa::vec_t<data_t0,512/dw>`, mem_type별 14개 온칩 메모리 배치 매크로) 생성.

#### `data_generation/{generate_data,generate_weights,generate_biases}.py`
- `generate_data.py`: onnxruntime로 모든 중간 텐서 추출(골든 레퍼런스), 입력을 **DL1(패딩 `[i1][h][w][i2]`)/DL2(no-pad 타일 `[i1][w1][h1][h2][w2][i2]`)** 레이아웃으로 reorg. (49-138)
- `generate_weights.py`: Conv `[O,I,kh,kw]→[O,kh,kw,I]`, TConv는 tstride·K별 하드코딩 순열 테이블로 reorder(49-73) 후 `[o1][i1][o2][p][q][i2]` 타일 레이아웃(in_num_t/out_num_t 정렬). zero-pad로 SA 차원 맞춤.

### 3.4 SW 호스트 / TF 통합 — `libsacc/`
- **`sacc.hpp`** (libsacc/inc/sacc.hpp): `REGISTER_OP("Sacc")`로 TensorFlow custom op 등록. 출력 shape=`{1, STAGE2L_OUT_H, STAGE2L_OUT_W, STAGE2R_OUT_NUM+STAGE2L_OUT_NUM}`(OpenPose stage 출력, 23-24). `class Sacc : public OpKernel`의 `Compute()`가 FPGA 호출 진입점(33). XRT/OpenCL 멤버(context/queue/buffer)와 cin/weight/bias/config 정적 벡터 보유(42-109).
- **`libsacc.cpp`**:
  - 파일 경로 상수: `inst_file_path=config/network.insts`, `weight=data/weight_reorg.dat`, `bias=data/bias_reorg.dat`, `xcl=config/binary_container_1.xclbin`. (29-37)
  - **3개 큐/커널 병렬**(`krnl_vadd/2/3` = 같은 `top_kernel` 3 인스턴스, 561-563): q/q2/q3로 멀티 DDR 뱅크에 입력 마이그레이션→`enqueueTask`→출력 회수를 오버랩(파이프라인/배치 처리). (113-244)
  - `reformat_input/output_data_layout` 3종(272~): TF 텐서 ↔ FPGA 레이아웃 변환.
  - `setArg`: arg0/1/2=buffer_cin(prev_cin/cout과 공유), 3=weight, 4=bias, 5=config. (288-293) — kernel 포트 순서와 대응.
  - `load_inst/load_weights/load_bias`(512~)로 파일에서 정적 벡터 적재, `init()`(551~)에서 device/program/kernel 설정.
- 의미: **FPGA 가속기를 TensorFlow 레이어로 노출** → 애플리케이션(OpenPose 등)의 CNN 부분을 FPGA로 offload. (README.md:9)

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 전체 컴파일→실행 파이프라인
```
ONNX 모델
  └─ extract_info.py  → pre_dse_models/{model}.csv  (레이어 융합 + en 비트 + 채널/커널)
  └─ dse.py           → post_dse_architectures/{model}_arch.json  (SA_ROWS/COLS/SIMD, 타일, dw, latency/resource)
  └─ inst_gen.py      → insts/{design_name}_instructions.dat  (레이어별 명령어 = CONFIG_PARAMS 워드)
  └─ generate.py(h_to_json + design_prepare)  → {design_name}/ (params.h, cnn_features.json, HLS/TAPA 소스)
  └─ generate_weights/biases/data.py  → *_reorg.dat (타일 레이아웃 가중치/바이어스/검증 데이터)
런타임:
  host(.exe / libsacc) → DRAM(cin/weight/bias/config) → top_kernel
     → while(layer): engine() 데이터플로우 1회 = 1레이어(또는 layer_batch)
         cin_load → (cin_load_prev) → weight_load → depth_conv → relu6 → conv(SA) → relu → add → upsample → merge → cout_write
  → DRAM(cout) → host
```

### 4.2 온칩 데이터플로우(engine)
- 모든 모듈이 `hls::stream` FIFO로 연결된 **단일 DATAFLOW 영역**(kernel.cpp:5352). 각 모듈은 `while(!done)`로 레이어/타일을 순회하며 II=1 파이프라인. 명령어는 데이터와 함께 config FIFO로 전파되어 모듈마다 독립 디코드.
- **conv만 systolic array**, 나머지(depth_conv/relu/bn/pool/upsample/add)는 line-buffer stencil. → "SA(연산집약) + 경량 stencil(메모리집약)" 혼합.

### 4.3 메모리 계층
- **DRAM/HBM**: global_cin/prev_cin/cout/weight/bias/config (512b AXI burst). (kernel.cpp:5655-5660)
- **URAM**: cin_load·cin_load_prev의 ping/pong 더블버퍼. (kernel.cpp:429-430, 702-703)
- **BRAM**: weight/cout/BN buf, stencil line buffer, SA 피더 버퍼(desp_gen `DFC_BUF_SIZE`). mem_type(0~4)으로 URAM↔BRAM 배치 조정(h_to_json params.h 매크로).
- **PE 로컬 레지스터**: output-stationary 부분합(`LOCAL_REG_NUM`개). (2DPE compute:792)

### 4.4 병렬화 차원
- **SIMD_LANE=8**(util.h:25, stencil 모듈) / SA의 **SIMD_FACTOR**(=lane, reduction 병렬, 2DPE:26) — 입력 채널 방향 벡터화.
- **SA_ROWS × SA_COLS** 2D PE 어레이 — 출력채널(rows) × 출력width(cols) 공간 병렬. (design_desp.json:89-90)
- **double buffering** — DRAM read와 compute 오버랩(cin_load task_cnt%2).
- **멀티 커널(libsacc q/q2/q3)** — DDR 뱅크 3개 병렬 처리.

### 4.5 양자화 / 데이터타입
- **HLS_Codes 레퍼런스 백본은 float32 고정**(util.h:41-48: data_t0~3=float/uint, 512b 버스).
- **auto_compile 생성 디자인은 dw로 선택**: 32=float, 16=ap_fixed<16,8>, 8=ap_fixed<8,4>. (dse.py:414-437) DSP/MAC 비용도 정밀도에 비례(float 5 → fixed8 0.25, resource_est.py:26-31).
- 즉 **양자화는 "비트폭 선택 + ap_fixed 치환" 수준**(PTQ/QAT 학습 파이프라인은 코드에 없음 — 확인 불가).

---

## 5. HW/SW 매핑

| 계층 | 구성요소 | 위치 |
|---|---|---|
| **SW: 컴파일러** | ONNX 파싱·융합·DSE·명령어/코드 생성 (Python) | `auto_compile/*` |
| **SW: 호스트** | XRT/OpenCL로 DRAM 적재·커널 실행·레이아웃 변환 | `libsacc/src/libsacc.cpp`, `SDx_project/src/host.cpp` |
| **SW: 프레임워크 통합** | TensorFlow custom op `Sacc` | `libsacc/inc/sacc.hpp` |
| **HW: 명령어 디코드** | layer_config → engine 모듈별 LAYER_EN/타일/주소 | `top_kernel`, `cin_load`(kernel.cpp:516-568) |
| **HW: 데이터 이동** | URAM 더블버퍼 + 512b AXI burst + 레이아웃 언팩 | `cin_load`, `weight_load`, `cout_write` |
| **HW: 경량 연산** | depthwise/pool/upsample/relu/bn/add = line-buffer stencil | `util.h` 템플릿, `depth_conv`/`relu`/`pool`/`upsample` |
| **HW: 핵심 연산** | NConv/DConv/TConv = 2D output-stationary systolic array | `conv`→`kernel()`, 생성된 `2DPE/2DDataFeed/2DDataCollect` |
| **HW: SA 코드젠** | 타일/IL/SIMD 매개화 VSA → HLS/TAPA C++ | `desp_gen.py`/`code_template.py`/`codegen.py` |

- **재구성형(overlay) 특성**: 비트스트림은 고정, **명령어 스트림으로 네트워크/레이어 변경** → 다양한 CNN을 재합성 없이 실행(SW가 명령어만 새로 생성).

---

## 6. 빌드 · 실행

(README.md:39-80, generate_design.sh, run.sh 근거)
1. 환경: `source env.sh`, `mkdir auto_compile/data/onnx` 후 ONNX 배치. (README.md:42-50)
2. 디자인 생성: `./generate_design.sh` 또는 `echo 13 | auto_compile/run.sh ENet ENet 32 u250 4 TAPA_1`. (README.md:53-59) → `designs/TAPA_1_ENet_8_9_8_MEM_4_32/` 생성.
3. C-simulation: `cd designs/...; source env.sh; cd sim; make csim_all; ./host.exe 1 89`(ENet은 89 inst). 결과 `sim/outputs/csim_output.log`. (README.md:62-71)
4. 비트스트림: `make hw_all; ./host.exe 1 89 --bitstream=...xclbin`. (README.md:72-80)
- HLS_Codes 단독: `hls_script.tcl`(Vitis HLS), `design_prepare.sh`.
- libsacc: cmake(>=3.0) + googletest + TensorFlow, `lib_base_path` 수정 후 `cmake .. && make all`로 `libsacc.so` 생성. (libsacc/README.md:32-64)

---

## 7. 의존성

- **HW**: Xilinx Alveo U250/U280, Vitis 2021.2, **TAPA HLS**(P&R 최적 코드), Vitis HLS. (README.md:28-37)
- **SW(컴파일러)**: Python 3.8, `onnx`/`onnxruntime`, `numpy`, `networkx`(그래프/의존성), `multiprocessing`. (requirements.txt, dse.py/extract_info.py import)
- **SW(호스트)**: Xilinx XRT/OpenCL(`xcl2`), cmake, googletest, TensorFlow(libsacc). (libsacc/README.md:21-29)
- **레거시**: libsacc는 XRT v2018.3/SDAccel/VCU1525 기준(구버전). (libsacc/README.md:13-26)
- **외부 데모**: `tf_DSA`는 tf-pose(OpenPose) 의존 — FlexCNN 자체 아님.

---

## 8. 강점 / 한계 / 리스크

### 강점
- **완전 엔드투엔드 자동화**: ONNX→DSE→명령어→HLS/TAPA 소스까지. analytical latency/resource 모델로 빠른 DSE.
- **명령어 기반 overlay**: 재합성 없이 명령어만으로 다양한 CNN/레이어 실행 — 빠른 turnaround.
- **conv 타입 풍부**: NConv/DepthWise/Dilated/Transposed를 sub-kernel 분해(K_NUM)로 단일 SA에서 처리.
- **연산/메모리 모듈 분리**: SA(연산) + line-buffer stencil(메모리효율) 혼합, double buffering + 512b coalescing.
- **범용 SA 코드젠**: desp_gen이 cnn/mm/mv/nw를 모두 지원 — Transformer GEMM(mm_pass)에 그대로 재사용 가능(시사점 9 참조).
- **TF 통합**(libsacc)로 실제 애플리케이션 offload 경로 제시.

### 한계
- **수작업 HLS_Codes 백본은 float32 고정**(util.h:41-48), 일부 하드코딩 의존: `stencil_w3 assert(w_t==26||50||98)`(util.h:211), `stencil_w1 assert(24||48||96)`(util.h:494) — **동적 타일링이 특정 width만 합성**. inst_gen `calculate_addresses`도 ENet/UNet/VGG16 초기주소 하드코딩(서브분석 169-191).
- **검증 범위**: U-Net/E-Net/VGG16만 테스트(README.md:12). mapper의 unsupported op(MatMul/Softmax/GlobalAvgPool) → Transformer류 직접 지원 안 됨.
- **코드 위생**: `* copy`/`HLS copy`/모델전용 inst_gen 등 중복본 다수, 주석 처리된 미사용 모듈(inter_load/pool/inter_write) 혼재 → 유지보수 리스크.
- **양자화 깊이 얕음**: dw 비트폭 선택+ap_fixed 치환 수준, QAT/캘리브레이션 파이프라인 부재(확인 불가).
- **libsacc 구버전 의존**(XRT 2018.3) — 최신 Vitis와 정합성 확인 불가.

### 리스크
- analytical latency 모델(FRE 220MHz 고정, DRAM 250cy 가정)이 실측과 괴리 가능 — DSE 최적값이 실제 최적 아닐 수 있음(추정).
- 명령어 포맷이 HLS 디코드/inst_gen/h_to_json 3곳에 분산 정의 → 한 곳 변경 시 동기화 깨질 위험.

---

## 9. 우리 프로젝트 관점 시사점

> 추정 컨텍스트: 우리 프로젝트는 **ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적**. FlexCNN은 CNN/systolic/HLS 자동컴파일이 핵심이라 직접 이식보다 **재사용 가능한 메커니즘**이 가치.

1. **systolic array 코드젠(desp_gen.py)의 `mm_pass` 직접 재사용 가능** (desp_gen.py:323-563)
   - Transformer의 QKV projection, attention score(QK^T), context(AV), FFN은 전부 **GEMM**이다. desp_gen은 이미 `mm_pass`(matrix-matrix)와 `mv_pass`(matrix-vector)를 보유 → ViT의 GEMM을 타일/IL/SIMD 매개화된 output-stationary SA로 자동 합성하는 기반으로 차용 가능. (`cnn_pass`만 conv 특화, 나머지는 GEMM 범용)
   - PE 마이크로아키텍처(`2DPE`의 SIMD MAC + op0=Down/op1=Right/res=Down output-stationary, 2DPE_U1.cpp:9-51,91-97)는 HG-PIPE 류 GEMM 어레이 설계의 레퍼런스로 유용.

2. **명령어 기반 overlay 패턴** (top_kernel + engine + LAYER_EN, kernel.cpp:5646-5694, 548-560)
   - ViT는 동일 블록(LN→MHA→LN→MLP)이 L회 반복 → FlexCNN처럼 **비트스트림 1개 + 명령어 스트림으로 레이어 반복**하면 ViT 깊이/해상도 변경 시 재합성 불필요. config FIFO로 명령어를 파이프라인에 전파하는 구조(engine)는 그대로 패턴 차용 가능.

3. **DSE analytical 모델(latency_est/resource_est)** (latency_est.py:117,217-219, resource_est.py:35-36)
   - `compute = M*N*K/(rows*cols*simd)`, `DSP = rows*cols*simd*DSP_per_MAC`(정밀도별 5/1/0.25)는 GEMM에 그대로 적용 가능 → ViT 가속기의 SA 형상/타일 자동 탐색 + Pareto(latency vs BRAM) 프레임워크로 재사용.
   - dynamic tiling + 레이어 간 타일 의존성 제약(dse.py:204-308) 개념은 Transformer의 시퀀스 길이/헤드 분할 타일링에 응용 가능.

4. **양자화 추상화** (dse.py:414-437)
   - `dw`(32/16/8) → float/ap_fixed 자동 치환 + DSP 비용 모델 연동 구조는, ViT의 INT8/INT4 양자화 가속기에서 "정밀도→자원/지연" 트레이드오프 자동화에 차용 가능. (단 FlexCNN엔 QAT 없음 → 우리 양자화 모듈과 결합 필요)

5. **데이터 레이아웃 reorg + double buffering** (cin_load:46-77, generate_data.py:49-138)
   - 512b coalescing + URAM 더블버퍼 + 타일 인터리브 레이아웃은 ViT의 토큰×차원 텐서를 HBM에서 SA로 공급할 때 동일하게 필요. reorg 코드(SW)와 언팩 코드(HW)의 짝 구조가 참고 템플릿.

6. **stencil 모듈은 ViT엔 부적합** — depth_conv/pool/upsample line-buffer는 conv 전용이므로 ViT 본체엔 불필요. 단 **XR 시선추적 전처리**(눈 영상 CNN 백본, 예: 경량 conv 전처리)에는 그대로 유용 → "CNN 전처리(FlexCNN stencil) + ViT 본체(mm_pass SA)" 하이브리드 파이프라인 구성 가능(추정).

7. **주의점**: FlexCNN의 mapper는 MatMul/Softmax/LayerNorm을 unsupported로 스킵(mapper.py:405-481) → **attention/LN/GELU/Softmax HW 모듈은 우리가 신규 구현 필요**. FlexCNN에서 가져올 것은 "SA 코드젠 + overlay + DSE 인프라"이고, Transformer 특화 연산기는 자체 개발 대상.

---

## 10. 근거 표기 규칙

- 본문의 모든 `파일:라인` 표기는 Read/Grep으로 실제 확인한 소스 근거다. (예: kernel.cpp:5352, util.h:25)
- **"추정"**: 코드 구조상 합리적이나 명시적 라인 근거가 없는 추론(예: relu의 BN 수식, stencil 전처리 활용 시나리오, FRE 가정의 실측 괴리).
- **"확인 불가"**: 해당 기능/사실이 분석 범위 소스에 존재하지 않거나 외부 산출물이라 검증 못 함(예: QAT 학습 파이프라인, libsacc 구버전과 최신 Vitis 정합성, tf_DSA 내부 OpenPose 세부).
- `auto_compile/` 라인 근거 일부는 병렬 서브분석가가 동일 제약(Read 전용)으로 조사한 결과를 통합했으며, 핵심 식·비트필드·매핑 규칙은 라인번호와 함께 명시했다.
