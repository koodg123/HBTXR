# FlexCNN 모듈 통합 가이드

> 1차 요약: [`../FlexCNN.md`](../FlexCNN.md) — 본 문서는 그 요약을 **codegen 단위**로 심화한 통합 가이드다.
> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\CNN-Accel\FlexCNN` (UCLA-VAST, TRETS 2022)
> 작성 원칙: 실제 소스 Read 후 `파일:라인` 근거 표기. 라인 근거 없는 추론은 "추정", 코드로 확인 불가는 "확인 불가"로 명시.
> 형제 가이드(동형 구조): [`../../CNN-Accel/ESDA/MODULE_GUIDE.md`](../ESDA/MODULE_GUIDE.md). ESDA가 "ILP+gen_code H-HLS"라면, FlexCNN은 "ONNX→DSE→inst→**PolySA SA codegen + 모듈 조립 codegen**"의 H-HLS 변형이다.

---

## 0. 문서 머리말

### 0.1 대표 케이스 (대표 생성 인스턴스)

FlexCNN은 단일 비트스트림이 아니라 **모델×보드×데이터폭마다 새 HLS 인스턴스를 codegen**한다. 본 가이드는 repo에 실제로 생성되어 있는 descriptor를 대표 인스턴스로 고정한다.

- **대표 생성 인스턴스: `HLS / ENet / SA(16×14×8) / MEM_4 / dw=32`**
  근거: `generate_design.sh:3`이 `echo 13 | run.sh ENet ENet 32 u250 4 TAPA_1`을 호출(=ENet, dw=32 float, u250, mem_type=4). repo의 생성 descriptor `auto_compile/design_generation/HLS/systolic_array_kernel/output/design_desp.json:89-90,113,130-131`가 **SA_ROWS=16, SA_COLS=14, SIMD_FACTOR=8, ROW_IL_FACTOR=4, COL_IL_FACTOR=8**로 baked-in 되어 있어 정량의 1차 근거로 삼는다.
  - ※ **1차 요약(`../FlexCNN.md`)이 인용한 "SA_ROWS=4, SA_COLS=6, SIMD=4"는 다른(과거) design point**다. 실제 repo에 남아 있는 descriptor는 16×14×8이므로 본 가이드는 후자로 정정한다(근거: 위 라인).
- **대표 conv 커널**: 3×3 NConv, IN_NUM=OUT_NUM=512, IN_IMG_H/W=226(=224+K-1), 타일 OUT_NUM_T=64, OUT_IMG_W_T=112 (`design_desp.json:4-16`). 이 타일이 `2DPE/2DDataFeed`의 카운터·버퍼 차원을 모두 결정.
- **대표 가변 conv**: TConv(transposed). inst_gen이 `conv_type=1, k_num=tstride²`로 인코딩(`inst_gen.py:513-534`), `2DPE_U1.cpp:798-803`의 `LAYER_LOCAL_ACCUM_NUM_ARR[4]`로 sub-kernel별 누적 길이를 분기 → 동일 SA로 TConv 처리.
- **대표 엔진 그래프**: ENet (`arch_connectivity/ENet.json` 11 노드: cin_load→SA→upsample→concat→add→act_and_bn→cout_write + bias_load/cin_load_prev/pool). 이 그래프가 codegen으로 `engine()` DATAFLOW 본체를 결정.

### 0.2 수치 표기 규약

- **MAC lanes(생성 파라미터)** = `SA_ROWS × SA_COLS × SIMD_FACTOR`. 대표 인스턴스 = 16×14×8 = **1792 float MAC/cycle**(각 PE가 8-lane SIMD reduction, `2DPE_U1.cpp:38-57`, 224 PE). DSP는 정밀도별 가중(아래 0.5).
- **scalar MACs(대표 CNN conv)** = `IN_NUM × OUT_NUM × KH × KW × OUT_H × OUT_W`. dse.py 동일식(`dse.py:112`). TConv는 입력차원 사용(`dse.py:134`).
- **loop trips** = SA: `TASK_NUM1 = ⌈in/in_t⌉·⌈out/out_t⌉·⌈in_h/h_t⌉·⌈in_w/w_t⌉`(`inst_gen.py:32`), 각 task 내부 `LOCAL_ACCUM_NUM × K_NUM × LOCAL_REG_NUM` 누적(`2DPE_U1.cpp:864-895`).
- **memory(payload)** = 버퍼원소수 × DATA_W(bit). cin/weight/cout 더블버퍼는 ×2. URAM vs BRAM 배치는 mem_type(0~4)이 14개 버퍼별로 결정(`h_to_json.py:516-590`).
- **codegen 자동화 범위** = ONNX→csv(extract_info)→arch.json/inst(dse,inst_gen)→**params.h(h_to_json) + SA 6파일(codegen/desp_gen) + engine top(code_template flexcnn_*) + 모듈 본체(modules/*.cpp 조립)** → C++ 디자인. 가중치/바이어스/검증데이터까지 생성.
- **합성 PPA**: repo에 csynth/cosim/vivado 리포트 **없음 → 확인 불가**. 보드 자원 상한만 존재(`boards/u250.json`: LUT 1.728M, FF 3.456M, BRAM18K 8000, DSP 12288, DSP_THRES 0.6).

### 0.3 운영 경로 (모델→auto_compile→HLS→보드)

```
[SW: ONNX 모델 (U-Net/E-Net/VGG16, data/onnx/*.onnx)]
      │  extract_info.py (-g onnx -a pre_dse_arch.json)
      ▼  graph→networkx, conv타입분류, 레이어융합, en비트
[pre_dse_models/{model}.csv]  (inst dict 1줄/행, header=입력shape)
      │  dse.py (-b boards/{board}.json -dw {32|16|8})
      │   get_layers_configs→MACS/IN_DATA_LAYOUT, 설계공간 sweep(멀티프로세스),
      │   latency_est+resource_est, Pareto, input()로 design id 선택
      ▼  마지막 줄=ROWS_COLS_SIMD → run.sh design_name
[post_dse_architectures/{model}_arch.json + post_dse_models/{model}.csv]
      │  inst_gen.py  (tested CNN은 {model}_inst_gen.py)
      ▼  en비트마스크 + SA카운터 + conv_type + DRAM주소
[insts/{design_name}_instructions.dat]  (7줄/inst, 53 CONFIG_PARAMS)
      │  generate.py → h_to_json.py (params.h + cnn_features.json)
      │             → design_prepare.sh:
      │                ① codegen.py: desp_gen(design_desp.json)→SA 6파일 + top.cpp(engine codegen)
      │                ② cat SA/* > hw_kernel; hw_kernel_modify.py
      │                ③ engine top = arch_connectivity 그래프로 FIFO+task 배선 + modules/*.cpp 본체 조립
      ▼
[designs/{code}_{model}_{R}_{C}_{S}_MEM_{mt}_{dw}/]  (hw_kernel.cpp, params.h, host, sim/)
      │  generate_weights/biases/data.py → *_reorg(타일레이아웃 가중치/골든)
      ▼  Vitis HLS 2021.2 / TAPA → xclbin
[보드: Alveo U250/U280] host.exe (libsacc로 TF custom op offload 가능)
```
- 타깃: **Xilinx Alveo U250/U280**, Vitis 2021.2, **TAPA HLS**(P&R 최적) 또는 Vitis HLS (`README.md:28-37`). FRE 가정 220MHz(`dse.py:772`).

### 0.4 데이터타입

- **대표(dw=32)**: `data_t0/1/2 = float`, 512b DRAM 버스, BUS_PACK_FACTOR0=16(=512/32), SIMD_LANE=8 (`h_to_json.py:443-450`, `util.h:42-51`). PE 피드 버스=256b(=8×32, `common_header_U1.h` Data0SIMDType).
- **양자화 선택(dse.py:414-437)**: dw=32→`float`(DSP/MAC=5), 16→`ap_fixed<16,8>`(=1), 8→`ap_fixed<8,4>`(=0.25). `K_T=16`(`dse.py:441`). **QAT/캘리브레이션 없음 → 비트폭+ap_fixed 치환 수준**(확인 불가).
- 토큰/inst: `ConfigInst = ap_uint<192>`(6 inst 그룹), `CONFIG_PARAMS=53`, `INST_PER_LAYER=6` (`util.h:38-39,56`).

---

## 1. 개요 (codegen/HW 맵) + 호출계층 + 제외

### 1.1 두 갈래 codegen + HW

FlexCNN의 핵심은 **두 종류의 codegen**이 합쳐져 하나의 `hw_kernel.cpp`를 만드는 것이다.

| 갈래 | 입력 | 출력(생성물) | 드라이버 |
|---|---|---|---|
| **(A) SA codegen** (PolySA/AutoSA, Jie Wang) | `cnn_features.json`(h_to_json 생성) | `design_desp.json` → `2DPE/2DDataFeed/2DDataCollect/2DDataFeedCollect/common_header/top.cpp` | `codegen.py`+`desp_gen.py`+`code_template.py` |
| **(B) engine 조립 codegen** | `arch_connectivity/{model}.json` + `modules/*.cpp` | `top.cpp`의 `engine()`(FIFO 배선 + task 호출) + 모듈 함수 본체 | `code_template.flexcnn_functions/fifos/tasks` |
| **(C) params codegen** | inst.dat + design_params | `params.h`(MAX타일/버퍼크기/data_t/mem배치 14매크로) | `h_to_json.py` |

HW 구조:
- **conv = 2D output-stationary systolic array** (갈래 A). cin(op0)=Down, weight(op1)=Right, cout(res)=Down (`design_desp.json:91-97`).
- **나머지(cin_load/weight_load/bias_load/act_and_bn(relu+bn)/pool/upsample/concat/add/cout_write) = line-buffer stencil / 버스언팩 모듈**(갈래 B의 `modules/*.cpp`).
- engine = 단일 `#pragma HLS DATAFLOW`로 위 모듈을 config FIFO + data FIFO로 연결(`code_template.py:525`, FIFO depth=128/16 `code_template.py:599,612`).

### 1.2 호출계층 (생성 시점 / 런타임 시점)

```
generate_design.sh → run.sh
 ├─ extract_info.py (PBPredictor)        # ONNX→csv
 ├─ dse.py (get_layers_configs, param_sweep, model_latency_est, res_est)
 │     └─ latency_est.py / resource_est.py / utils.py
 ├─ inst_gen.py (get_en_value, get_SA_insts, calculate_addresses, run)
 └─ generate.py
       └─ h_to_json.py (get_mem_params → params.h + cnn_features.json)
       └─ design_prepare.sh
             └─ codegen.py.run
                   ├─ desp_gen.run (cnn_pass | mm_pass | mv_pass | nw_pass)  → design_desp.json
                   └─ code_template (PE_MAC/op_transfer/compute/res_transfer/kernel ; df ; dc ; loader ;
                                     flexcnn_functions/fifos/tasks/engine_header/top_function)
런타임(생성된 hw_kernel.cpp):
 top_kernel → while(layer): engine(...)   # DATAFLOW
   cin_load → (cin_load_prev) → weight_load/bias_load → [SA(=kernel→2DPE 어레이)] → act_and_bn → add/concat/upsample/pool → cout_write
```

### 1.3 제외 (이름만)

- **중복 작업본**: `design_generation/{HLS copy, TAPA_1 copy}`, `data_generation/generate_data copy*.py`, `inst_generation/inst_gen_old.py`, `systolic_array_kernel/h_to_json copy.py`, `TAPA_1 copy/{tmp.cpp, FlexCNN copy.cpp}`.
- **구버전 파이프라인**: `auto_compile/auto_compile_old/`(protobuf 기반).
- **모델전용 변형**(정식 일반본만 분석): `inst_generation/{VGG16,UNet,ENet}_inst_gen.py`, `ENet_inst_gen_16.py`.
- **타깃 변형**: `TAPA_1`(=대표), `TAPA_2`, `HLS`는 동형 codegen — 본 가이드는 **HLS 타깃 트리**(가장 완비)를 라인 근거로 인용하되 구조는 3타깃 공통(추정).
- **외부 프레임워크/산출물**: `tf_DSA/`(tf-pose OpenPose), `libsacc/data/*.dat`, `design/`의 sim 산출물, `*.xclbin/*.pyc`, `SDx_project/src/xcl2.*`(Xilinx OpenCL 유틸).
- **생성물(수치 확인용만 인용)**: `SA/2D*`, `output/*`, `params.h`, `design_desp.json`, `cnn_features.json`.

---

## 2. 모듈: SA codegen 드라이버 — `codegen.py` + `desp_gen.py`

### 2.1 역할 + 상위/하위
- **역할**: VSA(Virtual Systolic Array) descriptor를 만들고(`desp_gen`) 그로부터 6개 SA C++ 파일 + engine top을 방출(`codegen`). **CNN 전용이 아니라 범용 SA 합성기**: `cnn_pass/mm_pass/mv_pass/nw_pass` 4개 패스 보유(`desp_gen.py:14,323,566,769`).
- **상위**: `design_prepare.sh:5-7`. **하위**: `code_template.py`(조각 텍스트).

### 2.2 데이터플로우
```
cnn_features.json ──desp_gen.run──▶ design_desp.json (LOCAL_REG/ACCUM_NUM, DF/DC 카운터, DFC_BUF_SIZE, HEAD_CODE, SW_KERNEL_CODE)
design_desp.json ──codegen.run──▶ generate_PE/DF/DC/Loader/header/tb/top  (output/2D*.cpp, common_header_U1.h, top.cpp)
```

### 2.3 대표 코드 위치
`desp_gen.py:14-321`(cnn_pass), `codegen.py:101-141`(run), `codegen.py:44-53`(PE 조립 순서).

### 2.4 대표 코드 블록 (cnn_pass 핵심 식)
```python
vsa['LOCAL_REG_NUM']   = OUT_IMG_H_T * ROW_IL_FACTOR * COL_IL_FACTOR        # desp_gen.py:16  (=14*4*8=448)
vsa['LOCAL_ACCUM_NUM'] = IN_NUM_T * K * K / SIMD_FACTOR                     # desp_gen.py:18  (=64*9/8=72)
vsa['MAC_STAT'] = 'sum += op0_u[i] * op1_u[i];\n'                           # desp_gen.py:23
vsa['DFC_BUF_SIZE'] = [ IN_NUM_T*(OUT_IMG_H_T+K-1)*(COL_IL+K-1),            # cin feeder  :29
                        ROW_IL*IN_NUM_T*K*K,                               # weight feeder:30
                        ROW_IL*SA_ROWS*OUT_IMG_H_T*COL_IL ]                # cout collect :31
# DF_FEED_COUNTER 6중: c0=OUT_IMG_H_T, c1=ROW_IL, c2=COL_IL, c3=K, c4=K, c5=IN_NUM_T/SIMD  (:49-95)
# DC_COLLECT_COUNTER 4중: c0=OUT_IMG_H_T, c1=ROW_IL, c2=COL_IL, c3=OUT_NUM_T/ROW_IL        (:100-130)
```
→ 대표값: LOCAL_REG_NUM=448, LOCAL_ACCUM_NUM=72 (단, 빌드된 `params.h`는 이전 design point의 128로 baked — `design_desp.json:171-172` vs subagent 확인). `SW_KERNEL_CODE`(`desp_gen.py:135-148`)는 6중 naive conv 검증 레퍼런스.

mm/mv/nw 패스는 `LOCAL_ACCUM_NUM`만 다름: mm=`K_T/SIMD`(`:327`), mv=`J_T/SIMD`(`:570`), nw=`BLEN_T/SIMD`(`:778`) — **GEMM/MV로 재사용 가능한 동일 골격**.

### 2.5 마이크로아키텍처(codegen 관점)
- `cal_width(range)=ceil(log2(range))+3`(`code_template.py:30-35`)로 모든 카운터 비트폭 자동 산정.
- codegen.run: desp_gen → design_desp.json 로드 → tb/top/header/PE/DF/DC/Loader 순 방출(`codegen.py:109-140`). PE 파일은 `PE_MAC+op_transfer+compute+res_transfer+kernel` 순 concat(`codegen.py:49-53`).

### 2.6 정량/병목
- DFC_BUF_SIZE는 타일·IL·SIMD로 매개화 → DSE 타일 선택이 SA 온칩 버퍼를 직접 결정.
- 병목: design_desp의 LOCAL_REG_NUM(448)과 빌드 params.h(128) 불일치는 **design point 변경 시 재생성 필수**임을 시사(추정).

---

## 3. 모듈: 생성된 PE 어레이 — `2DPE_U1.cpp` (conv 가속 핵심)

### 3.1 역할 + 상위/하위
- **역할**: 16×14 output-stationary PE 격자. 각 PE = 8-lane SIMD float MAC + 로컬 누산 레지스터파일. cin↓ weight→ cout↓.
- **상위**: `SA.cpp`의 `kernel()` 호출(`SA.cpp:242`). **하위**: `U1_PE_MAC`(프리미티브).

### 3.2 데이터플로우 (PE 격자)
```
        weight(op1) ──R──▶
cin(op0)   ┌────┬────┬─ ... ─┐  (14 cols)
  │        │PE00│PE01│       │
  ▼ D      ├────┼────┤       │
           │PE10│PE11│       │   각 PE: local_buffer[LOCAL_REG_NUM] 누산
  ...      └────┴────┴───────┘  (16 rows)
res(cout) ──D──▶ 하단 collector(열당 1개, 14개)
```

### 3.3 Function call stack
`SA.cpp:242 kernel(...)` → `2DPE_U1.cpp:1259 U1_kernel`(`#pragma HLS DATAFLOW :1266`) → 224개 `U1_op0/op1_transfer_wrapper` + `U1_compute_wrapper` + `U1_res_transfer[_first]_wrapper` 인스턴스(예: PE(0,0) `:6552-6592`, PE(1,0) `:7136-7177`).

### 3.4 대표 코드 위치
`2DPE_U1.cpp`: PE_MAC `:9-59`, op0_transfer `:62-393`, op1_transfer `:395-727`, compute `:728-919`, res_transfer `:921-1093`, kernel(격자배선) `:1259+`.

### 3.5 대표 코드 블록
```cpp
// U1_PE_MAC (output-stationary): 8-lane 곱 + 3단 가산트리 + 누산  (2DPE_U1.cpp:36-59)
U1_data_t2 sum = (init==1)? 0 : *op2;        // :36  부분합을 PE 로컬 op2에서 read-modify-write
U1_data_t2 mult0 = op0_u[0]*op1_u[0]; ...mult7;   // :38-45  (256b=8×float32 언팩)
U1_data_t2 sum2_0 = mult0+mult1; ... ; sum0_0 = sum1_0+sum1_1;  // :47-55
sum += sum0_0;  *op2 = sum;                  // :57,59

// U1_compute: 가변 conv (NConv/TConv) sub-kernel 분해  (2DPE_U1.cpp:798-895)
ap_uint<10> LAYER_LOCAL_ACCUM_NUM_ARR[4] = { in_ch_factor*unpack(KH,0)*unpack(KW,0), ...i=1,2,3 };  // :798
U1_PE_MAC(.., &local_buffer[local_reg_id], (init && la_counter==0 && i<K_NUM)?1:0);  // :873
if (la_counter==LAYER_LOCAL_ACCUM_NUM_ARR[i]-1 && last) fifo2_local.write(local_buffer[local_reg_id]); // :874
```
→ `K_NUM`개 sub-kernel을 순회하며 sub-kernel `i`는 `local_buffer[i*LOCAL_REG_NUM …]` 영역 사용 → **transposed/dilated conv를 동일 어레이로 처리**(inst_gen `conv_type`/`k_num`과 짝).

### 3.6 마이크로아키텍처
- **격자 배선 증거**: PE(1,0) op0가 PE(0,0)의 `fifo0_feed1_0`를 소비(`:7137`)=Down, op1은 행 내 우측 전파=Right, res는 `fifo2_collect0_0`→`collect1_0` 하단 누적=Down(`:7169`). → `OP_CHANNEL_DIR=["D","R"]`, `RES_CHANNEL_DIR=["D"]`(`design_desp.json:91-97`) 일치.
- **MAC lanes** = 16×14×8 = **1792 float MAC/cyc**. II=1(`:865`), `DEPENDENCE inter false`(`:866`).
- **메모리**: PE 로컬 `local_buffer[U1_LOCAL_REG_NUM]`(=output-stationary 누산파일), res_transfer 버퍼 `[U1_TRANSFER_REG_NUM]`.

### 3.7 정량/병목
- scalar MACs(대표 3×3 conv 512→512, 224×224 출력) = 512·512·9·224·224 ≈ **118 GMAC/layer**(추정 계산). peak 1792 MAC/cyc·220MHz ≈ 394 GMAC/s.
- 병목: float MAC당 DSP=5(`resource_est.py:27`) → 1792×5 = 8960 DSP(U250 12288의 73%, DSP_THRES 0.6 초과 가능 → DSE가 제외할 수 있음, 추정).

---

## 4. 모듈: SA 피더/콜렉터 — `2DDataFeed/2DDataCollect/2DDataFeedCollect_U1.cpp`

### 4.1 역할 + 상위/하위
- **역할**: DRAM↔온칩 reorg + 1D 데이지체인으로 cin/weight를 PE 격자에 공급(feed)하고 cout 회수(collect). dilation/stride/TConv 오프셋을 주소생성에서 처리.
- **상위**: `U1_kernel`(`2DPE_U1.cpp:1259`). **하위**: AXI(Head 단) / PE FIFO.

### 4.2 데이터플로우
```
DRAM ─Head(memcpy reorg)→ Engine0..EngineLast(데이지체인) ─▶ PE feed FIFO   (feed)
PE res FIFO ─▶ Engine0..EngineLast(데이지체인) ─Head(ping/pong)→ DRAM        (collect)
```

### 4.3 Function call stack
feed: `2DDataFeedCollect_U1.cpp:9 U1_DataFeed0Head`(cin)/`:305`(weight) → `2DDataFeed_U1.cpp:9 U1_Data0FeedData0`(cin)/`:126`(weight) → Engine 래퍼. collect: `2DDataCollect_U1.cpp:185 U1_Data2ReadData0` → `:9 Write` → `:472 DataCollect2Head`(ping/pong).

### 4.4 대표 코드 위치
cin feeder `2DDataFeed_U1.cpp:9-125`, weight feeder `:126-236`, collector read `2DDataCollect_U1.cpp:185-247`, Head들 `2DDataFeedCollect_U1.cpp:9,305,472`.

### 4.5 대표 코드 블록
```cpp
// cin 피더: stride/dilation 주소생성  (2DDataFeed_U1.cpp:65-71)
if (STRIDE==1){ w_idx=c2+c4*DILATION_RATE; h_idx=c0+c3*DILATION_RATE; }
else if (STRIDE==2){ w_idx=c2*2+c4; h_idx=c0*2+c3; }
// weight 피더: TConv 2×2 sub-kernel 오프셋 테이블 (2DDataFeed_U1.cpp:154-159, K_NUM==4)
// collector: 결과 행순서 역전 (Down 도착 → 자연 OUT_NUM 순서 복원)  (2DDataCollect_U1.cpp:211)
((U1_SA_ROWS-1-c3_counter)*LAYER_ROW_IL_FACTOR + c1_counter)
```

### 4.6 마이크로아키텍처 + 정량
- **OP_ENGINE_NUM=[14,16]**(cin 피더=SA_COLS=14, weight 피더=SA_ROWS=16), **RES_ENGINE_NUM=[14]**(`design_desp.json:164-170`).
- 데이지체인: `LAST_ENGINE=(engine_id==14/SPLIT-1)`(cin), `local_transfer_size=data0_buf_size*(14-engine_id)*GROUP`(`2DDataFeed_U1.cpp:263`) → 각 엔진이 하류 엔진 몫만 forward.
- collector Head ping/pong 더블버퍼(`2DDataCollect_U1.cpp:451-480`)로 PE 수집과 DRAM write-back 오버랩.
- 병목: feed 데이지체인이 SA 채움/비움 latency 지배(`latency_est`의 compute_drain·extra_latency, `latency_est.py:118,218`).

---

## 5. 모듈: engine 조립 codegen — `code_template.py` (flexcnn_*) + `arch_connectivity/{model}.json`

### 5.1 역할 + 상위/하위
- **역할**: ESDA의 `gen_code.py`에 대응하는 FlexCNN의 핵심 SW↔HW 브리지. **arch_connectivity 그래프(nodes/edges)를 위상정렬해 engine() DATAFLOW의 FIFO 선언 + 모듈 task 호출을 자동 배선**하고, 모듈 함수 본체(`modules/*.cpp`)를 조립한다.
- **상위**: `codegen.generate_top`(`codegen.py:80-99`). **하위**: `modules/*.cpp`, `arch_connectivity/{model}.json`.

### 5.2 데이터플로우
```
arch_connectivity/ENet.json (nodes 11, edges 12)
   │ flexcnn_functions  → modules/<node>[_<model>].cpp 본체 concat  (code_template.py:494-506)
   │ flexcnn_fifos      → 위상정렬 후 edge별 data FIFO(depth128) + 인접 노드 config FIFO(depth16)  (:585-614)
   │ flexcnn_tasks      → 노드별 task 호출, in/out edge로 인자 자동 결선  (:616-693)
   │ engine_header/top_function → engine() {#pragma HLS DATAFLOW} + top_kernel(while layer)  (:508-583)
   ▼  top.cpp
```

### 5.3 Function call stack
`codegen.generate_top:89-93` → `flexcnn_functions`(:494) + `engine_header`(:508) + `flexcnn_fifos`(:585) + `flexcnn_tasks`(:616) + `top_function`(:529).

### 5.4 대표 코드 위치
`code_template.py:494-693`(flexcnn_*), `arch_connectivity/ENet.json:3-29`(그래프).

### 5.5 대표 코드 블록
```python
# 모듈 본체 조립: pool/act_and_bn/bias_load는 모델별 변형  (code_template.py:498-504)
if module in ('pool','act_and_bn','bias_load'):
    open('../modules/'+module+'_'+cnn_name+'.cpp')   # 예: act_and_bn_ENet.cpp
else: open('../modules/'+module+'.cpp')
# FIFO 자동 선언  (code_template.py:598-599)
hls::stream<CinLoadData0Type> {src}_to_{dst}_{idx};  #pragma HLS STREAM ... depth=128
# task 인자 자동 결선  (code_template.py:638-691): in_edges→입력FIFO, out_edges→출력FIFO, config FIFO 전파
```
```cpp
// top_kernel: 명령어 기반 overlay (code_template.py:557-580)
memcpy(config, &layer_config[5 + CONFIG_PARAMS*layer_id], 4*CONFIG_PARAMS*cur_layer_batch); // :570
while(layer_id < end_layer){ engine(...); layer_id += 1; }                                   // :567-580
```

### 5.6 마이크로아키텍처 + 정량/병목
- **AXI 포트**(생성): gmem1(cin/cout), gmem3(prev_cin), gmem2(weight), gmem4(bias), gcontrol(layer_config) + s_axilite control (`code_template.py:542-555`). depth는 하드코딩(cout 826274, weight 34234, bias 1026 등 — 모델별, `:544-547`).
- **재구성형(overlay)**: 비트스트림 고정, `layer_config` 명령어 스트림으로 임의 깊이 CNN 실행. config FIFO가 6 inst 그룹을 파이프라인 따라 전파(각 모듈이 독립 디코드).
- 병목: edge별 data FIFO depth=128 고정(`:599`) → 레이어-파이프 불균형 시 stall 가능(추정). bias/gamma·beta 분기는 코드에 주석(현재 act_and_bn이 bias_load에서 직접 수신, `:601-605` 주석).

---

## 6. 모듈: cin/weight 로드 + 디코드 — `cin_load`/`weight_load`/`bias_load` (modules + FlexCNN.cpp)

### 6.1 역할 + 상위/하위
- **역할**: DRAM에서 입력특징맵/가중치/바이어스를 URAM 더블버퍼로 burst load하고 512b 버스를 SIMD_LANE(8)로 언팩해 FIFO write. **명령어(53워드 config) 디코드의 진입점**(cin_load).
- **상위**: engine task(codegen 배선). **하위**: AXI master.

### 6.2 데이터플로우
```
global_cin(DRAM 512b) ─cin_load_ddr_read(memcpy)→ URAM ping/pong ─cin_load_fifo_write(SIMD언팩)→ fifo_cin
config[53] ─디코드→ inst0..inst5(ConfigInst FIFO) ─전파→ 하류 모듈
```

### 6.3 Function call stack
`FlexCNN.cpp:430 cin_load` → `:27 cin_load_ddr_read`(memcpy) + `:89 cin_load_fifo_write`(언팩). prev: `:778 cin_load_prev`. (생성 모듈 본체는 `modules/cin_load.cpp`, FlexCNN.cpp는 동일본 조립 결과.)

### 6.4 대표 코드 위치
`FlexCNN.cpp`: cin_load `:430-776`, ddr_read `:27-83`, fifo_write `:89-259`, config 디코드 `:564-665`, cin_load_prev `:778-911`.

### 6.5 대표 코드 블록 (명령어 디코드)
```cpp
// 53워드 config → 6 inst 그룹  (FlexCNN.cpp:564-643)
LAYER_IN_NUM_HW=config[0..]; ... OUT_H_NPD/SPD/W_WPD/EPD=config[4..7];   // inst0: HW크기+비대칭패딩(N/S/W/E)
LAYER_EN=config[22..]; IN/OUT_NUM_T,IN_H_T,IN_W_T=config[24..27];        // inst3: enable비트+타일
TASK_NUM1/2, LOCAL_ACCUM/REG_NUM, ROW/COL_IL=config[29..34];            // inst4: SA카운터
CONV_TYPE=config[35]; FILTER_D0/D1; DILATION; TCONV_STRIDE; K_NUM; KH/KW(4바이트); POOL_NUM=config[35..52]; // inst5: 가변conv
// LAYER_EN 비트맵 (FlexCNN.cpp:602-616)
CONV_1ST[0] DEPTH_CONV[1] CONV[2] RELU[3] RELU6[4] POOL[5] UP_SAMPLE[6] BIAS[7]
INTER_LOAD[8] INTER_WRITE[9] BATCH_NORM[10] LOAD_PREV_CIN[11] BATCH_NORM_DEPTH[12]
// 더블버퍼 (FlexCNN.cpp:702-708): task_cnt%2로 ping/pong, read와 fifo_write 오버랩
```

### 6.6 마이크로아키텍처 + 정량/병목
- **URAM 더블버퍼**: `cin_burst_buf_ping/pong[CIN_BUFF/BUS_PACK_FACTOR0]`, mem_type에 따라 BRAM/URAM(`FlexCNN.cpp:441-447`).
- **두 데이터 레이아웃**: `change_layout`(=다음 필터 1)이면 `[num_tile][Th][Tw][Tn]` 통째 memcpy(`:54-63`), 아니면 halo 포함 행단위 burst(`:64-81`). dse의 `IN_DATA_LAYOUT`(1/2, `dse.py:170`)와 짝.
- **버스 언팩**: `DATA_SEL_FACTOR0=BUS_PACK_FACTOR0/SIMD_LANE`(=16/8=2)별 switch 언팩(`:123-227`).
- **CONFIG_PARAMS=53**, inst_gen이 7줄(6 데이터행)로 직렬화(`inst_gen.py:580-589`) → h_to_json `get_mem_params`가 53필드 복원(`h_to_json.py:25-77`).
- 병목: cin_load는 레이어마다 전체 입력 재로드(conv2d는 채널타일마다 반복, `:727-735`) → 메모리 바운드 레이어에서 latency 지배(`latency_est.py:45-46`).

---

## 7. 모듈: 경량 연산 — `act_and_bn`(relu+bn)/`pool`/`upsample`/`add`/`concat` (modules + util.h)

### 7.1 역할 + 상위/하위
- **역할**: conv 전/후 경량 연산. ReLU/ReLU6+BatchNorm+bias(act_and_bn), 2×2 maxpool(pool), 2× bilinear upsample(upsample), residual 가산(add), 채널 concat(concat). 모두 line-buffer stencil 또는 element-wise.
- **상위**: engine task. **하위**: util.h 템플릿(`maxpool_w2`).

### 7.2 데이터플로우 (ENet engine)
```
SA → upsample → concat ← pool ← cin_load_prev
          concat → add → act_and_bn(relu/bn/bias) → cout_write   (arch_connectivity/ENet.json:16-29)
```

### 7.3 Function call stack
codegen 조립: `act_and_bn_ENet.cpp`, `pool_ENet.cpp`, `upsample.cpp`, `add.cpp`, `concat.cpp`(모델별 변형은 `_<model>` 접미, `code_template.py:499-501`). pool은 `util.h:159 maxpool_w2` 래핑.

### 7.4 대표 코드 위치
`util.h:159-406`(maxpool_w2: 2×2 line-buffer max), `modules/{act_and_bn_ENet,pool_ENet,upsample,add,concat,cout_write}.cpp`.

### 7.5 대표 코드 블록
```cpp
// maxpool_w2: 동적 타일링용 하드코딩 mux (util.h:211-219)
if (layer_in_w_t==MAX_IN_W_T) tmp1=line_buf1[dup][MAX_IN_W_T-1];
else if (==MAX_IN_W_T/2) ... /4 ... /8   // ← 특정 width만 합성 지원
// 4-입력 max (util.h:349-351)
mux_0_0=max(line_buf2[dup][WS-1],line_buf2[dup][WS-2]); ...; sums[dup]=mux_1_0;
// stride 출력 마스킹 (util.h:366-367): col_skip/row_skip = (trans_cnt%stride!=0)
```

### 7.6 마이크로아키텍처 + 정량/병목
- **act_and_bn**: relu/relu6/bn/bias 통합, 입력채널타일 누적 완료 시점에만 후처리 적용(추정, ESDA quantize와 동형). BN gamma/beta는 bias_load→act_and_bn FIFO 전달.
- **pool**: `maxpool_w2` line_buf1[UNROLL][IN_W_T] + line_buf2[UNROLL][WS], II=1.
- **병목**: util.h `maxpool_w2`의 width 하드코딩 mux(`:211-219`)는 **MAX_IN_W_T/{1,2,4,8}만 동적 타일 지원** → 임의 width 미지원(한계). (1차 요약의 `stencil_w3 assert(26/50/98)`와 동일 성격의 하드코딩.)

---

## 8. 모듈: 결과 기록 + 호스트 통합 — `cout_write` + `libsacc`/`SDx_project`

### 8.1 역할 + 상위/하위
- **역할(cout_write)**: FIFO 결과를 512b로 패킹해 `global_cout` burst write. **역할(libsacc)**: FPGA 가속기를 TensorFlow custom op `Sacc`로 노출, XRT/OpenCL로 DRAM 적재·커널 실행.
- **상위**: engine task(cout_write) / TF 런타임(libsacc). **하위**: AXI / XRT.

### 8.2 데이터플로우
```
fifo → cout_write(패킹) → global_cout(DRAM) ─host─→ TF 텐서
libsacc: TF cin → reformat → DRAM(q/q2/q3 3큐 병렬) → top_kernel ×3 → reformat → TF
```

### 8.3 Function call stack
`modules/cout_write.cpp` (engine 말단). 호스트: `libsacc/src/libsacc.cpp`(load_inst/weights/bias, init, 3큐 enqueueTask), `libsacc/inc/sacc.hpp`(REGISTER_OP("Sacc"), OpKernel::Compute).

### 8.4 대표 코드 위치
`modules/cout_write.cpp`, `libsacc/inc/sacc.hpp`(REGISTER_OP, Compute), `libsacc/src/libsacc.cpp`(krnl_vadd/2/3 3 인스턴스, setArg 0~5).

### 8.5 대표 코드 블록 (근거: 1차 요약 통합 + sacc.hpp 구조)
- 파일 경로 상수: `inst_file=config/network.insts`, `weight=data/weight_reorg.dat`, `bias=data/bias_reorg.dat`, `xclbin=config/binary_container_1.xclbin`.
- setArg: arg0/1/2=cin(prev/cout 공유), 3=weight, 4=bias, 5=config — `top_kernel` 포트순서 대응.
- 3개 큐(q/q2/q3) = 같은 top_kernel 3 인스턴스로 멀티 DDR뱅크 오버랩.

### 8.6 마이크로아키텍처 + 정량/병목
- **TF offload**: OpenPose 등 애플리케이션의 CNN을 FPGA로 위임(`README.md:9`).
- 병목/리스크: libsacc는 **XRT 2018.3/SDAccel 구버전 기준**(`libsacc/README.md`) → 최신 Vitis 정합성 **확인 불가**. 호스트 reorg가 SW 측 오버헤드(추정).

---

## 9. 모듈: DSE + inst 생성 (codegen 자동화 두뇌) — `dse.py`/`latency_est`/`resource_est`/`inst_gen`/`extract_info`

### 9.1 역할 + 상위/하위
- **역할**: ONNX→csv 융합(extract_info) → 설계공간 sweep+Pareto(dse) → 명령어 직렬화(inst_gen). 갈래 C(params) 입력 생성.
- **상위**: `run.sh`. **하위**: `latency_est`/`resource_est`/networkx/multiprocessing.

### 9.2 데이터플로우
```
ONNX ─extract_info(PBPredictor)→ csv(conv타입+en비트+융합)
csv ─dse(get_layers_configs→MACS/IN_DATA_LAYOUT, param_sweep→latency_est+res_est, Pareto)→ arch.json + 타일모델
arch.json+모델 ─inst_gen(en마스크+SA카운터+conv_type+주소)→ instructions.dat(7줄/inst)
```

### 9.3 대표 코드 위치 + 식 (서브분석 통합, 모두 라인 근거)
- **extract_info**: conv타입 `get_conv_info`(`:299-370`): group==1&dil==1→NConv(`:312`), group==1&dil>1→DConv(`:323`), group>1→DWConv(`:334`), ConvTranspose→TConv(in/out swap, `:345-347`). 융합 `get_insts`(`:495-715`, fan-out>1에서 중단 `:598-601`). en벡터 `get_enabled_modules`(`:717-738`).
- **dse**: MACS=`in·out·kh·kw·out_h·out_w`(`:112`), TConv=입력차원(`:134`), padding=`dilated_kernel-stride`(`:119-120`), **IN_DATA_LAYOUT=2 if (filter-stride==0)||(filter==tstride) else 1**(`:170`). 정밀도 dw 32/16/8→float/ap_fixed16/8(`:414-437`), K_T=16(`:441`). LANE∈{2,4,8,16,32,64}(`:449`), SA_ROWS=OUT_NUM_T약수(`:469`), SA_COLS=IN_W_T약수(`:470`), SA_SIMD=LANE(`:471`). pruning `IN_W_T%COLS==0 & IN_NUM_T%SIMD==0 & IN_NUM_T%ROWS==0`(`:759`), `DSP>DSP_THRES·보드DSP`제외(`:762`), FRE=220(`:772`). Pareto: latency→DIM_SUM(=R+C+S)→min BRAM(`:513-570`).
- **latency_est**: dram_latency=250(`:21`), **conv compute=`in_t·out_t·out_h_t·out_w_t·fh·fw/(rows·cols·simd)`**(`:117`), layer_latency=`extra+stage·total_iter`(`:219`), peak=`cols·rows·simd`(`:233`).
- **resource_est**: DSP/MAC float=5/fixed16=1/fixed8=0.25(`:27,29,31`), **DSP=SA_ROWS·SA_COLS·SA_SIMD·DSP/MAC**(`:35`), BRAM=cin(×2)+weight+point_conv(5버퍼)+cout(×2)(`:40-57`), SDP모델(`:3-10`).
- **inst_gen**: en마스크 `conv<<2+relu<<3+pool<<5+upsample<<6+bias<<7+bn<<10+prev<<11+concat<<16+add<<17`(`:163`). SA카운터 `task_num1=⌈in/in_t⌉⌈out/out_t⌉⌈in_h/h_t⌉⌈in_w/w_t⌉`(`:32`), `local_accum=in_t/simd·fh·fw`(`:34`), `local_reg=(h_t/stride)(w_t/cols/stride)(out_t/rows)`(`:35`). conv_type **NConv=0/DConv=2/TConv=1**, **k_num=tstride²**(`:502,521,532`). 직렬화 7줄/inst(`:580-589`).

### 9.4 정량/병목
- 멀티프로세스 sweep(CPU 90%, `dse.py`). analytical 모델(FRE 220MHz·DRAM 250cy 고정) → 실측 괴리 가능(추정).
- 명령어 포맷이 HLS 디코드(`FlexCNN.cpp:564-643`)·inst_gen(`:580-589`)·h_to_json(`:25-77`) 3곳 분산 정의 → 동기화 리스크.

---

## 10. 모듈 한눈 요약 표

| 모듈 | 파일 | 핵심 함수(라인) | 역할 | 대표 정량(ENet 16×14×8 dw32) |
|---|---|---|---|---|
| SA codegen | systolic_array_kernel/codegen.py, desp_gen.py | run(codegen:101), cnn_pass(desp:14) | VSA descriptor→SA 6파일 | LOCAL_REG=448, ACCUM=72 |
| PE 어레이 | SA/2DPE_U1.cpp | PE_MAC(:9), compute(:728) | output-stationary 16×14 PE | 1792 float MAC/cyc |
| 피더/콜렉터 | SA/2DDataFeed/Collect_U1.cpp | Feed(:9), Collect(:185) | 데이지체인 cin↓/wt→/cout↓ | ENGINE_NUM [14,16],[14] |
| engine 조립 | code_template.py | flexcnn_tasks(:616), fifos(:585) | arch그래프→DATAFLOW 배선 | data FIFO depth128 |
| cin/weight 로드 | modules/cin_load.cpp(FlexCNN.cpp:430) | cin_load(:430), 디코드(:564) | URAM더블버퍼+SIMD언팩+inst디코드 | CONFIG_PARAMS=53 |
| 경량연산 | modules/act_and_bn/pool/upsample/add/concat | maxpool_w2(util.h:159) | relu/bn/pool/upsample/add | width mux /1,/2,/4,/8 |
| 결과기록/호스트 | modules/cout_write.cpp, libsacc | Sacc::Compute | 512b패킹+TF offload(3큐) | XRT 2018.3(구버전) |
| DSE | dse/dse.py,latency_est,resource_est | param_sweep(:730), res_est(:26) | sweep+Pareto+자원/지연 | DSP=R·C·S·{5/1/0.25} |
| inst 생성 | inst_generation/inst_gen.py | get_en_value(:110), get_SA_insts(:13) | en마스크+SA카운터+conv_type | NConv=0/TConv=1/DConv=2 |
| ONNX 융합 | graph_translation/extract_info.py | get_conv_info(:299), get_insts(:495) | conv분류+레이어융합+en | group/dilation→4타입 |
| params codegen | systolic_array_kernel/h_to_json.py | get_mem_params(:9) | inst→params.h+cnn_features | mem_type 0~4 14매크로 |

---

## 11. 읽기 순서 / 코드 추적 순서

1. **운영 전체상**: `README.md:39-80` + `run.sh` + `generate_design.sh` → 6단계 파이프라인.
2. **SW 진입**: `extract_info.py` get_conv_info(`:299`)/get_insts(`:495`) → conv 분류·융합 직관.
3. **DSE**: `dse.py` get_layers_configs(`:18`)·IN_DATA_LAYOUT(`:170`)·param_sweep(`:730`) → `latency_est.py:117,219` → `resource_est.py:35`.
4. **명령어**: `inst_gen.py` get_en_value(`:163`)+get_SA_insts(`:32-37`)+직렬화(`:580-589`) → `h_to_json.py:25-77` 복원.
5. **SA 본체**: `desp_gen.py` cnn_pass(`:14-130`) → `2DPE_U1.cpp` PE_MAC(`:36-59`)+compute(`:798-895`)+격자배선(`:6552-6592,7136-7177`).
6. **engine 조립**: `code_template.py` flexcnn_fifos(`:585`)/flexcnn_tasks(`:616`) + `arch_connectivity/ENet.json` → engine DATAFLOW가 어떻게 그래프에서 나오는지.
7. **모듈 본체**: `FlexCNN.cpp` cin_load 디코드(`:564-643`) + `SA.cpp` kernel 호출(`:242`) + `util.h` maxpool_w2(`:159`).
8. **호스트**: `libsacc/inc/sacc.hpp`(Sacc op) → `libsacc/src/libsacc.cpp`(3큐).
9. **대표 수치 확인**: `design_desp.json:89-131` + `cnn_features.json` + `boards/u250.json`.

---

## 12. 병목 후보 & 병렬도/DSE 노브

### 12.1 병목 후보
1. **float MAC당 DSP=5**(`resource_est.py:27`): 16×14×8×5 = 8960 DSP로 U250(12288)의 73% → DSP_THRES 0.6 초과 가능 → dw=16/8 양자화로 1/5·1/20 절감 여지. dw=32가 자원 1차 제약(추정).
2. **maxpool/stencil width 하드코딩 mux**(`util.h:211-219`): MAX_IN_W_T/{1,2,4,8}만 동적 타일 → 임의 해상도 미지원(합성 실패 가능).
3. **cin_load 레이어별 전체 재로드**(`FlexCNN.cpp:727-735`): conv2d 채널타일마다 입력 반복 read → 메모리 바운드 레이어 latency 지배.
4. **SA 채움/비움(데이지체인)**: feed/collect 1D 체인 + compute_drain(`latency_est.py:118`)이 작은 타일에서 효율 저하.
5. **명령어 포맷 3곳 분산**(FlexCNN.cpp 디코드 / inst_gen 직렬화 / h_to_json 복원): 한 곳 변경 시 동기화 깨짐.
6. **design point/params.h 불일치**(`design_desp.json:171` LOCAL_REG=448 vs 빌드 params 128): 재생성 강제 안 하면 잘못된 버퍼 크기.
7. **libsacc XRT 2018.3 구버전**: 최신 Vitis 정합성 확인 불가.

### 12.2 병렬도/DSE 노브
- **SA 형상 `SA_ROWS×SA_COLS×SA_SIMD`**: dse가 OUT_NUM_T약수×IN_W_T약수×LANE로 sweep(`dse.py:469-471`), Pareto가 latency·DIM_SUM·BRAM로 선택. 대표=16×14×8.
- **타일 `IN_NUM_T/OUT_NUM_T/IN_H_T/IN_W_T`**: 짝수·약수 후보(`dse.py:445-448`), dynamic_tiling_level 0/1/2로 레이어별 가변(`latency_est.py:299-320`). 레이어간 의존성(connected_components)으로 공유 제약.
- **데이터폭 dw 32/16/8**: float/ap_fixed16/ap_fixed8 + DSP/MAC 5/1/0.25 트레이드오프(`dse.py:414-437`, `resource_est.py:27-31`).
- **mem_type 0~4**: 14개 온칩 버퍼(Head/Engine/cin/weight/bias/cout)의 BRAM(0)/URAM(1) 배치(`h_to_json.py:516-590`). 대표 mem_4 = Engine0+cin/weight/bias/cout URAM.
- **IN_DATA_LAYOUT 1/2**: halo 패딩 vs no-pad 타일(`dse.py:170` ↔ `FlexCNN.cpp:54-81`).
- **arch_connectivity 그래프**: 모듈 추가/제거로 engine 파이프 재구성(codegen 자동 배선, `code_template.py:494-693`).
- **conv_type/k_num**: NConv/DConv/TConv를 동일 SA에서 sub-kernel 분해(`inst_gen.py:502-534`, `2DPE_U1.cpp:798`).

---

## 13. 근거 표기 규칙

- 본문 `파일:라인`은 Read/Grep으로 확인한 실제 소스 근거(예: `2DPE_U1.cpp:36-59`, `dse.py:170`, `code_template.py:616`). `auto_compile/.../HLS/` 트리를 1차 인용(가장 완비). TAPA_1/TAPA_2 변형은 동형 codegen(구조 동일, 라인은 HLS 기준).
- **"추정"**: 코드 구조상 합리적이나 명시 라인 없음(예: act_and_bn 누적 후처리 시점, DSP 73% DSE 제외 가능성, 데이지체인 latency 지배).
- **"확인 불가"**: 분석 범위 소스에 없음(예: 합성 PPA 리포트 부재, QAT 파이프라인, libsacc·최신 Vitis 정합성, 빌드 params.h vs descriptor 불일치 원인).
- 정량 1차 근거 인스턴스 = repo 잔존 `design_desp.json`(SA 16×14×8, ROW_IL=4, COL_IL=8) + `h_to_json.py`(CONFIG_PARAMS=53, SIMD=8) + `boards/u250.json`(DSP 12288). **1차 요약의 SA 4×6×4는 과거 design point로 정정**.

---

*근거 파일(절대경로)*:
`\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\CNN-Accel\FlexCNN\`
`README.md, generate_design.sh, auto_compile\run.sh`,
`auto_compile\graph_translation\{extract_info.py, mapper.py}`,
`auto_compile\dse\{dse.py, latency_est.py, resource_est.py, utils.py}`,
`auto_compile\inst_generation\inst_gen.py`,
`auto_compile\design_generation\generate.py`,
`auto_compile\design_generation\HLS\design_prepare.sh`,
`auto_compile\design_generation\HLS\systolic_array_kernel\{codegen.py, desp_gen.py, code_template.py, h_to_json.py}`,
`auto_compile\design_generation\HLS\systolic_array_kernel\output\design_desp.json`,
`auto_compile\design_generation\HLS\SA\{2DPE_U1.cpp, 2DDataFeed_U1.cpp, 2DDataCollect_U1.cpp, 2DDataFeedCollect_U1.cpp, common_header_U1.h}`,
`auto_compile\design_generation\HLS\FlexCNN.cpp`,
`auto_compile\design_generation\HLS\modules\{cin_load.cpp, weight_load.cpp, SA.cpp, act_and_bn_ENet.cpp, pool_ENet.cpp, upsample.cpp, add.cpp, concat.cpp, cout_write.cpp, cin_load_prev.cpp, bias_load_ENet.cpp}`,
`auto_compile\design_generation\HLS\design\src\{util.h, params.h}`,
`auto_compile\data\arch_connectivity\ENet.json`, `auto_compile\data\boards\u250.json`,
`libsacc\inc\sacc.hpp`, `libsacc\src\libsacc.cpp`.
