# ESDA 코드베이스 정밀 분석

> 분석 대상: `REF/CNN-Accel/ESDA`
> 작성 원칙: 실제 소스 Read/Grep 후 `파일:라인` 근거 표기. 라인 근거 없는 추론은 "추정", 코드로 확인 불가는 "확인 불가"로 명시.
> 분석 제외(이름만 언급): `.bit`/`.hwh`(비트스트림), `weight.h`(코드 생성물·거대 ROM 데이터), `data/*_w.txt`/`*_s.txt`(gen된 weight/scale txt), `.npy`/`.npz`(데이터셋), `benchmark_results/`·`baseline_results/`(실험 산출물 디렉토리, 평가 스크립트만 분석), `software/dataset/preprocess/dvs/PyAedatTools`(third-party AEDAT 파서), `.Xil`/build 산출물.

---

## 1. 개요

ESDA(추정: **E**fficient **S**parse-dense **D**ataflow **A**ccelerator)는 이벤트 기반 비전(event-based vision)용 희소(sparse) CNN을 FPGA로 가속하는 HW/SW 통합 프레임워크다.

- **목적**: 이벤트 카메라 입력처럼 공간적으로 희소한(spatially sparse) 입력에 대해, MobileNetV2 계열의 inverted-residual 블록(1x1 → 3x3 depthwise → 1x1)을 토큰(token) 기반 dataflow로 처리하여, 0이 아닌(non-zero) 픽셀만 연산하는 가속기를 생성한다. 입력 희소성 근거: `cfgs` 설정의 `"input_sparsity": 0.0497`(`SEE/ESDA/hardware/cfgs/DVS_1890_shift16-zcu102_80res.json:8` — ESDA 본체엔 cfg 미포함이라 동형 SEE 사본 인용), 즉 약 5% 픽셀만 유효.
- **원논문 추정**: 클래스명·README 구조상 "Efficient Sparse-dense Dataflow Accelerator for event-based vision" 계열. 정확한 논문/저자(MIT-HAN-Lab 여부 포함)는 코드 내 명시가 없어 **확인 불가**. (README/주석에 인용·저자 표기 없음. `hardware/README.md` 전체에 논문 링크 없음.)
- **타깃 보드**: ZCU102 + PYNQ 오버레이. 근거 `hardware/README.md:5` "We use ZCU102 board with PYNQ overlay", `hardware/HWConfig/zcu102.json:1-5`(dsp 2520, bram36 912).
- **워크플로**: (a) `optimization`에서 ILP/DSE로 레이어별 병렬도(parallelism) 결정 → `en-result.json` 생성, (b) `hardware/gen_prj.py`로 HLS 프로젝트 복제, (c) 프로젝트 내 `gen_code.py`가 cfg.json → `top.cpp`/`para.h`/`weight.h` 생성, (d) Vitis HLS·Vivado 합성, (e) `board/`의 PYNQ 스크립트로 보드 실측. 근거: `hardware/README.md:9-181`.

> **중요 관찰(우리 프로젝트 직접 관련)**: `optimization/pipeline.py`의 경로 상수가 `/vol/datastore/eye_tracking/eventModel`, `/vol/datastore/eye_tracking/eventHWConfig`, `/vol/datastore/eye_tracking/eventNetHW`로 설정되어 있고(`SEE/ESDA/optimization/pipeline.py:6-8`), 모델명이 `0324_cfg1551_biasBit16` 등(`:11`)이다. ESDA가 실제로 **eye-tracking 데이터/모델 파이프라인**에 연결되어 운용되었음을 시사(추정).

---

## 2. 디렉토리 구조 (자체 소스 + 제외 이유)

```
ESDA/
├── optimization/                  # SW: DSE/ILP 자원배분 솔버 (부분 체크아웃)
│   ├── eventnet.mk                # 모델×보드 배치 실행 Makefile  [분석]
│   └── solver/scip_solver.py      # SCIP ILP 솔버 래퍼            [분석]
│   # (주의) base_solver/utils/dse_var/dse_constr/eventnet.py = import만 존재, 파일 없음 → 확인 불가
├── hardware/                      # HW: HLS 템플릿 + 코드 생성 + 평가
│   ├── README.md                  # 빌드·평가 절차                [분석]
│   ├── gen_prj.py                 # 프로젝트 복제기               [분석]
│   ├── common.py                  # 네이밍 람다 헬퍼              [분석]
│   ├── baseline_extract.py        # csynth/cosim/power 리포트 추출 [분석]
│   ├── benchmark_extract.py       # sparse/dense 벤치 CSV 추출    [분석]
│   ├── template_e2e/              # 핵심 HLS 커널 템플릿          [정밀 분석]
│   │   ├── conv.h                 #   sparse conv·quant 커널
│   │   ├── conv_pack.h            #   inverted-residual 모듈 조립
│   │   ├── linebuffer.h           #   토큰 기반 line buffer
│   │   ├── mem.h                  #   mask→token, M2S, weight load
│   │   ├── type.h                 #   T_K(토큰)/BundleT 정의
│   │   ├── top.h / top.cpp.tpl    #   top 인터페이스/플레이스홀더
│   │   ├── gen_code.py            #   cfg→top.cpp/para.h/weight.h
│   │   ├── gen_input.py/gen_data.py/sw_e2e.py/common.py/tb.cpp/tb_func.h
│   │   ├── fixgmp.h               #   (third-party GMP 호환 헤더, 이름만)
│   │   ├── Makefile / prj/*.tcl   #   HLS/Vivado tcl
│   │   └── cfg.json               #   템플릿 더미 cfg
│   ├── template_e2e_roshambo/     # Roshambo 변형 템플릿(MW=64)  [동형, 이름만]
│   ├── HWConfig/zcu102*.json      # 자원 모델(dsp/bram)          [구조만]
│   └── benchmark_results/, baseline_results/  # 실험 산출물      [제외: 평가 스크립트만]
├── eventNet/
│   ├── DSE/                       # en-result.json 등 DSE 출력   [제외: 생성물]
│   └── hw/<dataset>/full/         # 생성된 프로젝트 사본          [제외: gen_code 산출물]
└── software/                      # 학습/전처리 (PyAedatTools = third-party 제외)
```

- **제외 근거**: `weight.h`/`data/*.txt`는 `gen_code.py`/`gen_data.py`가 생성(생성 로직만 분석). `eventNet/DSE`·`eventNet/hw`는 솔버·gen_prj 출력. `benchmark_results`·`baseline_results`는 합성/실측 로그(추출 스크립트만 분석).

---

## 3. 핵심 모듈 정밀 분석

### 3.A 최적화/DSE 솔버 (SW)

#### 3.A.1 `solver/scip_solver.py` — `ScipSolver` (SCIP 기반 ILP 자원배분)
파일: `optimization/solver/scip_solver.py` (전체 147줄), `BaseSolver` 상속(`:12`).

- **역할**: 레이어별 병렬도/자원 변수를 정수계획(MILP)으로 풀어 latency를 최소화. 의존성: `pyscipopt`(`:6`), 그리고 프레임워크 측 `utils.dse_constr.DSEConstr`·`utils.dse_var.DSEVar`·`solver.base_solver.BaseSolver`(`:7-9`).
- **`__init__`(`:19-31`)**: 기본 설정 주입 — `scip-use_cmd=False`(`:27`), `scip-maxnthreads=8`(`:28`), `scip-timelimits=None`(`:29`), `scip-quiet=True`(`:30`). 즉 기본은 pyscipopt 인-프로세스, 8스레드 병렬, 무제한 시간.
- **`build`(`:33-62`)**: 모델 조립.
  - `pyscipopt.Model()` 생성(`:37`).
  - 변수 추가: `var.in_scip`인 항목만 `addVar(name, vtype, lb, ub)`(`:39-44`). `var.is_obj`이면 목적변수로 지정(`:45-46`).
  - 제약 추가: `constr.in_scip`인 항목에 대해 `constr.get_constr_expr(...)`로 식 생성 후 `addCons`(`:48-52`). → DSE 제약(자원 상한 등)이 외부 `DSEConstr` 객체에 캡슐화됨(**해당 클래스 구현은 미포함 → 세부 제약식 확인 불가**).
  - 목적함수: 단일 목적변수 `setObjective(scip_obj)`(`:54-55`). objective는 cfg의 `"obj"`(latency 추정치)에 대응(추정; cfg 예시 `"obj": 2948.8`, `"lat_max": 2748`은 `cfgs` json에서 확인).
  - 모델 직렬화: `.cip` 포맷으로 `writeProblem`(`:57,62`).
- **`solve`(`:64-146`)**: 두 경로.
  - **CLI 경로**(`scip-use_cmd=True`, `:70-123`): `$SCIPOPTDIR/bin/scip`을 subprocess로 호출(`:71-74`). 멀티스레드면 `concurrentopt`, 아니면 `optimize`(`:82-85`). 해를 `.sol`로 기록 후 다시 읽어 `checkSol`로 검증 — valid / integrality violated / other violation 3단계 로깅(`:104-111`). 타임아웃·에러는 `status="fail"`(`:112-123`).
  - **API 경로**(기본, `:124-136`): `hideOutput`(`:126`), `parallel/maxnthreads`·`limits/time` 파라미터 세팅(`:128-132`), `optimize()`(`:134`), `getBestSol()`→`writeSol`(`:135-136`).
  - **해 변환**(`:138-145`): 모든 변수 순회, `vtype=="INTEGER"`면 `int(round(val))`, 아니면 float로 `sol_dict`에 저장 → `self.solution`, `status="done"`.
- **핵심 통찰**: 본 솔버는 **순수 ILP 래퍼**이며, 실제 latency/자원 모델식과 변수 정의는 미포함 프레임워크(`DSEVar`/`DSEConstr`/`BaseSolver`)에 있다. 따라서 "어떤 자원을 어떻게 배분하는가"의 수식 자체는 **이 체크아웃만으로는 확인 불가**. 다만 출력(`en-result.json`)의 `layers[].parallelism`(예: `[PIC, PC, POC]`)이 ILP 결정변수임은 하류 코드(`gen_code.py`)로 역추적 가능(아래 3.B.4).

#### 3.A.2 `eventnet.mk` — 배치 DSE 실행
파일: `optimization/eventnet.mk` (136줄).

- 모델×보드 데카르트곱을 `eventnet.py`/`eventnetblock.py`로 일괄 실행하는 Makefile. 모델 목록 `DVS_mobilenet_0703/0707_0p75/0707_0p5`(`:3`), 보드 목록 `zcu102_50res ... zcu102_full`(`:4`), 곱집합 `MODEL_HW_LIST`(`:5`).
- 타깃군: `all`(기본 `eventnet.py`, `:63-74`), `nas`(NAS 모델, `--nas`, `:78-90`), `bm`/`e2e`(블록 단위 `eventnetblock.py`, `:94-120`), `bl`(baseline, `:124-135`). 각 타깃은 `en-result.json` 존재 시 skip(`:64-66`)하는 idempotent 구조.
- **주의**: 실제 솔버 드라이버 `eventnet.py`/`eventnetblock.py`는 체크아웃에 **부재**(Glob 결과 0건) → DSE 목적함수·자원 모델의 진입점은 **확인 불가**.

### 3.B 하드웨어 코드 생성 + HLS 커널 (HW)

ESDA HW의 핵심 아이디어: **희소 토큰 스트림(T_K) + dataflow 파이프라인 + 패킹된 정수 MAC**. 각 유효 픽셀 좌표를 토큰 `T_K{end,x,y}`(`type.h:2-6`)로 표현하고, 0이 아닌 픽셀의 활성값만 스트림으로 흘려보내며, 토큰의 `(x,y)`로 3x3 line buffer 윈도우를 재구성한다.

#### 3.B.0 자료형 — `type.h`
- `T_K`(`type.h:2-6`): `ap_uint<1> end; ap_uint<8> x; ap_uint<8> y;` — 좌표가 8비트라 **최대 255×255 해상도** 제약(라인 근거 그대로). end=1이 스트림 종료 토큰.
- `BundleT<N,T>`(`type.h:8-11`): `T data[N]` — 병렬도 N개의 채널을 한 번에 묶는 패킷 타입. dataflow FIFO의 원소.
- `T_OFFSET = ap_uint<4>`, `#define end_3x3 15`(`type.h:15-16`): 3x3 윈도우 내 9개 위치(0~8) 인덱스 + 종료 마커 15.

#### 3.B.1 토큰 기반 Line Buffer — `linebuffer.h` (sparse 핵심 ①)
세 변형이 있다. 공통 구조: 3행 순환 버퍼(`BUFFER_ROWS=3`)와 `valid[BUFFER_ROWS][WIDTH]` 비트맵으로 "어떤 셀에 실제 데이터가 들어왔는가"를 추적하여, **희소 입력에서 윈도우가 채워졌는지** 판단.

- **`conv_3x3_line_buffer_stride1_serial`(`linebuffer.h:3-177`)**:
  - 버퍼: `line_buff[3][WIDTH*IC/PI]`(`:38`), 토큰 FIFO `token_fifo[FIFO_DEPTH=WIDTH*3]`(`:37,20`), `valid` 비트맵(`:43-44`, dim2 complete reshape).
  - 메인 루프(`:71`): `rep < HEIGHT*WIDTH*2`. 매 스텝 최신 토큰을 읽어(`:75-77`) FIFO에 적재, 가장 오래된 토큰을 봐서 `jump_y`(행 진행량) 계산(`:80`). `jump_y>0`이면 새로 진입할 행의 `valid`를 0으로 클리어(`:82-94`) — 순환 버퍼 행 재사용.
  - **출력 조건**(`:99-103`): `out_valid_one_line = (y_delta==1 && x차>=1)`, `out_valid_multi_line = (y_delta>=2)`, 또는 `end`. 즉 최신 토큰이 oldest보다 충분히 진행되어 3x3 윈도우 중심행이 확정될 때 출력.
  - **데이터 적재**(`:122-136`): `data_read_enable`이면 `IC/PI`개 패킷을 읽어 `line_buff`에 repack 저장.
  - **윈도우 방출**(`:138-175`): 중심 oldest_token 기준 `ki,kj∈[-1,1]` 9개 위치 순회. 패딩 검사(`:144-147`) + `valid_point` 검사(`:151-153`) 후 **유효한 위치만** `offset_s.write(offset)`로 위치 인덱스를 알리고 채널들을 `act_out`으로 방출(`:154-170`). 윈도우 끝에 `offset_s.write(end_3x3)`(`:174`). → **0인(존재 안 하는) 이웃은 아예 스트림에 싣지 않음** = 희소 연산의 본질.
- **`conv_3x3_line_buffer_stride2_fifo_serial`(`:179-455`)**: stride-2용. 짝/홀 행을 **분리된 even/odd 토큰 FIFO**로 관리(`:217-218`)하고, `key = x + y*WIDTH`로 두 FIFO의 우선순위를 비교해(`:321-347`) 순서 머지. 출력 토큰은 `x>>1, y>>1`로 다운샘플(`:386-387`). 나머지 valid/윈도우 로직은 stride1과 동형.
- **`conv_3x3_line_buffer_first_layer`(`:457-667`)**: 첫 레이어 전용. `valid`를 행당 `ap_uint<WIDTH>` 비트벡터로(`:511`) 압축, `line_buff` dim1 complete partition(`:509`), 전체 루프 `#pragma HLS PIPELINE`(`:558`). 출력은 9-원소 윈도우 묶음 `BundleT<9, ap_int<PI*AW>>`을 **그대로** `act_out`(`:460,665`) — 첫 레이어는 dense한 3x3 conv를 받으므로 윈도우 전체를 한 패킷으로 전달.

#### 3.B.2 정수 MAC 커널 — `conv.h` (sparse 핵심 ②)
- **`DSP_AM`(`conv.h:1-8`)**: DSP 1개에 2-MAC 패킹의 핵심 프리미티브. `(in1+in2)*in3` 형태 — `add_temp = in1+in2; mul_temp = add_temp*in3`(`:5-6`). Xilinx DSP48의 (A+D)×B 사전가산기를 이용해 한 곱셈에 두 가중치를 실어 보내는 트릭.
- **`conv_3x3_dw_kernel`(`:10-57`)**: 3x3 depthwise. 가중치 ROM `w_buffer[9][IC/PI]`를 BRAM `rom_2p`로 바인딩(`:19`). 9개 위치를 unroll(`:36-39`), 채널 PI를 unroll하여 `psum += activation*weight`(`:40-51`). II=1 파이프라인(`:34`). (단순 버전; 실제 top에서는 아래 serial 버전 사용.)
- **`conv_3x3_dw_kernel_serial`(`:325-387`)**: line buffer의 `offset_s`와 연동되는 **희소 dw 커널**. `psum_buffer[IC]`를 0으로 초기화(`:350-356`) 후, 토큰마다 최대 10회(`:362`) `offset = offset_s.read()`로 윈도우 내 유효 위치를 받아(`:363`) `end_3x3`이면 종료(`:364`), 아니면 그 위치의 가중치 `w_buffer[offset][ic]`로 누적(`:371-373`). 유효 위치만 누적하므로 **희소 입력에서 곱셈 횟수가 nz 수에 비례**. 마지막에 `psum_buffer`를 방출하고 0 리셋(`:377-385`).
- **`conv_1x1_kernel_dsp`(`:220-323`)**: 1x1 pointwise, **DSP 2-MAC 패킹** 적용. 활성값을 `act_buffer[IC/PI]`에 모은 뒤(`:259-273`), 출력채널을 PO/2 쌍으로 처리(`:284`): 두 가중치 `w_0,w_1`를 27비트 워드에 `w_1`을 상위 9비트(`<<18`)로 배치(`:299-302`)하고 `DSP_AM(w_1_shift, w_0_expend, in_expend)`로 한 번에 계산(`:303-304`), 48비트 결과의 하위/상위에서 두 부분곱을 분리해 두 psum에 누적(`:305-312`). w_buffer는 `rom_2p` BRAM + cyclic partition(PO/2)(`:229-232`).
- **`conv_3x3_kernel_dsp_first_layer`(`:390-487`)**: 첫 레이어 3x3 standard conv. 가중치를 **LUTRAM**으로(`:401`, dense·소형이라) 바인딩, 9×PO 가중치를 한 OC타일에 모아(`:442-446`) 동일한 DSP 2-MAC 패킹(`:465-478`)으로 처리.
- **`quantize`(`:59-123`)**: psum→int8 재양자화. `scale_buffer[IC]`에서 scale·bias를 비트필드로 분리(`:96-99`), `(psum+bias)*scale + round_shift`(`:102`) 후 `>>EXP`(`:106`), `[low,high]` 클리핑(`:108-109`), `relu` 옵션(`:112-117`). round_shift는 `1<<(EXP-1)`(`:67`).
- **`quantize_id_add`(`:125-218`)**: residual 가산 포함 재양자화. main psum 경로(`:184-194`)와 identity 경로 `id*id_scale>>EXP`(`:196-204`)를 각각 양자화 후 `final_sum = quantize_psum + quantize_id`(`:206`) → 클리핑. residual 블록의 skip-connection 합산.
- **`global_avgpool_linear`(`:490-559`)**: 마지막 GAP+FC 융합. 공간 누적 `sum[IC/PI]`(`:505,529-537`) 후 `out_buffer[oc] += s*w`(`:541-553`)로 분류 logit 직접 산출 → `c_out[i]`(`:555-558`). 비트폭은 정적 log2로 자동 산정(`:495-502`).

#### 3.B.3 inverted-residual 모듈 조립 — `conv_pack.h`
- **`conv_1x1_module`(`:1-22`)**: `#pragma HLS DATAFLOW`(`:8`) 하에 `conv_1x1_kernel_dsp` → `quantize`를 FIFO(`pusm_1` depth=2)로 연결(`:9-21`).
- **`conv_3x3_dw_module_stride1_serial`(`:49-85`)** / **`_stride2_serial`(`:87-121`)**: line buffer → `conv_3x3_dw_kernel_serial` → `quantize`를 dataflow로 연결. offset_stream(depth=9, `:72`)이 line buffer와 dw 커널을 동기화.
- **`conv_1x1_3x3_dw_1x1_stride1`(`:123-165`)** / **`_stride2`(`:167-209`)**: **MobileNetV2 inverted-residual 블록 전체**. expand(1x1) → dw(3x3) → project(1x1) 3단을 dataflow로 직렬 연결. 템플릿 파라미터 `PF_0/PF_1/PF_2`가 단계별 병렬도(=DSE 결정변수).
- **`conv_1x1_3x3_dw_1x1_stride1_residual`(`:211-268`)**: residual 블록. 입력을 `duplicate_stream`으로 분기(`:248`)하여 한 갈래는 3단 연산, 다른 갈래(`act_id`)는 깊은 FIFO `depth=(WIDTH+2)*OC/PF_2`(`:244`)로 지연시킨 뒤 `conv_1x1_module_residual`의 `quantize_id_add`로 합산(`:264-267`). residual 지연버퍼 깊이가 공간 크기에 비례 = BRAM 비용의 주요인.
- **`conv_3x3_first_layer`(`:271-302`)**, **`conv8`(`:304-314`)**: 첫 레이어/마지막 1x1.

#### 3.B.4 코드 생성기 — `gen_code.py` (SW↔HW 브리지, 정밀)
파일: `hardware/template_e2e/gen_code.py` (410줄). cfg.json(=DSE 결과) → `top.cpp`/`para.h`/`weight.h`를 자동 생성. **이것이 ILP 출력과 HLS 커널을 잇는 핵심 자동화.**

- **`gen_code`(`:306-376`)**: 태그별 코드 버퍼(`fifo/load/comp/store/para.h/weight.h`, `:307-310`) 생성 → prologue/per-layer/epilogue 순으로 append → `top.cpp.tpl`의 `/*gen_code-<tag>*/` 플레이스홀더(`:365-368`)에 삽입.
- **데이터셋별 양자화 비트 분기**(`:312-317`): `NCAL`이면 `CFG_SW/BW/EXP=32`(`:312-315`), `Roshambo`이면 `CFG_MW=64`(`:316-317`). 기본 AW=WW=8, PSUMW=32(`:320-322`).
- **`append_perlayer_code`(`:99-285`)**: 레이어 타입별 코드 방출.
  - `type=="conv"`(`:124-173`): `conv1`이면 `conv_3x3_first_layer`(`:135`), `conv8`이면 `conv8`(`:155`). PSUM 비트폭을 `CFG_AW+CFG_WW+ceil(log2(channels))`로 산정(`:128-131`) — 누적 오버플로 방지.
  - `type=="block"`(`:174-257`): stride·residual 조합으로 `conv_1x1_3x3_dw_1x1_stride{1,2}[_residual]` 함수명 결정(`:191-195`). 가정 검증 `stride∈[1,2]`, `not(stride==2 and residual)`(`:175-176`) — **stride2와 residual은 동시 불가**(공간 크기 변화 시 skip 불가). 3개 가중치/스케일 버퍼 선언(`:208-249`), residual이면 `ibuf`(`:250-257`).
  - `type=="linear"`(`:258-276`): `global_avgpool_linear`.
- **병렬도 매핑**(`:125-126,180`): `def_v_list = layer["parallelism"] + layer["channels"] + layer["input_shape"]`. 즉 cfg의 `parallelism`(ILP 결정)이 `CFG_<NAME>_PIC/PC/POC`로 직결 → HLS 템플릿 파라미터 = DSE 변수. **이 한 줄이 SW DSE ↔ HW 병렬도의 연결점.**
- 검증: `assert len(def_k_list)==len(def_v_list)`(`:283`).

#### 3.B.5 데이터 이동(M2S/마스크) — `mem.h`
- **`M2S_mask`(`:75-122`)**: 마스크 비트맵을 읽어, 비트=1인 좌표마다 `T_K` 토큰을 발행(`:101-115`) + 종료 토큰(`:117-121`). `M2S_mask_merge`(`:11-71`)는 토큰+활성값을 함께 방출하는 버전.
- **`mask_stride2`(`:124-176`)**: 인접 2×2를 OR로 묶어 stride2 토큰 생성(`:155-168`).
- **`read_sparse_input`(`:178-194`)**: AXI로 `num_nz*IC/PI`개 패킷만 읽음 — **0 픽셀은 메모리에서 아예 안 읽음**(대역폭 절감).
- **`duplicate_stream`(`:285-333`)**: residual용 스트림 복제. `PI==PO`면 단순 복제(`:303-312`), 아니면 LCM 버퍼로 폭 변환(`:313-330`).
- **`write_output`(`:262-283`)**: 토큰 종료까지 출력 패킷 기록.

#### 3.B.6 top 인터페이스 — `top.h` / `top.cpp.tpl`
- `top.h:27-29`: `top(act_in, act_out, mask, num_nz)` — 입력 활성·출력·마스크·nz개수.
- `top.cpp.tpl:18-28`: AXI master 3채널(`gmem0/1/2`, depth 65536) + AXI-lite 제어. `wrapper`(`:3-14`)가 `#pragma HLS DATAFLOW` 하에 생성 코드(fifo/load/comp/store) 삽입 지점.

### 3.C HWConfig 자원 모델 (구조만)
- `HWConfig/zcu102.json:1-5`: `{name, dsp:2520, bram36:912}`. `zcu102_80res.json:1-5`: `{dsp:2016, bram36:730}` — 80% 자원 예산 시나리오. 파일군: `zcu102_{40,50,60,75,80}res / full / mini / extra / roshambo.json` — DSE가 참조하는 보드별 DSP/BRAM 상한 테이블(이름·구조만). cfg.json의 `total_dsp`/`total_bram`(예 1804/1049)와 대조하여 ILP 제약 우변으로 사용(추정).

---

## 4. 데이터플로우

```
[mask 비트맵] --M2S_mask--> token(T_K) 스트림 ─┐
[희소 act]   --read_sparse_input--> act 패킷 ─┤
                                              ▼ (HLS DATAFLOW, 레이어별 병렬 파이프)
  conv1: line_buffer_first → conv_3x3_kernel_dsp_first → quantize
   │ (act 패킷 + token 패킷 + offset)
   ▼
  block_k: [1x1 expand → (line_buffer → 3x3 dw serial) → 1x1 project] (+residual: duplicate→id_add)
   │  각 단: kernel_dsp → quantize, FIFO depth=2로 연결
   ▼  ... (DSE가 정한 PIC/PC/POC 병렬도로 레이어별 II=1)
  conv8(1x1) → global_avgpool_linear → act_out(logit)
```
- 토큰과 활성값은 **별도 스트림**으로 흐르며, line buffer가 토큰의 (x,y)로 윈도우 유효성을 판정해 0 이웃을 건너뛴다(`linebuffer.h:138-175`). 전 파이프가 한 번에 켜지는 layer-pipelined dataflow(각 함수가 `top.cpp.tpl:6`의 단일 DATAFLOW 영역에 나열).

---

## 5. HW/SW 매핑

| 단계 | SW 산출물 | HW 반영 | 근거 |
|---|---|---|---|
| DSE/ILP | `en-result.json`의 `layers[].parallelism` | `CFG_<L>_PIC/PC/POC` (HLS 템플릿 파라미터) | `gen_code.py:125-126,180`; `scip_solver.py:138-145` |
| 자원예산 | `HWConfig/zcu102_*res.json` (dsp/bram) | ILP 제약 우변(추정) | `zcu102_80res.json:1-5` |
| 양자화 | cfg `CFG_SW/BW/EXP`, dataset 분기 | `quantize`/`quantize_id_add`의 scale/bias/shift | `gen_code.py:312-327`; `conv.h:96-106` |
| 코드생성 | `gen_code.py` | `top.cpp`/`para.h`/`weight.h` | `gen_code.py:306-376` |
| 프로젝트 | `gen_prj.py gen_full` | template_e2e 복제 + cfg 주입 | `gen_prj.py:81-85,107-110` |
| 실측 | `board/evaluate.py` 등 | PYNQ Overlay register_map | (SEE.md 참조) |

- DSP 2-MAC 패킹(`conv.h:299-312`)으로 INT8 MAC 처리량을 ×2 — DSP 예산이 ILP의 1차 제약(추정).

---

## 6. 빌드·실행

근거: `hardware/README.md:9-181`.
1. **DSE**: `optimization`에서 `eventnet.py`(부재) 실행 → `eventNet/DSE/<cfg>/en-result.json`.
2. **프로젝트 생성**: `python gen_prj.py gen_full --cfg_name <cfg> --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW`(`README.md:46`). gen_prj는 `en-result.json`을 `cfg.json`으로 복사(`gen_prj.py:107-110,52`)하고 `po2_axi` 옵션으로 첫/마지막 레이어 병렬도를 2의 거듭제곱+채널 약수로 내림(`po2_p`, `:12-20,45-51`).
3. **코드 생성**: `make gen`(내부적으로 `gen_code.py`, `gen_data.py`) → `top.cpp`/`weight.h`/`data/`.
4. **HLS IP**: `make ip_all` → `prj/`(`README.md:108-125`).
5. **Vivado HW**: `make hw_all` → `hw/top.bit`(`README.md:128-144`).
6. **평가**: `make evaluate_hw` 또는 `board/evaluate.py`(`README.md:149-181`).

리포트 추출: `baseline_extract.py`(csynth DSP/BRAM, cosim latency, vivado power; `:33-60`), `benchmark_extract.py`(sparse vs dense nzr별 cosim/실측 latency CSV; `:72-141`).

---

## 7. 의존성
- SW: `pyscipopt`(SCIP MILP, `scip_solver.py:6`), `numpy`, `json`, `xml.etree`(리포트 파싱, `baseline_extract.py:7`).
- HW: Vitis HLS(`hls_stream.h`, `ap_int.h`, `top.h:7-8`), **Boost** static metafunctions(`static_log2`/`static_min_max`/`common_factor`, `top.h:13-15` — 비트폭·LCM을 컴파일타임 산정), GMP 호환 `fixgmp.h`(third-party, 이름만).
- 보드: PYNQ, Vivado/Vitis 2020.x대(추정, 버전 명시 없음 → 확인 불가).
- 외부 도구 경로: `$SCIPOPTDIR`(`scip_solver.py:71`), 절대경로 `/vol/datastore/...`(`eventnet.mk:1-2`, `gen_prj.py:119`) — **하드코딩, 재현 시 수정 필요**.

---

## 8. 강점·한계

**강점**
- 토큰+마스크 기반 **진짜 희소 dataflow**: 0 픽셀을 메모리 read·MAC·윈도우 방출 전 단계에서 모두 제거(`mem.h:178-194`, `linebuffer.h:154-175`).
- **DSP 2-MAC 패킹**으로 INT8 처리량 배가(`conv.h:299-312`).
- ILP 기반 **레이어별 병렬도 자동 배분** → 자원예산 시나리오(40~80%)별 프로젝트 자동 생성(`gen_code.py`+`gen_prj.py`).
- 양자화·residual·stride·first/last 레이어를 모두 템플릿 + 코드생성으로 커버 → 모델 교체가 cfg 교체로 환원.

**한계**
- `T_K.x/y`가 8비트 → **최대 255×255 해상도**(`type.h:4-5`). 고해상도 비전엔 부적합.
- DSE 프레임워크 본체(`eventnet.py`, `DSEVar/DSEConstr`, `BaseSolver`)가 체크아웃에 **부재** → 목적함수·자원모델식 **확인 불가**(이 repo만으로 DSE 재현 불가).
- tcl/경로가 ZCU102+PYNQ에 **하드코딩**(`README.md:5,61`).
- 입력이 spatially-sparse라는 전제(이벤트 카메라)에 의존 — dense 입력에서는 이득 소멸(`benchmark_extract.py`가 sparse vs dense를 분리 측정하는 이유).
- stride2+residual 동시 불가(`gen_code.py:176`) — 아키텍처 표현력 제약.

---

## 9. 우리 프로젝트(ViT/Transformer FPGA 가속기 + XR 시선추적) 시사점

1. **eye-tracking 직접 연결 증거**: `pipeline.py:6-8`가 `/vol/datastore/eye_tracking/eventModel|eventHWConfig|eventNetHW`를 가리킨다. ESDA는 우리 도메인(이벤트 기반 시선추적)에 이미 적용된 흔적이 있어, **재사용 1순위 후보**(추정).
2. **희소성 재사용**: 이벤트 카메라 기반 XR 시선추적은 본질적으로 희소. 토큰(T_K) + 마스크 + line-buffer-valid 메커니즘(`linebuffer.h`)은 우리 파이프의 전처리/입력 스테이지에 그대로 차용 가능. ViT에도 token sparsity(중요 패치만 처리) 형태로 응용 여지.
3. **DSP 2-MAC 패킹**(`conv.h:299-312`): HG-PIPE 계열 ViT MAC 어레이의 INT8 처리량을 같은 트릭으로 배가 가능. DSP48 (A+D)×B 사전가산기 활용은 도메인 불문.
4. **DSE→코드생성 자동화 패턴**(`gen_code.py`): "ILP 결정변수(parallelism) → HLS 템플릿 파라미터" 직결 구조는 우리 ViT 가속기의 head/embedding 병렬도 자동 튜닝에 차용 가능. 단, 목적/자원 모델식은 우리가 별도 구축해야 함(ESDA 본체 부재).
5. **layer-pipelined dataflow + per-stage 재양자화**(`conv_pack.h` + `quantize`): Transformer 블록(MHA→FFN)을 동일하게 stage별 dataflow + 중간 INT 재양자화로 구성하는 설계 참조.
6. **주의**: 8비트 좌표·MobileNet 전제·ZCU102 하드코딩은 우리 보드/모델로 일반화 시 재작성 필요.

---
*근거 파일(절대경로)*: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\CNN-Accel\ESDA\optimization\solver\scip_solver.py`, `.../optimization/eventnet.mk`, `.../hardware/gen_prj.py`, `.../hardware/common.py`, `.../hardware/baseline_extract.py`, `.../hardware/benchmark_extract.py`, `.../hardware/README.md`, `.../hardware/template_e2e/{conv.h,conv_pack.h,linebuffer.h,mem.h,type.h,top.h,top.cpp.tpl,gen_code.py}`, `.../hardware/HWConfig/{zcu102.json,zcu102_80res.json}`. (pipeline.py는 ESDA 본체 부재로 동형 `SEE/ESDA/optimization/pipeline.py` 인용.)
