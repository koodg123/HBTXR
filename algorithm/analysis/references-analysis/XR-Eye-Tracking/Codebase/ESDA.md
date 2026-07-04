# ESDA 코드베이스 정밀 분석

분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\ESDA`
분석 도구: Glob / Grep / Read (실제 소스 라인 근거 기반). 본 문서의 모든 "확인" 항목은 실제 코드 라인을 직접 읽고 작성했으며, 추정/미확인은 명시했다.

---

## 1. 개요

### 목적 / 원논문
- **ESDA = Efficient Sparse Dataflow Accelerator** (추정 — repo 내 약어 풀네임 명시는 없으나 디렉토리/구조가 sparse dataflow 가속기와 일치). 디렉토리 명 `eventNet/`, cfg `name: MobileNetV2`로 보아 "EventNet"이 모델/시스템 이름으로 쓰임 (확인: `eventNet/hw/.../cfg.json:2`).
- **원논문: FPGA'24 논문**. `evaluation.md:3` "Reproduce Results in **FPGA'24 Paper**", `evaluation.md:6` "Table 1", `evaluation.md:191` "Figure 13" 명시. (정확한 제목은 repo 내 미기재 → 확인 불가. ACM/SIGDA FPGA 2024 학회로 추정.)
- 목적: **이벤트 카메라 데이터를 sparse하게 표현한 뒤, 양자화된 Sparse MobileNetV2를 FPGA에서 dataflow(layer-pipeline) 방식으로 저지연 추론**하는 SW/HW 통합 reproducibility 패키지.

### 입력 / 출력
- **입력 = 이벤트(event) 히스토그램 프레임**. `software/README.md:50` "we mainly use **histogram** for in a fixed time interval", 일부 데이터셋은 denoise로 sparsity 증가. HW 입력은 **sparse 표현**: 비영점(non-zero) 활성만 묶은 feature 배열(`act_in`) + 비트마스크(`mask`) + 비영점 개수(`num_nz`) (확인: `top.cpp:127-128` `top(... act_in, act_out, mask, num_nz)`).
- **출력 = 분류 logit**. 최종단이 `global_avgpool_linear`로 클래스 수만큼 `ap_int<32>` logit을 출력 (확인: `top.cpp:122`, `conv.h:578-647`). 즉 본 repo가 다루는 태스크는 **이벤트 기반 분류**(제스처/문자/물체)이며, **동공/시선 좌표 회귀는 이 repo에는 없음**. (XR-Eye-Tracking 상위 폴더의 참조 코드베이스로서 포함된 것으로 추정 — 시선추적 헤드 자체는 미구현, 확인 불가.)
- 데이터셋(분류): ASL-DVS(수어 알파벳), DVSGesture, N-MNIST, RoShamBo17(가위바위보), N-Caltech101 (확인: `evaluation.md:20-27`, `software/README.md:5-30`).

### HW 타깃
- **Xilinx ZCU102 (Zynq UltraScale+ MPSoC)** + PYNQ overlay (확인: `hardware/README.md:5`). 설계 폴더명에 `zcu102_80res / 60res / 50res` 등 리소스 예산 비율이 인코딩됨.
- 합성 흐름: **Vitis HLS → Vivado → bitstream(.bit/.hwh) → 보드 평가** (확인: `hardware/README.md:52-58`, `prj/hls.tcl`, `prj/vivado.tcl`).
- 리소스 예산: `zcu102_80res.json` = `dsp: 2016, bram36: 730` (확인). ZCU102 전체 자원의 약 80% 의미로 추정.

---

## 2. 디렉토리 구조

```
ESDA/
├── evaluation.md            # FPGA'24 결과 재현 메인 가이드(SW정확도/HW지연/전력/Fig13)
├── software/                # [자체] 양자화 학습·추론·int 익스포트
│   ├── main.py                  # 학습/평가 엔트리
│   ├── int_inference.py         # int8 모델 → HW용 cfg/weight/입력 익스포트
│   ├── data_preprocess.py       # 이벤트 → 히스토그램 프레임 전처리
│   ├── search_sw.py             # 랜덤 아키텍처 cfg 생성기(NAS류)
│   ├── models/
│   │   ├── HAWQ_mobilenetv2.py      # [핵심] 양자화 MobileNetV2(inverted residual)
│   │   ├── HAWQ_quant_module/
│   │   │   ├── quant_modules.py     # [핵심] QuantBnConv2d/QuantAct/QuantLinear
│   │   │   └── quant_utils.py       # 대칭 양자화 파라미터·STE
│   │   ├── mink_mobilenetv2.py / mink_resnet.py / mobilenet_base.py / mobilenet_settings.py
│   │   └── sparsity.py
│   ├── MinkowskiEngine/         # [외부] sparse tensor/conv 프레임워크(번들 포함)
│   ├── src/ , pybind/           # [외부] MinkowskiEngine C++/pybind 백엔드
│   ├── dataset/                 # 데이터셋 로더·이벤트 전처리(DVS/AEDAT 등)
│   └── config/                  # YAML 학습 설정
├── hardware/                # [자체] HLS 템플릿·프로젝트 생성·보드 평가
│   ├── gen_prj.py               # cfg → HLS 프로젝트 생성
│   ├── baseline_extract.py      # 결과 추출(Table 1)
│   ├── template_e2e/            # [핵심] HLS 소스 템플릿(타 데이터셋 공통)
│   ├── template_e2e_roshambo/   # roshambo 전용 템플릿
│   ├── HWConfig/                # [자체] FPGA 자원 예산 json (zcu102_*res.json)
│   ├── board/                   # 보드 측 평가 스크립트(evaluate.py/hw_e2e.py)
│   └── benchmark_results/       # [생성/대용량] sparse vs dense 블록별 co-sim(Fig13)
├── optimization/           # [자체] 하드웨어 설계공간탐색(DSE)
│   ├── eventnet.py              # [핵심] DSE 엔트리(ILP/MIQP 솔버)
│   └── formulation/, utils/     # 자원·지연 모델·SCIP/gpkit 정식화 (추정: 본문 미열람)
└── eventNet/               # [생성 산출물] 실제 모델/DSE결과/HW설계
    ├── model/                   # int 익스포트(model.json, input/output .npy)
    ├── DSE/                     # DSE 결과(en-config/result.json 등)
    └── hw/<dataset>_shift*-zcu102_*res/full/  # [생성] 데이터셋별 완성 HLS설계+bitstream
        ├── top.cpp/top.h            # dataflow 최상위
        ├── conv.h/conv_pack.h       # 연산 커널 + 블록 조립
        ├── linebuffer.h/mem.h       # sparse 라인버퍼 + DRAM 입출력
        ├── type.h/para.h/weight.h   # 자료형/비트폭·차원 매크로/가중치 ROM
        ├── gen_code.py/top.cpp.tpl  # [핵심] HLS 코드 생성기 + 템플릿
        └── prj/*.tcl, hw/top.bit    # 합성 스크립트 + 비트스트림
```

**자체 핵심 코드**: `software/models/HAWQ_*`, `hardware/template_e2e`(+생성된 `eventNet/hw/*/full/*.h/.cpp`), `optimization/`.
**제외(외부/대용량)**: `software/MinkowskiEngine/`·`software/src/`·`software/pybind/`(MinkowskiEngine 번들 원본), `eventNet/hw/*/full/hw/top.bit`·`*.hwh`·`*.npy`(비트스트림/대용량), `software/dataset/preprocess/dvs/PyAedatTools`(AEDAT 파서 서드파티), `hardware/benchmark_results/`의 블록별 생성물(Fig13용 대량 생성 코드).

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

### 3-A. 소프트웨어 (양자화 Sparse CNN)

#### (1) 모델: 양자화 Sparse MobileNetV2
- `software/models/HAWQ_mobilenetv2.py:12 Q_LinearBottleneck` — MobileNetV2 **inverted residual(linear bottleneck)** 블록을 양자화로 구현.
  - 구조: `conv1(1x1 expand) → ReLU6 → conv2(3x3 depthwise) → ReLU6 → conv3(1x1 project)` + 옵션 residual add (확인: `HAWQ_mobilenetv2.py:49-60, 75-94`).
  - 각 conv는 `QuantBnConv2d(per_channel=True, bias_bit=...)`, 각 활성 뒤에 `QuantAct(shift_bit=...)`. residual은 `quant_act_int32`로 정수 도메인에서 더한 뒤 재양자화 (확인: `:93-100`).
  - 활성함수는 `ME.MinkowskiReLU6` — **MinkowskiEngine sparse tensor** 위에서 동작(=sparse CNN). 즉 SW도 좌표(C)+특징(F) 형태의 희소 표현으로 학습/추론.
  - `int_folder`가 주어지면 레이어별 정수 입력/가중치/출력/스케일을 `.npy`로 덤프 → **HW 검증용 골든 데이터** 생성 (확인: 곳곳의 `np.save(...)`, 예 `:111-116`).

#### (2) HAWQ 양자화 모듈 (`HAWQ_quant_module/quant_modules.py`)
- `QuantBnConv2d` (`:327`) — **BN-fold된 conv를 대칭 양자화**.
  - 기본 `weight_bit=8, bias_bit=32`(클래스 기본; 실행 시 `--bias_bit`로 16/32 지정), per-channel 대칭 양자화 (확인: `:354-355,:369`, `:441-442 symmetric_linear_quantization_params(...)`).
  - BN 통계로 가중치/바이어스를 스케일링 후 정수화하고 `convbn_scaling_factor`·`bias_integer` 산출 (확인: `:502-510`).
- `QuantAct` (`:148`) — 활성 양자화. `act_range_momentum=0.95` 러닝 통계로 범위 추적, `shift_bit`(=HW EXP)로 재양자화 시프트 결정. residual/identity 경로 인자 보유 (확인: `:222 forward(... identity ...)`).
- `QuantLinear` (`:12`) — FC(분류기) 대칭 양자화. `weight_integer = SymmetricQuantFunction.apply(...)`, 정수 GEMM 후 스케일 복원 (확인: `:119-145`).
- 결론: **HAWQ 프레임워크(대칭 per-channel, BN-fold, integer-only inference)** 기반 QAT. 다만 **배포(HW) 비트폭은 가중치=활성=INT8 균일**이며(아래 HW 절), HAWQ의 mixed-precision(레이어별 비트 다름)은 **이 배포 경로에는 적용되지 않음**(확인: `para.h:5-6` `CFG_AW 8 / CFG_WW 8`). bias/scale/EXP만 16/32로 데이터셋별 가변(NCal=32, 그 외 16) (확인: `gen_code.py:312-327`, `evaluation.md:20-27`).

#### (3) 학습/익스포트 파이프라인
- `main.py` — float32 학습 → int8 QAT(`--shift_bit`, `--fixBN_ratio`) → 평가(`-e`) (확인: `software/README.md:96-127`).
- `int_inference.py` — 학습된 int8 모델에서 `--int_dir`로 **HW용 model.json + weight + 입력/출력 .npy** 생성 (확인: `software/README.md:130-141`).
- `search_sw.py` — 채널/확장비/다운샘플을 랜덤 샘플하는 **아키텍처 cfg 생성기**(NAS류). Hessian 민감도 기반 mixed-precision 탐색이 아님(확인: `search_sw.py:1-19`; "sensitivity/hessian" 검색 0 매치).

### 3-B. 하드웨어 (HLS dataflow 가속기) — 핵심

#### (1) 최상위 dataflow 구조 (`top.cpp`)
- `top()`은 AXI 인터페이스만 정의: `m_axi`로 `act_in/act_out/mask` 3개 메모리뱅크, `s_axilite control` (확인: `top.cpp:127-142`). 즉 **PYNQ 호스트가 DRAM에 sparse 입력·마스크를 올리고 결과를 회수**하는 표준 MPSoC 흐름.
- 실제 네트워크는 `wrapper()` 내부 단일 **`#pragma HLS DATAFLOW`** 영역 (확인: `top.cpp:6`).
- **데이터플로우 토폴로지**: 모든 레이어가 `hls::stream`(활성 `a_*` + 토큰 `t_*` 쌍)으로 연결되어 **레이어-파이프라인(공간적 데이터플로우)** 을 형성. 레이어마다 전용 PE가 있고 단일 재사용 systolic array가 아님 (확인: `top.cpp:8-95` 스트림 선언, `:103-122` 함수 체인).
  - 로드: `read_sparse_input`(비영점 특징) + `M2S_mask`(마스크→토큰) + `mask_stride2`(stride2용 토큰 사전생성) (확인: `top.cpp:98-100`).
  - 연산: `conv_3x3_first_layer` → `conv_1x1_3x3_dw_1x1_stride1/stride2/_residual` ×17블록 → `conv8`(1x1) → `global_avgpool_linear` (확인: `top.cpp:103-122`). 이는 cfg.json의 MobileNetV2 7-stage(블록0~16) 구성과 일치 (확인: `cfg.json:9-1003`).
- **스트림 폭이 레이어별로 다름**: `BundleT<CFG_BLOCK_k_POC, ap_int<CFG_AW>>` — 각 레이어 병렬도 `POC`만큼 채널을 한 묶음으로 흘림 (확인: `top.cpp:20-87`).

#### (2) 자료형 / 토큰 (`type.h`)
- `T_K { ap_uint<1> end; ap_uint<8> x; ap_uint<8> y; }` — **희소 좌표 토큰**(end=종료 플래그) (확인: `type.h:2-6`).
- `BundleT<N,T> { T data[N]; }` — N개 채널 묶음(병렬 처리 단위) (확인: `type.h:8-11`).
- `T_OFFSET = ap_uint<4>`, `end_3x3 = 15` — 3x3 윈도우 내 9개 위치 인덱스 + 종료 마커 (확인: `type.h:15-16`).
- 비트폭(`para.h`): `CFG_AW 8`(활성), `CFG_WW 8`(가중치), `CFG_PW 32`(psum), `CFG_SW 16`(scale), `CFG_BW 16`(bias), `CFG_EXP 16`(재양자화 시프트), `CFG_MW 128`(마스크 워드), `CFG_TW 8`(토큰 좌표) (확인: `para.h:5-12`).

#### (3) Sparse 라인버퍼 (`linebuffer.h`) — 본 설계의 차별점
- `conv_3x3_line_buffer_stride1_serial` (`:3`) — **3행 순환 라인버퍼 + valid 비트맵**으로 희소 입력을 받아 3x3 윈도우를 재구성.
  - `line_buff[BUFFER_ROWS=3][WIDTH*IC/PI]`, `valid[3][WIDTH]` (확인: `:39,:44`).
  - 토큰 FIFO로 "가장 최신/가장 오래된" 좌표를 추적하며, y_delta≥1 또는 한 줄 내 x 진행 시 출력 유효 판정 → 윈도우 9개 위치를 순회하되 **valid(=실제 비영점)인 위치만** offset과 함께 conv로 전달 (확인: `:104-204`, 특히 `:183-203` `if(valid_point){ offset_s.write(offset); ... act_out.write(...) }`).
  - 즉 **연산을 비영점 위치에만 수행**(zero-skipping) — sparse 가속의 핵심. 패딩 영역은 not_padding 검사로 제외 (확인: `:173-182`).
- `conv_3x3_line_buffer_stride2_fifo_serial` (`:209`) — stride2용. **even/odd y에 대해 분리된 토큰 FIFO**를 두어 출력 좌표를 정렬·머지(key 비교로 pop_even/pop_odd 결정) (확인: `:247-393`). stride2 다운샘플의 좌표 정합을 희소 도메인에서 처리하는 부분이 가장 복잡.
- `conv_3x3_line_buffer_first_layer` (`:542`) — 입력 채널이 작은 첫 레이어 전용(밀집 윈도우 패킹, `#pragma HLS PIPELINE` 전체 루프) (확인: `:641-643`).

#### (4) 연산 커널 (`conv.h`)
- **`DSP_AM` (`:1-8`) — DSP 패킹 트릭**: 한 DSP48에 `(in1+in2)*in3` 형태로 **2개의 MAC을 동시 수행**. 1x1 커널에서 `w_1`을 18비트 상위로 시프트(`w_1_shift.range(26,18)`)해 한 곱셈 결과 상·하위 비트로 2채널 psum을 분리 (확인: `conv.h:298-320`, `:531-555`). → **DSP 효율 2배**.
- `conv_1x1_kernel_dsp` (`:226`) — pointwise(1x1). 활성 버퍼에 채널 적재 후 `OC/PO × IC/PI` 루프로 GEMM, `PO/2`개를 DSP 패킹으로 처리, `#pragma HLS PIPELINE` (확인: `:283-324`).
- `conv_3x3_dw_kernel`/`conv_3x3_dw_kernel_serial` (`:10,:334`) — depthwise(3x3). `offset_s` 스트림으로 라인버퍼가 보낸 유효 위치만 누적, `psum_buffer[IC]` 채널별 누산 (확인: `:367-396`).
- `conv_3x3_kernel_dsp_first_layer` (`:462`) — 첫 3x3 conv(입력채널=PI), 가중치를 LUTRAM ROM에 저장(`impl=LUTRAM`) + DSP 패킹 (확인: `:473-475,:546-555`).
- **`quantize` (`:62)** — psum→int8 재양자화: `psum_mul = (psum+bias)*scale + round_shift; >> EXP; clip[low,high]; ReLU`. scale/bias는 `scale_buffer[IC]`(SCALEW+BIASW 묶음)에서 추출 (확인: `:99-121`). 정수-only inference로 SW HAWQ와 정합.
- **`quantize_id_add` (`:129)** — residual용. psum 경로와 identity 경로를 각각 스케일링·시프트 후 정수 도메인에서 더하고 clip (확인: `:188-219`). SW `quant_act_int32`(`HAWQ_mobilenetv2.py:98`)와 1:1 대응.
- `global_avgpool_linear` (`:578`) — GAP(채널별 합) 후 `N_CLASS×IC` 가중치로 logit 산출, `ap_int<32>` 출력 (확인: `:611-647`).
- **주의(코드 품질)**: `quantize`/`quantize_id_add` 내부에 `cout<<...`(`:108,:122` 등) 디버그 출력이 **합성 경로에 남아 있음**. HLS에서는 무시되나 csim에서 대량 출력 → 정리 권장.

#### (5) 블록 조립 (`conv_pack.h`)
- `conv_1x1_module`(`:1`) = 1x1 conv + quantize, `conv_1x1_module_residual`(`:24`) = 1x1 conv + quantize_id_add.
- `conv_3x3_dw_module_stride1/2_serial`(`:49,:87`) = 라인버퍼 + dw커널 + quantize를 한 `#pragma HLS DATAFLOW`로 묶음.
- **`conv_1x1_3x3_dw_1x1_stride1/stride2/_residual`** (`:123,:167,:215`) = MobileNetV2 inverted-residual 블록 전체(1x1확장→3x3dw→1x1축소[+residual])를 하나의 dataflow 서브그래프로 구성 (확인: `:148-164` 3-stage 체인).
- residual 경로는 `duplicate_stream`(`mem.h:431`)으로 입력을 복제해 identity FIFO에 보관(`act_id depth=(WIDTH+2)*OC/PF_2`)했다가 더함 (확인: `conv_pack.h:243-248`).
- 각 stage 병렬도 `PF_0/PF_1/PF_2`(=PIC/PC/POC)가 **레이어마다 독립 템플릿 인자** → 파이프라인 균형용 가변 병렬도 (확인: `conv_pack.h:123-126` 템플릿 시그니처).

#### (6) 메모리 / 입출력 (`mem.h`)
- `read_sparse_input` (`:231`) — DRAM의 비영점 특징을 `num_nz*IC/PI`개 읽어 스트림화 (확인: `:238-246`). **밀집 H×W가 아니라 비영점만 전송** → 대역폭 절감.
- `M2S_mask` (`:123`) — 비트마스크를 좌표 토큰 스트림으로 변환(`token.x/y`, 마지막 end 토큰) (확인: `:150-172`).
- `mask_stride2` (`:175`) — 2×2 OR로 stride2 출력 좌표 토큰을 사전 생성 (확인: `:206-222`).
- 가중치는 `weight.h`에 `#include "data/*.txt"`로 ROM 상수 임베드, HLS `bind_storage ... rom_2p impl=BRAM/LUTRAM`으로 온칩 저장 (확인: `gen_code.py:38-45,:141-146`; `conv.h:19,:72,:235,:473`).

#### (7) HWConfig 코드 생성기 (`gen_code.py` + `top.cpp.tpl`)
- cfg.json(레이어 리스트)을 읽어 **top.cpp / para.h / weight.h를 자동 생성** (확인: `gen_code.py:306-376`).
- `top.cpp.tpl`의 `/*gen_code-fifo|load|comp|store*/` 마커 위치에 생성 코드를 삽입 (확인: `:363-370`).
- 레이어 type별 함수·템플릿 인자·가중치 ROM 차원을 자동 산출. 예: block은 wbuf0/1/2 + sbuf0/1/2(+ibuf residual), conv1/conv8/linear은 전용 분기 (확인: `:124-276`).
- psum 비트폭을 `CFG_AW+CFG_WW+ceil(log2(IC))`로 레이어별 자동 산정(오버플로 방지) (확인: `:128-131,:185-187`).
- 데이터셋별 비트폭 분기: NCAL→SW/BW/EXP=32, Roshambo→MW=64 (확인: `:312-316`).

### 3-C. 최적화 (DSE) — `optimization/eventnet.py`
- **목적: 모델 구조 + FPGA 자원 예산을 입력받아 레이어별 병렬도(PIC/PC/POC)를 최적화**하여 지연 최소화 (확인: `optimization/README.md:3-9`).
- `EventNetFormulation`으로 정식화 후 `en.solve()` — **SCIP 솔버(ILP/MIQP) + gpkit** 사용. cfg에 `scip-use_cmd`, `scip-timelimits=900`, gpkit 모델 산출(`en-gpkit.model/.sol`) (확인: `eventnet.py:7-8,:160-189,:244`; `hardware/README.md:31-37`).
- 제약: 총 DSP ≤ `hw["dsp"]`, 총 BRAM ≤ `hw["bram36"]*2` (확인: `eventnet.py:154-157`). 자원 모델은 `dsp_vfunc_dict`/`buf_vfunc_dict`로 레이어 type별 함수 정의 (확인: `:90-108`).
- 목적함수 `form-obj`: `lat_max`(=레이어 최대지연 최소화→파이프라인 균형) 및 `lat_max+uti`(자원 활용 가중) (확인: `:160-189`).
- 결과는 `eventNet/DSE/<model>-<hw>/en-result.json` → 이후 `hardware/gen_prj.py`로 HW 프로젝트화. **즉 본 repo의 "mixed-precision/탐색"은 비트폭이 아니라 HW 병렬도 매핑 탐색**.

---

## 4. 알고리즘 / 데이터 표현

### Sparse 이벤트 표현
- 전처리: 이벤트 → **고정 시간창 히스토그램 2D 프레임**(+선택적 denoise) (확인: `software/README.md:48-62`).
- SW: MinkowskiEngine **SparseTensor(좌표 C + 특징 F)** 로 학습/추론(밀집 텐서 아님).
- HW: 입력을 **(비영점 특징 배열 `act_in`) + (비트마스크 `mask`) + (개수 `num_nz`)** 3요소로 표현 (확인: `top.cpp:127`, `mem.h:231,:123`). 입력 sparsity는 cfg에 기록(ASL `input_sparsity: 0.0114`, 확인 `cfg.json:8`). 레이어가 깊어질수록 sparsity 상승(`block_16 sparsity 0.4953`, 확인 `cfg.json:892`).
- HW 데이터플로우는 토큰(좌표) 스트림과 특징 스트림을 **쌍으로** 흘려 **비영점 위치에서만 MAC** 수행(zero-skipping) → 이벤트 카메라 특유의 고희소성을 직접 활용.

### Mixed-precision 양자화 (정밀 구분)
- **학습(QAT)**: HAWQ — 대칭 per-channel 가중치 양자화, BN-fold, integer-only inference (확인: `quant_modules.py:327,:441,:502`).
- **배포(FPGA)**: 가중치/활성 **균일 INT8**(CFG_AW=CFG_WW=8) (확인: `para.h:5-6`). scale=16/32b, bias=16/32b, 재양자화 시프트 EXP=16/32(데이터셋별) (확인: `para.h:8-12`, `gen_code.py:312-327`). → **레이어별 비트가 다른 진정한 mixed-precision은 배포 경로에 없음**(추정상 HAWQ의 일부 기능만 활용). psum만 32b 누산.

### Dataflow 매핑
- **레이어-파이프라인(layer-as-PE) 공간적 데이터플로우**: 모든 레이어가 동시에 살아있고 스트림으로 연결(`#pragma HLS DATAFLOW`, `top.cpp:6`). 단일 재사용 PE 어레이(systolic)가 **아님**.
- 레이어별 병렬도(PIC/PC/POC)를 DSE로 정해 **각 레이어 지연을 균형화**(throughput 매칭) → 파이프라인 stall 최소화.
- DSP 패킹(`DSP_AM`)으로 DSP당 2 MAC, 가중치는 온칩 ROM(BRAM/LUTRAM) 임베드 → 외부 가중치 트래픽 0.

---

## 5. 학습 · 평가

### 데이터셋 / 메트릭
- 분류 정확도 Top-1/Top-5 (확인: `evaluation.md:34-41`, N-MNIST 예시 Top-1 99.0).
- 8개 설계(데이터셋×폭): ASL_0p5, ASL_2929, DVS_1890, DVS_w0p5, NMNIST, Roshambo, NCal_2751, NCal_w0p5 (확인: `evaluation.md:64-102`).

### 리소스 / 지연 보고
- **HLS 합성 리포트(.rpt)는 repo에 미커밋**(확인: `**/*.rpt` 0건). LUT/FF의 실측 합성치는 직접 확인 불가.
- 대신 **DSE 추정 자원이 cfg.json에 레이어별로 기록**: 예(ASL_0p5) 레이어별 dsp/bram 합과 `total_dsp: 1998, total_bram: 1268, lat_max: 7626, obj: 7786.9` (확인: `cfg.json:60-64`). 예산은 `zcu102_80res: dsp 2016, bram36 730`(=BRAM18 730×2 기준 1460) (확인: `zcu102_80res.json`).
- 보드 실측: `make evaluate_hw ... --enable_pm`으로 **지연+전력** 측정, `baseline_extract.py`로 Table 1 재구성 (확인: `evaluation.md:62-113`). 전력은 `power_record.npy`로 저장(대용량, 미열람).
- **Fig13**: `hardware/benchmark_results/{sparse,dense}/...`에 블록별 co-sim → sparse vs dense 비교(확인: `evaluation.md:191-208`, 디렉토리 `single_blk-block_0..16`).

### 주요 명령어 (확인: README/evaluation)
- SW 평가: `python main.py --bias_bit 16 --settings_file=weights/<ds>/settings.yaml --load ... --shift_bit 16 -e`
- int 익스포트: `python int_inference.py --bias_bit 16 --settings_file=... --shift_bit 16 --int_dir ../eventNet/model/<ds>`
- DSE: `python eventnet.py --model_path ../eventNet/model --hw_path ../hardware/HWConfig --model_name <ds> --hw_name zcu102_80res --results_path ../eventNet/DSE` (README 인자명은 `--eventNet_path`로 표기되나 실제 코드 인자는 `--hw_path`/`--hw_name` — README와 코드 불일치, 확인: `optimization/README.md:9` vs `eventnet.py:200-206`)
- HW 생성: `python gen_prj.py gen_full --cfg_name <ds> --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW`
- 합성/평가: `make gen → make ip_all → make hw_all → make evaluate_hw → make e2e_inference`

---

## 6. 의존성
- SW: PyTorch, **MinkowskiEngine**(sparse conv, repo에 번들), numpy, tonic(데이터셋), conda env `ESDA` (확인: `evaluation.md:12-14`, `software/setup.py`).
- 양자화: HAWQ 계열 자체 모듈(`HAWQ_quant_module`), STE/대칭양자화 유틸 (확인: `quant_utils.py`).
- HW: **Vitis HLS / Vivado**(Xilinx), `ap_int`/`hls::stream`, **Boost**(`boost::static_log2`, `static_lcm` — 확인 `conv.h:15`, `mem.h:436`), C++ 템플릿 메타프로그래밍.
- DSE: **SCIP** 솔버, **gpkit**(geometric programming) (확인: `eventnet.py:7-8`, `hardware/README.md`).
- 보드: **PYNQ** overlay, ZCU102 (확인: `hardware/README.md:5`).

---

## 7. 강점 · 한계

### 강점
- **이벤트 희소성을 HW까지 일관되게 활용**: SW(MinkowskiEngine)→sparse 입력 표현→토큰 기반 라인버퍼 zero-skipping까지 end-to-end (확인: `linebuffer.h:183-203`).
- **DSP 패킹(2 MAC/DSP)** 으로 INT8 연산 밀도 2배 (확인: `conv.h:298-320`).
- **레이어-파이프라인 + DSE 자동 병렬도 매핑**으로 파이프라인 균형(throughput 매칭) (확인: `eventnet.py`, `cfg.json` per-layer parallelism).
- **완전 자동 HW 생성**: cfg.json→HLS 소스(top/para/weight) 코드 생성기 (확인: `gen_code.py`).
- **재현성 완비**: 8개 설계의 bitstream·골든 .npy·평가 스크립트 포함.

### 한계
- **분류 전용** — 동공/시선 좌표 회귀 헤드 없음(태스크 미스매치 가능, 확인: 최종단이 `global_avgpool_linear` 분류기).
- 배포가 **균일 INT8** — HAWQ mixed-precision의 정확도/효율 이점을 HW에서 활용하지 않음(추정).
- **레이어-파이프라인 = 모델 전용 비트스트림**: 모델/입력해상도 바뀌면 재합성 필요(유연성↓), 온칩 가중치 ROM이라 큰 모델은 BRAM 한계.
- **HLS 합성 리포트(.rpt) 미커밋** → LUT/FF/주파수 실측 직접 확인 불가(추정만 가능).
- **코드 위생**: 합성 경로에 `cout` 디버그 다수(`conv.h:108,:122` 등), README↔코드 인자명 불일치(`--eventNet_path` vs `--hw_path`), 주석처리 코드 블록 다량.

---

## 8. 우리 프로젝트 시사점 — "XR 시선추적 + FPGA 저지연 on-device 가속"

**직접 참조 가치 = 매우 높음.** 본 repo는 우리가 목표하는 "이벤트카메라 → FPGA on-device 저지연" 파이프라인의 거의 모든 빌딩블록을 제공한다.

1. **Sparse dataflow 라인버퍼 재사용**: `linebuffer.h`의 토큰(좌표)+특징 쌍 스트림 + valid 비트맵 zero-skipping은 이벤트 동공/시선추적에 그대로 이식 가능. 이벤트 입력의 극희소성(0.01~0.5)을 HW에서 직접 활용하는 검증된 레퍼런스.
2. **INT8 + DSP 패킹 MAC**: `DSP_AM`(2 MAC/DSP)은 우리 가속기 DSP 효율 2배 확보용으로 즉시 차용 가능.
3. **HLS 모듈 구조**: inverted-residual 블록을 `1x1+3x3dw+1x1(+residual)` dataflow로 조립한 `conv_pack.h` 패턴은 경량 backbone HW화의 모범. quantize/quantize_id_add(정수 residual)는 SW HAWQ와 1:1 정합되어 SW/HW 검증 루프가 안정적.
4. **DSE(병렬도 매핑) 프레임워크**: `optimization/eventnet.py`의 자원제약 ILP로 레이어별 PIC/PC/POC를 자동 결정 → 우리 보드(자원 예산만 교체)에 재사용 가능. `algo2fpga` DSE와 결합 시 강력.
5. **자동 HLS 코드 생성기**: `gen_code.py`+cfg.json 흐름은 모델 변형마다 수작업 HLS를 피하게 해줌 → 시선추적 모델로 바꿔도 cfg만 교체.

**보완 필요(우리가 추가할 것)**: (a) 분류 헤드 → **동공/시선 좌표 회귀 헤드**로 교체(GAP+linear 대신 좌표 regression), (b) **균일 INT8 → 정확도 민감 레이어 mixed-precision** 적용 여부 검토(HAWQ 활용), (c) HLS 합성 리포트 확보로 LUT/FF/주파수 실측, (d) 시계열(이벤트 스트림) 연속추론을 위한 상태유지/스트리밍 인터페이스 설계.

---

## 9. 근거 표기 정리
- **확인(코드 직접 열람)**: top.cpp/conv.h/conv_pack.h/linebuffer.h/mem.h/type.h/para.h/cfg.json/gen_code.py(전체), quant_modules.py(부분: QuantLinear/QuantBnConv2d/QuantAct 시그니처·forward), HAWQ_mobilenetv2.py(Q_LinearBottleneck), eventnet.py(전체), zcu102_80res.json, evaluation.md/각 README(전체).
- **추정**: ESDA 약어 풀네임, FPGA'24 논문 정확한 제목, MinkowskiEngine/PyAedatTools/formulation 세부(외부·미열람), 본 repo가 XR 시선추적 상위 폴더에 포함된 정확한 사유.
- **확인 불가**: HLS 합성 LUT/FF/주파수 실측(.rpt 미커밋), 전력 수치(.npy 미열람), DSE formulation 내부 자원·지연 함수 정확식(`formulation/` 미열람).
