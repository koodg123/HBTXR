# dac_sdc_2021_designs-main 정밀 분석

> 대상: `REF/CNN-Accel/dac_sdc_2021_designs-main/`
> 분석 범위: 서브팀 3개(SJTU_microe / SkrSkr / iSmart)의 자체 핵심 HLS 커널 + 양자화 파이썬
> 근거 표기: `파일:라인`. 추론은 "추정", 확인 불가는 "확인 불가"로 명시.
> 직전 분석의 "소스 미동봉" 판정은 재귀 Glob 타임아웃으로 인한 오판이며, 본 분석은 개별 절대경로 Read로 실재 소스를 직접 확인함.

---

## 1. 개요

본 디렉토리는 **DAC System Design Contest 2021**(저전력 임베디드 객체검출 트랙)에 출전한 여러 서브팀의 FPGA 가속기 설계 모음이다. 한 모델/한 가속기가 아니라 **서로 다른 팀의 HLS 가속기를 나란히 비교할 수 있게 모아 놓은 컬렉션**이라는 점이 특징이다.

- **공통 태스크/모델 백본**: 세 팀 모두 입력 해상도 `640x360` 원본을 받아 on-chip에서 `320x160`(SJTU·iSmart) 또는 동일 계열로 리사이즈한 뒤(`SJTU_microe/hls/top.cpp:24-28`, `SkrSkr/skynet_flow.h:22-31`), 8단계 안팎의 conv 스택 + bypass(reorg/concat) + 검출 헤드로 구성된 경량 검출망을 돌린다.
  - SJTU·iSmart: **UltraNet 계열**(3x3 conv 8개 + 1x1 conv 헤드, reorg bypass). `SJTU_microe/hls/top.cpp:41`의 top 함수명이 `UltraNet_Bypass`로 직접 확인됨. iSmart의 `config.h`는 SJTU와 거의 동일한 레이어 형상(conv_0~conv_8)을 가짐(`iSmart/src/config.h:1-229`).
  - SkrSkr: **SkyNet 계열**(depthwise + pointwise = MobileNet형 bundle 7개 + bypass reorg). `SkrSkr/skynet_flow.cpp:11`의 top 함수명이 `skynet_flow`로 직접 확인됨.
- **타깃 보드**: 코드 자체에 보드명 매크로는 없으나, DAC-SDC 2021의 FPGA 트랙 표준 보드인 **Xilinx Ultra96(ZU3EG)** 또는 **ZCU104(ZU7EV)** 계열로 추정. 세 팀 모두 AXI-Stream 인터페이스(`my_ap_axis`, `ap_axiu`)와 single-DDR 스트리밍 구조를 쓰는 점이 Ultra96 PYNQ 흐름과 일치(추정). 정확한 보드 핀/클럭은 본 디렉토리에 미동봉이라 **확인 불가**.
- **데이터 정밀도**:
  - SJTU: weight 4bit / activation 4bit(4w4a), 첫 레이어 입력만 8bit(`SJTU_microe/hls/config.h:13-15`, `:34`).
  - SkrSkr: activation 8bit, weight 5bit, conv 누산 16bit(`SkrSkr/skynet_flow.h:13-17`).
  - iSmart: 첫·마지막 레이어 weight 8bit, 중간 4bit, activation 4bit(`iSmart/src/config.h:13-15`, `:206`).

---

## 2. 디렉토리 구조 (서브팀별 + 제외 이유)

```
dac_sdc_2021_designs-main/
├── SJTU_microe/                 # UltraNet_Bypass (4w4a, 명시적 2D PE array)
│   ├── hls/
│   │   ├── PE_array.h           [핵심] 1D/2D PE array, 4MUL/DSP, 2MUL/DSP(L1)
│   │   ├── conv3x3.h            [핵심] 3x3 conv 데이터플로우 래퍼
│   │   ├── conv1x1.h            [핵심] 1x1 conv (검출 헤드)
│   │   ├── maxpool.h            MaxPool 2x2/s2
│   │   ├── reorg.h              ReOrg(space-to-depth) bypass
│   │   ├── shift_reg.h          [핵심] line-buffer/sliding-window (1in2out, 1in1out)
│   │   ├── function.h           BN+QUReLU 융합, padding, resize
│   │   ├── stream_tools.h       폭변환/DEMUX/MUX/PISO/concat 유틸
│   │   ├── top.cpp              [핵심] UltraNet_Bypass 전체 그래프 배선
│   │   └── config.h             레이어별 형상/병렬도/비트폭
│   ├── quantization/
│   │   ├── quant_ultra.py       [핵심] DoReFa형 양자화 모듈
│   │   └── qnn_mem_process.py   [핵심] 가중치→HLS 배열 패킹/배치
│   └── readme.md               (외부 repo 링크만)
│
├── SkrSkr/                      # SkyNet im2col 스트리밍 (DW/PW)
│   ├── skynet_flow.cpp          [핵심] 7-bundle 전체 그래프 배선
│   ├── skynet_flow.h            [핵심·일부만] 전역 config + 가중치 상수배열(620KB)
│   ├── im2col.h                 [핵심] 3x3 im2col 라인버퍼
│   ├── dwconv.h                 [핵심] depthwise conv3x3
│   ├── pwconv.h                 [핵심] pointwise conv + 2MUL/DSP MAC
│   ├── norm_actv.h              BN(scale+bias)+ReLU 융합
│   ├── maxPool.h                MaxPool 2x2
│   ├── bypass_unit.h            bypass reorg send/recv (BRAM FIFO)
│   ├── findMax.h                검출 헤드 후처리(confidence argmax)
│   └── stream_tools.h           reduceWidth/expandWidth/copy/comb/add_last
│
└── iSmart/                      # conv2d_DSPopt DSP 패킹
    ├── src/
    │   ├── conv2d.h             기준(naive) 3x3/1x1 conv + MVU 래퍼
    │   ├── conv2d_DSPopt.hpp    [핵심] DSP 패킹 conv3x3 (FIR식 4-tap/DSP)
    │   ├── conv1x1DSP2.hpp      [핵심] 1x1 conv 2MUL/DSP
    │   ├── bn_qrelu2d.h         BN+QUReLU 2D 래퍼
    │   └── config.h             레이어별 형상/SIMD/PE/DSP6 병렬도
    └── readme.md               (외부 repo 링크만)
```

**본 분석에서 제외한 항목과 이유**(작업 지침에 따름):
- `*.bit/.hwh/.bin/.npy/.pt/.so`: 비트스트림/하드웨어 핸드오프/학습 산출물 바이너리. 소스가 아님.
- 외부 yolov5/yolov3 학습 포크: 학습 코드는 본 가속기 핵심이 아니며 본 디렉토리에 미동봉(`readme.md`가 외부 GitHub 링크만 제시 — `SJTU_microe/readme.md:1`, `iSmart/readme.md:1`).
- 거대 `param.h` 가중치 생성물: `SJTU_microe/hls/top.cpp:16`이 `#include "param.h"`로 참조하나, 이는 `qnn_mem_process.py`가 자동 생성하는 weight 상수 헤더이므로 내용 분석 제외.
- `SkrSkr/skynet_flow.h`는 620KB로 대부분이 자동 생성 weight 상수배열(`L0_DW`, `L0_PW`, ... — `SkrSkr/skynet_flow.h:38-130` 등). 상단 160행의 **config/typedef 정의부만** 분석 대상으로 삼고, 가중치 hex 본문은 생성물로 간주해 제외.

---

## 3. 핵심 모듈 정밀 분석 (서브팀별)

세 팀 모두 "**스트림 기반 dataflow + SIMD(채널방향) × PE(출력채널방향) 병렬 + DSP 멀티펌핑(1 DSP에 여러 MUL 패킹)**"이라는 공통 골격을 쓰되, **PE array를 명시적으로 노출하는 방식(SJTU)**, **im2col로 depthwise/pointwise를 분리하는 방식(SkrSkr)**, **3x3을 FIR로 보고 DSP 한 개에서 4-tap을 뽑는 방식(iSmart)**으로 갈라진다.

### 3.1 SJTU_microe — 명시적 2D PE Array + 4MUL/DSP 패킹

#### (a) 1D PE — DSP 1개에 4개 MUL을 욱여넣는 핵심 트릭

`SJTU_microe/hls/PE_array.h:19-42`의 `_1D_PE_array`가 핵심이다. `IN_CH_PARA`개의 입력채널을 병렬 누산하되, **두 개의 weight와 두 개의 activation을 비트시프트로 한 operand에 합쳐** 단일 곱셈기(DSP48의 27x18 곱셈)로 4개의 부분곱을 동시에 만든다.

```
weight_shrink = (w1 << 11) + w0        // PE_array.h:34, 두 weight를 한 18bit operand로
in_act_shrink = (in1 << 22) + in0      // PE_array.h:35, 두 activation을 한 27bit operand로
result_shrink = weight_shrink * in_act_shrink   // PE_array.h:37, DSP 1회로 4곱
```

이 한 번의 곱셈 결과 45bit 안에 `w0*in0`, `w0*in1`, `w1*in0`, `w1*in1` 네 개의 부분곱이 서로 다른 비트 구역에 자리잡는다. 곱 결과를 다시 11bit씩 잘라(`acc_shrink(43,33)`, `(32,22)`, `(21,11)`, `(10,0)`) 부호 보정(`+ acc_shrink[하위비트]`로 carry 보정)을 거쳐 4개의 누산기로 분배한다(`PE_array.h:136-143`). 시프트 폭 11/22는 4bit weight × 4bit act에 가드비트를 둔 값으로 추정.

> 의미: 2개의 conv window(out0/out1, 즉 가로로 인접한 두 출력픽셀)와 2개의 출력채널을 동시 처리 → 한 DSP가 4 MAC을 담당(`PE_array.h:13-15`, `:44-47` 주석 명시).

#### (b) 2D PE Array — IN_CH_PARA × (OUT_CH_PARA/2) DSP 격자

`SJTU_microe/hls/PE_array.h:67-179`의 `_2D_PE_array_act`가 2차원 PE 배열 본체다.

- 행 방향 병렬 = `IN_CH_PARA`(입력채널), 열 방향 병렬 = `OUT_CH_PARA`(출력채널, 단 4MUL/DSP 때문에 실제 DSP 열 수는 `OUT_CH_PARA/2`). 실효 병렬도 = `IN_CH_PARA × OUT_CH_PARA × 2`(`PE_array.h:45-46` 주석).
- 입력 재사용 버퍼: `in_buffer0/1[IN_CH_ITER]`를 BRAM(RAM_S2P)으로 두고(`PE_array.h:84-87`), 출력채널 타일 첫 반복(`out_ch_iter_cnt==0`)에서만 스트림을 읽어 저장하고, 이후 출력채널 타일들은 버퍼에서 재사용(`PE_array.h:103-112`). → **input stationary 류의 재사용**.
- 누산 루프는 `#pragma HLS PIPELINE II=1`로 1사이클당 한 타일 처리(`PE_array.h:100`). 출력채널 루프는 `OUT_CH_PARA/2`까지 UNROLL(`PE_array.h:125-126`).
- 모든 입력채널 타일을 다 돌면(`in_ch_iter_cnt==IN_CH_ITER`) 그 즉시 BN+양자화 ReLU를 적용해 출력 스트림에 쓴다(`PE_array.h:150-166`). → 연산-정규화 융합.
- 데이터 교체 순서는 "(안쪽)conv window → OUT_CH → IN_CH → inner_core(바깥)"로 설계 주석에 명시(`PE_array.h:47`).

`_2D_PE_array_act_L1`(`PE_array.h:238-346`)은 **첫 레이어 전용 변형**으로, 입력 activation이 8bit이므로 4MUL이 아니라 **2MUL/DSP**만 패킹한다(시프트가 16, operand 분해도 16bit 단위 — `PE_array.h:206`, `:306-309`).

#### (c) conv 래퍼 — line buffer → width 조정 → PE array

`SJTU_microe/hls/conv3x3.h:40-75`의 `conv3x3_bn_act`가 위 PE array를 감싸는 dataflow 파이프라인이다:
1. `padding<...,1>`로 P=1 제로패딩(`conv3x3.h:61`).
2. `StreamingDataWidthConverter_Batch`로 `IN_CH` 폭을 `IN_CH_PARA` 폭으로 조정(`conv3x3.h:65`).
3. `Shift_Register_2O`로 **두 개의 conv window를 동시에 추출**(out0/out1) → 4MUL/DSP의 두 window를 공급(`conv3x3.h:70`).
4. `_2D_PE_array_act` 호출(`conv3x3.h:73-74`).
전체가 `#pragma HLS DATAFLOW`(`conv3x3.h:49`)로 묶여 task-level 파이프라인.

`Shift_Register_2O`(`shift_reg.h:23-113`)는 BRAM으로 라인버퍼를 흉내내며(`shift_reg.h:42-43`), 버퍼가 가득 차면 K×K×IN_CH_ITER 사이클에 걸쳐 `(j)`와 `(j+S)` 두 위치의 window를 동시에 out0/out1로 내보낸다(`shift_reg.h:76-89`). 이것이 2-window 병렬의 데이터 공급원.

#### (d) conv1x1 — 검출 헤드

`SJTU_microe/hls/conv1x1.h:31-141`은 sliding window 없이 `_1D_PE_array`(4MUL/DSP)를 그대로 재사용해 1x1 conv를 수행하며, BN을 적용하지 않고 raw psum(`M_BIT_CONV1x1=32`, `top.cpp:32`)을 그대로 출력한다(`conv1x1.h:122-128`). 최종 검출 출력은 좌표 회귀라 양자화하지 않는 설계로 추정.

#### (e) top 그래프 — UltraNet_Bypass

`SJTU_microe/hls/top.cpp:41`의 `UltraNet_Bypass`가 전체 배선이다. 흐름:
입력 AXIS → ExtractPixels → 폭변환 → `resize_batch`(640x360→320x160, `top.cpp:103`) → conv0(L1특수) → maxpool0 → conv1~conv2 (+maxpool) → conv3에서 **broadcast로 분기**(`top.cpp:416`) → 한쪽은 reorg(space-to-depth, `top.cpp:427`), 다른쪽은 maxpool3→conv4~conv6 → `StreamConcat`로 reorg결과와 conv6 결과 결합(`top.cpp:594`) → conv7 → conv8(1x1 헤드) → AddLast로 AXIS 출력(`top.cpp:699`). 이 bypass+concat은 YOLOv2-tiny의 passthrough(reorg) 구조와 동형(추정).
각 레이어 weight/inc/bias는 `#pragma HLS ARRAY_PARTITION ... complete dim=1`로 PE차원 완전분할(`top.cpp:47-79`).

#### (f) BN+양자화 융합 (function.h)

`SJTU_microe/hls/function.h:29-54`의 `BN_QUReLU`가 BatchNorm을 affine 정수곱(`in*inc + bias`, `function.h:38`)으로 표현하고, DoReFa식으로 `(W_BIT-1+DATA_BIT+L_SHIFT)`만큼 우시프트해 4bit(0~15)로 클램프한다(`function.h:43-48`). 음수는 ReLU로 0(`function.h:49-51`). 즉 BN scale/bias가 학습 후 정수 `inc/bias`로 폴딩된다.

#### (g) 양자화 학습/배치 (quantization)

- `quant_ultra.py:7-26` `uniform_quantize`: DoReFa의 `round(x*n)/n` 균일 양자화. `weight_quantize_fn`(`:29-51`)은 `tanh` 정규화 후 부호있는 k-1 bit 양자화(`:45-50`). `activation_quantize_fn`(`:54-67`)은 `clamp(x,0,1)` 후 a_bit 양자화. `batchNorm2d_Q_fn`(`:87-116`)은 BN을 weight/bias로 폴딩 후 양자화. → 본 양자화가 HLS의 4w4a와 정확히 대응.
- `qnn_mem_process.py:65-203` `QNNLayerMemProcess`: 학습된 weight를 **HLS 메모리 레이아웃으로 재배열**. `w_to_hls_array`(`:82-116`)가 weight를 `[PE][tiles]` 2D로 펴고, `array_to_string`(`:9-22`)로 SIMD개 원소를 하나의 큰 정수(`SIMD*W_BIT` 비트)로 패킹한다. weight 텐서를 `(out_ch, k, k, in_ch)`로 transpose하는 부분(`:135-137`)이 PE array의 입력채널-우선 교체순서와 정합. inc/bias 비트폭은 실제 최댓값에서 자동 산출(`:222-233`). 결과적으로 `config.h`의 `*_W_TILES`, `*_INC_BIT` 등이 이 스크립트 출력과 일치(예: `config.h:18` `CONV_0_W_TILES 9`).

### 3.2 SkrSkr — SkyNet im2col 스트리밍 (Depthwise + Pointwise)

SkyNet의 핵심 사상은 **표준 conv를 depthwise(공간) + pointwise(채널)로 쪼개고**, depthwise 입력만 im2col로 3x3 패치를 펼친 뒤 채널별 독립 곱셈으로 처리하는 것이다.

#### (a) im2col_3x3 — 라인버퍼 기반 패치 전개

`SkrSkr/im2col.h:11-104`. 3줄 라인버퍼 `line[3][COL][FOLD]`(`im2col.h:26`)를 회전 인덱스 `idx[3]`로 운영(`im2col.h:24`, `:57-61`)하면서, 각 출력픽셀마다 9개 위치(3x3)를 `out.write`로 직렬 전개한다(`im2col.h:78-95`). 경계는 삼항연산으로 `PAD_AP`(패딩값) 대체 — 위/아래/좌/우 경계 조건이 9개 write 각각에 개별 인코딩됨(`im2col.h:78-95`). 출력 II=9(`im2col.h:53`), 즉 픽셀당 9사이클로 패치를 쏟아낸다.

> SJTU의 shift_reg가 PE array에 직접 window를 공급하는 것과 달리, SkrSkr는 **스트림 상에서 명시적으로 9배로 늘린 데이터**를 만든 뒤 그 뒤단 dwconv가 단순 채널곱만 하게 만든다 — 모듈 간 결합도를 낮추는 대신 중간 스트림 트래픽이 9배.

#### (b) dwconv_3x3 — 채널별 독립 3x3

`SkrSkr/dwconv.h:16-80`. `N_IO`개 채널을 병렬 누산기 `acc[N_IO]`(`dwconv.h:32-33`)로 두고, im2col이 흘려준 9-tap을 `k=0..8` 동안 채널별로 `acc[i] += x*y` 누산(`dwconv.h:53-59`), k==8에서 출력(`dwconv.h:62-72`). depthwise라 입력채널-출력채널 교차곱이 없어 MAC 수가 `N_IO`로 적다. II=1 파이프라인(`dwconv.h:47`).

#### (c) pwconv — 1x1 채널혼합 + 2MUL/DSP MAC

`SkrSkr/pwconv.h:17-113`의 `pwconv`가 pointwise(=1x1) conv 본체이자 채널 GEMM이다. 핵심은 `MAC` 인라인 함수(`pwconv.h:7-13`):

```
concatnum = (mul_a1 << 14) + mul_a2     // pwconv.h:9, weight 2개를 한 operand로
result = concatnum * mul_b              // pwconv.h:10, 8bit activation 1개 곱 → DSP 1회로 2곱
rel1 = result(26,14) + result(13,13)    // pwconv.h:11, carry 보정 후 상위곱
rel2 = result(12,0)                     // pwconv.h:12, 하위곱
```

즉 한 activation에 대해 2개의 출력채널 weight를 동시에 곱한다(2MUL/DSP). `PE_loop`(`pwconv.h:78-97`)에서 `N_IN` 입력 × `N_OUT/2` 출력쌍을 unroll하여 누산기 `acc[2*o]`, `acc[2*o+1]`에 분배(`pwconv.h:94-95`). 입력 재사용은 `line[FOLD_I]` 버퍼로 출력타일 첫 회(`fo==0`)에만 읽기(`pwconv.h:67-71`).
`pwconv_single`(`pwconv.h:126-201`)은 DSP패킹 없이 naive하게 `acc[o] += x*y`만 하는 최종 헤드용 변형(`pwconv.h:180-184`). `pwconv_old`(`pwconv.h:215-294`)는 단일 평탄 루프 버전(주석상 구버전).

#### (d) norm_actv — BN + ReLU 융합

`SkrSkr/norm_actv.h:18-77`. `x = ((a + b) * m) >> R_SHIFT`(`norm_actv.h:57`)로 bias 가산 후 mult 스케일, 우시프트 `R_SHIFT=16`(`skynet_flow.h:20`). 0~`MAXOUT(=2^BIT_OUT-1)` 클램프(`norm_actv.h:59-67`). SJTU의 `BN_QUReLU`와 수식 구조는 같으나 SkrSkr는 BN을 (bias, mult) 두 상수배열로 분리해 둠.

#### (e) bypass reorg — BRAM FIFO로 space-to-depth

`SkrSkr/bypass_unit.h:21-98`. `bypass_send_reOrg`가 `(40,80,192)` 텐서를 `(20,40,768)`로 reshape/transpose(`bypass_unit.h:23-26` 주석)하면서 두 개의 BRAM FIFO(`bp_fifo0/1`)로 분배(`bypass_unit.h:36-55`). 후단 `bypass_recv`가 다시 읽어 합친다(`bypass_unit.h:63-98`). bypass 깊이는 `4*BP_COL*BP_BLK`, `3*BP_COL*BP_BLK`로 라인 지연을 흡수(`bypass_unit.h:17-18`). SJTU의 `ReOrg_2D`(stream 한 줄로 처리)보다 큰 FIFO를 쓰는 이유는 DW/PW bundle 간 깊은 파이프 지연을 메우기 위함으로 추정.

#### (f) maxPool2x2 / findMax (검출 헤드)

- `maxPool.h:30-114`: `line[COL/2][FOLD]` 한 줄 버퍼로 2x2 풀링, (행짝/열짝) 상태 4가지로 분기(`maxPool.h:56-84`).
- `findMax.h:10-73`: 최종 1채널(`L6_PW_NOCH`) confidence map에서 top-2 위치를 argmax(`findMax.h:44-53`)하고 box 4값+conf+pos(r,c)를 출력(`findMax.h:57-67`). `FINDMAX_NLINE=14`(`findMax.h:8`) = 2개 검출 × 7값. → 후처리(NMS류)를 HW에 일부 내장.

#### (g) top 그래프 — 7 bundle dataflow

`SkrSkr/skynet_flow.cpp:11-751` 전체가 한 `#pragma HLS DATAFLOW`(`skynet_flow.cpp:17`). 각 bundle = `im2col → dwconv → reduceWidth → norm_actv → expandWidth → pwconv → norm_actv → maxpool` 패턴이 7회 반복(Bundle #1~#7, `skynet_flow.cpp:44,141,251,391,489,613,714`). 인접 bundle 간 형상 정합을 `static_assert`로 컴파일타임 검증(예: `skynet_flow.cpp:142-143` `L0_PW_NOCH==L1_DW_NCH`). bundle 사이사이 `reduceWidth`/`expandWidth`(`stream_tools.h:29-88`)로 스트림 폭을 DW/PW의 서로 다른 병렬도(`L*_DW_NIO` vs `L*_PW_NIN`)에 맞춘다 — 이 폭변환 빈도가 SkrSkr 구조의 특징.

### 3.3 iSmart — conv2d_DSPopt (3x3을 FIR로 보는 DSP 패킹)

iSmart의 차별점은 **3x3 conv의 가로 슬라이딩을 1D FIR 필터로 재해석**해서, 한 번의 곱셈으로 인접 출력 4개의 부분곱을 동시에 만드는 것이다(FINN의 DSP packing을 확장한 형태로 추정).

#### (a) weight/input 패킹

`iSmart/src/conv2d_DSPopt.hpp:192-205` `pack_weight_data`: 한 SIMD lane에서 3개의 weight(w0,w1,w2 = 같은 행의 3-tap)를 `w0*2^(2*PROD_BIT) + w1*2^PROD_BIT + w2` 형태로 하나의 큰 정수에 적층(`conv2d_DSPopt.hpp:202-203`).
`pack_input_data`(`:180-190`): 두 인접 입력픽셀 A,B를 `(A, 0, B)` 형태로 한 operand에 적층(`:186-188`).

#### (b) simd_MAC — 한 곱셈에서 4 부분곱 추출 (FIR 4-tap)

`iSmart/src/conv2d_DSPopt.hpp:245-282` `simd_MAC`. `wpack * ipack`(`:263`) 한 번으로 나온 `m`을 `PROD_BIT` 폭으로 4등분해 carry 보정 후 `p0..p3`를 뽑는다(`:266-271`):
- p0 = x0*w2, p1 = x0*w1 + x1*w2, p2 = x0*w0 + x1*w1, p3 = x1*w0 (대응관계는 `simd_MAC_compare`의 검증식 `:319-328` 및 `simd_MAC_normal`(`:208-243`)의 `r0..r3` 정의와 동일).

즉 **2개 입력픽셀 × 3-tap weight를 컨볼브하면 4개의 출력 위치 기여가 자연히 생긴다**는 FIR 성질을 이용해 1 DSP에서 4 MAC을 얻는다. `CASCADE`(`:258-264`)로 여러 SIMD lane을 한 DSP의 누산기 체인에 묶어 추가로 DSP 효율을 올림(CASCADE ≤ 4 — `:357`).

#### (c) convDSPOpt — 행 단위 FIR 누산 + BN 융합

`iSmart/src/conv2d_DSPopt.hpp:346-473`. 출력행 h, PE타일, 폭 w(2씩 증가), infold(K×SIMDNUM) 4중 루프(`:392-395`). 각 스텝에서 `simd_MAC`으로 4 부분곱(`:423-425`)을 얻어, **이전 윈도우의 잔여 부분곱(`firPartialRes`)을 다음 윈도우로 이월**하며 출력 누산기에 합친다(`:428-438`) — 이것이 3x3의 3번째 행 tap을 인접 출력으로 넘기는 FIR 시프트의 정수구현. `o_out` 시점에 `bn_qurelu_fixed`로 BN+ReLU 융합 후 2픽셀치 출력(`:451-466`). 출력 폭 `OUT_BIT*PE*2`(2픽셀 동시).

#### (d) conv3padding / row_buffer (line buffer)

`conv2d_DSPopt.hpp:111-145` `conv3padding`이 `row_buffer[SIMD/IN_PE][4][...]`(4행 순환버퍼, `:119-120`, RAM_S2P_BRAM `:122`)로 라인버퍼를 운영. `stream_in_row`(`:18-45`)가 입력을 2픽셀 폭으로 받아 버퍼에 적재(`:31` `(data1,data0)=in.read()`), `stream_out_data`(`:49-109`)가 K행 × SIMD를 읽어 패딩 처리(`:83-92`) 후 conv 입력으로 공급. 더블버퍼링(store/load idx 분리, `:126-127`)으로 행 파이프라인.

#### (e) conv1x1DSP2 — 1x1 2MUL/DSP

`iSmart/src/conv1x1DSP2.hpp:169-186` `simd_mac_DSP2`: `w1*2^PROD_BIT + w0`로 두 출력채널 weight를 적층 후 input 곱(`:179`), 누산 `acc`를 상/하위로 분해해 2채널 결과(`:184-185`). `conv1x1DSP2`(`:194-254`)가 SIMD×PE 격자로 호출하며 PE를 2씩 짝지어(`:230`) DSP당 2채널. `conv1x1convert`(`:81-107`)는 1x1 입력을 채널-인터리브 레이아웃으로 재배열하는 라인버퍼.

#### (f) 기준(naive) 경로와 BN 래퍼

`iSmart/src/conv2d.h:26-70` `conv3x3_bn_act`는 DSP패킹 없는 표준 경로(padding→SWU sliding window→폭조정→`matrix_vector_act_unit`→폭조정). DSPopt 버전과의 **A/B 비교용 baseline**으로 추정. `conv2d.h:88-126`엔 LUT 곱셈 변형(`conv3x3_bn_act_lut`)도 있어 DSP/LUT 자원 트레이드오프 실험 흔적.
`bn_qrelu2d.h:20-44`는 BN+QUReLU를 독립 스트림 모듈로 분리한 버전(`adjust_width`로 채널 직렬화 후 `bn_qurelu` 적용 — `:27-34`). `convDSPOpt`처럼 conv에 융합하지 않는 경로용.
`config.h`에 `*_SIMD_DSP6`, `*_PE_DSP6`, `*_SIMD_DSP2`, `*_PE_DSP2`(`config.h:22-23,210-211`)와 `*_INC_BIT_NEW`/`*_BIAS_BIT_NEW`(`config.h:213-229`)가 별도로 정의된 것은 DSP패킹 경로 전용 병렬도/비트폭 재튜닝의 증거.

---

## 4. 데이터플로우

세 팀 공통: **단일 AXI-Stream in → on-chip resize → 레이어별 task가 FIFO로 연결된 거대 dataflow → AXI-Stream out**. off-chip DRAM 재방문 없이 한 번에 흘려보내는 fully-streamed 구조.

- **SJTU**: AXIS→Extract→폭변환→resize→[conv(L1)→pool→...→conv3]→broadcast 분기→{reorg / pool→conv4~6}→concat→conv7→conv8(1x1)→AddLast. 각 conv 내부는 `padding→DWC→shift_reg(2window)→2D PE array→BN/quant`(`conv3x3.h:49-74`). PE array 내부 재사용은 input-stationary, 출력은 out0/out1 두 줄을 PISO로 직렬화(`top.cpp:171`).
- **SkrSkr**: AXIS→reshape→resize→[im2col→dwconv→norm→pwconv→norm→pool] ×7 bundle, bundle3 후 bypass(BRAM FIFO)로 reorg하여 bundle6 입력에 concat(`comb_stream`, `skynet_flow.cpp:606`)→findMax→add_last. 핵심 트래픽 특징: DW/PW 병렬도 차이를 메우는 reduce/expandWidth가 레이어마다 끼어듦.
- **iSmart**: SJTU와 동일 백본(config 동형)이지만 conv 내부가 `conv3padding(4행 순환버퍼)→convDSPOpt(FIR 4-tap/DSP)→BN융합`. 출력이 항상 2픽셀 폭(`OUT_BIT*PE*2`)으로 흘러 후단 폭변환 필요.

데이터 재사용 축의 차이가 핵심 분기점:
- SJTU = 출력채널 타일 동안 입력 activation 재사용(BRAM in_buffer) + 4MUL/DSP.
- iSmart = 가로 슬라이딩의 FIR 중첩으로 입력픽셀 재사용 + 4-tap/DSP.
- SkrSkr = depthwise/pointwise 분리로 연산량 자체를 줄임(MobileNet형) + 2MUL/DSP.

---

## 5. HW/SW 매핑

| 구분 | SW(호스트/학습/전처리) | HW(FPGA HLS 커널) |
|---|---|---|
| SJTU | `quant_ultra.py`(DoReFa 4w4a 학습), `qnn_mem_process.py`(weight→`param.h` 패킹) | `UltraNet_Bypass`(top.cpp): conv/pool/reorg/PE array, AXIS in/out |
| SkrSkr | weight 양자화/패킹 SW는 본 디렉토리 미동봉(생성된 `skynet_flow.h` 상수배열만 존재) → **확인 불가** | `skynet_flow`(.cpp): 7-bundle DW/PW + bypass + findMax |
| iSmart | 양자화/패킹 SW 미동봉(`config.h`만 존재) → **확인 불가** | `conv3x3_bn_act_DSPopt`/`conv1x1_DSPopt` 등 |

- **HW/SW 경계**: 세 팀 모두 top 함수가 `#pragma HLS INTERFACE axis ... port=in/out` + `s_axilite`(SJTU `top.cpp:42-45`) 또는 `ap_ctrl_none`(SkrSkr `skynet_flow.cpp:13`)로, **호스트는 이미지 DMA만 하고 전체 추론은 HW가 단발 실행**. SkrSkr의 `ap_ctrl_none`은 free-running 스트림 가속기(호스트 제어 최소화)임을 의미.
- **가중치 적재**: SJTU는 weight를 HLS 소스 내 `const` 배열(`param.h`)로 컴파일타임 박아넣고 `ARRAY_PARTITION complete`로 레지스터/BRAM 상주(`top.cpp:47-79`). SkrSkr도 `skynet_flow.h`에 `const ap_uint<...> L*_DW/PW[...]`로 상수 상주(`skynet_flow.h:38-130`). → 모두 **weight on-chip 상주, DRAM weight fetch 없음**.
- **양자화 학습 ↔ HW 정합**: SJTU에서만 SW→HW 전체 체인 확인 가능. `quant_ultra.py`의 4w4a + BN폴딩이 `function.h`의 `BN_QUReLU` 정수연산과 1:1 대응(BN을 inc/bias 정수로 폴딩).

---

## 6. 빌드·실행

- 본 디렉토리에는 **Vivado/Vitis HLS 프로젝트 파일(tcl/solution)·Makefile·호스트 노트북이 동봉되어 있지 않음**(각 `readme.md`가 외부 repo만 안내 — `SJTU_microe/readme.md:1`, `iSmart/readme.md:1`). 따라서 정확한 빌드 절차는 **확인 불가**이며 외부 원본 repo(heymesut/SJTU_microe, xliu0709/DACSDC2021) 참조 필요.
- 코드로부터 추정되는 빌드 전제:
  - 의존 헤더: Vitis/Vivado HLS의 `ap_int.h`, `hls_stream.h`, `hls_video.h`(resize에 `Mat`/`Resize_opr_linear` 사용 — `function.h:9`, `:64-73`), `ap_axi_sdata.h`(SkrSkr — `stream_tools.h:11`).
  - `AP_INT_MAX_W 2048` 정의 필요(`top.cpp:2`, `stream_tools.h:6`) — 큰 비트폭 패킹 때문.
  - SJTU는 합성 전 `quant_ultra.py`로 학습→`qnn_mem_process.py`로 `param.h`/`config.h` 생성→HLS 합성 순서로 추정.
  - top 함수명이 곧 IP 이름: `UltraNet_Bypass`(SJTU), `skynet_flow`(SkrSkr). iSmart는 top 통합 파일이 본 발췌 범위 밖이라 **확인 불가**.

---

## 7. 의존성

- **Xilinx HLS 런타임 헤더**: `ap_int.h`, `hls_stream.h`(전 팀 공통), `hls_video.h`(SJTU resize), `ap_axi_sdata.h`(SkrSkr AXIS).
- **PyTorch + NumPy**: SJTU 양자화(`quant_ultra.py:1-4`), weight 배치(`qnn_mem_process.py:1-3`). `qnn_param_reader`(`qnn_mem_process.py:1`)는 본 디렉토리에 미동봉 → 외부 의존, **확인 불가**.
- **자동생성 파라미터 헤더**: `param.h`(SJTU, top.cpp:16) — 미동봉 생성물. SkrSkr는 `skynet_flow.h`에 인라인.
- **팀 간 의존 없음**: 세 서브팀은 서로 독립(같은 함수명 `conv3x3_bn_act`를 각자 다른 시그니처로 정의 — SJTU `conv3x3.h:40` vs iSmart `conv2d.h:26`). 공유 라이브러리 없음.
- iSmart는 `matrix_vector_unit.h`, `sliding_window_unit.h`, `function.h`에 의존(`conv2d.h:7-10`) — 이 중 `function.h`/`matrix_vector_unit.h` 본문은 본 발췌 범위 밖이라 세부 **확인 불가**(단 `bn_qurelu_fixed`, `SWU`, `matrix_vector_act_unit` 등이 그쪽에 정의됨을 호출부에서 확인).

---

## 8. 강점 · 한계

### 강점
- **DSP 멀티펌핑이 세 가지 서로 다른 방식으로 구현**되어 있어 비교연구에 이상적:
  - SJTU 4MUL/DSP(2 window × 2 out_ch, `PE_array.h:34-37`),
  - SkrSkr 2MUL/DSP(2 out_ch, `pwconv.h:7-13`),
  - iSmart 4MUL/DSP(FIR 4-tap, `conv2d_DSPopt.hpp:263-271`).
- 전 팀 fully-streamed dataflow로 **DRAM weight/activation 재방문 제거** → 저전력에 유리.
- **연산-정규화 융합**(BN+quant ReLU)을 PE 출력 직후 적용(SJTU `PE_array.h:161`, iSmart `conv2d_DSPopt.hpp:457`, SkrSkr `norm_actv.h`) → 중간 정밀도 저장 비용 절감.
- SJTU는 학습→패킹→HLS 전 체인이 동봉되어 **재현성·교육 가치 높음**.

### 한계
- **빌드 인프라 미동봉**: tcl/Makefile/host 부재로 즉시 합성·실행 불가, 외부 repo 의존(`readme.md`들).
- **SkrSkr·iSmart 양자화 SW 미동봉**: weight가 이미 패킹된 hex 상수로만 존재 → 재학습/재양자화 불가, 정밀도-자원 트레이드오프 재현 **확인 불가**.
- **하드코딩 비트시프트**(SJTU `PE_array.h:34` `<<11`, `<<22`)는 특정 비트폭(4w4a)에 강결합 — 비트폭 변경 시 시프트/추출 폭을 손으로 다시 맞춰야 함(가드비트 오류 위험).
- **im2col 9배 트래픽**(SkrSkr `im2col.h`): depthwise 입력을 9배로 부풀려 중간 FIFO 대역폭 압박. SJTU의 shift_reg 직결 방식보다 메모리 트래픽 큼.
- 검출 후처리(findMax)가 top-2 고정(`findMax.h:8` `NLINE=14`)이라 일반 검출로의 확장성 낮음.
- 모든 weight on-chip 상주는 모델이 커지면 BRAM/URAM 한계에 직면(현재는 4/5bit 초경량 모델이라 가능).

---

## 9. 우리 프로젝트(ViT/Transformer XR 시선추적 가속기, HG-PIPE 계열) 시사점

1. **DSP 멀티펌핑은 Transformer MatMul에 직접 이식 가능**. 우리 ViT의 INT4/INT8 QKV·FFN GEMM에서, SJTU/iSmart의 "두 operand를 비트시프트로 한 DSP에 패킹"(`PE_array.h:34-37`, `conv2d_DSPopt.hpp:263-271`) 기법을 그대로 채택하면 DSP당 2~4 MAC을 확보할 수 있다. ViT는 conv보다 GEMM 비중이 커서 SkrSkr의 `pwconv` 2MUL/DSP MAC(`pwconv.h:7-13`)이 가장 이식하기 쉬운 원형(pointwise=1x1=GEMM이므로).

2. **systolic/dataflow 선택지의 실증 비교**. HG-PIPE류는 layer-pipelined dataflow를 지향하는데, 세 팀 모두 동일 태스크에 대해 fully-streamed dataflow를 택했다는 점이 우리의 "전 레이어 on-chip 파이프" 방향과 정합. 특히 SkrSkr의 **bundle 간 reduce/expandWidth 폭변환 패턴**(`stream_tools.h:29-88`)은 Transformer에서 attention(작은 head 병렬) ↔ FFN(큰 채널 병렬) 사이 병렬도가 급변할 때 그대로 필요한 어댑터다.

3. **BN 폴딩 = LayerNorm/스케일 폴딩의 선례**. `BN_QUReLU`(SJTU `function.h:38-48`)와 `norm_actv`(SkrSkr `norm_actv.h:57`)의 "affine 정수곱 + 우시프트 + 클램프" 융합은, ViT의 LayerNorm 후 affine·dequant를 정수로 폴딩하는 우리 양자화 후처리에 동일 패턴으로 적용 가능(단 ViT는 LN의 평균/분산 reduction이 추가로 필요 — 본 repo엔 없음).

4. **bypass/reorg = residual/skip의 HW 원형**. SkrSkr `bypass_unit.h`의 BRAM FIFO 기반 skip(send/recv 분리, 깊이 사전산정 `bypass_unit.h:17-18`)은 Transformer의 residual add를 깊은 파이프라인에서 구현할 때의 정확한 템플릿이다. residual 한쪽 경로를 FIFO로 지연시켜 합류시키는 구조가 동일.

5. **시선추적 검출 헤드 재사용**. XR 시선추적이 동공/glint 좌표 회귀라면, SkrSkr `findMax`(confidence argmax + 좌표 출력, `findMax.h:44-67`)와 SJTU의 양자화 없는 1x1 회귀 헤드(`conv1x1.h`)가 그대로 후처리 원형이 된다. 특히 HW 내장 argmax는 호스트 후처리 지연을 없애 XR 저지연 요구에 부합.

6. **재사용 축 설계 교훈**. 세 팀의 재사용 축(SJTU=input-stationary, iSmart=FIR 픽셀 재사용, SkrSkr=연산량 자체 절감)은 ViT 가속기 설계 시 "weight-stationary vs output-stationary vs row-stationary" 선택의 구체적 정수구현 레퍼런스다. ViT는 weight(파라미터) 재사용 기회가 토큰 수만큼 크므로 SJTU의 input/weight 재사용 BRAM 버퍼(`PE_array.h:84-112`) 패턴이 가장 참고할 만함.

---

### 부록: 라인 근거가 약하거나 확인 불가한 항목 (정직성 표기)
- 타깃 보드(Ultra96/ZCU104): 코드에 명시 매크로 없음 → **추정**.
- iSmart top 통합 그래프/함수명: 본 발췌 범위 밖 → **확인 불가**.
- SkrSkr·iSmart 양자화 학습/패킹 SW: 미동봉 → **확인 불가**.
- 빌드 절차(tcl/Makefile): 미동봉 → **확인 불가**.
- SJTU 비트시프트 11/22의 정확한 가드비트 근거: 4bit×4bit + carry로 **추정**(주석엔 "4 MUL in 1 DSP"만 명시, `PE_array.h:14`).
