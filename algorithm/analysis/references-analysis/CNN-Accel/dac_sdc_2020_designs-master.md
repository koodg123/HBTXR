# dac_sdc_2020_designs-master 정밀 분석

> 대상 경로: `REF/CNN-Accel/dac_sdc_2020_designs-master`
> 분석 방식: Glob/Grep/Read로 자체 핵심 HLS/Python을 함수·HLS커널 단위로 정독, 라인 근거(파일:라인) 명기.
> 추론은 "추정", 코드만으로 확인 불가한 항목은 "확인 불가"로 표기한다.

---

## 1. 개요 (목적 / 대회 / 타깃 보드)

- **목적**: DAC System Design Contest(SDC) 2020 — 저전력 임베디드 FPGA에서 단일 객체검출(single-object detection)을 고정밀·고속·저전력으로 수행하는 챔피언/입상 설계 모음. 입력 이미지에서 1개의 바운딩 박스를 회귀로 예측한다.
- **공통 태스크 구조**: 입력 RGB 이미지 → 양자화 CNN backbone → 마지막 conv(1x1)에서 anchor 기반 box 회귀 → 호스트(PS)에서 bbox 디코딩.
- **타깃 보드**:
  - ShanghaiTech_SkrSkr(SkyNet), iSmart(SkyNet 변형): HLS tcl에서 `set_part {xczu3eg-sbva484-1-e}` 확정 — **Xilinx Zynq UltraScale+ ZU3EG(Ultra96-V2)** (`ShanghaiTech_SkrSkr/hw/hls.tcl:15`, `iSmart/HLS/script_4ns.tcl:11`). 즉 GPU 트랙이 아닌 FPGA 트랙.
  - BJUT_Runner(UltraNet): HLS readme에 클럭만 명시(Period 8ns), board part는 HLS tcl 미수록이라 코드 기준 **확인 불가**. 다만 `vivado/ultranet_vivado.tcl`, `deploy/dac_sdc.bit/.hwh`, PYNQ용 `deploy/dac_sdc.ipynb`가 존재하므로 PYNQ 계열 Zynq US+ 보드(Ultra96 추정).
- **클럭 목표**:
  - BJUT_Runner UltraNet: Period 8ns(=125MHz) (`BJUT_Runner/hls/readme.md:4-5`).
  - ShanghaiTech_SkrSkr SkyNet: `create_clock -period 3` (=333MHz 목표) (`ShanghaiTech_SkrSkr/hw/hls.tcl:11`).
  - iSmart SkyNet: `create_clock -period 4` (=250MHz 목표) (`iSmart/HLS/script_4ns.tcl:11`).

세 설계는 동일 대회의 서로 다른 **가속기 아키텍처 패러다임**을 대표한다:
1. **BJUT_Runner = UltraNet** — FINN/spooNN 계열 스트리밍 데이터플로우, 4w4a 균일 양자화, SIMD×PE MVU + sliding-window. (`BJUT_Runner/hls/readme.md:8-10`에서 spooNN/BNN-PYNQ 참조 명시)
2. **ShanghaiTech_SkrSkr = SkyNet** — depthwise-separable(DW3x3+PW1x1) backbone, 출력-스테이셔너리 on-chip 타일 버퍼, 8bit feature/6bit weight 고정소수점, 단일 가속기 함수가 19개 레이어를 순차 호출.
3. **iSmart = SkyNet 변형** — 동일 SkyNet 토폴로지지만 `ap_fixed` 고정소수점 + 16-입력 MAC 트리(compute_engine_16) + DDR 머지버퍼 타일링, ReLU6, 채널 reorg.

---

## 2. 디렉토리 구조 (서브팀별 + 제외 이유)

```
dac_sdc_2020_designs-master/
├── BJUT_Runner/                  (UltraNet 4w4a — 스트리밍 데이터플로우)
│   ├── hls/ultra_net_accelerator_code/   ★ 핵심 HLS 커널
│   │   ├── ultranet.cpp          top + do_compute 데이터플로우
│   │   ├── conv2d.h              conv3x3_bn_act / conv1x1_bn_act / conv1x1
│   │   ├── matrix_vector_unit.h  MVU(SIMD×PE) + simd_mul/simd_mul_lut
│   │   ├── sliding_window_unit.h SWU/sliding_window_unit (line buffer)
│   │   ├── bn_qrelu2d.h          BN+양자화ReLU 2D 래퍼
│   │   ├── function.h            padding / bn_qurelu(융합 양자화 활성)
│   │   ├── pool2d.h              max_pool2d (SWU 재사용)
│   │   ├── stream_tools.h        폭변환/AXIS/유틸
│   │   ├── config.h              레이어별 K/S/P/CH/SIMD/PE/BIT 매크로
│   │   ├── param.h               생성된 가중치/inc/bias 초기화(거대 생성물 → 정독 제외)
│   │   ├── conv_test.cpp/res_test.cpp  테스트벤치(보조)
│   │   └── Makefile              g++ csim 빌드
│   ├── quantization/             ★ PyTorch QAT + HLS 파라미터 생성
│   │   ├── quant_ultra.py        DoReFa식 weight/act 양자화 fn
│   │   ├── ultranet_param_gen.py 레이어별 simd/pe 배정 + param.h/config.h 생성
│   │   ├── qnn_mem_process.py    w→[PE][tiles] 재배치, inc/bias 정수화
│   │   ├── qnn_param_reader.py   .npz 파라미터 리더(보조)
│   │   └── ultranet_4w4a.pt      (제외: .pt 가중치)
│   ├── model/                    학습 모델 정의(보조)
│   ├── deploy/                   .bit/.hwh/.ipynb/.so/.npy (제외: 산출물/바이너리)
│   ├── train/yolov3/             (제외: 외부 yolov3 학습 포크)
│   └── vivado/                   bd.tcl/vivado.tcl/bd.pdf (빌드 스크립트)
│
├── ShanghaiTech_SkrSkr/          (SkyNet — DW-separable, 출력 스테이셔너리)
│   ├── hw/src/                   ★ 핵심 HLS
│   │   ├── SkyNet.cpp            top SkyNet() + 모든 커널(DWCONV/PWCONV/POOL/REORG/ACT/Compute_BBOX)
│   │   ├── SkyNet.h              자료형(ADT/WDT/BDT) + 레이어 오프셋 매크로
│   │   ├── transform.cpp         stitch/distitch/DT↔DT32 패킹 (배치 4-타일 합성)
│   │   ├── utils.cpp             load/check/show 유틸(보조)
│   │   └── main.cpp              테스트벤치(보조)
│   ├── hw/hls.tcl, rtl.tcl       빌드 스크립트 (part=xczu3eg)
│   ├── SkrSkr.ipynb              PYNQ 추론 노트북
│   └── dac_sdc.bit/.hwh, *.bin   (제외: 산출물/바이너리)
│
└── iSmart/                       (SkyNet 변형 — ap_fixed, DDR 머지버퍼)
    ├── HLS/                      ★ 핵심 HLS
    │   ├── net_hls.cc            top SkyNet() + 버퍼관리/reorg/pool/bbox
    │   ├── net_hls.h             ap_fixed typedef + 함수 선언
    │   ├── conv1x1.cc            CONV_1x1_bias + compute_engine_16(16-MAC 트리)
    │   ├── dwconv3x3.cc          DW_CONV_3x3_bias + MAC_16_16
    │   ├── reorder_weight.cc     호스트측 weight 재배열 + 512-bit 패킹 .bin 생성
    │   ├── golden_c.cc/tb.cc/output_verify.cc  검증(보조)
    │   ├── script_4ns.tcl        빌드(part=xczu3eg, period 4)
    │   └── *.bin                 (제외: 가중치/feature 바이너리)
    ├── RTL/script.tcl
    └── Deploy/                   .bit/.hwh/.ipynb/.bin (제외)
```

**제외 항목 근거**: `*.bit/.hwh/.bin/.npy/.pt/.so`는 합성·배포 산출물 또는 거대 바이너리; `train/yolov3/`는 외부 yolov3 학습 포크(자체 가속기 코드 아님); `param.h`는 quantization 스크립트가 생성하는 거대 가중치 초기화물이라 함수 단위 정독 대상에서 제외(생성 로직은 `qnn_mem_process.py`로 분석).

---

## 3. 핵심 모듈 정밀 분석 (서브팀별 HLS 커널)

### 3.1 BJUT_Runner = UltraNet (스트리밍 데이터플로우, SIMD×PE MVU)

#### 3.1.1 네트워크 구조 / 양자화 스펙
- 9개 conv: conv0~conv7은 3x3(K=3,S=1,P=1), conv8은 1x1(K=1,P=0) 출력 36채널 박스회귀 (`config.h:2-184`).
- 입력 conv0: IFM 3ch @160×320, IN_BIT=8, W_BIT=4, OUT_BIT=4 (`config.h:5-15`). conv1~8은 IN_BIT=4(4w4a) (`config.h:34-36` 등).
- **다운샘플은 max_pool 4회로 처리**: conv0/1/2/3 뒤에만 pool 삽입(`ultranet.cpp:166-299`), conv4~7은 동일 10×20 해상도 유지(`config.h:90-157`).
- 입력 해상도: 호스트가 640×360을 보내고 온칩에서 320×160으로 resize (`ultranet.cpp:24-28`, `66-76`의 `hls::Resize_opr_linear`).

#### 3.1.2 Top 데이터플로우 — `do_compute` (`ultranet.cpp:84-462`)
- `#pragma HLS DATAFLOW`로 전 레이어가 stream FIFO로 연결된 **레이어-파이프라인**(`ultranet.cpp:85`).
- AXIS 64bit 입력 → `ExtractPixels`(`:91`) → `StreamingDataWidthConverter_Batch`로 64→192→24bit 변환(`:95-100`) → `resize_batch`(`:108`) → conv0_bn_act(`:131-153`) → pool(`:166`) … → conv8 `conv1x1`(`:433-448`) → `AddLast`로 AXIS 출력(`:460`).
- 모든 레이어 가중치/inc/bias는 `param.h`의 `const` 배열이며 top에서 `#pragma HLS ARRAY_PARTITION ... complete dim=1`로 PE 차원 완전 분할(`ultranet.cpp:470-502`) → PE 병렬 동시접근 보장.
- 인터페이스: in/out은 AXIS, reps와 return은 s_axilite (`ultranet.cpp:465-468`).

#### 3.1.3 Sliding Window Unit — im2col 라인버퍼 (`sliding_window_unit.h`)
- 두 구현 존재: 일반 `SWU`(`:10-125`)와 메모리 최적 `sliding_window_unit`(`:127-214`).
- `SWU`: line_buffer 크기 `K*Din_W`(`:26`), `#pragma HLS RESOURCE ... core=RAM_2P`(`:33`). 순환 큐 포인터로 K×K 윈도우를 steps×K×K 순서로 출력(`:96-122`).
- `sliding_window_unit`: 버퍼를 `(K-1)*IN_COL + K`로 축소(`:151`) — "K=3일 때 완전한 3행이 아니라 2×IN_COL+3만 있으면 의존성 해소"라는 주석(`:147-151`)이 핵심 최적화 포인트. 우→하 슬라이딩 카운터(`right_slid/down_slid`)로 stride/줄넘김 관리(`:159-211`).
- conv3x3 경로는 `SWU<3,1,...>`를 사용(`conv2d.h:58`).

#### 3.1.4 Matrix-Vector Unit — SIMD×PE (`matrix_vector_unit.h`)
- 핵심 곱셈 커널 `simd_mul`(`:42-57`): SIMD개 weight×activation을 `#pragma HLS UNROLL`로 동시 곱 후 누산. `simd_mul_lut`(`:17-32`)은 동일하나 `#pragma HLS RESOURCE variable=result core=Mul_LUT`(`:28`)로 **DSP 대신 LUT 강제** — DSP 부족 시 LUT로 오프로드하는 옵션.
- `matrix_vector_act_unit`(`:191-282`): MVU + 융합 양자화활성. 핵심 루프(`:225-280`):
  - `INPUT_FOLD=MAT_ROW/SIMD`, `OUTPUT_FOLD=MAT_COL/PE`(`:202-203`).
  - `#pragma HLS PIPELINE II=1`(`:226`).
  - `out_fold_cnt==0`일 때만 입력 읽어 `row_store`에 저장, 이후 재사용(`:230-236`) → **입력 활성 재사용**(여러 출력 타일이 같은 입력 윈도우 공유).
  - PE 루프 UNROLL로 PE개 출력채널 동시 누산(`:250-258`), `acc[p] += simd_mul<...>`.
  - INPUT_FOLD 완료 시 PE개 결과를 `bn_qurelu`로 즉시 양자화활성 후 `out_buf`에 패킹(`:266-272`) → **conv+BN+ReLU+양자화가 한 모듈 내 융합**.
- `row_store`는 `core=RAM_2P_BRAM`(`:210`).
- LUT 버전(`matrix_vector_act_unit_lut` `:418-507`)도 동일 구조, `simd_mul_lut` 사용.

#### 3.1.5 BN + 양자화 ReLU 융합 — `bn_qurelu` (`function.h:166-197`)
- 정수 도메인 BN+활성 단일 함수: `bn_res = in*inc + bias`(`:182`) — BN의 scale(inc)·shift(bias)를 정수로 흡수.
- `D = 1 << (W_BIT-1 + DATA_BIT + L_SHIFT)`(`:180`)로 재스케일 분모 결정, `bn_res>0`이면 라운딩 후 우측 시프트(`:186`), 0~15 클램프(`:187-191`)로 **4bit 부호없는 출력**, 음수면 0(ReLU). 즉 BN·ReLU·재양자화를 곱1회+가산1회+시프트1회로 구현.
- 2D 래퍼 `bn_qrelu2d`(`bn_qrelu2d.h:20-44`)는 conv8 같은 비융합 경로용(별도 `bn_qurelu<IN_BIT,OUT_BIT,INC_BIT,BIAS_BIT>` 4-인자 버전 호출, `:34`).

#### 3.1.6 Conv 래퍼 (`conv2d.h`)
- `conv3x3_bn_act`(`:35-70`): `#pragma HLS DATAFLOW`(`:43`)로 padding(`:55`)→SWU(`:58`)→폭변환(`:61`)→MVU_act(`:66`)→폭변환(`:69`)을 파이프 연결. K=3,P=1 전용(INTER_ROW=IN_ROW+2, `:45`).
- `conv1x1`(`:190-201`): SWU 없이 MVU만 호출(1x1은 윈도우 불필요).

#### 3.1.7 Max Pool (`pool2d.h:115-134`)
- K=2,S=2 전용. SWU를 재사용(`:129`)해 윈도우 생성 후 `pool_cal`(`:72-103`)로 K×K 최대값(채널별 UNROLL, `:86-95`). max_pool도 SWU를 공유하는 점이 코드 재사용의 핵심.

#### 3.1.8 양자화·파라미터 생성 (PyTorch → HLS)
- `quant_ultra.py`: DoReFa 스타일. `uniform_quantize(k)`(`:7-26`)로 round 양자화, `weight_quantize_fn`(`:29-51`)은 tanh 정규화 후 부호 1bit 포함 (k=w_bit-1) 양자화(`:35,45-50`). `activation_quantize_fn`(`:54-67`)은 [0,1] 클램프 후 a_bit 양자화 — UltraNet 4w4a의 근거.
- `BatchNorm2d_Q`(`:87-116`): BN 파라미터를 w=γ/√var, b=bias−mean·γ/√var로 폴딩(`:103-104`) 후 양자화 — 학습 시점에 BN-fold를 양자화 친화적으로 강제.
- `ultranet_param_gen.py`: 레이어별 simd/pe 수동 배정 테이블(`:15-16`, conv0 simd3/pe16 … conv4~7 simd8/pe2) → `QNNLayerMemProcess`로 `param.h`/`config.h` 생성(`:34-47`). w_bit/in_bit/out_bit 테이블(`:8-10`)이 4w4a, 마지막 layer out_bit=32(`:10`).
- `qnn_mem_process.py`: `w_to_hls_array`(`:82-116`)가 4D weight를 (out_ch,k,k,in_ch)로 transpose→2D→ `array_to_string`(`:9-22`)로 SIMD개를 한 정수로 비트패킹→ `[PE][tiles]` 배열로 재배치(`:107-116`). HLS MVU의 `weights[PE][...]` 레이아웃과 정확히 대응. `inc_bias_to_hls_array`(`:119-126`)는 inc/bias를 [PE,tiles]로 전치. inc/bias 비트폭은 실제 max값에서 자동 산출(`:222-233`).

### 3.2 ShanghaiTech_SkrSkr = SkyNet (DW-separable, 출력-스테이셔너리 타일)

#### 3.2.1 네트워크 / 자료형
- 19-레이어 config 테이블(`SkyNet.cpp:3-23`): conv1(3x3 DW)→conv2(1x1 PW)→pool … reorg/concat 포함, conv13(1x1)에서 box 회귀. backbone은 **DW3x3 + PW1x1 반복(MobileNet식 depthwise-separable)**.
- 자료형(`SkyNet.h:25-33`): activation `ap_uint<8>`(ADT), bias/중간 `ap_int<16>`(BDT), **weight `ap_int<6>`(WDT)** — 8bit feature / 6bit weight. 256bit 벡터형(ADT32/WDT32/BDT16)으로 32채널 묶음 DDR 전송.
- 양자화 상수: `nm=17`, `qm=131072.0=2^17`, amax=255, bmax=32767 (`SkyNet.h:47-57`).

#### 3.2.2 온칩 버퍼 — 32채널 완전분할 타일
- FM1~FM4 = `[32][43][83]` 타일 버퍼(`SkyNet.cpp:25-28`), WBUF3x3 `[3][32][3][3]`, WBUF1x1 `[2][32][32]`, BBUF/MBUF `[3][32]`(`:30-33`).
- 모든 커널이 `#pragma HLS ARRAY_PARTITION variable=... dim=1 complete`로 **채널(32) 차원 완전분할** → 32채널 동시 처리.

#### 3.2.3 Depthwise 3x3 — `DWCONV3X3` (`SkyNet.cpp:73-93`)
- IFM/OFM/W 모두 dim=1 complete partition(`:75-77`).
- 루프 순서 i,j(커널) → h,w(공간) → c(채널) (`:79-88`). `#pragma HLS PIPELINE II=1`(`:83`)는 w 루프, c는 암시적 UNROLL(32채널 partition).
- 누산은 OFM에 직접 in-place(`odata = OFM[c][h][w]; odata += IFM[..]*WBUF3x3[c][i][j]`) (`:85-87`) → **출력-스테이셔너리**(출력 픽셀이 버퍼에 머물며 9 tap 누적). `clamp_BDT`로 16bit 포화(`:87`).

#### 3.2.4 Pointwise 1x1 — `PWCONV1X1` + `compute_engine_16` (`SkyNet.cpp:95-210`)
- `compute_engine_16`(`:95-154`): 16개 weight×activation 곱 후 **balanced adder tree**(8→4→2→1 단계, `:134-152`)로 누산. 16-입력 MAC 트리가 PW의 곱셈 기본 단위.
- `PWCONV1X1`(`:166-210`): ci를 16씩 끊어(`:176`) `LOAD_W1x1`(`:156-164`)로 16ch weight 로드, h,w 루프 `#pragma HLS PIPELINE II=2`(`:183`), co 루프 UNROLL(32 출력채널, `:184-186`)로 32×16 MAC 동시. OFM in-place 누적(`:187-205`).

#### 3.2.5 활성 / 풀 / REORG
- `ACTIVATION`(`:253-276`): BDT IFM에 BBUF 더하기→ReLU→MBUF 곱→`>>nm`(=÷2^17) 재스케일→`clamp_adt`로 8bit(`:264-267`). 경계(h/w=0 또는 끝)는 0(`:269-272`). BN의 scale은 MBUF(곱)+nm 시프트, bias는 BBUF로 흡수 — **BN+ReLU+재양자화 융합**.
- `POOL`(`:220-232`): 2×2 max(`MAX` 4-입력, `:212-218`).
- `REORG`(`:35-61`): 80×40→40×20×4(공간→채널) space-to-depth, Cx/Rx 인덱싱으로 stride-2 재배치(`:47-56`) — SkyNet의 reorg(YOLO passthrough식) 구현.

#### 3.2.6 Top — `SkyNet()` 단일 함수 오케스트레이션 (`SkyNet.cpp:581-933`)
- m_axi 4개 번들(img/fm/weight/biasm), s_axilite return (`:583-587`).
- **`#pragma HLS ALLOCATION instances=... limit=1 function`** (`:589-597`)로 PWCONV/DWCONV/REORG/POOL/ACT/Load/Export 각 1인스턴스만 → **레이어를 시간 다중화(time-multiplexing)**해 단일 연산기로 19레이어 순차 실행(자원 절약형, 면적 최소화 전략).
- 4-타일 배치 처리: Load_IMG로 4개 이미지 영역 ping-pong 로드(`:612-659`), 레이어별 Load_WBUF/Load_BBUF로 가중치 스트리밍.
- bbox: `Compute_BBOX`(`:490-558`)가 conf 채널(OFM[4],OFM[9]) argmax로 2-anchor 중 최적 박스 선택(`:506-549`), 4-quadrant 배치별 박스(`:497-505`).

#### 3.2.7 배치 4-타일 패킹 — `transform.cpp`
- `stitch`(`:3-30`): 4개 분할 출력(2×2 quadrant, offset_h/w `:5-9`)을 패딩 128로 채운 큰 OFM에 합성. `distitch`(`:32-56`) 역연산.
- `img_DT_2_DT4`(`:73-82`): 3채널 8bit를 32bit(ADT4)로 패킹, `fm_DT_2_DT32`(`:59-71`)는 32채널을 256bit로 패킹 → DDR 버스트 효율.

### 3.3 iSmart = SkyNet 변형 (ap_fixed, 16-MAC 트리, DDR 머지버퍼)

#### 3.3.1 자료형 — ap_fixed 고정소수점
- `net_hls.h`: feature `ap_fixed<9,3>`(FIX_FM), 누산 `ap_fixed<13,4>`(FIX_FM_acc), **weight `ap_fixed<11,4>`(FIX_WT)** — 모두 `AP_RND,AP_SAT`(`:45-47`). ShanghaiTech의 정수 양자화와 달리 **ap_fixed 부동→고정 변환** 채택(round+saturate). CSIM_DEBUG 시 전부 float로 치환(`:22-42`)해 골든 비교.
- 256bit feature(uint256), 512bit weight(uint512) 버스 (`:66-67`).

#### 3.3.2 Depthwise 3x3 — `DW_CONV_3x3_bias` (`dwconv3x3.cc:33-76`)
- `MAC_16_16`(`:23-30`): 2-입력 MAC(`w1*b1+w2*b2`), `ap_fixed<24,7>` 누산.
- 본체(`:47-62`): h,w 루프 `#pragma HLS pipeline II=5`(`:49`), co 루프 UNROLL 32채널(`:50-51`). 9-tap을 5개 MAC_16_16(8쌍+1단일)로 묶어 누산(`:52-57`) 후 bias 가산(`:59`).
- ReLU6: `relu_single`(`:12-18`)이 6 상한 클램프(MobileNet ReLU6). relu==1일 때 별도 패스(`:65-75`).

#### 3.3.3 Pointwise 1x1 — `CONV_1x1_bias` + `compute_engine_16` (`conv1x1.cc`)
- `compute_engine_16`(`:11-71`): ShanghaiTech와 동형의 16-입력 곱 + balanced adder tree(`:50-69`), 단 `FIX_32_10` 고정소수점 누산.
- `load_weights`(`:76-93`): 16ch weight를 시프트레지스터로 로드(`weight_buf[co][i]=weight_buf[co][i+1]`, `:88`) — 슬라이딩 weight 버퍼.
- `CONV_1x1_bias`(`:98-155`): ci 16씩(`:116`), h,w 루프 `#pragma HLS pipeline II=2`(`:123`), coo UNROLL 32출력(`:125-126`). `first_ci_flag`/`skip`로 누산 시작·partial-sum 이어받기 제어(`:127-131`) → **여러 1x1 호출에 걸쳐 채널 누산을 분할**(채널이 32 초과일 때).

#### 3.3.4 Top — `SkyNet()` + DDR 머지버퍼 타일링 (`net_hls.cc:824-1240`)
- 인터페이스: image/weight_1x1/weight_3x3/bias/DDR_buff_merge/predict_boxes/constant 각 m_axi, return s_axilite (`:836-847`). ALLOCATION limit=1로 CONV/DW/Pool/Load_image 단일화(`:850-853`) — ShanghaiTech와 동일한 면적-우선 시간다중화.
- **DDR 머지버퍼 인덱싱**: 단일 큰 DDR 영역(`DDR_buff_merge`)을 buf_id로 분할(예: DW1_POOL_OFFSET=524288, DW2_POOL_OFFSET=786432, `:11-12`). 각 레이어 출력을 `relu_copy_buf_to_DDR(_acc)`(`:301-347`)로 DDR에 쓰고 다음 레이어가 `load_buf_from_DDR`/`load_dwX_pool_from_DDR`(`:351-531`)로 타일 로드 — **on-chip 버퍼는 작게 유지, 중간 feature는 DDR ping-pong**.
- 이미지 정규화 LUT: `img_norm_ch[256]`(`:551-568`)로 0~255 픽셀을 [-2,2] 정규화값 테이블 매핑(`load_image_chunk_norm` `:572-603`) — 나눗셈/곱 대신 LUT.
- **레이어 융합 최적화**: DW4의 1x1 출력을 DDR 안 쓰고 바로 DW5 3x3 입력으로(`Relu_Convert_FIX`→`DW_CONV_3x3_bias`, `:1062-1064`), DW5도 동일(`:1120-1122`) — 인접 레이어 fusion으로 DDR 왕복 절감.
- 채널 reorg: `load_and_reorg_part`(`:689-766`)가 256bit DATA를 32 ap_uint<8> 분해 후 ping-pong h/w 주소로 4개 출력버퍼(buf_out_1~4)에 분배(`:720-760`) — concat+reorg를 로드 단계에서 처리.
- bbox: `compute_bounding_box`(`:29-274`)가 4-quadrant×2-anchor conf argmax로 박스 선택(ShanghaiTech Compute_BBOX와 같은 의미, 코드 길게 펼침).

#### 3.3.5 호스트측 weight 재배열 / 패킹 — `reorder_weight.cc`
- `reorder_weight_fix`(`:119-666`): float weight를 (1) DW6의 768채널 4분할 인터리브 재배열(`:123-168`), (2) ap_fixed 변환 + 채널 3→32/48→64 zero-pad(`:171-328`), (3) 32채널 단위로 `fix_conv_weight_1x1_all`/`3x3_all`에 reorder(`:330-565`), (4) **32 weight를 512bit 한 워드로 패킹**(`:628-663`, `DATA.range(j*16+WT_RG, j*16)`)해 `weights_fixed.bin` 출력(`:625,664`). HLS의 512bit weight 버스 레이아웃과 정확히 대응. **이 패킹은 HLS 합성 전 호스트(SW) 단계 실행** (csim 컨텍스트).

---

## 4. 데이터플로우

### 4.1 BJUT_Runner UltraNet (레이어 파이프라인 / 스트리밍)
```
AXIS(64b) → ExtractPixels → WidthConv(64→192→24b) → resize(640×360→320×160)
→ [conv0_3x3_bn_act → pool] → [conv1 → pool] → [conv2 → pool] → [conv3 → pool]
→ conv4 → conv5 → conv6 → conv7  (10×20 유지)
→ WidthConv → conv8_1x1(36ch box) → AddLast → AXIS out
```
- 전체가 `#pragma HLS DATAFLOW` 하나로 연결된 **생산자-소비자 FIFO 스트림**. 각 conv 내부도 padding→SWU→MVU→폭변환의 서브 dataflow(`conv2d.h:43`). 가중치는 온칩 const(스트리밍 아님), feature만 스트림.

### 4.2 ShanghaiTech / iSmart SkyNet (타일 + DDR 왕복 / 시간다중화)
```
DDR(img) → 온칩타일 로드 → DWCONV3x3 → ACT → PWCONV1x1 → ACT → POOL
→ DDR 저장 → (다음 레이어) 로드 → … → REORG/CONCAT → 최종 1x1 → BBOX → DDR
```
- 단일 연산기(ALLOCATION limit=1)를 레이어마다 재호출. feature map은 `[32][43][83]` 온칩 타일 ↔ DDR(ShanghaiTech: `fm` 포인터 / iSmart: `DDR_buff_merge`)을 왕복. 가중치는 레이어별 DDR→온칩 버퍼 스트리밍.
- **두 SkyNet 차이**: ShanghaiTech는 정수(ap_int<8/6>)+`>>nm` 재스케일, 4-quadrant stitch 버퍼; iSmart는 ap_fixed+ReLU6+img정규화 LUT+인접레이어 fusion+512bit weight 패킹. 토폴로지(DW-separable+reorg+2anchor bbox)는 동일.

---

## 5. HW/SW 매핑

| 항목 | BJUT_Runner | ShanghaiTech_SkrSkr | iSmart |
|---|---|---|---|
| PL(가속기) 진입점 | `ultra_net()` AXIS (`ultranet.cpp:463`) | `SkyNet()` m_axi (`SkyNet.cpp:581`) | `SkyNet()` m_axi (`net_hls.cc:824`) |
| PS↔PL 인터페이스 | AXI-Stream(feature) + s_axilite(ctrl) (`:465-468`) | m_axi(img/fm/wt/bias) + s_axilite (`:583-587`) | m_axi(다중 버스) + s_axilite (`:836-847`) |
| 가중치 위치 | 온칩 const 배열(`param.h`), top에서 partition | DDR→온칩 WBUF 스트리밍 | DDR(512b)→온칩, 호스트 reorder 후 .bin |
| 중간 feature | 온칩 FIFO 스트림(off-chip 없음) | DDR `fm` 타일 왕복 | DDR `DDR_buff_merge` 타일 왕복 |
| SW(호스트) 역할 | resize 입력/박스 디코딩(PYNQ ipynb), QAT+param 생성(`quantization/`) | 입력 패킹/박스 후처리(SkrSkr.ipynb), stitch | weight reorder+패킹(`reorder_weight.cc`), 박스 후처리 |
| 양자화 도메인 | 정수 4w4a, BN정수폴딩 | 정수 8a6w, `>>17` 재스케일 | ap_fixed(9,3)/(11,4), AP_RND_SAT |
| 병렬화 축 | SIMD(입력ch)×PE(출력ch) | 채널32 완전분할(UNROLL) | 채널32 분할 + 16-MAC 트리 |
| 면적 전략 | 레이어별 전용 인스턴스(공간 펼침) | ALLOCATION limit=1(시간다중화) | ALLOCATION limit=1(시간다중화) |

- **공통**: 학습은 GPU(PyTorch), 양자화 파라미터는 호스트 스크립트로 HLS 친화 레이아웃 변환, 추론 가속기는 PL, 박스 디코딩·전후처리는 PS(PYNQ 노트북). 즉 명확한 SW(학습·후처리) / HW(추론 MAC) 분리.

---

## 6. 빌드 · 실행

- **BJUT_Runner**: HLS top=`ultra_net.cpp/ultra_net`, clk period 8ns (`hls/readme.md:1-5`). csim은 `Makefile`로 g++(Vivado 2018.3 include, `Makefile:4,10-11`). Vivado bd/impl은 `vivado/ultranet_vivado.tcl`. PYNQ 배포는 `deploy/dac_sdc.ipynb`(+`.bit/.hwh`). 파라미터 생성: `quantization/ultranet_param_gen.py`(→`param.h`,`config.h`).
- **ShanghaiTech_SkrSkr**: `hw/hls.tcl`(`open_project HLS`, `set_top SkyNet`, part `xczu3eg-sbva484-1-e`, period 3, SDx target, m_axi 64bit addr, reset low, prefix `a0_`, export ip_catalog)(`hls.tcl:6-23`). RTL은 `rtl.tcl`. README는 `vivado_hls hls.tcl` → `vivado rtl.tcl`(`hw/README.txt:1-2`).
- **iSmart**: `HLS/script_4ns.tcl`(`open_project project_1`, set_top SkyNet, conv1x1/dwconv3x3/net_hls 추가, part xczu3eg, period 4, csynth+export)(`script_4ns.tcl:6-19`). RTL `RTL/script.tcl`. weight bin은 csim에서 `reorder_weight_fix()`가 `weights_fixed.bin` 생성(`reorder_weight.cc:625-664`).

---

## 7. 의존성

- **공통**: Xilinx Vivado HLS 2018.x~2019.x (`ap_int.h`/`ap_fixed.h`/`hls_stream.h`).
- BJUT_Runner: 추가로 `hls_video.h`(온칩 resize `hls::Resize_opr_linear`, `ultranet.cpp:13,74`); 양자화는 PyTorch+numpy(`quant_ultra.py:1-4`), 설계 참조 FINN-hlslib/spooNN/BNN-PYNQ(`hls/readme.md:8-10`, Makefile의 finn-hlslib include 주석 `Makefile:3`).
- ShanghaiTech: `sds_lib.h`(SDSoC, `SkyNet.h:16-21`), `ap_int.h`. SDx 빌드 흐름.
- iSmart: `ap_fixed.h`, `dsp_builtins.h`(`net_hls.h:9`), `hls_math.h`(`reorder_weight.cc:4`).
- 학습/데이터: BJUT는 `train/yolov3`(외부 포크, 분석 제외), `model/quant_ultra.py`.

---

## 8. 강점 · 한계

### BJUT_Runner UltraNet
- **강점**: (1) 전 레이어 DATAFLOW 스트리밍으로 off-chip feature 트래픽 없음 → 높은 처리량·낮은 지연. (2) SIMD×PE 파라미터화로 레이어별 병렬도 튜닝 자유(quantization 스크립트가 자동 비트팩). (3) conv+BN+ReLU+양자화 완전 융합(`bn_qurelu`), Mul_LUT 옵션으로 DSP/LUT 균형. (4) 4w4a 극저비트로 자원 효율 우수.
- **한계**: (1) 가중치를 전부 온칩 const로 두므로 큰 모델엔 비적합(UltraNet은 작아서 가능). (2) 레이어별 전용 인스턴스라 면적이 레이어 수에 비례. (3) `param.h` 거대 생성물(가독성·재합성 비용). (4) resize/입력처리가 640×360 고정.

### ShanghaiTech_SkrSkr SkyNet
- **강점**: (1) DW-separable로 연산량 최소, 채널32 완전분할로 32-way 병렬. (2) ALLOCATION limit=1 시간다중화로 면적 최소(ZU3EG 소형 보드 적합). (3) 정수+시프트 재스케일로 단순·빠름(period 3ns 목표). (4) 4-quadrant 배치로 1프레임 4영역 동시.
- **한계**: (1) 단일 연산기 재사용 → 처리량은 데이터플로우형보다 낮음(레이어 직렬). (2) 중간 feature DDR 왕복 대역폭 의존. (3) `[32][43][83]` 타일 크기 고정(해상도 종속). (4) bbox 로직이 2-anchor·고정 quadrant로 하드코딩.

### iSmart SkyNet
- **강점**: (1) ap_fixed(round+sat)로 정밀도 관리 용이, CSIM에서 float 골든 비교 가능. (2) 인접 레이어 fusion(DW1x1→DW3x3)으로 DDR 왕복 절감. (3) 512bit weight 패킹·img 정규화 LUT로 대역·연산 최적. (4) 호스트 weight reorder로 HLS 내부 단순화.
- **한계**: (1) DDR 머지버퍼 인덱싱이 매우 복잡(오프셋 매직넘버 다수) → 유지보수 난이도 높음. (2) 함수 in-place 누적·skip 플래그 등 제어가 산재. (3) reorder가 호스트 단계라 별도 .bin 관리 필요. (4) ShanghaiTech 대비 코드 장황(같은 bbox 로직 4회 펼침).

---

## 9. 우리 프로젝트(ViT/Transformer FPGA + XR 시선추적) 시사점

1. **SIMD×PE MVU 패턴(BJUT)**은 Transformer의 Linear/QKV projection·FFN GEMM에 직접 재사용 가능. `matrix_vector_act_unit`의 INPUT_FOLD/OUTPUT_FOLD 타일링과 입력 재사용(`row_store`)은 attention의 Q·Kᵀ matmul 타일링에 그대로 매핑된다. HG-PIPE식 레이어-파이프라인을 노린다면 BJUT의 전-레이어 DATAFLOW 스트리밍이 참조 모델.

2. **융합 양자화활성(`bn_qurelu`, ACTIVATION)**: BN·ReLU·재양자화를 곱1+가산1+시프트1로 흡수하는 정수 도메인 기법은 ViT의 LayerNorm/GELU를 정수·시프트 근사로 융합하는 설계에 응용 가능(단 LayerNorm은 reduction 필요 → softmax/LN은 별도 reduction 데이터패스 필요, 직접 재사용은 "추정" 수준).

3. **DSP/LUT 균형(`simd_mul_lut`의 Mul_LUT 강제)**: 4bit 곱을 LUT로 오프로드하는 옵션은 DSP가 부족한 ViT 가속기에서 저비트 MAC을 LUT로 돌려 DSP를 attention의 고정밀 누산에 집중시키는 전략에 유용.

4. **호스트 weight 재배치·비트패킹(`qnn_mem_process.w_to_hls_array`, `reorder_weight.cc`)**: 가중치를 `[PE][tiles]`·512bit 워드로 사전 패킹하는 SW 전처리는 Transformer weight(큰 차원)를 HLS 버스폭에 맞춰 패킹하는 데 필수 패턴. 양자화 스크립트와 HLS 레이아웃의 일대일 대응 설계를 그대로 차용 권장.

5. **면적 전략 선택지**: BJUT(공간 펼침=처리량 우선) vs SkyNet(ALLOCATION limit=1 시간다중화=면적 우선)의 대비는 XR 시선추적의 실시간 제약(저지연) vs 소형 보드 자원의 트레이드오프 결정에 직접적 가이드. 시선추적은 저지연·소모델이므로 BJUT식 스트리밍이 유리할 "추정", 다만 ViT backbone이 크면 SkyNet식 DDR 타일링이 불가피.

6. **DDR 타일 ping-pong(iSmart `DDR_buff_merge`)**: ViT의 토큰×차원 중간 텐서가 온칩에 안 들어갈 때의 타일 스케줄링·인접연산 fusion 사례로 참조. 단 매직넘버 오프셋의 복잡성은 반면교사(파라미터화 권장).

7. **depthwise-separable backbone(SkyNet)**: 순수 ViT가 아닌 하이브리드(conv-stem + transformer)나 MobileViT류를 XR 경량 backbone으로 쓸 경우, DW3x3+PW1x1 커널과 출력-스테이셔너리 버퍼가 재사용 가능.

---

### 부록: 분석 시 확인된 사실 / 한계
- BJUT_Runner의 board part는 HLS tcl이 repo에 없어 코드만으로 **확인 불가**(클럭 8ns만 확정). deploy 산출물 존재로 PYNQ Zynq US+ 추정.
- `param.h`(BJUT), `*.bin`(가중치)은 생성물이라 함수 단위 정독에서 제외했고, 생성 로직(`qnn_mem_process.py`/`reorder_weight.cc`)으로 레이아웃을 역추적했다.
- 세 설계 모두 정확한 board 리소스(LUT/DSP/BRAM) 수치는 합성 리포트가 repo에 없어 **확인 불가**.
