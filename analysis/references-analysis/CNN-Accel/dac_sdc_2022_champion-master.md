# dac_sdc_2022_champion-master 정밀 분석

> 경로: `REF/CNN-Accel/dac_sdc_2022_champion-master/`
> 분석 방식: Glob으로 좁은 경로 탐색 후 핵심 `.cpp/.hpp/.h` 개별 Read. 라인 근거(파일:라인) 표기.

---

## 1. 개요 (목적/대회/타깃보드)

- **정체**: SEUer 그룹(동남대, Southeast University)의 **DAC-SDC 2022 FPGA 트랙 우승** 설계. README.md:1, README.md:5, README.md:9.
- **목적**: 임베디드 FPGA에서 저전력 객체 검출/추적 (DJI 제공 드론 데이터셋). README.md:5, README.md:17.
- **타깃보드**: **Ultra96 v2** (Zynq UltraScale+ MPSoC). README.md:17.
- **모델 계열**: **UltraNet** 기반 (UltraNet은 DAC-SDC 2020 우승작). `ultranet.cpp`의 top 함수명 `ultra_net` (ultranet.cpp:187). 9-레이어 풀 컨볼루션 YOLO형 검출기(conv0~conv7 = 3x3, conv8 = 1x1, 36채널 출력 = 6×(4 anchor 좌표+1)형). config_opt3.h:200 (`CONV_8_OFM_CH 36`).
- **양자화**: **W4A4** (weight 4-bit, activation 4-bit). 단 첫 레이어 conv0는 입력 8-bit, weight 8-bit (config_opt3.h:13-15). 나머지 conv1~7은 `IN_BIT=4, OUT_BIT=4, W_BIT=4` (config_opt3.h:37-39 등).
- **핵심 기여**: INT-Packing 방식 DSP 패킹(1 DSP에서 4개 4-bit MAC). 본 repo는 conv2d_DSPopt3.hpp의 `simd_MAC`로 구현.

---

## 2. 디렉토리 구조 (자체 + 제외 이유)

확인된 파일 (Glob 결과):
```
README.md, ranking.png
scripts/hls_script.tcl, scripts/rtl_script.tcl
src/config_opt3.h          ← 레이어 하이퍼파라미터(자체 핵심)
src/conv1x1DSP2.hpp        ← conv8(1x1) DSP 패킹 커널(자체 핵심)
src/conv2d_DSPopt3.hpp     ← conv1~7(3x3) DSP 패킹 커널(자체 핵심)
src/conv2d_l0_opt.hpp      ← conv0(첫 레이어, LUT MAC) 커널(자체 핵심)
src/debug.hpp
src/function.h             ← bn_qurelu 양자화 활성(자체 핵심)
src/param.h                ← 가중치 생성물(거대) — 제외
src/pool_reord.hpp         ← max_pool 커널
src/stream_tools.h         ← AXIS↔스트림 폭변환(자체 핵심)
src/ultranet.cpp           ← top dataflow(자체 핵심)
src/weights_opt3.hpp       ← 가중치 생성물(거대) — 제외
```
- **제외**: `param.h`, `weights_opt3.hpp` — 양자화 가중치 상수 배열 생성물(거대). `ranking.png` — 이미지. 둘 다 알고리즘 정보 없음.
- **소스 미동봉 여부**: 해당 없음. HLS 커널 소스 전부 실재 확인.
- **특이**: 본 repo 소스는 designs-main/SEUer/ 폴더와 **동일 파일군**(아래 designs-main 문서 참조). designs-main/SEUer/README.md:1이 본 champion repo를 가리킴.

---

## 3. 핵심 모듈 정밀 분석

### 3.1 Top dataflow `do_compute2` / `ultra_net` (ultranet.cpp:25-238)

- `#pragma HLS DATAFLOW` (ultranet.cpp:27) — 전 레이어가 스트림으로 연결된 **레이어-파이프라인(dataflow) 아키텍처**. 각 레이어가 별도 하드웨어 인스턴스로 동시 동작(레이어 간 병렬).
- 입력 처리: AXIS 64-bit → `ExtractPixels`(ultranet.cpp:33) → `StreamingDataWidthConverter_Batch`로 64→192→(IN_BIT×IFM_CH)bit 단계적 폭변환 (ultranet.cpp:37-45). `num_per_rep = 160*320*3*8/64` (ultranet.cpp:29) = 입력 해상도 160×320 RGB.
- 레이어 매핑 (config_opt3.h 기준):
  - conv0: 3ch→16ch, 160×320, 첫 레이어 LUT-MAC 커널 `conv3x3_l0_bn_act_DSPopt` (ultranet.cpp:56-62).
  - pool0~pool3: `max_pool2x2` 4회 (ultranet.cpp:68, 87, 106, 125) → 공간 160×320→10×20로 16배 축소.
  - conv1~7: 3x3 `conv3x3_bn_act_DSPopt` (DSP 패킹). conv4~7은 풀링 없음(10×20 유지) — config_opt3.h:101-191.
  - conv8: 1x1 `conv1x1_DSPopt`, 64ch→36ch (ultranet.cpp:178-181).
  - `AddLast` (ultranet.cpp:183) → AXIS tlast 부여 후 출력.
- 가중치 배열은 top에서 `#pragma HLS ARRAY_PARTITION ... complete dim=1/dim=2`로 완전 분할(레지스터화) — ultranet.cpp:195-236. PE/SIMD 병렬 접근을 위함.

### 3.2 INT-Packing DSP MAC — conv2d_DSPopt3.hpp (핵심 중 핵심)

UltraScale+ DSP48E2(27×18 곱셈기, 48-bit 누산)에 **4-bit MAC 4개를 1 DSP에 패킹**하는 기법.

**입력 패킹** `pack_input_data` (conv2d_DSPopt3.hpp:212-222):
- 두 픽셀 A, B(각 4-bit×SIMD)를 받아 `ipack[i] = (A_seg, 0패딩, B_seg)` 형태로 27-bit 피연산자 안에 활성값 2개를 배치. PROD_BIT 간격 분리.

**가중치 패킹** `pack_weight_data` (conv2d_DSPopt3.hpp:224-237):
- 3개 가중치 w0,w1,w2를 `wpack[i] = w0·2^(2·PROD_BIT) + w1·2^PROD_BIT + w2` 로 18-bit 피연산자 안에 3개 배치 (conv2d_DSPopt3.hpp:234-235).

**SIMD MAC** `simd_MAC` (conv2d_DSPopt3.hpp:240-279):
- `m += wpack[i+cs] * ipack[i+cs]` 단일 곱셈으로 **(2 활성 × 3 가중치) = 부분곱 다수**를 동시 계산 (conv2d_DSPopt3.hpp:260). 곱 결과 m을 PROD_BIT 단위로 슬라이스해 partial0~3로 분리 (conv2d_DSPopt3.hpp:263-271).
- 캐리 보정: `r1 += (p1>>1)+(p1&1)` 형태 반올림 보정 (conv2d_DSPopt3.hpp:269-271).
- `CASCADE` 파라미터로 1 DSP에 cascade개 누산 (conv2d_DSPopt3.hpp:254, static_assert CASCADE<=4 conv2d_DSPopt3.hpp:298).

**컨볼루션 본체** `convDSPOpt` (conv2d_DSPopt3.hpp:282-449):
- 4중 루프 `h → peIdx → w(2픽셀씩) → infoldIdx(SIMD fold)` (conv2d_DSPopt3.hpp:354-357), `#pragma HLS pipeline` (conv2d_DSPopt3.hpp:358).
- K=3의 3행(wpacks0/1/2)을 각각 `simd_MAC` 3회 호출(conv2d_DSPopt3.hpp:389-399) — 행별 부분합 누산.
- psum 누산 상태머신: `m_clear`(w==0), `o_clear`(infold==0), `o_out`(infold==마지막) 플래그로 윈도우 경계 가로 누산 처리 (conv2d_DSPopt3.hpp:359-425). 2픽셀(out_buf0/out_buf1) 동시 출력 (conv2d_DSPopt3.hpp:429-438).

### 3.3 Sliding window / line buffer — conv3padding (conv2d_DSPopt3.hpp:107-170)

- `row_buffer[SIMD/IN_PE][4][IN_W/2·IN_CH/SIMD]` — **4-행 순환 라인버퍼**, BRAM(`RAM_S2P_BRAM`) (conv2d_DSPopt3.hpp:115-120).
- `stream_in_row`(conv2d_DSPopt3.hpp:16-35)로 한 행 적재, `stream_out_data`(conv2d_DSPopt3.hpp:40-105)로 3행 윈도우 동시 방출. 한 출력 픽셀에 대해 3×3 윈도우의 6개 워드 `(data1[0],data0[0],...,data1[2],data0[2])` 패킹 출력 (conv2d_DSPopt3.hpp:90). 2픽셀(data0/data1) 동시 처리로 폭 절반의 라인버퍼.
- 패딩은 행 인덱스 경계 검사로 zero-fill (conv2d_DSPopt3.hpp:77-87).

### 3.4 첫 레이어 LUT-MAC — conv2d_l0_opt.hpp

- conv0는 입력 8-bit×3ch로 DSP 패킹 이점이 적어 **LUT 곱셈** 사용. `conv_mul_lut`에 `#pragma HLS RESOURCE core=Mul_LUT` (conv2d_l0_opt.hpp:121-127), `simd_mac9_LUT`(conv2d_l0_opt.hpp:131-153)에서 9-탭(3×3) 누산.
- DSP2 버전(`simd_mac9_DSP2`, conv2d_l0_opt.hpp:157-178)도 존재(2가중치 패킹)하나 본체 `convDSPOpt_l0`(conv2d_l0_opt.hpp:195-274)는 LUT 버전 호출(conv2d_l0_opt.hpp:253-258).
- 라인버퍼 `row_buffer[4][IN_W+2]`로 패딩 포함 적재 (conv2d_l0_opt.hpp:97), 9-워드 윈도우 방출 (conv2d_l0_opt.hpp:87).

### 3.5 conv1x1 DSP 패킹 — conv1x1DSP2.hpp

- `simd_mac_DSP2` (conv1x1DSP2.hpp:167-184): 입력 1개 × 가중치 2개를 `w1·2^PROD_BIT + w0` 패킹 후 1곱셈으로 out0/out1 동시 산출 (conv1x1DSP2.hpp:177-183). 즉 **2-출력채널/DSP**.
- `conv1x1convert`(conv1x1DSP2.hpp:79-105): 1x1엔 윈도우 불필요, 2픽셀 폭 재배열만 수행.

### 3.6 BN + 양자화 ReLU — function.h:119-142

- `bn_qurelu_fixed`: `bn_res = in*inc + bias` (BN 융합, function.h:128) → 양수면 `(bn_res + D/2) >> (W_BIT-1+DATA_BIT+L_SHIFT)` 라운딩 시프트 후 [0,15] 클램프 → 4-bit 출력 (function.h:131-140). 음수는 0(ReLU). DoReFa식 균일 양자화. `D = 1<<(W_BIT-1+DATA_BIT+L_SHIFT)` (function.h:126).
- 스트림 래퍼 `streamBnRelu`는 conv2d_DSPopt3.hpp:175-210에 위치, M_BIT psum→OUT_BIT(4) 변환, `#pragma HLS pipeline II=2`.

---

## 4. 데이터플로우

```
AXIS64 →ExtractPixels →WidthConv(64→192→24) 
  →conv0(LUT,3→16) →pool0 →conv1(DSPpack)→pool1 →conv2→pool2 →conv3→pool3
  →conv4→conv5→conv6→conv7 (10×20 고정) →conv8(1x1,64→36) →AddLast →AXIS
```
- 전 구간 `hls::stream<ap_uint<...>>` + `#pragma HLS STREAM depth` (ultranet.cpp:32,36,41 등)로 연결. 레이어 내부도 `conv3x3_bn_act_DSPopt`가 `#pragma HLS DATAFLOW`로 padding→conv→bnrelu 3-stage 파이프 (conv2d_DSPopt3.hpp:468-487).
- 데이터 폭은 항상 `(BIT × PE × 2)` — **2픽셀 동시(폭절반 라인버퍼)** 가 일관된 설계 원칙.

---

## 5. HW/SW 매핑

- **HW(PL)**: conv0~8 + pool + 폭변환 전부 단일 IP `ultra_net`. AXIS 입출력 + s_axilite control (ultranet.cpp:190-193).
- **SW(PS)**: README상 HLS/RTL 스크립트로 IP 생성 후 Vivado 블록디자인 통합. testbench `main`(ultranet.cpp:262-288)은 `boat6_0.bin`(360×640) 로드해 C-sim. 박스 후처리(NMS 등)는 본 repo에 미동봉(추정: 호스트 Python에서 수행).
- `reps` 인자 = 배치 반복 횟수 (ultranet.cpp:188).

---

## 6. 빌드 · 실행

- HLS: `cd scripts; vivado_hls hls_script.tcl` (README.md:21-26).
- Vivado: `vivado -mode tcl -source rtl_script.tcl` (README.md:28-32).
- C-sim: `ultranet.cpp`의 `main`이 단독 검증용(ultranet.cpp:262).

---

## 7. 의존성

- Xilinx Vivado HLS (`ap_int.h`, `hls_stream.h`). 외부 라이브러리 없음(순수 HLS). 가중치는 `weights_opt3.hpp`/`param.h` 헤더 상수.

---

## 8. 강점 · 한계

**강점**
- INT-Packing DSP MAC로 4-bit MAC 4개/DSP → DSP 효율 극대화 (conv2d_DSPopt3.hpp:240-279).
- 완전 dataflow + 2픽셀 동시 처리 → 고처리량.
- BN을 inc/bias 정수상수로 융합(function.h:128) → 곱 1회로 BN+양자화.

**한계**
- 입력 해상도/레이어가 config 매크로에 하드코딩(config_opt3.h 전체) → 모델 변경 시 재합성 필요.
- W4A4 고정. 다른 비트폭은 패킹 로직(PROD_BIT 슬라이싱) 재설계 필요.
- 첫 레이어는 LUT 의존(conv2d_l0_opt.hpp:121) → LUT 자원 소모.
- 박스 후처리 미동봉(확인 불가, 호스트측 추정).

---

## 9. 우리 프로젝트 시사점 (ViT/Transformer + XR 시선추적)

- **DSP 패킹 재사용**: ViT의 Q/K/V projection·FFN GEMM도 저비트(4-bit)면 동일 INT-Packing(`pack_weight_data`/`simd_MAC` 패턴, conv2d_DSPopt3.hpp:224-279)으로 DSP당 다중 MAC 적용 가능. HG-PIPE류 dataflow ViT의 MAC 효율 핵심 차용점.
- **레이어-파이프라인(dataflow)**: 본 설계의 `#pragma HLS DATAFLOW` 전계층 스트리밍은 HG-PIPE의 "전 파이프라인" 철학과 직결. 시선추적 같은 저지연 실시간 추론에 적합.
- **BN 융합 양자화 활성**(function.h:128)은 ViT의 LayerNorm 후 양자화 단계로 변형 가능(단 LN은 통계 의존이라 직접 융합은 한계, 추정).
- **2픽셀/2출력 동시(폭절반 버퍼)** 기법은 ViT 토큰 2개 동시 처리로 매핑 가능(추정).
- 한계: 본 설계는 CNN 전용 라인버퍼/슬라이딩윈도우 구조라 attention의 전역 행렬곱엔 직접 재사용 불가 — GEMM 엔진 부분(simd_MAC)만 발췌 재사용이 현실적.
