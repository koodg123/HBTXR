# dac_sdc_2023_champion-main 정밀 분석

> 경로: `REF/CNN-Accel/dac_sdc_2023_champion-main/`
> 분석 방식: 좁은 Glob 후 핵심 소스 Read. 라인 근거 표기.

---

## 1. 개요 (목적/대회/타깃보드)

- **정체**: SEUer 그룹(동남대) **DAC-SDC 2023 FPGA 트랙 우승** 설계 **UltraSpeed**. README.md:1, README.md:9, README.md:13.
- **목적**: 임베디드 FPGA에서 다중 객체 검출/추적 + 자율주행 (Baidu 제공 데이터셋). README.md:5, README.md:24.
- **타깃보드**: **Xilinx Kria KV260** FPGA (또는 GPU 트랙은 Jetson Nano). README.md:5.
- **모델**: UltraNet 진화형 **UltraSpeed** (저비트 양자화 강화). README.md:24. top 함수 `ultra_speed`(ultraspeed.cpp:172). 9-레이어(conv0~7 3x3, conv8 1x1, 72ch 출력 config.h:200).
- **입력 해상도**: **320×640** (config.h:6-7) — 2022(160×320)의 4배 픽셀.
- **양자화**: 대부분 W4A4 (conv0도 W_BIT=4 config.h:15, 입력만 8-bit config.h:13). 2022 champion은 conv0 W8이었음 → **conv0까지 4-bit화**가 차이.
- **핵심 기여 3가지** (README.md:22-32):
  1. UltraNet 기반 저비트 양자화 모델.
  2. **UINT-Packing** — unsigned int DSP 패킹(기존 INT-Packing 대비 MAC 효율↑, "100% 이상 성능 향상" 주장, README.md:26-28).
  3. **Non-aligned bit-width converter** — AXI 폭불일치를 1-step 변환, FIFO 깊이 192→80bit 절감 (README.md:30-32).
- DAC'23 논문 인용 (README.md:44-55, "Uint-Packing", Zhang et al.).

---

## 2. 디렉토리 구조 (자체 + 제외 이유)

Glob 결과:
```
README.md, ranking.png
src/config.h            ← 레이어 하이퍼파라미터(자체 핵심)
src/conv1x1.hpp         ← conv8 1x1 커널(자체 핵심)
src/conv2d.hpp          ← conv1~7 UINT-Packing 3x3 커널(자체 핵심★)
src/conv2d_l0.hpp       ← conv0 첫 레이어 LUTopt 커널(자체 핵심)
src/function.h          ← bn_qurelu/qurelu 양자화
src/pool_reord.hpp      ← max_pool
src/stream_tools.h      ← 비정렬 폭변환(자체 핵심, 64to24)
src/ultraspeed.cpp      ← top dataflow(자체 핵심)
src/weights.hpp         ← 가중치 생성물(거대) — 제외
src/script/script.tcl
```
- **제외**: weights.hpp(가중치 상수), ranking.png(이미지).
- **소스 미동봉**: 해당 없음. 핵심 커널 전부 실재.

---

## 3. 핵심 모듈 정밀 분석

### 3.1 Top dataflow `do_compute`/`ultra_speed` (ultraspeed.cpp:20-223)
- `#pragma HLS DATAFLOW`(ultraspeed.cpp:23). 2022와 동형의 전계층 스트리밍.
- 입력: `num_per_rep = 320*640*3*8/64`(ultraspeed.cpp:25) → `ExtractPixels`(ultraspeed.cpp:28) → **`StreamingDataWidthConverter_64to24`** (ultraspeed.cpp:33) — README의 "비정렬 1-step 변환"(64-bit AXI → 24-bit=8bit×3ch를 **한 번에**). 2022는 64→192→24 2단계였음(`StreamingDataWidthConverter_Batch` 2회) → **이것이 변환기 개선의 코드 증거**.
- conv0: `conv3x3_l0_bn_act_LUTopt`(ultraspeed.cpp:40) — 첫 레이어 LUT MAC.
- conv1~7: `conv3x3_bn_act_DSPopt`(UINT-Packing, ultraspeed.cpp:59 등). conv4~7은 풀링 없이 20×40 유지(config.h:101-191).
- conv8: `conv1x1_DSPopt` 64→72ch(ultraspeed.cpp:163, config.h:200).
- pool0~3 + AddLast(ultraspeed.cpp:168).

### 3.2 ★ UINT-Packing DSP MAC — conv2d.hpp (2022 대비 핵심 차이)

2022 INT-Packing은 부호있는 가중치를 그대로 패킹했으나, 2023은 **활성을 unsigned로 두고 가중치를 양수화 시프트**하여 곱셈을 unsigned 영역에서 수행 → 음수 캐리 오염 제거로 가드비트 절약/효율↑.

**입력 패킹** `pack_input_data` (conv2d.hpp:237-252):
- 두 활성 i1,i0(unsigned)을 `ipack[i]=(i1, 0, i0)`로 27-bit에 배치(conv2d.hpp:248). 동시에 **보정용 부분합** `subdata[1]+=i1; subdata[0]+=i0`(conv2d.hpp:249-250) 누적 — UINT 보정의 핵심.

**가중치 시프트 패킹** `pack_shiftweight_data` (conv2d.hpp:269-281):
- w0,w1,w2를 **unsigned로** `wpack[i]=(w0,0,w1,0,w2)` 패킹(conv2d.hpp:279). 즉 부호 보정을 가중치에 `(W_BIT-1)` 시프트로 흡수.

**SIMD MAC** `simd_MAC` (conv2d.hpp:283-319):
- `m += wpack[i+cs] * ipack[i+cs]` 후 PROD_BIT 단위 4슬라이스 p0~p3, 각각 r0~r3 누산(conv2d.hpp:300-313).
- **보정 차감**: `partial0 = r0 - sub0; partial1 = r1 - sub0 - sub1; partial2 = r2 - sub0 - sub1; partial3 = r3 - sub1`(conv2d.hpp:315-318). 이 sub0/sub1이 위 `subdata`를 `(W_BIT-1)` 시프트한 값(conv2d.hpp:423-424). → **unsigned 곱의 바이어스를 정확 복원**하는 것이 UINT-Packing의 수학적 핵심.

**대안 LUT 경로** `simd_MAC_DSPLUT`(conv2d.hpp:330-358): DSP 대신 `Mul_LUT`(conv2d.hpp:322-328)로 동일 4부분곱 산출 — DSP/LUT 자원 트레이드오프 선택지.

**본체** `convDSPOpt`(conv2d.hpp:360-489): 2022와 동형 4중 루프(`h→peIdx→w→infoldIdx`, conv2d.hpp:410-413, II=1 pipeline) + 동일 psum 상태머신(m_clear/o_clear/o_out, conv2d.hpp:440-464). 단 패킹이 unsigned 경로.

### 3.3 Sliding window — conv3padding (conv2d.hpp:92-154)
- `row_buffer[SIMD/IN_PE][4][...]` 4행 순환 BRAM(conv2d.hpp:100-103). `stream_in_row`(conv2d.hpp:15) 한 행 적재, `stream_out_data`(conv2d.hpp:35)에서 K×SIMD fold로 윈도우 방출. 2픽셀 동시(data0/data1) 처리(conv2d.hpp:75).

### 3.4 conv1x1 — conv1x1.hpp
- 2022 conv1x1DSP2.hpp와 동형: `streamInOneRowTwoPix`/`streamOutOneRowTwoPix`로 2픽셀 재배열(conv1x1.hpp:13-70) 후 DSP2 패킹 MAC(2출력/DSP).

### 3.5 BN/양자화 — function.h + conv2d.hpp:streamBnRelu/streamRelu
- `bn_qurelu_fixed` 사용(conv2d.hpp:229) — 2022와 동일 라운딩 시프트식 4-bit 양자화. `streamRelu`(BN 없는 버전, conv2d.hpp:159)와 `streamBnRelu`(conv2d.hpp:200) 둘 다 제공. `#pragma HLS pipeline II=INFOLD`(conv2d.hpp:177,218).

---

## 4. 데이터플로우
```
AXIS64 →ExtractPixels →64to24(1-step) 
  →conv0(LUTopt,3→16) →pool0 →conv1(UINT-pack)→pool1 →conv2→pool2 →conv3→pool3
  →conv4~conv7 (20×40) →conv8(1x1,64→72) →AddLast →AXIS
```
- 2022 대비: 입력 4배(320×640), conv0 4-bit화, 폭변환 1-step, conv1~7 UINT-Packing.

---

## 5. HW/SW 매핑
- HW(PL): conv0~8+pool+변환 단일 IP `ultra_speed`, AXIS+s_axilite(ultraspeed.cpp:175-178).
- SW(PS): HLS 스크립트로 IP 생성. 박스 후처리/호스트 코드는 본 repo 미동봉(확인 불가, KV260 PYNQ 추정).

---

## 6. 빌드 · 실행
- `vivado_hls ./src/script/script.tcl` (README.md:38-40).

---

## 7. 의존성
- Vivado HLS(`ap_int.h`,`hls_stream.h`). 가중치 weights.hpp 상수.

---

## 8. 강점 · 한계
**강점**
- **UINT-Packing**(conv2d.hpp:237-318)으로 INT-Packing 대비 가드비트/보정 효율 개선, 동일 DSP에서 처리량↑(README 100%+ 주장).
- **1-step 비정렬 폭변환**(`64to24`, ultraspeed.cpp:33)으로 FIFO 자원 절감.
- DSP/LUT 두 경로 제공(`simd_MAC` vs `simd_MAC_DSPLUT`, conv2d.hpp:283/330) → 자원 균형 튜닝 용이.

**한계**
- UINT-Packing은 보정 부분합(subdata) 추가 계산/저장 필요(conv2d.hpp:249) — 약간의 오버헤드.
- 여전히 config 하드코딩, W4A4 특화.
- 호스트/후처리 미동봉.

---

## 9. 우리 프로젝트 시사점 (ViT/Transformer + XR)
- **UINT-Packing은 ViT GEMM에 직접 매력적**: ViT의 활성은 GELU/Softmax 후 비대칭(주로 양수)이라 unsigned 가정이 자연스럽고, `subdata` 보정(conv2d.hpp:315-318)으로 정확도 손실 없이 DSP MAC을 2배 채울 수 있음 → HG-PIPE류 dataflow ViT의 MAC 밀도 향상 1순위 차용 후보.
- **비정렬 1-step 폭변환**(64to24): ViT 패치 임베딩 입력(RGB 24-bit)을 AXI 64-bit에서 직접 추출하는 데 그대로 재사용 가능.
- **DSP/LUT 듀얼 경로**: KV260급 보드에서 DSP 한계 도달 시 LUT MAC으로 일부 레이어 오프로딩 — 자원 제약 ViT 매핑 전략에 참고.
- 한계: CNN 슬라이딩윈도우 라인버퍼는 attention에 직접 재사용 불가, MAC/패킹/폭변환만 발췌 재사용이 현실적(2022 champion과 동일 결론).
