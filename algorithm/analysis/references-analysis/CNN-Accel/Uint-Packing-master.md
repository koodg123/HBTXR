# Uint-Packing-master 정밀 분석

> 분석 대상 경로: `REF/CNN-Accel/Uint-Packing-master`
> 모든 라인 근거는 `파일:라인` 형식. 라인 근거 없는 서술은 "추정"/"확인 불가"로 명시.
> 핵심 주제: **DSP 패킹(한 DSP에 여러 저비트 MAC) + UltraNet 4w4a 양자화**.

---

## 1. 개요 (목적 / 원논문 추정 / 타깃보드)

- **목적**: 하나의 FPGA **DSP 블록 안에 여러 개의 저비트 곱셈(MAC)을 패킹**하여 DSP 사용량을 줄이는 "uint-packing" 일반화 모델을 HLS로 구현. README가 직접 명시: 기존 packing은 signed 곱 기반이라 정확도 손실·면적 증가 → 이를 **unsigned integer DSP packing**으로 동시 해결(`README.md:5`).
- **원논문 추정**: README의 "unsigned integer DSP packing generalization model called uint-packing"(`README.md:5`)이라는 명명으로 보아 **동명의 논문(uint-packing) 구현체로 추정**. 정확한 서지정보는 저장소 내 없음("서지 확인 불가").
- **베이스 모델**: **UltraNet**(DAC-SDC 2020 우승 계열, 드론/보트 객체검출). 8개 3x3 conv + 1x1 검출헤드, 4w4a 양자화. 톱 함수 `ultra_net`(`ultranet.cpp:187`), 입력 160x320x3(`config_opt3.h:5-10`), 테스트 입력 `boat6_0.bin` 360x640(`ultranet.cpp:263-264`).
- **타깃보드**: 코드 내 보드 직접 명시 없음. **Vivado_HLS(vivado_hls hls_script.tcl) + Vivado RTL 흐름**(`README.md:9-17`). UltraNet 원본은 Ultra96(ZU3EG)이나 본 repo는 확인 불가 → "보드 미명시(추정: Zynq UltraScale+)".
- **정밀도**: 입력/활성 4bit(`IN_BIT=4`), 가중치 4bit(`W_BIT=4`)가 conv1~7 기본(`config_opt3.h:37-39`). 단 conv0은 입력 8bit·가중치 8bit(RGB)(`config_opt3.h:13-15`), conv8(1x1 헤드)은 가중치 8bit(`config_opt3.h:204-206`).

---

## 2. 디렉토리 구조 (자체 포함 + 제외 항목 이유)

### 자체 핵심 소스
- `src/conv2d_DSPopt3.hpp` — **3x3 conv DSP 패킹 본체**(2픽셀×3가중치 패킹, 한 곱셈에서 4결과 추출). 본 분석의 핵심.
- `src/conv1x1DSP2.hpp` — **1x1 conv DSP2 패킹**(2가중치 패킹, 한 곱셈에서 2결과). conv8 헤드용.
- `src/conv2d_l0_opt.hpp` — **레이어0(RGB, 8bit) 전용** conv. LUT 곱 + 2가중치 DSP2 패킹 fallback.
- `src/function.h` — 패딩 + **BN+양자화ReLU** `bn_qurelu_fixed`(고정소수 BN→4bit 출력).
- `src/stream_tools.h` — AXI 스트림 구조체/폭변환/ExtractPixels/AddLast 유틸.
- `src/pool_reord.hpp` — 2x2 맥스풀 `max_pool2x2`(2픽셀 패킹 스트림 대응).
- `src/config_opt3.h` — 레이어별 형상/비트폭/SIMD/PE/패킹 파라미터.
- `src/ultranet.cpp` — 톱 함수 `ultra_net`, dataflow 파이프라인 `do_compute2`, TB `main`.
- `src/param.h` — **레거시 샘플 가중치 상수 헤더**(구버전, 12bit 패킹 예시). 활성 경로(weights_opt3.hpp)와 별개.

### 제외 항목(이름만 언급)
- `src/weights_opt3.hpp` — **가중치 생성물**(conv_*_w_new/inc_new/bias_new). 대용량 상수 데이터, 분석 제외.
- `src/debug.hpp` — 디버그 보조(빈/주석).
- `scripts/hls_script.tcl`, `scripts/rtl_script.tcl` — 빌드 스크립트(이름만).
- `param.h`는 자체 소스이나 **비활성 레거시**이므로 본문에서는 패킹 비트폭 예시로만 인용(`param.h:1-21`).

---

## 3. 핵심 모듈 정밀 분석 — DSP 패킹 기법

### 3.1 패킹의 수학적 원리 (가장 중요)

#### (A) DSP2: 2가중치 패킹 → 1곱셈에서 2결과 — `simd_mac_DSP2` (`conv1x1DSP2.hpp:167-184`)
한 입력 `a`(unsigned)와 두 가중치 `w0,w1`(signed)에 대해:
- 패킹 가중치: `rst = w1*(1<<PROD_BIT) + w0`(`conv1x1DSP2.hpp:177`).
- 한 번의 곱: `m = a * rst = (a*w1)<<PROD_BIT + (a*w0)`(`conv1x1DSP2.hpp:178`).
- SIMD개 누산: `acc += m`(`conv1x1DSP2.hpp:175-180`).
- 언팩: `out0 = acc[PROD_BIT-1:0]`(하위=Σa·w0), `out1 = acc[2·PROD_BIT-1:PROD_BIT] + acc[PROD_BIT-1]`(상위=Σa·w1, 하위 부호 borrow 보정)(`conv1x1DSP2.hpp:182-183`).
- `PROD_BIT = IN_BIT + W_BIT + 2`(가드비트 2)(`conv1x1DSP2.hpp:197`). → **두 출력채널(w0=PE p, w1=PE p+1)을 한 곱셈기로 동시 계산**. conv1x1DSP2에서 `p+=2`로 PE쌍 처리(`conv1x1DSP2.hpp:228-240`).

#### (B) DSP "6"(opt3): 3가중치 × 2픽셀 패킹 → 1곱셈에서 4결과 — `conv2d_DSPopt3.hpp`
가장 정교한 패킹. 한 multiply가 **2 입력픽셀 × 2 가중치 = 4 부분곱**을 산출.
- **입력 패킹** `pack_input_data`(`conv2d_DSPopt3.hpp:212-222`): 두 픽셀 A,B를 `ipack = (A, 0…0, B)`로 배치. 즉 `ipack = A·2^PROD_BIT + B`(가운데 PROD_BIT-IN_BIT 제로 가드)(`conv2d_DSPopt3.hpp:218-220`).
- **가중치 패킹** `pack_weight_data`(`conv2d_DSPopt3.hpp:224-237`): 세 가중치 w0,w1,w2를 `wpack = w0·2^(2·PROD_BIT) + w1·2^PROD_BIT + w2`(`conv2d_DSPopt3.hpp:234-235`).
- **한 곱** `m = wpack * ipack`이 4개 세그먼트로 분리됨(`simd_MAC` `conv2d_DSPopt3.hpp:263-266`):
  - `p0 = m[PROD_BIT-1:0]` (≈ B·w2)
  - `p1 = m[2·PROD_BIT-1:PROD_BIT-1]` (≈ B·w1 + A·w2 교차항 → 부분합1)
  - `p2 = m[3·PROD_BIT-1:2·PROD_BIT-1]`
  - `p3 = m[4·PROD_BIT-1:3·PROD_BIT-1]`
  - 누산 시 `(p>>1)+(p&1)`로 인접 세그먼트 carry/round 보정(`conv2d_DSPopt3.hpp:268-271`).
- 비트폭 정의(`conv2d_DSPopt3.hpp:301-303`): `PROD_BIT=W_BIT+IN_BIT+GUARD_BIT`, `WPACK_BIT=3·W_BIT+2·IN_BIT+2·GUARD_BIT`, `IPACK_BIT=2·IN_BIT+W_BIT+GUARD_BIT`. GUARD_BIT=3(`ultranet.cpp:59`에서 conv3padding 호출 시 `3`, `conv2d_DSPopt3.hpp:481` convDSPOpt 템플릿 인자 `3`).
- **CASCADE**: `simd_MAC`에서 SIMD를 CASCADE 단위로 묶어 `m += wpack[i+cs]*ipack[i+cs]` 후 분해(`conv2d_DSPopt3.hpp:254-273`). CASCADE≤4(`conv2d_DSPopt3.hpp:298`). → DSP 캐스케이드 체인 활용(추정).

> **정리**: DSP2는 ×2(2가중치), DSPopt3는 한 곱셈에서 (2픽셀)×(서로 다른 K행 가중치)를 동시에 → 3x3 K=3행을 wpacks0/1/2로 나눠 처리(`conv2d_DSPopt3.hpp:369-399`). 이것이 "한 DSP에 여러 저비트 MAC" 패킹의 구현.

### 3.2 3x3 conv 본체 `convDSPOpt` (`conv2d_DSPopt3.hpp:287-449`)
- 루프: `h(OUT_H) → peIdx(OUT_CH/PE) → w(OUT_W/2) → infoldIdx(IN_CH/SIMD)`, 최내 `PIPELINE`(`conv2d_DSPopt3.hpp:354-358`). **w를 /2 처리**(2픽셀 동시) = 패킹의 공간적 의미.
- 입력 한 read에 K=3행 × (data0,data1)=2픽셀 묶음 수신: `(data1[0],data0[0],...,data1[2],data0[2])=vec.read()`(`conv2d_DSPopt3.hpp:365`).
- 3행 각각 `pack_input_data`로 ipack0/1/2 생성(`conv2d_DSPopt3.hpp:366-368`).
- PE별 3행 가중치 패킹 wpacks0/1/2(K=3 탭씩)(`conv2d_DSPopt3.hpp:369-383`).
- PE별 `simd_MAC` 3회(행0/1/2) → firPartial00..23(`conv2d_DSPopt3.hpp:385-399`).
- **부분합 상태머신**(m_clear=w==0, o_clear=infold==0, o_out=마지막 infold): 픽셀쌍 경계 firPartialRes(이전 픽셀의 잔여)와 현재 outPartialArr0/1 누적을 4분기로 관리(`conv2d_DSPopt3.hpp:359-427`). 컨볼루션 윈도우가 가로로 겹치는 2픽셀 구조의 부분합 캐리 처리.
- 마지막 infold(`o_out`)에서 PE 결과 2픽셀(out_buf0/1)을 합쳐 write(`conv2d_DSPopt3.hpp:429-438`).

### 3.3 라인버퍼/패딩 `conv3padding` + `stream_in_row`/`stream_out_data` (`conv2d_DSPopt3.hpp:14-170`)
- **4행 회전 라인버퍼** `row_buffer[SIMD/IN_PE][4][...]`, BRAM 바인딩(`conv2d_DSPopt3.hpp:115-120`).
- `stream_in_row`(`conv2d_DSPopt3.hpp:14-35`): 2픽셀폭(IN_PE·IN_BIT·2) 단위로 한 행 적재.
- `stream_out_data`(`conv2d_DSPopt3.hpp:38-105`): 현재 outRowIdx 기준 3행(K=3) 윈도우를 추출, 경계는 0패딩(`conv2d_DSPopt3.hpp:77-87`), data0/data1(2픽셀)을 K행 묶어 출력(`conv2d_DSPopt3.hpp:90`).
- `conv3padding`(`conv2d_DSPopt3.hpp:107-170`): 2행 선적재 후 in/out을 행단위 ping-pong, storeBufferIdx/loadBufferIdx 2bit 회전(`conv2d_DSPopt3.hpp:124-169`). 명시적 im2col 대신 **라인버퍼 스트리밍**.

### 3.4 레이어0(RGB) `conv2d_l0_opt.hpp`
- 입력 8bit 3채널이라 채널수 적음 → **DSP 패킹 대신 LUT 곱**을 기본 사용: `conv_mul_lut`(`#pragma HLS RESOURCE core=Mul_LUT`)(`conv2d_l0_opt.hpp:120-127`), `simd_mac9_LUT`로 9탭 누산(`conv2d_l0_opt.hpp:131-153`).
- DSP2 패킹 버전 `simd_mac9_DSP2`도 정의되어 있으나(`conv2d_l0_opt.hpp:156-178`), 실제 호출은 LUT 버전(`convDSPOpt_l0` `conv2d_l0_opt.hpp:253-258`) → **레이어0은 LUT 곱 채택**(채널 적어 패킹 이득 작음 → DSP 절약을 LUT로 흡수, 추정).
- 9탭(3x3)을 ivec/ivec1/ivec2(3행)으로 분리 처리(`conv2d_l0_opt.hpp:228-261`).
- 라인버퍼는 별도 l0 버전 `conv3padding_l0`(4행, IN_W+2 패딩)(`conv2d_l0_opt.hpp:91-118`).

### 3.5 BN + 양자화 ReLU `bn_qurelu_fixed` (`function.h:119-142`)
- 고정소수 BN: `bn_res = in*inc + bias`(int, `function.h:128`).
- 우측시프트 라운딩 + ReLU + 4bit 클램프: `D=1<<(W_BIT-1+DATA_BIT+L_SHIFT)`, 양수면 `(bn_res+D/2)>>(W_BIT-1+DATA_BIT+L_SHIFT)`, `>15`면 15, 음수면 0(`function.h:126-141`). → **출력 4bit unsigned(0~15) 양자화**(4a).
- L_SHIFT=8(`config_opt3.h:20`), INC/BIAS 비트폭은 레이어별(`config_opt3.h:16-17`). `bn_qurelu`(비-fixed)는 `in`을 IN_BIT로 받는 구버전(`function.h:94-117`).
- conv2d_DSPopt3의 `streamBnRelu`가 2픽셀(PE·2)씩 BN 적용(`conv2d_DSPopt3.hpp:175-210`).

### 3.6 풀링 `max_pool2x2` (`pool_reord.hpp:23-58`)
- 2픽셀 패킹 스트림(`PE·IN_BIT·2`) 입력. `max2_PE`로 PE병렬 elementwise max(`pool_reord.hpp:9-21`).
- row_store로 2x2 윈도우(가로 max → 세로 max) 처리, `w%2&&h%2`에서 출력(`pool_reord.hpp:45-57`).

### 3.7 1x1 헤드 + reorder `conv1x1DSP2.hpp`
- `conv1x1convert`(`conv1x1DSP2.hpp:77-105`): 2픽셀 스트림을 SIMD폭으로 재배열(ping-pong row_buffer)(`conv1x1DSP2.hpp:92-104`).
- `conv1x1DSP2`(`conv1x1DSP2.hpp:186-252`): SIMD누산, PE쌍(p,p+1) DSP2 패킹, 마지막 infold에서 bias 더해 출력(`conv1x1DSP2.hpp:228-248`).
- `conv1x1_DSPopt`(`conv1x1DSP2.hpp:254-269`): convert→DSP2를 DATAFLOW로 연결. conv8(36채널 헤드)에 사용.

### 3.8 톱 dataflow `do_compute2` / `ultra_net` (`ultranet.cpp:25-238`)
- `#pragma HLS DATAFLOW`로 **전 레이어를 온칩 스트림으로 연결**(`ultranet.cpp:27`). DRAM 왕복 없는 **완전 스트리밍 파이프라인**.
- 입력: AXI64 → ExtractPixels → 폭변환 2단(64→192→IN_CH·IN_BIT)(`ultranet.cpp:29-45`).
- CONV0(l0,LUT) → POOL0 → CONV1~3(+POOL) → CONV4~7(풀링 없음) → CONV8(1x1) → AddLast(`ultranet.cpp:52-184`). 각 스트림 depth 명시(`ultranet.cpp:53-177`).
- `ultra_net`: AXIS in/out + s_axilite reps, 모든 가중치/inc/bias 배열 ARRAY_PARTITION(`ultranet.cpp:187-238`).

### 3.9 레이어별 파라미터 — `config_opt3.h`
- conv0: 3→16ch, IN_BIT8/W_BIT8, SIMD3/PE8, INPE3(`config_opt3.h:1-23`).
- conv1~3: 채널 점증(16→32→64→64), 4w4a, SIMD/PE 레이어별(`config_opt3.h:25-95`).
- conv4~7: 64→64, SIMD2/PE2, DSP6용 SIMD_DSP6=4(`config_opt3.h:97-191`).
- conv8: 1x1, 64→36ch, W_BIT8, SIMD_DSP2=4/PE_DSP2=2(`config_opt3.h:193-211`).
- INC/BIAS 비트폭 두 세트(구/신) 존재, 현재 `*_NEW` 사용(`config_opt3.h:214-232` vs ultranet.cpp 호출이 `_NEW` 참조).

---

## 4. 데이터플로우

```
AXI64 in → ExtractPixels → WidthConv(64→192→48) (ultranet.cpp:29-45)
  → CONV0 (l0, LUT 곱, 8w8a→4a) → POOL0           (ultranet.cpp:56-69)
  → CONV1(3x3 DSPopt, 4w4a) → POOL1                (ultranet.cpp:75-88)
  → CONV2 → POOL2 → CONV3 → POOL3                  (ultranet.cpp:94-126)
  → CONV4 → CONV5 → CONV6 → CONV7 (풀링 없음)       (ultranet.cpp:132-174)
  → CONV8 (1x1 DSP2 패킹 헤드)                       (ultranet.cpp:178-181)
  → AddLast → AXI64 out                            (ultranet.cpp:183)

[각 3x3 conv 내부]  (conv3x3_bn_act_DSPopt, conv2d_DSPopt3.hpp:461-488)
  conv3padding(라인버퍼,2픽셀×3행 윈도우)
    → convDSPOpt(입력2픽셀+가중치3 패킹 → 1곱셈 4결과, output-stationary 부분합)
    → streamBnRelu(고정소수 BN+ReLU → 4bit)
```

- **전체가 HLS DATAFLOW 온칩 스트리밍**(`ultranet.cpp:27`): repo2와 달리 레이어 간 DRAM 왕복 없음 → HG-PIPE식 파이프라인에 더 가까움.

## 5. HW/SW 매핑

- **거의 100% HW(PL)**: `ultra_net`이 AXIS 입출력만 가진 단일 가속기. 호스트 코드는 TB(`ultranet.cpp:262-288`)에서 bin 로드/스트림 주입만 담당.
- SW 후처리(NMS/박스디코딩)는 저장소 내 미동봉 → "후처리 SW 확인 불가"(TB는 출력 크기만 확인 `ultranet.cpp:285`).
- 양자화 파라미터(inc/bias/Q)는 오프라인 학습+추출 산출물(weights_opt3.hpp, 제외).

## 6. 빌드 · 실행

- HLS: `cd scripts && vivado_hls hls_script.tcl`(`README.md:11-13`).
- RTL/Vivado: `vivado -mode tcl -source rtl_script.tcl`(`README.md:15-17`).
- TB 실행: `main`이 `../data/boat6_0.bin` 로드 후 `ultra_net` 호출(`ultranet.cpp:262-288`).
- 입력 포맷: 160x320x3, 8bit, 64bit 라인당 8픽셀(`ultranet.cpp:267-279`).

## 7. 의존성

- Xilinx HLS: `ap_int.h`, `hls_stream.h`(`conv2d_DSPopt3.hpp:4-5`, 전 파일 공통).
- C++ 표준: `<fstream>`, `<iostream>`(TB)(`ultranet.cpp:8-10`).
- 가중치 헤더 `weights_opt3.hpp`(생성물, 제외)(`ultranet.cpp:13`).
- 외부 라이브러리 없음(순수 HLS).

## 8. 강점 · 한계

**강점**
- **DSP 패킹의 정수**: 1 DSP에서 DSP2=2 MAC, DSPopt3=최대 4결과(2픽셀×2가중치) → DSP 효율 수배(`conv1x1DSP2.hpp:167-184`, `conv2d_DSPopt3.hpp:212-279`).
- unsigned 입력 + signed 가중치 패킹으로 부호처리 단순화(carry/round 보정 `>>1+&1`)(`conv2d_DSPopt3.hpp:268-271`).
- 완전 스트리밍 DATAFLOW(레이어 융합, DRAM 왕복 0)(`ultranet.cpp:27`).
- 레이어별 SIMD/PE/비트폭 튜닝 가능(`config_opt3.h`).
- 4w4a 초저비트로 메모리/대역폭 극소화.

**한계**
- 패킹 비트폭 관리가 복잡·오류민감: GUARD_BIT 부족 시 세그먼트 오버랩 손상(코드가 `(p>>1)+(p&1)` 보정에 의존)(`conv2d_DSPopt3.hpp:263-271`). 정확도 영향은 코드만으로 검증 불가("확인 불가").
- K=3 하드코딩(`conv2d_DSPopt3.hpp:113` static_assert), 형상 변경 시 재작성 필요.
- UltraNet 전용으로 형상이 고정(config_opt3.h), 범용성 낮음.
- 후처리/정확도 평가 코드 부재(TB가 크기만 확인).
- DSPopt3 부분합 상태머신(4분기)이 난해 → 유지보수 비용 높음(`conv2d_DSPopt3.hpp:401-425`).

## 9. 우리 프로젝트 시사점 (ViT/HG-PIPE + XR 시선추적 관점)

- **DSP 패킹 재사용(가장 가치 큼)**: ViT의 핵심 연산(QKV/Attention/MLP의 GEMM)은 저비트 양자화(INT4/INT8) 시 `simd_mac_DSP2`(`conv1x1DSP2.hpp:167`)·`simd_MAC`(`conv2d_DSPopt3.hpp:240`)의 패킹을 **systolic PE 곱셈기에 그대로 이식** 가능. 특히 1x1 conv=행렬곱이므로 `conv1x1DSP2`는 ViT 선형층 가속의 직접 프로토타입. HG-PIPE가 DSP 한정 자원에서 throughput을 올리려면 이 패킹이 결정적 레버.
- **dataflow 관점**: 본 repo의 `#pragma HLS DATAFLOW` 전레이어 융합(`ultranet.cpp:27`)은 HG-PIPE의 "완전 파이프라인" 철학과 일치. 우리 ViT 가속기도 인코더 블록을 온칩 스트림으로 연결(레이어 간 DRAM 왕복 제거)하는 방향이 옳음을 입증하는 레퍼런스.
- **양자화 재사용**: `bn_qurelu_fixed`의 고정소수 BN+라운딩시프트+클램프(`function.h:119-142`)는 ViT의 양자화 GELU/LayerNorm affine에 응용 가능(activation을 정수 도메인에 유지). 단 ViT는 ReLU 대신 GELU/softmax라 클램프 대신 LUT/PWL 필요(본 repo엔 없음 → 우리가 추가해야 함).
- **systolic 관점**: 본 repo는 systolic이 아닌 스트리밍+공간 PE이나, **패킹된 곱셈기 셀**은 systolic array의 PE로 캡슐화하기에 이상적. "한 PE = 한 패킹 DSP = 2~4 MAC"로 두면 동일 DSP 예산으로 어레이를 2~4배 키울 수 있음(추정).
- **XR 시선추적 특화**: 시선추적은 저지연·저전력이 필수. 4w4a + DSP패킹은 **소형 백본/회귀헤드를 극저자원으로** 실현 가능. 본 repo의 스트리밍 입력(ExtractPixels/WidthConv `ultranet.cpp:29-45`)은 카메라 직결 저지연 프론트엔드로 차용 가능. 다만 ViT는 4bit에서 정확도 저하 위험이 커서, attention은 INT8·MLP는 INT4 같은 **혼합 비트폭 + 레이어별 패킹**(config_opt3.h식 레이어별 파라미터화)이 현실적 전략.
- **주의(안티패턴 경고)**: GUARD_BIT/세그먼트 폭 설계가 틀리면 패킹 결과가 조용히 손상됨. 우리가 채택 시 **각 패킹 곱에 대한 비트정확 골든 검증**(repo1의 testbench식)을 반드시 병행해야 함.
