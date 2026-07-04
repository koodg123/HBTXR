# yolo-fpga-accelerator 모듈 통합 가이드

> 1차 요약: [`../yolo-fpga-accelerator-main.md`](../yolo-fpga-accelerator-main.md) — 본 문서는 그 요약을 모듈 단위로 심화한 통합 가이드다.
> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\CNN-Accel\yolo-fpga-accelerator-main`
> 작성 원칙: 실제 소스 Read 후 `파일:라인` 근거 표기. 라인 근거 없는 추론은 "추정", 코드로 확인 불가는 "확인 불가"로 명시.
> 형제 가이드와 동형(H-HLS): `REF/Analysis/CNN-Accel/ESDA/MODULE_GUIDE.md`.

---

## 0. 문서 머리말

### 0.1 대표 케이스 선정
- **대표 모델: YOLOv2 (416×416, COCO 80-class), INT16 고정소수 KV260 배포 패스.** 근거: `README.md:5`가 KV260 검증을 YOLOv2 INT16으로 명시. 23개 conv 레이어 오프셋 테이블이 `model_config.cpp:4-10`에 하드코딩.
- **대표 conv layer: layer 13 (3×3 standard conv, IC=512→OC=1024, 입력 13×13, stride1, pad1).** 근거: 가중치 오프셋 `kYolo2WeightOffsets[13]=4718592`(`model_config.cpp:6`) = OC×IC×K×K = 1024×512×3×3. 가장 채널이 큰 후반 레이어라 PE 어레이(Tm32×Tn4=128 MAC)의 채널 타일링(`OFM/Tm=32`회, `IFM/Tn=128`회)이 최대로 노출되어 분석 가치가 큼. 공간(13×13)은 단일 타일(Tr=Tc=13)에 들어가 타일 루프는 채널 방향만 활성.
- **대표 첫 레이어: layer 0 (3×3 conv, IC=3→OC=32, 입력 416×416, stride1).** 근거: `kYolo2WeightOffsets[0]=864`=32×3×3×3(`model_config.cpp:4`). 공간이 가장 커서 r/c 타일 루프(416/13≈32×32 타일)와 256bit 정렬 버스트 로드가 전부 활성.
- **대표 다운샘플: MAXPOOL(2×2 s2, LayerType=1) + REORG(stride2 passthrough, LayerType=2).** 근거: `core_compute.cpp:266`(`pool_yolo2`), `:354`(`reorg_yolo2`), `core_scheduler.cpp:63/88`(분기). REORG는 양자화 스킵연결 스케일 정렬(`yolo2_model.cpp:379-399`)이 등장해 ViT residual 차용에 직결.

### 0.2 수치 표기 규약
- **MAC lanes** = `compute()`에서 `ARRAY_PARTITION complete`로 펼친 차원 곱 = **Tm×Tn = 32×4 = 128 MAC**. 근거: `input_buffer` dim1(Tn) 완전분할(`core_compute.cpp:28`), `output_buffer` dim1(Tm)(`:29`), `weight_buffer` dim1,2(Tm,Tn)(`:30-31`). 최내 `tn`(Tn) 곱이 II=1 파이프 안에서 펼쳐지고 `tm`(Tm)이 공간 병렬. systolic 아님(0.4 참조).
- **scalar MACs**(dense 기준) = 출력H×출력W×Cout×Cin×Kh×Kw. 대표 layer 13 = 13×13×1024×512×3×3 = **약 7.97 G MAC**. layer 0 = 416×416×32×3×3×3 = 약 149.5 M MAC. (입력채널이 Tn배수 아니면 zero-pad 무효 MAC 포함, `core_io.cpp:196-197`/`core_compute.cpp` 경계.)
- **loop trips**(대표 layer 13) = 공간타일 `⌈Output_h/TR⌉×⌈Output_w/TC⌉` = ⌈13/13⌉×⌈13/13⌉ = 1×1, 출력채널 타일 `OFM/Tm = 1024/32 = 32`회(`yolo2_accel.cpp:136`), 입력채널 타일 `(IFM+TN)/TN = (512+4)/4 = 129`회(`core_scheduler.cpp:45`), 커널 9탭 × Tr·Tc(13×13) 픽셀(`core_compute.cpp:65-71`). layer 0은 공간타일 ⌈416/13⌉×⌈416/13⌉ = 32×32 = 1024개.
- **memory size**(payload bit, INT16 IO_Dtype=16b 기준):
  - `input_buffer0/1`(ping-pong) = Tn×OnChipIB_Height×OnChipIB_Width = 4×27×27 = 2,916 워드 × 16b = **46,656 bit/버퍼**(`yolo2_accel.cpp:103-106`, OnChipIB=27 from `params.hpp` 생성, `hw_params_gen.py:42-43`).
  - `output_buffer/output_buffer1`(M-방향 ping-pong, LUTRAM) = Tm×Tr×Tc = 32×13×13 = 5,408 워드 × 16b = **86,528 bit/버퍼**(`yolo2_accel.cpp:107-112`).
  - `weight_buffer0/1`(ping-pong) = Tm×Tn×K×K = 32×4×3×3 = 1,152 워드 × 16b = **18,432 bit/버퍼**(`core_scheduler.cpp:21-27`).
  - `beta_buffer` = MAX_BETA_LENGTH=1024 워드 × 16b = 16,384 bit(`yolo2_accel.cpp:113`).
- **per-layer Q 도메인** = 4-파라미터 `Qw`(가중치)/`Qa_in`(입력활성)/`Qa_out`(출력활성)/`Qb`(bias). `shift_out = Qa_in+Qw-Qa_out`, `shift_bias = Qb-Qa_out`(`core_compute.cpp:48-50`). 호스트가 per-layer 인덱싱(`yolo2_model.cpp:312-321`).
- **타깃 데이터타입**: INT16 모드 `IO_Dtype=int16_t`, `Acc_Dtype=int32_t`(`types.hpp:9-10`); 곱은 int32, 누산은 int64(`core_compute.cpp:103-106`); 출력 INT16 포화 [-32768,32767](`:116-117`). FP32 모드는 둘 다 float(`types.hpp:12-13`). 컴파일 스위치 `INT16_MODE`.
- **합성 PPA**: `yolo2_report.json:3`이 HLS 리포트 디렉터리(`yolo2_int16/solution1/syn/report`)와 Vivado impl 경로만 가리키며 **저장소에 실제 리포트(rpt/util/timing)는 미동봉** → LUT/FF/DSP/BRAM/주파수/latency는 **확인 불가**. 정량은 전부 정적(소스 기반).

### 0.3 운영 경로 (호스트 스케줄러 ↔ HLS 엔진)
```
[SW 학습/양자화: Darknet YOLOv2 + nn-weight-extractor (본 repo 외부, README.md:9)]
      │ weights_reorg_int16.bin / bias_int16.bin / weight_int16_Q.bin / bias_int16_Q.bin / iofm_Q.bin
      ▼
[호스트 레이어 스케줄러: yolov2_hls_ps()  (yolo2_model.cpp:229-449)]
      │  레이어 루프 for i in net->n:
      │   CONV  → 타일 TR/TC/TM/TN 계산 → per-layer Q 인덱싱 → YOLO2_FPGA(...)
      │   MAXPOOL/REORG → YOLO2_FPGA(LayerType=1/2) + reorg는 reorg_cpu(host)
      │   ROUTE → skip(주소 핑퐁) ; REGION → dequant(2^-q)+forward_region_layer
      │   메모리 더블버퍼 배치 generate_iofm_offset() (Memory_top/bottom 핑퐁)
      ▼
[HLS 타일 엔진: YOLO2_FPGA()  (yolo2_accel.cpp:25-171)]
      │  AXI m_axi 4버스(Input/Output/Weight/Beta) + s_axilite CTRL_BUS
      │  3중 타일 루프 r(TR)→c(TC)→m(TM), m-방향 ping-pong(compute↔write_back)
      ▼
[채널 ping-pong: intra_pingpong_wrapper()  (core_scheduler.cpp:14-113)]
      │  LayerType==0 conv: n(TN) ping-pong [ copy_input_weight ‖ compute ]
      │  LayerType==1/2 pool/reorg: load ‖ pool_yolo2/reorg_yolo2
      ▼
[연산 코어 compute() (core_compute.cpp:22-173) = Tm×Tn=128 MAC, output-stationary, INT16 Q정렬]
[IO core_io.cpp(256bit burst) / write_back_output_reorg(leaky+memcpy 더블버퍼)]
      ▼
[KV260 런타임: linux_app/ — mmap CTRL_BUS(0xA0000000) + Q via AXI GPIO(0xA0010000~0xA0040000)]
```
- 타깃: **AMD Kria KV260**, Vitis/Vivado **2024.2**, HLS IP `xilinx.com:hls:YOLO2_FPGA:1.0`(`README.md:1,21,23,71`). 제어 베이스 `0xA0000000`(`yolo2_config.h:18`).
- **핵심 경계**: PL은 "범용 conv/pool/reorg 타일 엔진" 1개를 레이어마다 재사용(가중치 재로드, 레이어 간 DRAM 왕복). 호스트(PS)가 레이어 순서·메모리 배치·Q 스케일을 전담.

---

## 1. Repo / 모듈 개요

YOLOv2 FPGA accelerator = **모델 비종속 타일 엔진(`hls/core/`) + YOLOv2 모델 래퍼(`hls/models/yolov2/`) + 호스트 레이어 스케줄러 + KV260 런타임(`linux_app/`)**의 풀스택 가속기. 핵심 설계는 (1) Tm32×Tn4=128 MAC output-stationary 어레이, (2) 3중 중첩 ping-pong으로 IO-compute 은폐, (3) per-layer 4-파라미터 양자화 도메인 정렬이다. 1차 요약(`../yolo-fpga-accelerator-main.md`) 1-3절 참조.

### 1.1 호출 계층 (HW PL / SW PS / 런타임)

| 구분 | 파일(자체 소스) | 역할 |
|---|---|---|
| **HLS top(HW)** | `hls/models/yolov2/yolo2_accel.cpp` | `YOLO2_FPGA` 톱 함수: AXI 인터페이스 + 3중 타일 루프 오케스트레이션 |
| **HLS 스케줄러(HW)** | `hls/core/core_scheduler.cpp` | `intra_pingpong_wrapper`: 채널 ping-pong, LayerType 분기 |
| **HLS 연산코어(HW)** | `hls/core/core_compute.cpp` | `compute`(128 MAC), `pool_yolo2`, `reorg_yolo2`, `nonlinear_leaky_row`, `write_back_output_reorg` |
| **HLS IO(HW)** | `hls/core/core_io.cpp` | `input_load`/`weight_load_reorg`/`copy_input_weight`(256bit 정렬 버스트) |
| **자료형/상수** | `hls/core/types.hpp`, `params.hpp`(생성물) | IO/Acc dtype, Tm/Tn/Tr/Tc/OnChipIB |
| **모델 래퍼(SW↔HW)** | `hls/models/yolov2/yolo2_model.cpp` | `yolov2_hls_ps` 호스트 레이어 스케줄러 + `load_weights`/`generate_iofm_offset`/`reorg_cpu` |
| **모델 상수** | `hls/models/yolov2/model_config.cpp` | 23-conv 가중치/bias 오프셋, mem_len, route/conv 길이 |
| **프래그마 헬퍼** | `include/models/yolov2/yolov2_acc_pragmas.h` | `HLS_PRAGMA`/`DO_PRAGMA` 매크로(호스트 빌드 시 경고 억제) |
| **파라미터 생성(SW)** | `scripts/hw_params_gen.py` | 타일 파라미터 → `params.hpp` + linux config 동기화 |
| **호스트 검증 진입** | `src/models/yolov2/yolov2_main.cpp`, `yolov2_weight_gen.cpp` | CPU 추론/가중치 변환 드라이버 |
| **KV260 런타임(C)** | `linux_app/src/*`, `linux_app/include/yolo2_config.h` | mmap 레지스터/udmabuf/V4L2/MJPEG/후처리 |
| **빌드/배포** | `vitis/*.tcl`, `vivado/*.tcl`, `scripts/run_pipeline.py` | HLS 합성·BD·KV260 패키징 |

### 1.2 제외 목록(이름만 언급)
- **third-party**: `include/third_party/{stb_image.h,stb_image_write.h}`, `linux_app/include/third_party/stb_image*`(이미지 IO 라이브러리), `src/stb_image_implementation.cpp`. `nn-weight-extractor`(외부 가중치/Q 생성기, `README.md:9`).
- **생성물**: `hls/core/params.hpp`(미동봉 — `hw_params_gen.py`가 생성, 기본값은 스크립트 하드코딩), `weights/*.bin`(가중치/Q 바이너리, 저장소 미동봉), `linux_app/accel_package/yolov2_accel/{*.bit.bin,*.dtbo,*.dtsi}`(비트스트림/디바이스트리), `yolo2_int16/solution1/syn/report/*`(합성 리포트, 미동봉).
- **데이터/설정**: `examples/test_images/*.jpg`, `data/labels/*`, `config/*.cfg`, `config/coco.names`.
- **부분 정독(라인근거 일부만)**: `linux_app/src/yolo2_*.c`(V4L2/MJPEG/postprocess) — `yolo2_config.h` 레지스터 맵만 정독, 본문 로직은 1차 요약 9절 참조.
- **경로 주의**: 과제 지시의 `src/core/yolo_*.cpp`는 **CPU 참조 구현**(`yolo_cfg/image/layers/math/net/post/region/utils.cpp`)으로 존재하나, 실제 HW/SW 핵심 스케줄러는 `hls/models/yolov2/yolo2_model.cpp`에 위치 → 후자를 분석. `hls/api.hpp`는 `yolo2_accel.hpp`를 include하는 얇은 진입점(`api.hpp:1-3` 추정, 1차 요약 2절).

### 1.3 대표 모델 레이어 구성(YOLOv2)
근거: `model_config.cpp:4-10`(23개 conv 오프셋), `yolo2_model.cpp:294-446`(레이어 타입 분기). conv0(3×3,3→32) → maxpool → conv2(3→ ...) → ... → conv13(512→1024) → ... → reorg(passthrough) → route(skip concat) → conv → region(검출 디코딩). LayerType 코드: 0=CONV, 1=MAXPOOL, 2=REORG(`core_scheduler.cpp:33,63,88`), ROUTE/REGION은 호스트 전담(`yolo2_model.cpp:404-441`).

---

## 2. 모듈: 자료형 / 타일 파라미터 — `types.hpp` + `params.hpp`(생성물)

### 2.1 역할 + 상위/하위
- **역할**: 정밀도별 IO/누산 dtype 별칭과 어레이 형상을 결정하는 합성 상수. 모든 HLS 커널의 차원이 여기서 파생.
- **상위**: 모든 `hls/core/*` 및 `hls/models/yolov2/*`가 의존(`core_compute.hpp:5-6` include). **하위**: 없음(원자 정의).

### 2.2 데이터플로우
```
INT16_MODE? ──yes──► IO_Dtype=int16_t, Acc_Dtype=int32_t  (types.hpp:9-10)
            └─no───► IO_Dtype=float,   Acc_Dtype=float    (types.hpp:12-13)
hw_params_gen.py(기본 Tn4/Tm32/Tr13/Tc13/S2/K3) ──► params.hpp
   ├─ OnChipIB_Width=(Tc-1)*S+K=27, OnChipIB_Height=(Tr-1)*S+K=27 (hw_params_gen.py:42-43)
   └─ TRow_max=OnChipIB_Height, TCol_max=OnChipIB_Width (:44-45)
```

### 2.3 대표 코드 위치
`hls/core/types.hpp`(16줄 전체), `scripts/hw_params_gen.py:16-23`(기본값)+`:42-63`(헤더 렌더), `linux_app/include/yolo2_config.h:97-102`(linux 측 동일값).

### 2.4 대표 코드 블록
```cpp
#ifdef INT16_MODE
using IO_Dtype = int16_t;  using Acc_Dtype = int32_t;   // types.hpp:9-10
#else
using IO_Dtype = float;    using Acc_Dtype = float;     // types.hpp:12-13
#endif
```
```python
DEFAULT_TN=4; DEFAULT_TM=32; DEFAULT_TR=13; DEFAULT_TC=13; DEFAULT_S=2; DEFAULT_K=3  # hw_params_gen.py:16-22
onchip_ib_width = (tc-1)*stride + kernel   # = 27 (hw_params_gen.py:42)
onchip_ib_height = (tr-1)*stride + kernel  # = 27 (hw_params_gen.py:43)
```
→ **PE 어레이 규모 = Tm×Tn = 32×4 = 128 MAC.** OnChipIB=27은 Tr·Tc 타일(13×13) + 커널 윈도우(K=3, S=2 최대) 여유.

### 2.5 마이크로아키텍처
- **메모리**: params는 합성 상수(`constexpr`, `hw_params_gen.py:52-62`)라 자원 0. 어레이 차원만 결정.
- **동기화 메커니즘**: `hw_params_gen.py:66-94`의 `_sync_linux_config`가 정규식으로 `yolo2_config.h`의 `#define Tm/Tn/Tr/Tc/OnChipIB_*`를 동시 갱신 → HLS IP와 런타임 헤더 불일치 방지. **이 자동 동기화가 타일 파라미터 변경 시 단일 진실원천(SSOT) 보장.**
- **병목**: Tm/Tn 확대 시 `compute()`의 `complete` 분할(`core_compute.cpp:28-31`)이 라우팅/팬아웃 비용으로 직결(systolic 아님, 4절 참조). 128 MAC은 비교적 소규모 → 고해상도 처리량 제약(추정).

---

## 3. 모듈: PE 어레이 / MAC 연산코어 — `compute()` (`core_compute.cpp:22-173`) **[가장 중요]**

### 3.1 역할 + 상위/하위
- **역할**: 본 가속기의 연산 코어. Tm×Tn=128 MAC을 공간 병렬로 펼쳐 **output-stationary 누산**으로 부분합을 출력버퍼에 적층. INT16 경로는 per-layer Q 도메인 정렬 + 라운딩 시프트 + 포화를 내장.
- **상위**: `intra_pingpong_wrapper`(LayerType==0 conv)에서 호출(`core_scheduler.cpp:52,58`). **하위**: 없음(버퍼 직접 연산). INT16/FP32 두 버전이 `#ifdef INT16_MODE` 분기(`core_compute.cpp:32/121`).

### 3.2 데이터플로우 (output-stationary 누산)
```
enable==false (bias 프리로드 패스):
   local_beta_buffer[tm] = beta_buffer[m+tm]; return   (core_compute.cpp:36-43)

enable==true (연산 패스):
   precompute shift_out = Qa_in+Qw-Qa_out, shift_bias = Qb-Qa_out  (:48-50)
   for i(Ksize) for j(Ksize) for tr(TR_MIN) for tc(TC_MIN):  [PIPELINE II=1]
     for tm(Tm):  [공간 병렬, DEPENDENCE output_buffer inter false]
        base = (i==0&&j==0&&n==0) ? bias>>shift_bias : output_buffer[tm][tr][tc]
        partial_sum = Σ_{tn(Tn)} weight[tm][tn][i][j] * input[tn][...]    (int32 곱, int64 누산)
        scaled = partial_sum >> shift_out (라운딩)
        acc = base + scaled;  clip[-32768,32767];  output_buffer[tm][tr][tc] = acc
```

### 3.3 Function call stack
`core_scheduler.cpp:52/58` `compute(input_buffer0/1, output_buffer, weight_buffer0/1, beta_buffer, n0/n1, ..., n!=0, Qw, Qa_in, Qa_out, Qb)`. enable=`n!=0`로 첫 호출은 bias 프리로드, 이후는 연산(채널 ping-pong과 중첩).

### 3.4 대표 코드 위치
`hls/core/core_compute.cpp`: INT16 `:32-120`, FP32 `:121-172`. 배열분할 `:28-31`, shift 사전계산 `:48-63`, conv 루프 `:65-118`, FP32 `:139-168`.

### 3.5 대표 코드 블록
```cpp
HLS_PRAGMA(HLS ARRAY_PARTITION variable=input_buffer  complete dim=1)  // Tn 채널 펼침
HLS_PRAGMA(HLS ARRAY_PARTITION variable=output_buffer complete dim=1)  // Tm 출력채널 펼침
HLS_PRAGMA(HLS ARRAY_PARTITION variable=weight_buffer complete dim=1)  // Tm
HLS_PRAGMA(HLS ARRAY_PARTITION variable=weight_buffer complete dim=2)  // Tn   (core_compute.cpp:28-31)
```
→ **Tm(32) 출력채널 × Tn(4) 입력채널이 동시에 HW로 펼쳐짐 = 128 MAC 병렬.**

```cpp
const int shift_out  = Qa_in + Qw - Qa_out;   // 누산기를 Qa_out 도메인으로 정렬
const int shift_bias = Qb - Qa_out;           // (core_compute.cpp:48-50)
```
→ **per-layer Q 4-파라미터 도메인 정렬의 핵심 한 줄.** ViT 선형층/QKV에 그대로 차용 가능.

```cpp
const bool use_bias_init = (i==0)&&(j==0)&&first_input_tile;     // 첫 입력타일+첫 커널탭만 bias 초기화
base = use_bias_init ? (bias>>shift_bias) : output_buffer[tm][tr][tc];  // 그 외 누적   (:77,86-97)
for(tn=0;tn<Tn;tn++){
   int32_t weight_val=weight_buffer[tm][tn][i][j]; int32_t input_val=input_buffer[tn][...];
   partial_sum += (int64_t)(weight_val*input_val);              // INT16xINT16→int32, 누산 int64  (:103-106)
}
scaled = out_shift_right ? (partial_sum+out_round)>>out_shift_mag : partial_sum<<out_shift_mag;  // 라운딩 (:108-113)
acc = base + scaled;  if(acc>32767)acc=32767; if(acc<-32768)acc=-32768;  // INT16 포화 (:115-118)
```
→ **output-stationary**: 출력버퍼에 적층 누산. 시프트 크기 30 클램프(`:56,62`)와 라운딩 상수(`+round = 1<<(mag-1)`, `:57,63`)로 비트-정확 고정소수 재양자화.

### 3.6 마이크로아키텍처
- **Stage 분해**: ① bias 프리로드(enable=false, `:36-43`) ② shift 사전계산(`:48-63`) ③ 9탭×Tr×Tc 픽셀 II=1 파이프(`:65-74`) ④ Tm 출력채널 공간병렬(`:82-84`) ⑤ Tn 부분곱(`:100-106`) ⑥ 재양자화+포화 라이트백(`:108-118`).
- **MAC lanes**: 128(Tm32×Tn4). FP32 버전은 `partial_mul[Tm][Tn]` 완전분할(`:133-135`)로 동일 병렬.
- **systolic 여부**: **비-systolic.** 명시적 PE-to-PE 시프트 레지스터 없음. `ARRAY_PARTITION complete` 기반 공간 MAC 트리 + `DEPENDENCE output_buffer inter false`(`:84`)로 II=1 파이프 보장 → **"output-stationary spatial array"로 분류.** Tm/Tn 확대 시 팬아웃 부담(추정).
- **정량(대표 layer 13)**: 9탭×13×13픽셀 = 1,521 파이프 반복/타일, 각 반복 Tm32×Tn4=128 MAC → 타일당 194,688 scalar MAC. 입력채널 타일 129회(`core_scheduler.cpp:45`)×출력채널 타일 32회(`yolo2_accel.cpp:136`) 누적. 누산 정밀도: int64 partial_sum(`:105`)로 512채널×INT16² 누적 오버플로 방지.
- **병목**: (1) Tn=4가 작아 `tn` 곱이 4-입력 가산트리에 그침 → 입력채널 재사용 낮음. (2) shift 분기(우/좌/무, `:88-94,109-113`)가 II에 영향 가능(추정, II=1 명시되었으나 cosim 필요). (3) FP32/INT16 코드 중복(별도 유지보수 부담).

---

## 4. 모듈: 풀링 / reorg / 활성화 / 라이트백 — `core_compute.cpp`(보조 커널)

### 4.1 역할 + 상위/하위
- **역할**: conv 외 레이어 연산. maxpool(`pool_yolo2`), space-to-depth(`reorg_yolo2`), Leaky ReLU(`nonlinear_leaky_row`), 출력 라이트백 더블버퍼(`write_back_output_reorg`).
- **상위**: pool/reorg는 `intra_pingpong_wrapper`(LayerType 1/2, `core_scheduler.cpp:74,84,99,109`). write_back은 `YOLO2_FPGA`(`yolo2_accel.cpp:155,164`). **하위**: leaky/memcpy 헬퍼.

### 4.2 데이터플로우
```
maxpool: tmp[Tn] init=-32768(i==0&&j==0) → max(Input) → 출력기록(i==1&&j==1, 2x2 가정)  (core_compute.cpp:289-301)
reorg  : Output[(ky<<1)+kx][y][x] = Input[0][2y+ky][2x+kx]  (stride2 passthrough, :373-377)
leaky  : INT16 tmp<0 ? tmp/10(≈0.1 정수근사) : tmp ; FP32 tmp<0 ? tmp*0.1f : tmp  (:194-203)
write_back: ping-pong(local_buf0/1) [ nonlinear_leaky_row ‖ ofm_mmcpy_row ] → DDR  (:242-255)
```

### 4.3 Function call stack
`core_scheduler.cpp:74/84` `pool_yolo2`, `:99/109` `reorg_yolo2`. `yolo2_accel.cpp:155/164` `write_back_output_reorg` → 내부 `nonlinear_leaky_row`(`core_compute.cpp:247/252`) ‖ `ofm_mmcpy_row`(`:248/253`).

### 4.4 대표 코드 위치
`hls/core/core_compute.cpp`: `nonlinear_leaky_row` `:175-210`, `ofm_mmcpy_row` `:212-220`, `write_back_output_reorg` `:222-264`, `pool_yolo2` `:266-305`, `reorg_yolo2` `:354-379`. (미사용 잔존: `zero_output` `:307`, `accumulate_conv` `:319`, `apply_bias_nonlinear` `:339` — 모듈식 재구현 흔적, 호출처 없음.)

### 4.5 대표 코드 블록
```cpp
if(i==0&&j==0) tmp[of] = (IO_Dtype)(-32768);          // INT16 max 초기값  (pool_yolo2, :289-291)
if(Input[of][tr*Kstride+i][tc*Kstride+j] > tmp[of]) tmp[of] = Input[...];
if(i==1&&j==1) Output[of][tr][tc] = tmp[of];          // 2x2 윈도우 끝에 기록  (:300-301)
```
```cpp
Output[(ky<<1)+kx][y][x] = Input[0][(y<<1)+ky][(x<<1)+kx];  // reorg space-to-depth  (:373-377)
```
```cpp
if(pp){ nonlinear_leaky_row(local_buf0, output_buffer, ..., t!=TM_MINxTR_MIN);   // 활성화(현재 행)
        ofm_mmcpy_row(Output, local_buf1, ..., t!=0); }                          // memcpy(이전 행)
   else{ nonlinear_leaky_row(local_buf1, ...);  ofm_mmcpy_row(Output, local_buf0, ...); }  // (:245-255)
```
→ **write_back 내부 ping-pong**(3중 중첩의 3번째 레벨): leaky 계산과 DDR memcpy를 행단위로 중첩.

### 4.6 마이크로아키텍처
- **메모리**: `tmp[Tn]` complete 분할(`:275`), write_back `local_buf0/1[Tc]`(=13워드×2). pool은 LayerType=1일 때 input pad_value=-32768(`core_io.cpp:97-103`)로 경계 처리.
- **정량**: write_back 루프 `TM_MIN×TR_MIN + 1`회(`:242`, +1은 파이프 드레인). pool은 Tr×Tc×K×K×Tn 반복(`:277-302`).
- **병목**: (1) leaky INT16이 `tmp/10` 정수 나눗셈(`:195`) → 정확히 0.1 아님(정확도 손실, 추정). (2) reorg가 `Input[0]`만 읽음(단일채널 가정, `:377`) → 호스트 reorg_cpu(`yolo2_model.cpp:373`)와 역할 중복. (3) maxpool 2×2 하드코딩(`i==1&&j==1`, `:300`) → 일반 KxK 미지원.

---

## 5. 모듈: IO 로드 (256bit 정렬 버스트) — `core_io.cpp`

### 5.1 역할 + 상위/하위
- **역할**: IFM/가중치를 DRAM에서 온칩 버퍼로 256bit(8워드) 정렬 버스트 로드, ping-pong으로 mmcpy(DRAM read)와 unpack(버퍼 채움)을 중첩. 경계 패딩 처리.
- **상위**: `copy_input_weight`가 `intra_pingpong_wrapper`에서 호출(`core_scheduler.cpp:50,56,72,82,97,107`). **하위**: `input_load`/`weight_load_reorg`.

### 5.2 데이터플로우
```
copy_input_weight: TN_MIN=MIN(TN, IFM_num-n); n_next[0]=n  (core_io.cpp:209-211)
  ├─ input_load: ping-pong [ ifm_mmcpy_row(256b align) ‖ ifm_copy_lbuf2ibuf(unpack+pad) ]  (:108-137)
  │     8워드 정렬 오프셋 begin_num=offset&0x7 (:24), pad_value=-32768(maxpool, :97-103)
  └─ weight_load_reorg: Woffset 누적, 8워드 슬라이딩, weight_buffer[Tm][Tn][K][K] 채움  (:154-198)
        유효범위 밖(t1>=TM_MIN || t2>=TN_MIN)은 0패딩 (:196-197)
```

### 5.3 Function call stack
`core_scheduler.cpp:50` `copy_input_weight(...)` → `core_io.cpp:212` `input_load` + `:214/217` `weight_load_reorg`(REORG_TEST 무관 동일 함수, `:213-218`).

### 5.4 대표 코드 위치
`hls/core/core_io.cpp`: `ifm_mmcpy_row` `:19-42`, `ifm_copy_lbuf2ibuf` `:44-80`, `input_load` `:82-138`, `weight_load_reorg` `:140-199`, `copy_input_weight` `:201-219`, `copy_local_beta` `:221-232`, `beta_copy` `:234-237`.

### 5.5 대표 코드 블록
```cpp
const int ifm_trans_offset = (ifm_offset >> 3) << 3;   // 8워드(256bit) 정렬
const uint8_t begin_num = ifm_offset & 0x7;            // 정렬 내 시작 오프셋  (core_io.cpp:23-24)
for(uint16_t t=0;t<loop_cnts;t++) memcpy(local_buf[t], input+ifm_trans_offset+t*8, 8*sizeof(IO_Dtype));  // 버스트 (:33-37)
```
→ **256bit AXI 버스트 정렬.** Input_w도 256b 정렬(`IW_align_256b`, `yolo2_accel.cpp:89-91`).

```cpp
bool XEnable=(xoffset>=0)&&(xoffset<Input_w);  bool YEnable=(yoffset>=0)&&(yoffset<Input_h);
input_buffer[t1][t2][t3] = (XEnable&&PEnable) ? buf_256b[bn_local] : pad_value;  // 경계 0/-32768 패딩 (:64-70)
```
```cpp
if(pp){ ifm_mmcpy_row(input, local_buf0, ...);  ifm_copy_lbuf2ibuf(input_buffer, local_buf1, ..., t!=0); }  // ping-pong
   else{ ifm_mmcpy_row(input, local_buf1, ...);  ifm_copy_lbuf2ibuf(input_buffer, local_buf0, ..., t!=0); }  // (:111-128)
```
→ **input_load 내부 ping-pong**: DRAM read와 unpack 중첩.

### 5.6 마이크로아키텍처
- **메모리**: `local_buf0/1[OnChipIB_Width/8+3][8]`(=6×8워드, `:85-86`), weight `local_buf[(Tm*Tn*K*K)/8+3][8]`(=147×8워드, `:144`). `Woffset` static(`:145`)으로 레이어 내 가중치 스트리밍 위치 유지.
- **정량**: input_load 루프 `Tn*TRow+1`회(`:108`). weight loop_cnts = ⌈(TM_MIN×TN_MIN×KxK+begin_num)/8⌉(`:162-164`). 대표 layer 13: weight 타일당 32×4×9=1,152워드 → 144 버스트.
- **병목**: (1) `unpack`이 8워드 경계마다 memcpy 재로드(`:73-78`) → 직렬 의존. (2) weight를 매 채널타일 재로드(레이어 간/타일 간 가중치 캐싱 없음) → 대표 layer 13에서 9.4M워드급 가중치 반복 read(추정). (3) `IW_align_256b`가 Input_w 미정렬 시 8워드 round-up(`yolo2_accel.cpp:89-91`) → 패딩 워드 낭비.

---

## 6. 모듈: 채널 ping-pong 스케줄러 — `intra_pingpong_wrapper()` (`core_scheduler.cpp:14-113`)

### 6.1 역할 + 상위/하위
- **역할**: 입력채널 n방향으로 **다음 타일 로드와 현재 타일 compute를 ping-pong 중첩**(IO-compute 오버랩의 핵심). LayerType으로 conv/pool/reorg 분기.
- **상위**: `YOLO2_FPGA`의 m-루프(`yolo2_accel.cpp:150,159`). **하위**: `copy_input_weight`/`compute`/`pool_yolo2`/`reorg_yolo2`.

### 6.2 데이터플로우 (LayerType==0 conv)
```
for n=0; n<IFM_num+TN; n+=TN:                              (core_scheduler.cpp:45)
   pingpong==1: copy_input_weight(→buf1, weight1, n<IFM_num)  ‖  compute(buf0, ..., weight0, n!=0)  (:50-52)
   pingpong==0: copy_input_weight(→buf0, weight0)             ‖  compute(buf1, ..., weight1)        (:56-58)
```
→ `n<IFM_num`로 마지막 로드 비활성(드레인), `n!=0`로 첫 compute 비활성(워밍업). 가중치도 weight_buffer0/1 더블버퍼(`:21-27`).

### 6.3 Function call stack
`yolo2_accel.cpp:150/159` `intra_pingpong_wrapper(...)` → (LayerType 0) `core_scheduler.cpp:50,56` `copy_input_weight` ‖ `:52,58` `compute`; (LayerType 1) `:72,82` `copy_input_weight` ‖ `:74,84` `pool_yolo2`; (LayerType 2) `:97,107` ‖ `:99,109` `reorg_yolo2`.

### 6.4 대표 코드 위치
`hls/core/core_scheduler.cpp`: weight 더블버퍼 `:21-27`, conv 분기 `:33-62`, pool 분기 `:63-87`, reorg 분기 `:88-112`.

### 6.5 대표 코드 블록
```cpp
for(n=0;n<IFM_num+TN;n+=TN){
   if(pingpong==1){
      copy_input_weight(Input,Weight,...,input_buffer1,weight_buffer1, n1, n<IFM_num,1,...);  // 다음 타일 로드
      compute(input_buffer0,output_buffer,weight_buffer0,beta_buffer, n0,..., n!=0, Qw,Qa_in,Qa_out,Qb); // 현재 compute
      pingpong=0;
   } else { /* buf0 로드 ‖ buf1 compute */ pingpong=1; }  // (core_scheduler.cpp:45-60)
}
```
→ **채널 n방향 ping-pong**(3중 중첩의 1번째 레벨). pool/reorg는 `tmp_x`/`tmp_tx_min` static으로 m 타일 정보를 한 스텝 지연 전달(`:67-69`).

### 6.6 마이크로아키텍처
- **메모리**: `weight_buffer0/1[Tm][Tn][K][K]` static + dim1,2 complete(`:21-27`), input_buffer0/1은 `YOLO2_FPGA`에서 전달(`yolo2_accel.cpp:103-106`).
- **정량**: conv 루프 `(IFM_num+TN)/TN = ⌈512/4⌉+1 = 129`회(대표 layer 13). LOOP_TRIPCOUNT max=2048(`:47`).
- **병목**: conv ping-pong은 입력채널 단위라 Tn=4면 채널타일이 많아짐(layer 13에서 129회) → 가중치/입력 로드 반복. pool/reorg는 `pingpongx`를 외부(`YOLO2_FPGA`)에서 받아 m-방향 ping-pong과 연동(2단 핑퐁 결합).

---

## 7. 모듈: HLS 톱 함수 타일 오케스트레이션 — `YOLO2_FPGA()` (`yolo2_accel.cpp:25-171`)

### 7.1 역할 + 상위/하위
- **역할**: AXI 인터페이스 정의 + 3중 타일 루프(r/c/m) + m-방향 ping-pong(compute↔write_back). 단일 IP가 conv/pool/reorg 모두 처리.
- **상위**: 호스트 `yolov2_hls_ps`가 레이어마다 호출(`yolo2_model.cpp:322,353`). **하위**: `intra_pingpong_wrapper` + `write_back_output_reorg`.

### 7.2 데이터플로우
```
m_axi 4버스(Input depth=6.92M / Output 5.54M / Weight 50.9M / Beta 10761), burst 64/128, outstanding 4  (:43-46)
s_axilite CTRL_BUS: IFM_num/OFM_num/Ksize/.../LayerType + 포인터 (:48-73)
if LayerType==0: memcpy(beta_buffer, Beta, OFM_num)  (:124-125)
for r(Output_h step TR):  for c(Output_w step TC):  pingpongm=0; for m(OFM_num_bound step TM):  (:127-168)
   input_flag/process_flag/write_flag = f(LayerType, m, mLoopsxTM, ...)  (:144-146)
   pingpongm==0: intra_pingpong_wrapper(→output_buffer1, ..., m1, pingpongm, input_flag, process_flag)
                 write_back_output_reorg(output_buffer, ..., m0[0], write_flag)  (:150-155)
   pingpongm==1: intra_pingpong_wrapper(→output_buffer, ...)  write_back(output_buffer1, ...)  (:159-164)
```

### 7.3 Function call stack
`yolo2_model.cpp:322` `YOLO2_FPGA(in_ptr[i], out_ptr[i], Weight+woff, Beta+boff, ..., Qw,Qa_in,Qa_out,Qb)` → `yolo2_accel.cpp:150/159` `intra_pingpong_wrapper` + `:155/164` `write_back_output_reorg`.

### 7.4 대표 코드 위치
`hls/models/yolov2/yolo2_accel.cpp`: m_axi `:43-46`, s_axilite `:48-73`, assert(경계) `:75-87`, 256b 정렬 `:89-94`, 온칩 버퍼 `:103-113`, 3중 루프 `:127-170`, flag 계산 `:144-146`.

### 7.5 대표 코드 블록
```cpp
HLS_PRAGMA(HLS INTERFACE m_axi depth=50941792 port=Weight bundle=DATA_BUS1 num_read_outstanding=4 max_read_burst_length=128)  // :45
HLS_PRAGMA(HLS INTERFACE s_axilite register port=LayerType bundle=CTRL_BUS)  // :68
```
```cpp
static IO_Dtype output_buffer[Tm][Tr][Tc];
HLS_PRAGMA(HLS BIND_STORAGE variable=output_buffer type=RAM_S2P impl=LUTRAM)  // M-방향 ping-pong (:107-109)
static IO_Dtype output_buffer1[Tm][Tr][Tc];  // BIND_STORAGE LUTRAM (:110-112)
```
```cpp
bool input_flag   = LayerType ? MnemLps&&MneMLps_a1 : MnemLps;   // 파이프 워밍업/드레인 제어
bool process_flag = LayerType ? Mne0&&MneMLps_a1   : MnemLps;
bool write_flag   = LayerType ? Mne0&&Mne1         : Mne0;       // (:144-146)
```
→ **3개 flag로 m-방향 파이프라인 워밍업·정상·드레인 단계 분리.** conv(LayerType=0)와 pool/reorg(≠0)의 타이밍 오프셋이 다름.

### 7.6 마이크로아키텍처
- **메모리**: input_buffer0/1(46,656bit×2), output_buffer/1(LUTRAM 86,528bit×2), weight는 wrapper 내부(6절), beta_buffer 16,384bit. **output을 LUTRAM에 바인딩**(`:109,112`)해 BRAM 절약 + 분할 접근 용이.
- **정량(대표 layer 13)**: r-루프 1회, c-루프 1회, m-루프 ⌈OFM_num_bound/TM⌉ = `(mLoops+1)*TM/TM` = mLoops+1 = 33회(mLoops=⌈1024/32⌉=32, `yolo2_model.cpp:309`). layer 0: r 32회 × c 32회 × m ⌈32/32⌉+1.
- **병목**: (1) 레이어마다 IP 재호출 → 레이어 간 DRAM 왕복(스트리밍 대비 손해). (2) Weight m_axi depth 50.9M(`:45`) → 큰 outstanding 필요. (3) Q값이 CTRL_BUS에 없고 함수인자로만 전달(`:30`) → KV260에선 AXI GPIO 별도 경로(9절).

---

## 8. 모듈: 호스트 레이어 스케줄러 — `yolov2_hls_ps()` (`yolo2_model.cpp:229-449`) **[HW/SW 핵심]**

### 8.1 역할 + 상위/하위
- **역할**: per-layer로 `YOLO2_FPGA`를 반복 호출해 전체 YOLOv2 구성(레이어 단위 오프로딩). 메모리 더블버퍼 배치, per-layer Q 인덱싱, 스킵연결 스케일 정렬, 최종 dequant+디코딩.
- **상위**: `yolov2_main.cpp`(CPU 드라이버, 추정). **하위**: `YOLO2_FPGA`, `load_weights`, `generate_iofm_offset`, `reorg_cpu`, `forward_region_layer`.

### 8.2 데이터플로우
```
load_weights(net, precision): INT16 시 per-layer wpad/bpad 처리 + weight_q/bias_q/iofm_Q 로드  (:158-227)
generate_iofm_offset: Memory_top/bottom 사이 짝/홀 레이어 핑퐁, route16 영역 별도  (:56-110)
입력 양자화: input*2^Qa_in, INT16 클립  (:261-272)
for i in net->n:  switch(l.type):
   CONVOLUTIONAL: TR/TC/TM/TN 클램프(OnChipIB·Tr/Tc·l.n/l.c) → Q 인덱싱 → YOLO2_FPGA → woffset/boffset 누적  (:299-339)
   MAXPOOL: YOLO2_FPGA(LayerType=1, Q=0)  (:341-357)
   REORG: reorg_cpu(host) + INT16 스킵 스케일 정렬(route24_q vs current_Qa, shift)  (:358-403)
   ROUTE: skip(no-op, 주소 핑퐁이 배치를 대신)  (:404-405)
   REGION: dequant(*2^-q_out) → forward_region_layer  (:406-441)
```

### 8.3 Function call stack
`yolo2_main.cpp` → `yolov2_hls_ps`(`:229`) → `load_weights`(`:158`)+`generate_iofm_offset`(`:56`) → 레이어 루프(`:294`) → `YOLO2_FPGA`(`:322,353`)/`reorg_cpu`(`:373`)/`forward_region_layer`(`:440`).

### 8.4 대표 코드 위치
`hls/models/yolov2/yolo2_model.cpp`: `generate_iofm_offset` `:56-110`, `reorg_cpu` `:112-129`, `load_weights` `:158-227`, conv 분기 `:299-340`, maxpool `:341-357`, reorg+스케일정렬 `:358-403`, region `:406-442`.

### 8.5 대표 코드 블록
```cpp
TR = std::min(((OnChipIB_Height-l.size)/l.stride+1), Tr);  TR = std::min(output_h, TR);   // 타일 클램프
TM = std::min(l.n, Tm);  TN = std::min(l.c, Tn);  mLoops = ceil((float)l.n/TM);            // (:303-309)
Qa_in  = act_q[offset_index];   Qa_out = act_q[offset_index+1];                            // per-layer 활성 Q
Qw     = weight_q[offset_index];  Qb = bias_q[offset_index];                               // per-layer 가중치/bias Q (:314-317)
if(pending_route_q>=0) Qa_in = pending_route_q;   // route 후 입력 Q를 정렬된 값으로 강제 (:318-320)
YOLO2_FPGA(in_ptr[i], out_ptr[i], Weight_buf+woffset, Beta_buf+boffset, l.c,l.n,l.size,..., Qw,Qa_in,Qa_out,Qb);  // (:322-326)
```
```cpp
// REORG: 스킵연결(route24) 스케일을 reorg 갈래와 일치시킴
const int target_q = std::min(route24_q, current_Qa);  const int shift = current_Qa - target_q;
v = (shift>0) ? (v>>shift) : (v<<-shift);  clip[-32768,32767];  current_Qa=target_q;  pending_route_q=current_Qa;  // (:381-398)
```
→ **양자화 도메인 일치(스킵연결 스케일 정렬)** — ViT residual add 전 스케일 정렬에 직결.

```cpp
in_ptr[x]  = (x%2==0) ? Memory_top : out_ptr[x-1];                       // 짝/홀 레이어 핑퐁
out_ptr[x] = (x%2==0) ? Memory_bottom - (out_c*out_h*out_w_align) : Memory_top;  // (:67-77)
```
→ **메모리 더블버퍼(Memory_top/bottom 핑퐁)**로 레이어 간 in/out 버퍼 교대 — DRAM 스크래치 재사용.

### 8.6 마이크로아키텍처
- **메모리**: `Memory_buf = calloc(mem_len+512*2)`(`:244`, mem_len=6,922,240, `model_config.cpp:19`), 32-엔트리 in_ptr/out_ptr 테이블(`:249-250`), region_buf/region_buf2(13×16×425, `:280-282`).
- **정량**: 레이어 23 conv + pool/reorg/route/region. weight/bias offset은 per-layer `cfg.weight_offsets[]`/`beta_offsets[]` 누적(`:328-329`). INT16 가중치 odd-count 패딩 처리(`:217-223`).
- **병목**: (1) reorg를 호스트 CPU(`reorg_cpu`, `:373`)로 처리 → PS 부하 + PL 유휴. (2) 레이어마다 YOLO2_FPGA 동기 호출(시뮬 shim)이라 레이어 간 직렬. (3) region/route가 호스트 전담 → 저지연엔 PS 왕복 비용.

---

## 9. 모듈: 모델 상수 + KV260 런타임 매핑 — `model_config.cpp` + `linux_app/include/yolo2_config.h`

### 9.1 역할 + 상위/하위
- **역할(model_config)**: 23-conv 가중치/bias 오프셋 테이블, mem_len, route16/conv24/conv27 길이, detection_workspace. **역할(yolo2_config)**: KV260 제어 레지스터 맵 + Q값 AXI GPIO 경로 + 메모리/타일 상수.
- **상위**: `yolo2_model.cpp`(model_config), `linux_app/src/*`(yolo2_config). **하위**: 없음(상수 정의).

### 9.2 데이터플로우
```
model_config: weight_offsets[23] + beta_offsets[23] → woffset/boffset 누적 (yolo2_model.cpp:328-329)
              mem_len=6,922,240 → generate_iofm_offset 배치 (:59)
yolo2_config: YOLO2_CTRL_BASE 0xA0000000 → mmap → AP_CTRL/주소레지스터/파라미터레지스터
              Q값은 별도 AXI GPIO: QW 0xA0010000 / QA_IN 0xA0020000 / QA_OUT 0xA0030000 / QB 0xA0040000
```

### 9.3 대표 코드 위치
`hls/models/yolov2/model_config.cpp`: 가중치 오프셋 `:4-7`, bias 오프셋 `:9-10`, ModelConfig `:18-26`. `linux_app/include/yolo2_config.h`: 베이스주소 `:18-22`, CTRL 레지스터 `:38-75`, Q via GPIO 주석 `:77-78`, 메모리 `:84-91`, 타일 `:97-102`.

### 9.4 대표 코드 블록
```cpp
constexpr std::array<int,32> kYolo2WeightOffsets = {864, 18432, 73728, 8192, 73728, 294912, ...
   4718592, 524288, 4718592, 524288, 4718592, 9437184, 9437184, 32768, 11796480, 435200, 0...};  // model_config.cpp:4-7
static const ModelConfig cfg{ /*mem_len=*/6922240, /*route16_len=*/26*32*512, /*conv27_len=*/13*16*256,
   /*conv24_len=*/13*16*1024, /*detection_workspace=*/3*13*425, kYolo2WeightOffsets, kYolo2BetaOffsets };  // :18-26
```
→ 오프셋[13]=4718592=1024×512×9(대표 layer 13), [0]=864=32×3×9(첫 레이어). mem_len=416²×32+208²×32.

```c
#define YOLO2_CTRL_BASE     0xA0000000UL   // 제어 레지스터
#define AXI_GPIO_QW_BASE    0xA0010000UL   // 가중치 Q (CTRL_BUS에 Q 레지스터 없음, :77-78)
#define CTRL_LAYER_TYPE_OFFSET 0xd0        // 마지막 파라미터 레지스터
#define WEIGHTS_SIZE_BYTES  (50941792 * 2) // ~97MB (INT16)   (yolo2_config.h:18-22,75,90)
```

### 9.5 마이크로아키텍처
- **메모리**: 가중치 ~97MB(INT16, `yolo2_config.h:90`), mem_len 6.92M워드(~13.8MB INT16). detection_workspace 3×13×425.
- **정량/병목**: (1) **Q값이 CTRL_BUS에 없음**(HLS IP 제약) → 4개 AXI GPIO(0xA0010000~0xA0040000)로 별도 전달(`:19-22,77-78`). per-layer Q 변경마다 GPIO 쓰기. (2) 가중치 97MB → DRAM 대역폭/용량 압박(INT8/INT4 대비 큼). (3) model_config 테이블이 linux 측(`linux_app/src/yolo2_inference.c`)에 복제 → SSOT 위반 위험.

---

## 10. 모듈: 파라미터 생성 + 빌드/배포 — `hw_params_gen.py` + `vitis/*.tcl` + `run_pipeline.py`

### 10.1 역할 + 상위/하위
- **역할**: 타일 파라미터 → `params.hpp` 생성 + linux config 동기화, HLS 합성/cosim TCL, 자동화 파이프라인.
- **상위**: 사용자 CLI / Makefile. **하위**: Vitis HLS, Vivado.

### 10.2 대표 코드 위치
`scripts/hw_params_gen.py`(178줄), `vitis/{yolo2_cli.tcl,yolo2_int16_cli.tcl,run_cosim.tcl,yolo2_cosim_tb.cpp}`, `vivado/{build_from_bd.tcl,bd/kv260_yolov2_int16_bd.tcl}`, `scripts/run_pipeline.py`, `scripts/yolo2_report.py`, `pipeline.yaml`, `yolo2_report.json`.

### 10.3 대표 코드 블록
```python
output_path.write_text(_render_header(stride, kernel, max_beta_length, tn, tm, tr, tc))  # params.hpp 생성 (:153)
_sync_linux_config(config_path, tm, tn, tr, tc, onchip_ib_width, onchip_ib_height)        # linux 동기화 (:162-170)
```
→ `python3 scripts/hw_params_gen.py [--tn --tm --tr --tc --stride --kernel]`로 타일 파라미터 변경 → HLS + 런타임 헤더 동시 갱신.

### 10.4 마이크로아키텍처(빌드 관점)
- **흐름**: `make test`(params 자동생성)+`./yolov2_detect --backend hls`(FP32 검증) / `make test-int16`(INT16) → `vitis-run --tcl vitis/yolo2_int16_cli.tcl`(합성) → `vivado/build_from_bd.sh`(BD) → `create_accel_package.sh`/`deploy_to_kv260.sh`(배포)(1차 요약 6절).
- **리포트**: `yolo2_report.py`+`yolo2_report.json`이 HLS/Vivado/KV260 리포트 경로 지정(`yolo2_report.json:2-4`)하나 **실제 리포트 미동봉** → PPA 확인 불가. KV260 측정은 `enabled:false`(`:6`).
- **병목 없음**(빌드타임 스크립트). 절대경로 하드코딩 주의(추정, 미정독).

---

## 11. 모듈 한눈 요약 표

| 모듈 | 파일 | 핵심 함수(라인) | 역할 | 대표 정량(layer 13) |
|---|---|---|---|---|
| 자료형/파라미터 | types.hpp, params.hpp(gen) | (types 전체), hw_params_gen(:42-63) | dtype + Tm/Tn/Tr/Tc/OnChipIB | Tm32×Tn4=128 MAC |
| PE 어레이/MAC | core_compute.cpp | compute(:22-173) | output-stationary 128 MAC + INT16 Q정렬 | 9.97G scalar MAC |
| 양자화 정렬 | core_compute.cpp | compute shift(:48-50,108-118) | shift_out=Qa_in+Qw-Qa_out, 라운딩/포화 | int64 누산 |
| pool/reorg/leaky | core_compute.cpp | pool_yolo2(:266), reorg_yolo2(:354), leaky(:175) | maxpool/space-to-depth/LeakyReLU | 2×2 s2 |
| 라이트백 | core_compute.cpp | write_back_output_reorg(:222) | leaky‖memcpy 더블버퍼 | local_buf[Tc]×2 |
| IO 로드 | core_io.cpp | input_load(:82), weight_load_reorg(:140) | 256bit 버스트 + ping-pong + 패딩 | weight 144 burst/tile |
| 채널 스케줄러 | core_scheduler.cpp | intra_pingpong_wrapper(:14) | n방향 load‖compute ping-pong | 129 채널타일 |
| HLS top | yolo2_accel.cpp | YOLO2_FPGA(:25-171) | AXI + 3중 타일 + m-pingpong | m-루프 33회 |
| 호스트 스케줄러 | yolo2_model.cpp | yolov2_hls_ps(:229-449) | per-layer 오프로딩 + Q인덱싱 + skip정렬 | 23 conv 레이어 |
| 메모리 배치 | yolo2_model.cpp | generate_iofm_offset(:56) | Memory_top/bottom 핑퐁 | mem_len 6.92M |
| 모델 상수 | model_config.cpp | yolo2_model_config(:13) | 23-conv 오프셋 + mem_len | woff[13]=4718592 |
| KV260 런타임 | yolo2_config.h | (매크로) | CTRL 0xA0000000 + Q via GPIO | 가중치 ~97MB |
| 파라미터 생성 | hw_params_gen.py | main(:131), _sync(:66) | params.hpp + linux 동기화 | 빌드타임 |

---

## 12. 읽기 순서 / 코드 추적 순서

1. **자료형/파라미터 먼저**: `types.hpp` + `hw_params_gen.py:16-63` → Tm/Tn/Tr/Tc/OnChipIB와 dtype 직관. **128 MAC = Tm32×Tn4** 확인.
2. **연산 코어**: `core_compute.cpp` `compute()`(`:22-173`) → 배열분할(`:28-31`, 128 MAC 펼침) → shift 사전계산(`:48-50`, Q 도메인) → output-stationary 누산(`:77,86-118`)이 본 가속기의 본질.
3. **IO**: `core_io.cpp` `input_load`(`:82`, 256b 버스트+ping-pong) + `weight_load_reorg`(`:140`) → DRAM↔온칩 변환.
4. **채널 스케줄러**: `core_scheduler.cpp` `intra_pingpong_wrapper`(`:14`) → n방향 load‖compute 중첩(IO 은폐 핵심).
5. **HLS top**: `yolo2_accel.cpp` `YOLO2_FPGA`(`:25`) → AXI 인터페이스(`:43-73`) + 3중 타일 루프(`:127-168`) + m-pingpong(`:148-166`) + flag 제어(`:144-146`).
6. **라이트백/보조**: `core_compute.cpp` `write_back_output_reorg`(`:222`, leaky‖memcpy) + `pool_yolo2`(`:266`)/`reorg_yolo2`(`:354`).
7. **호스트 스케줄러(HW/SW 경계)**: `yolo2_model.cpp` `yolov2_hls_ps`(`:229`) → 레이어 루프(`:294`) + per-layer Q 인덱싱(`:312-321`) + 스킵 스케일 정렬(`:379-399`) + 메모리 핑퐁(`generate_iofm_offset`, `:56`).
8. **상수/런타임**: `model_config.cpp`(오프셋 테이블) → `yolo2_config.h`(CTRL/Q-GPIO 맵).
9. **빌드 확인**: `hw_params_gen.py`(params 생성) → `vitis/yolo2_int16_cli.tcl`(합성).

---

## 13. 병목 후보 & 병렬도/노브

### 13.1 병목 후보
1. **PE 어레이 128 MAC(소규모)**(`core_compute.cpp:28-31`, Tn=4): Tn이 작아 입력채널 재사용 낮고 채널타일 多(layer 13에서 129회). 고해상도/대모델 처리량 제약.
2. **레이어 단위 오프로딩 → DRAM 왕복**(`yolo2_model.cpp:322` 레이어마다 YOLO2_FPGA 재호출): 스트리밍 dataflow(예: HG-PIPE 완전 파이프) 대비 대역폭/지연 손해.
3. **가중치 매 타일 재로드**(`core_io.cpp:140-198`, 캐싱 없음): 대표 layer 13에서 9.4M워드급 가중치 반복 read(추정).
4. **reorg를 호스트 CPU로 처리**(`yolo2_model.cpp:373`): PS 부하 + PL 유휴, region/route도 PS 전담(`:404,440`).
5. **비-systolic 공간 어레이**(`core_compute.cpp` `complete` 분할): Tm/Tn 확대 시 라우팅/팬아웃 부담(추정).
6. **INT16 메모리/DSP 비용**(가중치 ~97MB, `yolo2_config.h:90`): INT8/INT4 대비 큼.
7. **Q via AXI GPIO 별도 경로**(`yolo2_config.h:77-78`): per-layer Q 변경마다 GPIO 쓰기(CTRL_BUS에 Q 레지스터 없음).
8. **leaky INT16 `tmp/10` 정수근사**(`core_compute.cpp:195`): 정확히 0.1 아님 → 정확도 손실(추정).
9. **params.hpp 미동봉**: 빌드 전 정적 상수 부재(생성 필요).

### 13.2 병렬도/노브
- **타일 파라미터 Tm/Tn/Tr/Tc/S/K**(`hw_params_gen.py:101-106`): `--tm --tn --tr --tc`로 PE 규모·온칩버퍼 조정 → params.hpp + linux config 동시 갱신(`:162-170`). Tm32×Tn4가 128 MAC, Tm/Tn↑하면 MAC↑(라우팅 비용 trade-off).
- **per-layer Q 4-파라미터**(`yolo2_model.cpp:312-317`): `Qw/Qa_in/Qa_out/Qb`를 레이어별 테이블에서 인덱싱 → `shift_out/shift_bias` 자동(`core_compute.cpp:48-50`). 스킵연결은 `min(route24_q, current_Qa)`로 정렬(`:381`).
- **AXI 버스트/outstanding**(`yolo2_accel.cpp:43-46`): Weight burst=128/outstanding=4, IO burst=64. 대역폭 노브.
- **3중 ping-pong 깊이**: (1) 채널 n방향(`core_scheduler.cpp:45-60`), (2) m방향(`yolo2_accel.cpp:148-166`), (3) write_back 내부(`core_compute.cpp:242-255`). IO 지연 은폐의 3중 노브.
- **output_buffer LUTRAM 바인딩**(`yolo2_accel.cpp:109,112`): BRAM↔LUTRAM 선택으로 자원 균형.
- **INT16/FP32 스위치**(`types.hpp:8`, `INT16_MODE`): 정밀도↔비용 trade-off.

---

*근거 파일(절대경로)*:
`\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\CNN-Accel\yolo-fpga-accelerator-main\hls\core\{types.hpp,core_compute.cpp,core_compute.hpp,core_io.cpp,core_scheduler.cpp}`,
`...\hls\models\yolov2\{yolo2_accel.cpp,yolo2_accel.hpp,yolo2_model.cpp,model_config.cpp,model_config.hpp}`,
`...\include\models\yolov2\yolov2_acc_pragmas.h`,
`...\linux_app\include\yolo2_config.h`,
`...\scripts\hw_params_gen.py`,
`...\yolo2_report.json`.
