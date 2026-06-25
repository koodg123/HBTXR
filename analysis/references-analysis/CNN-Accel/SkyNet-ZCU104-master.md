# SkyNet-ZCU104-master 정밀 분석

분석 대상 경로: `REF/CNN-Accel/SkyNet-ZCU104-master/`
분석 기준: 자체 핵심 HLS C 커널 + 가중치 재배치/양자화 + PYNQ 호스트 드라이버 + GPU 학습측 모델. 모든 라인 근거는 `파일:라인` 형식.

---

## 1. 개요

- **목적**: DAC-SDC(System Design Contest) 계열 저전력 객체 검출용 경량 DNN인 **SkyNet**을 Xilinx **ZCU104** 보드(MPSoC, PL 부분 `xczu7ev`)로 포팅한 FPGA 가속기. 원본은 Ultra96 구현이며, 본 저장소는 ZCU104 이식 + **DVFS(동적 전압/주파수 스케일링)** 지원을 추가한 변형판이다 (`README.md:2-3`).
- **원논문**: SkyNet, *"A Champion Design for DAC-SDC on Low Power Object Detection"* (Zhang et al., arXiv:1906.10327) (`README.md:115-122`). 설계 동기/방법론은 ICML'19 워크숍 논문(arXiv:1905.08369)과 DAC'19 FPGA/DNN Co-Design 논문(arXiv:1904.04421)에 기술 (`README.md:124-141`).
- **타깃 보드**: ZCU104, PL 디바이스 `xczu7ev-ffvc1156-2-e` (`FPGA/HLS/script.tcl:34`), HLS 클록 주기 3 ns(=333 MHz 목표) (`script.tcl:35`). 호스트 데모는 기본 주파수 330 MHz / 전압 750 mV로 실행 (`README.md:24`).
- **태스크**: 단일 객체 검출(1 bounding box / 이미지). 입력 320x160(학습) / 640x360(원본 좌표 복원). 출력은 confidence가 최대인 한 개 박스의 (x, y, w, h, conf). **배치 4** 이미지를 644x324 캔버스에 stitch하여 한 번에 추론 (`FPGA/Deploy/SkyNet.py:88-118`, `README.md:14-15`).
- **백본**: depthwise-separable conv(MobileNet 스타일) 6 블록 + ReorgLayer skip-concat + 최종 pointwise(1x1) detection head. PyTorch 정의에서 확인 (`GPU/models.py:34-86`).

---

## 2. 디렉토리 구조 (자체 코드 vs. 제외)

```
SkyNet-ZCU104-master/
├── README.md                         # 전체 빌드/배포 가이드 (분석함)
├── LICENSE
├── FPGA/
│   ├── HLS/                          # ★ 자체 핵심 HLS C 소스
│   │   ├── net_hls.h                 # 타입 정의(ap_fixed) + 함수 프로토타입 (분석함)
│   │   ├── net_hls.cc                # ★ 최상위 SkyNet() 데이터플로우 오케스트레이션 (분석함, 1224행)
│   │   ├── conv1x1.cc                # ★ pointwise 1x1 PE + compute_engine_16 MAC 트리 (분석함)
│   │   ├── dwconv3x3.cc              # ★ depthwise 3x3 커널 + relu6 (분석함)
│   │   ├── reorder_weight.cc         # ★ float→fixed 양자화 + 가중치 재배치 + .bin 생성 (분석함, 668행)
│   │   ├── golden_c.cc               # CSIM 검증용 float reference (테스트벤치, 미정밀분석)
│   │   ├── output_verify.cc          # golden 대비 비교 (테스트벤치)
│   │   ├── tb.cc                     # CSIM 테스트벤치 + 전역 weight 버퍼 선언 (부분분석)
│   │   ├── script.tcl                # Vivado HLS 빌드 스크립트 (분석함)
│   │   ├── test_image_bins/*.bin     # [제외] 테스트 입력 생성물(.bin)
│   │   ├── weights_fixed.bin         # [제외] 양자화 가중치 생성물(.bin)
│   │   └── weights_floating.bin      # [제외] float 가중치 입력(.bin)
│   ├── RTL/script.tcl                # Vivado 시스템 통합(bitstream 생성) 스크립트
│   └── Deploy/                       # ★ PYNQ 호스트 배포
│       ├── SkyNet.py                 # ★ ARM 코어 호스트 드라이버(추론 + DVFS) (분석함)
│       ├── SkyNet.bit/.hwh/.bin      # [제외] bitstream/HW핸드오프/가중치 생성물
│       ├── dfs / dvs                 # [제외] DVFS 제어 바이너리(컴파일됨)
│       └── test_images/*.jpg         # [제외] 데모 입력 이미지
└── GPU/                              # 학습측 (PyTorch, darknet 변형)
    ├── models.py                     # ★ SkyNet PyTorch 모델 정의 (분석함)
    ├── region_loss.py / dataset.py / utils.py  # darknet/YOLOv2 유래 (외부포크성, 미정밀분석)
    ├── run.py / demo.py / prepare.py / *.sh
    ├── dac.weights                   # [제외] darknet 가중치(.weights)
    └── samples/*.jpg                 # [제외] 샘플 이미지
```

**제외 사유**: `*.bin / *.bit / *.hwh / *.weights / *.jpg` 는 이름만으로 제외(빌드/학습 생성물 또는 데이터). `GPU/`의 `region_loss.py`, `dataset.py`, `utils.py`는 YOLOv2/darknet 학습 포크 성격이 강해 본 분석의 핵심(HLS 가속기)에서 제외하고 모델 정의(`models.py`)만 정밀 분석함.

---

## 3. 핵심 모듈 정밀 분석

### 3.0 자료형 / 양자화 비트폭 (net_hls.h)

가속기는 16-bit 컨테이너 안에 다양한 비트폭의 `ap_fixed`를 매핑한다 (`net_hls.h:45-67`):

| 용도 | 타입 | 정의 | 정수.소수 비트 | 라인 |
|------|------|------|------|------|
| Feature map | `FIX_FM` | `ap_fixed<9,3,AP_RND,AP_SAT>` | 9비트(정수3) | `net_hls.h:45` |
| 누적(acc) | `FIX_FM_acc` | `ap_fixed<12,4,AP_RND,AP_SAT>` | 12비트(정수4) | `net_hls.h:46` |
| 가중치 | `FIX_WT` | `ap_fixed<11,4,AP_RND,AP_SAT>` | 11비트(정수4) | `net_hls.h:47` |
| MAC 곱셈중간 | `FIX_32_10` | `ap_fixed<32,10>` | 32비트 | `net_hls.h:57`, 사용처 `conv1x1.cc:28-31` |

- **양자화 모드**: PTQ(Post-Training Quantization) 성격. 학습은 float로 수행되고, `reorder_weight.cc`에서 `(FIX_WT)` 캐스팅으로 float→fixed 변환 (예 `reorder_weight.cc:176,217,249`). 활성화는 `AP_RND`(반올림) + `AP_SAT`(포화) 적용. **scale/zero-point 기반 정수 양자화가 아닌 고정소수점(fixed-point) 방식**으로, dynamic range는 비트폭에 의해 정적으로 결정됨.
- **비트 패킹 상수**: `FM_RG=8`(FM 9비트 중 상위인덱스), `FM_ACC_RG=11`, `WT_RG=10` (`net_hls.h:17-19`). `.range(WT_RG,0)` = 11-bit 가중치 추출, `.range(FM_RG,0)` = 9-bit FM 추출. 단, 16-bit 슬롯에 패킹할 때 부호확장 없이 하위 비트만 복사하므로 음수 표현은 ap_fixed 내부 표현에 의존(확인: `reorder_weight.cc:581` 가중치는 `.range(10,0)` 11비트만 복사).
- **AXI 데이터 폭**: 모든 FM/가중치는 **512-bit 버스**로 전송. 512/16 = **32채널을 1클럭에 병렬 처리**하는 것이 본 설계의 기본 데이터 단위 (`net_hls.h:66-67`, `net_hls.cc:283-350`).

### 3.1 Pointwise 1x1 Conv — `CONV_1x1` + `compute_engine_16` (conv1x1.cc)

**MAC 트리 — `compute_engine_16` (`conv1x1.cc:11-71`)**
- 16쌍 `(weight, activation)`을 입력받아 16개 병렬 곱셈(`conv1x1.cc:33-48`) 후 **4단 가산 트리**(16→8→4→2→1, `conv1x1.cc:50-67`)로 누적. 즉 **16-input dot-product 하드 PE**. 곱셈 결과는 `FIX_32_10`으로 폭 확장하여 누적 오차 억제 (`conv1x1.cc:28-29`).

**메인 루프 `CONV_1x1` (`conv1x1.cc:91-142`)**
- 시그니처: `bottom[32][44][84]`, `top[32][44][84]`(acc 타입), `weights[32][32]` (`conv1x1.cc:91-93`).
- 입력 채널 32개를 **16개씩 2회**(`ci += 16`, `conv1x1.cc:106`) 분할 → `compute_engine_16`이 16-input이므로 32-입력 누적을 2 패스로 처리.
- `load_weights`로 32x16 가중치 타일을 로컬 버퍼에 복사(`conv1x1.cc:76-86, 109`).
- **병렬도(parallelism)**:
  - `array_partition ... dim=1 complete`: top/bottom/weights의 채널 차원(32)을 완전 분할(레지스터화) → 32채널 동시 접근 (`conv1x1.cc:96-99`).
  - 출력 채널 루프 `coo`(0..31)에 `#pragma HLS unroll` → **32개 compute_engine_16 인스턴스 병렬** (`conv1x1.cc:116-117`).
  - 공간 루프 `h,w`에 `#pragma HLS pipeline II=2` (`conv1x1.cc:115`).
- **MAC 병렬도 추정**: 32(출력ch) x 16(PE당 입력) = **클럭당 512 MAC** (단 II=2이므로 처리량은 2클럭당 512). 2회 ci 패스로 32x32 곱을 완성.

### 3.2 Depthwise 3x3 Conv — `DW_CONV_3x3` (dwconv3x3.cc)

- 시그니처: `bottom[32][44][84]`, `top[32][44][84]`, `weights[32][3][3]`, `bias[32]`, `int relu` (`dwconv3x3.cc:21-25`).
- **depthwise** 특성: 채널 간 합산이 없고 `top[co] += weights[co][i][j] * bottom[co][h+i-1][w+j-1]` 로 채널별 독립 (`dwconv3x3.cc:41`). co가 input=output 채널.
- 루프 구조: 커널 `i,j`(3x3) 바깥, 공간 `h,w`(1..42 / 1..82) 안쪽, 채널 `co`(0..31) 최내곽 (`dwconv3x3.cc:34-42`).
- **병렬도**:
  - `array_partition dim=1 complete` (채널 32 완전 분할, `dwconv3x3.cc:28-31`).
  - `co` 루프 `#pragma HLS unroll` → **32채널 곱셈-누적 병렬** (`dwconv3x3.cc:39-40`).
  - 공간 루프에 `#pragma HLS pipeline` (`dwconv3x3.cc:38`).
  - **MAC 병렬도 추정**: 32(채널) MAC/클럭. 9(3x3) x 42 x 82 회 공간/커널 반복.
- **ReLU6 융합**: `relu` 인자가 1이면 별도 패스에서 `relu_single`(0~6 clamp, `dwconv3x3.cc:12-18`) 적용 (`dwconv3x3.cc:48-58`). h를 2씩 증가시키며 2행 동시 처리(`dwconv3x3.cc:49,53-54`)로 II 개선.
- **참고**: 누적 시작 전 bias 세팅은 `net_hls.cc`의 `set_bias_3x3`가 담당(`net_hls.cc:413-425`)하고, conv는 그 위에 `+=` 누적.

### 3.3 Relu + Max Pooling 융합 — `Relu_Max_Pooling` (net_hls.cc)

- 2x2 stride-2 max pooling을 ReLU와 융합 (`net_hls.cc:582-623`).
- `relu_max(a,b,c,d)`: 4개 입력 각각 relu 후 max (`net_hls.cc:565-577`).
- 출력 좌표는 `layer` 인자에 따라 3개 DDR 목적지로 분기:
  - `layer==1` → `ddr_dw1_pool` (`net_hls.cc:608-610`)
  - `layer==2` → `ddr_dw2_pool` (`net_hls.cc:611-613`)
  - `layer==3` → `ddr_buf`(범용 버퍼) (`net_hls.cc:614-616`)
- 32채널 unroll + 16-bit 패킹으로 512-bit DATA 1워드 생성 (`net_hls.cc:600-605`), `#pragma HLS pipeline II=2` (`net_hls.cc:598`).

### 3.4 가중치 재배치 + 양자화 — `reorder_weight_fix` (reorder_weight.cc)

CSIM 단계에서 한 번 실행되어 `weights_fixed.bin`을 생성하는 **호스트측(오프라인) 전처리**다. 핵심 3단계:

1. **채널 패딩(32 정렬)** (`reorder_weight.cc:171-328`): 하드웨어가 32채널 단위로만 동작하므로, 32의 배수가 아닌 레이어를 0-패딩.
   - dw1: 입력 3ch→32, 출력 48→64 (`reorder_weight.cc:66-70, 174-204`)
   - dw2: 48→64 패딩 (`reorder_weight.cc:72-76, 207-233`)
   - pw7: 출력 10→32 패딩 (`reorder_weight.cc:102, 318-328`)

2. **dw6 채널 재배치(reorder)** (`reorder_weight.cc:119-168`): dw3(192ch→reorg시 768ch)와 dw5(512ch)를 concat한 1280ch을 하드웨어 reorg 패턴에 맞게 재정렬. ch를 4씩 묶어 `ch/4, ch/4+192, +192*2, +192*3` 순으로 인터리브 (`reorder_weight.cc:123-138`). 768 이후 512채널(dw5)은 그대로 (`reorder_weight.cc:139-147`). 이는 `net_hls.cc`의 `load_and_reorg_part`(아래 4.2)가 기대하는 4-way 채널 인터리브와 짝을 이룬다.

3. **32-tile 분할 + 512-bit 패킹**:
   - 모든 1x1 가중치를 `[index][32co][32ci]` 청크로 분할 정렬(예 `reorder_weight.cc:352-364`). 각 레이어는 `(CO/32) x (CI/32)` 청크로 분해.
   - 11-bit 가중치를 16-bit 슬롯에 패킹 후(`reorder_weight.cc:577-606`) 32개를 모아 512-bit 워드로 직렬화 (`reorder_weight.cc:628-664`).
   - 출력 파일: `weights_fixed.bin` — 1x1, 3x3, bias 순서로 기록 (`reorder_weight.cc:625-664`). 호스트가 이 순서대로 다시 읽음(아래 5).

### 3.5 최상위 데이터플로우 — `SkyNet()` (net_hls.cc:773-1223)

- **인터페이스**: 7개 `m_axi` 슬레이브 포트(image, 1x1 wt, 3x3 wt, bias, dw1 pool, dw2 pool, 범용 buf) + 출력(predict_boxes, constant, debug) + `s_axilite` 제어 (`net_hls.cc:789-802`). bundle은 `INPUT`/`OUTPUT` 2개 HP AXI로 묶임(`README.md:91`과 일치).
- **자원 제약(ALLOCATION limit=1)**: `CONV_1x1`, `DW_CONV_3x3`, `Relu_Max_Pooling`, `load_image_chunk_norm` 각각 **인스턴스 1개로 제한**(`net_hls.cc:805-808`) → 면적 절감 위해 커널을 시분할 재사용(layer-by-layer 직렬 실행).
- **온칩 FM 버퍼**: `FM_buf1~4 [32][44][84]` + `FM_buf_acc` (`net_hls.cc:12-16`). 32채널 x (44x84) 타일이 처리 기본 단위. 44x84는 패딩 포함 타일(유효 42x82 + 1픽셀 halo, conv 루프가 1..42/1..82 사용).
- **가중치 온칩 버퍼**: `weight_buf_1x1[4][32][32]`, `weight_buf_3x3[4][32][3][3]`, `bias_buf[4][32]` (`net_hls.cc:19-21`).

**레이어 실행 시퀀스** (모두 `dw_conv(3x3 depthwise) → pointwise(1x1) → (pool)` 패턴):

| 블록 | 입력→출력 ch | 처리 | 라인 |
|------|------|------|------|
| DW1 | 32→64 | 3x3 + 1x1 + pool(layer1). 이미지 norm 로드, 8x8 타일 순회, ping-pong(FM_buf1/3) | `net_hls.cc:814-856` |
| DW2 | 64→96 | dw1_pool DDR 로드 → 3x3 x2 → 1x1 누적 → pool(layer2). 4x4 타일 | `net_hls.cc:859-901` |
| DW3 | 96→192 | dw2_pool 로드 → 3x3 x3 → 1x1 → pool(layer3). **출력을 DDR_buf[100..]에 별도 저장(reorg용)** | `net_hls.cc:904-961` |
| DW4 | 192→384 | DDR_buf[6..] 로드 → 3x3 → 1x1. ping-pong DDR 더블버퍼 | `net_hls.cc:964-1024` |
| DW5 | 384→512 | 3x3 → 1x1. DDR_buf[18..]→[42..] | `net_hls.cc:1027-1085` |
| DW6 | 1280→96 | **concat(dw3 reorg 768 + dw5 512) → 3x3 → 1x1**. 전반부 reorg 로드, 후반부 직접 로드 | `net_hls.cc:1087-1195` |
| PW7 | 96→32(유효10) | 1x1 only detection head. 결과 FM_buf_acc에 누적 | `net_hls.cc:1198-1214` |

- **skip-concat(reorg) 구현**: DW3의 1x1 출력을 `relu_copy_buf_to_DDR_acc`로 DDR_buf[100~123]에 따로 보관(`net_hls.cc:954`). DW6 전반부에서 `load_and_reorg`로 이 4개 이미지 타일을 공간 stride-2 reorg하며 채널을 4배(192→768)로 펼쳐 로드(`net_hls.cc:1109-1111`). 이는 PyTorch `ReorgLayer`(stride2, ch x4)와 대응(`GPU/models.py:12-31, 82-84`).
- **검출 후처리(온칩)**: `compute_bounding_box`가 FM_buf_acc(10채널 출력)를 4개 사분면(20x40 그리드)으로 나눠 각 영역에서 conf 최댓값(채널4 또는 채널9) 위치를 찾고 박스 좌표/anchor 인덱스를 추출 (`net_hls.cc:25-270`). 4 배치 이미지에 대응하는 4 사분면. sigmoid/exp는 PL이 아니라 호스트에서 적용(주석처리된 `net_hls.cc:39,53` 참조, 실제 적용은 `SkyNet.py:139-148`).

---

## 4. 데이터플로우

### 4.1 전체 흐름
```
[ARM/PYNQ 호스트]
  이미지 4장 stitch(644x324) → cma_array(img)
  weights_fixed.bin → conv_1x1/3x3/bias cma_array
        │  (물리주소를 s_axilite 레지스터에 write)
        ▼
[PL: SkyNet IP]  (512-bit AXI, 32채널/워드)
  DW1(3x3→1x1→pool) → dw1_pool DDR
  DW2 → dw2_pool DDR
  DW3 → DDR_buf[100..] (reorg 원본 보존) + pool→DDR_buf
  DW4 → DDR_buf  (DDR 더블버퍼 타일링)
  DW5 → DDR_buf
  DW6: reorg(DW3) ++ DW5 = 1280ch → 3x3 → 1x1
  PW7: 1x1 head → FM_buf_acc(10ch)
  compute_bounding_box → predict_boxes[4][5], constant[4][3]
        ▼
[ARM 호스트]  sigmoid/exp 후처리 → 픽셀 좌표 박스
```

### 4.2 핵심 데이터 이동 패턴
- **타일 기반(tiled) + DDR 더블버퍼링**: 온칩 버퍼 용량(32x44x84)이 전체 FM(예 192x20x40)보다 작아, 채널/공간을 타일로 잘라 DDR을 경유. ping-pong(예 `FM_buf1`/`FM_buf3`, `net_hls.cc:837-843, 985-992`)으로 load와 compute 중첩.
- **512-bit burst 패킹/언패킹**: `relu_copy_buf_to_DDR`(쓰기, `net_hls.cc:283-304`), `load_buf_from_DDR`(읽기, `net_hls.cc:333-350`)가 32채널을 16-bit씩 512-bit 워드에 패킹. AXI 대역폭 = 32채널 동시 전송.
- **reorg 인터리브 로드**: `load_and_reorg_part` (`net_hls.cc:626-742`)는 16개 인접 512-bit 워드를 읽어 공간 2x2 reorg + 채널을 4개 출력버퍼(buf_out_1~4)로 분배(채널 8씩 그룹). DW3의 192ch이 공간축소(stride2)로 768ch가 되는 과정을 메모리 레이아웃 변환으로 구현.
- **데이터 흐름 스타일**: 전형적 **dataflow/tiled accelerator** (systolic array는 아님). PE는 16-input dot-product 트리(`compute_engine_16`)를 32개 병렬 배치한 **SIMD형 broadcast** 구조. 가중치는 매 타일 재로드(weight-reload), 출력은 acc 버퍼에 머무름(output-stationary 성향).

---

## 5. HW/SW 매핑

| 항목 | 위치 | 근거 |
|------|------|------|
| 학습(float) | GPU/PyTorch | `GPU/models.py:34-86` (depthwise-separable, reorg, head) |
| 가중치 양자화+재배치 | HLS CSIM(호스트 1회) | `reorder_weight.cc` 전체, 출력 `weights_fixed.bin` |
| conv/pool 연산 | PL(FPGA) | `conv1x1.cc`, `dwconv3x3.cc`, `Relu_Max_Pooling` |
| 박스 좌표 raw 추출 | PL | `net_hls.cc:25-270 compute_bounding_box` |
| sigmoid/exp 후처리 | PS(ARM) | `SkyNet.py:139-148` (PL은 logit만 출력) |
| 이미지 stitch/전처리 | PS(ARM, multiprocess) | `SkyNet.py:88-118` |
| IP 제어/메모리 할당 | PS(PYNQ) | `SkyNet.py:171-226` |
| DVFS 제어 | PS(쉘) | `SkyNet.py:65-68` (`dvs`/`dfs` 바이너리 호출) |

- **PS↔PL 인터페이스**: PYNQ `Overlay`로 bitstream 로드(`SkyNet.py:61`), `cma_array`로 연속 물리메모리 할당(`SkyNet.py:35-44`), 물리주소를 IP 레지스터(0x10~0x58)에 write(`SkyNet.py:173-181`). 시작은 `write(0x00,1)`, 완료 폴링은 `read(0x00)` (`SkyNet.py:216-219`) — `ap_ctrl` 핸드셰이크.
- **가중치 파일 레이아웃 일치성**: 호스트가 `conv_weight_1x1_all (413,32)`, `conv_weight_3x3_all (64,3,3)`, `bias_all (106)` 순으로 읽음(`SkyNet.py:37-56`). 각 원소는 `dtype='B,'*63+'B'` = 64바이트(512-bit) 구조체(`SkyNet.py:33`). reorder_weight.cc의 1x1→3x3→bias 기록 순서(`reorder_weight.cc:638,653,663`)와 동일.
- **DVFS**: 데모 인자 `--frequency`(20~500MHz), `--voltage`(660~850mV) (`SkyNet.py:20-23`). 12V 레일 전력을 `pynq.DataRecorder`로 측정해 에너지(J) 산출(`SkyNet.py:184-185, 206, 235`).

---

## 6. 빌드 · 실행

1. **HLS 합성** (`README.md:67-80`, `script.tcl`): `vivado_hls -f script.tcl`.
   - csim → csynth → (cosim 주석) → export_design(ip_catalog) (`script.tcl:38-49`).
   - top 함수 `SkyNet`, part `xczu7ev`, period 3ns (`script.tcl:7,34-35`).
   - CSIM 중 `reorder_weight.cc`가 `weights_fixed.bin` 생성(테스트벤치로 등록, `script.tcl:27`).
2. **Vivado 시스템 통합** (`README.md:84-92`, `FPGA/RTL/script.tcl`): `vivado -mode batch -source script.tcl -tclargs SkyNet . ../HLS/`. Zynq 300MHz, 2개 HP AXI(INPUT/OUTPUT) 연결. bitstream 생성은 스크립트 종료 후 추가 40~60분 소요(`README.md:92`).
3. **호스트 배포** (`README.md:95-108`): `.bit`/`.hwh`/`weights_fixed.bin`을 보드 업로드 후 `SkyNet.py` 실행. 데모: `sudo python3 SkyNet.py --frequency 330 --voltage 750` (`README.md:23-24`).

---

## 7. 의존성

- **HLS 측**: Vivado HLS(2018 추정, `script.tcl:4` Copyright 2018), `ap_fixed.h`, `hls_stream.h`, `hls_math.h` (`net_hls.h:4-5`, `net_hls.cc:1-6`). 표준 C++ I/O(`<fstream>`).
- **호스트 측**: PYNQ(`pynq.Overlay`, `pynq.Xlnk`, `pynq.get_rails`, `DataRecorder`), numpy, PIL, cv2, multiprocessing (`SkyNet.py:4-17`). DVFS용 외부 바이너리 `dvs`/`dfs`(소스 미포함, 제외).
- **학습 측**: PyTorch(`torch`, `torch.nn`), darknet 유래 `region_loss.RegionLoss`, `utils.*` (`GPU/models.py:1-9`). anchors 2개(`models.py:75-77`).

---

## 8. 강점 · 한계

**강점**
- depthwise-separable conv를 두 개의 전용 HLS 커널(3x3 depthwise, 1x1 pointwise)로 명확히 분리, 각각 32채널 SIMD 병렬 + ReLU/pool 융합으로 효율적.
- **16-bit 슬롯 패킹 + 512-bit AXI**로 32채널/워드 전송 — 대역폭 활용 단순·일관(`load_buf_from_DDR` 등 재사용성 높음).
- 단일 인스턴스 재사용(ALLOCATION limit=1)으로 ZCU104 면적에 맞춤. DVFS + 전력측정 통합으로 에너지 효율 평가가 가능한 완결형 배포.
- 학습(float, PyTorch)부터 양자화/재배치(reorder_weight)·합성·배포까지 end-to-end 흐름이 한 저장소에 존재.

**한계**
- **고정소수점(fixed-point) PTQ**: scale/zero-point 정수 양자화가 아니라 비트폭 고정 + 포화. 비트폭(FM 9b/WT 11b)이 하드코딩되어 정밀도-면적 trade-off 탐색이 제한적이고 QAT 미적용(`net_hls.h:45-47`).
- **하드코딩 토폴로지**: `SkyNet()` 본문이 7블록 채널수/타일링/DDR 인덱스를 모두 수작업 전개(약 1200행). 다른 네트워크/해상도로의 일반화 어려움.
- **systolic array 아님**: 16-input dot 트리 broadcast 구조라 데이터 재사용(weight stationary 정도)이 제한적이며, 가중치를 타일마다 재로드.
- **단일 박스/단순 head**: 사분면당 1박스(배치=4 고정) 검출 후처리(`net_hls.cc:25-270`)는 일반 다객체 검출로 확장 어려움. NMS 없음.
- **CSIM 의존 양자화**: weights_fixed.bin 생성이 HLS CSIM 실행에 묶여 있어 가중치 갱신 파이프라인이 무겁다.

---

## 9. 우리 프로젝트(ViT/Transformer FPGA 가속기 + XR 시선추적) 시사점

1. **채널-병렬 512-bit 패킹 패턴 재사용성**: 32채널을 16-bit씩 512-bit 워드에 패킹하는 `load_buf_from_DDR`/`relu_copy_buf_to_DDR`(`net_hls.cc:283-350`) 방식은 ViT의 임베딩/토큰 차원(예 D=64/128) 타일 전송에 직접 응용 가능. HG-PIPE 계열의 토큰-병렬 dataflow에서 "한 워드 = N개 element"의 정렬 단위를 잡는 좋은 레퍼런스.
2. **dot-product 트리 PE → MatMul/Attention 재해석**: `compute_engine_16`(16-input adder tree, `conv1x1.cc:11-71`)는 1x1 conv = 1x1 GEMM이다. Transformer의 Q·Kᵀ, attn·V, FFN이 모두 GEMM이므로, 이 가산 트리 PE를 reduction 길이에 맞춰 확장하면 systolic 대안(broadcast SIMD)으로 활용 가능. 단, 재사용 효율은 systolic/output-stationary 대비 낮음을 유의.
3. **고정소수점 비트폭 분리 설계**: FM/ACC/WT를 서로 다른 ap_fixed 비트폭으로 분리(`net_hls.h:45-47`)하고 곱셈 중간을 32비트로 확장하는 패턴은, 우리의 양자화 가속기에서 LayerNorm/Softmax 누적 오차 관리에 시사적. 다만 ViT는 동적 range가 커서 단순 fixed보다 per-tensor scale 양자화가 더 안전(이 repo의 한계를 보완점으로 채택 권장).
4. **오프라인 weight reorder 분리**: 하드웨어 접근 패턴(reorg/4-way 인터리브, `reorder_weight.cc:119-168` ↔ `net_hls.cc:626-742`)을 컴파일타임 가중치 재배치로 흡수하는 설계는, attention head/패치 reshape를 HW에 맞춰 사전 변환하는 우리 파이프라인에 그대로 차용 가능.
5. **DVFS + 전력측정 통합 배포 플로우**: `SkyNet.py`의 PYNQ overlay + cma_array + 12V 레일 에너지 측정(`SkyNet.py:184-235`)은 XR 엣지(저전력 시선추적)에서 latency/energy 동시 평가 harness로 거의 그대로 재사용 가능.
6. **PS/PL 후처리 분담 모델**: sigmoid/exp 등 비선형 후처리를 PS로 넘기고 PL은 logit/raw만 산출(`net_hls.cc:39 주석` + `SkyNet.py:139-148`)하는 분담은, Softmax/GELU의 비선형부를 PS 또는 LUT로 분리하려는 우리 설계 결정에 참고점. 단 latency-critical 경로(시선추적 실시간성)에서는 PL 내 LUT 처리가 더 유리할 수 있어 trade-off 검토 필요.

---

### 부록: 검증 메모
- HLS 핵심 소스(net_hls.cc 1224행 전체, conv1x1.cc, dwconv3x3.cc, reorder_weight.cc 668행, net_hls.h) 및 SkyNet.py, models.py, script.tcl을 직접 Read로 확인함.
- golden_c.cc/output_verify.cc는 CSIM 검증 reference(테스트벤치)로 핵심 가속기 경로가 아니어서 정밀분석에서 제외(존재만 확인, `script.tcl:25-26`).
- `*.bin/.bit/.hwh/.weights/.jpg`는 제약에 따라 이름만 확인하고 내용 미열람.
- DVFS 바이너리 `dvs`/`dfs`는 소스 미포함(컴파일 산출물)이라 동작은 `SkyNet.py:65-68` 호출부 기준으로만 기술. 내부 구현은 **확인 불가**.
