# yolov2_xilinx_fpga-flex 코드베이스 정밀 분석

> 분석 대상: `REF/CNN-Accel/yolov2_xilinx_fpga-flex/`
> 분석 방식: HLS 커널 / SDK 호스트 / 가중치 재배치 스크립트를 함수 단위로 Read 후 (파일:라인) 인용. 추론은 "추정", 확인 불가는 "확인 불가" 명시.

---

## 1. 개요 (목적 / 원논문 / 타깃보드)

- **목적**: YOLOv2(다크넷) 객체검출을 Xilinx Zynq-7000 / Zynq UltraScale+ FPGA에 가속하는 **풀스택 데모**. 입력 416x416x3, 31 레이어, 29.47 GOP(`README.md:82,177`).
- **두 정밀도 버전**:
  - **FT32**(float32): 곱셈 2 DSP + 덧셈 3 DSP 소모 → 자원 큼(`README.md:58`).
  - **INT16**(고정소수점 16bit): 곱셈 1 DSP, 덧셈은 LUT → 자원·전력 효율 대폭 개선(`README.md:58`). 같은 29.47 GOP에서 INT16이 12→62 GOP/s, 전력효율 5.8→59.6 GOP/s/W (`README.md:84-85`).
- **원논문(확인됨)**: 저자 석사학위논문 "Research of Scalability on FPGA-based Neural Network Accelerator"(Jiangnan University, 2019) + 저널 2편(`README.md:3-6`). 즉 학위연구 산출물.
- **타깃 보드**(`README.md:1-2,62`): PYNQ-z2, Zedboard, ZU3EG(EdgeBoard), ZCU102. 평가표 기준은 **EdgeBoard ZU3EG**(A53 4코어 1.2GHz + 4GiB DDR4 + FPGA, `README.md:62`).
- **툴체인**: Vivado / Vivado HLS **2019.2 권장**(`hls/README.md:4-6`, `README.md:55`). 2019.2 이후 Vitis HLS는 수동 ping-pong 버퍼 인식 실패 우려(`README.md:55`).
- **설계 출처(참고문헌)**: Going Deeper(Qiu), Caffeine, Optimizing FPGA Accelerator(Zhang), DianNao 등(`README.md:190-197`) — 고전적 loop-tiling + Tn/Tm 병렬 + ping-pong 패턴.

---

## 2. 디렉토리 구조 (자체 + 제외 이유)

### 자체 핵심 (정밀 분석 대상)
```
hls/                          # Vivado HLS 가속기 커널(IP)
  src_int16/   cnn_hls.cpp    # ★INT16 conv/pool 커널 + load/compute/store 파이프라인
               cnn_t.h        # 파라미터 #define 템플릿(#DEFINE_HEADER# 치환)
  src_int16_128b/ acc_i16c.cpp# INT16 + 128bit 데이터버스 + LANE_NUM 벡터화 변형
  src_float32/ cnn.cpp        # FT32 커널(곱셈 2DSP/덧셈 3DSP)
  src_float32_mp/, _fusion/   # FT32 멀티포트(4r2w)·layer-fusion 변형
SDK/                          # PetaLinux/베어메탈 호스트(/dev/mem 직접 매핑)
  src_int16/   cnn.cpp        # ★FPGA_Acc: AXI-Lite 레지스터 라이트 + /dev/mem 매핑
               yolov2.h       # darknet 파서/네트워크/region 디코딩
               yolov2_acc_sim.h # ★yolov2_hls_ps: 레이어 스케줄러 + Q 로딩 + iofm 오프셋
               main.cpp       # 이미지 로드→yolov2_hls_ps→NMS→그리기
  src_int16_128b/, src_float32*/ # 각 HLS 버전 대응 호스트
software_version/
  02_ReorganizeWeight_Int16/  # ★가중치 재배치+양자화(float32→fixed16) 스크립트/Makefile
  02_ReorganizeWeight_Float32(_4r2w)/ hw_params_gen.py  # HLS 파라미터 #define 생성기
pynq/        yolov2.ipynb     # PYNQ Overlay 기반 노트북 흐름
             softmax.c/.so    # region softmax(호스트 .so)
petalinux/, vivado/  README.md# 빌드 가이드
```

### 제외 (이름만 언급)
- `software_version/01_ExtractWeightAndBiasFromDarknet/darknet/` — **darknet 원본**(third-party, pjreddie). 가중치 추출만 담당. 분석 제외(README만 참조: `01.../README.md`).
- `pynq/yolov2.bit`, `*.hwh`/`.tcl`, `*.so`, `*.weights`, 결과 이미지(`*.jpg/.png`), `.Xil` — 비트스트림/바이너리/산출물.
- `hls/src_*/stb_image.h`, `stb_image_write.h` — third-party 이미지 IO 라이브러리.
- `software_version/01.../darknet/python/*.py`, `scripts/*.py`, `labels/make_labels.py` — darknet 유틸/라벨 생성.
- 생성물: `weight_reorg_ap16.bin`, `bias_ap16.bin`, `iofm_Q.bin`, `weights_ap16_Q.bin` 등(런타임 로드 대상).

> 02_Reorganize* 의 **재배치 로직(C/Makefile)**은 핵심으로 분석. 단 `hw_params_gen.py`는 단순 헤더 치환기임이 확인됨(아래 3.5).

---

## 3. 핵심 모듈 정밀 분석

### 3.1 INT16 HLS 커널 — `hls/src_int16/cnn_hls.cpp` (★)

탑 함수 `FPGA_Acc`(`:578`)는 **단일 IP가 conv/maxpool/reorg를 모두 처리**(layer type를 인자로 받음). 구조: 4 AXI master(ifm/ofm/weight/bias) + 1 AXI-Lite(CTRL_BUS) (`:582-612`).

**파라미터 패킹**: 32bit 레지스터에 여러 필드를 비트필드로 압축. 예 `k_s_pad_ltype`= ksize<<24 | kstride<<16 | pad<<8 | ltype (`:624-627`), `TMTN`=TM<<16|TN, `iofm_num`=ifm<<16|ofm 등(`:614-631`). AXI-Lite 레지스터 수 절감 목적(추정).

**타일링**: 외부 3중 루프 `tr(ofm_h)/tc(ofm_w)/tm(OFM_num_bound)` (`:691-698`). Tile 크기 Tr/Tc/Tm/Tn은 컴파일타임 #define(int16 기본 Tn=2,Tm=60,Tr=Tc=26 — `hw_params_gen.py:6-9`).

**연산 엔진 `conv2d_tile`** (`:378-441`): 핵심 MAC.
- 루프: `i,j(커널) → tr,tc(출력) → tm(출력채널) → tn(입력채널)`, `tr/tc` 루프에 `PIPELINE II=1`(`:418`).
- `partial_mul[Tm][Tn]` 완전분할(`:404-405`) → **Tm×Tn개 곱셈기 병렬**(output-parallel Tm × input-parallel Tn). 이것이 README의 "parallel multiplication units + add tree"(`README.md:48`).
- MAC: `partial_mul[tm][tn] = weight[tm][tn][i][j]*input[tn][...] >> WeightAddInputSubInter` (`:430`). **곱 직후 우시프트**로 중간 비트폭(INTERWIDTH=19) 맞춤 — 고정소수점 스케일 정렬.
- add tree: `partial_sum += partial_mul[tm][tn]` (`:434-437`), 누산 `output_buffer += local_beta + partial_sum`. `#pragma HLS DEPENDENCE variable=output_buffer inter false`(`:421`)로 누산 의존성 완화(II=1 달성 핵심; README Q&A에서 이 dependency pragma가 stall 원인일 수 있다 경고 — `README.md:9-10`).
- bias 초기화: 첫 입력타일(`i==0&&j==0&&ne0`)에 `local_beta_buffer`로 초기화(`:423-424`), `copy_local_beta`(`:363-376`)가 beta를 `<< InterSubBeta`로 스케일.

**고정소수점 스케일 수식**(`:651-657`):
- `InterSubBeta = INTERWIDTH - BetaQ` (bias를 중간폭으로)
- `WeightAddInputSubInter = WeightQ + InputQ - INTERWIDTH` (곱셈 후 중간폭으로 우시프트량)
- `InterSubOutput = INTERWIDTH - OutputQ` (출력 양자화 우시프트량)
- 즉 레이어마다 **WeightQ/InputQ/OutputQ/BetaQ**(소수점 위치)를 받아 동적 스케일 — 레이어별 동적 fixed-point.

**LeakyReLU + 출력양자화 `nonlinear_leaky_row`** (`:164-224`): conv 후 `tmp1<0`이면 `(int64)tmp1*0xccc >>15`(≈0.1 곱, leaky), 그 후 `>> InterSubOutput`로 출력 Q 양자화(`:190-200`), int16 2개를 32bit 워드에 패킹(`:206-212`) → 256bit 정렬 메모리로 write.

**Maxpool `maxpool_tile`** (`:326-361`): Tn채널 병렬, `tr/tc → i,j(커널)` 순서로 윈도우 최대값(`:351-357`). 초기값 `MIN_NEG=0x8001`(`cnn_t.h:25`).

**메모리 256bit(32bit*?) 정렬 로직**: ifm/ofm 폭을 짝수 정렬(`IW_align_256b`, `:665-667`). `int32_t*` 포인터로 받아 **int16 2개를 32bit에 패킹**(`ifm_copy_lbuf2ibuf`의 `buf_256b[2]`, `:39-42`). 즉 128/256bit 버스를 int16 데이터에 활용.

**Ping-pong (이중버퍼)**:
- ifm/weight 로드 vs compute: `load_compute_wrapper`(`:471-544`)에서 `ifm_buffer0/1`, `weight_buffer0/1` 교대(`:502-518`), 로드(`ifm_weight_load_wrapper`)와 `conv2d_tile`를 다른 버퍼로 동시 진행 → 로드/연산 오버랩.
- ofm 출력 vs compute: 탑루프에서 `ofm_buffer0/1` pingpongm 교대(`:705-722`), compute와 `write_back_output_reorg`(write) 오버랩. README "ping-pong operation"(`README.md:50-51`)의 실제 구현.

**입력 로드 `input_load`** (`:68-109`): ifm row를 `local_buf0/1` ping-pong으로 memcpy(`ifm_mmcpy_row`, `:4-24`)하며 동시에 패딩 처리해 on-chip buffer로 복사(`ifm_copy_lbuf2ibuf`, `:26-66`). 패딩값 `pad_val`, 경계 밖은 pad로 채움(`:49-54`).

### 3.2 INT16 128bit 변형 — `hls/src_int16_128b/acc_i16c.cpp`

- src_int16와 동일 알고리즘이나 **데이터 타입 `DT_IO`(128bit) + `LANE_NUM` 벡터 차원** 도입. 버퍼가 `[T*/LANE_NUM][...][LANE_NUM]` 형태(`:477-479`).
- `conv2d_tile`(`:477-541`): MAC 인덱싱이 `weight[tm/LANE_NUM][tn][i][j][tm%LANE_NUM]`, `input[tn/LANE_NUM][...][tn%LANE_NUM]`(`:519-520`) — **LANE_NUM 단위 SIMD 벡터화로 128bit 버스 1트랜잭션에 여러 채널** 읽기.
- 스케일: `tmp_mul << ComQ`(C_SL_EN) 또는 `>> ComQ`(`:523-526`) — 곱 후 가변 시프트(좌/우 모두 지원, src_int16의 우시프트만보다 유연).
- 평가표 C/D(`README.md:68-69`): Tn=8/Tm=24, II_CONV=1, 128/128 bit 버스, 62 GOP/s. 즉 128b 버전이 **버스폭 확대(32→128) + Tn 증대(2→8)**로 처리량 향상.

### 3.3 FT32 커널 — `hls/src_float32/cnn.cpp`

- 동일 loop-tiling/ping-pong 골격이나 `float` 누산. `postproc`(`:306-322`)에서 bias 덧셈 + leaky(`tmp0*0.1f`) — 정수 시프트 대신 float 곱.
- `write_back_output_reorg`(`:328-...`): **burst length 최적화** — ofm이 전부 연속(IsAllCont)/열연속(IsColCont)/비연속에 따라 burstlen 분기(`:348-360`). DRAM 대역폭 극대화(README "weight arrangement", `README.md:44-45`).
- FT32는 곱셈 2DSP/덧셈 3DSP라 자원 큼(`README.md:58`) → mp(멀티포트 4r2w)·fusion 변형으로 보완(`src_float32_mp`, `_fusion`).

### 3.4 SDK 호스트: FPGA 제어 — `SDK/src_int16/cnn.cpp` (★)

- `FPGA_Acc`(`:110-196`): **베어메탈/PetaLinux 직접 레지스터 제어**. `/dev/mem`을 `mmap`(`:120-123`)해 가속기 base(`ACC_BASEADDR`)에 접근.
- 흐름: ① `ap_idle` 폴링(`:130-135`) → ② 모든 인자를 AXI-Lite 레지스터에 `WriteReg`(IFM/OFM/Weight/Bias 주소 + 패킹 파라미터 + Q값들, `:143-178`) → ③ `AP_CTRL=0x1` Start(`:181`) → ④ `ap_done` 폴링(`:182-187`). 즉 **폴링 기반 동기 호출**(인터럽트 비활성 `GIE=0`, `:180`).
- 주소: `WEIGHT_BASEADDR + Weight_offset*4`, `BETA_BASEADDR + Beta_offset*4`(`:145-146`) — weight/bias는 별도 물리메모리 영역.
- `copy_mem2dev`/`copy_dev2mem`/`copy_file2mem`(`:4-108`): `/dev/mem` mmap으로 DRAM ↔ 호스트버퍼 ↔ 파일 복사. 페이지(HPAGESIZE) 정렬.

### 3.5 SDK 호스트: 레이어 스케줄러 — `SDK/src_int16/yolov2_acc_sim.h` (★ 가장 중요한 호스트 로직)

`yolov2_hls_ps`(`:117-382`)가 **네트워크 31레이어를 순회하며 FPGA_Acc를 호출**.

**(1) Q(소수점 위치) 로딩** (`:119-180`): 오프라인 생성 바이너리들을 로드.
- `iofm_Q.bin` → `iofmQ[]`(레이어별 입출력 fixed-point Q). route 공유 레이어 20/21은 작은 쪽으로 통일(`:136-139`).
- `weight_reorg_ap16.bin`(재배치+양자화된 가중치) → `WEIGHT_BASEADDR`로 적재(`:145-146`), `bias_ap16.bin` → `BETA_BASEADDR`(`:148-149`).
- `weights_ap16_Q.bin`→`WeightQ[]`, `bias_ap16_Q.bin`→`BetaQ[]`(`:158-180`). 즉 **레이어별 Weight/Bias/IO Q를 모두 외부 파일에서 받아 동적 스케일**.
- weight/bias offset 테이블은 코드에 하드코딩(`weight_offset[32]`, `beta_offset[32]` — `:119-124`).

**(2) IO 메모리 오프셋 — `generate_iofm_offset`** (`:36-95`): **ping-pong DRAM 버퍼 + route용 고정주소**.
- 짝/홀 레이어가 Memory_top/Memory_bottom을 교대 사용(`:52-61`) — 레이어 간 in-place 방지 더블버퍼.
- route(layer16,24,27)용 고정 영역 예약: `ROUTE16_LEN`, `CONV24_LEN`, `CONV27_LEN`(`:38-40`, `:82-93`). README "routing layer는 미리 특정 주소 설정으로 구현"(`README.md:36`)의 실체.

**(3) 레이어별 호출**(`:212-371`):
- CONV/MAXPOOL(`:217-318`): Tr/Tc/Tm/Tn을 on-chip buffer 한계와 ofm 크기 min으로 산출(`:240-245`), maxpool은 TN=0(`:249-251`). `OFM_num_bound`·`mLoopsxTM`·`mLoops_a1xTM`로 ping-pong 경계 계산(`:254-257`). en_bits에 {IsReLU,LoadBias,IsNotConv} 패킹(`:275-279`). leaky면 IsReLU set(`:288-289`). FPGA_Acc에 패킹 파라미터 + `WeightQ/BetaQ/InputQ(iofmQ)/OutputQ` 전달(`:297-299`).
- REORG(layer27, `:323-345`): **호스트 CPU에서 reorg_cpu 수행**(HW 아님). `reorg_cpu`(`:97-115`)는 darknet space-to-depth. 26x26x64→13x13x256. 메모리 정렬(out_align/out_left_num) 처리(`:336-343`).
- ROUTE(`:346-352`): 출력만 로그(주소는 generate_iofm_offset에서 이미 배치 — 연산 없음).
- REGION(layer31, `:353-368`): DRAM에서 회수 후 `*pow(2,-iofmQ)`로 **fixed→float 역양자화**(`:360-365`), `forward_region_layer`로 softmax/bbox 디코딩(호스트).

> 즉 conv/pool은 FPGA, reorg/route/region(후처리)은 호스트 CPU. **HW/SW 분할이 명확**.

### 3.6 SDK 호스트: darknet 파서 — `SDK/src_int16/yolov2.h`

- darknet의 `layer`/`network` 구조체(`:63-373`), cfg 파서(`parse_network_cfg`, `:1940-2016`), region/yolo box 디코딩(`get_region_box`, `correct_region_boxes`, `do_nms_sort` — `:2304-2528`), 이미지 letterbox/resize(`:925-1008`)를 그대로 차용. **darknet 호환 호스트**(yolov2.cfg/coco.names 사용, `main.cpp:25,16`).
- 자체 추가는 파일 끝 `#include "yolov2_acc_sim.h"`(`:2769`) — 가속기 스케줄러 주입.

### 3.7 가중치 재배치/양자화 — `software_version/02_ReorganizeWeight_Int16/`

- README(`02.../README.md:1-8`): darknet에서 추출한 `weights.bin`/`bias.bin`을 **(a) 타일 단위 연속 블록으로 재배치**(DRAM burst 대역폭 위해, `README.md:44-45`) **(b) float32→fixed16 양자화**. `make gen_i16; ./test_layers` 로 `weight_reorg_ap16.bin` 등 생성, `make test_i16`로 SW 검증(forward_region_layer까지).
- `hw_params_gen.py`(`02_ReorganizeWeight_Int16/hw_params_gen.py`): **HLS 파라미터 헤더 생성기**. Tn=2/Tm=60/Tr=Tc=26/INTERWIDTH=19/OnChipIB 크기를 문자열로 만들어(`:4-29`), 템플릿(`cnn_t.h`/`acc_*_t.h`)의 `#DEFINE_HEADER#` 토큰을 치환해 `cnn.h`/`acc_*.h` 생성(`:31-71`). 즉 **하드웨어 병렬도(Tn/Tm/Tr/Tc)를 한 곳에서 정의→HLS·SW 양쪽에 주입**. 실제 양자화 수치 로직은 C(`test_layers`)에 있음(스크립트 자체는 치환만 — 확인됨).
- OnChipIB: `(Tc-1)*HW_S+K`, `(Tr-1)*HW_S+K`(`:12-13`) — stride 고려한 입력타일 버퍼 크기. cnn_t.h(`:39-40`)와 일치.

### 3.8 PYNQ 흐름 — `pynq/`

- `pynq/README.md`: `Overlay("yolov2.bit")` 로드 후 `yolov2.ipynb` 실행, 이미지 경로/비트스트림 교체 가능(`pynq/README.md:8-14`). PYNQ(Python) 추상화로 베어메탈 `/dev/mem`보다 간편.
- `softmax.c`(`:6-34`): region softmax(over class). `.so`로 컴파일해 노트북에서 호출(a53_64bit용 별도 .so).

---

## 4. 데이터플로우

### 오프라인 (PC)
```
01_Extract(darknet) → weights.bin / bias.bin
02_Reorganize_Int16 → (재배치+fixed16 양자화) weight_reorg_ap16.bin, bias_ap16.bin,
                       weights_ap16_Q.bin(WeightQ), bias_ap16_Q.bin(BetaQ), iofm_Q.bin
hw_params_gen.py    → cnn.h / acc_*.h (Tn/Tm/Tr/Tc/INTERWIDTH #define)
Vivado HLS(2019.2)  → YOLOv2 IP → Vivado BD → .bit + .hdf → PetaLinux/PYNQ
```

### 온보드 (런타임, yolov2_hls_ps 기준)
```
이미지 → letterbox 416x416 → input*2^iofmQ[0] → DRAM in_ptr[0] (int16)
for layer in 0..30:
  CONV/POOL → FPGA_Acc(AXI-Lite 레지스터 라이트 → Start → ap_done 폴링)
              [input_load(ping-pong) → conv2d_tile(Tm×Tn MAC, II=1) → leaky+>>OutputQ → write]
  REORG/ROUTE/REGION → 호스트 CPU (reorg_cpu / 주소배치 / 역양자화+region 디코딩)
→ NMS → draw_detections → predictions.png
```

### 메모리 모델
- ifm/ofm: Memory_top/bottom **ping-pong 더블버퍼**(레이어 짝/홀 교대, `acc_sim:52-61`).
- weight/bias: `WEIGHT_BASEADDR`/`BETA_BASEADDR` 고정영역, 레이어 offset 누적(`acc_sim:301-302`).
- route 출력: 예약된 고정 주소(`ROUTE16_LEN` 등).
- on-chip: ifm/ofm/weight buffer 모두 `ARRAY_PARTITION complete dim=1`(채널 병렬, `cnn_hls:476,481,660`).

---

## 5. HW/SW 매핑

| 기능 | 담당 | 근거 |
|---|---|---|
| Conv MAC (Tm×Tn 병렬, add tree, II=1) | **FPGA** `conv2d_tile` | `cnn_hls.cpp:378-441` |
| MaxPool | **FPGA** `maxpool_tile` | `cnn_hls.cpp:326-361` |
| LeakyReLU + 출력 fixed-point 양자화 | **FPGA** `nonlinear_leaky_row` | `cnn_hls.cpp:188-200` |
| ifm/weight/ofm ping-pong | **FPGA** load_compute_wrapper / 탑루프 | `cnn_hls.cpp:502-518,705-722` |
| 레이어 스케줄·타일 파라미터 산출 | **호스트** yolov2_hls_ps | `acc_sim:212-318` |
| 레지스터 라이트·Start/Done 폴링 | **호스트** FPGA_Acc | `cnn.cpp:130-187` |
| Reorg(space-to-depth) | **호스트 CPU** reorg_cpu | `acc_sim:97-115,323-345` |
| Route(주소 배치) | **호스트**(연산無) | `acc_sim:36-95,346-352` |
| Region(softmax/bbox) + 역양자화 | **호스트 CPU** forward_region_layer | `acc_sim:353-368` |
| 가중치 재배치+양자화 | **오프라인 PC** 02_Reorganize | `02.../README.md` |

---

## 6. 빌드 · 실행

1. **가중치 추출**: darknet에서 `weights.bin`/`bias.bin`(01 폴더, README만).
2. **재배치+양자화**: `cd 02_ReorganizeWeight_Int16; make gen_i16; ./test_layers [img]` → ap16 bin들 생성; `make test_i16; ./test_layers`로 SW 검증(`02.../README.md:6-8`).
3. **HLS 파라미터**: `python hw_params_gen.py` → `cnn.h` 등(`hw_params_gen.py:59-71`).
4. **HLS 합성**: Vivado HLS 2019.2로 `hls/src_int16` 합성(target `xazu3eg-sfvc784-1-i`, clk 3ns, `hls/README.md:4-6`). C-sim만 통과, C-RTL co-sim은 testbench overflow로 미구현(`README.md:24`, `hls/README.md:9`).
5. **Vivado BD**: YOLOv2 IP 연결, clock wizard Reset=Active Low, HP0/1/2 포트로 동시 write(`README.md:25-26`).
6. **배포**: .hdf + .bit → PetaLinux 생성(`petalinux/README.md`) 또는 PYNQ(`Overlay("yolov2.bit")` + ipynb).
7. **실행**: `./yolov2_ft32.elf <img>` / int16 elf — `static -lm -O2`(`README.md:57,92`).

> SDK는 hw_cfg 변경 시 **수동 코드 수정**(스크립트 없음, `SDK/src_int16/README.md:2`).

---

## 7. 의존성

- **Vivado / Vivado HLS 2019.2**(강력 권장; Vitis HLS는 ping-pong 인식 문제, `README.md:55`).
- `ap_int.h`(Xilinx 고정소수점), AXI4 master/AXI4-Lite.
- darknet(가중치/cfg/coco.names/parser) — third-party.
- PetaLinux(`/dev/mem` 직접 mmap) 또는 PYNQ(Python Overlay).
- stb_image(이미지 IO) — third-party.
- 호스트 빌드: gcc `-static -lm -O2`.

---

## 8. 강점 · 한계

**강점**
- **단일 IP가 conv/pool 다용도** + layer-type 인자화 → 자원 재사용. 31레이어 전체를 한 엔진으로 순차 처리.
- **3중 ping-pong**(ifm/weight 로드, 연산, ofm 쓰기 모두 오버랩) → 계산엔진 동적 이용률↑(`README.md:50`).
- **레이어별 동적 fixed-point**(WeightQ/InputQ/OutputQ/BetaQ를 런타임 레지스터로) → 레이어별 최적 소수점.
- INT16에서 곱셈 1DSP/덧셈 LUT → FT32 대비 5배 처리량·10배 전력효율(`README.md:84-85`).
- 멀티 배포 경로(베어메탈/PetaLinux/PYNQ) + 다보드(PYNQ-z2~ZCU102) — 교육/이식성 우수.
- DRAM burst 최적 weight arrangement + 연속/열연속 분기 write(`cnn.cpp(f32):348-360`).

**한계**
- **C-RTL co-simulation 미구현**(testbench overflow, `README.md:24`) → RTL 검증 부재, C-sim만.
- Q&A에 "Layer0 stall" / "출력이 SW와 다름/백색" 등 **HLS dependency pragma 안정성 이슈**(`README.md:9-13`) — 버전 민감.
- reorg/route/region이 **호스트 CPU**라 그 구간 latency는 가속 제외(README latency 측정도 post-process 제외, `README.md:87`).
- weight/bias offset, route 메모리 길이 등이 **YOLOv2 31레이어에 하드코딩**(`acc_sim:119-124,38-40`) → 다른 네트워크 이식 어려움.
- SDK는 hw_cfg 변경 시 수동(`SDK/src_int16/README.md:2`).
- INT16 C-sim이 너무 느려 미실행(`hls/README.md:15`).
- 폴링 기반 동기 호출(인터럽트 미사용, `cnn.cpp:180`) — 호스트 CPU busy-wait.

---

## 9. 우리 프로젝트(ViT/Transformer FPGA + XR 시선추적) 시사점

- **단일 다용도 IP + 호스트 스케줄러 패턴**(`acc_sim:212-371`)은 ViT의 이종 연산(attention GEMM, FFN GEMM, LayerNorm, softmax)을 **소수 범용 PE로 시분할** 처리하는 우리 HG-PIPE류 설계에 직접 참고. layer-type 인자화 + 비트필드 파라미터 패킹(`cnn_hls:624-631`)은 AXI-Lite 레지스터 절감 기법으로 채택 가능.
- **레이어별 동적 fixed-point(WeightQ/InputQ/OutputQ)**(`cnn_hls:651-657`, `acc_sim:297-299`)는 transformer의 레이어/헤드별 스케일 차이가 큰 양자화에 매우 적합 — 동적 시프트량을 레지스터로 받는 구조 그대로 이식 권장. 단 ViT는 LayerNorm/softmax의 동적 통계가 있어 conv-BN처럼 정적 흡수는 불가(주의).
- **3중 ping-pong 오버랩**(`cnn_hls:705-722`)은 attention의 K/V 로드 vs QK^T 연산 vs 출력 write 오버랩 설계의 검증된 템플릿. `DEPENDENCE inter false`로 누산 II=1 달성(`cnn_hls:421`)도 GEMM 누산기에 동일 적용 가능(단 stall 위험 주의 — `README.md:9`).
- **HW/SW 분할 원칙**: 규칙적 MAC(conv/GEMM)은 FPGA, 불규칙·저빈도 연산(reorg/route/region; ViT라면 softmax 정규화·token 재배열·NMS)은 호스트 CPU로 — XR 시선추적의 경량 후처리는 호스트가 효율적일 수 있음(추정).
- **DRAM burst-aware 데이터 레이아웃**(타일 연속 배치 + 짝/홀 ping-pong DRAM 버퍼, `acc_sim:36-95`, `README.md:44`)은 ViT의 token×dim 텐서를 DRAM에 둘 때 그대로 유효. attention의 reorg 격인 head split/merge 주소를 "미리 배치"하는 route 기법(연산 없이 주소만)은 데이터무브 절감에 응용.
- **128bit 버스 + LANE_NUM 벡터화**(`acc_i16c:519-520`)는 우리 INT4/INT8 transformer에서 한 트랜잭션에 다채널 적재하는 방식의 참고. AnyPackingNet의 DSP-packing(SIMD<<wbit)과 결합하면 버스폭+DSP 양쪽 이용률을 동시 최적화(추정).
- **반면교사**: C-RTL co-sim 미구현·하드코딩 오프셋·버전 민감 pragma는 우리가 피해야 할 지점. 골든 비교(SW int 모델 == HLS C-sim == RTL co-sim)를 처음부터 갖추고, 네트워크 토폴로지를 파라미터화(하드코딩 금지)할 것.
