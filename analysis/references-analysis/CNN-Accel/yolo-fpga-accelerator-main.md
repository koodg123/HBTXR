# yolo-fpga-accelerator-main 정밀 분석

> 분석 대상 경로: `REF/CNN-Accel/yolo-fpga-accelerator-main`
> 모든 라인 근거는 `파일:라인` 형식. 라인 근거 없는 서술은 "추정"/"확인 불가"로 명시.

---

## 1. 개요 (목적 / 원논문 추정 / 타깃보드)

- **목적**: YOLOv2(416x416, COCO 80-class)를 **모델→가중치추출/INT16 양자화→HLS 가속기→Vivado 통합→KV260 배포→Linux 유저스페이스 추론**까지 end-to-end 재현하는 풀스택 가속기 (`README.md:1-6`).
- **원논문**: 특정 논문이 아니라 **Darknet YOLOv2 + 타일링 기반 HLS 가속기**의 엔지니어링 구현으로 추정. 코드 구조(`Tn/Tm/Tr/Tc` 타일, ping-pong, OnChipIB)는 전형적인 "loop tiling CNN accelerator"(Zhang et al. FPGA'15 계열) 패턴 → "방법론 차용(추정)". 가중치 추출은 외부 `nn-weight-extractor` 사용(`README.md:9`).
- **타깃보드**: AMD Kria **KV260**, Vivado/Vitis HLS **2024.2** (`README.md:1,21,23`). HLS IP 이름 `xilinx.com:hls:YOLO2_FPGA:1.0`(`README.md:71`). 제어 베이스 `0xA0000000`(`linux_app/include/yolo2_config.h:18`).
- **정밀도**: FP32(호스트 검증용)와 **INT16 고정소수(KV260 배포용)** 듀얼 패스. `Precision{FP32,INT16}`(`include/core/precision.hpp:6-9`), 컴파일 스위치 `INT16_MODE`(`hls/core/types.hpp:8-14`). KV260 검증은 **YOLOv2 INT16**(`README.md:5`).

---

## 2. 디렉토리 구조 (자체 포함 + 제외 항목 이유)

### 자체 핵심 소스
- `hls/core/` — **모델 비종속 가속기 빌딩블록**:
  - `core_compute.cpp/.hpp` — PE 어레이(MAC), 풀링, 활성화, 출력 라이트백.
  - `core_io.cpp/.hpp` — IFM/가중치/bias DRAM↔온칩 로드(256bit 정렬 버스트).
  - `core_scheduler.cpp/.hpp` — 채널 ping-pong 래퍼.
  - `types.hpp` — IO/Acc dtype 별칭.
- `hls/models/yolov2/` — YOLOv2 래퍼:
  - `yolo2_accel.cpp/.hpp` — 톱 함수 `YOLO2_FPGA`(타일 루프 오케스트레이션).
  - `yolo2_model.cpp` — 호스트측 레이어별 스케줄러 `yolov2_hls_ps`.
  - `model_config.cpp/.hpp` — 레이어별 가중치/bias 오프셋 테이블, 메모리 길이.
  - `yolo2_accel_internal.hpp`, `yolov2_acc_pragmas.h`(프래그마 헬퍼).
- `include/core/` — `precision.hpp`, `yolo.h`, `yolo_cfg.hpp`(Darknet cfg 파서 선언).
- `linux_app/src/` — KV260 유저스페이스 앱(추론 오케스트레이션, DMA, V4L2, MJPEG, 후처리).
- `scripts/hw_params_gen.py` — `params.hpp` 생성기. `scripts/run_pipeline.py`(미정독, README 근거).

### 제외 항목(이름만 언급)
- `include/third_party/stb_image.h`, `stb_image_write.h`, `linux_app/.../stb_image*` — **third-party 이미지 IO 라이브러리**. 분석 제외.
- `linux_app/accel_package/yolov2_accel/*.bit.bin`, `*.dtbo`, `*.dtsi` — **비트스트림/디바이스트리 생성물**. 제외.
- `examples/test_images/*.jpg`, `data/labels/*`, `config/*.cfg`, `config/coco.names` — 데이터/설정 입력물.
- `build/`, `.gitignore`, `LICENSE`, `*.md`(README류는 근거로만 인용) — 빌드/메타.
- `weights/*.bin` (본문 참조됨, 저장소엔 미동봉) — 가중치 생성물.

> **주의**: 과제 지시의 `hls/api.hpp`는 `models/yolov2/yolo2_accel.hpp`를 include하는 얇은 진입점일 뿐임(`hls/api.hpp:1-3`). 또한 지시의 `src/core/yolo_*.cpp`, `scripts/run_pipeline.py` 중 **`src/core/` 경로는 저장소에 존재하지 않음**(Glob 확인). 동등 SW는 `linux_app/src/yolo2_*.c`와 `hls/models/yolov2/yolo2_model.cpp`에 위치 → 후자를 분석함("경로 상이, 실제 위치로 대체").

---

## 3. 핵심 모듈 정밀 분석

### 3.0 타일 파라미터(합성 상수) — `params.hpp`(생성물)
`params.hpp`는 `hw_params_gen.py`가 생성하며 저장소에는 없으나(미생성 상태), **기본값**이 스크립트에 하드코딩됨(`scripts/hw_params_gen.py:16-23`):
- `S=2, K=3, MAX_BETA_LENGTH=1024, Tn=4, Tm=32, Tr=13, Tc=13`.
- 파생: `OnChipIB_Width=(Tc-1)*S+K=27`, `OnChipIB_Height=(Tr-1)*S+K=27`(`scripts/hw_params_gen.py:42-43`). Linux 측 동일값(`linux_app/include/yolo2_config.h:97-102`).
- **PE 어레이 규모 = Tm×Tn = 32×4 = 128 MAC**(아래 3.2에서 확인).

### 3.1 자료형 / 양자화 도메인 — `types.hpp`, `precision.hpp`
- INT16 모드: `IO_Dtype=int16_t`, `Acc_Dtype=int32_t`(`hls/core/types.hpp:9-10`). FP32 모드: 둘 다 float(`types.hpp:12-13`).
- per-layer Q 인자 4종: `Qw`(가중치), `Qa_in`(입력활성), `Qa_out`(출력활성), `Qb`(bias). YOLO2_FPGA 인자로 전달(`hls/models/yolov2/yolo2_accel.hpp:10-17`).

### 3.2 PE 어레이 / MAC — `compute()` (`core_compute.cpp:22-173`) **[가장 중요]**

`compute()`가 본 가속기의 **연산 코어(PE 어레이)**다. 두 버전(INT16/FP32)이 `#ifdef INT16_MODE`로 분기.

#### (a) 버퍼 분할로 정의되는 어레이 형상
- `input_buffer[Tn][...][...]` dim=1(채널) 완전분할(`core_compute.cpp:28`).
- `output_buffer[Tm][Tr][Tc]` dim=1(출력채널) 완전분할(`core_compute.cpp:29`).
- `weight_buffer[Tm][Tn][K][K]` dim=1,2 완전분할(`core_compute.cpp:30-31`).
→ **Tm(32)개 출력채널 × Tn(4)개 입력채널이 동시에 하드웨어로 펼쳐짐** = 128 MAC 병렬.

#### (b) bias 캐싱(enable=false 1패스)
`enable==false`면 출력채널 블록 bias를 `local_beta_buffer`에 적재 후 리턴(`core_compute.cpp:36-43` INT16 / `122-129` FP32). 즉 호출 1회로 bias 프리로드, 다음 호출로 계산.

#### (c) 컨볼루션 루프 순서 (output-stationary 누산)
루프: `i(Ksize) → j(Ksize) → tr(TR_MIN) → tc(TC_MIN)`, 최내 `PIPELINE II=1`(`core_compute.cpp:65-74`). 픽셀 위치마다:
- 출력채널 `tm(Tm)` 루프 + `DEPENDENCE output_buffer inter false`(`core_compute.cpp:82-84`).
- 입력채널 `tn(Tn)` 루프로 부분곱 누산(`core_compute.cpp:100-106`).
- **base**: 첫 입력타일(`n==0`)+커널 첫 탭(`i==0&&j==0`)일 때만 bias로 초기화, 그 외 `output_buffer`에서 읽어 누적(`core_compute.cpp:77,86-97`) → **출력 정주(output-stationary)**.

#### (d) INT16 양자화 누산 정밀도 관리 (핵심)
- 시프트 사전계산: `shift_out = Qa_in + Qw - Qa_out`, `shift_bias = Qb - Qa_out`(`core_compute.cpp:48-49`). → **고정소수 도메인 정렬**.
- 곱은 int32로 유지(`weight_val*input_val`), 누산은 int64(`core_compute.cpp:103-106`). 주석: "INT16xINT16 product stays within signed 32-bit"(`core_compute.cpp:102`).
- 라운딩 시프트: 우/좌 시프트 분기 + 라운딩 상수(`+round`)(`core_compute.cpp:53-63,88-113`).
- 출력 INT16 포화 `[-32768,32767]`(`core_compute.cpp:116-117`).
- FP32 버전은 단순 `partial_mul`/`partial_add` float 누산(`core_compute.cpp:133-168`).

#### (e) systolic 여부
- 명시적 PE-to-PE 데이터 전달(시스톨릭 시프트 레지스터)은 **없음**. `ARRAY_PARTITION complete`로 펼친 **공간 병렬 MAC 트리 + II=1 파이프라인**(`core_compute.cpp:28-31,74`). → **"output-stationary spatial array(비-systolic)"로 분류**.

### 3.3 풀링 / reorg / 활성화 — `core_compute.cpp`
- `pool_yolo2()`(`core_compute.cpp:266-305`): KxK 맥스풀, `tmp[Tn]` 분할, `i==1&&j==1`에서 출력 기록(2x2 가정).
- `reorg_yolo2()`(`core_compute.cpp:354-379`): stride-2 space-to-depth(YOLOv2 passthrough). `Output[(ky<<1)+kx][y][x]=Input[0][2y+ky][2x+kx]`.
- `nonlinear_leaky_row()`(`core_compute.cpp:175-210`): Leaky ReLU. INT16은 `tmp/10`(≈0.1, 정수근사), FP32는 `tmp*0.1f`(`core_compute.cpp:194-203`).
- `write_back_output_reorg()`(`core_compute.cpp:222-264`): **더블버퍼(local_buf0/1) ping-pong**으로 활성화↔memcpy를 중첩(`core_compute.cpp:242-255`).

### 3.4 IO 로드 (256bit 정렬 버스트) — `core_io.cpp`
- `input_load()`(`core_io.cpp:82-138`): IFM을 ping-pong 로컬버퍼(`local_buf0/1`)로 행단위 적재, `ifm_mmcpy_row`↔`ifm_copy_lbuf2ibuf` 이중화로 로드/언팩 중첩(`core_io.cpp:108-137`). 8워드(256bit) 정렬 오프셋 처리(`core_io.cpp:22-30`). 패딩 처리: 좌표 범위 밖이면 `pad_value`(maxpool은 -32768/-1024K)(`core_io.cpp:62-70,97-103`).
- `weight_load_reorg()`(`core_io.cpp:140-199`): reorg된 가중치를 `weight_buffer[Tm][Tn][K][K]`로 적재, `Woffset` 누적, 8워드 버퍼 슬라이딩(`core_io.cpp:154-198`). 유효범위 밖은 0패딩(`core_io.cpp:196-197`).
- `copy_input_weight()`(`core_io.cpp:201-219`): 위 둘을 묶고 `TN_MIN=MIN(TN,IFM_num-n)` 경계 계산(`core_io.cpp:209-212`).

### 3.5 채널 ping-pong 스케줄러 — `intra_pingpong_wrapper()` (`core_scheduler.cpp:14-113`)
- 가중치 더블버퍼 `weight_buffer0/1`(`core_scheduler.cpp:21-27`).
- **LayerType==0(conv)**: 입력채널 `n`을 `TN(4)`씩 진행하며 **로드(다음 타일)와 compute(현재 타일)를 ping-pong으로 중첩**(`core_scheduler.cpp:45-61`). 이것이 IO-compute 오버랩의 핵심.
- **LayerType==1(pool)/2(reorg)**: 로드와 pool/reorg를 ping-pong(`core_scheduler.cpp:63-112`).

### 3.6 톱 함수 타일 오케스트레이션 — `YOLO2_FPGA()` (`yolo2_accel.cpp:25-171`)
- AXI m_axi 4버스: Input(depth 6.92M)/Output(5.54M)/Weight(50.9M)/Beta(10761), 버스트 64/128, outstanding 4(`yolo2_accel.cpp:43-46`).
- s_axilite로 IFM_num/OFM_num/Ksize/Kstride/TM/TN/TR/TC/LayerType/Q값 등 전달(`yolo2_accel.cpp:48-73`).
- 온칩: `input_buffer0/1`, `output_buffer/output_buffer1`(**M-방향 ping-pong**, LUTRAM 바인딩)(`yolo2_accel.cpp:103-112`).
- **3중 타일 루프** `r(Output_h step TR) → c(Output_w step TC) → m(OFM step TM)`(`yolo2_accel.cpp:127-168`).
- `m` 루프에서 `pingpongm`으로 compute(다음 m타일)↔write_back(이전 m타일) 중첩(`yolo2_accel.cpp:148-166`).
- `input_flag/process_flag/write_flag`로 파이프라인 워밍업/드레인 제어(`yolo2_accel.cpp:144-146`).

### 3.7 호스트 레이어 스케줄러 — `yolov2_hls_ps()` (`yolo2_model.cpp:229-449`) **[HW/SW 핵심]**
- **per-layer로 `YOLO2_FPGA`를 반복 호출**하여 전체 YOLOv2를 구성(레이어 단위 오프로딩).
- 메모리 더블버퍼 배치 `generate_iofm_offset()`(`yolo2_model.cpp:56-110`): Memory_top/bottom 사이를 짝/홀 레이어가 핑퐁 사용(`yolo2_model.cpp:67-95`). route16(skip) 영역 별도 확보(`yolo2_model.cpp:89,97-101`).
- CONVOLUTIONAL: 타일 `TR/TC/TM/TN` 계산(OnChipIB·Tr/Tc·l.n/l.c로 클램프)(`yolo2_model.cpp:303-309`), `mLoops` 계산 후 `YOLO2_FPGA` 호출(`yolo2_model.cpp:322-326`). INT16 시 per-layer Q 인덱싱(`yolo2_model.cpp:312-321`).
- REORG: 호스트에서 `reorg_cpu`로 space-to-depth 후, **INT16 스킵연결 스케일 정렬**(`route24_q`와 `current_Qa` min, shift)(`yolo2_model.cpp:358-401`). → 양자화 도메인 일치 시키는 중요한 디테일.
- REGION: 최종 출력 dequant(`*2^-q_out`) 후 `forward_region_layer`(`yolo2_model.cpp:406-441`).

### 3.8 모델 상수 테이블 — `model_config.cpp`
- 23개 conv 레이어의 가중치 오프셋(`kYolo2WeightOffsets`)/bias 오프셋(`kYolo2BetaOffsets`)(`model_config.cpp:4-10`).
- `mem_len=6,922,240`(=416²·32+208²·32), route16/conv27/conv24 길이, detection_workspace(`model_config.cpp:18-27`). Linux 측 동일 테이블 복제(`linux_app/src/yolo2_inference.c:30-40`).

### 3.9 KV260 런타임 매핑 — `linux_app/include/yolo2_config.h`
- 제어 레지스터 맵: `AP_CTRL=0x00`, Input/Output/Weight/Beta 64bit 주소 레지스터(0x10/0x1c/0x28/0x34), 파라미터 레지스터(IFM_NUM 0x40 … LAYER_TYPE 0xd0)(`yolo2_config.h:38-75`).
- **Q값은 별도 AXI GPIO로 전달**(`AXI_GPIO_QW/QA_IN/QA_OUT/QB_BASE` 0xA0010000~0xA0040000)(`yolo2_config.h:19-22,77-78`) → CTRL_BUS에 Q 레지스터 없음(HLS IP 제약).

---

## 4. 데이터플로우

```
[호스트(PS) yolov2_hls_ps]  (yolo2_model.cpp:294-446)
  레이어 루프:
   CONV  → 타일 계산 → YOLO2_FPGA(in_ptr[i],out_ptr[i],W+woff,Beta+boff, Q...)
   MAXPOOL/REORG → YOLO2_FPGA(LayerType=1/2) or reorg_cpu
   ROUTE → skip(주소 핑퐁) ; REGION → dequant+디코딩
        │ (in_ptr/out_ptr Memory_top/bottom 핑퐁)
        ▼
[가속기(PL) YOLO2_FPGA]  (yolo2_accel.cpp:127-168)
  for r(TR) for c(TC) for m(TM):
     intra_pingpong_wrapper ──► (LayerType 분기)
        conv: n(TN) 핑퐁 [ copy_input_weight ‖ compute ]   (core_scheduler.cpp:45-61)
           load: input_load + weight_load_reorg (256b burst) (core_io.cpp)
           compute: Tm×Tn=128 MAC, output-stationary, INT16 Q정렬 (core_compute.cpp:65-118)
        pool/reorg: pool_yolo2 / reorg_yolo2
     write_back_output_reorg: leaky+memcpy 더블버퍼 → DDR  (core_compute.cpp:222-264)
```

- **3중 중첩 ping-pong**: (1) 채널 n방향(입력/가중치 로드 ↔ compute), (2) m방향(compute ↔ write_back), (3) write_back 내부(leaky ↔ memcpy). → IO 지연 은폐가 설계의 핵심.

---

## 5. HW/SW 매핑

| 기능 | HW(PL) | SW(PS) |
|---|---|---|
| conv MAC | `compute()` (core_compute.cpp:22) | — |
| pool/reorg | `pool_yolo2`/`reorg_yolo2` (core_compute.cpp:266,354) | reorg는 호스트 `reorg_cpu`도 사용 (yolo2_model.cpp:373) |
| 타일 루프 | `YOLO2_FPGA` (yolo2_accel.cpp:127) | 레이어 루프 `yolov2_hls_ps` (yolo2_model.cpp:294) |
| 양자화 정렬 | compute 내 shift (core_compute.cpp:48-63) | per-layer Q 인덱싱·route 정렬 (yolo2_model.cpp:312-401) |
| DMA/제어 | AXI m_axi+s_axilite (yolo2_accel.cpp:43-73) | mmap 레지스터/udmabuf (yolo2_config.h, linux_app/src) |
| 후처리(NMS/디코딩) | — | REGION/postprocess (yolo2_model.cpp:406, linux_app/src/yolo2_postprocess.c) |

- **경계 특징**: PL은 "범용 conv/pool/reorg 타일 엔진" 1개를 레이어마다 재사용(가중치 재로드). 호스트가 레이어 순서·메모리배치·Q스케일을 전담 → **유연하지만 레이어 간 DRAM 왕복 비용 존재**.

---

## 6. 빌드 · 실행

- **호스트 검증**: `make test`(params 자동생성) → `make gen && ./yolov2_weight_gen` → `./yolov2_detect --backend hls`(`README.md:38-51`). INT16: `make test-int16`(`README.md:53-59`).
- **params 생성**: `python3 scripts/hw_params_gen.py [--tn --tm --tr --tc --stride --kernel]` → `hls/core/params.hpp` + `linux_app/include/yolo2_config.h` 동기화(`scripts/hw_params_gen.py:97-171`).
- **HLS IP**: `vitis-run --mode hls --tcl vitis/yolo2_int16_cli.tcl` → `yolo2_int16/solution1/impl/ip`(`README.md:73-79`).
- **Vivado**: `vivado/build_from_bd.sh --bd-tcl ... --ip-repo ...`(`README.md:88-95`).
- **KV260 패키징/배포**: `create_accel_package.sh`, `deploy_to_kv260.sh`, `start_yolo.sh -i image`(`README.md:107-146`).
- **자동화**: `scripts/run_pipeline.py --config pipeline.local.yaml`(`README.md:179-184`).

---

## 7. 의존성

- HLS: `ap_int.h`/Xilinx HLS, 프래그마 헬퍼 `yolov2_acc_pragmas.h`(`core_compute.cpp:1-8`, `include/models/yolov2/yolov2_acc_pragmas.h:19-22`).
- 호스트(C++): `<filesystem>`, Darknet 호환 `network/layer`(`include/core/yolo.h`)(`yolo2_model.cpp:22-31`).
- Linux 앱(C): POSIX mmap, udmabuf, V4L2, ffmpeg(옵션), MJPEG 서버(`README.md:149-152`, `linux_app/src/*`).
- 이미지 IO: third_party stb_image(제외).
- 외부: `nn-weight-extractor`(가중치/Q 생성, 제외)(`README.md:9`).

---

## 8. 강점 · 한계

**강점**
- **모델 비종속 코어(core/) + 모델 래퍼 분리** 아키텍처 → 재사용성 높음(`hls/core` vs `hls/models/yolov2`).
- 3중 ping-pong으로 IO-compute 중첩(`core_scheduler.cpp:45-61`, `yolo2_accel.cpp:148-166`, `core_compute.cpp:242-255`).
- INT16 per-layer Q 도메인 정렬 + 라운딩 시프트(`core_compute.cpp:48-117`), 스킵연결 스케일 일치(`yolo2_model.cpp:379-399`) → 정확도 보존 설계.
- 256bit 정렬 버스트 로드(`core_io.cpp:22-35`)로 DRAM 대역폭 활용.
- end-to-end 재현 문서/스크립트 완비(`README.md`, `hw_params_gen.py`, `run_pipeline.py`).

**한계**
- PE 어레이가 **128 MAC(Tm32×Tn4)로 비교적 소규모**(`hw_params_gen.py:19-20`) → 고해상도/대모델 처리량 제약.
- 레이어 단위 오프로딩 → **레이어 간 DRAM 왕복**(스트리밍 dataflow 대비 대역폭/지연 손해). reorg를 호스트 CPU로 처리(`yolo2_model.cpp:373`).
- systolic 아님: `complete` 분할 기반 공간 어레이라 Tm/Tn 확대 시 라우팅/팬아웃 부담(추정).
- INT16(16비트)로 메모리·DSP 비용이 INT8/INT4 대비 큼(가중치 ~97MB)(`yolo2_config.h:90`).
- `params.hpp` 미동봉(생성 필요) → 빌드 전 정적 상수 부재.

---

## 9. 우리 프로젝트 시사점 (ViT/HG-PIPE + XR 시선추적 관점)

- **dataflow 관점**: 본 repo의 "범용 타일 엔진 + 호스트 레이어 루프"는 ViT처럼 레이어 구조가 균일(반복 인코더 블록)한 경우 **재사용에 유리**. 다만 HG-PIPE의 강점인 **완전 파이프라인(레이어 간 온칩 스트리밍)** 과 달리 레이어마다 DRAM 왕복이 있어, 우리 ViT 가속기는 **인코더 블록을 온칩 융합**하는 방향이 더 적합(본 repo는 대조 레퍼런스).
- **양자화 재사용 관점(가장 직접적)**: per-layer `Qw/Qa_in/Qa_out/Qb` 4-파라미터 도메인 정렬과 `shift_out=Qa_in+Qw-Qa_out` 사전계산(`core_compute.cpp:48-49`)은 **ViT 선형층/QKV에 그대로 차용 가능**. 특히 residual/skip 연결의 스케일 일치 로직(`yolo2_model.cpp:379-399`)은 ViT residual add 전 스케일 정렬에 직결 → 채택 권장.
- **systolic 관점**: 본 repo는 비-systolic 공간 MAC. ViT의 큰 GEMM(QK^T, attn·V, MLP)에는 **systolic/output-stationary 어레이가 더 효율적**. 본 repo의 output-stationary 누산 패턴(`core_compute.cpp:77,86-118`)은 systolic 설계 시 누산 제어 참고용.
- **HW/SW 분담 관점**: 호스트가 레이어 스케줄·메모리배치·Q관리를 전담하는 구조(`yolo2_model.cpp:294-446`)는 ViT의 LayerNorm/Softmax 같은 **비-GEMM 연산을 PS로 빼는 분담 설계**의 좋은 템플릿. 시선추적 저지연 요구상, Softmax/LN은 가능한 PL 내 처리해 PS 왕복을 줄이는 게 우리 방향(본 repo와 반대 선택이 필요).
- **재사용 가능 모듈**: (1) `hw_params_gen.py`식 **타일 파라미터 자동생성 + 헤더 동기화** 흐름(`scripts/hw_params_gen.py:66-94`)은 ViT의 `D_model/heads/seq_len` 파라미터화에 그대로 응용. (2) 3중 ping-pong IO 은폐 패턴은 ViT 가중치 로드(MLP가 큼)에 필수.
- **시선추적 특화**: 본 repo의 V4L2 카메라 + MJPEG 스트리밍 인프라(`linux_app/src`)는 **XR 시선추적의 입력 카메라 파이프라인 레퍼런스**로 직접 재활용 가능(추정, 코드 상세 미정독 부분 존재).
