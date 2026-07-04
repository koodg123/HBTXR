# Kria-YOLOv4-Tiny-FPGA-Accelerator 정밀 분석

> 분석 대상 경로: `REF/CNN-Accel/Kria-YOLOv4-Tiny-FPGA-Accelerator`
> 모든 라인 근거는 `파일:라인` 형식으로 표기. 라인 근거 없는 서술은 "추정"/"확인 불가"로 명시.

---

## 1. 개요 (목적 / 원논문 추정 / 타깃보드)

- **목적**: YOLOv4-Tiny 객체검출의 **연산집약 컨볼루션 1개 레이어**를 ARM(PS)에서 FPGA(PL)로 오프로딩하여 CPU 대비 가속을 시연하는 HW/SW 코디자인 학부/연구 프로젝트. (`README.md:6`, `README.md:8`)
- **원논문**: 별도 학술 논문 기반이 아니라 **YOLOv4-Tiny(Darknet) 모델 + Vitis HLS 커스텀 가속기**의 교육용 구현으로 추정. README에 Chennai Institute of Technology 팀 명시(`README.md:198-204`), `docs/FPGA_YOLO_Accelerator_Final_Report.pdf` 최종보고서 존재(`README.md:175`). 학술 인용 없음 → "원논문 없음(추정)".
- **타깃보드**: AMD Kria **KV260**, 디바이스 `xck26-sfvc784-2LV-c`, Vitis Kernel Flow, XRT 런타임, OpenCL 프로그래밍 모델 (`README.md:28-33`).
- **수치(README 자체 보고)**: LUT 18% / FF 8% / BRAM 13% / DSP 13% (`README.md:49-54`); CPU 850ms→1.18FPS, FPGA 368ms→2.7FPS, 2.31x speedup (`README.md:64-84`). 이 수치는 코드로 재현 불가 → "보고값(검증 불가)".
- **정밀도**: INT8 연산 (`README.md:44`, `cnn_accelerator.cpp:3`).

---

## 2. 디렉토리 구조 (자체 포함 + 제외 항목 이유)

### 자체 핵심 소스(분석 대상)
- `hardware/HLS/cnn_accelerator.cpp` — HLS 커널 톱 함수 `yolo_conv_kv260` (단일 3x3 conv).
- `hardware/HLS/testbench.cpp` — C 시뮬레이션 테스트벤치(골든 레퍼런스 비교).
- `software/fpga_arm.cpp` — ARM(PS) 측 `/dev/mem` mmap 기반 베어메탈식 드라이버.
- `software/cpu_baseline/yolo4tiny_cpu.cpp` — OpenCV DNN 기반 CPU 베이스라인.
- `README.md` — 프로젝트 개요/빌드.

### 제외 항목(이름만 언급, 분석 제외)
- `.git/**` — 버전관리 메타데이터.
- `hardware/kernal/YOLO_wrapper.bit`, `YOLO_wrapper.xsa` — **생성물(비트스트림/하드웨어 핸드오프)**. 바이너리, 코드 근거 분석 대상 아님.
- `hardware/VIVADO_PROJECT/*.png`, `docs/*.jpeg`, `docs/*.pdf` — 이미지/문서 산출물.
- `software/cpu_baseline/yolov4-tiny.weights`, `*.cfg`, `coco.names`, `y4t_cpu.exe` — **외부 Darknet 사전학습 가중치/설정/실행 바이너리(vendor 생성물)**. 분석 제외.

---

## 3. 핵심 모듈 정밀 분석

### 3.1 HLS 커널 톱 함수 `yolo_conv_kv260` (`cnn_accelerator.cpp:25-160`)

이 커널은 **YOLOv4-Tiny 전체가 아니라 단일 3x3 컨볼루션 1개 레이어**만 가속한다. 고정 형상은 다음과 같다(`cnn_accelerator.cpp:9-16`):
- 입력 `H=W=26`, `CIN=128`, 출력 `COUT=128`, 커널 `K=3`, 출력 `OH=OW=24`(`cnn_accelerator.cpp:18-19`).
- 타일: `CI_TILE=8`(입력채널), `CO_TILE=4`(출력채널) (`cnn_accelerator.cpp:15-16`).

#### (a) 인터페이스 / 메모리맵
- AXI4 master 4개 번들: `in_fm`/`out_fm`/`weights`/`bias`를 각각 `gmem0~gmem3`로 분리(`cnn_accelerator.cpp:31-34`). `depth` 명시: in/out 86528, weights 147456(=128·128·3·3), bias 128.
- 제어는 `s_axilite port=return`(`cnn_accelerator.cpp:35`).

#### (b) 온칩 버퍼 구조 (라인버퍼 방식)
- `linebuf[K][W][CI_TILE]` — **3행 라인버퍼**(슬라이딩 윈도우용). dim=1(행), dim=3(채널) 완전분할(`cnn_accelerator.cpp:40-42`).
- `window[K][K][CI_TILE]` — 현재 3x3 윈도우, 완전분할(`cnn_accelerator.cpp:44-45`).
- `wbuf[CO_TILE][CI_TILE][K][K]` — 가중치 타일, dim=1,2 완전분할(`cnn_accelerator.cpp:47-49`).
- `acc[CO_TILE]` — 출력채널별 누산기, 완전분할(`cnn_accelerator.cpp:54-55`).

#### (c) 레이어 매핑 / 루프 구조 (출력채널-입력채널 타일링)
이중 외부 루프로 출력/입력 채널을 타일링한다(`cnn_accelerator.cpp:70-71`):
- `for co0 in 0..COUT step CO_TILE` → `for ci0 in 0..CIN step CI_TILE`.
- 가중치 로드: 4중 루프로 `wbuf`에 적재, `PIPELINE II=1`(`cnn_accelerator.cpp:76-86`). 인덱스 계산 `(co0+co)*CIN*K*K + (ci0+ci)*K*K + ky*K+kx`(`cnn_accelerator.cpp:81-85`) → **가중치 레이아웃은 OIHW(out,in,ky,kx)**.

#### (d) 슬라이딩 윈도우 / im2col 등가 처리
명시적 im2col 버퍼는 없고, **라인버퍼 + 윈도우 재구성**으로 컨볼루션을 수행한다:
- 한 행 입력 적재: `linebuf[row_ptr][x][ci] = in_fm[(y*W+x)*CIN + (ci0+ci)]`(`cnn_accelerator.cpp:97-104`) → **입력 텐서 레이아웃은 HWC(채널 마지막)**.
- 윈도우 유효 조건 `y >= K-1`에서만 계산(`cnn_accelerator.cpp:108`).
- 윈도우 재구성: 회전 라인버퍼 인덱싱 `r = row_ptr + K - ky; if(r>=K) r-=K`로 **모듈로 없이** 행 선택(`cnn_accelerator.cpp:113-120`). 주석에 "NO modulo on RAM" 명시(`cnn_accelerator.cpp:112`).
- 라인버퍼 회전: `row_ptr++; if(row_ptr==K) row_ptr=0`(`cnn_accelerator.cpp:154-156`).

#### (e) PE 어레이 / MAC 구조 (병렬도 분석)
- 공간 루프 `for x` 에 `PIPELINE II=4`(`cnn_accelerator.cpp:109-110`).
- 출력채널 루프 `for co (CO_TILE)`는 `UNROLL`(`cnn_accelerator.cpp:123-124`) → CO_TILE=4 병렬.
- 입력채널 루프 `for ci (CI_TILE)`는 `UNROLL factor=2`(부분 언롤, "safe partial unroll" 주석)(`cnn_accelerator.cpp:128-129`).
- 내부 MAC: `acc[co] += window[ky][kx][ci] * wbuf[co][ci][ky][kx]`(`cnn_accelerator.cpp:130-134`).
- **병렬 MAC 추정**: CO 완전언롤 4 × CI 부분언롤 2 = **약 8개 곱셈기 동시**(K×K=9는 미언롤). 이것이 systolic array가 아니라 **언롤+파이프라인 기반 단순 MAC 데이터패스**임을 보여줌(systolic 아님).

#### (f) 양자화 / 누산 / 출력 처리
- `ci0==0`일 때만 bias로 누산 초기화(`cnn_accelerator.cpp:125-126`).
- 마지막 입력채널 타일(`ci0==CIN-CI_TILE`)에서만 출력 기록(`cnn_accelerator.cpp:138`).
- **양자화 스케일**: `out = acc >> 8` (우측시프트 8, 고정 스케일) 후 INT8 포화 `[-128,127]`(`cnn_accelerator.cpp:139-141`). → per-tensor 동적 스케일이 아니라 **하드코딩된 8비트 시프트**. 학습된 스케일과 무관 → 실제 검출 정확도에 대한 근거 없음("확인 불가").
- 출력 인덱스 `((y-(K-1))*OW + (x-(K-1)))*COUT + (co0+co)`(`cnn_accelerator.cpp:143-148`) → **출력도 HWC**.

### 3.2 테스트벤치 `golden_conv` / `main` (`testbench.cpp`)
- 골든 레퍼런스 `golden_conv`(`testbench.cpp:40-80`): 동일 형상의 6중 루프 직접 컨볼루션. 입력 인덱스 `((y+ky)*W+(x+kx))*CIN+ci`(HWC), 가중치 `co*CIN*K*K+ci*K*K+ky*K+kx`(OIHW), 누산 후 `acc>>8` + 포화(`testbench.cpp:64-76`) → **커널과 동일한 양자화 규약** 검증.
- 결정론적 입력 생성: `in_fm[i]=(i%13)-6`, `weights[i]=(i%7)-3`, `bias[i]=i%16`(`testbench.cpp:101-108`).
- HW/REF 비트일치 비교 후 PASS/FAIL(`testbench.cpp:128-143`).

### 3.3 ARM 드라이버 `main` (`fpga_arm.cpp`)
- **XRT/OpenCL이 아닌 베어메탈식 `/dev/mem` mmap 직접 제어**(README의 OpenCL 모델과 불일치 → 두 가지 호스트 경로 존재 추정).
- FPGA 제어 베이스 `0xA0000000`, 맵 크기 `0x10000`(`fpga_arm.cpp:13-14`).
- HLS 자동생성 레지스터 오프셋: `REG_CTRL=0x00`, in/out/wgt/bias 주소 레지스터 0x10~0x38(`fpga_arm.cpp:16-24`).
- **주의(잠재 버그)**: 물리주소로 가상 배열 주소를 그대로 사용 `in_phys=(uint64_t)in_fm`(`fpga_arm.cpp:73-76`). 주석에 "assumes identity mapping ... For production, use CMA or u-dma-buf"라고 한계 명시(`fpga_arm.cpp:28-31`). → 실디바이스에서 DMA가 동작하려면 물리연속 버퍼 필요. 현 코드로는 정상 DMA 보장 불가("확인 불가").
- 실행: `ap_start`=1 후 `ap_done` 폴링 `while((CTRL&0x2)==0)`(`fpga_arm.cpp:93-94`), 시간측정(`fpga_arm.cpp:91-99`).

### 3.4 CPU 베이스라인 `yolo4tiny_cpu.cpp`
- **OpenCV DNN(Darknet 리더) 기반** 전체 YOLOv4-Tiny 추론. `readNetFromDarknet(cfg,weights)`, CPU 타깃(`yolo4tiny_cpu.cpp:40-42`).
- 입력 320, conf 0.35, NMS 0.45(`yolo4tiny_cpu.cpp:50-52`), `blobFromImage`로 1/255 스케일(`yolo4tiny_cpu.cpp:64`).
- 검출 파싱+NMS(`yolo4tiny_cpu.cpp:79-114`), 전/추론/후처리 시간 분리 측정(`yolo4tiny_cpu.cpp:121-133`).
- 이 베이스라인은 **HLS 커널과 직접 비교 불가**(전체 모델 vs 단일 레이어). README의 2.31x는 시스템 레벨 비교값으로 추정.

---

## 4. 데이터플로우

```
[CPU 베이스라인 경로]
 카메라 → blobFromImage(1/255,320) → OpenCV DNN forward → NMS → 화면 (yolo4tiny_cpu.cpp)

[FPGA 가속 경로 (단일 conv 레이어)]
 DDR(in_fm HWC, weights OIHW, bias)
   → AXI m_axi 버스트 read (gmem0~3)
   → linebuf[3][26][8] 행단위 적재 (cnn_accelerator.cpp:97-104)
   → window[3][3][8] 슬라이딩 재구성 (cnn_accelerator.cpp:113-120)
   → MAC: CO언롤4 × CI언롤2 (cnn_accelerator.cpp:123-134)
   → acc>>8 + INT8 포화 (cnn_accelerator.cpp:139-141)
   → out_fm HWC write (cnn_accelerator.cpp:148)
   → DDR
 호스트: /dev/mem mmap → 주소레지스터 write → ap_start → ap_done 폴링 (fpga_arm.cpp)
```

- 채널 타일 누산은 **출력버퍼(acc) 정주(output-stationary)**: `ci0` 루프 동안 acc 유지, 마지막 타일에서 flush(`cnn_accelerator.cpp:125-148`).

---

## 5. HW/SW 매핑

| 구분 | 위치 | 근거 |
|---|---|---|
| HW(PL): 단일 3x3 conv | `cnn_accelerator.cpp` `yolo_conv_kv260` | `cnn_accelerator.cpp:25` |
| SW(PS) 드라이버: mmap 제어 | `fpga_arm.cpp` | `fpga_arm.cpp:44-110` |
| 검증(SW): 골든 비교 | `testbench.cpp` | `testbench.cpp:117-123` |
| 전체 모델(SW): CPU 추론 | `yolo4tiny_cpu.cpp` | `yolo4tiny_cpu.cpp:40-71` |

- **경계**: PL은 conv 1개만, 나머지 레이어/전처리/NMS/디코딩은 모두 PS(SW). 즉 YOLOv4-Tiny의 극히 일부만 HW화. 전체 파이프라인 통합 코드는 저장소 내 확인 불가("확인 불가").
- 레지스터 맵(`fpga_arm.cpp:16-24`)은 HLS가 생성하는 `xyolo*_hw.h`와 일치해야 하나 해당 헤더는 저장소에 없음("확인 불가").

---

## 6. 빌드 · 실행

- **HLS 커널**: Vitis HLS에서 합성 → RTL 커널(.xo) export → 링크하여 .xclbin 생성(`README.md:133-136`).
- **호스트(OpenCL 경로, README)**: `g++ host.cpp -o host -I/opt/xilinx/xrt/include -lOpenCL ...`(`README.md:147-151`), 실행 `./host binary_container_1.xclbin`(`README.md:160`). 단, 저장소의 실제 호스트는 `fpga_arm.cpp`(mmap)로 README와 불일치.
- **CPU 베이스라인**: OpenCV 의존, cfg/weights/names 필요(`yolo4tiny_cpu.cpp:34-38`). 빌드 스크립트는 저장소 내 미확인("확인 불가").
- 요구사항: Vitis 2023.x, Vivado, XRT, OpenCL, Ubuntu(`README.md:120-125`).

---

## 7. 의존성

- HLS: `ap_int.h` (Xilinx 고정소수 타입)(`cnn_accelerator.cpp:1`, `testbench.cpp:5`).
- 호스트 드라이버: POSIX `fcntl/sys/mman/unistd`, C++ `chrono`(`fpga_arm.cpp:1-7`).
- CPU 베이스라인: **OpenCV(opencv2/opencv.hpp, dnn)**(`yolo4tiny_cpu.cpp:1-2`).
- 외부 모델: Darknet YOLOv4-Tiny cfg/weights(vendor, 제외).

---

## 8. 강점 · 한계

**강점**
- 라인버퍼+회전 인덱싱으로 BRAM 효율적, 모듈로 제거(`cnn_accelerator.cpp:112-120`).
- 출력정주 채널타일 누산으로 부분합 DRAM 왕복 제거(`cnn_accelerator.cpp:125-148`).
- 골든 비트일치 TB로 기능검증 체계 보유(`testbench.cpp`).

**한계**
- **단일 레이어만 HW화** → 전체 모델 가속 아님. 확장성 미검증.
- 양자화가 `>>8` 고정시프트로 **학습된 스케일/zero-point 미반영**(`cnn_accelerator.cpp:139`) → 정확도 보장 불가.
- 병렬도 낮음: CO4×CI2 ≈ 8 MAC, II=4(`cnn_accelerator.cpp:110,123-129`). DSP 13%만 사용.
- 드라이버가 가상=물리 주소 가정(`fpga_arm.cpp:73-76`, 주석 인정) → 실배포 위험.
- README(OpenCL) ↔ 실코드(mmap) 불일치.

---

## 9. 우리 프로젝트 시사점 (ViT/HG-PIPE + XR 시선추적 관점)

> 우리 프로젝트는 ViT/Transformer FPGA 가속기(HG-PIPE 계열)+XR 시선추적으로 추정. 본 repo는 **가장 단순한 학부형 CNN 가속기 레퍼런스**로, 우리 설계의 "안티패턴 대조군"으로 가치가 있다.

- **systolic/dataflow 관점**: 본 repo는 systolic이 아닌 언롤+파이프라인 단일 PE 데이터패스(`cnn_accelerator.cpp:123-134`). HG-PIPE의 완전 파이프라인 dataflow와 대비됨 → 우리는 레이어 단위 오프로딩이 아니라 **레이어 융합/스트리밍 dataflow**(repo 3 UltraNet식)로 가야 함을 재확인.
- **양자화 재사용 관점**: `>>8` 고정시프트(`cnn_accelerator.cpp:139`)는 **금지 패턴**. ViT는 LayerNorm/Softmax로 동적 레인지가 커서 per-tensor 학습 스케일이 필수. 단, "마지막 채널타일에서만 출력+포화" 누산 패턴(`cnn_accelerator.cpp:138-148`)은 채널 타일링 시 재사용 가능.
- **HW/SW 경계 관점**: PL=conv1개, PS=나머지의 극단적 분할은 **ViT 가속에 부적합**(전처리·디코딩 PS 병목). 시선추적은 저지연이 핵심이므로 본 repo의 `ap_done` 폴링/단일레이어 모델을 그대로 차용하면 안 됨.
- **재사용 가능 요소**: (1) 라인버퍼 회전 인덱싱(`cnn_accelerator.cpp:113-120`)은 ViT 패치 임베딩 conv(stem)에 적용 가능. (2) 골든 비트일치 TB 구조(`testbench.cpp`)는 우리 HLS 검증 템플릿으로 채택 권장.
- **시선추적 특화**: 입력 26x26x128 같은 소형 텐서를 단일 레이어로 빠르게 처리하는 구조는, 시선추적 후단 소형 회귀헤드(예: gaze regression)에는 부분적으로 유효할 수 있음(추정).
