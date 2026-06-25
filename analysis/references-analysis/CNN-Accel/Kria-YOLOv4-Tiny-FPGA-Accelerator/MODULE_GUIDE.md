# Kria-YOLOv4-Tiny-FPGA-Accelerator 모듈 통합 가이드

> 1차 요약: [`../Kria-YOLOv4-Tiny-FPGA-Accelerator.md`](../Kria-YOLOv4-Tiny-FPGA-Accelerator.md) — 본 문서는 그 요약을 모듈 단위로 심화한 통합 가이드다.
> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\CNN-Accel\Kria-YOLOv4-Tiny-FPGA-Accelerator`
> 작성 원칙: 실제 소스 Read 후 `파일:라인` 근거 표기. 라인 근거 없는 추론은 "추정", 코드로 확인 불가는 "확인 불가"로 명시.
> **본 repo의 분석 가치**: 형제 가이드(TATAA/ESDA/UltraNet)와 달리 이 repo는 **안티패턴 대조군**이다. 단일 conv 1개만 오프로딩, `>>8` 고정시프트 양자화, ~8 MAC의 빈약한 병렬도 — 비효율 지점을 정량 근거로 드러내는 데 초점을 둔다.

---

## 0. 문서 머리말

### 0.1 대표 케이스 선정
- **대표 케이스: 단일 3×3 conv 레이어 `yolo_conv_kv260`** (`hardware/HLS/cnn_accelerator.cpp:25-160`). 입력 `H=W=26, CIN=128`, 출력 `COUT=128, OH=OW=24`, `K=3` 고정(`cnn_accelerator.cpp:9-19`). 이 repo는 **이 한 함수가 가속기의 전부**다. YOLOv4-Tiny 전체(~21개 conv 레이어 추정)가 아니라 그중 conv 1개만 PL로 내린다. 따라서 "대표"이자 "유일" 케이스.
- **나머지 SW 경로**: 전처리(`blobFromImage 1/255`)·전체 추론·NMS·디코딩은 전부 PS(ARM/OpenCV). FPGA 경로는 conv 1개에 한정(`fpga_arm.cpp:44-110`, `yolo4tiny_cpu.cpp:33-138`).

### 0.2 수치 표기 규약
- **MAC lanes** = HLS `#pragma HLS UNROLL` 차원 곱. 본 설계는 출력채널 `CO_TILE=4` 완전언롤(`cnn_accelerator.cpp:123-124`) × 입력채널 `factor=2` 부분언롤(`:128-129`) = **8 lanes**. 커널 `K×K=9` 루프는 미언롤(`:130-131`)이라 lanes에 포함 안 됨.
- **scalar MACs**(dense, 단일 레이어) = `OH·OW·COUT·CIN·K·K` = `24·24·128·128·9` = **84,934,656 MAC**(~84.9 MMAC). 골든 레퍼런스 6중 루프(`testbench.cpp:46-67`)와 동형.
- **오프로드 비중** = (가속 레이어 MAC) / (YOLOv4-Tiny 1프레임 전체 MAC). YOLOv4-Tiny@320 전체는 약 3.4 GMAC(외부 일반치, 추정) → 본 레이어 84.9 MMAC은 **약 2.5%**. 즉 모델의 ~2.5%만 HW화(추정 분모, 분자는 코드 확정).
- **loop trips** = 정적 루프 반복수. 외부 채널 타일 = `(COUT/CO_TILE)·(CIN/CI_TILE)` = `32·16` = **512회**, 각 타일 안에서 행 적재·윈도우·MAC 반복.
- **memory size**(payload bit) = 온칩 버퍼 깊이 × 폭(bit). 모두 정적 배열(`cnn_accelerator.cpp:40-55`).
- **타깃**: AMD Kria **KV260** (`xck26-sfvc784-2LV-c`), **INT8**(`ap_int<8>`, `cnn_accelerator.cpp:3`), psum INT32(`cnn_accelerator.cpp:4`). 목표 클럭/주기 정보는 저장소 코드에 없음(확인 불가; .tcl 부재).

### 0.3 운영 경로
```
[CPU 베이스라인 (전체 모델, 비교 기준)]
  카메라 → blobFromImage(1/255, 320) → OpenCV DNN forward(Darknet) → NMS → 화면
  (yolo4tiny_cpu.cpp:40-71, 79-117)
                                  │  850ms/1.18FPS (README.md:64-67, 검증불가)
─────────────────────────────────┼────────────────────────────────────────────
[FPGA 가속 경로 (conv 1개만)]
  DDR(in_fm HWC, weights OIHW, bias INT32)
     │ AXI4 m_axi 4번들 gmem0~3 (cnn_accelerator.cpp:31-34)
     ▼
  bias_buf 적재(1회) → [co0 32 × ci0 16 타일 루프]
     ├ wbuf 가중치 타일 적재 PIPELINE II=1 (:76-86)
     ├ linebuf[3][26][8] 행단위 적재 (:97-104)
     ├ window[3][3][8] 회전 인덱싱 재구성 (NO modulo) (:113-120)
     ├ MAC: CO언롤4 × CI언롤2 ≈ 8 lanes, PIPELINE II=4 (:109-110, 123-134)
     └ acc>>8 + INT8 포화 → out_fm HWC write (:139-148)
     ▼
  DDR  ──►  ap_done 폴링 (fpga_arm.cpp:94)
[호스트: /dev/mem mmap → 주소레지스터 write → ap_start → 폴링 (fpga_arm.cpp:44-110)]
                                  │  368ms/2.7FPS, 2.31x (README.md:73-84, 검증불가)
[검증: testbench.cpp 골든 비트일치 비교 (HLS C-sim, :117-143)]
```
- 합성 PPA 보고값(README 자체): LUT 18% / FF 8% / BRAM 13% / DSP 13%(`README.md:49-54`). `docs/ARM_5_REPORT.pdf` 존재하나 **PDF 텍스트 추출 불가(pdftoppm 미설치)** → 리포트 본문 수치는 **확인 불가**. README 표 외 PPA 근거 없음.

---

## 1. Repo 개요 + 호출 계층 + 제외

### 1.1 자체 소스 4파일 (전부 분석 대상)

| 구분 | 파일 | 핵심 함수(라인) | 역할 |
|---|---|---|---|
| **HW(PL) 커널** | `hardware/HLS/cnn_accelerator.cpp` | `yolo_conv_kv260`(:25-160) | 단일 3×3 conv 가속기 톱(AXI m_axi + s_axilite) |
| **HW 검증(C-sim)** | `hardware/HLS/testbench.cpp` | `golden_conv`(:40-80), `main`(:85-155) | 골든 6중루프 vs 커널 비트일치 비교 |
| **SW(PS) 드라이버** | `software/fpga_arm.cpp` | `main`(:44-110) | /dev/mem mmap MMIO, 주소레지스터 write, ap_start/done 폴링 |
| **SW 베이스라인** | `software/cpu_baseline/yolo4tiny_cpu.cpp` | `main`(:33-139) | OpenCV DNN 전체 YOLOv4-Tiny CPU 추론(비교 기준) |

### 1.2 호출 계층
```
[HW C-sim]  testbench.cpp:main → yolo_conv_kv260(=DUT)  // :117
                              → golden_conv(=REF)        // :123
                              → 비트일치 비교             // :129-138
[실보드]    fpga_arm.cpp:main → mmap(/dev/mem) → reg write → ap_start → poll  // :53,79-94
            (PL에서 yolo_conv_kv260이 RTL로 동작; 호스트는 RTL과 레지스터로만 통신)
[베이스라인] yolo4tiny_cpu.cpp:main → readNetFromDarknet → net.forward → NMS  // :40,71,107
```
- **세 실행 경로는 서로 독립**: C-sim(testbench)·실보드(fpga_arm)·CPU(yolo4tiny)는 빌드/실행이 분리. testbench의 `yolo_conv_kv260`은 C로 직접 호출하지만 fpga_arm는 RTL을 레지스터로만 제어 → 호스트↔커널 함수 직접 호출 없음.
- **README ↔ 실코드 불일치**: README는 OpenCL/XRT 호스트(`host.cpp`, `README.md:99,147-160`)를 기술하나, 저장소의 실제 호스트는 `fpga_arm.cpp`(mmap)다. `host.cpp`는 저장소에 **부재**(확인 불가).

### 1.3 제외 목록(이름만 언급, 분석 제외)
- **생성물(바이너리/HW 핸드오프)**: `hardware/kernal/YOLO_wrapper.bit`(비트스트림), `hardware/kernal/YOLO_wrapper.xsa`(하드웨어 핸드오프). 코드 근거 분석 대상 아님.
- **vendor 모델 가중치/설정**: `software/cpu_baseline/yolov4-tiny.cfg`(Darknet 설정), 동급 `.weights`·`coco.names`(외부 사전학습). 분석 제외.
- **문서/이미지 산출물**: `docs/ARM_5_REPORT.pdf`(PPA 인용 시도 → 추출 불가), `docs/*` 이미지, `hardware/VIVADO_PROJECT/*.png`(존재 시).
- **부재(확인 불가)**: OpenCL 호스트 `host.cpp`(README 기술 but 미포함), HLS 자동생성 레지스터맵 헤더 `x*_hw.h`(드라이버 오프셋 정합 검증 불가), 합성/구현 .tcl·Makefile(빌드 자동화 미포함).

---

## 2. 모듈: HLS conv 커널 톱 `yolo_conv_kv260` — `cnn_accelerator.cpp`

### 2.1 역할 + 상위/하위
- **역할**: 고정 형상(26×26×128 → 24×24×128, 3×3) INT8 conv 1개를 PL에서 수행하는 가속기 톱. 가속기의 사실상 전부.
- **상위**: C-sim에서 `testbench.cpp:117`이 DUT로 호출 / 실보드에서 호스트가 RTL로 제어. **하위**: 없음(자기완결, 외부 함수 호출 없음).

### 2.2 인터페이스 / 메모리맵
- AXI4 master 4번들 분리: `in_fm`/`out_fm`/`weights`/`bias` → `gmem0`/`gmem1`/`gmem2`/`gmem3`(`cnn_accelerator.cpp:31-34`). depth 명시: in/out=86528(=26·26·128), weights=147456(=128·128·3·3), bias=128.
- 제어: `s_axilite port=return`(`:35`). → ap_ctrl(ap_start/ap_done) 레지스터 + 4개 포인터 주소 레지스터를 자동 생성(드라이버 `fpga_arm.cpp:16-24` 오프셋과 정합해야 함).
- **bias만 INT32 포인터**(`:29`), 나머지 INT8(`:26-28`).

### 2.3 온칩 버퍼 구조
| 버퍼 | 선언 | 분할 | 크기(payload) |
|---|---|---|---|
| `linebuf[3][26][8]` | `:40` | dim1·dim3 complete(`:41-42`) | 3·26·8·8b = **4,992 bit** (~0.6 KB) |
| `window[3][3][8]` | `:44` | complete(`:45`) | 3·3·8·8b = 576 bit |
| `wbuf[4][8][3][3]` | `:47` | dim1·dim2 complete(`:48-49`) | 4·8·9·8b = 2,304 bit |
| `bias_buf[128]` | `:51` | cyclic factor=4(`:52`) | 128·32b = 4,096 bit |
| `acc[4]` | `:54` | complete(`:55`) | 4·32b = 128 bit |
- **총 온칩 버퍼 ≈ 12.1 Kb (~1.5 KB)**. 입력 전체 86,528B를 온칩에 못 올리므로 **co0 타일마다 in_fm을 DRAM에서 재독출**(아래 5절 정량 비효율).

### 2.4 루프 구조 (출력채널-입력채널 타일링)
```
for co0 in 0..128 step 4:        // 32회 (:70)
  for ci0 in 0..128 step 8:      // 16회 (:71)
    load wbuf  PIPELINE II=1     // 4·8·3·3=288 trip (:76-86)
    for y in 0..26:              // 26행 (:93)
      load linebuf row  PIPELINE // 26·8 trip (:97-104)
      if y>=2:                   // 유효 윈도우 (:108)
        for x in 2..26:          // 24열, PIPELINE II=4 (:109-110)
          build window (회전인덱싱, no modulo) (:113-120)
          for co in 0..4 UNROLL  (:123-124)
            for ci in 0..8 UNROLL factor=2  (:128-129)
              acc[co] += window·wbuf  // K×K=9 미언롤 (:130-134)
```
- **누산 정책(output-stationary)**: `ci0==0`에서만 bias로 acc 초기화(`:125-126`), `ci0==CIN-CI_TILE`(마지막 입력타일)에서만 출력 flush(`:138`). → 부분합 acc를 ci0 루프 동안 레지스터에 유지, DRAM 왕복 제거.

### 2.5 슬라이딩 윈도우 / 회전 라인버퍼
- 행 적재: `linebuf[row_ptr][x][ci] = in_fm[(y*W+x)*CIN + (ci0+ci)]`(`:100-103`) → **입력 레이아웃 HWC**(채널 last).
- 윈도우 재구성(모듈로 제거): `r = row_ptr + K - ky; if(r>=K) r-=K`(`:116-117`), 주석 "NO modulo on RAM"(`:112`). 회전 row_ptr: `row_ptr++; if(row_ptr==K) row_ptr=0`(`:155-156`).
- 가중치 인덱스 `(co0+co)*CIN*K*K + (ci0+ci)*K*K + ky*K+kx`(`:81-85`) → **가중치 레이아웃 OIHW**.

### 2.6 MAC 구조 (병렬도 — ★안티패턴 핵심)
- 공간 루프 `for x`에 **PIPELINE II=4**(`:110`) — II=1이 아니라 **II=4**. 즉 출력 픽셀 1개당 4사이클 소요(이상적 1사이클 대비 4배 느림).
- 출력채널 `for co` UNROLL → CO_TILE=4 병렬(`:123-124`).
- 입력채널 `for ci` UNROLL **factor=2**(부분언롤, "safe partial unroll" 주석)(`:128-129`).
- 커널 `for ky`,`for kx`(K×K=9)는 **미언롤**(`:130-131`).
- 내부 MAC `acc[co] += window[ky][kx][ci] * wbuf[co][ci][ky][kx]`(`:132-134`).
- **병렬 MAC = CO언롤4 × CI언롤2 = 8 lanes**. systolic array 아님 — 단순 언롤+파이프라인 데이터패스. README는 "Parallel MAC Units"라 표현(`README.md:41`)하나 실제는 8 lanes에 II=4.

### 2.7 양자화 / 출력 (★안티패턴 핵심)
- **재양자화 = `out = acc >> 8`** 고정 우측시프트 8비트(`:139`) 후 INT8 포화 `[-128,127]`(`:140-141`). per-tensor 학습 스케일/zero-point **전무**. 모든 레이어·텐서에 동일 시프트량 하드코딩 → 실제 학습 스케일과 무관, **검출 정확도 보장 근거 없음**(확인 불가). 골든도 동일 `>>8`(`testbench.cpp:69`)이라 C-sim은 통과하지만 이는 "양자화 정확성"이 아니라 "HW=REF 비트일치"만 검증.
- 출력 인덱스 `((y-2)*OW + (x-2))*COUT + (co0+co)`(`:143-146`) → **출력 HWC**.

### 2.8 정량 / 병목
- scalar MAC 84.9 MMAC, lanes 8, II=4 → **이상적 사이클 ≈ scalar MAC / (lanes/II) = 84.9M / (8/4) ≈ 42.5M cycle**(K×K 미언롤·파이프 오버헤드 무시한 하한, 추정). 8 lanes를 II=4로 나누면 유효 처리량은 사실상 **2 MAC/cycle 수준**.
- DSP 13%만 사용(`README.md:53`) → KV260 DSP(1248개)의 ~162개 추정인데 8 MAC밖에 안 쓰므로 **대부분 idle 또는 비효율 매핑**(추정; csynth 미확인).

---

## 3. 모듈: 테스트벤치 `golden_conv` + `main` — `testbench.cpp`

### 3.1 역할 + 상위/하위
- **역할**: HLS C-sim에서 커널 출력과 CPU 골든의 **비트일치** 검증. **상위**: 없음(엔트리). **하위**: `yolo_conv_kv260`(`:117`), `golden_conv`(`:123`).

### 3.2 골든 레퍼런스
- 6중 직접 conv(`:46-67`): 입력 `((y+ky)*W+(x+kx))*CIN+ci`(HWC, `:56-57`), 가중치 `co*CIN*K*K+ci*K*K+ky*K+kx`(OIHW, `:59-62`), `acc=bias[co]` 초기화(`:50`).
- **커널과 동일 양자화 규약**: `acc>>=8` + 포화(`:69-71`)(`cnn_accelerator.cpp:139-141`과 일치). → TB는 양자화 규약 정합을 보장하지만 규약 자체의 타당성은 검증 못함.

### 3.3 결정론적 입력 / 비교
- 입력 `in_fm[i]=(i%13)-6`, `weights[i]=(i%7)-3`, `bias[i]=i%16`(`:101-108`) → 외부 npy 불요, 자기완결.
- 전체 OH·OW·COUT(=24·24·128=73,728) 원소 비교, mismatch 10개까지 출력(`:129-138`), PASS/FAIL 종료코드(`:140-154`).

### 3.4 정량 / 병목
- **병목 없음**(검증 코드). 단 **양자화 정확도 검증 불가**: 골든이 동일 `>>8`이라 PASS는 "기능 등가"만 의미, 모델 정확도와 무관. cosim(RTL)·합성 리포트는 저장소에 없음(확인 불가).

---

## 4. 모듈: ARM 드라이버 `main` — `fpga_arm.cpp`

### 4.1 역할 + 상위/하위
- **역할**: PS에서 PL을 제어하는 베어메탈식 호스트. **상위**: 없음(엔트리). **하위**: POSIX `mmap`/`open`만.

### 4.2 MMIO / 레지스터맵
- 베이스 `0xA0000000`, 맵 `0x10000`(`:13-14`), `mmap(/dev/mem)`(`:53-57`).
- 레지스터 오프셋: `REG_CTRL=0x00`, in/out/wgt/bias 주소 L/H 0x10~0x38(`:16-24`). HLS 자동생성 `x*_hw.h`와 정합해야 하나 헤더 부재(확인 불가).
- 실행: `fpga[REG_CTRL/4]=1`(ap_start, `:93`) → `while((fpga[REG_CTRL/4]&0x2)==0)`(ap_done **바쁜대기 폴링**, `:94`) → chrono 시간측정(`:91-99`).

### 4.3 정량 / 병목 (★안티패턴)
- **잠재 버그 — 가상=물리 주소 가정**: `in_phys=(uint64_t)in_fm` 등 **유저공간 가상주소를 그대로 DMA 물리주소로 사용**(`:73-76`). 주석이 한계 인정: "assumes identity mapping ... For production, use CMA or u-dma-buf"(`:29-31`). 일반 malloc/정적배열은 물리연속 보장 없음 → 실디바이스 DMA 정상동작 **확인 불가**, 실배포 위험.
- **바쁜대기 폴링**(`:94`): interrupt/event 기반이 아니라 CPU 점유 spin. 저지연 시스템엔 부적합.
- **데이터 전송 오버헤드**: bias 제외 in(86,528B)+out(86,528B)+weights(147,456B) = **약 320 KB/호출**을 DDR↔PL로 이동. conv 1개치곤 전송량 대비 연산밀도 낮음(84.9 MMAC / 320KB ≈ 265 MAC/byte; 단일 레이어라 가중치 재사용 없음).

---

## 5. 모듈: CPU 베이스라인 `main` — `yolo4tiny_cpu.cpp`

### 5.1 역할 + 상위/하위
- **역할**: OpenCV DNN으로 **전체** YOLOv4-Tiny를 CPU에서 추론(README 2.31x의 기준). **상위**: 없음. **하위**: OpenCV `dnn::Net`.

### 5.2 파이프라인
- `readNetFromDarknet(cfg,weights)` + CPU 타깃(`:40-42`), 카메라(`:44`).
- 입력 320, conf 0.35, NMS 0.45(`:50-52`), `blobFromImage(1/255, 320)`(`:64`).
- forward(`:71`) → 검출 파싱(obj·class score, `:79-104`) → `NMSBoxes`(`:107`) → draw(`:117`).
- pre/inf/post/tot 시간 분리 측정(`:121-133`).

### 5.3 정량 / 병목 (★비교 타당성)
- **HLS 커널과 직접 비교 불가**: 베이스라인은 전체 모델(~3.4 GMAC 추정), HLS는 conv 1개(84.9 MMAC). README의 850ms→368ms·2.31x(`README.md:64-84`)는 **시스템 레벨 추정 비교**이며, conv 1개 오프로딩이 어떻게 전체 850ms를 368ms로 줄이는지 코드 근거 없음 → **검증 불가**. (conv 1개가 전체의 ~2.5%인데 2.31x는 산술적으로 비정합 — 측정 조건 불명, 확인 불가.)

---

## 6. 모듈 한눈 요약 표

| 모듈 | 파일 | 핵심(라인) | 역할 | 대표 정량 |
|---|---|---|---|---|
| conv 커널 톱 | cnn_accelerator.cpp | yolo_conv_kv260(:25) | 단일 3×3 conv INT8 | 84.9 MMAC, 8 lanes, II=4 |
| 인터페이스 | cnn_accelerator.cpp | m_axi gmem0~3(:31-34) | AXI4 4번들 + s_axilite | depth 86528/147456/128 |
| 버퍼 | cnn_accelerator.cpp | linebuf/window/wbuf(:40-55) | 회전 라인버퍼 윈도우 | 온칩 ~12 Kb |
| MAC | cnn_accelerator.cpp | UNROLL co4·ci2(:123-134) | 언롤+파이프 데이터패스 | 8 MAC, K²미언롤 |
| 양자화 | cnn_accelerator.cpp | acc>>8 포화(:139-141) | 고정시프트 재양자화 | **학습스케일 없음** |
| 골든 TB | testbench.cpp | golden_conv(:40), main(:85) | 비트일치 검증 | 73,728 원소 비교 |
| ARM 드라이버 | fpga_arm.cpp | main(:44) mmap(:53) | MMIO 제어·폴링 | 가상=물리 가정(:73) |
| CPU 베이스라인 | yolo4tiny_cpu.cpp | main(:33) forward(:71) | 전체 모델 CPU 추론 | 비교 분모(검증불가) |

---

## 7. 읽기 순서 / 코드 추적 순서

1. **형상 먼저**: `cnn_accelerator.cpp:9-19`(H/W/CIN/COUT/K/타일) → 무엇을 가속하는지 한 줄로.
2. **인터페이스**: `:31-35`(AXI 4번들 + s_axilite) → 호스트와의 계약.
3. **버퍼·루프**: `:40-71`(온칩 버퍼 + co0/ci0 타일) → output-stationary 누산 골격.
4. **윈도우 핵심**: `:113-120`(회전 인덱싱, no modulo) → 라인버퍼 슬라이딩 본질.
5. **MAC·양자화(★)**: `:123-148` → 8 lanes·II=4·`>>8` 안티패턴 직시.
6. **검증**: `testbench.cpp:40-80`(골든) → `:117-138`(비트일치) → 규약 정합.
7. **호스트(★)**: `fpga_arm.cpp:73-94`(물리주소 가정·폴링) → 실배포 위험.
8. **비교 기준**: `yolo4tiny_cpu.cpp:40-71` → 베이스라인이 "전체 모델"임을 확인(비교 비정합).

---

## 8. 병목·병렬도 노브 (★안티패턴 정량 요약 + 개선 방향)

### 8.1 안티패턴 정량 비효율
1. **단일 레이어만 오프로드** — 가속 대상 84.9 MMAC은 YOLOv4-Tiny@320 전체(~3.4 GMAC 추정)의 **약 2.5%**(`cnn_accelerator.cpp:9-19` 형상 확정 / 분모 추정). 나머지 97.5%는 PS(OpenCV). 전체 모델 가속이 아니므로 README의 2.31x는 코드로 재현 불가(검증 불가).
2. **빈약한 MAC 병렬도** — CO4×CI2 = **8 lanes**(`:123-129`), 게다가 공간 루프 **II=4**(`:110`)라 유효 처리량 ~2 MAC/cycle. K×K=9 미언롤(`:130-131`). DSP 13%만 사용(`README.md:53`)하면서도 8 MAC → DSP 활용 비효율(idle 다수, 추정). 형제 systolic/dataflow(TATAA/ESDA/UltraNet)의 수백~수천 MAC 대비 2~3 자릿수 열위.
3. **고정시프트 양자화** — `acc>>8`(`:139`)는 모든 텐서 동일 시프트. per-tensor 학습 scale/zero-point 없음 → 정확도 보장 불가(확인 불가). 골든도 `>>8`(`testbench.cpp:69`)이라 C-sim PASS는 정확도가 아니라 비트일치만 의미.
4. **입력 DRAM 재독출** — `linebuf` 행 적재 루프가 co0(32회) 안에 있어(`:70,97-104`) **입력 텐서를 co0 타일마다 재독출** = 32 × 86,528 B ≈ **2.77 MB**(필요량 86,528 B의 32배). 가중치는 타일당 1회씩 총 147,456 B 독출(재사용 없음 — 단일 레이어라 배치/공간 재사용 부재).
5. **호스트 안티패턴** — 가상=물리 주소 가정(`fpga_arm.cpp:73-76`, 주석 인정)·바쁜대기 폴링(`:94`). 실보드 DMA 정상성·저지연 보장 불가.
6. **README↔코드 불일치** — OpenCL/XRT(README) vs mmap(실코드), `host.cpp` 부재. 빌드 .tcl/Makefile 부재(확인 불가).

### 8.2 개선 방향 (대조군으로서의 교훈)
- **레이어 융합·스트리밍 dataflow**: 단일 레이어 오프로딩 대신 `#pragma HLS DATAFLOW`로 다수 레이어를 파이프(ESDA/UltraNet식). 입력 재독출(2.77 MB) 제거.
- **MAC 어레이 확대 + II=1**: 8 lanes·II=4 → 공간/채널 동시 언롤로 수백 lanes·II=1. K×K도 언롤. DSP packing(INT8 2/DSP)으로 효율 ×2.
- **학습 스케일 양자화**: `>>8` → per-tensor(또는 per-channel) 학습 scale·bias 적용. ViT처럼 LayerNorm/Softmax 동적레인지 큰 연산엔 필수.
- **DMA 정공법**: CMA/u-dma-buf 물리연속 버퍼 + interrupt 기반 완료통지.
- **재사용 가능 요소**(긍정): 회전 라인버퍼 인덱싱(`:113-120`)·output-stationary 채널타일 누산(`:125-148`)·골든 비트일치 TB 구조(`testbench.cpp`)는 우리 HLS 검증/stem conv 템플릿으로 차용 가능.

---

*근거 파일(절대경로)*:
`\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\CNN-Accel\Kria-YOLOv4-Tiny-FPGA-Accelerator\hardware\HLS\{cnn_accelerator.cpp,testbench.cpp}`,
`...\software\fpga_arm.cpp`,
`...\software\cpu_baseline\yolo4tiny_cpu.cpp`,
`...\README.md`.
*확인 불가*: `docs\ARM_5_REPORT.pdf`(PDF 텍스트 추출 불가, pdftoppm 미설치), OpenCL `host.cpp`(README 기술 but 부재), HLS 레지스터맵 헤더 `x*_hw.h`, 합성 .tcl/Makefile, 합성/cosim PPA 리포트.
