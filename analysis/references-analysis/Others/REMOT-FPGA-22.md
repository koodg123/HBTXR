# REMOT-FPGA-22 정밀 분석

> 분석 대상 경로: `REF/Others/REMOT-FPGA-22`
> 분석 방식: 자체 핵심 소스(HLS 커널 `top.cpp`/`amap.h`/`au_amap.h`/`hash.h`, Python drive/software/eval)를 Read로 직접 읽고 라인 근거 기반 작성. 비트스트림(.bit/.hwh)/IP/.git/대용량 .mat/.avi 제외.

---

## 1. 개요

- **목적**: Dynamic Vision Sensor(DVS, 이벤트 카메라)의 비동기 이벤트 스트림을 FPGA에서 **다중 객체 추적(Multi-Object Tracking, MOT)**하는 HW/SW 통합 아키텍처. 핵심은 "Attention Unit(AU)" — 각 AU가 특정 관심영역(ROI)에만 주목하고, 객체 이동에 따라 ROI가 변하도록 설계.
- **한줄요약**: "**병렬 Attention Unit(AU) 레이어를 FPGA에 구현해 DVS 이벤트 스트림을 어텐션 맵(AMAP) 갱신/조회로 처리하는, HLS 기반 이벤트-구동 MOT 가속기. AMAP을 dense 배열(FULL) 또는 해시테이블(HASH)로 저장하는 3가지 구현(FULL-AMAP / HASH-AMAP / FIFO-ONLY)을 제공**".
- **원논문**: Y. Gao, S. Wang, H. K.-H. So, "REMOT: A Hardware-Software Architecture for Attention-Guided Multi-Object Tracking with Dynamic Vision Sensors on FPGAs", ACM/SIGDA FPGA 2022 (README L1-14 근거).
- **타깃 디바이스**: PYNQ-Z2, Ultra96. PYNQ v2.6 이미지, Vivado 2020.2, Python 3.7, Matlab R2021b (README L30-36). 보드별 디렉토리(`hls/pynq`, `hls/ultra96`)에 동일 3종 구현.
- **저자 그룹**: Hayden K.-H. So (HKU) 그룹 — AGNA-FCCM2023과 동일 그룹(공저자 Yizhao Gao, Hayden So 중복). software/hardware 분리 컨벤션 공유.

---

## 2. 디렉토리 구조 (자체 소스 트리 + 제외 목록)

### 자체 핵심 소스
```
REMOT-FPGA-22/
├── README.md, LICENSE, requirements.txt
├── software/                              # Python/Matlab: 알고리즘 시뮬레이션
│   ├── software_basic/
│   │   ├── au_functions.py                # bbox/IoU/클러스터링 등 추적 유틸
│   │   ├── sw_wrap.py                      # 소프트웨어 AU 래퍼
│   │   ├── software_evaluation.ipynb       # 3개 데이터셋 시뮬레이션
│   │   ├── hota.m / evalhota.m            # HOTA(추적 정확도) 평가 (Matlab)
│   │   ├── config/*.yml, dataset/*.mat    # 입력/GT
│   │   └── result/                         # (생성물)
│   └── hash/                               # HASH-AMAP 정확도 실험
│       ├── hash_au.py, hash_sw.py, au_functions.py
│       ├── hash_acc_experiment.ipynb       # 해시 테이블 크기별 정확도(fig.7f)
│       └── eval.py, hota.m
├── hardware/
│   ├── README.md
│   ├── hls/                                # ★HLS 커널 (FULL/HASH/FIFO-ONLY)
│   │   ├── README.md, Makefile
│   │   ├── pynq/  (full, hash, fifo, *_final)
│   │   │   ├── full/                        # FULL-AMAP
│   │   │   │   ├── top.cpp/top.h            # 탑 커널 + 인터페이스 pragma
│   │   │   │   ├── amap.h                   # query / update_square (dense AMAP)
│   │   │   │   ├── au_amap.h                # au_full / init / return 함수군
│   │   │   │   ├── para.h                   # 파라미터/타입 정의
│   │   │   │   └── (script.tcl, vivado/)
│   │   │   ├── hash/                        # HASH-AMAP
│   │   │   │   ├── hash.h                   # hash_map / 2-way insert / query
│   │   │   │   ├── au_hash.h, top.h, para.h
│   │   │   ├── fifo/, fifo_final/           # FIFO-ONLY
│   │   │   │   ├── fifo.h, au_fifo.h, top.cpp/top.h
│   │   └── ultra96/  (동일 3종 구조)
│   ├── drive/                              # ★PYNQ overlay 파이썬 드라이버
│   │   ├── pynq/  au_hardware_{full,hash,fifo_only}.py, au_functions.py
│   │   └── ultra96/  (동일)
│   └── eval/  eval.py, hota.m, evalhota.m  # 정확도 평가
```

### 분석 제외 (이름만 언급)
- `.git/`, 모든 `bitfile/`(`.bit`, `.hwh`, `ps7_init*`, `xtop*` 드라이버 = Vivado/Vitis 생성물), `*.mat`/`*.avi`/`*.csv`(데이터셋·결과 = 대용량/생성물), `vivado/*.tcl`·`*_proj.tcl`(프로젝트 생성 스크립트), Matlab `hota.m`/`evalhota.m`(외부 HOTA 메트릭 구현).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 hardware/hls/pynq/full/para.h — 파라미터/타입 (FULL-AMAP)
- 이벤트 비트필드(L1-12): `XBITS=10, YBITS=10, TBITS=32, PBITS=1`(좌표 10bit씩 → 최대 1024×1024), `VBITS=16`(AMAP 값), `HEIGHT=260, WIDTH=346`(DAVIS346 센서 해상도), **`N_AU=2`**(동시 AU 수), `FIFO_DEPTH=128`(AU당 이벤트 FIFO 깊이), **`d=3`**(ROI 반경 → 7×7 갱신창).
- 타입(L14-42): `event` 구조체(x/y/t/p), `event_pack_t=ap_uint<64>`(이벤트 1개 패킹), `amap_type=ap_uint<16>`, `event_axi=hls::axis<...,1,1,1>`(AXI-Stream), `event_stream`(입력), `local_stream`(브로드캐스트). → **이벤트 = 64bit 워드 1개로 압축 스트리밍**.

### 3.2 amap.h — FULL-AMAP 코어 연산 (dense 2D 배열)
#### `query<Number>()` (L3-25)
- 역할: 한 이벤트 좌표 (x,y)에 대해 **모든 AU의 AMAP에서 해당 픽셀 값을 조회**, 0보다 크면 그 AU가 "interested"(이 이벤트에 주목)로 표시.
- 구현: `AMAP[n][y*WIDTH+x]`을 `N_AU` 전부 `#pragma HLS UNROLL`로 동시 조회(L17-23). → **모든 AU 병렬 조회**(이벤트당 1 클럭 목표).

#### `update_square<Number>()` (L28-80) — ★핵심 갱신 커널
- 역할: interested AU들의 ROI(중심 (x[n],y[n]), 반경 `d`=3 → 7×7=49 픽셀)에 `value`(+1 또는 -1)를 누적.
- 2단계 dataflow(`#pragma HLS DATAFLOW`, L36): (1) **QL1/QL2 루프**(L43-63, `II=1`)에서 7×7 좌표를 경계검사 후 (x,y,v)를 AU별 `queue` 스트림에 패킹(`pack.range(...)`). (2) **IL1 루프**(L65-79)에서 큐를 읽어 `AMAP[n][y_i*WIDTH+x_i] += v`로 누적, `#pragma HLS dependence variable=AMAP inter false`(L76)로 **메모리 의존성 무시 → II=1 파이프라인**.
- `value=+1`로 새 이벤트 위치 강화, 이후 `value=-1`로 가장 오래된 이벤트(FIFO에서 pop) 위치 약화 → **슬라이딩 윈도우 어텐션 맵**(au_amap에서 호출 순서로 확인).

### 3.3 au_amap.h — AU 본체 + 초기화/반환 (FULL)
#### `au_full<Number>()` (L14-98) — ★AU 메인 루프
- 역할: `N`개 이벤트를 순차 처리하며 AMAP 갱신 + AU 생성/소멸 + AU별 이벤트 FIFO 관리.
- 흐름(이벤트당): (1) `pack2e`로 64bit→event 디코드(L41-42). (2) `query`로 interested AU 판정(L43). (3) **어느 AU도 주목 안 하면(`!found_interested`) 빈(status==1) AU 슬롯을 하나 할당**(L51-64) → **새 객체 등장 시 AU 자동 spawn**. (4) interested AU의 ROI에 `+1` 누적(`update_square(...,1,...)`, L72). (5) AU별 이벤트 FIFO에서 가장 오래된 이벤트를 꺼내 그 위치를 `-1` 대상으로 잡고, 새 이벤트를 FIFO에 push(원형버퍼 `last_i %= FIFO_DEPTH`, L77-93). (6) `update_square(...,-1,...)`로 오래된 위치 약화(L96). → **이벤트 기반 ROI가 객체를 따라 이동(attention-guided)**.
- `#pragma HLS ARRAY_PARTITION ... complete`(L34-37)로 interested/update_x/y/old를 완전분할 → AU 병렬.

#### 초기화/반환 함수군
- `init_amap_array`(L120-134)/`init_efifo`(L138-156): host가 준 AMAP/FIFO 상태를 `init_n` AU에 로드(checkpoint 복원 — kill 후 재생성 시 사용).
- `return_amap`(L101-117)/`return_fifo`(L237-252)/`return_status`(L221-234)/`return_depth`(L256-266): host로 AMAP/FIFO/상태/깊이 회수.
- `broadcast<NA>()`(L178-193): 입력 AXIS 이벤트를 `local_stream`으로 전달(`II=1`) — dataflow 진입점.
- `init_status`(L196-216): 최초 1회 모든 AU를 빈 상태(status=1)로, 이후 host status_in으로 갱신(L210-215).

### 3.4 top.cpp / top.h — 탑 HLS 커널 (FULL)
- `top()` (top.cpp L47-97): 인터페이스 — `in_stream`(AXIS 이벤트), `out_amap/init_amap/out_fifo/init_fifo`(각 별도 `m_axi bundle=gmem0..3`, L62-65), 스칼라 제어(`N,return_n,init_n,in_fifo_depth,status_in/out`)는 `s_axilite bundle=control`(L67-73). → **DMA 4채널 + AXI-Lite 제어**의 전형적 PYNQ 오버레이 구조.
- `#pragma HLS DATAFLOW`(L75) + `broadcast_stream`(depth=1024, L78)로 broadcast → `Au_full_wrapper<0>` 파이프.
- `Au_full_wrapper()` (top.cpp L4-42): `static amap[N_AU][HEIGHT*WIDTH]`을 **on-chip 상주**(`#pragma HLS ARRAY_PARTITION variable=amap dim=1 complete`, L21 → AU 차원 완전분할). init→au_full→return 순서로 호출(L32-40). → **AMAP은 BRAM에 상주, host와는 init/return으로만 교환**.

### 3.5 hash/ — HASH-AMAP 구현 (메모리 효율 변형)
#### para.h (hash) 차이
- `HEIGHT=180, WIDTH=240`(다른 데이터셋), **`N_AU=10`**(FULL의 2배 이상), `T_SIZE=8192`, `TABLE_BITS=13`(=log2 T_SIZE), `FIFO_DEPTH=64`. → **dense 배열 대신 8192엔트리 해시테이블**로 AMAP을 저장 → AU당 메모리를 HEIGHT×WIDTH(=43200)에서 8192로 축소 → 더 많은 AU 탑재.

#### hash.h
- `hash_map<>()` (L2-10): 좌표를 키로 `key = x + y*WIDTH + 1`(0을 빈 슬롯 표시로 예약).
- `hash_forward_A<>()` (L12-21): **Knuth 곱셈 해싱** — `key * A`(A=72189)의 상위 비트를 테이블 인덱스로(`mul.range(KBITS-1, KBITS-TABLE_BITS)`). → 단순 곱셈 1회로 해시(HLS 친화적).
- `insert_2way<>()` (L25-68): 1-way(단일 슬롯) insert. 키 일치 시 값 누적(음수면 0 클램프, L48-56), 빈 값(value>0)이면 신규 삽입, 아니면 유지(L57-64). 주석상 2way지만 구현은 1슬롯(확인됨).
- `query`/`update_square` (L71-161): amap.h와 동일 구조이나 dense 배열 접근이 **해시 조회/삽입**으로 치환(L93, L154). `#pragma HLS dependence ... inter false`(L156-157)로 파이프라인.
- **트레이드오프**: 해시 충돌 시 값 손실 가능(빈 슬롯 우선 정책) → 정확도 vs 메모리. `hash_acc_experiment.ipynb`가 테이블 크기별 정확도(논문 fig.7f)를 측정.

### 3.6 software/software_basic/au_functions.py — 추적 후처리 유틸 (SW 레퍼런스)
- `bbox`(L8)/`bbArea`(L12)/`bbox2points`(L16): AMAP 활성 픽셀의 바운딩박스 계산.
- `bboxOverlapRatio`(L25-36): **IoM(Intersection over Min-area)** — 교집합/min(면적). IoU 변형으로 작은 박스 포함관계 민감.
- `clusterAu`(L39-58): 박스들을 IoM>=임계 기준 그리디 클러스터링(같은 객체에 붙은 여러 AU 병합). → **AU 출력 → 최종 트랙 박스**로 변환하는 후처리. 하드웨어는 AU/AMAP만, 박스화·클러스터링·트랙ID는 호스트 SW가 담당.

### 3.7 hardware/drive/pynq/au_hardware_full.py — PYNQ 드라이버 (host↔FPGA)
- `Au_full` 클래스(L10-): `Overlay(bitfile)` 로드, `top_0` IP 핸들, `allocate`로 DMA 버퍼(event_buffer 8192, output/init_buffer = H×W, in/out_fifo) 확보(L24-28). AXI-Lite 레지스터 오프셋 하드코딩(L30-44, `0x10`~`0x68`) → `top` 스칼라 포트와 1:1 매핑.
- `stream_in_events()` (L140-158): 이벤트를 64bit 패킹(`pack_event`, L93-100) 후 `axi_dma_0.sendchannel.transfer`로 스트림, `ap_done`(0x0 bit1) 폴링(L154). throughput 측정(Meps).
- AU 라이프사이클: `write_au`(L113-138, host 계산 AMAP/FIFO를 특정 AU에 주입), `dump_single_au`/`dump_all_au`(L176-188, AU 상태 회수), `kill_au`(L228-249, AU 슬롯 해제 + 버퍼 클리어), `read_status`(status 레지스터로 점유 AU 파악). `rebuild_amap_with_event`(L209-213): event FIFO로부터 AMAP 재구성(`amapAddlocal` 7×7 누적, L190-200) — **하드웨어 update_square의 SW 미러**.
- → **host가 AU 풀을 관리(spawn/kill/checkpoint)하고, FPGA는 고속 이벤트 처리 엔진** 역할 분담.

### 3.8 eval/eval.py, software/hash/eval.py — 정확도 평가
- HOTA(Higher Order Tracking Accuracy) 기반 평가(Matlab `hota.m`/`evalhota.m` 호출, Matlab Engine API). 데이터셋: shapes_6dof(이벤트 카메라 공개셋), inbound/outbound traffic(자체 라벨링). (README, software/README L4-16 근거).

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 하드웨어 이벤트 처리 파이프 (FULL-AMAP)
```
DVS events → (host pack64) → AXI-DMA → in_stream(AXIS)
  → broadcast() → local_stream
  → au_full() [이벤트당]:
      query(모든 AU AMAP 병렬 조회) → interested[]
      (no interest → 빈 AU spawn)
      update_square(+1, ROI 7×7 강화)
      event FIFO push(new) / pop(oldest)
      update_square(-1, oldest 위치 약화)
  → (host) return_amap/return_fifo/return_status
  → au_functions.bbox/cluster → 트랙 박스/ID
```
- **메모리 계층**: AMAP은 `static`으로 on-chip BRAM 상주(AU 차원 완전분할), 이벤트 FIFO도 on-chip. host와는 init/return DMA로만 교환 → **이벤트 처리 중 외부 메모리 접근 최소화 = 고처리량**.
- **병렬화**: `N_AU` 전 AU를 `#pragma HLS UNROLL`/`ARRAY_PARTITION complete`로 동시 처리. ROI 갱신은 `update_square`의 2-stage DATAFLOW로 II=1.
- **데이터타입**: 모두 `ap_uint`/`ap_int` 고정폭(좌표 10bit, AMAP 값 16bit, 이벤트 64bit 패킹). 부동소수점 없음 — **완전 정수/비트조작 파이프라인**(HLS·자원 효율적).

### 4.2 3가지 구현의 차이
| 구현 | AMAP 저장 | 메모리/AU | N_AU(예) | 특징 |
|---|---|---|---|---|
| **FULL-AMAP** | dense 2D `[N_AU][H*W]` | H×W×16bit | 2 | 정확, 메모리 큼 |
| **HASH-AMAP** | 해시테이블 `[N_AU][T_SIZE]` | T_SIZE×(key+val) | 10 | 메모리 절감→AU 다수, 충돌 손실 |
| **FIFO-ONLY** | AMAP 없이 FIFO만 | FIFO만 | (fifo.h) | 최소자원, AMAP 재구성 host 의존 |
- `_final` 접미사 = 최대 AU 수 구성(논문 fig.7c), 비접미사 = shapes_6dof 데모용(hardware/hls/README L4).

---

## 5. HW/SW 매핑

| 소프트웨어(host Python) | → | 하드웨어(HLS top) |
|---|---|---|
| `pack_event()` (64bit 패킹) | → | `pack2e()` 디코드 (au_amap.h L3-11) |
| `stream_in_events()` → AXI-DMA | → | `in_stream`(AXIS) → `broadcast` |
| AXI-Lite 레지스터(0x10~0x68) | → | `top` 스칼라 포트(`s_axilite control`) |
| `write_au`/`init_buffer` | → | `init_amap_array`/`init_efifo` (m_axi gmem0-3) |
| `dump/kill_au`, `read_status` | → | `return_*` / `init_status` (status_in/out) |
| `au_functions.bbox/cluster` | → | (HW는 AMAP만, 박스화는 SW) |
| `rebuild_amap_with_event` | → | `update_square` (SW 미러) |

→ **HW = 이벤트→AMAP 갱신 고속 엔진, SW = AU 풀 관리·박스화·트랙ID·평가**의 명확한 분담. 동일 알고리즘을 software_basic(순수 SW)·hash(SW)·HLS(HW)로 3중 구현해 **검증 가능성** 확보.

---

## 6. 빌드·실행
- **SW 시뮬레이션**: `conda create -n au python=3.7; pip install -r requirements.txt`; Matlab Engine API 설치(README L48-58). `software_evaluation.ipynb`(3개 데이터셋), `hash_acc_experiment.ipynb`(해시 크기별 정확도).
- **비트스트림 생성**: `cd hardware/hls/pynq; make all`(12개 비트스트림) 또는 개별 폴더 `make all` → `.bit/.hwh`가 `hardware/drive`로 자동 export(hardware/hls/README L6-19).
- **보드 실행**: drive 폴더(프리빌트 비트스트림 포함)를 보드에 업로드, `au_hardware_*.py`로 구동(README, hardware/README L7-11). 결과(영상·트랙)를 다운로드해 `eval/`로 HOTA 평가.

## 7. 의존성
- Python 3.7, numpy, PYNQ v2.6(`pynq.Overlay/allocate/Xlnk`), Vivado 2020.2(HLS+합성), Matlab R2021b(+Engine API, HOTA 평가). HLS 라이브러리: `ap_int.h`, `hls_stream.h`, `ap_axi_sdata.h`(Xilinx — 제외). 보드: PYNQ-Z2(Zynq-7020), Ultra96(Zynq UltraScale+).

## 8. 강점 / 한계 / 리스크
- **강점**: (1) **이벤트-구동 어텐션**을 HLS로 우아하게 표현(query→update_square(+1/-1) 슬라이딩 윈도우). (2) AMAP on-chip 상주 + AU 완전분할로 고처리량(Meps). (3) FULL/HASH/FIFO 3종 + 2보드로 **자원-정확도 트레이드오프 탐색** 제공. (4) SW(2종)·HW 3중 구현으로 검증 견고. (5) `dependence inter false`·`DATAFLOW`·`ARRAY_PARTITION`의 교과서적 HLS 최적화.
- **한계/리스크**: (1) **DNN/Transformer가 아님** — MAC 어레이·GEMM 없음, AMAP 누적/조회·해시가 전부. (2) `N_AU`가 컴파일타임 상수(`#define`) → AU 수 변경 시 재합성. (3) 해시 충돌 시 값 손실(정확도 영향, fig.7f로 정량화). (4) DAVIS346 등 특정 센서 해상도/이벤트 포맷에 종속. (5) AU 풀 관리·박스화가 host SW 의존 → end-to-end 지연에 PS-PL 통신 포함.

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA + XR 시선추적) 관점 시사점
- **XR 시선추적 직접 관련성(최상)**: REMOT는 **이벤트 카메라(DVS) 기반 저지연 객체 추적**으로, XR 시선추적의 **이벤트-기반 동공/시선 추적** 파이프라인과 문제구조가 매우 유사(추정). `update_square`의 ROI 슬라이딩 윈도우(+1/-1)는 동공 영역을 따라가는 어텐션 맵으로 재해석 가능 — **시선 ROI tracker의 HLS 프로토타입 출발점**.
- **이벤트 스트리밍 인터페이스 재사용**: 64bit 이벤트 패킹 + AXIS broadcast + on-chip 상태 상주 + AXI-Lite 제어 패턴은 우리 XR 가속기의 **센서→가속기 저지연 스트리밍 프런트엔드**로 직접 차용 가능.
- **해시 기반 sparse 표현(SpMV 관점)**: hash.h의 Knuth 곱셈 해싱 + 충돌 정책은 **sparse activation/attention(예: 토큰 프루닝, sparse MHSA)**을 dense 배열 없이 표현하는 데 응용 가능. SpMV/sparse GEMM 가속의 인덱싱 메커니즘으로 재사용 검토.
- **HLS 최적화 패턴 라이브러리**: `DATAFLOW`+`dependence inter false`+`ARRAY_PARTITION complete`+더블 stream 패턴은 ViT의 토큰-병렬 처리(N축 언롤)에 그대로 이식 가능한 HLS 관용구.
- **자원-정확도 DSE 사고**: FULL/HASH/FIFO 3종 비교 방법론은 우리 Mamba/ViT 양자화(INT8/INT4)·메모리압축의 트레이드오프 탐색 프레임으로 차용. AGNA의 MIGP DSE와 결합하면 "표현방식×비트폭" 2D 탐색 가능(추정).
- **주의**: REMOT 자체에는 GEMM/systolic/MAC이 없어 **ViT 연산 코어 재사용은 불가** — 본 repo는 "프런트엔드(이벤트 스트리밍/ROI 추적)와 HLS 스타일"에서, AGNA/ViT-Accelerator는 "연산 코어"에서 차용하는 역할 분리가 적절.

## 10. 근거 표기
- 라인 근거는 본문 (파일 Lxx) 형식. README/소스 직접 인용.
- **"추정"**: XR 시선추적과의 직접 적용성(문제구조 유사성 기반 추정, 본 repo는 일반 MOT 대상), `_final` 접미사의 정확한 AU 수(README 설명 기반), "표현×비트폭" 2D DSE 결합 아이디어.
- **"확인 불가"**: 비트스트림 실제 AU 수/주파수/자원(bitfile = 생성물, 제외), Ultra96 구현이 PYNQ와 알고리즘 동일한지 세부(디렉토리 구조상 동일 추정이나 ultra96 소스 미정독 — full/hash/fifo 동일 패턴 가정), `insert_2way`가 실제 2-way가 아닌 1-slot인 점(코드상 1슬롯 확인, 명명과 불일치).
- third-party(Xilinx HLS lib, PYNQ, Matlab HOTA), `.git`, 비트스트림/데이터셋(.mat/.avi/.bit/.hwh)은 이름만 언급하고 분석 제외.
