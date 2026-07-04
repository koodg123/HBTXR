# TMMA 정밀 분석

> 작성 기준: 실제 소스(`hls/MatMul_SA/`의 HLS 커널·테스트벤치·합성 리포트)를 Read로 직접 읽고 라인 근거 기반으로 작성. 라인 번호는 해당 파일 내용 기준.

---

## 1. 개요

- **목적**: Transformer/LLM의 Self-Attention **Q/K/V 프로젝션 행렬곱**을 에지 FPGA에서 가속하는 타일드(tiled) 행렬곱 가속기.
- **한줄요약**: INT8 입력·INT32 출력의 단일 HLS 커널 `mmult_accel`이, **행렬 A를 온칩 BRAM에 상주(persistent)**시키고 B를 컬럼 블록 단위로 스트리밍하며, `TILE_SIZE×TILE_SIZE`(=32×32) 출력 타일을 완전 언롤된 MAC 어레이로 계산하는 **블록/타일 기반 GEMM 가속기**.
- **원논문**: TMMA (arXiv:2503.16731), Richie Li & Sicheng Chen, UC Irvine, 2025-03-23 (`CITATION.cff:3-24`, `README.md:64`). 부제: "A Tiled Matrix Multiplication Accelerator for Self-Attention Projections in Transformer Models".
- **타깃 디바이스**:
  - `README.md:1-13` → **Xilinx KV260 Vision AI Starter Kit**(에지) 최적화.
  - 합성 리포트 `Target device: xck26-sfvc784-2LV-c`(KV260의 Zynq UltraScale+ MPSoC) (`mmult_accel_csynth.rpt:12`).
  - 대상 모델: **DistilBERT**의 MHA Q/K/V 프로젝션(`README.md:11`, `pynq/Quant_distilBERT_forward-pass.ipynb`).
- **출처/서브모듈**: HLS 코드는 별도 GitHub 서브모듈 `Richielee630/MatMul_SA`(`.gitmodules:1-4`). 본 레포는 그 위에 vivado/pynq/model 통합.
- **상태**: 진행 중 연구(WIP), 비트스트림/전체 Vivado 프로젝트 미포함(`README.md:5-8, 72-73`).

---

## 2. 디렉토리 구조

### 자체 핵심 소스 트리
```
TMMA/
├── README.md, LICENSE.md, CITATION.cff, .gitmodules
├── hls/MatMul_SA/                 # ★ 가속기 HLS 소스 (서브모듈, 본 분석 핵심)
│   ├── mmult_accel.cpp            # 탑레벨 커널 mmult_accel (행렬곱 본체)
│   ├── mmult_accel.hpp            # 인터페이스 프로토타입(extern "C")
│   ├── mmult_accel_tb.cpp         # 테스트벤치(Q/K/V 3-프로젝션 + CPU 레퍼런스 검증)
│   ├── README.md                  # HLS 사용법
│   ├── _ide/                      # Vitis IDE 워크스페이스 저널(생성물)
│   └── MatMul_SA/.../hls/syn/report/  # 합성 리포트(csynth) — 생성물이나 자원 근거로 참조
├── pynq/                          # KV260 런타임 노트북(생성/문서)
│   ├── TMMA_pynq_benchmark.ipynb  # FPGA 벤치마크
│   ├── Quant_distilBERT_forward-pass.ipynb  # 양자화 DistilBERT 포워드패스
│   └── *.pdf                      # 노트북 PDF / FLOPs·MACs 계산서
├── vivado/2-27_bd&df/             # 블록디자인/스키매틱(이미지/PDF, 생성물)
└── model/distilBert/              # PyTorch DistilBERT 참조 노트북
    ├── distilBert_pytorch-cpu.ipynb
    └── distilBERT_layer_shape_verify.ipynb
```

### 제외 목록 (이름만 언급, 분석 제외)
- `.git/`, `.git/modules/hls/MatMul_SA/` (pack/hooks) — 버전관리 메타.
- `hls/MatMul_SA/_ide/` (workspace_journal_*.py, settings.json) — Vitis IDE 생성물.
- `hls/MatMul_SA/MatMul_SA/.../syn/report/*.rpt|xml` — 합성 산출물(단, 자원 수치는 근거로 인용).
- `vivado/.../*.png|*.pdf`, `pynq/*.pdf` — 이미지/문서 바이너리.
- `.ipynb`(pynq/model) — 노트북은 핵심 RTL/HLS 아님(섹션 4에서 역할만 언급).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `hls/MatMul_SA/mmult_accel.hpp` — 인터페이스
- `extern "C" void mmult_accel(const int8_t *A, const int8_t *B, int32_t *C, int N, int K, int M, int update_A)` (`mmult_accel.hpp:21-29`).
- 의미론: `A[N×K]`, `B[K×M]`, `C[N×M]`(row-major). `update_A` 플래그로 A 재적재 제어.
- 주석(`mmult_accel.hpp:14-15`)에 "16x16 INT8 systolic array"라 적혀 있으나 **실제 구현은 32×32 타일**(아래 .cpp 참조) — 헤더 주석과 구현 불일치(헤더가 구버전 잔재로 추정).

### 3.2 `hls/MatMul_SA/mmult_accel.cpp` — 탑레벨 GEMM 커널 (가장 중요)

#### (a) 파라미터/자료형 (`mmult_accel.cpp:11-38`)
- `MAX_N=64`, `MAX_K=768`, `MAX_M=768` — 지원 최대 차원. DistilBERT-base hidden=768과 정합(`:11-13`).
- `BLOCK_M=256` — B를 256 컬럼 블록 단위로 처리(온칩 메모리 절감) (`:21`).
- `TILE_SIZE=32` — 계산 타일(출력 32×32, 부분합 레지스터) (`:29`).
- `DTYPE_IN=int8_t`, `DTYPE_OUT=int32_t` — **INT8 입력 / INT32 누산 출력**(에지 양자화) (`:37-38`).

#### (b) 인터페이스 프래그마 (`mmult_accel.cpp:66-81`)
- `#pragma HLS INTERFACE m_axi` × 3: A→`gmemA`(depth MAX_N·MAX_K), B→`gmemB`(MAX_K·MAX_M), C→`gmemC`(MAX_N·MAX_M). **세 개의 독립 AXI 마스터 번들**로 동시 메모리 접근.
- `#pragma HLS INTERFACE s_axilite ... bundle=control`: 모든 스칼라 인자 + return을 단일 control 번들로 → 호스트(PYNQ)가 레지스터로 제어.

#### (c) A 상주 BRAM (`mmult_accel.cpp:88-102`)
- `static DTYPE_IN A_bram[MAX_N][MAX_K]` + `#pragma HLS BIND_STORAGE type=ram_2p impl=bram` (`:88-89`).
- `static` + `update_A` 분기(`:93-102`)로 **호출 간 A 지속**: 첫 호출(`update_A=1`)에만 A를 DDR→BRAM 복사(`copy_A` 루프 `II=1`), 이후 호출은 재사용.
- **핵심 설계 의도**: Self-Attention에서 같은 입력 X(=A)에 Q/K/V 세 가중치(B)를 곱하므로, **A를 한 번만 로드하고 B만 바꿔 3회 호출** → DDR 트래픽 1/3 절감(테스트벤치가 정확히 이 패턴 검증, 3.3 참조).

#### (d) B 컬럼 블록 처리 (`mmult_accel.cpp:109-124`)
- `outer_j_block`: `j_block`을 `BLOCK_M`(256) 간격으로 순회, `current_block_M`로 잔여 처리(`:110-111`).
- `DTYPE_IN B_bram[MAX_K][BLOCK_M]` (`:114`, `ram_2p impl=bram`) 에 현재 블록만 DDR→BRAM 적재(`copy_B_block`, `II=1`) (`:118-124`).
- **B 전체를 온칩에 안 올리고 256컬럼씩** → BRAM 사용량 상한 고정(에지 자원 적합).

#### (e) 타일드 행렬곱 본체 (`mmult_accel.cpp:131-227`)
- `tile_i`(N을 `TILE_SIZE` 간격), `tile_j`(블록 내 M을 `TILE_SIZE` 간격) 이중 루프 (`:131-134`).
- **출력 타일 레지스터화**: `DTYPE_OUT localC[32][32]` + `#pragma HLS ARRAY_PARTITION dim=0 complete` → 1024개 레지스터로 완전 분해(`:139-140`). `init_c`를 2중 `UNROLL`로 병렬 0초기화(`:144-151`).
- **입력 타일 버퍼**: `localA[32][32]`, `localB[32][32]` 모두 `ARRAY_PARTITION dim=0 complete`(`:155-158`).
- **K 타일 루프 `k_loop`** (`:165-214`):
  - `loadA`(`:169-180`): `A_bram`→`localA`, 경계 검사 후 미달은 0 패딩, `II=1`.
  - `loadB`(`:184-195`): `B_bram`→`localB`, 동일.
  - **`compute` MAC 어레이** (`:199-213`): 외부 `kk`(K, `II=1` 파이프라인) × `ii`(32, `UNROLL`) × `jj`(32, `UNROLL`). 즉 **한 사이클에 32×32=1024 MAC 동시 수행**, K 방향으로 누산. `a_val=(int32)localA[ii][kk]`를 행당 1회 캐스트 후 `localC[ii][jj] += a_val * b_val`.
  - 이 구조가 **출력-고정(output-stationary) 32×32 MAC 어레이**의 본질. 합성 리포트에서 `compute` 인스턴스가 **DSP 1024개**(=32×32) 사용으로 확인(`mmult_accel_csynth.rpt:93`).
- **`writeC`** (`:218-227`): `localC`→DDR `C`, 경계 검사, `II=1`.

#### (f) 합성 결과 근거 (`mmult_accel_csynth.rpt`)
- Target clock 10ns(100MHz), Estimated 8ns(`:23`).
- **Total 자원**(`:77`): BRAM_18K=126(43%), DSP=1040(83%), FF=102741(43%), LUT=71046(60%), URAM=0(0%).
  - `compute` 단독: DSP 1024, FF 32783, LUT 27969 (`:93`) → 어레이가 DSP 대부분 점유.
  - 메모리: `A_bram` 24 BRAM(49152 words×8b), `B_bram` 96 BRAM(196608 words) (`:112-115`).
- 루프 레이턴시: `loadA/loadB` 각 1026 cycle, `writeC` 1032, `compute` 36(`:43-46`). 전체 탑레벨 레이턴시는 차원 의존이라 `?`(가변, `:32`, `tile_j max ~1.43e11` `:56`).
- **DSP 83%가 병목 자원** → KV260에서 어레이를 더 키우기 어려운 상한.

### 3.3 `hls/MatMul_SA/mmult_accel_tb.cpp` — 테스트벤치(검증 + 성능)
- **레퍼런스** `reference_mmult` (`:59-71`): 64비트 누산 CPU 행렬곱(`long sum`), 오버플로 방지.
- **테스트 케이스** (`:96-108`): 기본 `{16,768,768},{32,768,768},{64,768,768}`(N=토큰 수, K=M=768=DistilBERT hidden). `FAST_COSIM`은 `{8,64,64}`.
- **Q/K/V 프로젝션 시나리오** (`:184-375`): 동일 A에 대해
  - Q: `mmult_accel(A,B_q,C_hw,N,K,M,update_A=1)` — A를 BRAM에 적재 (`:204`).
  - K: `...,B_k,...,update_A=0` — A 재사용 (`:269`).
  - V: `...,B_v,...,update_A=0` — A 재사용 (`:333`).
  - 각 단계 CPU 레퍼런스와 전수 비교(`C_hw[i]!=C_sw[i]`), GFLOPs 측정(`ops=2·N·M·K`).
- **HLS_COSIM**(`:39-43, 198-215`): `clock_start/end`로 사이클 정확 성능, 10ns/100MHz 가정. 일반 모드는 `std::chrono`.
- 메모리는 `MAX_*` 깊이로 할당하고 미사용부 0패딩(`:117-174`) → HLS 인터페이스 depth 정합.
- **의미**: 이 TB가 곧 TMMA의 핵심 활용 패턴(어텐션 Q/K/V 가중치 공유)을 그대로 모사하는 정확도/성능 검증 하니스.

### 3.4 보조 자산(노트북/Vivado — 역할만)
- `pynq/Quant_distilBERT_forward-pass.ipynb`: 양자화 DistilBERT 포워드패스(추정: INT8 변환·FPGA 오프로드).
- `pynq/TMMA_pynq_benchmark.ipynb`: KV260 런타임 벤치마크.
- `pynq/FLOPs_and_MACs_Calculation_for_DistilBERT-Base.pdf`: 연산량 산정서.
- `model/distilBert/*.ipynb`: PyTorch DistilBERT 레이어 shape 검증/CPU 참조.
- `vivado/2-27_bd&df/`: 블록디자인/구현 디바이스뷰 이미지(비트스트림·프로젝트 미포함, `README.md:72-73`).

---

## 4. 데이터플로우 / 실행 흐름

```
호스트(PYNQ, KV260 ARM):
  X(=A) 1회 → mmult_accel(update_A=1, B=W_Q) → C_Q
            → mmult_accel(update_A=0, B=W_K) → C_K   (A는 BRAM 상주 재사용)
            → mmult_accel(update_A=0, B=W_V) → C_V

커널 내부(mmult_accel):
  [update_A] DDR A → A_bram (1회)
  for j_block in M step 256:        # B 컬럼 블록
     DDR B[block] → B_bram
     for i0 in N step 32:           # 출력 행 타일
       for j0 in block step 32:     # 출력 열 타일
         localC[32][32] = 0 (완전 언롤)
         for k0 in K step 32:       # K 타일
           A_bram→localA, B_bram→localB
           compute: kk(II=1) × ii(unroll32) × jj(unroll32)  # 1024 MAC/cycle
         localC → DDR C
```

### 메모리 계층
- **DDR(외부)**: A/B/C, 세 독립 AXI 마스터(`gmemA/B/C`).
- **온칩 BRAM**: `A_bram`(64×768, 24 BRAM), `B_bram`(768×256, 96 BRAM) — 상주/블록 (`csynth:112-113`). URAM 미사용.
- **레지스터**: `localA/localB/localC`(각 32×32, complete partition) — MAC 어레이 직결.

### 병렬화 / dataflow
- `compute`의 ii/jj 완전 언롤 = **공간 병렬 32×32 MAC**(출력-고정), kk는 `II=1` 시간 파이프라인 누산.
- 단, 탑레벨에 `#pragma HLS dataflow`는 **없음** → load/compute/write가 순차(타일 단위 직렬). acap-gemm-sa의 이중버퍼 중첩과 대비되는 단순 구조.

### 양자화 / 데이터타입
- **INT8 입력 × INT8 → INT32 누산**(`:37-38, 205-210`). DistilBERT를 PTQ로 INT8 양자화하여 FPGA 오프로드(노트북 근거, 추정).
- `compute`에서 `(int32)` 캐스트 후 곱·누산 → 오버플로 안전.

---

## 5. HW/SW 매핑

| 구성요소 | 물리 위치 | 역할 | 근거 |
|---|---|---|---|
| `mmult_accel` | KV260 PL(HLS IP) | INT8 타일드 GEMM 본체 | `mmult_accel.cpp:58` |
| `compute` 32×32 | PL DSP(1024개) | 출력-고정 MAC 어레이 | `csynth.rpt:93` |
| `A_bram` | PL BRAM(24) | 입력 X 상주 | `mmult_accel.cpp:88` |
| `B_bram` | PL BRAM(96) | Q/K/V 가중치 블록 | `mmult_accel.cpp:114` |
| `gmemA/B/C` | DDR↔PL AXI4 마스터 | 외부 메모리 I/O | `mmult_accel.cpp:66-68` |
| control | AXI-Lite | 호스트 제어 레지스터 | `mmult_accel.cpp:74-81` |
| PYNQ 노트북 | KV260 ARM(PS) | 양자화/오프로드/벤치 | `pynq/*.ipynb` |
| Vivado BD | PL+PS 통합 | 블록디자인(이미지만) | `vivado/2-27_bd&df/` |

- 클럭 100MHz 타깃(`csynth.rpt:23`). DSP 83% 점유로 KV260 자원 거의 한계.

---

## 6. 빌드·실행 (README 근거)

HLS (`hls/MatMul_SA/README.md:30-49`):
```
vitis_hls -f mmult_accel.cpp          # HW 함수 합성
g++ -o mmult_accel_tb mmult_accel_tb.cpp -I/path/to/hls/include
./mmult_accel_tb                       # TB 실행
```
배포 (`README.md:34-41`):
1. `hls/`에서 Vivado HLS로 가속기 빌드 → IP.
2. Vivado 블록디자인으로 하드웨어 플랫폼 재구성(비트스트림 직접 생성, 레포 미포함).
3. `pynq/` 노트북으로 KV260 배포·벤치마크.
- 서브모듈 초기화 필요(`.gitmodules` → `git submodule update --init`).

---

## 7. 의존성

- **툴체인**: Vivado/Vitis HLS(2023 계열, KV260 `xck26` 파트), Vivado 합성/구현, PYNQ 런타임(KV260).
- **HLS 라이브러리**: `ap_int.h`, `hls_stream.h`, `stdint.h`(`mmult_accel.cpp:1-3`). (실제 본문은 hls_stream 미사용 — include만.)
- **SW/모델**: PyTorch(DistilBERT), 양자화 파이프라인, numpy(노트북, 추정).
- **참고문헌**(`README.md:63-70`): Attention Is All You Need, DistilBERT, SOCC2020 Transformer Accel, FlightLLM(FPGA'24), SSR(FPGA'24).
- **라이선스**: MIT (`LICENSE.md`, `README.md:75-76`).

---

## 8. 강점 / 한계 / 리스크

### 강점
- **A 상주 + Q/K/V 가중치 공유 최적화**(`update_A`, `mmult_accel.cpp:88-102`): 어텐션 프로젝션의 입력 재사용을 정확히 포착, DDR 트래픽 절감. 도메인 특화의 핵심 가치.
- **INT8/INT32 에지 양자화**: KV260급 소형 디바이스에 적합한 정밀도/자원 균형.
- **B 컬럼 블록(256)**: 온칩 메모리 상한 고정으로 큰 M에도 BRAM 폭주 방지.
- **명확한 출력-고정 32×32 MAC 어레이**: 1024 DSP 직매핑, 구조 단순·합성 안정.
- **완결된 검증 하니스**: TB가 실제 Q/K/V 시나리오 + CPU 레퍼런스 + GFLOPs 측정.
- **풀스택 자산**: HLS + Vivado BD + PYNQ 벤치 + PyTorch 참조까지 연구 재현 경로 제공.

### 한계
- **DSP 83% 병목**(`csynth.rpt:77`): 어레이 확장 여력 거의 없음. 처리량 상한 고정.
- **탑레벨 dataflow 부재**: load/compute/write 순차 → 통신-계산 미중첩(acap-gemm-sa 대비 처리량 손해).
- **헤더-구현 불일치**: hpp 주석 "16x16"(`mmult_accel.hpp:14`) vs 실제 32×32 — 문서 신뢰성 주의.
- **고정 차원 상한**(MAX_N=64, K=M=768): DistilBERT 외 모델/긴 시퀀스 일반화 제한. FFN·Softmax 미포함(README 로드맵상 진행 중).
- **WIP**: 비트스트림/Vivado 프로젝트 미포함, 일부 미완(`README.md:5-8, 55-58`).

### 리스크
- 100MHz 추정 클럭 기반 성능(`csynth.rpt:23`) — 실제 KV260 달성 주파수/대역폭은 "확인 불가".
- B 컬럼 블록 + 비dataflow 구조에서 DDR 재적재 오버헤드가 큰 M에서 누적될 수 있음(추정).

---

## 9. 우리 프로젝트(HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

> TMMA는 KV260 에지 + DistilBERT Q/K/V에 특화된 **소형 FPGA INT8 GEMM**으로, HG-PIPE 계열(고처리량 ViT)과 타깃·정밀도가 직접 겹침. 재사용 가치 높음.

1. **입력 상주 + 가중치 스왑 패턴**(`update_A`): ViT 어텐션의 Q/K/V 프로젝션, MLP의 두 FC도 동일 입력에 다른 가중치를 곱하는 구조 → HG-PIPE 파이프라인에서 활성(activation) 온칩 상주 + 가중치 스트리밍 설계의 직접 레퍼런스.
2. **INT8×INT8→INT32 양자화 데이터패스**: 우리 ViT/Mamba 양자화 가속기의 MAC 정밀도 정책과 직결. PTQ 기반 DistilBERT 오프로드 노트북(`pynq/Quant_distilBERT_*`)은 양자화→FPGA 매핑 절차의 참고 자료.
3. **출력-고정 32×32 MAC 어레이 + DSP 직매핑**: 1024 DSP=83% 점유라는 정량 자원 모델은, KV260급에서 우리 가속기 PE 어레이 크기 상한 산정의 현실 기준선.
4. **B 컬럼 블록(BLOCK_M) 기법**: 큰 출력 차원을 블록 스트리밍으로 처리하는 패턴은 ViT의 큰 channel/MLP hidden 차원 GEMM에 적용 가능.
5. **개선 포인트 식별(반면교사)**: 탑레벨 dataflow·이중버퍼 부재가 명확한 한계 → 우리 HG-PIPE 설계에서는 (acap-gemm-sa식) 통신-계산 중첩을 결합하면 TMMA 대비 처리량 우위 확보 가능(설계 차별화 근거).
6. **XR 시선추적 연계**(추정): TMMA 자체는 NLP(DistilBERT)지만, 시선추적이 경량 ViT/Transformer 백본을 쓸 경우 그 어텐션 프로젝션 GEMM 가속에 동일 커널 구조 차용 가능. 단, KV260 에지 폼팩터는 XR 디바이스 제약과 부합.
7. **풀스택 재현 경로**: HLS→Vivado BD→PYNQ→PyTorch 비교의 워크플로는 우리 가속기 검증 파이프라인 구성에 템플릿으로 활용.

---

## 10. 근거 표기

- **확인된 사실(라인 근거)**: 타깃 KV260/`xck26`, INT8/INT32, TILE_SIZE=32, BLOCK_M=256, MAX(64,768,768), A 상주(`update_A`), 32×32 출력-고정 MAC, DSP 1040(83%)/BRAM 126, 100MHz 타깃, Q/K/V TB 시나리오, arXiv:2503.16731 출처 — 모두 위 인용 라인 직접 확인.
- **추정**: 노트북 내부 양자화 절차, B 재적재 오버헤드 영향, XR/HG-PIPE 연계성, 헤더 "16x16" 주석이 구버전 잔재라는 해석.
- **확인 불가**: 실제 KV260 합성 후 달성 주파수/실측 GFLOPs/전력, 비트스트림·Vivado 프로젝트(레포 미포함), 노트북 셀 실행 결과 수치(.ipynb 미실행 분석).
