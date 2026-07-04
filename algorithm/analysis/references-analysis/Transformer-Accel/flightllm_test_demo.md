# FlightLLM `flightllm_test_demo` 정밀 분석

분석 대상: `REF/Transformer-Accel/flightllm_test_demo`
분석 도구: Glob / Read (bash 불가 — UNC 경로 제약)
분석 일자: 2026-06-20
근거 표기 규칙: **확인** = 실제 파일/라인 근거 있음 / **추정** = 정황 근거 / **확인 불가** = 소스 부재로 검증 불가

---

## 0. 핵심 결론 (가장 먼저 읽을 것)

> **이 repo에는 자체 소스코드(.py/.cpp/.h/.c/RTL/HLS/Makefile/tcl/셸)가 단 하나도 없다.**
> 전부 **사전 컴파일된 배포 산출물(deployment artifacts)** 로만 구성된 "닫힌 데모 번들"이다.

확인된 전체 구성 (Glob 전수 조사 결과, 아래 파일이 repo의 전부):

| 종류 | 파일 | 성격 | 근거 |
|------|------|------|------|
| 호스트 실행파일 | `fpga_implementation/host/fpgaHost` | x86-64 ELF 바이너리 (소스 없음) | **확인** (Read 시 `ELF ... /lib64/ld-linux-x86-64.so.2` 헤더) |
| FPGA 비트스트림 | `fpga_implementation/bitstream/stc-v1.xclbin` | Vitis/XRT xclbin (생성물) | **확인** (파일명·위치) |
| RTL 명령어 스트림 | `.../case/token_64_single_layer/inst/inst.slr{0,1,2}.rtl.bin` | SLR별 instruction 바이너리 | **확인** |
| RTL 입력 데이터 | `.../input/input.ch{00..28}.rtl.bin` (8개, 4채널 간격) | HBM 채널별 입력 텐서 | **확인** |
| RTL 파라미터 | `.../param/param.ch{00..28}.rtl.bin` (8개) | HBM 채널별 가중치 | **확인** |
| RTL 출력(기댓값) | `.../output/output.ch{00..28}.rtl.bin` (8개) | golden 출력 | **확인** |
| 어텐션 마스크 | `profile/.compiler_output/ir_output/attention_mask/attention_mask_{0..31}.mask.npy` (32개) | 컴파일러 IR 산출 마스크 | **확인** (NumPy 헤더 판독) |
| 설정 | `.gitignore` (내용: `__pycache__` 1줄) | — | **확인** |

→ **분석의 본질은 "코드 리뷰"가 아니라 "바이너리 아티팩트 역설계 기반 추론"이다.** 함수/커널/모듈 단위 라인 분석은 **소스 부재로 수행 불가**하며, 본 문서는 파일 구조·바이너리 헤더·디렉토리 규약(naming convention)에서 읽어낼 수 있는 사실만 근거로 기술한다.

---

## 1. 개요

### 1.1 프로젝트 정체
- **FlightLLM** (ISCA/FPGA'24, Infinigence/칭화대): sparse LLM을 FPGA(AMD Alveo U280급)에서 가속하는 "Mapping Flow + Configurable Sparse DSP Chain + Always-On-Chip Decode" 아키텍처. (배경 지식 — repo 내 논문/README는 **확인 불가**)
- 본 `flightllm_test_demo`는 그 **테스트/데모 배포본**으로 추정. (**추정** — 디렉토리명 `flightllm_test_demo` + 산출물 구성이 정확히 "비트스트림 + 호스트 + 케이스 데이터" 패턴)

### 1.2 데모 케이스 식별
- 단일 케이스: `token_64_single_layer` (**확인** — 디렉토리명)
  - "token 64 / single layer" → **시퀀스 길이 64 토큰, Transformer 1개 레이어** 추론을 검증하는 최소 케이스. (**추정** — 명명 규칙)
- 어텐션 마스크 shape = **`(32, 32, 32)`**, dtype `|b1`(bool) (**확인** — `attention_mask_0.mask.npy` NumPy v1.0 헤더: `{'descr': '|b1', 'fortran_order': False, 'shape': (32, 32, 32)}`)
  - 해석: **헤드 32개 × 32×32 토큰 어텐션 행렬**, 또는 32×32 블록 × 32 단위. 마스크 파일이 0~31번 32개 존재 → 레이어/헤드/타일 인덱스 32개. (**추정** — 정확한 축 의미는 컴파일러 소스 부재로 단정 불가)
  - 주의: 마스크 32×32×32는 "token_64"와 직접 일치하지 않음. 64토큰을 32-블록 타일로 쪼갠 블록-스파스(block-sparse) 표현일 가능성. (**추정**)

### 1.3 타깃 HW (산출물에서 역추론)
- **다중 SLR (Super Logic Region) 디바이스**: `inst.slr0/slr1/slr2` → **3개 SLR** = AMD Alveo U280(3 SLR) 계열. (**확인**=파일명 / **추정**=U280 단정)
- **HBM 다채널**: 채널 인덱스 `ch00, ch04, ..., ch28` (4씩 증가, 8개) → 최소 29채널 주소공간 중 8채널 사용 또는 4채널 그룹핑. U280 HBM은 32 pseudo-channel → `ch00..ch31` 체계와 정합. (**확인**=채널명 / **추정**=U280 HBM 매핑)
- 비트스트림 `stc-v1.xclbin`의 `stc` = sparse transformer core 등으로 **추정** (**확인 불가**).

---

## 2. 디렉토리 구조

### 2.1 자체(분석 대상) — **없음**
자체 작성 소스코드 0건. (**확인** — `*.py / *.cpp / *.h / *.c / *.sh / *.tcl / *.v / *.sv / *.md / Makefile` 전부 Glob 결과 0건)

### 2.2 전체 트리 (산출물 = 모두 "생성물/제외 대상"이지만 데모 이해상 명시)

```
flightllm_test_demo/
├── .gitignore                         # "__pycache__" (한 줄)  [확인]
├── fpga_implementation/
│   ├── bitstream/
│   │   └── stc-v1.xclbin              # [생성물] FPGA 비트스트림
│   ├── host/
│   │   └── fpgaHost                   # [생성물] x86-64 ELF 호스트 실행파일 (소스 없음)
│   └── case/
│       └── token_64_single_layer/
│           ├── inst/  inst.slr{0,1,2}.rtl.bin     # [생성물] SLR별 명령어 (3개)
│           ├── input/ input.ch{00,04,08,12,16,20,24,28}.rtl.bin   # [생성물] 채널별 입력 (8개)
│           ├── param/ param.ch{00,04,08,12,16,20,24,28}.rtl.bin   # [생성물] 채널별 파라미터 (8개)
│           └── output/output.ch{00,04,08,12,16,20,24,28}.rtl.bin  # [생성물] 채널별 골든 출력 (8개)
└── profile/
    └── .compiler_output/             # (숨김 디렉토리)
        └── ir_output/
            └── attention_mask/
                └── attention_mask_{0..31}.mask.npy   # [생성물] 어텐션 마스크 32개
```

### 2.3 제외 항목 (third-party/vendor/생성물 — 이름만 언급, 분석 안 함)
- `bitstream/stc-v1.xclbin` — Xilinx/AMD xclbin (FPGA 합성 산출물)
- `host/fpgaHost` — 컴파일된 ELF (XRT/OpenCL 링크 추정, **확인 불가**)
- `*.rtl.bin` (27개) — 컴파일러/패커가 만든 바이너리 입력/명령/파라미터/출력
- `*.mask.npy` (32개) — 컴파일러 IR(`ir_output`) 중간 산출물

> **참고**: 보통 FlightLLM 류 데모는 호스트 소스(`host.cpp`), Makefile, run 스크립트, README, 컴파일러(파이썬)를 동봉한다. **이 번들에는 그 어느 것도 없다** — 즉, "실행 가능한 닫힌 산출물만" 배포된 형태. (**확인** — Glob 전수)

---

## 3. 핵심 모듈 정밀 분석

> ⚠️ **소스코드가 없으므로 함수/커널/모듈 단위 라인 분석은 불가능하다.** 본 절은 (a) 바이너리 헤더에서 직접 읽은 사실, (b) 파일 명명 규칙이 강제하는 구조적 사실만 근거로 기술한다. 추론에는 모두 **추정** 표기.

### 3.1 호스트 실행파일 `fpgaHost` — 바이너리 헤더 분석 (**확인**)
- Read 결과 첫 바이트가 `ELF` 매직 + `>`(little-endian, 64-bit) + 동적 링커 `/lib64/ld-linux-x86-64.so.2` 문자열 포함.
  - 근거: `fpga_implementation/host/fpgaHost:1` (ELF 매직 `7f 45 4c 46`, `EM_X86_64`, INTERP `/lib64/ld-linux-x86-64.so.2`)
- **사실로 확정되는 것**:
  - x86-64 Linux 동적 링크 실행파일. 정적이 아님 → 런타임에 공유 라이브러리(XRT `libxrt_core`, OpenCL, pthread 등) 필요할 것. (라이브러리 목록은 `.dynamic`/`.dynstr` 미파싱으로 **확인 불가**, 정황상 **추정**)
- **추정되는 역할 (FlightLLM 호스트 드라이버 일반 패턴)**:
  1. `xclbin` 로드 → FPGA 프로그래밍
  2. `inst.slr*.rtl.bin`을 명령어 버퍼로, `param.ch*` / `input.ch*`를 HBM 채널 버퍼로 DMA 전송
  3. 커널 실행(enqueue) → 완료 대기
  4. 결과를 읽어 `output.ch*` (golden)와 비교 (test/verify 흐름)
  - 이 4단계는 **추정**이며, 소스 없이는 실제 API 호출 시퀀스 **확인 불가**.

### 3.2 명령어 스트림 `inst.slr{0,1,2}.rtl.bin` (**확인**=존재 / **추정**=의미)
- **3개 SLR 각각에 독립 명령어 스트림** → FlightLLM이 SLR 단위로 가속기 인스턴스(또는 파이프 스테이지)를 복제/분산 배치하고, **SLR별로 별도 instruction을 스케줄링**함을 시사. (**추정**)
- "RTL bin" = RTL(실HW)이 직접 소비하는 인코딩된 명령어. FlightLLM의 컴파일러가 모델 그래프를 **사용자 정의 instruction set**(matmul/softmax/layernorm/load/store 등)으로 낮춘 뒤 SLR별로 파티셔닝한 결과로 **추정**.
- instruction 포맷/opcode는 바이너리만으로 디코딩 불가 → **확인 불가**.

### 3.3 데이터/파라미터 채널 바이너리 `*.ch{NN}.rtl.bin` (**확인**=존재 / **추정**=의미)
- `input` / `param` / `output` 3종이 **동일한 채널 인덱스 집합 `{00,04,08,12,16,20,24,28}`** 을 공유 → 입력·가중치·출력이 **같은 HBM 채널 토폴로지로 인터리빙**되어 배치됨. (**확인**=인덱스 일치)
- 4 간격(00,04,...,28) → HBM pseudo-channel을 4개씩 그룹핑하거나, 8개의 연산 lane이 각각 전용 채널을 갖는 구조로 **추정**.
- `output.ch*`가 함께 들어있는 것 → 이 데모가 **자가 검증(self-check)** 목적임을 강하게 시사 (실측 vs golden). (**추정**)

### 3.4 어텐션 마스크 `attention_mask_{0..31}.mask.npy` (**확인**=헤더)
- NumPy v1.0, `descr='|b1'`(bool), `shape=(32,32,32)`, C-order.
  - 근거: `profile/.compiler_output/ir_output/attention_mask/attention_mask_0.mask.npy:1`
- **확정 사실**:
  - 32개의 bool 텐서, 각 32×32×32 = 32,768 bit ≈ 4 KB/파일.
  - bool 마스크 = **block-sparse 어텐션** 또는 causal/padding 마스크의 비트맵. FlightLLM의 핵심인 "configurable sparse" 어텐션을 위한 **희소성 패턴 메타데이터**로 **추정**.
  - 위치가 `ir_output`(컴파일러 IR 단계 산출) → 호스트/HW가 아니라 **컴파일 타임에 미리 결정된 정적 sparsity**임을 시사. (**추정**)
- 32라는 수: 헤드 수(32) 또는 레이어/타일 인덱스. token_64 케이스이므로 64-토큰을 32-크기 블록으로 분할한 어텐션의 블록 마스크일 가능성. (**추정**, 확정 불가)

### 3.5 비트스트림 `stc-v1.xclbin` (**확인**=존재 / 내용 분석 불가)
- xclbin은 메타데이터(`.json` 섹션) + 비트스트림을 담는 컨테이너지만, 본 분석 도구로 내부 IP-LAYOUT/MEM_TOPOLOGY 섹션을 파싱하지 않음 → 커널 이름·HBM 토폴로지 **확인 불가**.
- 파일명 `stc-v1` → "Sparse Transformer Core v1" 등으로 **추정**.

---

## 4. 데이터 플로우 (산출물 기반 재구성)

소스가 없어 호스트 코드 흐름은 **추정**이나, 파일 토폴로지가 강제하는 데이터 경로는 다음과 같다:

```
[컴파일 타임 — 이 repo에 컴파일러 소스 없음, 산출물만 존재]
 모델(가중치) + 그래프
        │  (FlightLLM 컴파일러, repo 외부)  [확인 불가]
        ▼
 ir_output/attention_mask/*.npy   ← sparsity 패턴 (정적)        [확인: npy 존재]
 case/.../inst/inst.slr{0,1,2}    ← SLR별 명령어 스트림          [확인]
 case/.../param/param.ch*         ← HBM 채널별 가중치             [확인]

[런타임 — fpgaHost(ELF)가 수행, 내부 동작은 추정]
 fpgaHost  ──load──▶  stc-v1.xclbin → FPGA program             [추정]
    │
    ├─DMA(H2D)─▶ HBM ch00..28 : param.ch* + input.ch*          [추정]
    ├─push─────▶ SLR0/1/2 instruction queues : inst.slr*       [추정]
    ├─run──────▶ 커널 실행 (sparse matmul/attn/FFN, 1 layer)    [추정]
    └─DMA(D2H)─▶ 결과 ← 비교 ← output.ch* (golden)             [추정]
```

- **정적 sparsity → 컴파일 타임 결정 → instruction에 인코딩 → SLR별 분산 실행 → HBM 채널 병렬 I/O** 가 이 데모가 드러내는 핵심 파이프라인 구조. (**추정**, 산출물 토폴로지 근거)

---

## 5. HW/SW 매핑

| 계층 | 산출물 | 역할(추정) | 근거 |
|------|--------|-----------|------|
| SW (호스트) | `fpgaHost` | XRT/OpenCL로 xclbin 로드·DMA·실행·검증 | **확인**=ELF / **추정**=역할 |
| SW→HW 인터페이스 | `*.rtl.bin` | 호스트가 HW에 주입하는 명령/데이터 패킷 | **추정** |
| HW 명령 | `inst.slr{0,1,2}` | SLR별 가속기 instruction stream | **확인**=존재 / **추정**=의미 |
| HW 메모리 | `*.ch{NN}` | HBM pseudo-channel별 텐서 배치 | **확인**=채널명 / **추정**=HBM 매핑 |
| HW 로직 | `stc-v1.xclbin` | sparse transformer 커널 비트스트림 | **확인**=존재 / 내부 **확인 불가** |
| 컴파일러 산출 | `*.mask.npy` | 정적 sparsity 메타데이터 | **확인**=헤더 |

- **HW/SW 경계**: 호스트(x86) ↔ FPGA(3 SLR, HBM) — 전형적인 Alveo 데이터센터 카드 모델. (**추정**)
- **이 repo가 보여주지 못하는 것**: PE 배열 구조, 양자화 비트폭(INT4/INT8 등), instruction opcode, softmax/layernorm 하드웨어 구현 — 모두 **확인 불가**. (FlightLLM 논문/원본 RTL 필요)

---

## 6. 빌드·실행

- **빌드 산출물만 존재. 빌드 시스템 부재.** Makefile/CMake/tcl/스크립트 0건 → 이 번들만으로 재빌드 **불가**. (**확인** — Glob 0건)
- **실행**: 정확한 실행 커맨드 라인(인자, xclbin 경로 지정 방식)을 적은 README/run 스크립트가 없어 **확인 불가**. 일반 패턴상 `./fpgaHost <xclbin> <case_dir>` 형태로 **추정**.
- **재현성**: golden `output.ch*`가 동봉되어 자가 검증은 가능하나, 환경(XRT 버전, 드라이버, 카드)이 맞아야 함 — 동봉 정보 없음 → **확인 불가**.

---

## 7. 의존성

- 직접 확인: 동적 ELF (`/lib64/ld-linux-x86-64.so.2`) → glibc + 공유 라이브러리 런타임 의존. (**확인**)
- 추정 의존: AMD XRT(`libxrt_core`, `libxrt_coreutil`), OpenCL ICD, pthread, 표준 C++ 런타임. (**추정** — `.dynstr` 미파싱)
- 데이터 포맷: NumPy `.npy` v1.0 (마스크). (**확인**)
- **확인 불가**: Python 버전, 컴파일러 의존성, Vitis/Vivado 버전 — 해당 소스/메타 전무.

---

## 8. 강점 · 한계

### 강점 (이 번들이 보여주는 것)
- **End-to-end 배포 가능 형태**: 비트스트림+호스트+케이스 데이터+golden이 한 묶음 → 즉시 실행/검증 지향 데모. (**확인**=구성)
- **명확한 HW 토폴로지 단서**: 3 SLR + 다채널 HBM + SLR별 instruction 분리 → 멀티-die 파티셔닝 설계가 산출물 레벨에서 명시됨. (**확인**=명명)
- **정적 sparsity 메타데이터 노출**: `ir_output/attention_mask` 가 컴파일러→HW 파이프라인의 중간 표현을 엿보게 함. (**확인**)

### 한계 (분석 관점)
- **소스 0건 → 화이트박스 분석 불가**. 함수/커널/모듈/라인 근거 분석은 원천적으로 수행 불가. (**확인**)
- 빌드/실행 스크립트·README 부재 → 재현·재빌드 불가. (**확인**)
- instruction/xclbin/rtl.bin 포맷 비공개 → 알고리즘(양자화, PE dataflow, sparse matmul 방식) 검증 불가. (**확인 불가**)
- 본 repo는 "참조 구현"이 아니라 "참조 산출물"에 가까움 — 설계 학습용 가치는 토폴로지 단서로 제한됨.

---

## 9. 우리 프로젝트(PRJXR-HBTXR: 고처리량 ViT/Transformer FPGA 가속기[HG-PIPE 계열] + XR 시선추적) 시사점

> 우리 프로젝트 성격은 **추정**(디렉토리/형제 분석 문서 정황). 아래는 그 가정 하의 시사점.

1. **배포 번들 설계의 레퍼런스로 활용**: 우리도 FPGA 산출물 배포 시 `bitstream + host + case(input/param/inst/golden)` 구조를 따르면 자가 검증형 데모를 깔끔히 패키징 가능. (산출물 구성은 **확인**된 좋은 선례)
2. **멀티-SLR 파티셔닝**: HG-PIPE류 고처리량 파이프라인을 U280급(3 SLR)에 올린다면, **SLR별 instruction/데이터 분리** 패턴이 직접 참고가 됨. 우리 파이프라인 스테이지를 SLR 경계로 나누는 설계 결정에 근거 제공. (**추정**)
3. **HBM 채널 인터리빙(ch 4간격)**: ViT 가속에서 입력/가중치/출력을 동일 채널 토폴로지로 배치해 대역폭을 lane별로 균등 분배하는 전략 참고. (**추정**)
4. **정적 sparsity vs XR 실시간성**: FlightLLM은 **컴파일 타임 정적 마스크**(`ir_output`)를 쓴다. XR 시선추적은 프레임마다 동적이므로, 이 정적-sparsity 접근은 **그대로는 부적합** — 우리는 동적/구조적 sparsity 또는 dense 고처리량(HG-PIPE식 fully-pipelined) 쪽이 더 맞음. (**추정**, 중요한 차별점)
5. **반면교사**: 소스·빌드·문서 없는 "닫힌 번들"은 후속 연구/재현에 치명적. 우리 산출물은 **반드시 소스+빌드+README+실행스크립트를 동봉**해야 함. (**확인**된 한계의 교훈)

> ⚠️ **주의**: FlightLLM은 LLM(decode 위주, autoregressive) 가속기다. 우리의 ViT/시선추적(encoder 위주, 저지연 단발 추론)과는 워크로드 특성이 달라, 아키텍처를 직접 차용하기보다 **패키징·파티셔닝·검증 방법론** 수준에서 참고하는 것이 타당. (**추정**)

---

## 10. 근거 표기 요약

| 항목 | 상태 | 근거 위치 |
|------|------|----------|
| 자체 소스 0건 | **확인** | Glob 전수(`.py/.cpp/.h/.c/.sh/.tcl/.v/.sv/.md/Makefile` 0건) |
| `fpgaHost`=x86-64 ELF | **확인** | `fpga_implementation/host/fpgaHost:1` (ELF 헤더 + `/lib64/ld-linux-x86-64.so.2`) |
| 어텐션 마스크 shape (32,32,32) bool | **확인** | `profile/.compiler_output/ir_output/attention_mask/attention_mask_0.mask.npy:1` |
| 3 SLR / 다채널 HBM | **확인**(명명) / **추정**(U280) | `inst.slr{0,1,2}`, `*.ch{00..28}` |
| 케이스 = token_64_single_layer | **확인** | 디렉토리명 |
| 호스트 실행 흐름(load/DMA/run/verify) | **추정** | 산출물 구성 정황 |
| xclbin 내부 IP/메모리 토폴로지 | **확인 불가** | xclbin 섹션 미파싱 |
| instruction/rtl.bin 포맷·opcode | **확인 불가** | 비공개 바이너리 |
| 양자화 비트폭·PE dataflow·sparse matmul 방식 | **확인 불가** | 소스/RTL 부재 |
| 빌드/실행 커맨드 | **확인 불가** | 빌드 시스템·README 부재 |

---

## 부록: 사용 분석 명령
- `Glob`: repo 경로 하위 `*`, `**`, 확장자 필터(`*.py`, `*.{cpp,h,c}`, `*.{sh,tcl,mk,...}`, `*.{so,whl,pyc,...}` 등) — 자체 소스 0건 확인
- `Read`: `.gitignore`(전체), `fpgaHost`(ELF 헤더 5줄), `attention_mask_0.mask.npy`(NumPy 헤더 2줄)
- bash 계열(`mcp__workspace__bash`)은 UNC 경로 미지원으로 사용 불가 확인
