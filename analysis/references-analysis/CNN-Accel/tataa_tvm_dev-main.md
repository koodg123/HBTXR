# tataa_tvm_dev-main 정밀 분석

> 대상 repo: `REF/CNN-Accel/tataa_tvm_dev-main`
> 분석 일자: 2026-06-20
> 분석 도구: Glob/Grep/Read (bash 미사용, UNC 경로 제약 준수)

---

## 0. 핵심 결론 (먼저 읽을 것)

**이 repo는 디렉토리명("tataa_tvm_dev")과 달리, 실제로는 Apache TVM `0.23.dev0` 개발 브랜치의 거의 순수(vanilla) 체크아웃이며, TATAA 관련 자체 추가/수정 코드는 본 디렉토리 트리 내에서 발견되지 않았다.**

- `tataa` / `TATAA` 키워드: `src/`, `apps/`, `tests/`, `docs/`, `CMakeLists.txt`, `python/` 어디에서도 **0건** (근거: 아래 §3 검색 로그).
- `transformable` 키워드: 매칭 없음(0건). `bfloat16`/`systolic`/`mixed precision` 등 TATAA 연상 키워드는 매칭되나, 전부 **TVM 원본 코어의 표준 기능**(예: `topi/nn/conv2d.py`, `relax/transform/to_mixed_precision.cc`)에 해당하며 TATAA 가속기와 무관함.
- 커스텀 FPGA target / codegen / relax·tir 패스 / contrib 추가분: **없음**. `python/tvm/relax/backend/` 하위는 표준 백엔드(cuda, rocm, metal, adreno, nnapi, cpu_generic 등)만 존재.
- README.md: Apache TVM 표준 README 원문 그대로(TATAA 언급 없음).

따라서 본 문서는 (a) "이 트리에는 TATAA 자체분이 없다"는 사실을 라인 근거와 함께 확정하고, (b) TATAA 논문/아키텍처가 무엇인지 외부 지식 기반으로 짧게 정리하며, (c) 우리 프로젝트(HG-PIPE 계열 ViT FPGA 가속기 + XR 시선추적) 관점에서 이 "TVM 베이스"가 갖는 시사점을 정리한다.

> 추정: 디렉토리명으로 보아 이 repo는 TATAA 연구진이 **컴파일러 베이스로 fork만 떠 놓고** 아직 TATAA 커스텀 패스/타깃을 커밋하지 않았거나, 자체 추가분이 **이 스냅샷에 포함되지 않은(별도 브랜치/별도 repo)** 상태일 가능성이 높다. `.git` 디렉토리도 존재하지 않아(§3) 브랜치/커밋 이력으로 추가 확인은 **확인 불가**.

---

## 1. 개요

| 항목 | 내용 |
|---|---|
| 목적 | (선언상) TATAA 가속기를 위한 TVM 기반 컴파일 환경. **실제 트리에는 TATAA 자체분 부재** |
| 한줄요약 | Apache TVM 0.23.dev0 vanilla 체크아웃 (Relax + TensorIR 크로스레벨 컴파일러). TATAA 커스텀 코드 미발견 |
| 베이스 프레임워크 | Apache TVM (오픈소스 ML 컴파일러). 본 분석에서 **TVM 원본 코어는 "외부 베이스"로 처리** |
| TVM 버전 | `0.23.dev0` (근거: `version.py:48` `__version__ = "0.23.dev0"`) |
| 원논문(추정) | TATAA: *Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture* (FPGA 분야, INT8 systolic + bfloat16 부동소수 혼합정밀 트랜스포머 가속기) — **이 repo 내부에 논문/문서 근거 없음, 외부 지식 기반 추정** |
| 타깃 디바이스(추정) | FPGA (TATAA 아키텍처). 단, **이 트리 내 FPGA target/codegen 근거 없음 → 확인 불가** |

---

## 2. 디렉토리 구조 (TATAA 자체 추가분 vs TVM 베이스)

```
tataa_tvm_dev-main/
├── README.md            # ← TVM 표준 README (TATAA 무관)  [TVM 베이스]
├── version.py           # ← __version__ = "0.23.dev0"      [TVM 베이스]
├── CMakeLists.txt       # ← TATAA 문자열 0건                [TVM 베이스]
├── .gitmodules          # ← dmlc-core, cutlass, zlib, tvm-ffi 등 표준 서브모듈만 [TVM 베이스]
├── 3rdparty/            # ← [제외 대상] cutlass, picojson, mlperftiny, tensorrt_llm 등
├── apps/                # ← android_rpc, cpp_rpc, hexagon_*, ios_rpc (전부 TVM 표준) [TVM 베이스]
├── src/                 # ← relax/transform 등 전부 표준 TVM 패스 [TVM 베이스]
├── python/tvm/          # ← relax, tir, topi, contrib(msc/cutlass/hexagon) 전부 표준 [TVM 베이스]
├── tests/               # ← 표준 TVM 테스트 [TVM 베이스]
├── docs/                # ← 표준 TVM 문서 [TVM 베이스]
└── (.git 디렉토리 없음 → 커밋 이력 추적 불가)
```

**TATAA 자체 추가분: 식별된 파일/디렉토리 없음.**

검증한 핵심 영역(모두 표준 TVM):
- `python/tvm/relax/transform/*.py` (33개): `fast_math.py`, `fuse_transpose_matmul.py`, `legalize_ops/`, `optimize_layout_transform.py` 등 — TVM 표준 Relax 패스. TATAA 패스 없음.
- `src/relax/transform/*.cc` (57개): `to_mixed_precision.cc`, `fuse_ops.cc`, `combine_parallel_matmul.cc`, `run_codegen.cc` 등 — TVM 표준. TATAA 패스 없음.
- `python/tvm/relax/backend/*`: `cuda/`, `rocm/`, `metal/`, `adreno/`, `contrib/nnapi.py`, `cpu_generic/`, `gpu_generic/` — 표준 백엔드만. **FPGA/TATAA 백엔드 없음**.
- `python/tvm/contrib/*`: `cutlass/`, `hexagon/`, `msc/`(Model Serving Compiler), `cblas/cudnn/mkl` 등 — 전부 TVM 표준 contrib. TATAA contrib 없음.

---

## 3. 핵심 모듈·파일별 정밀 분석 (= TATAA 추가분 부재 입증)

TATAA 자체 추가분이 없으므로, 본 섹션은 **"없음"을 입증하는 검색 근거**로 대체한다. (지시: 모르면/없으면 근거와 함께 명시)

### 3.1 키워드 검색 로그 (라인 근거)

| 검색 키워드 | 검색 경로 | 결과 |
|---|---|---|
| `tataa\|TATAA` | `src/` | **No files found** (0건) |
| `tataa\|TATAA\|transformable` (-i) | `apps/` | **No files found** (0건) |
| `tataa\|TATAA\|transformable` (-i) | `tests/` | **No files found** (0건) |
| `tataa\|TATAA\|transformable` (-i) | `python/` | **No files found** (0건) |
| `tataa\|TATAA\|Transformable` (-i) | `docs/` | **No files found** (0건) |
| `tataa\|TATAA` (-i) | `CMakeLists.txt` | **No matches found** (0건) |
| glob `**/*tataa*` | repo 전체 | **No files found** (TATAA 명명 파일 0개) |
| glob `**/*TATAA*` | repo 전체 | **No files found** |

> 참고: repo 전체 단일 Grep은 TVM 트리 규모로 20초 타임아웃 발생 → 위와 같이 `src/`, `apps/`, `tests/`, `python/`, `docs/`, 루트 파일로 분할 검색하여 전 영역 커버. 모두 0건.

### 3.2 "TATAA 연상" 키워드가 매칭된 파일 (전부 TVM 표준 — TATAA 무관)

`bfloat16|systolic|mixed.?precision` (-i) 검색 시 `python/`에서 다음이 매칭되나 **전부 TVM 원본 코어**임:
- `python/tvm/topi/nn/conv2d.py`, `dense.py`, `batch_matmul.py`, `conv2d_transpose.py` 등 → TOPI 표준 연산 정의(dtype 인자로 bfloat16 등을 받을 뿐).
- `python/tvm/relax/transform/transform.py` → TVM 표준 `ToMixedPrecision` 패스 래퍼.
- `python/tvm/tir/tensor_intrin/metal.py`, `script/ir_builder/tir/ir.py` → 표준 TIR intrinsic.

이들은 TVM이 원래 제공하는 일반 혼합정밀/데이터타입 기능으로, **TATAA의 "INT8 systolic ↔ bfloat16 transformable" 아키텍처 매핑과는 무관**(코드 내 TATAA 하드웨어/ISA 연결 없음).

### 3.3 빌드/메타 근거
- `version.py:48` → `__version__ = "0.23.dev0"` : 최신 TVM dev 브랜치 확정.
- `.gitmodules` (전체 8개 서브모듈) → `dmlc-core`, `rang`, `cutlass`, `OpenCL-Headers`, `cnpy`, `cutlass_fpA_intB_gemm`, `libflash_attn`, `zlib`, `tvm-ffi` : **전부 TVM 표준 서브모듈**. TATAA/FPGA 관련 서브모듈 없음.
- `.git` 디렉토리 부재(glob `.git/*` → No files found) → fork 출처/커스텀 커밋 이력 **확인 불가**.

---

## 4. 데이터플로우 / 컴파일 흐름

TATAA 커스텀 흐름은 **이 트리에 없다**. 베이스 TVM의 표준 흐름만 존재한다(외부 베이스이므로 요약만):

```
모델(PyTorch/ONNX)
  → Relax frontend (relax/frontend/torch/fx_translator.py 등)   [TVM 표준]
  → Relax IRModule (그래프 레벨 IR)
  → Relax transform 패스들 (fuse_ops, to_mixed_precision, legalize_ops ...)  [TVM 표준]
  → TensorIR(TIR) 레벨로 lower
  → codegen (run_codegen.cc / BYOC)   [TVM 표준, CUDA/LLVM/Metal 등]
  → 런타임 모듈
```

- **TATAA 양자화/데이터타입 매핑(INT8 linear + bfloat16 nonlinear) 패스: 없음 → 확인 불가.**
- 표준 `ToMixedPrecision`(`src/relax/transform/to_mixed_precision.cc`)이 fp16/bf16 혼합정밀을 지원하나, 이는 TATAA의 "transformable arithmetic"(같은 MAC 배열을 INT8/bf16로 시분할 전환)과는 **개념·구현 모두 별개**.

---

## 5. HW/SW 매핑 (TATAA 하드웨어 ISA와의 연결)

**이 트리 내에서 TATAA 하드웨어 ISA / 명령어 생성 / FPGA target과의 연결 코드: 발견되지 않음 → 확인 불가.**

- TVM 표준 BYOC/codegen은 존재하나, TATAA 전용 target(`tvm.target.Target("tataa")` 등) 정의·등록 없음.
- `python/tvm/relax/backend/`에 FPGA/TATAA 백엔드 디렉토리 부재.
- 따라서 "Relax/TIR → TATAA 명령어" 매핑은 본 repo 스냅샷 기준 **구현 미존재**.

> 참고(외부 지식, 추정): TATAA 아키텍처 자체는 INT8 systolic array를 bfloat16 vector 유닛으로 "변형(transform)"하여 트랜스포머의 선형(matmul, INT8)과 비선형(Softmax/GELU/LayerNorm, bf16)을 단일 PE 배열로 처리하는 것이 핵심. 그 ISA/RTL은 **별도 하드웨어 repo에 존재할 것으로 추정**되며 본 TVM 트리에는 없음.

---

## 6. 빌드·실행

표준 TVM 빌드 절차를 따른다(본 repo에 TATAA 전용 빌드 옵션 없음 — `CMakeLists.txt`에 TATAA 문자열 0건):
- CMake로 `libtvm` 빌드 → `python/` 패키지 설치(`pip install -e .` 또는 `PYTHONPATH`).
- 서브모듈(`3rdparty/`) 초기화 필요(dmlc-core, cutlass, tvm-ffi 등).
- **TATAA 전용 실행 스크립트/앱: 없음** (`apps/`는 RPC/hexagon/ios 표준만).

---

## 7. 의존성

`.gitmodules` 기준(전부 TVM 표준): `dmlc-core`, `rang`, `cutlass`, `cutlass_fpA_intB_gemm`, `OpenCL-Headers`, `cnpy`, `libflash_attn`, `zlib`, `tvm-ffi`. Python 측은 표준 TVM 의존(numpy, ml_dtypes 등). **TATAA/FPGA 전용 의존성 추가 없음.**

---

## 8. 강점 / 한계 / 리스크

**강점 (베이스 TVM 관점)**
- 최신 TVM 0.23.dev0 — Relax(그래프) + TensorIR(텐서) 크로스레벨 IR, Python-first 커스텀 패스 작성 용이.
- BYOC, `ToMixedPrecision`, `legalize_ops`, meta-schedule 등 가속기 통합에 재사용 가능한 인프라가 풍부.

**한계 / 리스크**
- **치명적 한계: 이 스냅샷에는 TATAA 자체 코드가 전혀 없다.** "TATAA 컴파일 흐름 참고"를 기대하고 이 repo를 분석하면 얻을 것이 없음.
- `.git` 부재로 fork 베이스 커밋·출처 추적 불가 → 어떤 TATAA 브랜치에서 분기했는지 **확인 불가**.
- TATAA 실제 컴파일러 자산(커스텀 패스/타깃/codegen)은 **별도 위치에 있을 가능성** — 본 repo만으로는 미완/공백.

---

## 9. 우리 프로젝트 관점 시사점
(우리 프로젝트: ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적 — 추정)

이 repo가 TATAA 자체분 없는 vanilla TVM이라는 점을 감안할 때, **TATAA 컴파일 흐름의 직접 재사용은 불가**하다. 다만 "TVM 베이스" 자체에서 우리 프로젝트에 끌어다 쓸 수 있는 부분은 있다:

1. **Relax → TIR 크로스레벨 IR + 커스텀 패스 작성 패턴**
   - 우리가 ViT를 FPGA 가속기로 내릴 때, `python/tvm/relax/transform/`의 패스 작성 스타일(예: `fuse_transpose_matmul.py`, `optimize_layout_transform.py`)을 템플릿 삼아 HG-PIPE용 레이아웃/퓨전 패스를 작성 가능.
   - 추정: HG-PIPE의 파이프라인 친화적 레이아웃(채널 블로킹 등)을 `convert_layout`/`alter_op_impl` 계열 패스로 모델링 가능.

2. **BYOC(`run_codegen.cc`) + 커스텀 target 등록**
   - 우리 FPGA 가속기 전용 codegen을 붙이려면 TVM BYOC 경로가 표준 진입점. 단, **이 repo에 예시(TATAA target)가 없으므로** 별도 TVM BYOC 튜토리얼/HG-PIPE 자체 흐름을 참조해야 함.

3. **혼합정밀 처리(`ToMixedPrecision`)**
   - ViT 양자화(INT8 linear) + 비선형(LayerNorm/Softmax/GELU)을 부동소수로 처리하는 우리 설계와 개념적으로 닮음. TATAA의 "transformable" 사상을 우리 패스로 직접 구현할 때, TVM의 dtype 전파/캐스팅 삽입 로직(`infer_amp_utils.cc`, `to_mixed_precision.cc`)이 참고 자료가 됨.

4. **재사용 시 주의**: 이 repo에서 "TATAA가 어떻게 했는가"를 배우는 것은 불가능(코드 부재). 우리는 TATAA *논문*과 (있다면) TATAA *하드웨어/컴파일러 별도 repo*를 따로 확보해야 함. → REF 인벤토리에 "TATAA 컴파일러 자산 위치 확인" 액션 아이템 추가 권장.

---

## 10. 근거 표기 규칙

- **확정 사실**: Read/Grep/Glob 결과로 직접 확인한 항목(파일경로·라인번호 명시). 예: `version.py:48`, "src/ 검색 0건".
- **추정**: 외부 지식(TATAA 논문 일반 상식)·디렉토리명 추론에 기반. 본문에 "추정"으로 명시.
- **확인 불가**: 본 repo 스냅샷 내 근거 부재로 단정 불가한 항목(예: fork 베이스 커밋, TATAA 하드웨어 ISA 연결, 타깃 FPGA 보드). 본문에 "확인 불가"로 명시.

---

### 부록: 본 분석에서 실제 Read/검증한 근거 파일
- `README.md` (1–66행, TVM 표준 README 확인)
- `version.py` (`__version__ = "0.23.dev0"`)
- `.gitmodules` (서브모듈 8종, 전부 표준)
- `python/tvm/relax/transform/*.py` 디렉토리 목록 (33개, 표준)
- `src/relax/transform/*.cc` 디렉토리 목록 (57개, 표준)
- `python/tvm/relax/backend/*` 디렉토리 목록 (표준 백엔드만)
- `python/tvm/contrib/*` 디렉토리 목록 (표준 contrib만)
- `apps/*` 디렉토리 목록 (RPC/hexagon/ios 표준)
- Grep: `tataa/TATAA/transformable` (src, apps, tests, python, docs, CMakeLists) → 전부 0건
- Glob: `**/*tataa*`, `**/*TATAA*`, `.git/*` → 모두 없음
