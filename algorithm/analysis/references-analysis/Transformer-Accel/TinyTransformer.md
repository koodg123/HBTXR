# TinyTransformer 코드베이스 정밀 분석

> 대상 repo: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\Transformer-Accel\TinyTransformer`
> 분석 일자: 2026-06-20
> 분석 방법: Glob/Grep/Read (bash 미사용)
> 근거 표기 규칙: **확인** = 실제 파일 라인 근거 / **추정** = 정황 기반 / **확인 불가** = 파일 부재로 검증 불가

---

## 0. 핵심 결론 (가장 먼저)

**이 repo는 사실상 비어 있다.** 추적 가능한 파일은 `.gitignore` **단 1개**뿐이며, 실제 소스 코드(`.py`, `.cpp`, `.h`, `.v`, `.sv`, HLS 커널, RTL, Makefile, 노트북 등)는 **하나도 존재하지 않는다.**

- **확인**: `\\...\TinyTransformer\.gitignore` (22행) — repo 내 유일하게 발견된 파일.
- **확인**: 다음 Glob 패턴 전부 "No files found" 반환 (TinyTransformer 경로 한정):
  - `**/*.py`, `**/*.{cpp,h,hpp,c,v,sv,ipynb,md,json,yaml,yml,toml,cfg,ini,sh,tcl}`
  - `**/Makefile`, `**/*.{txt,hex,pth,pt,log,csv}`, `**/*.ipynb`
  - `*/` (서브디렉토리), `TinyTransformer/.*` (점파일)
- **확인**: `*` (최상위) Glob 결과 = `.gitignore` 1건만.

따라서 "TinyTransformer가 SW 모델인지 HW 가속기인지"를 **소스 코드로 직접 판정하는 것은 불가능**하다. 아래 분석은 유일한 단서인 `.gitignore`의 내용에서 도출 가능한 **간접 근거**만을 기술하며, 모두 **추정**으로 명시한다.

---

## 1. 개요

TinyTransformer는 디렉토리 골격과 `.gitignore`만 커밋된 상태로, **실질적 작업물이 형상관리(git)에 포함되지 않은 빈 저장소**다. 일반적으로 이런 상태는 다음 중 하나를 의미한다(**추정**):

1. 작업 초기 단계에서 골격(`.gitignore`)만 먼저 푸시하고 본 코드는 아직 미작성/미커밋.
2. 본 코드가 로컬에만 존재하거나 `.gitignore`에 의해 통째로 제외됨(예: 산출물 디렉토리 패턴에 걸림).
3. 노트북(Colab/Jupyter) 기반 실험 repo로, 실제 작업물은 `.ipynb`였으나 `.ipynb_checkpoints`만 무시되고 정작 노트북 본체는 커밋 누락.

`.gitignore` 주석이 베트남어(`Caches và log của Colab/Jupyter`, `.gitignore:6` 확인)로 작성되어 있어, **외부 베트남어권 작성자의 학습/실습용 repo**일 가능성이 있다(**추정**). "TinyTransformer"라는 명칭과 Colab/Jupyter 친화 설정으로 미루어 **PyTorch 기반 소형 Transformer SW 학습 모델 + 하드웨어용 weight 추출(.hex) 실험**을 의도했던 repo로 보인다(**추정**).

---

## 2. 디렉토리 구조

### 2.1 자체 소스 (실재)

```
TinyTransformer/
└── .gitignore        # 유일 파일 (22행) — 확인
```

- 서브디렉토리: **없음** (확인 — `*/` Glob "No files found").
- `__pycache__/`, `data/`, `venv/`, `env/` 등은 `.gitignore`에 선언만 되어 있을 뿐 **물리적으로 존재 확인 불가**(생성/커밋 안 됨).

### 2.2 third-party / vendor / 생성물 (제외 대상)

- **해당 없음.** vendor 디렉토리, 서브모듈, 생성물(.pth/.hex/.log 등) 모두 **물리적으로 존재하지 않음**(확인 — 해당 패턴 Glob 전부 "No files found").
- 단, `.gitignore`가 *제외하도록 선언한* 항목은 다음과 같다(아래 §7 참조). 이들은 "제외 의도"일 뿐 실재하지 않는다.

---

## 3. 핵심 모듈 정밀 분석

> 분석 대상 소스 파일이 **0개**이므로 함수/클래스/모듈/RTL/HLS 커널 단위 정밀 분석은 **수행 불가**다. attention/FFN/embedding/양자화 로직, RTL 모듈, HLS 커널 어느 것도 **코드 레벨에서 확인 불가**.

유일 분석 가능 산출물인 `.gitignore`를 라인 단위로 해부한다.

### 3.1 `.gitignore` 라인별 정밀 분석

파일: `\\...\TinyTransformer\.gitignore` (전체 22행, 확인)

| 라인 | 내용 | 해석 |
|---|---|---|
| 1 | `# 1. Byte-compiled / Optimized / DLL files` | Python 산출물 섹션 헤더 → **Python 프로젝트**임을 시사 (확인) |
| 2 | `__pycache__/` | CPython 바이트코드 캐시 → Python 사용 (확인) |
| 3 | `*.py[cod]` | `.pyc/.pyo/.pyd` 컴파일 산출물 (확인) |
| 4 | `*$py.class` | Jython 클래스 (관용적 표준 패턴) (확인) |
| 6 | `# 2. Caches và log của Colab/Jupyter` | **베트남어 주석**. "Colab/Jupyter의 캐시와 로그" → **노트북 기반 워크플로** (확인) |
| 7 | `.ipynb_checkpoints/` | Jupyter 체크포인트 디렉토리 (확인) |
| 8 | `*.ipynb_checkpoints` | 체크포인트 파일 패턴 (확인) |
| 10 | `# 3. Model weights & Data` | **모델 가중치/데이터 섹션** → ML 학습 repo (확인) |
| 11 | `/data` | 루트 `data/` 디렉토리(데이터셋) 제외 (확인) |
| 12 | `*.pth` | **PyTorch 가중치 직렬화** 포맷 → **PyTorch 프레임워크** (확인) |
| 13 | `*.pt` | PyTorch 모델/텐서 포맷 (확인) |
| 14 | `*.hex` | **HEX 메모리 이미지** → FPGA/ASIC ROM/BRAM 초기화용 가중치 덤프 가능성 → **HW 연계 의도** (확인된 패턴, 용도는 추정) |
| 15 | `*.txt` | 텍스트 산출물(로그/덤프) 제외 (확인) |
| 16 | `*.log` | 로그 파일 (확인) |
| 18 | `# 4. Virtual Environment` | 가상환경 섹션 (확인) |
| 19 | `venv/` | Python venv (확인) |
| 20 | `env/` | 대체 venv 디렉토리명 (확인) |
| 21 | `.env` | 환경변수 파일 (확인) |

### 3.2 코드 부재 항목 (전부 확인 불가)

| 분석 요구 항목 | 상태 | 근거 |
|---|---|---|
| Attention (QKV, softmax) | **확인 불가** | `.py`/`.cpp`/RTL 0개 |
| FFN / MLP | **확인 불가** | 동일 |
| Embedding / Positional Encoding | **확인 불가** | 동일 |
| 양자화(INT8/PTQ/QAT) | **확인 불가** | 동일 |
| RTL 모듈(.v/.sv) | **확인 불가** | `.v`/`.sv` 0개 |
| HLS 커널(.cpp + pragma) | **확인 불가** | `.cpp`/`.h` 0개 |
| 빌드 스크립트(Makefile/tcl) | **확인 불가** | Makefile/.tcl 0개 |

---

## 4. 데이터 플로우

**확인 불가.** 소스가 없어 실제 텐서/데이터 흐름(입력 → embedding → attention → FFN → 출력, 또는 HW 측 DMA → PE 배열 → 누산기)을 추적할 수 없다.

`.gitignore` 기반 **추정 워크플로**(검증 불가):
```
(추정) Colab/Jupyter 노트북에서 PyTorch로 소형 Transformer 학습
   → 가중치를 .pth/.pt 로 저장 (.gitignore:12-13)
   → 가중치를 .hex 로 변환/덤프 (.gitignore:14)  ← HW 메모리 초기화용 추정
   → (가설) FPGA/RTL/HLS 측에서 .hex 를 BRAM/ROM 으로 로드
```
위 화살표 중 `.pth → .hex` 변환 스크립트, HW 측 로더 모두 **파일로 확인되지 않음**.

---

## 5. HW/SW 매핑

**코드 레벨 매핑 불가.** SW 측(PyTorch)·HW 측(RTL/HLS) 어느 쪽 소스도 없다.

- **SW 단서**: `*.pth`/`*.pt`(.gitignore:12-13), `__pycache__`(.gitignore:2), Colab/Jupyter(.gitignore:6-8) → **PyTorch SW 모델 학습 환경**이 명백히 의도됨 (확인).
- **HW 단서**: `*.hex`(.gitignore:14) **단 하나**. HEX는 통상 FPGA/ASIC 메모리 초기화 포맷이므로 **HW 가속기 연계 가능성**을 시사하나, RTL/HLS 파일이 전무하여 **HW 구현체 존재 여부는 확인 불가**.
- **종합 판정(추정)**: TinyTransformer는 "HW 가속기 본체"라기보다 **SW(PyTorch) 학습 모델 + HW용 가중치 추출을 염두에 둔 경량 Transformer 실험 repo**일 가능성이 높다. 단, **현 상태로는 SW/HW 어느 쪽도 코드로 입증 불가**.

---

## 6. 빌드·실행

**확인 불가.** Makefile, `setup.py`, `requirements.txt`, `pyproject.toml`, tcl, 빌드 스크립트, 노트북 어느 것도 존재하지 않는다(확인 — 해당 Glob 전부 "No files found"). 빌드·실행 절차를 기술할 근거가 없다.

`.gitignore`로부터 **추정**되는 실행 환경: Colab/Jupyter + Python venv (`venv/`, `env/`, `.env` — .gitignore:19-21). 즉 노트북 셀 실행 방식이었을 것(**추정**).

---

## 7. 의존성

**확인 불가** (의존성 명세 파일 부재). `.gitignore` 패턴에서 **추정**되는 스택:

- **Python** (확인 — `__pycache__`, `*.py[cod]`)
- **PyTorch** (확인 — `*.pth`, `*.pt` 포맷; 이 포맷은 사실상 PyTorch 전용)
- **Jupyter/Colab** (확인 — `.ipynb_checkpoints`, 주석 라인 6)
- 그 외 구체적 패키지(`torch` 버전, `numpy`, `timm` 등)는 **명세 파일 없어 확인 불가**.

---

## 8. 강점·한계

### 강점
- 분석할 코드 자산이 없어 "강점"을 코드 근거로 제시할 수 없음. 유일한 긍정 요소는 `.gitignore`가 ML 산출물(.pth/.pt/.hex/data)을 적절히 제외하도록 **구조화된 섹션 주석**으로 잘 작성되어 있다는 점(.gitignore:1,6,10,18 — 4개 섹션 헤더, 확인).

### 한계 (치명적)
- **소스 코드 전무**: 모델/가속기 구현체가 git에 없음 (확인). 본 repo만으로는 어떠한 기술적 재현·이식·학습도 불가.
- **참고 가치 거의 없음**: 우리 프로젝트(FPGA Transformer 가속기 + XR 시선추적)에 직접 이식하거나 벤치마크할 코드가 0개.
- **정보 추적 불가**: README/docs 없음 → 의도/스코프/완성도 추정만 가능.

---

## 9. 우리 프로젝트 시사점

> 우리 프로젝트(추정): **고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적**.

- **현 상태로는 활용 가치 사실상 없음.** TinyTransformer에는 ViT/attention/FFN/양자화/RTL/HLS 어떤 구현도 없어, HG-PIPE 계열 파이프라인 설계나 XR 시선추적 모델 어디에도 차용할 코드가 없다 (확인 — 소스 0개).
- **유일한 시사점(추정)**: `.gitignore`의 `*.hex` 패턴(.gitignore:14)은 "SW에서 학습한 Transformer 가중치를 HEX 메모리 이미지로 떨궈 FPGA BRAM/ROM에 적재"하는 워크플로를 암시한다. 우리 프로젝트에서도 **PyTorch weight → (양자화) → .hex/.coe → BRAM 초기화** 흐름이 필요하므로, 이 명명/제외 컨벤션 정도가 참고 수준의 약한 시사점이다. 단, 변환 스크립트 자체가 없으므로 **구현 참조는 불가**.
- **권고**: 이 repo는 REF 인벤토리에서 "빈 저장소(코드 없음)"로 기록하고, 분석 우선순위에서 제외할 것. 만약 원작자의 노트북/가중치가 별도로 존재한다면(로컬·Colab) 그 산출물을 직접 확보해야만 실질 분석이 가능하다.

---

## 10. 근거 표기 요약

| 분류 | 항목 | 근거 |
|---|---|---|
| **확인** | repo 내 유일 파일 = `.gitignore` (22행) | `*` Glob 결과 |
| **확인** | `.py/.cpp/.h/.v/.sv/Makefile/.ipynb` 등 소스 0개 | 다중 Glob "No files found" |
| **확인** | Python + PyTorch + Jupyter 의도 | `.gitignore:2,6-8,12-13` |
| **확인** | 베트남어 주석(외부 작성자 정황) | `.gitignore:6` |
| **확인** | `*.hex` 제외 = HW 메모리 이미지 의도 단서 | `.gitignore:14` |
| **추정** | SW(PyTorch) 학습 + HW용 weight 추출 실험 repo | `.gitignore` 종합 |
| **추정** | Colab/Jupyter + venv 실행 환경 | `.gitignore:6-8,19-21` |
| **확인 불가** | attention/FFN/embedding/양자화/RTL/HLS, 데이터플로우, 빌드, 의존성 명세, HW/SW 매핑 | 해당 소스 전부 부재 |

---

### 분석 한계 고지
본 repo는 코드 자산이 부재하여 "정밀 분석" 요건(함수/클래스/모듈/RTL/HLS 라인 근거)을 충족할 대상이 물리적으로 존재하지 않는다. 위 문서는 가용한 단일 파일(`.gitignore`)에 한해 라인 근거를 제시했으며, 그 외 모든 기술 항목은 "추정" 또는 "확인 불가"로 정직하게 표기했다.
