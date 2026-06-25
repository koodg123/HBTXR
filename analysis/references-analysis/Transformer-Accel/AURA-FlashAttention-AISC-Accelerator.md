# AURA-FlashAttention-AISC-Accelerator (Transformer-Accel) — 교차참조

> 대상 repo: `REF/Transformer-Accel/AURA-FlashAttention-AISC-Accelerator`
> 작성일: 2026-06-20 / 방식: Glob 파일트리 대조 + 대표 파일(README, expmul_stage.sv) 직접 확인

## 결론: **동일 (사실상 같은 repo)**

본 repo의 설계 본체(RTL/C++/Python/합성/문서)는
**`REF/Analysis/ViT-Accelerator/AURA-FlashAttention-AISC-Accelerator.md`** 의 분석 대상과 **동일**하다.
설계 분석 전체는 위 기존 .md를 그대로 참조할 것. (중복 분석 생략)

## 동일 근거

- **소스 트리 완전 일치**: `verilog/*.sv`(AURA, PE, dot_product, tree_reduce, max, expmul, expmul_stage, vector_division, int_division, memory_controller, K/V/Q/OSRAM, q_convert 계열), `cpp/*`, `python/*`, `test/*`, `synth/AURAsynth.tcl`, `Makefile`, `include/sys_defs.svh`, `_deprecated/*` 까지 파일 구성·이름 동일. (양쪽 Glob 대조)
- **README 동일**: 동일한 문제정의/동기/Related Work(SwiftTron, ExpMul, FLASH-D 언급). ASIC·edge·FlashAttention 지향 그대로. (`README.md:1-11`)
- **핵심 커널 동일**: `expmul_stage.sv` 의 ExpMul 근사 주석/공식 `Log2Exp(X) = −⌊X + (X≫1) − (X̂≫4)⌉` 동일. (`verilog/expmul_stage.sv:12`)
- **부수 파일 동일**: `LICENSE`, `docs/AURA_RTL.pdf`, `initialnovas.rc`, `scratch.txt` 양쪽 모두 존재.

## 차이점 (생성 데이터 한정, 설계 무관)

설계/RTL/SW 코드 차이는 **없음**. 유일한 차이는 `models/` 하위 **테스트 벡터 데이터**와 GitHub 이슈 템플릿 일부뿐:

| 항목 | Transformer-Accel (본 repo) | ViT-Accelerator (기존 분석) |
|---|---|---|
| `models/bert-base-uncased/` | 존재 | 존재 (동일) |
| `models/bert-large-cased/` | **존재** (`.mem`+`.dec` 다수) | 없음 |
| `models/random_test1/` | 없음 | **존재** (`O_fixed_clean.mem` 등) |
| `.github/ISSUE_TEMPLATE` | research/verilog/verification/bug 등 다수 | documentation/config 위주(일부) |

→ `models/*`는 `Generate_QKV.py`/`fp32_to_f8` 가 만드는 **자동 생성 산출물**(기존 .md §2.2에서 분석 제외 명시). 설계 동일성에 영향 없음.

## 반환 요약

- **동일 여부**: 동일 (소스/문서 일치, 생성 데이터만 상이)
- **참조**: `REF/Analysis/ViT-Accelerator/AURA-FlashAttention-AISC-Accelerator.md`
- **차이 1줄**: `models/` 테스트벡터만 다름(본 repo는 bert-large-cased 포함, random_test1 없음). RTL/SW 코드 차이 없음.
