# A.U.R.A. — FlashAttention ASIC Accelerator (교차참조)

> 분석 대상 repo: `REF/Transformer-Accel/A.U.R.A.---FlashAttention-ASIC-Accelerator`
> 작성일: 2026-06-20
> **결론: 이 repo는 set-2 에서 이미 정밀 분석된 `REF/ViT-Accelerator/AURA-FlashAttention-AISC-Accelerator` 와 동일 프로젝트다. 중복 전체분석을 피하기 위해 본 문서는 짧은 교차참조(cross-ref)로 갈음한다.**

---

## 1. 동일성 판정 (코드 직접 대조)

작업 지시(세트2의 AURA와 동일 가능 → 핵심파일 Read 비교: 동일하면 짧은 cross-ref)에 따라 핵심 파일을 직접 Read 하여 두 repo의 동일성을 확인했다.

| 대조 항목 | 본 repo (`A.U.R.A.---FlashAttention-ASIC-Accelerator`) | 기존 분석 repo (`AURA-FlashAttention-AISC-Accelerator`) | 판정 |
|---|---|---|---|
| `README.md` 본문 | "A.U.R.A. is a SystemVerilog based ASIC accelerator for the FlashAttention kernel ..." / SwiftTron·ExpMul·FLASH-D 언급 (`README.md:2,7,10`) | 동일 문구 (기존 분석 §1, §8 근거) | **동일** |
| `include/sys_defs.svh` 파라미터 | `INTEGER_WIDTH 8`(`:22`), `MAX_EMBEDDING_DIM 64`(`:24`), `MAX_SEQ_LENGTH 512`(`:26`), `MEM_BLOCK_SIZE_BITS 64`(`:28`), `NUM_PES 4`(`:36`), `NUM_TILES = 512/4`(`:38`), `MAX_NUM_PES` 산식(`:34`) | 기존 분석 §1·§3.0과 동일 수치 | **동일** |
| `verilog/` RTL 모듈 집합 | `AURA.sv, PE.sv, dot_product.sv, tree_reduce.sv, reduction_step.sv, max.sv, expmul.sv, expmul_stage.sv, vector_division.sv, int_division.sv, memory_controller.sv, KSRAM/VSRAM/QSRAM/OSRAM.sv, q_convert/q_align_frac/q_align_int/q_saturate/q_sign_extend.sv` (Glob 확인) | 기존 분석 §2.1 RTL 트리와 1:1 일치 | **동일** |
| `cpp/` SW 참조모델 집합 | `attention_fp32.cpp, attention_f8.cpp, fp32_to_f8.cpp, fp32_to_f16.cpp, input_to_f8.cpp, output_to_f8.cpp, generate_mem.cpp, generate_output_f8.cpp, generate_output_fp64.cpp, precision_measure.cpp, precision_measuref16.cpp` (Glob 확인) | 기존 분석 §2.1 cpp 트리와 동일 | **동일** |
| `python/` (Generate_QKV.py 등), `test/`(aura_test.sv, mem.sv, *_test.sv), `synth/AURAsynth.tcl`, `Makefile`, `models/` 디렉토리 | 모두 존재 (Glob 확인) | 기존 분석 §2.1·§6 동일 | **동일** |

> **유일한 차이**: 디렉토리 이름 표기뿐이다. 본 repo는 `A.U.R.A.---FlashAttention-ASIC-Accelerator`(점·하이픈 포함, "ASIC" 정확 표기), 기존은 `AURA-FlashAttention-AISC-Accelerator`("AISC" 오타 표기). 내용물(소스/파라미터/구조)은 동일 프로젝트의 동일 스냅샷으로 **판정**한다. (단, 두 repo의 git 커밋 해시까지 바이트 단위로 일치하는지는 미대조 → 사소한 리비전 차이 가능성은 "확인 불가".)

추가로 본 repo에는 `docs/AURA_RTL.pdf`, `.vscode/settings.json`, `initialnovas.rc`(Novas/Verdi 초기화), `convert_to_decimal.sh`, `models/{bert-base-uncased,bert-large-cased,random_test1}/*.{mem,dec}` 생성 테스트벡터가 함께 존재한다(Glob 확인). 즉 본 스냅샷은 **생성된 테스트 벡터(models/*.mem/.dec)가 동봉**되어 있다는 점이 기존 분석 시점("models 생성물 없음")과 다르다 — 그러나 이는 데이터 산출물이므로 분석 제외 대상이며 설계 동일성 판정에는 영향 없음.

---

## 2. 전체 분석 위치 (재사용)

본 repo의 함수/모듈 단위 정밀 분석(개요·디렉토리·핵심모듈·데이터플로우·HW/SW매핑·빌드·의존성·강점한계·우리프로젝트 시사점·근거표기)은 이미 다음 문서에 완비되어 있다. **본 repo에도 그대로 적용된다.**

→ `REF/Analysis/ViT-Accelerator/AURA-FlashAttention-AISC-Accelerator.md`

해당 문서가 다루는 핵심(요지만):
- **online-softmax(running max/sum) 기반 streaming FlashAttention**을 전용 RTL로 구현한 **edge 지향 ASIC**(250nm 표준셀, Synopsys DC 합성, `synth/AURAsynth.tcl`).
- **ExpMul**: 곱셈기 없이 `log_e_x = x + (x>>1) − (x>>4)`로 log2(e) 곱을 근사 후 5단 배럴시프트로 `2^(-L)·V` 계산 (`expmul_stage.sv`).
- **65-원소 누적 벡터**(index 0 = softmax 분모 exp-sum, 1..64 = 출력)로 출력·분모를 **동일 회로로 동시 online 누적**(FLASH-D식), 마지막에 1회만 나눗셈으로 정규화 (`expmul.sv`, `vector_division.sv`).
- 재사용 IP: 파라미터화 트리 리덕션(`tree_reduce.sv`), 합성가능 양자화 변환 5종(`q_convert` 계열), valid/ready elastic pipeline, QSRAM/OSRAM ping-pong + KSRAM/VSRAM FIFO 메모리 계층.
- SW/HW co-verify: FP32 골든(`attention_fp32.cpp`) → INT8 골든(`attention_f8.cpp`) → RTL → `precision_measure.cpp`(MAE/RMSE/Top-1).

---

## 3. 우리 프로젝트(HG-PIPE 계열 고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 — 요약

(전체 논의는 기존 문서 §9 참조. 핵심만 재기술)

- **즉시 재사용(High value)**: ① ExpMul 곱셈없는 exp 근사(FPGA에서 DSP 절약), ② 1-pass online softmax + 분모 동봉 누적(HG-PIPE의 "끊김 없는 파이프라인" 철학과 부합), ③ `q_convert` 양자화 변환 5종(임의 Qm.n↔Qp.q, 라운딩/포화/부호확장 — INT8/혼합정밀 데이터패스에 이식), ④ 파라미터화 트리 리덕션(내적/LayerNorm/softmax-sum 공용), ⑤ valid/ready elastic pipeline(멀티사이클 나눗셈을 throughput 손실 없이 통합).
- **구조 참고**: 메모리 더블버퍼링(QSRAM/OSRAM ping-pong, KSRAM/VSRAM 재사용)을 FPGA BRAM/URAM으로 매핑; HW/SW co-verification 흐름.
- **차용 시 주의**: ASIC(250nm) 합성 스크립트·표준셀·SRAM 매크로는 FPGA에 무의미 → **RTL 연산 블록만 추출**, 메모리는 FPGA primitive로 교체. VCS/SystemVerilog 고급문법(union packed, generate) → Vivado/verilator 호환성 점검. `dk=64`·INT8 하드코딩 일반화 필요.
- **XR 시선추적**: AURA의 edge/저전력 지향 + 곱셈없는 softmax는 XR 헤드셋 전력예산에 부합. 단, 시선추적 백본의 attention 비중에 비례해 효과가 결정되므로 **프로파일링 선행 권장**.

---

## 4. 근거 / 한계 표기

- **확실(코드 직접 확인)**: 본 repo의 `README.md`, `include/sys_defs.svh`(라인 22/24/26/28/36/38/34), `verilog/`·`cpp/`·`python/`·`test/`·`synth/` 파일 집합을 Glob/Read로 직접 대조 → 기존 분석 repo와 동일 프로젝트로 판정.
- **추정**: 디렉토리명만 다르고 내용이 동일하므로 두 repo는 같은 프로젝트의 같은(또는 거의 같은) 스냅샷.
- **확인 불가**: 두 repo의 git 커밋 단위 완전 일치 여부(해시 미대조); 본 스냅샷 동봉 `models/*.mem/.dec` 테스트벡터의 정합성/생성출처(데이터 산출물이라 분석 제외).
- **의도적 제외**: `_deprecated/`(현 빌드 미포함), `models/*.mem/.dec/.dump`(생성 데이터), 표준셀 라이브러리 `lec25dscc25_TT.db`, `__pycache__`, `.vscode/`, `docs/AURA_RTL.pdf`(문서) — 이름만 언급.

---

> **요약 한 줄**: 본 repo는 set-2에서 이미 전체 분석된 AURA(FlashAttention ASIC, ExpMul + online-softmax)와 **동일 프로젝트**이므로, 정밀 분석은 `REF/Analysis/ViT-Accelerator/AURA-FlashAttention-AISC-Accelerator.md` 를 참조한다(중복 전체분석 생략).
