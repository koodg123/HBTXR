# TATAA — Cross-Reference

> 본 문서는 교차참조(cross-ref)이며, 정밀 분석 본문은 **`REF/Analysis/ViT-Accelerator/TATAA.md`** 를 참조한다.
> 대상 repo: `REF/Transformer-Accel/TATAA`
> 기준 repo: `REF/ViT-Accelerator/TATAA` (기존 정밀 분석 대상)

---

## 동일 여부: **동일 (same)** — 같은 코드베이스의 동일 사본

대상 repo는 기존 정밀 분석 대상과 **동일한 TATAA 코드베이스**다. 디렉토리 구조·파일 구성·핵심 RTL 내용이 일치한다. 별도의 새 정밀 분석은 불필요하며, `REF/Analysis/ViT-Accelerator/TATAA.md`의 모든 분석 내용(TAPU 4×16 PE, `mode_sel` INT8↔bfloat16 변형, `dm_quant` 4모드, 64b ISA 2계층 디코더, BFP+bfloat16 양자화, DeiT 컴파일러 등)이 그대로 적용된다.

## 근거 (파일/라인 직접 대조)

- **디렉토리 구조 일치**: 대상 repo `hardware/rtl/`에 `tapu.sv, pe_sys.sv, pe_stg_0~3.sv, proc_core.sv, dm_quant.sv, core_instr_ctrl.sv, pccmd_ctrl.sv, dmrf_x/y.sv` 등 기존 분석 §2의 파일 목록이 동일하게 존재. `quantization/`, `compilation/` 두 축도 동일.
- **`hardware/README.md`**: Alveo U280(L3), Verilog RTL(L15), mem/proc 커널 분리 + Vitis 2023.2 강제(L17–L18), "host 실행 전 모델 컴파일로 instr 바이너리 필요"(L11) — 기존 분석 §1·§6과 동일.
- **`hardware/vitis_kernel/README.md`**: 2개 RTL 커널(`tata_int8os_mem`/`tata_int8os_proc`) 분리, `add_files.tcl`/`add_ips.tcl`→패키지→`config_kernel.tcl`→`.xo`(L1–L18) — 기존 분석 §6 절차와 동일.
- **`hardware/rtl/pe_stg_0.sv`**: `mode_sel` 4모드 주석(L17 `00=matmul, 10=fp mul, 11=fp add, 01=fp mag`), INT8 DSP 패킹(`dsp_a_in={...,y_reg[15:8],19'd0}` L92, `dsp_b_in={...,left_in}` L102, `dsp_d_in={...,y_reg[7:0]}` L121), **isqrt 매직넘버 `27'h0005f37` (L123)** — 기존 분석 §3.3과 라인까지 일치.

## 핵심 차이 (유일한 차이점)

- **저장 경로만 상이**: 동일 코드가 `REF/ViT-Accelerator/TATAA`(기존 분석 기준) 와 `REF/Transformer-Accel/TATAA`(본 대상) 두 위치에 존재. 기존 분석 .md 본문 L3는 대상 경로를 `REF/ViT-Accelerator/TATAA`로 표기하고 있어, 본 대상 repo와는 **상위 디렉토리 명칭(ViT-Accelerator vs Transformer-Accel)만 다른 사본**으로 판단됨(재배치 또는 중복 체크아웃 추정).
- 코드 레벨 차이는 발견되지 않음(README 2종 + 대표 RTL `pe_stg_0.sv` 대조 기준). 전수 diff는 미수행이나, 대표 파일 라인 일치로 동일 사본으로 판정.

---

**기존 정밀 분석 .md 존재**: O — `REF/Analysis/ViT-Accelerator/TATAA.md` (10개 절, 라인 근거 포함 완본). 본 cross-ref는 그 문서로 위임한다.
