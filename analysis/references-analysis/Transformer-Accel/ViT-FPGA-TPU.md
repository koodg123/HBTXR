# ViT-FPGA-TPU 교차참조 (Transformer-Accel 사본)

> 대상 repo: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\Transformer-Accel\ViT-FPGA-TPU`
> 기존 분석 참조: **`REF/Analysis/ViT-Accelerator/ViT-FPGA-TPU.md`** (존재함, 정밀 분석 v2.0)
> 작성일: 2026-06-20 · 범위: README + 컨트롤러/어레이 대표 RTL 1~2개만 확인

---

## 결론: 동일 프로젝트, 다른(더 완전한) 스냅샷

같은 Harvard CS205 ViT-FPGA-TPU 프로젝트, 동일 저자(Eric/Hongyi/Wenyun/Sebastian), 동일 아키텍처(16×16 FP16 systolic OS GEMM + XDMA/MMIO ABI). 코드 내용은 **동일/유사**하나, 트리 구성이 **상이**하다.

| 항목 | 기존 분석(ViT-Accelerator 사본) | 본 사본(Transformer-Accel) | 판정 |
|---|---|---|---|
| 핵심 소스 위치 | `v2.0/hw`, `v2.0/sw` (풀 소스) | `v2.0/`는 README 스텁만, 실소스는 `code/fpga`, `code/code_cpu`, `code/code_fpga` | **상이(구조)** |
| systolic_array.sv | 16×16 generate, OS 전파 | 동일 구조 + `MAC_ID`/`USE_1_DSP_ID` 파라미터 추가 | **유사** |
| 컨트롤러 RTL | **확인 불가**(레지스터 ABI만 존재) | **존재함**: `accelerator_ctl.sv`, `accelerator_types.sv`, `spad_arbiter.sv` | **상이(여기서 추가 확인)** |
| 명령/상태 enum | `accelerator.h`(SW) | `accelerator_types.sv`(HW) — 동일 enum | **동일** |
| ViT 본체 | 루트 `code/code_cpu` libTorch(언급만) | `code/code_cpu`에 vit/msa/vit_block 풀 소스 | **유사** |

---

## 근거 (파일/라인)

1. **동일 프로젝트**: `README.md:1-2`("FPGA based Vision Transformer accelerator (Harvard CS205)"), 기여자 표 `README.md:37-44` — 기존 분석 `README.md:1-2`와 일치.

2. **systolic_array는 동일 설계**: `code/fpga/pci_mig_accelerator_1.0_16_auto/src/systolic_array.sv:7-8`(`NUM_ROWS/COLS=16`), `:39-85` generate 결선(weight 아래로 `:75`, input 오른쪽으로 `:80`, OS 출력 `:61`) — 기존 분석 §3.1과 라인 단위로 동일. **차이**: `:9` `USE_1_DSP_ID=250`, `:46` `MAC_ID=i*NUM_ROWS+j` 파라미터가 새로 추가됨(PE별 DSP 사용 모드 제어 의도, 기존 사본에는 없던 것).

3. **컨트롤러 RTL이 여기서 확인됨**(기존 분석 §8 한계·§10 "확인 불가"였던 `pci_mig_accelerator`): `code/fpga/.../src/accelerator_ctl.sv` 존재. FSM `S_IDLE→S_WAIT_ARW_READY→S_WAIT_RW→S_WAIT_COMPUTE→S_SEND_MAT_C`(`accelerator_ctl.sv:259-425`), AXI write 서브FSM `S_W_IDLE..S_W_DONE`(`:293-363`), MIG read 요청 `mig_send_rq`(`:182-187`), BRAM A/B 적재(`:81-89`). → DMA↔BRAM↔어레이 조율 로직이 본 사본에서 실제 검증 가능.

4. **MMIO ABI 동일**: `accelerator_types.sv:9-25`의 `accel_instr_e`(I_IDLE/I_R_MAT_A/B/C/I_GEMM/I_RESET)·`accel_state_e`(S_IDLE..S_DONE)가 기존 분석 `accelerator.h:35-51`(SW 측)과 **완전 일치**. `accelerator_ctl.sv:100-114` 13개 MMIO 레지스터도 `accelerator.h` 레지스터 맵과 일치.

5. **추가 차이 — 파라미터화/대형 어레이 변형**: `accelerator_ctl.sv:20` `SYS_ARR_SIZE=32`(32×32 변형도 존재), `:23` `SYS_MAC_LATENCY=18`, `:12` `DATA_PRECISION=16`(FP16 유지). 기존 분석은 16×16·MAC latency≈8(곱4+누산4)로 기술 — 본 컨트롤러는 어레이 크기/지연을 파라미터화하고 `1.0_16_auto`(16)·`1.0_32`(32) 두 IP 변형을 보유. FP16 정밀도는 동일.

---

## 요약

- **동일여부**: 동일 프로젝트의 **다른 스냅샷**(코드 유사, 트리 구조 상이).
- **핵심 차이**: 기존 사본에서 "확인 불가"였던 **컨트롤러 RTL(`accelerator_ctl.sv`/`spad_arbiter.sv`)이 본 사본에 존재**하며, 어레이 크기(16/32)와 MAC latency가 파라미터화됨.
- 재사용 시사점은 기존 분석 §9를 그대로 따르되, 컨트롤러 FSM(DMA→compute→writeback)은 이제 본 사본의 `accelerator_ctl.sv`로 직접 참조 가능.
