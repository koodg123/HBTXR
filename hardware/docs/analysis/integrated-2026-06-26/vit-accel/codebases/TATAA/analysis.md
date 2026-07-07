---
source_type: codebase
source_name: TATAA
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/TATAA/analysis.md -->

# TATAA Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA`
- repo_remote: `origin	https://github.com/CASR-HKU/TATAA (fetch)`
- category: `mixed-precision transformer accelerator`
- HGTXR relevance: `medium`
- matched_paper: `TATAA`
- paper_title: TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/TATAA.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `365` |
| 주요 언어 | `Python:75, .txt:55, SystemVerilog:51, Verilog:45, .do:42, Shell:24, .udo:12, Tcl:9, .vho:6, .veo:6` |
| LOC 추정 | `Verilog:22701, Python:17231, SystemVerilog:11903, Shell:4344, .txt:2281, C++:214, Tcl:173, C/C++ header:102, Markdown:89` |

### Directory Map
- `compilation/` (2 entries)
- `compilation/parser/` (5 entries)
- `hardware/` (4 entries)
- `hardware/host/` (6 entries)
- `hardware/rtl/` (30 entries)
- `hardware/vitis_kernel/` (3 entries)
- `quantization/` (2 entries)
- `quantization/hlbfp_quantization/` (2 entries)

### Metadata / Config Refs
- `README.md`
- `quantization/README.md`
- `hardware/README.md`
- `hardware/vitis_kernel/README.md`
- `hardware/vitis_kernel/tata_int8os_proc/prj/add_files.tcl`
- `hardware/vitis_kernel/tata_int8os_proc/prj/add_ips.tcl`
- `hardware/vitis_kernel/tata_int8os_proc/prj/config_kernel.tcl`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.gen/sources_1/ip/axis_register_256/axis_register_256_clocks.xdc`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.gen/sources_1/ip/axis_register_8/axis_register_8_clocks.xdc`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.gen/sources_1/ip/axis_register_32/axis_register_32_clocks.xdc`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/xsim/cmd.tcl`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/xsim/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/modelsim/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/xcelium/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/ies/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/riviera/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/activehdl/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/vcs/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_256/questa/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_8/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_8/xsim/cmd.tcl`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_8/xsim/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_8/modelsim/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_8/xcelium/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_8/ies/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_8/riviera/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_8/activehdl/README.txt`
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts/axis_register_8/vcs/README.txt`

### Core Source Refs
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/ipstatic/hdl/axis_register_slice_v1_1_vl_rfs.v`: 3005 lines
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.gen/sources_1/ip/axis_register_8/hdl/axis_register_slice_v1_1_vl_rfs.v`: 3005 lines
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.gen/sources_1/ip/axis_register_32/hdl/axis_register_slice_v1_1_vl_rfs.v`: 3005 lines
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.gen/sources_1/ip/axis_register_256/hdl/axis_register_slice_v1_1_vl_rfs.v`: 3005 lines
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/ipstatic/hdl/axis_infrastructure_v1_1_vl_rfs.v`: 1324 lines
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.gen/sources_1/ip/axis_register_8/hdl/axis_infrastructure_v1_1_vl_rfs.v`: 1324 lines
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.gen/sources_1/ip/axis_register_32/hdl/axis_infrastructure_v1_1_vl_rfs.v`: 1324 lines
- `hardware/vitis_kernel/tata_int8os_proc/prj/tata_int8os_proc.gen/sources_1/ip/axis_register_256/hdl/axis_infrastructure_v1_1_vl_rfs.v`: 1324 lines
- `quantization/hlbfp_quantization/hlibf_bfloat16_int8/hlibf_bert/hibf_bert.py`: 1181 lines
- `compilation/parser/bert/hibf_bert.py`: 1029 lines

### Hardware-Oriented Source Refs
- `hardware/rtl/pe_stg_3.sv`
- `hardware/rtl/tata_top_wrapper.v`
- `hardware/rtl/data_loader.sv`
- `hardware/rtl/dmrf_x.sv`
- `hardware/rtl/tapu.sv`
- `hardware/rtl/transpose_int.sv`
- `hardware/rtl/instr_loader.sv`
- `hardware/rtl/lzc_48b.sv`
- `hardware/rtl/pe_stg_0.sv`
- `hardware/rtl/exec_ctrl_int8.sv`
- `hardware/rtl/twos_sm_convert.sv`
- `hardware/rtl/pe_stg_1.sv`
- `hardware/rtl/dmrf_y.sv`
- `hardware/rtl/fifo_axis.sv`
- `hardware/rtl/sim_tata_top.sv`
- `hardware/rtl/mem_kernel.v`
- `hardware/rtl/pe_stg_2.sv`
- `hardware/rtl/bs_rsf.sv`
- `hardware/rtl/delay_chain.sv`
- `hardware/rtl/dm_quant.sv`
- `hardware/rtl/proc_kernel.v`
- `hardware/rtl/pe_sys.sv`
- `hardware/rtl/fifo_common.sv`
- `hardware/rtl/core_instr_ctrl.sv`
- `hardware/rtl/zout_ctrl.sv`
- `hardware/rtl/proc_core.sv`
- `hardware/rtl/ps_ctrl.v`
- `hardware/rtl/bram_sdp_wrapper.sv`
- `hardware/rtl/pccmd_ctrl.sv`
- `hardware/rtl/sm_twos_convert.sv`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/README.md`
- # TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture
- The open-source implementation of the paper "TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture" in ACM Transactions on Reconfigurable Technology and Systems.
- ## Compilation
- The compiler of TATAA to parse Transformer models, generate dataflow and instructions for TATAA processor.
- Please refer to the `./compilation` directory for more details.
- ## Hardware
- The hardware implementation of TATAA processor.
- Please refer to the `./hardware` directory for more details.
- ## Quantization
- The quantization tool to quantize Transformer models in TATAA (int8 + bfloat16).
- Please refer to the `./quantization` directory for more details.
- Also, we listed the required python environment in `./quantization`.
- ## Citation (ACM Format)
- ```
- @article{TATAA,
- author = {Wu, Jiajun and Song, Mo and Zhao, Jingmin and Gao, Yizhao and Li, Jia and So, Hayden Kwok-Hay},
- title = {TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture},
- year = {2025},
- issue_date = {March 2025},
- publisher = {Association for Computing Machinery},

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: int8 systolic mode와 bfloat16 SIMD mode를 runtime 전환하는 transformable arithmetic architecture를 제안한다.
- 알고리즘 축: linear op는 int8 PTQ, nonlinear op는 bf16 approximation으로 mapping하고 compiler가 instruction/dataflow를 생성한다.
- 하드웨어 축: systolic array mode와 SIMD vector mode를 같은 arithmetic fabric에서 전환한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/README.md:7:The compiler of TATAA to parse Transformer models, generate dataflow and instructions for TATAA processor.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:240:        // Register Control Attributes: Pipeline Register Configuration`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:241:        .ACASCREG(1),  // Number of pipeline stages between A/ACIN and ACOUT (0-2)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:242:        .ADREG(0),  // Pipeline stages for pre-adder (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:243:        .ALUMODEREG(0),  // Pipeline stages for ALUMODE (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:244:        .AREG(1),  // Pipeline stages for A (0-2)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:245:        .BCASCREG(1),  // Number of pipeline stages between B/BCIN and BCOUT (0-2)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:246:        .BREG(1),  // Pipeline stages for B (0-2)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:247:        .CARRYINREG(1),  // Pipeline stages for CARRYIN (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:248:        .CARRYINSELREG(1),  // Pipeline stages for CARRYINSEL (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:249:        .CREG(0),  // Pipeline stages for C (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:250:        .DREG(1),  // Pipeline stages for D (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:251:        .INMODEREG(1),  // Pipeline stages for INMODE (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:252:        .MREG(1),  // Multiplier pipeline stages (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:253:        .OPMODEREG(0),  // Pipeline stages for OPMODE (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_0.sv:254:        .PREG(0)  // Number of pipeline stages for P (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_3.sv:160:        // Register Control Attributes: Pipeline Register Configuration`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_3.sv:161:        .ACASCREG(1),  // Number of pipeline stages between A/ACIN and ACOUT (0-2)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_3.sv:162:        .ADREG(0),  // Pipeline stages for pre-adder (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/hardware/rtl/pe_stg_3.sv:163:        .ALUMODEREG(0),  // Pipeline stages for ALUMODE (0-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:36:    den = isqrt_gelu(denominator, mp)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:118:def isqrt_gelu(x, mp=None):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:282:            mp.parse_ops('matmul', in_data=self.weight, in_data_name='', `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:383:            mp.parse_ops('matmul', in_data=k_trans, in_data_name='', in_tmp=q, in_tmp_name='', out_data=x, out_data_name='', feature_dict={'scale': 0.1, 'quant_type': 'fp16'}, int_ops=True, keep_out=True)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:441:            mp.parse_ops('matmul', in_data=v, in_data_name='', in_tmp=s, in_tmp_name='', out_data=x`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:447:class HMQGeLU(nn.GELU):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:449:        super(HMQGeLU, self).__init__()`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:452:        mp.parse_layer(layer_name, 'gelu')`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:488:class HMQSoftmax(nn.Module):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:490:        super(HMQSoftmax, self).__init__()`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:524:class HMQLayerNorm(nn.LayerNorm):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/quant_module.py:526:        super(HMQLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/hibf_bert.py:25:    BaseModelOutputWithPastAndCrossAttentions,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/hibf_bert.py:26:    BaseModelOutputWithPoolingAndCrossAttentions,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TATAA/compilation/parser/bert/hibf_bert.py:30:    _prepare_4d_attention_mask_for_sdpa,`

## 5. HGTXR 적용 해석
- Q4/Q8에서 정확도 병목이 LayerNorm/Softmax/GELU라면 해당 경로만 higher precision으로 올리는 P1 실험에 적합하다.

### 적용 가능 모듈
- Quantization / scale calibration / weight generation
- HLS/Vivado/PYNQ integration and resource-flow references

## 6. 실험 옵션으로 변환할 때의 규칙
- 먼저 HGTXR-SW 또는 C++ reference에서 accuracy/bit-exact behavior를 검증한다.
- HLS에 반영할 때는 C3b signoff path를 덮어쓰지 말고 새로운 suffix variant로 추가한다.
- 완료된 `csynth.xml`, routed timing/power, PYNQ smoke JSON 없이는 resource matrix의 완료 variant로 승격하지 않는다.
- HGTXR 논문 범위를 벗어나는 구조 변경은 `paper_scope_review_required`로 둔다.

## 7. 리스크와 비적용 조건
- 동적 MoE routing, softmax-free attention, ternary/LUT-heavy compute는 정확도 또는 resource 정책과 충돌할 수 있다.
- 현재 사용자의 resource 방향은 DSP/URAM 활용 증가와 LUT 과사용 억제이므로 LUT-LLM/LUT-GEMM류는 기본값이 아니라 negative-control이다.
- codebase가 LLM 중심이면 HGTXR eye-tracking 적용은 kernel/system-flow 수준으로 제한한다.

## 8. 다음 분석/구현 액션
- 핵심 source file을 1개 선택해 line-by-line 분석을 수행한다.
- 해당 방법을 HGTXR `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml`의 planned experiment와 연결한다.
- SW accuracy metric과 HW resource metric을 같은 manifest에 기록한다.
