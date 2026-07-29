---
source_type: codebase
source_name: flightllm_test_demo
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/flightllm_test_demo/analysis.md -->

# flightllm_test_demo Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo`
- repo_remote: `not_available`
- category: `FlightLLM FPGA demo flow`
- HGTXR relevance: `low`
- matched_paper: `FlightLLM`
- paper_title: FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/FlightLLM.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `346` |
| 주요 언어 | `YAML:146, .csv:122, .npy:32, .bin:27, Python:12, Markdown:3, no_ext:2, .xclbin:1, .txt:1` |
| LOC 추정 | `YAML:3323358, Python:3295, Markdown:195, .txt:5` |

### Directory Map
- `fpga_implementation/` (4 entries)
- `fpga_implementation/bitstream/` (1 entries)
- `fpga_implementation/case/` (1 entries)
- `fpga_implementation/host/` (1 entries)
- `profile/` (7 entries)
- `profile/.compiler_output/` (2 entries)
- `profile/inst_gen/` (4 entries)
- `profile/utils/` (2 entries)

### Metadata / Config Refs
- `README.md`
- `fpga_implementation/README.md`
- `fpga_implementation/case/token_64_single_layer/info.yaml`
- `profile/config.yaml`
- `profile/README.md`
- `profile/.compiler_output/ir_output/llama2_prefill_token_1792.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_192.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_704.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_1072.yaml`
- `profile/.compiler_output/ir_output/llama2_prefill_token_384.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_2016.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_160.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_1264.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_560.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_240.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_832.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_400.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_928.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_1056.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_816.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_1648.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_1504.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_1920.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_528.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_752.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_624.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_1696.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_1488.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_1184.yaml`
- `profile/.compiler_output/ir_output/llama2_decode_token_912.yaml`

### Core Source Refs
- `profile/.compiler_output/ir_output/llama2_decode_token_992.yaml`: 23269 lines
- `profile/.compiler_output/ir_output/llama2_decode_token_976.yaml`: 23269 lines
- `profile/.compiler_output/ir_output/llama2_decode_token_960.yaml`: 23269 lines
- `profile/.compiler_output/ir_output/llama2_decode_token_96.yaml`: 23269 lines
- `profile/.compiler_output/ir_output/llama2_decode_token_944.yaml`: 23269 lines
- `profile/.compiler_output/ir_output/llama2_decode_token_928.yaml`: 23269 lines
- `profile/.compiler_output/ir_output/llama2_decode_token_912.yaml`: 23269 lines
- `profile/.compiler_output/ir_output/llama2_decode_token_896.yaml`: 23269 lines
- `profile/.compiler_output/ir_output/llama2_decode_token_880.yaml`: 23269 lines
- `profile/.compiler_output/ir_output/llama2_decode_token_864.yaml`: 23269 lines

### Hardware-Oriented Source Refs
- no obvious HLS/RTL source found in first scan

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/README.md`
- # FlightLLM Test Demo
- This demo is for testing FlightLLM implementation on the Xilinx Alveo U280 FPGA.
- Our submission can be divided into two parts.

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: configurable sparse DSP chain, always-on-chip decode, length-adaptive compilation을 제공한다.
- 알고리즘 축: sparsity pattern별 DSP chain mapping과 sequence length별 compilation strategy를 사용한다.
- 하드웨어 축: U280/VHK158 FPGA 대상 DSP48와 heterogeneous memory hierarchy를 적극 활용한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/README.md:15:    * Sparse attention masks (`.compiler_output/ir_output/attention_mask/*.npy`)  `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/config.yaml:32:SOFTMAX_PARALLEL_K: 16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/config.yaml:33:LAYERNORM_PARALLEL_K: 16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/isa.py:27:    "softmax",      # operation_flag = 2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/isa.py:28:    "layernorm",    # operation_flag = 3`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/isa.py:29:    "RMSlayernorm", # operation_flag = 4`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/inst_profiler.py:85:        elif misc_op_type == "softmax":`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/inst_profiler.py:86:            total_cycle = K * self.cfg.FSB_BLOCK_SIZE / self.cfg.SOFTMAX_PARALLEL_K * 2 # 需要先过一遍所有数据再计算，因此需要乘2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/inst_profiler.py:87:        elif misc_op_type in ("layernorm", "RMSlayernorm"):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/inst_profiler.py:88:            total_cycle = K * self.cfg.FSB_BLOCK_SIZE / self.cfg.LAYERNORM_PARALLEL_K * 2 # 需要先过一遍所有数据再计算，因此需要乘2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:7:# compile attention QKT or QKTV, support fusing attention with softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:21:    BATCH, N, M, K, N_block_num, M_block_num, K_block_num, mask_mode = tools.get_attention_layer_info(layer_ir, cfg, "MV")`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:22:    if fuse_misc_flag: # 融合attention和softmax层`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:30:        assert MISC_operation_name == "softmax"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:33:    # Q/Softmax: M * K, KT/V: K * N, Out: M * N`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:73:    MV的矩阵(KT/V)放在A Buffer, 序列(Q/Softmax)放在B Buffer`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:75:    Attention暂时不支持切K维度, 因为bank够大`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:77:    Load input sequence (Q/Softmax) to B Buffer * 1 [MAYBE DO NOT NEED]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:78:    for 所有attention head (BATCH)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:90:    LD_seq_in_inst = isa.generate_LD_inst( # LD Q/Softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/README.md:62:INFO 12-11 16:02:22 llm_engine.py:73] Initializing an LLM engine with config: model='meta-llama/Llama-2-7b-hf', tokenizer='meta-llama/Llama-2-7b-hf', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=None, seed=0)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mm.py:66:    each_block_size_B = cfg.FSB_BLOCK_SIZE * cfg.FSB_BLOCK_SIZE * 1 # 1代表INT8，每个数1Byte`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_linear_mv.py:48:    out_sequence_actual_size_B = M * N * 1 # 1 for int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_linear_mv.py:54:    fsb_info = np.ones((K_block_num, need_N_block_num), dtype=np.int8) * cfg.FSB_BLOCK_SIZE # 全dense`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_linear_mv.py:55:    assert fsb_info.dtype == np.int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_linear_mv.py:116:    sequence_batch_1_size_B = M * K * 1 # 1 for int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_linear_mv.py:124:    each_MV_output_size_B = M * cfg.MV_START_N_NUM * 1 # 1 for int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_linear_mv.py:372:                        LD_1d_length            = each_MV_N_block_num * cfg.FSB_BLOCK_SIZE * 1, # 1 for int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_misc.py:66:            LD_ST_hbm_addr_shift = (head_id * M * K + M_id * K) * 1 # 1 for int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_linear_mm.py:41:    fsb_info = np.ones((K_block_num, N_block_num), dtype=np.int8) * cfg.FSB_BLOCK_SIZE # 全dense`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_linear_mm.py:42:    assert fsb_info.dtype == np.int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:44:    out_sequence_actual_size_B = M * N * 1 # 1 for int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:50:    a_K_block_line_size_B = 1 * cfg.FSB_BLOCK_SIZE * K # 1 for int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:58:    sequence_batch_1_size_B = M * K * 1 # 1 for int8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/flightllm_test_demo/profile/inst_gen/layer_support/IR_attention_mv.py:66:    each_MV_output_size_B = M * cfg.MV_START_N_NUM * 1 # 1 for int8`

## 5. HGTXR 적용 해석
- DSP chain scheduling과 compile partitioning 참고용. ViT/eye tracking 직접성은 낮다.

### 적용 가능 모듈
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
