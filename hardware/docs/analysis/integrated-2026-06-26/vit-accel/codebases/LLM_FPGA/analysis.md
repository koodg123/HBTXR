---
source_type: codebase
source_name: LLM_FPGA
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/LLM_FPGA/analysis.md -->

# LLM_FPGA Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA`
- repo_remote: `not_available`
- category: `LLM FPGA kernel snapshot`
- HGTXR relevance: `low`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `353` |
| 주요 언어 | `C/C++ header:102, .rpt:67, .log:58, .pb:25, .mk:24, C++:20, .xutil:15, .rpx:10, .rpv:10, C++ header:5` |
| LOC 추정 | `C/C++ header:319335, JSON:12499, C++:7835, C++ header:590, Python:252` |

### Directory Map
- `W8A8_version/` (5 entries)
- `W8A8_version/GPT_decoding_24L/` (4 entries)
- `W8A8_version/GPT_prefilling_24L/` (18 entries)
- `W8A8_version/LLaMA_decoding_32L/` (19 entries)
- `W8A8_version/LLaMA_prefilling_32L/` (19 entries)
- `W8A8_version/bert_12L/` (19 entries)

### Metadata / Config Refs
- `W8A8_version/GPT_decoding_24L/reports/link/imp/impl_1_system_diagram.json`
- `W8A8_version/bert_12L/reports/link/imp/impl_1_system_diagram.json`
- `W8A8_version/GPT_prefilling_24L/reports/link/imp/impl_1_system_diagram.json`
- `W8A8_version/LLaMA_decoding_32L/reports/link/imp/impl_1_system_diagram.json`
- `W8A8_version/LLaMA_prefilling_32L/reports/link/imp/impl_1_system_diagram.json`

### Core Source Refs
- `W8A8_version/GPT_decoding_24L/const/buf7.h`: 65666 lines
- `W8A8_version/GPT_decoding_24L/const/buf5.h`: 65666 lines
- `W8A8_version/GPT_decoding_24L/const/buf3.h`: 65666 lines
- `W8A8_version/GPT_decoding_24L/const/buf1.h`: 65666 lines
- `W8A8_version/LLaMA_prefilling_32L/const/buf15.h`: 4098 lines
- `W8A8_version/LLaMA_prefilling_32L/const/buf14.h`: 4098 lines
- `W8A8_version/LLaMA_prefilling_32L/const/buf13.h`: 4098 lines
- `W8A8_version/LLaMA_decoding_32L/const/buf15.h`: 4098 lines
- `W8A8_version/LLaMA_decoding_32L/const/buf14.h`: 4098 lines
- `W8A8_version/LLaMA_decoding_32L/const/buf13.h`: 4098 lines

### Hardware-Oriented Source Refs
- `W8A8_version/GPT_decoding_24L/xcl2.hpp`
- `W8A8_version/GPT_decoding_24L/const/buf25.h`
- `W8A8_version/GPT_decoding_24L/const/buf10.h`
- `W8A8_version/GPT_decoding_24L/const/buf22.h`
- `W8A8_version/GPT_decoding_24L/const/buf12.h`
- `W8A8_version/GPT_decoding_24L/const/buf23.h`
- `W8A8_version/GPT_decoding_24L/const/buf2.h`
- `W8A8_version/GPT_decoding_24L/const/buf24.h`
- `W8A8_version/GPT_decoding_24L/const/buf6.h`
- `W8A8_version/GPT_decoding_24L/const/buf5.h`
- `W8A8_version/GPT_decoding_24L/const/buf13.h`
- `W8A8_version/GPT_decoding_24L/const/buf26.h`
- `W8A8_version/GPT_decoding_24L/const/buf1.h`
- `W8A8_version/GPT_decoding_24L/const/buf11.h`
- `W8A8_version/GPT_decoding_24L/const/buf4.h`
- `W8A8_version/GPT_decoding_24L/const/buf18.h`
- `W8A8_version/GPT_decoding_24L/const/buf20.h`
- `W8A8_version/GPT_decoding_24L/const/buf15.h`
- `W8A8_version/GPT_decoding_24L/const/buf3.h`
- `W8A8_version/GPT_decoding_24L/const/buf19.h`
- `W8A8_version/GPT_decoding_24L/const/buf14.h`
- `W8A8_version/GPT_decoding_24L/const/buf9.h`
- `W8A8_version/GPT_decoding_24L/const/buf17.h`
- `W8A8_version/GPT_decoding_24L/const/buf8.h`
- `W8A8_version/GPT_decoding_24L/const/buf16.h`
- `W8A8_version/GPT_decoding_24L/const/buf27.h`
- `W8A8_version/GPT_decoding_24L/const/buf7.h`
- `W8A8_version/GPT_decoding_24L/const/buf21.h`
- `W8A8_version/GPT_decoding_24L/const/buf28.h`
- `W8A8_version/bert_12L/bert_region_1.cpp`

## 3. README 기반 기능 요약
- README가 없어 기능 요약은 파일 구조 기반 추정이다.

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:2:sp=Bert_layer_dataflow_region_1_1.inp_addr_0:HBM[0]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:3:sp=Bert_layer_dataflow_region_1_1.inp_addr_1:HBM[0]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:4:sp=Bert_layer_dataflow_region_1_1.inp_addr_2:HBM[0]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:5:sp=Bert_layer_dataflow_region_1_1.wk_addr:HBM[1]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:6:sp=Bert_layer_dataflow_region_1_1.wv_addr:HBM[2]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:7:sp=Bert_layer_dataflow_region_1_1.wq_addr:HBM[3]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:8:sp=Bert_layer_dataflow_region_2_1.w_ds0_addr:HBM[4]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:9:sp=Bert_layer_dataflow_region_3_1.w_ds1_addr:HBM[5]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:10:sp=Bert_layer_dataflow_region_3_1.w_ds2_addr:HBM[6]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:11:sp=Bert_layer_dataflow_region_3_1.outp_addr:HBM[0]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:13:slr=Bert_layer_dataflow_region_1_1:SLR0`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:14:slr=Bert_layer_dataflow_region_2_1:SLR1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:15:slr=Bert_layer_dataflow_region_3_1:SLR2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:16:stream_connect=Bert_layer_dataflow_region_1_1.outp_k:Bert_layer_dataflow_region_2_1.outp_k:16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:17:stream_connect=Bert_layer_dataflow_region_1_1.outp_v:Bert_layer_dataflow_region_2_1.outp_v:16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:18:stream_connect=Bert_layer_dataflow_region_1_1.outp_q:Bert_layer_dataflow_region_2_1.outp_q:16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:19:stream_connect=Bert_layer_dataflow_region_1_1.outp_inp:Bert_layer_dataflow_region_2_1.outp_inp:16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.cfg:20:stream_connect=Bert_layer_dataflow_region_2_1.outp_ln0:Bert_layer_dataflow_region_3_1.outp_ln0:16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/makefile_us_alveo.mk:93:	v++ -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k Bert_layer_dataflow_region_1 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' $^`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/makefile_us_alveo.mk:97:	v++ -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k Bert_layer_dataflow_region_2 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' $^`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.h:17:#define gelu_len 11008`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.h:28:#define pack_gelu_len_inp gelu_len/inp_parallel`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/kernel.h:33:#define pack_gelu_len_w gelu_len/w_parallel`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_2.cpp:216:void Attention_layer(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_2.cpp:497:void Softmax_layer(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_2.cpp:707:    Attention_layer(inp_sfa, K, buf21, attn_outp);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_2.cpp:708:    Softmax_layer(attn_outp, buf22, sfm_outp);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_3.cpp:116:	data_load_AB:for (int k = 0; k < gelu_len; k++) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_3.cpp:136:			PE_int8_int16_r3(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C[m][n], gelu_len);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_3.cpp:140:	data_drain_AB:for (int k = 0; k < gelu_len; k++) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_3.cpp:186:    for(int jj = 0; jj < pack_gelu_len_w / 2; jj++){`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_3.cpp:227:  io_pack_int8 A[gelu_len];`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_3.cpp:241:    init_inp_buf: for (int j = 0; j < gelu_len; j++) {    // L19`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_3.cpp:251:      for(int k = 0; k < gelu_len; k++){`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LLM_FPGA/W8A8_version/LLaMA_prefilling_32L/bert_region_3.cpp:315:        block_w_ds1_load: for(int jj = 0; jj < pack_gelu_len_w / 2; jj++){`

## 5. HGTXR 적용 해석
- 직접 논문 근거가 약하므로 HGTXR 기본 경로에는 넣지 않는다.
- 단, HLS kernel, PYNQ packaging, AXI/DMA, testbench, golden model 등 구현 보조 자료로 활용 가능하다.

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
