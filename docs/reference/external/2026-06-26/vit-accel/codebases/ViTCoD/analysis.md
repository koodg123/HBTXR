---
source_type: codebase
source_name: ViTCoD
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/ViTCoD/analysis.md -->

# ViTCoD Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD`
- repo_remote: `origin	https://github.com/GATECH-EIC/ViTCoD (fetch)`
- category: `sparse attention ViT co-design`
- HGTXR relevance: `high`
- matched_paper: `ViTCoD`
- paper_title: ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/ViTCoD.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `94` |
| 주요 언어 | `Python:58, .txt:8, no_ext:5, Markdown:5, Shell:4, .png:3, .npy:3, JSON:2, .2:1, .0:1` |
| LOC 추정 | `Python:10792, .txt:2531, JSON:459, Shell:324, Markdown:301, YAML:175` |

### Directory Map
- `Algorithm/` (2 entries)
- `Algorithm/deit/` (35 entries)
- `Algorithm/levit/` (20 entries)
- `Figures/` (2 entries)
- `Hardware/` (1 entries)
- `Hardware/Simulator/` (8 entries)
- `Profile/` (4 entries)
- `Profile/GPU_benchmark/` (3 entries)
- `Profile/TX2_benchmark/` (6 entries)
- `Profile/models/` (3 entries)

### Metadata / Config Refs
- `README.md`
- `Profile/README.md`
- `Profile/TX2_benchmark/benchmark_logs/ViT_Transformer-seq_len_196-dim_192-heads_3.json`
- `Profile/GPU_benchmark/benchmark_logs/ViT_Transformer-seq_len_196-dim_192-heads_3/eic-2019gpu2_54364.1647906182908.pt.trace.json`
- `Algorithm/levit/README.md`
- `Algorithm/deit/README.md`
- `Hardware/Simulator/README.md`

### Core Source Refs
- `Algorithm/deit/timm/vision_transformer.py`: 1353 lines
- `Algorithm/deit/flops.txt`: 984 lines
- `Algorithm/levit/levit.py`: 717 lines
- `Hardware/Simulator/masks/deit_tiny_lowrank/vitcod_atten_0.95_wo.txt`: 630 lines
- `Algorithm/deit/shading_temp.py`: 519 lines
- `Algorithm/deit/main.py`: 487 lines
- `Algorithm/levit/main.py`: 486 lines
- `Algorithm/deit/cait_models.py`: 480 lines
- `Profile/TX2_benchmark/benchmark_logs/ViT_Transformer-seq_len_196-dim_192-heads_3.json`: 459 lines
- `Algorithm/levit/levit_c.py`: 457 lines

### Hardware-Oriented Source Refs
- no obvious HLS/RTL source found in first scan

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/README.md`
- # ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design
- [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)
- **Haoran You**, Zhanyi Sun, Huihong Shi, Zhongzhi Yu, Yang Zhao, Yongan Zhang, Chaojian Li, Baopu Li and Yingyan Lin
- Accepted by [**HPCA 2023**](https://hpca-conf.org/2023/). More Info:
- ---
- ## Why We Consider ViTCoD Given NLP Transformer Accelerators?
- This is because there is a large difference between ViTs and Transformers for natural language processing (NLP)
- <p align="center">
- </p>
- * ***New Opportunity***: The fixed sparse patterns in ViTs can alleviate the stringent need for adopting on-the-fly sparse attention pattern prediction and highly reconfigurable processing element (PE) designs.
- ---
- ## Overview of Our ViT Co-Design Framework
- We propose a dedicated algorithm and accelerator co-design framework dubbed ViTCoD for accelerating ViTs, i.e., Vision Tranformers.
- * ***On the algorithm level***, ViTCoD prunes and polarizes the attention maps to have either denser or sparser
- <p align="center">
- </p>
- ---
- ## Usage of the Provided Codebase
- For reproducing the results, we provides three kinds of codebases:
- * (2) hardware simulator for estimating the cycles given sparsity ratios and patterns, see `./Hardware/Simulator` for detailed implementation and usages.

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: attention map을 dense/sparse fixed pattern으로 polarize하고 autoencoder module로 data movement를 줄인다.
- 알고리즘 축: attention map pruning/polarization으로 두 workload class를 만들고 encoder/decoder가 data movement와 compute를 trade한다.
- 하드웨어 축: dense/sparse workload와 encoder/decoder engine을 동시에 coordinate하는 dedicated accelerator를 사용한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/README.md:34:* ***On the hardware level***, we develop a dedicated accelerator to simultaneously coordinate the aforementioned enforced denser and sparser workloads for boosted hardware utilization, while integrating on-chip encoder and decoder engines to leverage ViTCoD’s algorithm pipeline for much reduced data movements.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/README.md:15:tasks: ViTs have a relatively fixed number of input tokens, whose attention maps can be pruned by up to 90% even with fixed sparse patterns, without severely hurting the model accuracy (e.g., <=1.5% under 90% pruning ratio); while NLP Transformers need to handle input sequences of varying numbers of tokens and rely on on-the-fly predictions of dynamic sparse attention patterns for each input to achieve a decent sparsity (e.g., >=50%).`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/README.md:21:* ***New Opportunity***: The fixed sparse patterns in ViTs can alleviate the stringent need for adopting on-the-fly sparse attention pattern prediction and highly reconfigurable processing element (PE) designs.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/README.md:23:* ***New Challenge***: ViTs' allowed high sparsity in attention maps inevitably aggravates the extent of both irregular data accesses and processing, which could incur severe workload imbalance problems. Moreover, the high sparsity can cause undesired under-utilization when processing highly sparse attention regions, where efficiency is largely bounded by memory/bandwidth due to decreased computational density. That is because the non-zero elements in sparse attention maps of ViTs mostly concentrate along the diagonal lines, i.e., the most inefficient.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/README.md:31:* ***On the algorithm level***, ViTCoD prunes and polarizes the attention maps to have either denser or sparser`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/README.md:32:fixed patterns for regularizing two levels of workloads without hurting the accuracy, largely reducing the attention computations while leaving room for alleviating the remaining dominant data movements; on top of that, we further integrate a lightweight and learnable auto-encoder module to enable trading the dominant high-cost data movements for lower-cost computations.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/README.md:6:## Polarize the sparsed attention maps`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/README.md:9:    <img src="./figs/sparse_attention.jpg" width="520`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/README.md:13:To polarize the attention map stored in `./masks` (e.g., the attention maps of Deit-Tiny trained under 95% sparsity and stored in ./masks/deit_tiny_lowrank/info_0.95.npy) to be either denser or sparser for enhancing more regular workloads, run`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/README.md:17:which correspondingly generates a numpy file (e.g., ./masks/deit_tiny_lowrank/global_token_info_0.95.npy) indicating the number of tokens in the polarized denser attention map, and a numpy file (e.g., ./masks/deit_tiny_lowrank/reodered_info_0.95) representing the sparsity pattern of polarized sparser attention map. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/README.md:20:## Simulate the attention latency `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/README.md:22:To simulate the latency of attention computation, run`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/README.md:32:where we adopt a ***dynamic*** *PE allocation* between the ***denser*** and ***sparser engines*** to balance the workload of processing the denser and sparser patterns of different attention head, and leverage the on-chip ***decoder*** to reconstruct Q and K that are compressed by the on-chip ***encoder*** for saving data access costs. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/README.md:50:* then add the simulated latency with the previously simulated attention latency. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/ViT_FFN.py:32:# root = '/home/sheminghao/shh/ViTCoD/attention_mask'`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/ViT_FFN.py:56:    # TODO: load the masks of attention and global tokens`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/cait_models.py:21:class Class_Attention(nn.Module):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/cait_models.py:48:        attn = attn.softmax(dim=-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/cait_models.py:61:                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/cait_models.py:65:        self.attn = Attention_block(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/cait_models.py:87:class Attention_talking_head(nn.Module):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/environment.yml:55:  - libgfortran-ng=7.5.0=ha8ba4b0_17`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/environment.yml:56:  - libgfortran4=7.5.0=ha8ba4b0_17`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/vision_transformer_flop.py:67:        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/run_visualize_sparsity.py:12:quant_sparsity = [12, 15, 19, 24, 30, 34, 38, 42.5, 47, 52, 57.5]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/run_visualize_sparsity.py:13:quant_BLEU = [34.2, 34.3, 34.4, 34.45, 34.8, 34.3, 34, 33.5, 32.6, 29.5, 31.25]`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/run_visualize_sparsity.py:39:ax.plot(quant_sparsity[:], quant_BLEU, c='orange', marker=marker[3], markersize=4*lw, label=r"$\bf{NLP-Sf. quant}$", lw=lw)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/timm/vision_transformer.py:98:        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/reorder.py:57:    # ax.axis('off')`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Hardware/Simulator/reorder.py:81:        #         ax[i, j].axis('off')`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/levit/gen_mask.py:40:        stage_to_mask_map[stage] = np.apply_along_axis(info_cutoff, 3, arr=attn_map, info_cut_map=info_cut_map, stage=stage)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/vision_transformer_flop.py:182:        # FIXME look at relaxing size constraints`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/gen_mask.py:116:            mean = np.mean(data, axis=1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/gen_mask.py:118:            std = np.std(data, axis=1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTCoD/Algorithm/deit/run_visualize_sparsity.py:7:axis_sparsity = [10,30,50,70,90,100]`

## 5. HGTXR 적용 해석
- 고정 eye token pattern pruning 실험에 유용하지만 정확도 리스크가 높다. P1/P2 ablation으로 제한.

### 적용 가능 모듈
- Attention / Softmax / token sparsity ablation

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
