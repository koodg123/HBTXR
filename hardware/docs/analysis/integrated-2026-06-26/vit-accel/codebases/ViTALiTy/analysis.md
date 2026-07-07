---
source_type: codebase
source_name: ViTALiTy
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/ViTALiTy/analysis.md -->

# ViTALiTy Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy`
- repo_remote: `origin	https://github.com/GATECH-EIC/ViTALiTy (fetch)`
- category: `linear Taylor attention ViT accelerator`
- HGTXR relevance: `high`
- matched_paper: `ViTALiTy`
- paper_title: ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/ViTALiTy.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `30` |
| 주요 언어 | `Python:17, .png:7, Markdown:3, no_ext:2, .txt:1` |
| LOC 추정 | `Python:3741, Markdown:108, .txt:25` |

### Directory Map
- `.github/` (6 entries)
- `figures/` (3 entries)
- `src/` (17 entries)

### Metadata / Config Refs
- `README.md`

### Core Source Refs
- `src/quantize.py`: 589 lines
- `src/quant_utils.py`: 482 lines
- `src/main.py`: 460 lines
- `src/patchconvnet_models.py`: 350 lines
- `src/vision_transformer.py`: 341 lines
- `src/utils.py`: 238 lines
- `src/mlp.py`: 231 lines
- `src/resmlp_models.py`: 197 lines
- `src/models.py`: 178 lines
- `src/drop.py`: 148 lines

### Hardware-Oriented Source Refs
- no obvious HLS/RTL source found in first scan

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md`
- # ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention
- [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)
- Jyotikrishna Dass*, Shang Wu*, Huihong Shi*, Chaojian Li, Zhifan Ye, Zhongfeng Wang and Yingyan Lin
- (*Equal contribution)
- Accepted by [HPCA 2023](https://hpca-conf.org/2023/). More Info:
- \[ [**Paper**](https://arxiv.org/abs/2211.05109) | [**Slide**]() | [**GitHub**](https://github.com/GATECH-EIC/ViTaLiTy) \]
- ---
- ## Overview of the ViTALiTy Framework
- We propose a low-rank and sparse approximation algorithm and accelerator co-design framework dubbed ViTALiTy.
- <p align="center">
- </p>
- <p align = "center">
- Fig.1 - ViTALiTy workflow comprising the proposed (Low-Rank) Linear Taylor attention (order, m = 1): (i) Higher-order Taylor terms (m > 1)
- </p>
- <p align="center">
- </p>
- <p align = "center">
- Fig.2 - Computational steps (a) vanilla Softmax Attention and (b) our Taylor attention (see Algorithm 1), where the global context matrix G provides linear computation and memory benefits over the vanilla quadratic QK^T.
- </p>
- <p align="center">

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: first-order Taylor attention과 row-mean centering으로 low-rank component를 만들고 sparsity regularization을 결합한다.
- 알고리즘 축: softmax dot-product attention을 Taylor 근사로 linearize하고 low-rank/sparse 성분을 통합한다.
- 하드웨어 축: linear Taylor attention workload에 맞춘 pipeline accelerator가 attention matrix materialization을 줄인다.

### 핵심 모듈 근거
- `src/vision_transformer.py:92-137`의 `Attention`은 `vitality` 플래그에 따라 일반 softmax attention과 ViTALiTy attention을 분기한다. HGTXR에 적용할 때도 exact attention fallback을 유지한 ablation flag가 필수다.
- 같은 블록에서 ViTALiTy path는 q/k를 quantize하고 sparse mask를 만든 뒤, centered key와 `kv = k^T @ v`, `attn = sparse @ v + q @ kv` 형태로 계산한다. 이는 attention matrix 전체를 저장하지 않는 장점이 있지만 HGTXR 정확도 목표와 충돌할 수 있으므로 SW-first 실험으로 제한한다.
- `src/vision_transformer.py:140-160`의 `Block`은 attention 결과를 residual과 MLP에 다시 연결한다. 따라서 Taylor attention의 정확도 손실은 attention 단독 지표가 아니라 최종 eye-tracking accuracy, residual scale, downstream MLP activation 분포로 검증해야 한다.
- `src/quant_utils.py`의 quantization helper와 `QuantizedMatMul` 계층은 bit-width ablation 근거를 제공한다. 다만 현재 사용자의 자원 방향은 DSP/URAM 활용률 증가이므로 sparse/linear attention의 제어 로직이 LUT를 과도하게 늘리면 채택하지 않는다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:37:* ***On the hardware level***, we develop a dedicated accelerator to better leverage the algorithmic properties of ViTALiTy’s linear attention, where only a low-rank component is executed during inference favoring hardware efficiency. Specifically, ViTALiTy's accelerator features a chunk-based design integrating both a systolic array tailored for matrix multiplications and pre/post-processors customized for ViTALiTy attentions’ pre/post-processing steps. Furthermore, we adopt an intra-layer pipeline design to leverage the intra-layer data dependency for enhancing the overall throughput together with a down-forward accumulation dataflow for the systolic array to improve hardware efficiency.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:2:# ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:19:* ***On the algorithm level***, we propose a linear attention for reducing the computational and memory cost by decoupling the vanilla softmax attention into its corresponding “weak” and “strong” Taylor attention maps. Unlike the vanilla attentions, the linear attention in ViTALiTy generates a global context matrix G by multiplying the keys with the values. Then, we unify the low-rank property of the linear attention with a sparse approximation of “strong” attention for training the ViT model. Here, the low-rank component of our ViTALiTy attention captures global information with a linear complexity, while the sparse component boosts the accuracy of linear attention model by enhancing its local feature extraction capacity.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:25:Fig.1 - ViTALiTy workflow comprising the proposed (Low-Rank) Linear Taylor attention (order, m = 1): (i) Higher-order Taylor terms (m > 1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:26:when added results in vanilla softmax attention score, (ii) Training phase (unifying low-rank and sparse approximation) where higher-order Taylor terms are approximated as Sparse attention (computed using SANGER [28]), and (iii) Inference phase that uses only the (Low-Rank) Linear Taylor attention.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:31:<img src="./figures/TaylorAttentionFlow2.png" width="400">`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:34:Fig.2 - Computational steps (a) vanilla Softmax Attention and (b) our Taylor attention (see Algorithm 1), where the global context matrix G provides linear computation and memory benefits over the vanilla quadratic QK^T.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:37:* ***On the hardware level***, we develop a dedicated accelerator to better leverage the algorithmic properties of ViTALiTy’s linear attention, where only a low-rank component is executed during inference favoring hardware efficiency. Specifically, ViTALiTy's accelerator features a chunk-based design integrating both a systolic array tailored for matrix multiplications and pre/post-processors customized for ViTALiTy attentions’ pre/post-processing steps. Furthermore, we adopt an intra-layer pipeline design to leverage the intra-layer data dependency for enhancing the overall throughput together with a down-forward accumulation dataflow for the systolic array to improve hardware efficiency.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:50:### Training (DeiT-Tiny with vanilla softmax)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/README.md:56:### Inference (DeiT-Tiny with vanilla softmax)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/losses.py:55:                F.log_softmax(outputs_kd / T, dim=1),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/losses.py:59:                F.log_softmax(teacher_outputs / T, dim=1),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/models.py:65:        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/models.py:80:        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/models.py:95:        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/models.py:110:        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/models.py:125:        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/models.py:140:        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/models.py:155:        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/models.py:170:        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:437:class QuantizedMatMul(nn.Module):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:2:https://github.com/IntelLabs/nlp-architect/blob/master/nlp_architect/nn/torch/quantization.py`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:16:    """Calculate dynamic scale for quantization from input by taking the`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:24:    """Calculate scale for quantization according to some constant and number of bits"""`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:25:    return calc_max_quant_value(bits) / threshold`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:28:def calc_max_quant_value(bits):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:29:    """Calculate the maximum symmetric quantized value according to number of bits"""`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:33:def quantize(input, scale, bits):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:34:    """Do linear quantization to input according to a scale and number of bits"""`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:35:    thresh = calc_max_quant_value(bits)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:39:def dequantize(input, scale):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:40:    """linear dequantization according to some scale"""`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:45:class FakeLinearQuantizationWithSTE(torch.autograd.Function):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:46:    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViTALiTy/src/quant_utils.py:50:        """fake quantize input according to scale and number of bits, dequantize`

## 5. HGTXR 적용 해석
- eye-region token 수가 고정/작은 편이면 approximation ablation으로 가치가 있다. exact attention fallback 필수.

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
