---
source_type: codebase
source_name: P2-ViT
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/P2-ViT/analysis.md -->

# P2-ViT Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT`
- repo_remote: `origin	https://github.com/shihuihong214/P2-ViT (fetch)`
- category: `PoT PTQ and ViT accelerator`
- HGTXR relevance: `very_high`
- matched_paper: `P2-ViT`
- paper_title: P2-ViT: Power-of-Two Post-Training Quantization and Acceleration for Fully Quantized Vision Transformer
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/P2-ViT.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `43` |
| 주요 언어 | `Python:35, YAML:2, no_ext:2, .png:2, Shell:1, Markdown:1` |
| LOC 추정 | `Python:5906, Markdown:57, YAML:54, Shell:11` |

### Directory Map
- `.github/` (1 entries)
- `.github/workflows/` (1 entries)
- `figures/` (2 entries)
- `models/` (8 entries)
- `models/ptq/` (5 entries)
- `pyhessian/` (3 entries)
- `utils/` (4 entries)

### Metadata / Config Refs
- `.pre-commit-config.yaml`
- `README.md`

### Core Source Refs
- `models/swin_quant.py`: 901 lines
- `models/vit_fquant.py`: 894 lines
- `models/vit_quant.py`: 559 lines
- `test_quant.py`: 515 lines
- `models/layers_quant.py`: 493 lines
- `models/ptq/layers.py`: 415 lines
- `pyhessian/hessian.py`: 286 lines
- `models/ptq/observer/minmax.py`: 198 lines
- `models/utils.py`: 197 lines
- `generate_data.py`: 182 lines

### Hardware-Oriented Source Refs
- no obvious HLS/RTL source found in first scan

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md`
- # P2-ViT
- This repo contains the official implementation of **["P2-ViT: Power-of-Two Post-Training Quantization and Acceleration for Fully Quantized Vision Transformer"](https://arxiv.org/abs/2405.19915).**
- ## Abstract
- <div align=center>
- </div>
- To this end, we propose P$^2$-ViT, the first Power-of-Two (PoT) post-training quantization and acceleration framework to accelerate fully quantized ViTs.
- <div align=center>
- </div>
- ## Quantization
- ### Run
- Example: Evaluate quantized DeiT-S with MinMax quantizer.
- ```bash
- python test_quant.py deit_small <YOUR_DATA_DIR> --quant --quant-method minmax
- ```
- - `deit_small`: model architecture, which can be replaced by `deit_tiny`, `deit_base`, `vit_base`, `vit_large`, `swin_tiny`, `swin_small` and `swin_base`.
- - `--quant`: whether to quantize the model.
- - `--quant-method`: quantization methods of activations, which can be chosen from `minmax`, `ema`, `percentile` and `omse`.
- ## Citation
- If you find this repo useful in your research, please consider citing the following paper:
- ```BibTex

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: PoT scaling factor 기반 PTQ와 coarse-to-fine automatic mixed precision을 결합한다.
- 알고리즘 축: activation/weight scale을 power-of-two로 제한해 rescale multiply를 shift로 바꾸고, layer별 bit/scale 후보를 coarse-to-fine 탐색한다.
- 하드웨어 축: chunk-based accelerator와 row-stationary dataflow가 PoT requant pipeline을 이용해 throughput을 높인다.

### 핵심 모듈 근거
- `models/vit_quant.py:25-61`의 `Attention`은 `QLinear -> qkv reshape/permute -> q @ k^T -> QIntSoftmax -> attn @ v -> QLinear proj` 순서다. HGTXR 관점에서는 Q/K/V projection, attention score, softmax, output projection을 각각 별도 정수화/스케일 추적 지점으로 볼 수 있다.
- `models/vit_quant.py:120-196`의 `Block`은 `norm1 -> attn -> residual qact -> norm2 -> MLP -> residual qact` 구조다. 따라서 P2-ViT의 scale policy를 HGTXR에 붙일 때 attention만 보지 말고 residual boundary와 MLP input/output scale까지 같이 검증해야 한다.
- `models/layers_quant.py:13-36`은 `alpha_pool=[0.5]`, `bit_pool=[4,8]`, `smoothquant_process()`의 power-of-two `channel_scale`을 둔다. 현재 HGTXR Q4/Q8 방향과 직접 맞고, scale multiply를 shift-friendly하게 바꾸는 실험으로 변환 가능하다.
- `models/layers_quant.py:141-260`의 quantized `Mlp`는 `fc1/fc2`, activation quant, SmoothQuant/channel-scale 후보가 연결된다. DSP 매핑을 유지하려면 matmul 자체는 LUT화하지 않고 scale/requant 경로만 shift/constant화하는 것이 맞다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:17:In terms of hardware, we develop a dedicated chunk-based accelerator featuring multiple tailored sub-processors to individually handle ViTs' different types of operations, alleviating reconfigurable overhead. Additionally, we design a tailored row-stationary dataflow to seize the pipeline processing opportunity introduced by our PoT scaling factors, thereby enhancing throughput.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/layers_quant.py:147:                 act_layer=nn.GELU,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/quantizer/log2.py:15:        self.softmax_mask = None`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/quantizer/log2.py:19:        self.softmax_mask = rounds >= 2**self.bit_type.bits`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/quantizer/log2.py:25:        outputs[self.softmax_mask] = 0`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/observer/minmax.py:72:            attn = attn.softmax(dim=-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/__init__.py:3:from .layers import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:225:class QIntLayerNorm(nn.LayerNorm):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:228:        super(QIntLayerNorm, self).__init__(normalized_shape, eps,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:294:class QIntSoftmax(nn.Module):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:297:                 log_i_softmax=False,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:305:        super(QIntSoftmax, self).__init__()`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:307:        self.log_i_softmax = log_i_softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:331:    def int_softmax(x, scaling_factor):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:367:        if self.log_i_softmax and scale is not None:`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:368:            exp_int, exp_int_sum = self.int_softmax(x, scale)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:369:            softmax_out = torch.round(exp_int_sum / exp_int)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:370:            rounds = self.log_round(softmax_out)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:373:            deq_softmax = 2**(-qlog)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:374:            deq_softmax[mask] = 0`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/ptq/layers.py:375:            return deq_softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:3:This repo contains the official implementation of **["P2-ViT: Power-of-Two Post-Training Quantization and Acceleration for Fully Quantized Vision Transformer"](https://arxiv.org/abs/2405.19915).**`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:9:To tackle this limitation, prior works have explored ViT-tailored quantization algorithms but retained floating-point scaling factors, which yield non-negligible re-quantization overhead, limiting ViTs' hardware efficiency and motivating more hardware-friendly solutions. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:11:<img src="./figures/PoT.png" width="850px" />`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:15:To this end, we propose P$^2$-ViT, the first Power-of-Two (PoT) post-training quantization and acceleration framework to accelerate fully quantized ViTs.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:16:Specifically, as for quantization, we explore a **dedicated quantization scheme** to effectively quantize ViTs with **PoT scaling factors**, thus minimizing the re-quantization overhead. Furthermore, we propose **coarse-to-fine automatic mixed-precision quantization** to enable better accuracy-efficiency trade-offs.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:17:In terms of hardware, we develop a dedicated chunk-based accelerator featuring multiple tailored sub-processors to individually handle ViTs' different types of operations, alleviating reconfigurable overhead. Additionally, we design a tailored row-stationary dataflow to seize the pipeline processing opportunity introduced by our PoT scaling factors, thereby enhancing throughput.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:22:Extensive experiments consistently validate P$^2$-ViT's effectiveness. Particularly, we offer comparable or even superior quantization performance with PoT scaling factors when compared to the counterpart with floating-point scaling factors. Besides, we achieve up to $\mathbf{10.1\times}$ speedup and $\mathbf{36.8\times}$ energy saving over GPU's Turing Tensor Cores, and up to $\mathbf{1.84\times}$ higher computation utilization efficiency against SOTA quantization-based ViT accelerators.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:25:## Quantization`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:29:Example: Evaluate quantized DeiT-S with MinMax quantizer.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:32:python test_quant.py deit_small <YOUR_DATA_DIR> --quant --quant-method minmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:37:- `--quant`: whether to quantize the model.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:39:- `--quant-method`: quantization methods of activations, which can be chosen from `minmax`, `ema`, `percentile` and `omse`.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/README.md:49:  title={P$^2$-ViT: Power-of-Two Post-Training Quantization and Acceleration for Fully Quantized Vision Transformer},`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/P2-ViT/models/plot_distrib.py:72:def plot_distribution(a, name, quant):`

## 5. HGTXR 적용 해석
- 가장 직접적인 P0 후보. HGTXR Q4 weight/Q8 activation에서 scale을 shift-friendly하게 재보정하면 정확도와 resource를 동시에 개선할 수 있다.

### 적용 가능 모듈
- Quantization / scale calibration / weight generation

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
