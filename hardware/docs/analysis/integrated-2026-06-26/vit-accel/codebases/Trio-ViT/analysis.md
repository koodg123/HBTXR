---
source_type: codebase
source_name: Trio-ViT
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/Trio-ViT/analysis.md -->

# Trio-ViT Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT`
- repo_remote: `origin	https://github.com/shihuihong214/Trio-ViT (fetch)`
- category: `softmax-free efficient ViT PTQ accelerator`
- HGTXR relevance: `medium`
- matched_paper: `Trio-ViT`
- paper_title: Trio-ViT: Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/Trio-ViT.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `41` |
| 주요 언어 | `Python:37, no_ext:2, Shell:1, Markdown:1` |
| LOC 추정 | `Python:5138, Markdown:28, Shell:20` |

### Directory Map
- `EfficientViT/` (1 entries)
- `EfficientViT/models/` (6 entries)
- `data/` (1 entries)
- `linklink/` (3 entries)
- `models/` (6 entries)
- `quant/` (9 entries)

### Metadata / Config Refs
- `README.md`

### Core Source Refs
- `quant/quant_layer.py`: 609 lines
- `EfficientViT/models/nn/ops.py`: 468 lines
- `quant/quant_block.py`: 464 lines
- `models/regnet.py`: 456 lines
- `models/resnet.py`: 298 lines
- `main_imagenet_dist.py`: 284 lines
- `main_imagenet.py`: 264 lines
- `quant/block_recon.py`: 255 lines
- `quant/data_utils.py`: 206 lines
- `EfficientViT/models/efficientvit/seg.py`: 203 lines

### Hardware-Oriented Source Refs
- no obvious HLS/RTL source found in first scan

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md`
- # Trio-ViT
- This repo contains the official implementation of **["Trio-ViT: Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer"](https://arxiv.org/abs/2405.03882).**
- ## Abstract
- ViTs' huge model sizes and intensive computations hinder their deployment on embedded devices, calling for effective model compression methods, such as quantization.
- Furthermore, at the hardware level, we build **an accelerator** dedicated to the specific Convolution-Transformer hybrid architecture of efficient ViTs, thereby enhancing hardware efficiency.
- ## Quantization
- ### Run
- Example: Quantize EfficientViT-b1-r224 with 8bit.
- ```bash
- python main_imagenet.py --data_path PATH_TO_IMAGENET  --n_bits_w 8 --channel_wise --weight 0.5 --model b1-r224 --disable_8bit_head_stem  --n_bits_a 8  --act_quant --input_size 224 --test_before_calibration
- ```
- - `--n_bits_w `: the quantization bit-width of weights
- - `-channel_wise`: whether to use channel-wise quantization for quantizing weights
- - `--model`: the chosed model to quantize
- - `--n_bits_a`: the quantization bit-width of activation
- - `--act_quant`: whether to quantize activation
- - `--input_size`: input resolution for dataset

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: softmax-free efficient ViT를 대상으로 tailored PTQ engine과 dedicated accelerator를 구성한다.
- 알고리즘 축: efficient ViT activation distribution을 반영한 PTQ와 conv-transformer hybrid operator mapping을 사용한다.
- 하드웨어 축: convolution-transformer hybrid architecture에 맞춘 dedicated accelerator로 operation type을 분리/융합한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:3:This repo contains the official implementation of **["Trio-ViT: Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer"](https://arxiv.org/abs/2405.03882).**`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:8:Unfortunately, due to the existence of hardware-unfriendly and quantization-sensitive non-linear operations, particularly Softmax, it is non-trivial to completely quantize all operations in ViTs, yielding either significant accuracy drops or non-negligible hardware costs. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:9:In response to challenges associated with standard ViTs, we focus our attention towards the quantization and acceleration for **efficient ViTs**, which not only eliminate the troublesome Softmax but also integrate linear attention with low computational complexity, and propose Trio-ViT accordingly. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:11:Specifically, at the algorithm level, we develop a **tailored post-training quantization engine** taking the unique activation distributions of Softmax-free efficient ViTs into full consideration, aiming to boost quantization accuracy. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/quant_model.py:49:            elif isinstance(child_module, nn.LayerNorm):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/quant_block.py:334:                # lightweight global attention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/quant_block.py:341:                kv[i] = torch.matmul(trans_k[i], v[i]) `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/quant_block.py:351:                q_kv[i] = torch.matmul(q[i], torch.cat([kv_0[i], kv_1[i]], dim=-1))`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/quant_block.py:409:            # lightweight global attention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/quant_block.py:416:            kv = torch.matmul(trans_k, v)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/quant_block.py:417:            out = torch.matmul(q, kv)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/data_utils.py:183:                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/quant_layer.py:405:                shifted_bias = torch.squeeze(torch.matmul(weight_sum, torch.unsqueeze(self.z, dim=1)))`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/quant_layer.py:594:                shifted_bias = torch.squeeze(torch.matmul(weight_sum, torch.unsqueeze(self.z, dim=1)))`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/EfficientViT/models/nn/norm.py:13:    "ln": nn.LayerNorm,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/EfficientViT/models/nn/ops.py:244:class MatMul(nn.Module):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/EfficientViT/models/nn/ops.py:253:    r""" Lightweight multi-scale attention """`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/EfficientViT/models/nn/ops.py:306:        self.kv_matmul = MatMul()`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/EfficientViT/models/nn/ops.py:307:        self.qkv_matmul = MatMul()`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/EfficientViT/models/nn/ops.py:336:        # lightweight global attention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:3:This repo contains the official implementation of **["Trio-ViT: Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer"](https://arxiv.org/abs/2405.03882).**`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:7:ViTs' huge model sizes and intensive computations hinder their deployment on embedded devices, calling for effective model compression methods, such as quantization. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:8:Unfortunately, due to the existence of hardware-unfriendly and quantization-sensitive non-linear operations, particularly Softmax, it is non-trivial to completely quantize all operations in ViTs, yielding either significant accuracy drops or non-negligible hardware costs. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:9:In response to challenges associated with standard ViTs, we focus our attention towards the quantization and acceleration for **efficient ViTs**, which not only eliminate the troublesome Softmax but also integrate linear attention with low computational complexity, and propose Trio-ViT accordingly. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:11:Specifically, at the algorithm level, we develop a **tailored post-training quantization engine** taking the unique activation distributions of Softmax-free efficient ViTs into full consideration, aiming to boost quantization accuracy. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:14:## Quantization`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:18:Example: Quantize EfficientViT-b1-r224 with 8bit.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:20:python main_imagenet.py --data_path PATH_TO_IMAGENET  --n_bits_w 8 --channel_wise --weight 0.5 --model b1-r224 --disable_8bit_head_stem  --n_bits_a 8  --act_quant --input_size 224 --test_before_calibration`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:22:- `--n_bits_w `: the quantization bit-width of weights`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:23:- `-channel_wise`: whether to use channel-wise quantization for quantizing weights`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:24:- `--model`: the chosed model to quantize`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:25:- `--n_bits_a`: the quantization bit-width of activation`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/README.md:26:- `--act_quant`: whether to quantize activation`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/block_recon.py:3:from quant.quant_layer import QuantModule, StraightThrough, lp_loss, QuantModule_Shifted, QuantModule_Scaled`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Trio-ViT/quant/block_recon.py:4:from quant.quant_model import QuantModel`

## 5. HGTXR 적용 해석
- softmax-free 변경은 HGTXR 논문 범위를 벗어날 수 있으므로 P2 paper-scope review가 필요하다.

### 적용 가능 모듈
- Quantization / scale calibration / weight generation
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
