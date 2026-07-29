---
source_type: codebase
source_name: AHCPTQ
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/AHCPTQ/analysis.md -->

# AHCPTQ Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ`
- repo_remote: `origin	https://github.com/Keio-CSG/AHCPTQ (fetch)`
- category: `vision PTQ/SAM quantization`
- HGTXR relevance: `medium`
- matched_paper: `AHCPTQ`
- paper_title: AHCPTQ: Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/AHCPTQ.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `1591` |
| 주요 언어 | `Python:1273, Markdown:146, YAML:97, .txt:19, Shell:10, no_ext:9, .png:7, .rst:6, C/C++ header:3, Notebook:3` |
| LOC 추정 | `Python:148576, Markdown:18232, YAML:12323, Shell:378, .txt:157, C/C++ header:125, Notebook:100, C++:57, JSON:1` |

### Directory Map
- `ahcptq/` (3 entries)
- `ahcptq/model/` (1 entries)
- `ahcptq/quantization/` (7 entries)
- `ahcptq/solver/` (4 entries)
- `exp/` (3 entries)
- `mmdetection/` (26 entries)
- `mmdetection/.circleci/` (1 entries)
- `mmdetection/.dev_scripts/` (15 entries)
- `mmdetection/.github/` (5 entries)
- `mmdetection/configs/` (91 entries)
- `mmdetection/demo/` (10 entries)
- `mmdetection/docker/` (2 entries)
- `mmdetection/docs/` (2 entries)
- `mmdetection/mmdet/` (7 entries)
- `mmdetection/requirements/` (8 entries)
- `mmdetection/resources/` (6 entries)
- `mmdetection/tests/` (7 entries)
- `mmdetection/tools/` (11 entries)
- `projects/` (2 entries)
- `projects/configs/` (5 entries)
- `projects/instance_segment_anything/` (3 entries)
- `requirements/` (8 entries)

### Metadata / Config Refs
- `README.md`
- `projects/instance_segment_anything/models/focalnet_dino/models/dino/util/coco_id2name.json`
- `mmdetection/.pre-commit-config.yaml`
- `mmdetection/CITATION.cff`
- `mmdetection/README_zh-CN.md`
- `mmdetection/README.md`
- `mmdetection/configs/groie/README.md`
- `mmdetection/configs/resnest/README.md`
- `mmdetection/configs/convnext/README.md`
- `mmdetection/configs/deformable_detr/README.md`
- `mmdetection/configs/pascal_voc/README.md`
- `mmdetection/configs/scnet/README.md`
- `mmdetection/configs/resnet_strikes_back/README.md`
- `mmdetection/configs/fast_rcnn/README.md`
- `mmdetection/configs/regnet/README.md`
- `mmdetection/configs/lvis/README.md`
- `mmdetection/configs/ssd/README.md`
- `mmdetection/configs/ghm/README.md`
- `mmdetection/configs/pafpn/README.md`
- `mmdetection/configs/carafe/README.md`
- `mmdetection/configs/fpg/README.md`
- `mmdetection/configs/guided_anchoring/README.md`
- `mmdetection/configs/efficientnet/README.md`
- `mmdetection/configs/tridentnet/README.md`
- `mmdetection/configs/pisa/README.md`
- `mmdetection/configs/gn+ws/README.md`
- `mmdetection/configs/gcnet/README.md`
- `mmdetection/configs/autoassign/README.md`
- `mmdetection/configs/vfnet/README.md`
- `mmdetection/configs/maskformer/README.md`

### Core Source Refs
- `mmdetection/mmdet/datasets/pipelines/transforms.py`: 2968 lines
- `mmdetection/docs/en/changelog.md`: 1800 lines
- `mmdetection/mmdet/models/dense_heads/solo_head.py`: 1197 lines
- `mmdetection/mmdet/models/utils/transformer.py`: 1167 lines
- `mmdetection/tests/test_data/test_pipelines/test_transform/test_transform.py`: 1118 lines
- `projects/instance_segment_anything/models/focalnet_dino/models/dino/deformable_transformer.py`: 1104 lines
- `mmdetection/mmdet/core/mask/structures.py`: 1102 lines
- `mmdetection/mmdet/models/dense_heads/corner_head.py`: 1086 lines
- `mmdetection/mmdet/models/dense_heads/yolact_head.py`: 1018 lines
- `mmdetection/configs/hrnet/metafile.yml`: 971 lines

### Hardware-Oriented Source Refs
- `projects/instance_segment_anything/ops/src/vision.cpp`
- `projects/instance_segment_anything/ops/src/ms_deform_attn.h`
- `projects/instance_segment_anything/ops/src/cpu/ms_deform_attn_cpu.h`
- `projects/instance_segment_anything/ops/src/cpu/ms_deform_attn_cpu.cpp`
- `projects/instance_segment_anything/ops/src/cuda/ms_deform_attn_cuda.h`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/README.md`
- # AHCPTQ: Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model
- ## 1. Environment Settings
- ### 1.1 Create Environment
- We follow the environment settings of [PTQ4SAM](https://github.com/chengtao-lv/PTQ4SAM), please refer to the ``environment.sh`` in the root directory.
- 1. Install PyTorch
- ```
- conda create -n ahcptq python=3.7 -y
- pip install torch torchvision
- ```
- 2. Install MMCV
- ```
- pip install -U openmim
- mim install "mmcv-full<2.0.0"
- ```
- 3. Install other requirements
- ```
- pip install -r requirements.txt
- ```
- 4. Compile CUDA operators
- ```

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: Hybrid Log-Uniform Quantization(HLUQ)과 Channel-Aware Grouping(CAG)을 결합해 hardware-compatible PTQ를 구성한다.
- 알고리즘 축: HLUQ는 작은 값 dense 영역은 log2 형태로, sparse large 값은 uniform 형태로 다루며 CAG는 유사 activation channel을 묶어 공유 quant parameter를 사용한다.
- 하드웨어 축: FPGA 구현에서 HLUQ/CAG가 복잡한 full-precision 보정 없이 quantized execution에 맞게 설계된다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/README.md:113:- [customize data pipelines](docs/en/tutorials/data_pipeline.md)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/projects/configs/yolox/yolo_l-sam-vit-h.py:22:# test_pipeline, NOTE the Pad's size_divisor is different from the default`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/projects/configs/yolox/yolo_l-sam-vit-h.py:26:test_pipeline = [`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/projects/configs/yolox/yolo_l-sam-vit-h.py:52:        pipeline=test_pipeline),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/projects/configs/yolox/yolo_l-sam-vit-h.py:57:        pipeline=test_pipeline))`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py:41:train_pipeline = [`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py:63:test_pipeline = [`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py:92:    train=dict(pipeline=train_pipeline),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py:93:    val=dict(pipeline=test_pipeline),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py:94:    test=dict(pipeline=test_pipeline))`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py:41:train_pipeline = [`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py:63:test_pipeline = [`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py:92:    train=dict(pipeline=train_pipeline),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py:93:    val=dict(pipeline=test_pipeline),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py:94:    test=dict(pipeline=test_pipeline))`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py:41:train_pipeline = [`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py:63:test_pipeline = [`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py:92:    train=dict(pipeline=train_pipeline),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py:93:    val=dict(pipeline=test_pipeline),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py:94:    test=dict(pipeline=test_pipeline))`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/README.md:97:The Segment Anything Model (SAM) has demonstrated strong versatility across various visual tasks. However, its large storage requirements and high computational cost pose challenges for practical deployment. Post-training quantization (PTQ) has emerged as an effective strategy for efficient deployment, but we identify two key challenges in SAM that hinder the effectiveness of existing PTQ methods: the heavy-tailed and skewed distribution of post-GELU activations, and significant inter-channel variation in linear projection activations. To address these challenges, we propose AHCPTQ, an accurate and hardware-efficient PTQ method for SAM. AHCPTQ introduces hardware-compatible Hybrid Log-Uniform Quantization (HLUQ) to manage post-GELU activations, employing log2 quantization for dense small values and uniform quantization for sparse large values to enhance quantization resolution. Additionally, AHCPTQ incorporates Channel-Aware Grouping (CAG) to mitigate inter-channel variation by progressively clustering activation channels with similar distributions, enabling them to share quantization parameters and improving hardware efficiency. The combination of HLUQ and CAG not only enhances quantization effectiveness but also ensures compatibility with efficient hardware execution. For instance, under the W4A4 configuration on the SAM-L model, AHCPTQ achieves 36.6\% mAP on instance segmentation with the DINO detector, while achieving a $7.89\times$ speedup and $8.64\times$ energy efficiency over its floating-point counterpart in FPGA implementation.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/quantization/fake_quant.py:590:        softmax_mask = x_int >= levels`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/quantization/fake_quant.py:593:        X[softmax_mask] = torch.Tensor([0.0])`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/quantization/quantized_module_matmul.py:7:from tools.modifier import MatMul`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/quantization/quantized_module_matmul.py:175:        if 'MatMul' in str(type(module)):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/quantization/quantized_module.py:252:class QuantizedMatMul(QuantizedModule):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/quantization/util_quant.py:21:    softmax_mask = ((x_int >= levels))`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/quantization/util_quant.py:24:    X[softmax_mask] = torch.Tensor([0.0])`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/quantization/util_quant.py:91:    softmax_mask = (xq >= levels_log)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/quantization/util_quant.py:94:    xq[softmax_mask] = torch.Tensor([0.0])`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/solver/test_quant.py:35:from ahcptq.quantization.quantized_module import QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/solver/test_quant.py:441:            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.GELU)):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/solver/test_quant.py:507:            if isinstance(child_module, (QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul)):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/model/quant_model.py:8:from projects.instance_segment_anything.models.segment_anything.modeling.transformer import Attention, TwoWayAttentionBlock`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AHCPTQ/ahcptq/model/quant_model.py:10:from ahcptq.quantization.quantized_module import PreQuantizedLayer,QuantizedMatMul`

## 5. HGTXR 적용 해석
- 눈 영역 segmentation/ROI preprocessing을 SAM류 보조 네트워크로 붙일 경우 유용하다. HGTXR backbone 자체에는 P0가 아니라 보조 SW accuracy 실험으로 제한한다.

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
