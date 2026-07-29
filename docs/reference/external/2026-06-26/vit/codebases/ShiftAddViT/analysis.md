---
source_type: codebase
source_name: ShiftAddViT
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-vit-codebase
---

# ShiftAddViT Analysis

- corpus: `References/ViT`
- local_path: `/home/kjm26/project/PRJXR/References/ViT/ShiftAddViT`
- category: `Shift/add operator replacement and TVM/CUDA kernels`
- HGTXR relevance: `Operator-level cost study; use cautiously with DSP policy`
- recommended_priority: `P2`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/ViT` 하위의 `ShiftAddViT` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `166` |
| 주요 확장자 | `.py:85, [no_ext]:23, .sample:14, .txt:12, .o:7, .cpp:4, .egg:4, .so:4` |
| README 감지 | `README.md` |

## 3. 핵심 파일 / 구성요소

- `pvt/hw_utils.py`
- `OPs_Speedups/*`
- `custom CUDA/C++ kernels`

## 4. README / 문서 근거 요약

> # ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer
> [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)
> **Haoran You***, Huihong Shi*, Yipin Guo* and Yingyan Lin
> Accepted by [**NeurIPS 2023**](https://neurips.cc/). More Info:
> \[ [**Paper**](https://arxiv.org/abs/2306.06446) | [**Slide**](https://neurips.cc/media/neurips-2023/Slides/70751_L4FOulc.pdf) | [**Project**](https://neurips.cc/virtual/2023/poster/70751) | [**Poster**](https://drive.google.com/file/d/1QWsQXQc7hdXKd0WQqu_vTr8wU9833eox/view?usp=sharing) | [**Github**](https://github.com/GATECH-EIC/ShiftAddViT/) \]
> ---
> **Updates**
> * We have made the entire code for PVT models publicly available, encompassing training, evaluation, TVM compilation of the entire model, and subsequent throughput measurements and comparisons. For additional information, refer to the `./pvt` directory.
> * We have also released the unit test for our MatAdd and MatShift kernels constructed with TVM. This test enables you to replicate the comparison results illustrated in Figures 4 and 5 of our paper. Please refer to the `./Ops_Speedups` folder for more information.
> **ToDos**
> * Publish the pre-trained checkpoints and provide the corresponding expected TVM output in the form of a `.json` file for replicating our results.
> * Upload the presentation to Youtube and share the link.

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `Shift/add operator replacement and TVM/CUDA kernels` |
| HGTXR 적용성 | `Operator-level cost study; use cautiously with DSP policy` |
| 우선순위 | `P2` |
| 직접 HW 구현성 | 중간/간접 |

## 6. HGTXR 실험 변환

| 단계 | 작업 | 산출물 | 승격 조건 |
|---|---|---|---|
| SW/분석 | 원본 코드/문서에서 파라미터, 연산자, 데이터플로우를 추출 | source notes, mapping table | HGTXR baseline과 비교 가능한 metric 정의 |
| HLS/RTL 후보화 | HGTXR C3b successor에 맞는 bounded variant로 축소 | compile-time define 또는 별도 experiment variant | csim/csynth 통과 |
| 리소스 정책 | DSP-bound MAC, URAM large buffer, LUTRAM small table 원칙 적용 | block-level resource report | DSP/LUT/URAM 목표 방향 개선 |
| 검증 | accuracy/error/latency/resource를 기존 C3b evidence와 분리 보고 | validation report | static claim이 아닌 측정 artifact 확보 |

## 7. 예상 효과와 리스크

| 항목 | 내용 |
|---|---|
| 예상 효과 | `Operator-level cost study; use cautiously with DSP policy` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`ShiftAddViT`는 `P2` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
