---
source_type: codebase
source_name: RepViT
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-vit-codebase
---

# RepViT Analysis

- corpus: `References/ViT`
- local_path: `/home/kjm26/project/PRJXR/References/ViT/RepViT`
- category: `Mobile CNN/ViT hybrid deployment`
- HGTXR relevance: `Lightweight front-end or local feature extractor reference`
- recommended_priority: `P1`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/ViT` 하위의 `RepViT` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `213` |
| 주요 확장자 | `.py:117, [no_ext]:19, .sample:14, .sh:14, .jpg:12, .txt:12, .json:6, .md:6` |
| README 감지 | `README.md` |

## 3. 핵심 파일 / 구성요소

- `README.md`
- `export_coreml.py`
- `main.py`
- `sam/`
- `detection/`
- `segmentation/`

## 4. README / 문서 근거 요약

> # [RepViT-SAM: Towards Real-Time Segmenting Anything](https://arxiv.org/abs/2312.05760)
> # [RepViT: Revisiting  Mobile CNN From ViT Perspective](https://arxiv.org/abs/2307.09283)
> Official PyTorch implementation of **RepViT-SAM** and **RepViT**. CVPR 2024.
> <p align="center">
>   <img src="sam/figures/comparison.png" width=80%> <br>
>   Models are deployed on iPhone 12 with Core ML Tools to get latency.
> </p>
> <p align="center">
>   <img src="figures/latency.png" width=70%> <br>
>   Models are trained on ImageNet-1K and deployed on iPhone 12 with Core ML Tools to get latency.
> </p>
> [RepViT-SAM: Towards Real-Time Segmenting Anything](https://arxiv.org/abs/2312.05760).\

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `Mobile CNN/ViT hybrid deployment` |
| HGTXR 적용성 | `Lightweight front-end or local feature extractor reference` |
| 우선순위 | `P1` |
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
| 예상 효과 | `Lightweight front-end or local feature extractor reference` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`RepViT`는 `P1` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
