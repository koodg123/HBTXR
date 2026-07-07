---
source_type: codebase
source_name: Lightening-Transformer-AE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-hardware-codebase
---

# Lightening-Transformer-AE Analysis

- corpus: `References/Hardware`
- local_path: `/home/kjm26/project/PRJXR/References/Hardware/Lightening-Transformer-AE`
- category: `Transformer accelerator`
- HGTXR relevance: `Attention/encoder comparison`
- recommended_priority: `P2`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/Hardware` 하위의 `Lightening-Transformer-AE` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `114` |
| 주요 확장자 | `.py:41, .sample:14, .yaml:14, [no_ext]:13, .csv:11, .sh:10, .md:4, .log:3` |
| README 감지 | `readme.md` |

## 3. 핵심 파일 / 구성요소

- `transformer-focused hardware`

## 4. README / 문서 근거 요약

> # Lightening-Transformer HPCA 2024 AE
> This contains the codebases for the main codebases of the paper "Lightening-Transformer: A Dynamically-operated Optically-interconnected Photonic Transformer Accelerator".
> ---
> ## Usage of the Provided Codebase
> We provides three kinds of codebases:
> * (1) algorithm codes for training/running models on our photonic accelerator, with the analytic transformation of our unique photonic tensor core embedded in the computation process. See `./software_model` for detailed implementation and usages, including the [DeiT](https://arxiv.org/abs/2012.12877) case.
> * (2) hardware simulator for estimating the energy and latency running Transformers on our photonic accelerator. See `./hardware_simulator` for detailed implementation and usages.
> * (3) profile codes for profiling latency and power usage of running Transformers on GPU. See `./profile` for detailed implementation and usages. The implementation refers to [Neurips'23, Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers](https://github.com/zixuanjiang/pre-rmsnorm-transformer).
> ---
> ## Required Dependencies
> The DeiT requires to install PyTorch and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models).
> ```

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `Transformer accelerator` |
| HGTXR 적용성 | `Attention/encoder comparison` |
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
| 예상 효과 | `Attention/encoder comparison` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`Lightening-Transformer-AE`는 `P2` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
