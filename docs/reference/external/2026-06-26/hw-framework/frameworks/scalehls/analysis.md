---
source_type: framework
source_name: scalehls
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-hw-framework
---

# scalehls Analysis

- corpus: `References/HW_Framework`
- local_path: `/home/kjm26/project/PRJXR/References/HW_Framework/scalehls`
- category: `MLIR/ScaleHLS compiler`
- HGTXR relevance: `Compiler-level HLS generation reference`
- recommended_priority: `P2`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/HW_Framework` 하위의 `scalehls` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `121115` |
| 주요 확장자 | `.cpp:5469, .h:2659, .c:2376, .py:1514, .ll:1380, .s:1146, [no_ext]:970, .test:780` |
| README 감지 | `README.md` |

## 3. 핵심 파일 / 구성요소

- `lib/Translation/EmitHLSCpp.cpp`
- `tools/pyscalehls/pyscalehls.py`
- `build-scalehls.sh`

## 4. README / 문서 근거 요약

> # ScaleHLS Project
> [![Build and Test](https://github.com/hanchenye/scalehls/actions/workflows/buildAndTest.yml/badge.svg?branch=master)](https://github.com/hanchenye/scalehls/actions/workflows/buildAndTest.yml)
> ScaleHLS is a High-level Synthesis (HLS) framework on [MLIR](https://mlir.llvm.org). ScaleHLS can compile HLS C/C++ or PyTorch model to optimized HLS C/C++ in order to generate high-efficiency RTL design using downstream tools, such as Xilinx Vivado HLS.
> By using the MLIR framework that can be better tuned to particular algorithms at different representation levels, ScaleHLS is more scalable and customizable towards various applications coming with intrinsic structural or functional hierarchies. ScaleHLS represents HLS designs at multiple levels of abstraction and provides an HLS-dedicated analysis and transform library (in both C++ and Python) to solve the optimization problems at the suitable representation levels. Using this library, we've developed a design space exploration engine to generate optimized HLS designs automatically.
> For more details, please see our [HPCA'22](https://doi.org/10.1109/HPCA53966.2022.00060) and [DAC'22](https://doi.org/10.1145/3489517.3530631) paper:
> ```bibtex
> @inproceedings{yehpca2022scalehls,
>   title={ScaleHLS: A New Scalable High-Level Synthesis Framework on Multi-Level Intermediate Representation},
>   author={Ye, Hanchen and Hao, Cong and Cheng, Jianyi and Jeong, Hyunmin and Huang, Jack and Neuendorffer, Stephen and Chen, Deming},
>   booktitle={2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
>   year={2022}
> }

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `MLIR/ScaleHLS compiler` |
| HGTXR 적용성 | `Compiler-level HLS generation reference` |
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
| 예상 효과 | `Compiler-level HLS generation reference` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`scalehls`는 `P2` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
