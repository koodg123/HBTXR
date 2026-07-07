---
source_type: codebase
source_name: ViT-FPGA-TPU
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-hardware-codebase
---

# ViT-FPGA-TPU Analysis

- corpus: `References/Hardware`
- local_path: `/home/kjm26/project/PRJXR/References/Hardware/ViT-FPGA-TPU`
- category: `FPGA/TPU-style ViT`
- HGTXR relevance: `Host/accelerator interface reference`
- recommended_priority: `P1`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/Hardware` 하위의 `ViT-FPGA-TPU` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `336` |
| 주요 확장자 | `.sv:47, .v:26, .xci:24, [no_ext]:16, .sample:14, .h:13, .cpp:12, .tcl:11` |
| README 감지 | `README.md` |

## 3. 핵심 파일 / 구성요소

- `code_fpga/accel_driver/include/accelerator.h`

## 4. README / 문서 근거 요약

> # ViT-FPGA-TPU
> FPGA based Vision Transformer accelerator (Harvard CS205)
> ![accel_arch](images/accel_arch.png)
> # To Run ViT in libTorch (PyTorch C++ API)
> #### Go into the `code/code_cpu` directory
> - `cd code/code_cpu`
> - Everything below is done in the `code/code_cpu` directory!!!
> #### First we download the LibTorch library if it's not in the folder yet:
> - `wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip`
> - `unzip libtorch-shared-with-deps-latest.zip`
> #### Then build the project
> - `mkdir build`

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `FPGA/TPU-style ViT` |
| HGTXR 적용성 | `Host/accelerator interface reference` |
| 우선순위 | `P1` |
| 직접 HW 구현성 | 높음 |

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
| 예상 효과 | `Host/accelerator interface reference` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`ViT-FPGA-TPU`는 `P1` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
