# Markdown Prompt / Template Analysis for Uploaded Files

## 0. Analysis Coverage

| File | Lines | Primary Target | Status |
|---|---:|---|---|
| `general_algorithm_codebase_analysis_prompt.md` | 678 | General software / algorithm codebase analysis | Fully inspected as prompt structure |
| `general_algorithm_codebase_analysis_template.md` | 649 | General software / algorithm codebase analysis report | Fully inspected as template structure |
| `hardware_codebase_analysis_prompt.md` | 848 | RTL/HLS hardware repository analysis | Fully inspected as prompt structure |
| `hardware_codebase_analysis_template.md` | 994 | RTL/HLS hardware repository analysis report | Fully inspected as template structure |
| `Paper-Review-Summary-Prompt.md` | 371 | Research paper analysis | Fully inspected as prompt structure |
| `Paper-Review-Summary-Template.md` | 134 | Research paper analysis report | Inspected; visible template ends at Section 4.4, so later sections are absent in the provided template file |

분석 관점에서 업로드된 Markdown 파일은 세 부류로 나뉜다. 첫째, 일반 알고리즘 코드베이스 분석용 prompt/template은 software repository의 구조, 실행 흐름, 함수 의미, call graph, test/reproducibility를 강하게 요구한다. 둘째, hardware RTL/HLS 분석용 prompt/template은 module hierarchy, interface protocol, clock/reset, CDC/RDC, FSM, pipeline, memory, HLS pragma, synthesis/implementation report를 요구한다. 셋째, 논문 분석용 prompt/template은 논문 전체를 읽고 9개 고정 섹션으로 구조화하는 것을 목표로 한다.

---

# 1. General Algorithm Codebase Analysis Prompt 분석

## 1.1 목적

이 prompt는 단순 README 요약이 아니라 repository 전체를 정적 분석 대상으로 삼는다. 요구 범위에는 root structure, source code, package/module structure, entry points, configuration, dependency manifests, build scripts, training/inference/evaluation scripts, tests, CI/CD, examples, benchmark scripts, utility modules, TODO/FIXME/HACK, generated/vendored/submodule 여부가 포함된다.

## 1.2 강점

- **분석 범위가 넓음:** README, docs, examples 수준을 넘어 실제 코드와 실행 script까지 확인하도록 강제한다.
- **검증 중심:** 실행 가능성, dependency, dataset/checkpoint, benchmark 재현성, test coverage를 분리하여 평가하게 한다.
- **근거 기반:** file path, function/class/module/config/CLI/test/script 단위의 evidence를 요구한다.
- **심층 분석 부록:** call graph, function semantic table, algorithm extraction, control/data flow, dependency impact, code quality metrics까지 요구한다.

## 1.3 한계

- 매우 큰 repository에서는 모든 function/method의 완전한 call graph 생성이 현실적으로 어렵다.
- Python dynamic import, registry, callback, reflection, framework-level dispatch는 정적 분석만으로 불완전할 수 있다.
- 실행 검증을 요구하지만, dataset/checkpoint/private dependency가 없으면 실제 reproducibility 검증은 제한된다.

## 1.4 적합한 사용 대상

| 대상 | 적합도 | 이유 |
|---|---:|---|
| PyTorch/TensorFlow training repo | High | training/inference/eval/config/test 구조 분석에 적합 |
| Classical algorithm library | High | function-level semantic and algorithm extraction에 적합 |
| Compiler/runtime codebase | High | IR/pass/scheduler/executor 분석 구조로 확장 가능 |
| 논문 PDF 단독 분석 | Low | 코드 중심 prompt이므로 논문만 있는 경우 부적합 |
| RTL/HLS repo | Medium-Low | 일부 구조 분석은 가능하지만 hardware-specific interface/timing 검증이 약함 |

---

# 2. General Algorithm Codebase Analysis Template 분석

## 2.1 구조

Template은 다음 순서로 구성된다.

1. Repository Information
2. Overview
3. Codebase Background
4. Related Works / Dependency Ecosystem
5. Key Concepts
6. Architecture and Methodology
7. Experiments / Tests / Benchmark Results
8. Summary
9. Pros.
10. Cons.
11. Appendix A-H

## 2.2 장점

- 표 기반으로 repository metadata, entry point, dependency, concept map, module responsibility, runtime flow를 정리하기 좋다.
- Appendix가 실무 code review 문서로 바로 사용할 수 있을 정도로 상세하다.
- execution command와 validation gaps를 분리하여 “실행할 수 있는 코드인지”를 판단하기 좋다.

## 2.3 한계

- 모든 항목을 채우면 문서가 매우 길어지므로, 작은 코드베이스가 아닌 경우 문서 분할이 필요하다.
- line number와 call graph는 자동화 도구와 결합해야 품질이 안정된다.
- paper-only task에는 template 항목 대부분이 비어 `코드베이스에서 확인되지 않음`으로 남게 된다.

---

# 3. Hardware RTL/HLS Codebase Analysis Prompt 분석

## 3.1 목적

이 prompt는 RTL/HLS hardware repository의 구조와 구현 가능성을 검토하기 위한 전문 prompt다. 단순 top module/testbench 확인이 아니라 source hierarchy, interface, clock/reset, CDC/RDC, datapath/control path, FSM, pipeline, buffer/FIFO/SRAM/BRAM/URAM, DMA/NoC/interconnect, HLS pragma, constraints, simulation/synthesis/implementation reports까지 분석하게 한다.

## 3.2 강점

- **하드웨어 검증 관점이 명확함:** simulation, synthesis, implementation, timing closure, power, board validation을 구분한다.
- **RTL/HLS 구분:** HLS C/C++ 함수 의미와 합성 후 hardware meaning을 분리하도록 요구한다.
- **interface/protocol 중심:** AXI4, AXI4-Lite, AXI4-Stream, APB/AHB/Avalon/custom stream 등 protocol-level 분석에 적합하다.
- **timing/resource 현실성:** WNS/TNS, achieved clock, LUT/FF/BRAM/DSP/URAM, power report가 없으면 단정하지 않도록 한다.

## 3.3 한계

- vendor encrypted IP, generated RTL, block design, Vivado project metadata가 누락되면 완성도 있는 top hierarchy 검증이 어렵다.
- CDC/RDC safety는 정적 텍스트 분석만으로 확정할 수 없고 별도 lint/formal/CDC tool 검증이 필요하다.
- HLS schedule report 없이 pragma 효과를 단정할 수 없다.

## 3.4 적합한 사용 대상

| 대상 | 적합도 | 이유 |
|---|---:|---|
| Verilog/SystemVerilog accelerator repo | High | module hierarchy, FSM, memory, interface 분석에 적합 |
| Vitis/Vivado HLS project | High | top function, pragma, C-sim/Cosim/HLS report 분석에 적합 |
| FPGA board demo | High | constraints, scripts, bitstream/board integration 분석 가능 |
| ASIC RTL prototype | Medium-High | synthesis/timing/power report가 있으면 적합 |
| 논문 PDF 단독 분석 | Low | hardware paper를 읽을 수는 있으나 codebase verification prompt는 아님 |

---

# 4. Hardware RTL/HLS Codebase Analysis Template 분석

## 4.1 구조

Template은 Repository Information 이후 9개 본문 섹션과 Appendix A 이후의 hardware-specific 부록을 포함한다. 주요 구성은 Top-Level Architecture, Interface Architecture, Datapath, Control/FSM, Memory, Pipeline/Scheduling, Clock/Reset, HLS Pragmas, Build/Simulation/Synthesis Flow, Resource/Timing/Power 분석이다.

## 4.2 장점

- hardware review 문서로 바로 제출 가능한 수준의 항목을 갖고 있다.
- top module candidate, final top module, instance hierarchy, unused/blackbox module을 구분한다.
- HLS top function, call graph, algorithm-to-hardware mapping, signal dependency, FSM transition, performance model까지 포함한다.

## 4.3 한계

- 실제 design report가 없는 repository에서는 Simulation/Synthesis/Implementation Results 섹션이 대부분 미확인으로 남는다.
- fully generated RTL이나 IP-XACT/block design 중심 프로젝트는 원본 generator/script 분석이 병행되어야 한다.
- 논문 PDF 분석 결과를 이 template에 억지로 넣으면 “source code 근거”와 “paper claim”이 혼재될 위험이 있다.

---

# 5. Paper Review Summary Prompt 분석

## 5.1 목적

이 prompt는 논문 PDF 전체를 대상으로 연구 문제, 기존 연구 한계, 제안 방법의 설계 논리, 실험 결과의 의미, 장점과 한계를 구조적으로 분석하도록 설계되어 있다. 지정된 9개 섹션 순서는 다음과 같다.

1. Overview
2. Research Background
3. Related Works
4. Key Concepts
5. Methodology
6. Experiments Results
7. Summary
8. Pros.
9. Cons.

## 5.2 강점

- **전체 논문 검토 원칙:** Abstract/Introduction/Method/Experiment/Conclusion만 선택적으로 읽지 말고 Figure/Table/Equation/Appendix까지 보도록 한다.
- **근거 기반:** Section, page, equation, algorithm, figure, table 번호 근거를 요구한다.
- **비교 공정성:** Dataset, Backbone, Resolution, Training Setting, Metric이 동일한지 구분하도록 한다.
- **저자 주장과 분석자 해석 분리:** 과장된 논문 claim을 그대로 받아쓰지 않게 한다.
- **실험 해석 강화:** Quantitative/Qualitative/Ablation/Efficiency/Result Interpretation을 분리한다.

## 5.3 본 작업에 적용한 방식

본 패키지의 논문별 Markdown 분석 문서는 이 prompt를 기준으로 작성했다. 특히 각 문서에서 다음을 유지했다.

- 9개 섹션 순서 유지
- 논문에 없는 정보는 `논문에 명시되지 않음` 또는 `제공된 PDF 기준 확인 제한`으로 표시
- 주요 수치에는 가능한 경우 Table/Figure/Section 근거 표시
- 직접 비교가 어려운 경우 조건 차이를 명시
- 논문 저자의 claim과 분석자의 해석을 분리

---

# 6. Paper Review Summary Template 분석

## 6.1 확인된 구조

업로드된 `Paper-Review-Summary-Template.md`는 다음 부분까지 포함한다.

- Paper Information
- 1. Overview
- 2. Research Background
- 3. Related Works
- 4. Key Concepts
- 4.4 Concept Relationships

## 6.2 핵심 문제

제공된 template 파일은 Section 5 이후가 포함되어 있지 않다. 즉, prompt가 요구하는 `Methodology`, `Experiments Results`, `Summary`, `Pros.`, `Cons.`가 template 파일 내부에는 존재하지 않는다.

## 6.3 본 작업의 처리 방식

본 작업에서는 template의 visible section style을 최대한 유지하되, 누락된 Section 5-9는 `Paper-Review-Summary-Prompt.md`가 요구한 9개 섹션 순서에 맞추어 확장했다. 따라서 산출물은 “제공된 template + prompt-required sections” 형태다.

---

# 7. Prompt 선택 기준

| 분석 대상 | 사용할 Prompt | 사용할 Template | 이유 |
|---|---|---|---|
| 논문 PDF | Paper-Review-Summary-Prompt | Paper template를 확장 | 본 작업의 주 대상 |
| EV-Eye/FACET/SEE code zip | General Algorithm Codebase Analysis Prompt | General Algorithm Template | Python training/inference repo 분석에 적합 |
| HG-PIPE/ESDA HLS/RTL zip | Hardware RTL/HLS Codebase Analysis Prompt | Hardware Template | HLS/RTL accelerator repo 분석에 적합 |
| 논문 + 코드 동시 분석 | Paper prompt + codebase prompt 병행 | 문서 분리 권장 | claim과 implementation evidence를 분리해야 함 |

---

# 8. 품질 향상용 통합 프롬프트

```text
당신은 event-based eye tracking, XR gaze tracking, model compression, FPGA/ASIC accelerator co-design을 모두 이해하는 논문 분석가입니다.
첨부된 논문 PDF 전체를 읽고, Paper-Review-Summary-Prompt.md의 9개 섹션 순서를 유지하여 Markdown 분석 문서를 작성하십시오.
단, 업로드된 Paper-Review-Summary-Template.md는 Section 4.4까지만 제공되어 있으므로, Section 5-9는 prompt의 요구사항에 맞추어 확장하십시오.

분석 시 다음을 반드시 지키십시오.
1. 논문에 명시된 정보와 분석자의 해석을 구분합니다.
2. 주요 결과 수치는 Table/Figure/Section 근거와 함께 기록합니다.
3. Dataset, metric, evaluation frequency, hardware platform이 다른 결과는 직접 비교하지 않습니다.
4. 알고리즘 pipeline, event representation, temporal modeling, loss, training, inference, latency/energy/resource를 분리합니다.
5. 논문에 없는 정보는 `논문에 명시되지 않음`으로 표시합니다.
6. 최종 문서는 Markdown 형식으로 작성하고, 9개 섹션을 빠짐없이 포함합니다.
```

---

# 9. 결론

업로드된 Markdown 패키지는 software codebase, hardware RTL/HLS codebase, paper review라는 세 가지 서로 다른 분석 작업을 위한 prompt/template 세트다. 이번 요청의 주 대상은 `XR-Eye-Tracking.zip` 내부 논문 PDF이므로, 논문별 분석에는 `Paper-Review-Summary-Prompt.md`가 가장 적합하다. 다만 paper template가 Section 4.4에서 끝나는 구조적 누락이 있으므로, 본 산출물에서는 prompt의 9개 필수 섹션을 기준으로 template를 확장하여 각 논문별 Markdown 문서를 생성했다.
