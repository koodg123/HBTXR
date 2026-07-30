> **작성** 2026-07-05 · **갱신** 2026-07-05
> **상태** active — 내용은 2026-07-15 구 트리에서 이어받음, 재구성 진행분을 여기에 이어 씁니다
> **소유** hardware

# HGTXR Hardware ADR

## ADR-2026-06-15-01: Keep C3b As Third-Goal Gate While Adding VREF Experiments

- Status: accepted
- Context: ViT accelerator references suggest useful quantization, memory, pipeline, sparse attention, and MoE ideas.
- Decision: C3b remains the immediate board-smoke gate. VREF experiments are added as successors/ablations.
- Consequences:
  - P2-ViT and ME-ViT can proceed before board access as SW/audit tasks.
  - Attention replacement and MoE routing require paper-scope review.
  - LUT-heavy methods remain negative controls under current resource policy.

## ADR-2026-06-15-02: Memory Placement Policy

- Status: accepted
- Decision: large frame/global/QKV/hidden/deep FIFO buffers prefer URAM; small tables, short control buffers, and tiny FIFOs prefer LUTRAM or BRAM.
- Rationale: user explicitly requested higher URAM/DSP use with lower LUT pressure, while avoiding wasteful URAM use on small memories.

## ADR-2026-06-16-01: C3b Protection Before VREF Promotion

- Status: accepted
- Context: C3b is the current ZCU104 board-smoke candidate and final signoff still has external blockers.
- Decision: VREF work may proceed as software/static/audit work, but no VREF successor may replace or mutate the C3b baseline unless it passes the C3b protection checklist.
- Required successor gates: no C3b overwrite, latency `<= 37508072`, WNS `>= 4.415 ns`, DSP `<= 604`, LUT `<= 126506`, URAM `<= 64`, valid bit/hwh topology, physical smoke evidence, and XR-VITs reference policy.
- Consequence: C3b remains the reference baseline while VREF improvements are evaluated as bounded successors.

## ADR-2026-07-30-01: 코드 정리는 M3 이후 — `module/`이 정본이 된 뒤

- Status: accepted
- Context: `module/`의 이름·구조·내부 동작을 정리하려 했으나, tcl이 `hardware/hls/`를
  리터럴로 참조합니다 (`add_files [file join $hw_dir hls src ...]`). **`module/`을 고쳐도
  빌드되는 코드는 바뀌지 않습니다.**
- Decision: 정리를 M3(빌드 스크립트 이관 + 경로 재지정) **뒤로** 미룹니다.
- Rationale: 지금 고치면 M3에서 경로가 바뀔 때 두 번 봐야 합니다. M3가 끝나면
  `module/`이 정본이 되고, 같은 작업을 한 번에 끝냅니다.
- 부수 정정: `module/README.md`가 "전환은 M5"라고 적고 있었는데 틀렸습니다. 도구가 읽는
  `config/`는 M5, **HLS 빌드가 읽는 `module/`은 M3**입니다.
- Consequence: M3-2(경로 재지정)가 코드 정리의 선행 조건입니다. 그 안에
  `-I ../golden`(M2 부채)과 `run_cyclic_*` 러너 추가가 함께 들어갑니다.

## ADR-2026-07-30-02: 컴파일러 없이 "파일 합치기는 안전"은 성립하지 않습니다

- Status: accepted
- Context: ponytail 감사가 `module/`에서 **−2,100줄 · −13파일**을 찾았고, 그중 파편 헤더 7개를
  `common.h`로, 얇은 `src/` 4개를 `hgtxr_top.cpp`로 합치는 것을 "컴파일러 불필요, 안전"으로
  분류했습니다.
- 측정 결과 아니었습니다: `fixed_types.h`를 **4곳**, `config.h`를 **2곳**이 직접 include하고
  (총 11곳), 얇은 함수 6개가 **각 4~5곳**에서 호출됩니다.
- Decision: 철회. **파일 이동조차 include·호출 그래프를 건드리면 컴파일러가 필요합니다.**
- 컴파일러 없이 증명 가능한 것은 셋뿐: tcl 플래그로 도달 불가한 golden 2개(플래그 15 vs
  파일 17) · 짝 없는 `blocks2_spec.json` · 어떤 빌드도 세우지 않는 플래그 2개.
- Rationale: 이 트리의 테스트 426개는 전부 Python 도구 테스트이고 **HLS를 건드리는 것이
  0건**입니다. 잘못 합치면 조용히 깨지고 아무도 모릅니다.
