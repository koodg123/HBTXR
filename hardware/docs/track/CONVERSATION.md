> **작성** 2026-07-05 · **갱신** 2026-07-15
> **상태** active — 내용은 2026-07-15 구 트리에서 이어받음, 재구성 진행분을 여기에 이어 씁니다
> **소유** hardware
>
> **사용자 요구 변경 기록.** `archive/hardware/docs/track/CONVERSATION.md` 에서 2026-07-29 복사, 내용 무변경.
> 2026-07-15 이후 하드웨어 작업이 없어 그 시점 상태 그대로입니다.

# HGTXR Hardware Conversation Summary

Date: 2026-06-15

## User Decisions
- Main working directory is `HGTXR/hardware`.
- Use Codex-native model hierarchy and real sub-agents when available.
- Preserve choice records when multiple paths exist.
- Selected flow: A2 work followed by A1, and C path in parallel; E pending.
- Resource direction: increase DSP/URAM utilization, avoid mapping arithmetic wholesale into LUTs, use LUTRAM for small memories.
- Add ViT accelerator paper/codebase findings to the third-goal plan and continue work.

## Current Interpretation
Third-goal work must not reset C3b. Reference-derived improvements enter as VREF successor experiments or software-first ablations.

## 2026-07-05 Commit And Handover Snapshot

- The user requested that all HGTXR work progress and conversations be
  documented, then all changes tracked and committed.
- Current repository topology was rechecked: root, `software/`, and `hardware/`
  all resolve to `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`, so the commit target
  is the single root repository.
- A comprehensive continuation summary was added at
  `hardware/docs/track/SESSION_PROGRESS_AND_CONVERSATION_2026_07_05.md`.
- That document consolidates the conversation from git strategy, Codex
  environment setup, power/clock analysis, Search/Track shared-top semantics,
  full learned path restoration, AQ2 metrics, pressure-relief DSE, ZCU104
  reference analysis, and the current commit request.
- Current technical interpretation: AQ2 is the clean reporting baseline;
  fastest HLS Search/Track pair remains useful as a latency lower-bound
  reference but not promotable due DSP/LUT overuse; the next real closure work is
  resource-preserving Search latency reduction through GELU ROM banking and
  shared/time-multiplexed W2.
