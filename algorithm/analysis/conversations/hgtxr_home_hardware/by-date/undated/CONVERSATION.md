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
