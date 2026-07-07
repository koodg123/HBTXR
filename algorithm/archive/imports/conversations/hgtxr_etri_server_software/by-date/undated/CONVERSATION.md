# HGTXR-SW Conversation Context

Date: 2026-06-10

## 2026-06-27 Consolidated Conversation State

The active working directory is `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software`.

The conversation progressed through:

- NVIDIA driver/NVML mismatch diagnosis and recovery to a CUDA-ready state.
- Raw event-count readiness and training/result analysis.
- Stage1/Stage2 experiment planning and execution.
- XR-Eye-Tracking and PAPER_REF code/paper analysis.
- Experiment pause and documentation passes.
- `runs/` cleanup and categorization policy.
- Git hygiene discussion around generated run artifacts.
- Stage1-first pivot because frame Search accuracy was not strong enough.
- Broad Stage1 architecture/head/loss/optimizer/scheduler/augmentation experiments.
- Post-hoc gate diagnosis and closeout.
- S1-OC integrated Search candidate branch implementation and smoke validation.
- Current request: document all progress, plans, results, conversation context, and create HANDOVER.

Current durable decision:

- Keep the active Stage1 frame-search baseline unchanged.
- Do not repeat frozen-output/post-hoc gate ablations.
- Run S1-OC A/B for at least `50` epochs as the next meaningful experiment.
- Promote only if P10 improves over `28.27717937613433` while center remains `<= 17.257583906065744`.

New durable docs:

- `docs/resources/current_project_synthesis_2026_06_27.md`
- `docs/HANDOVER_2026_06_27.md`
- `HANDOVER.md`

## User Objective

1. Analyze papers under `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF`.
2. Plan accuracy-improvement experiments including head, loss, LR, optimizer, self-supervised distillation, and teacher training.
3. Push HGTXR software accuracy toward `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial` quality.

## Important Decisions

- Use `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software` as hard working directory.
- Use Codex-only agent policy.
- Use Caveman-style concise reporting.
- Spark sub-agent runtime was attempted but quota-blocked; GPT5.5 fallback is allowed.
- Do not rerun Stage1 immediately; start with Stage2-only ablations using existing Stage1 checkpoint.

## Baseline

- Stage1 checkpoint: `runs/raw_mode1_stage1_event_count_20260610_192838/train/best_search_p10.pt`
- Stage2 checkpoint: `runs/raw_mode1_stage2_event_count_20260610_193719/train/best_track_p10.pt`
- Stage2 best validation P10: `7.2653`
- Stage2 final center error: `43.9917 px`
