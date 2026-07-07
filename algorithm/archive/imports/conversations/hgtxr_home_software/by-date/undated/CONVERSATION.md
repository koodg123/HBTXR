# HGTXR-SW Conversation Context

Date: 2026-06-10

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
