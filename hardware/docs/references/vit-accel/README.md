# Documentation Guide

This is the main documentation index for `ViT_Accel`.

If you are not sure where to start, use this order:

1. [Project Structure](./PROJECT-STRUCTURE.md)
2. [Local Setup](./LOCAL-SETUP.md)
3. [Board Runbooks](./BOARD-RUNBOOKS.md)
4. [Deployment and Run Scripts](./DEPLOYMENT-RUN-SCRIPTS.md)

## Start Here

- [Project Structure](./PROJECT-STRUCTURE.md)
  - top-level tree, major directories, and canonical locations
- [Local Setup](./LOCAL-SETUP.md)
  - local environment, installed tools, and runtime expectations
- [Board Runbooks](./BOARD-RUNBOOKS.md)
  - full copy-paste command sequences per board
- [Deployment and Run Scripts](./DEPLOYMENT-RUN-SCRIPTS.md)
  - script layout and the main user-facing wrappers

## Flow Operation

- [VCK190 Docker Baseline](./VCK190-DOCKER-BASELINE.md)
  - VCK190-centric Docker workflow and step usage
- [Vivado-Only Flow](./VIVADO-ONLY-FLOW.md)
  - ZynqMP and Vivado-only reconstruction / bitstream flow overview
- [Function Call Stack](./FUNCTION-CALL-STACK.md)
  - wrapper to Python / Tcl / flow call mapping
- [Pipeline Overview](./PIPELINE-OVERVIEW.md)
  - configuration model and end-to-end generation stages

## Hardware Architecture

- [SRC Case Module Guide](./SRC-CASE-MODULE-GUIDE.md)
  - detailed integrated guide for `workspace/hardware/src/` and `case/`
- [SRC Case Parallelism Tuning](./SRC-CASE-PARALLELISM-TUNING.md)
  - practical `CIP/COP/TP/CHAP` tuning order for fit / latency trade-offs
- [VCK190 vs ZU15EG Parameter Comparison](./VCK190-ZU15EG-PARAMETER-COMPARISON.md)
  - current full-model parameter differences, expected impact, and VCK190 packaging notes
- ~~SRC Case Analysis / Diagrams / Microarchitecture~~ — **HBTXR 트리에서 제거됨 (2026-07-29)**
  - 셋 다 15~17줄짜리 리다이렉트 안내문이었고, 본문이 스스로 그렇게 밝히고 있었습니다.
    내용은 [SRC Case Module Guide](./SRC-CASE-MODULE-GUIDE.md)에 통합돼 있습니다.
  - 원본은 `archive/hardware/docs/references/vit_accel/` 에 삭제 전 이름 그대로 있습니다.

## Validation And Status

- [Multi-Board Validation](./MULTI-BOARD-VALIDATION.md)
  - board fit strategy, tuning choices, and expected impact
- [Project Checklist](./CHECKLIST.md)
  - current implementation and validation progress
- [Progress Snapshot 2026-04-15](./2026-04-15-progress.md)
  - current VCK190 / ZU15EG patch status, validation state, and next commands
- [Merge Status](./MERGE-STATUS.md)
  - integration status and source-tree mapping
- [Project History](./HISTORY.md)
  - merge and validation history
- [Execution Plan](./PLAN.md)
  - forward-looking work plan

## Metrics And Reports

- [Training, Evaluation, and Metrics](./TRAINING-EVALUATION-AND-METRICS.md)
  - what this repository does and does not measure
- [Experiment Report Archive](./experiments/README.md)
  - curated long-lived experiment markdowns
