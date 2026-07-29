# Documentation Guide

This is the main documentation index for `ViT_Accel`.

If you are not sure where to start, use this order:

1. [Project Structure](./PROJECT_STRUCTURE.md)
2. [Local Setup](./LOCAL_SETUP.md)
3. [Board Runbooks](./BOARD_RUNBOOKS.md)
4. [Deployment and Run Scripts](./DEPLOYMENT_RUN_SCRIPTS.md)

## Start Here

- [Project Structure](./PROJECT_STRUCTURE.md)
  - top-level tree, major directories, and canonical locations
- [Local Setup](./LOCAL_SETUP.md)
  - local environment, installed tools, and runtime expectations
- [Board Runbooks](./BOARD_RUNBOOKS.md)
  - full copy-paste command sequences per board
- [Deployment and Run Scripts](./DEPLOYMENT_RUN_SCRIPTS.md)
  - script layout and the main user-facing wrappers

## Flow Operation

- [VCK190 Docker Baseline](./VCK190_DOCKER_BASELINE.md)
  - VCK190-centric Docker workflow and step usage
- [Vivado-Only Flow](./VIVADO_ONLY_FLOW.md)
  - ZynqMP and Vivado-only reconstruction / bitstream flow overview
- [Function Call Stack](./FUNCTION_CALL_STACK.md)
  - wrapper to Python / Tcl / flow call mapping
- [Pipeline Overview](./PIPELINE_OVERVIEW.md)
  - configuration model and end-to-end generation stages

## Hardware Architecture

- [SRC Case Module Guide](./SRC_CASE_MODULE_GUIDE.md)
  - detailed integrated guide for `workspace/hardware/src/` and `case/`
- [SRC Case Parallelism Tuning](./SRC_CASE_PARALLELISM_TUNING.md)
  - practical `CIP/COP/TP/CHAP` tuning order for fit / latency trade-offs
- [VCK190 vs ZU15EG Parameter Comparison](./VCK190_ZU15EG_PARAMETER_COMPARISON.md)
  - current full-model parameter differences, expected impact, and VCK190 packaging notes
- [SRC Case Analysis](./SRC_CASE_ANALYSIS.md)
  - higher-level index for the source analysis set
- [SRC Case Diagrams](./SRC_CASE_DIAGRAMS.md)
  - diagram-oriented reading path
- [SRC Case Microarchitecture](./SRC_CASE_MICROARCH.md)
  - microarchitecture-oriented reading path

## Validation And Status

- [Multi-Board Validation](./MULTI_BOARD_VALIDATION.md)
  - board fit strategy, tuning choices, and expected impact
- [Project Checklist](./CHECKLIST.md)
  - current implementation and validation progress
- [Progress Snapshot 2026-04-15](./PROGRESS_2026_04_15.md)
  - current VCK190 / ZU15EG patch status, validation state, and next commands
- [Merge Status](./MERGE_STATUS.md)
  - integration status and source-tree mapping
- [Project History](./HISTORY.md)
  - merge and validation history
- [Execution Plan](./PLAN.md)
  - forward-looking work plan

## Metrics And Reports

- [Training, Evaluation, and Metrics](./TRAINING_EVALUATION_AND_METRICS.md)
  - what this repository does and does not measure
- [Experiment Report Archive](./experiments/README.md)
  - curated long-lived experiment markdowns
