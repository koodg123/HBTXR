# FACET Analysis Memory - 2026-06-27

## Scope

Analysis targets:

- Codebase: `/home/user/project/PRJXR-HBTXR/HBTXR/references/codebase/software/FACET`
- Report archive: `/home/user/project/PRJXR-HBTXR/HBTXR/references/report/FACET`

This memory summarizes FACET software architecture, HBTXR integration, reproduction-report state, validation evidence, and known risks for future HBTXR/FACET work.

## Prompt Brief

- Goal: 자세한 FACET codebase/report 분석을 수행하고 project-local memory로 저장한다.
- Evidence sources: FACET Python/YAML files, report Markdown/JSON/shell files, code-analyzer skill scripts, direct static reads, compile/shell syntax checks.
- Assumptions: 현재 경로 `/home/user/project/PRJXR-HBTXR/HBTXR`가 메인 작업 디렉토리이다.
- Unknowns: 실제 long-running training process의 최신 runtime 상태는 이 분석 시점의 saved reports 기준이며 live GPU/tmux 상태는 재검사하지 않았다.
- Constraints: 외부 dataset/checkpoint/GPU 실행은 하지 않고 read-only/static 분석과 문서 저장만 수행했다.
- Expected output: 재사용 가능한 Markdown memory.
- Acceptance criteria: 핵심 경로와 명령 근거 포함, 코드/리포트 양쪽 분석, 완료/미완료 gate 구분, 검증 결과 기록.

## Sub-Agent Note

The provided global instruction asked for sub-agent planning, but the available sub-agent runtime explicitly allows spawning only when the user asks for sub-agents or parallel agent work. Since the user asked for analysis and memory persistence, not sub-agent execution, this analysis was performed directly by the main agent. Task card used by the main agent:

```yaml
task_card:
  task_id: T-001
  sub_agent: "main-codex"
  role: "analyst | evaluator | artifact-manager"
  objective: "Analyze FACET codebase and report archive, then save reusable project memory"
  file_ownership:
    - "docs/memory/FACET-analysis-memory-2026-06-27.md"
  assigned_skill:
    - "code-analyzer"
  inputs:
    - "references/codebase/software/FACET"
    - "references/report/FACET"
  outputs:
    - "docs/memory/FACET-analysis-memory-2026-06-27.md"
  validation:
    - "repo overview scripts"
    - "dependency graph extraction"
    - "compileall syntax check"
    - "shell syntax check"
  dependencies: []
```

## Best-Output Criteria

| Criterion | Status | Evidence |
|---|---|---|
| Completeness | Met for static analysis | Both target directories scanned and key files read. |
| Evidence | Met | File paths, line references, command results recorded below. |
| Executability | Partial | Syntax checks passed; full train/eval not executed due external data/checkpoints/GPU runtime dependency. |
| Consistency | Met | Resolution and dataset contracts aligned with current report correction. |
| Safety | Met | No credentials or hidden reasoning stored. |
| Maintainability | Met | Findings are organized as reusable memory, not scattered chat notes. |

## Inventory

Codebase overview from `repo_overview.py`:

- Files: 147
- Lines: 19,346
- Python: 127 files, 17,700 LOC, 91.5%
- YAML: 19 files, 1,564 LOC, 8.1%
- Markdown: 1 file, 82 LOC
- Largest files:
  - `EvEye/model/DavisEyeEllipse/ElNet/ElNet.py`: 990 lines
  - `EvEye/utils/scripts/check_reproduction_status.py`: 976 lines
  - `EvEye/utils/scripts/build_full_dean_dataset_with_unet.py`: 563 lines
  - `EvEye/utils/visualization/visualization.py`: 549 lines
  - `EvEye/model/DavisEyeEllipse/HBTXR/Loss.py`: 504 lines

Report archive overview from `repo_overview.py`:

- Files: 116
- Lines: 16,358
- Markdown: 69 files, 11,694 LOC, 71.5%
- Shell: 35 files, 3,074 LOC, 18.8%
- JSON: 10 files, 1,113 LOC, 6.8%
- Python: 2 files, 477 LOC, 2.9%
- Largest report: `FACET_full_training_monitor_2026-06-26.md`, 2,138 lines.

## Codebase Architecture

FACET is a Lightning-based event-eye-tracking codebase. The important execution entry points are:

- `tools/train.py`: primary local training CLI. It loads YAML config, builds train/val dataloaders, builds model through `make_model`, applies optimizer config, then calls `Trainer.fit`. Evidence: `tools/train.py:16-88`.
- `tools/validate.py`: validation CLI. It loads only validation dataloader, model, callbacks/logger, and calls `Trainer.validate`. Evidence: `tools/validate.py:13-52`.
- `main.py`: SageMaker launcher, not a portable local CLI. It configures SageMaker estimator and points to `tools/train.py`. Treat as cloud submission script.

Factory pattern:

- `EvEye/dataset/dataset_factory.py` maps dataset type strings to dataset classes and constructs `DataLoader`. Evidence: `dataset_factory.py:23-31`, `dataset_factory.py:49-70`.
- `EvEye/model/model_factory.py` maps model type strings to import paths/classes. Current registered FACET-relevant entries include `EPNet`, `HBTXR`, `ElNet`, and `UNet`. Evidence: `model_factory.py:5-13`, `model_factory.py:16-26`.

Dependency extraction shows the main control flow:

```text
tools.train
  -> load_config
  -> logger_factory
  -> callback_factory
  -> dataset_factory
  -> model_factory

model_factory
  -> EPNet / HBTXR / ElNet / UNet

DavisEyeEllipseDataset
  -> event cache/load helpers
  -> ToFrameStack / CutMaxCount
  -> Albumentations replay
  -> CenterNet-style targets
```

## Dataset Contract

`DavisEyeEllipseDataset` is the central FACET ellipse dataset. It expects:

- `root/<split>/cached_data`
- `root/<split>/cached_ellipse`

Per sample, it:

1. Loads ellipse label and 5000-event segment.
2. Applies event augmentation during training.
3. Converts event stream to a two-channel event frame.
4. Applies image transform and replays it onto the ellipse.
5. Downsamples label targets by `down_ratio = 4`.
6. Produces CenterNet-style targets.

Returned keys:

```text
input, hm, reg_mask, ind, ab, ang, trig, mask, reg, center, close, ellipse
```

Evidence:

- Dataset paths: `DavisEyeEllipseDataset.py:37-40`
- 5000-event load: `DavisEyeEllipseDataset.py:190-194`
- event-to-frame conversion: `DavisEyeEllipseDataset.py:215-225`
- downsample ratio and 64x64 target contract from 256 input: `DavisEyeEllipseDataset.py:255-290`
- returned dict: `DavisEyeEllipseDataset.py:321-337`

Complexity note: `DavisEyeEllipseDataset.__getitem__` is high-risk maintenance code: 148 LOC, cyclomatic complexity 19. It mixes data load, event augmentation, rasterization, invalid-label recovery, target construction, and tensor packaging.

## HBTXR Implementation

Current HBTXR is an independent FACET model under:

```text
EvEye/model/DavisEyeEllipse/HBTXR
```

Architecture:

```text
input event tensor: (B, 2, 256, 256)
  -> DeiT patch embedding, patch_size=4
  -> token grid 64x64
  -> transformer encoder
  -> remove CLS token
  -> reshape patch tokens to (B, embed_dim, 64, 64)
  -> projection conv embed_dim -> 64
  -> HBTXRHead
  -> hm, ab, trig, reg, mask heads
```

Evidence:

- HBTXR constructor and DeiT config: `HBTXR.py:38-85`
- projection and head wiring: `HBTXR.py:86-103`
- forward path: `HBTXR.py:115-118`
- validation metrics and post-process use: `HBTXR.py:161-187`
- DeiT patch map reshape: `Backbone/DeiT.py:174-196`
- head output dict: `Head/HBTXRHead.py:10-44`
- loss composition: `Loss.py:399-504`

Output contract:

```text
hm:   (B, 1, 64, 64)
ab:   (B, 2, 64, 64)
trig: (B, 2, 64, 64)
reg:  (B, 2, 64, 64)
mask: (B, 1, 64, 64)
```

Important resolution decision:

- FACET paper/report `64x64` should be read as final feature/heatmap/metric resolution, not raw input resolution.
- Correct reproduction contract is `(B, 2, 256, 256)` input, `down_ratio = 4`, and `64x64` heads.
- Evidence: `FACET_resolution_contract_correction_2026-06-27.md:7-16`, `:23-27`, `:46-52`.

HBTXR training risk:

- `patch_size=4` creates 4096 tokens, so self-attention is expensive.
- Batch-size probe selected `batch_size=4` as best stable fp32 candidate on GPU1. Batch 6+ OOM; mixed precision did not improve throughput/memory enough. Evidence: `FACET_hbtxr_batch_probe_2026-06-26.md:43-72`.
- Live report snapshot at the time of the archived report showed HBTXR at epoch 0, 158830 / 291315 steps, 54.52%, 5.49 it/s, 0 checkpoints. Evidence: `FACET_full_training_progress_snapshot_2026-06-26.md:5-13`.

## EPNet Relationship

EPNet is the closest FACET paper implementation. It uses:

- MobileNetV3 backbone
- FPN-style feature fusion
- detection head with `hm`, `ab`, `trig`, `reg`, `mask`
- CenterNet-style focal/regression/GWD/mask losses

The report archive states HBTXR was intentionally made independent from EPNet direct imports by copying/renaming the head/loss/predict/metric components into the HBTXR namespace. Evidence:

- `FACET_HBTXR_independent_implementation_2026-06-25.md:5-10`
- implemented file list: `:11-47`
- validation result and forward shape smoke: `:95-174`

The paper-to-code analysis records an important mismatch:

- Paper describes four heads, but current code uses an extra `mask` head and `mask_loss`.
- Paper's fast causal event volume limit `l=25` is only partially traceable in active config/code; current active paths use causal accumulation and clipping but not clearly the paper's `l=25`.
- Evidence: `FACET_code_and_paper_analysis.md:12-17`, `:39-50`, `:252-264`.

## Report Archive Analysis

`references/report/FACET/README.md` is the report index and standing rule: FACET analysis, reproduction records, dataset generation logs, and validation results are stored under this directory. Evidence: `README.md:1-18`.

Chronology and roles:

- `FACET_code_and_paper_analysis.md`: code/paper mapping and quality risks.
- `FACET_reproduction_plan_2026-06-25.md`: reproduction phases.
- `FACET_reproduction_dataset_flow_2026-06-25.md`: dataset split and generated DeanDataset flow.
- `FACET_phase1_subset_smoke_2026-06-25.md`: subset EPNet pipeline smoke.
- `FACET_phase2_unet_dataset_prep_2026-06-25.md`: U-Net relabeling prep.
- `FACET_phase3_full_expansion_prep_2026-06-25.md`: full DeanDataset expansion prep.
- `FACET_HBTXR_DeiT_training_plan_2026-06-25.md`: HBTXR training architecture target.
- `FACET_HBTXR_independent_implementation_2026-06-25.md`: HBTXR standalone implementation record.
- `FACET_full_training_monitor_2026-06-26.md` and progress/status artifacts: long-running EPNet/HBTXR training monitoring.
- `FACET_reproduction_completion_audit_2026-06-26.md/json`: completion gate decision.
- `FACET_resolution_contract_correction_2026-06-27.md`: restores the correct 256 input / 64 output contract.

Current archived status:

- Overall status: `incomplete`
- Passed: 10
- Missing: 8
- Evidence: `FACET_reproduction_status_2026-06-26.md:1-9`

Passed gates:

- Gate 0 preflight
- Phase 1 subset DeanDataset
- Phase 2 U-Net labelled PNG dataset
- Phase 3 full DeanDataset_full_unet
- report artifacts
- U-Net visual samples
- Phase 1 EPNet smoke checkpoint
- Phase 2 U-Net smoke checkpoint
- Phase 2 full U-Net checkpoint
- Phase 4 full EPNet checkpoint

Important passed dataset counts:

- Phase 1 DeanDataset samples: 8,911. Evidence: `FACET_reproduction_status_2026-06-26.md:19-24`
- Phase 2 labelled PNG samples: 9,011. Evidence: `:25-30`
- Phase 3 full DeanDataset_full_unet valid ellipse count: 1,457,820. Evidence: `:31-36`
- U-Net visual samples: 10 records. Evidence: `:48-53`

Missing gates:

1. EPNet max_epochs=70 completion log.
2. Full HBTXR training output/checkpoint.
3. HBTXR max_epochs=70 completion log.
4. EPNet fpn_dw ablation training output.
5. EPNet fpn_dw max_epochs=70 completion log.
6. HBTXR effective-batch-32 training output.
7. HBTXR effective-batch-32 max_epochs=70 completion log.
8. Final full-validation evaluation/comparison/summary artifacts.

Evidence:

- `FACET_reproduction_status_2026-06-26.md:78-159`
- `FACET_missing_gate_summary_2026-06-26.md:18-29`
- Completion audit: `FACET_reproduction_completion_audit_2026-06-26.md:1-14`, `:25-36`, `:68-90`

Completion rule:

- Do not mark FACET reproduction complete until all items are `passed` and final evaluation artifacts are produced from full checkpoint and full validation split.
- Evidence: `FACET_reproduction_status_2026-06-26.md:160-162`.

## Automation And Validation Scripts

The report archive contains runners/watchers for:

- full EPNet/HBTXR checkpoint evaluation
- HBTXR effective-batch-32 evaluation
- EPNet fpn_dw ablation evaluation
- hourly status refresh guard
- long-running training watchdogs
- final artifact validation and pairwise comparison validation

Important validation suite:

- `run_validation_smoke_suite_2026-06-26.sh`
- It checks Python/shell syntax, artifact validation, pairwise input validation, training completion marker, evaluation runner completion gate, hourly refresh guard routing, interval defaults, and completion audit pass/fail behavior.
- Reported suite result: `FACET validation smoke suite passed`.
- Evidence: `FACET_validation_smoke_suite_2026-06-26.md:5-49`, `:51-79`.

## Quality And Risk Register

High / correctness:

- Full reproduction is not complete. Completion audit says `can_mark_goal_complete: False`. Evidence: `FACET_reproduction_completion_audit_2026-06-26.md:1-14`.
- HBTXR computational cost is large due 4096-token DeiT at patch size 4. Full completion can be slow and memory-constrained.

Medium / maintainability:

- `DavisEyeEllipseDataset.__getitem__` is too large and complex. Split into load, event transform, ellipse transform, invalid-label normalization, and target-building helpers before major changes.
- HBTXR constructor has many parameters. Consider dataclass/config object if further variants are added.
- Factories use `pop("type")`, mutating caller-owned config dicts. Evidence: `dataset_factory.py:57-59`, `model_factory.py:22-26`.
- Many configs and scripts contain machine-specific paths (`/home/kjm26`, `/mnt/data2T`). This is acceptable for archived reproduction but risky for portable reuse.

Medium / security and robustness:

- `MemmapCacheStructedEvents.load_memmap()` uses `eval(dtype_str)` for dtype metadata. Use safe dtype parsing before accepting untrusted cache metadata. Evidence from grep: `EvEye/utils/cache/MemmapCacheStructedEvents.py:61`.
- Several older utilities/tests use direct `.cuda()` calls or hardcoded CUDA behavior, especially one-off prediction/groundtruth scripts.

Low / hygiene:

- `tools/train.py` imports `pdb` but does not use it. Evidence: `tools/train.py:1-2`.
- Many notebook/test/demo paths are hardcoded and should not be treated as portable APIs.

## Verification Performed In This Analysis

Commands run:

```bash
python3 /home/user/.codex/skills/code-analyzer/scripts/repo_overview.py references/codebase/software/FACET
python3 /home/user/.codex/skills/code-analyzer/scripts/repo_overview.py references/report/FACET
python3 /home/user/.codex/skills/code-analyzer/scripts/dependency_graph.py references/codebase/software/FACET --lang python
PYTHONPYCACHEPREFIX=/tmp/facet_pycache_analysis python3 -m compileall -q references/codebase/software/FACET
bash -n references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
python3 /home/user/.codex/skills/code-analyzer/scripts/complexity.py references/codebase/software/FACET/EvEye/model/DavisEyeEllipse/HBTXR/HBTXR.py
python3 /home/user/.codex/skills/code-analyzer/scripts/complexity.py references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py
```

Results:

- Codebase/report overview collected successfully.
- Python dependency graph extracted successfully.
- Python syntax compile check passed for `references/codebase/software/FACET`.
- Shell syntax check passed for `run_validation_smoke_suite_2026-06-26.sh`.
- Complexity analysis completed for HBTXR and DavisEyeEllipseDataset.

Not run:

- Full training, validation, final evaluation, or live tmux/GPU checks. These require external datasets, checkpoints, long-running GPU jobs, and environment-specific paths.

## Recommended Next Actions

1. Treat FACET reproduction as incomplete until archived status gates are refreshed and all 8 missing gates pass.
2. Before changing HBTXR, preserve the 256 input / 64 output resolution contract unless intentionally running a separate low-resolution ablation.
3. For HBTXR speed work, profile the DeiT attention path first; batch-size-only tuning already showed limited wall-clock improvement.
4. Before final artifact generation, rerun the validation smoke suite and status checker.
5. If making code portable, prioritize config path parameterization, safe dtype metadata parsing, non-mutating factories, and removal of stale `.cuda()` assumptions.

