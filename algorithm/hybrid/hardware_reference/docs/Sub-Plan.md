# HGTXR-SW Sub-Agent And Worker Plan

Date: 2026-06-10

## 2026-06-27 Documentation And Handover Task Cards

```yaml
task_card:
  task_id: T-001
  sub_agent: gpt5.3-codex-spark
  role: analyst
  objective: Summarize current HGTXR software progress, experiment plans, and experiment results from existing docs only.
  file_ownership: []
  assigned_skill: [autosci-exp-status, agent-hierarchy-runtime-manager]
  inputs:
    - docs/resources/current_stage1_work_progress_2026_06_25.md
    - docs/resources/stage1_frame_search_escalation_plan_2026_06_22.md
    - docs/resources/current_work_summary_and_next_experiments_2026_06_18.md
    - docs/track/PROGRESS.md
    - docs/track/log.md
    - docs/track/CONVERSATION.md
  outputs:
    - progress summary
    - experiment result summary
    - next experiment plan summary
    - handover risks
  validation:
    - source file evidence
  dependencies: []
```

```yaml
task_card:
  task_id: T-002
  sub_agent: codex-gpt5.5
  role: evaluator
  objective: Define what a complete HANDOVER document must contain for current HGTXR software Stage1 work.
  file_ownership: []
  assigned_skill: [agent-hierarchy-runtime-manager]
  inputs:
    - docs/resources/current_stage1_work_progress_2026_06_25.md
    - docs/resources/stage1_frame_search_escalation_plan_2026_06_22.md
    - docs/Master-Plan.md
    - docs/Sub-Plan.md
    - docs/Spec.md
    - docs/track/PROGRESS.md
  outputs:
    - handover outline
    - continuation commands
    - residual risks
    - quality gate checklist
  validation:
    - do not invent unverified execution
    - separate implemented, validated, and executed states
  dependencies: []
```

Main-agent integration outputs:

- `docs/resources/current_project_synthesis_2026_06_27.md`
- `docs/HANDOVER_2026_06_27.md`
- `HANDOVER.md`

## Runtime Status

- Requested Spark sub-agents were attempted.
- `gpt-5.3-codex-spark` quota failed with: `You've hit your usage limit for GPT-5.3-Codex-Spark. Switch to another model now, or try again at Jun 15th, 2026 11:18 PM.`
- Fallback: GPT5.5 explorer agents plus main-agent local integration.

## Task Cards

```yaml
task_card:
  task_id: T-001R
  sub_agent: gpt5.5
  role: research
  objective: Extract paper-backed accuracy-improvement ideas from PAPER_REF.
  file_ownership: ["read-only:/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF"]
  assigned_skill: ["caveman", "computer-vision-expert"]
  inputs: ["tmp_pdf_text/*.txt", "eye_tracking_document_analysis_ko.md"]
  outputs: ["paper-method-metric-experiment table"]
  validation: ["file path + line/snippet evidence", "no fabricated metrics"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-002R
  sub_agent: gpt5.5
  role: analyst
  objective: Extract submission target claims and current code experiment surfaces.
  file_ownership:
    - "read-only:/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial"
    - "read-only:/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software"
  assigned_skill: ["caveman"]
  inputs: ["main.tex", "analysis/report*/analysis*.md", "configs/external", "src/hbtxr"]
  outputs: ["target-claim table", "code-surface table"]
  validation: ["file path + line evidence"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-003
  sub_agent: codex-native
  role: implementer
  objective: Create executable Stage2 ablation configs and scripts.
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_*"
    - "scripts/external/run_accuracy_experiment_matrix.sh"
    - "scripts/external/summarize_training_history.py"
    - "docs/*"
  assigned_skill: ["bash-scripting", "deep-learning-framework-expert"]
  inputs: ["current Stage1 checkpoint", "current Stage2 baseline history"]
  outputs: ["configs", "scripts", "tracking docs"]
  validation: ["bash -n", "config load", "dry-run command"]
  dependencies: ["T-001R evidence", "T-002R evidence"]
```

```yaml
task_card:
  task_id: T-004R
  sub_agent: gpt5.5
  role: analyst
  objective: Analyze XR-Eye-Tracking papers for method, reported metrics, and HGTXR-compatible experiment ideas.
  file_ownership:
    - "read-only:/home/kjm26/project/PRJXR/References/XR-Eye-Tracking/Papers"
  assigned_skill: ["cv-dl-expert", "caveman"]
  inputs: ["21 local PDF files", "pdftotext extracted snippets"]
  outputs: ["paper-by-paper analysis", "priority experiment table"]
  validation: ["local PDF path evidence", "reported claims marked as source-reported"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-005R
  sub_agent: gpt5.5
  role: analyst
  objective: Analyze XR-Eye-Tracking codebases for train/eval surfaces and HGTXR integration points.
  file_ownership:
    - "read-only:/home/kjm26/project/PRJXR/References/XR-Eye-Tracking/Codebase"
  assigned_skill: ["cv-dl-expert", "caveman"]
  inputs: ["README files", "source inventories", "selected model/train/eval files"]
  outputs: ["codebase-by-codebase analysis", "integration risk ranking"]
  validation: ["file path evidence", "read-only execution"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-006
  sub_agent: codex-native
  role: artifact-manager
  objective: Store XR-Eye-Tracking analysis and integrate prioritized experiments into existing HGTXR plans.
  file_ownership:
    - "anlaysis/xr-eye-tracking/**"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Master-Plan.md"
    - "docs/Sub-Plan.md"
    - "docs/track/*"
  assigned_skill: ["agent-hierarchy-runtime-manager", "cv-dl-expert", "caveman"]
  inputs: ["T-004R output", "T-005R output", "current HGTXR leader history"]
  outputs: ["analysis folders", "second-goal experiment integration"]
  validation: ["find/rg inventory checks", "plan consistency checks"]
  dependencies: ["T-004R", "T-005R"]
```

```yaml
task_card:
  task_id: T-007R
  sub_agent: gpt5.5
  role: analyst
  objective: Re-analyze XR-Eye-Tracking codebases at SRC_CASE_MODULE_GUIDE detail level.
  file_ownership:
    - "read-only:/home/kjm26/project/PRJXR/References/XR-Eye-Tracking/Codebase"
  assigned_skill: ["agent-hierarchy-runtime-manager", "cv-dl-expert", "caveman"]
  inputs:
    - "reference hardware guide: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/docs/SRC_CASE_MODULE_GUIDE.md"
    - "source files under XR-Eye-Tracking/Codebase"
  outputs:
    - "module-level role/hierarchy, dataflow, call-stack, code-evidence, experiment-surface, HGTXR-risk summary"
  validation:
    - "concrete source path and line evidence where available"
    - "read-only source analysis"
  dependencies: []
```

```yaml
task_card:
  task_id: T-008R
  sub_agent: gpt5.5
  role: analyst
  objective: Re-analyze XR-Eye-Tracking papers with detailed problem/method/algorithm/hardware/experiment/result/options fields.
  file_ownership:
    - "read-only:/home/kjm26/project/PRJXR/References/XR-Eye-Tracking/Papers"
  assigned_skill: ["cv-dl-expert", "caveman"]
  inputs:
    - "21 local PDF files"
    - "pdftotext extraction under /tmp/xr_eye_tracking_pdf_text_full"
  outputs:
    - "paper-by-paper detailed analysis"
    - "cross-paper HGTXR experiment priorities"
  validation:
    - "reported claims marked as source-reported"
    - "uncertainty explicitly marked where extraction is incomplete"
  dependencies: []
```

```yaml
task_card:
  task_id: T-009
  sub_agent: codex-native
  role: artifact-manager
  objective: Integrate detailed XR-Eye-Tracking refresh into durable analysis and progress artifacts.
  file_ownership:
    - "anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md"
    - "anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md"
    - "anlaysis/xr-eye-tracking/index.md"
    - "docs/Sub-Plan.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "cv-dl-expert", "caveman"]
  inputs: ["T-007R output", "T-008R output"]
  outputs: ["canonical detailed analysis documents", "updated index and tracking records"]
  validation: ["rg field checks", "file existence checks", "line-count sanity checks"]
  dependencies: ["T-007R", "T-008R"]
```

```yaml
task_card:
  task_id: T-010R
  sub_agent: gpt5.5
  role: evaluator
  objective: Audit where detailed XR-Eye-Tracking analyses should be integrated into second-goal planning.
  file_ownership:
    - "read-only:docs/Master-Plan.md"
    - "read-only:docs/Paper-Backed-Experiment-Plan.md"
    - "read-only:docs/track/TODO.md"
    - "read-only:docs/track/PROGRESS.md"
    - "read-only:anlaysis/xr-eye-tracking/**"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["detailed analysis docs", "current second-goal docs"]
  outputs: ["integration edit list", "ordering/risk audit"]
  validation: ["local file references", "read-only review"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-011R
  sub_agent: gpt5.5
  role: analyst
  objective: Identify safe implementation surface for EyeLoRiN-style no-retrain HGTXR refinement.
  file_ownership:
    - "read-only:scripts/external/eval_hbtxr.py"
    - "read-only:scripts/external/infer_hbtxr.py"
    - "read-only:src/hbtxr/loss/metrics.py"
    - "read-only:runs/**"
  assigned_skill: ["cv-dl-expert", "caveman"]
  inputs: ["EyeLoRiN detailed analysis", "current eval/infer outputs"]
  outputs: ["candidate files", "input/output format", "smoke strategy", "risks"]
  validation: ["local file references", "read-only review"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-012
  sub_agent: codex-native
  role: implementer
  objective: Implement and smoke-test XR-02 EyeLoRiN-style M2F refinement evaluation.
  file_ownership:
    - "scripts/external/eval_eyelorin_refinement.py"
    - "tests/test_eyelorin_refinement.py"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Sub-Plan.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["T-010R output", "T-011R output", "AdamW fixed255k best-center checkpoint"]
  outputs: ["M2F refinement evaluator", "unit tests", "smoke result", "updated plan/progress docs"]
  validation:
    - "python3 -m py_compile scripts/external/eval_eyelorin_refinement.py tests/test_eyelorin_refinement.py"
    - "PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_eyelorin_refinement.py"
    - "smoke64 run on AdamW fixed255k best-center checkpoint"
  dependencies: ["T-010R", "T-011R"]
```

```yaml
task_card:
  task_id: T-013R
  sub_agent: gpt5.3-codex-spark
  role: research/evaluator
  objective: Audit PAPER_REF 30-paper coverage against second-goal HGTXR experiment planning.
  file_ownership:
    - "read-only:/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF"
    - "read-only:/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial"
    - "read-only:docs/Paper-Backed-Experiment-Plan.md"
    - "read-only:docs/track/PROGRESS.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "cv-dl-expert", "caveman"]
  inputs: ["PAPER_REF corpus", "submission draft", "current second-goal docs"]
  outputs: ["coverage matrix", "missing experiment axis list"]
  validation: ["local file citations", "read-only review"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-014
  sub_agent: codex-native
  role: artifact-manager/implementer
  objective: Add PAPER_REF-specific experiment map and prepare XR-03 AdamW-leader ellipse-state runner.
  file_ownership:
    - "anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md"
    - "scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Master-Plan.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "cv-dl-expert", "caveman"]
  inputs: ["T-013R output", "AdamW fixed255k best-center checkpoint", "existing ellipsestate config"]
  outputs: ["PAPER_REF detailed experiment map", "XR-03 runner", "updated tracking docs"]
  validation:
    - "bash -n scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh"
    - "test -f runs/raw_mode1_stage2_count255000_adamw_lr8e_6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_112215/train/best_metric_track_center_px.pt"
  dependencies: ["T-013R"]
```

```yaml
task_card:
  task_id: T-015
  sub_agent: codex-native
  role: runtime-manager/evaluator
  objective: Resolve XR-01 AdamW fixed250k/fixed260k count-bracket eval status and choose the next branch.
  file_ownership:
    - "scripts/external/compare_xr01_adamw_bracket.py"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "cv-dl-expert", "caveman"]
  inputs: ["XR-01 tmux logs", "XR-01 eval summaries", "current AdamW fixed255k leader gate"]
  outputs: ["count-bracket promotion decision", "fixed255k LR branch launch", "scanner extension"]
  validation:
    - "scripts/external/compare_xr01_adamw_bracket.py"
    - "python3 -m py_compile scripts/external/compare_xr01_adamw_bracket.py"
    - "tmux/nvidia-smi runtime checks"
  dependencies: ["T-014"]
```

```yaml
task_card:
  task_id: T-016R
  sub_agent: gpt5.3-codex-spark
  role: analyst/evaluator
  objective: Audit whether updated second-goal documents reflect PAPER_REF-derived plan.
  file_ownership:
    - "read-only:docs/Paper-Backed-Experiment-Plan.md"
    - "read-only:anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md"
    - "read-only:docs/track/PROGRESS.md"
    - "read-only:docs/Execution.md"
    - "read-only:docs/Validation.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["updated second-goal docs"]
  outputs: ["read-only audit report"]
  validation: ["sub-agent execution attempted; errored due context-window limit, no output used"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-017
  sub_agent: codex-native
  role: implementer/evaluator
  objective: Create XR-04 failure-bucket diagnostic from eval rows and manifest metadata.
  file_ownership:
    - "scripts/external/summarize_eval_failure_buckets.py"
    - "runs/diagnostics/xr04_failure_buckets_adamw255k_leader_test_20260616.json"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["AdamW fixed255k best-center eval_rows", "manifest1 test split", "stage2 fixed255k config overrides"]
  outputs: ["failure-bucket diagnostic script", "full test split diagnostic JSON", "updated XR-04 plan"]
  validation:
    - "python3 -m py_compile scripts/external/summarize_eval_failure_buckets.py"
    - "64-row smoke with training.num_workers=0"
    - "full 2238-row diagnostic with zero missing predictions"
  dependencies: ["T-015"]
```

```yaml
task_card:
  task_id: T-018
  sub_agent: codex-native + gpt5.3-codex-spark sidecar
  role: runtime-manager/evaluator
  objective: Close out XR-01 fixed255k LR bracket, promote the correct leader, and prepare XR-03 geometry run from the promoted checkpoint.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["fixed255k LR bracket logs", "scripts/external/compare_xr01_adamw_bracket.py", "XR-03 runner"]
  outputs: ["LR bracket promotion decision", "new leader checkpoint", "XR-03 launch command"]
  validation:
    - "scripts/external/compare_xr01_adamw_bracket.py"
    - "checkpoint existence check"
    - "GPU availability check"
  dependencies: ["T-015", "T-017"]
```

```yaml
task_card:
  task_id: T-019
  sub_agent: codex-native + gpt5.3-codex-spark sidecar
  role: runtime-manager/evaluator
  objective: Close out XR-03 ellipse-state probes, promote the correct leader, and queue stronger geometry-weight refinement.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["XR-03 eval summaries", "XR-03 train logs", "current second-goal promotion gate"]
  outputs: ["XR-03 promotion decision", "new leader checkpoint", "XR-03A/B launch commands"]
  validation:
    - "find runs -path '*eval_fixed255k_xr03_ellipsestate*20260616*/eval/test/eval_summary.json'"
    - "test metrics use metric_track_center_px, metric_track_p10_pct, metric_track_p5_pct"
    - "GPU availability check before launch"
  dependencies: ["T-018"]
```

```yaml
task_card:
  task_id: T-020
  sub_agent: codex-native
  role: runtime-manager/evaluator
  objective: Close out XR-03A/B stronger geometry sweep, promote XR-03A as the current second-goal leader, and queue the next bounded geometry refinement.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["XR-03A/B eval summaries", "XR-03A/B train logs", "current XR-03 promotion gate"]
  outputs: ["XR-03A promotion decision", "new leader checkpoint", "XR-03C/D bounded geometry commands"]
  validation:
    - "XR-03A test center 20.453865163666862, P10 25.86862314088004, P5 8.310374430247716"
    - "checkpoint existence check for XR-03A best_metric_track_center_px.pt"
    - "bash -n scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh"
  dependencies: ["T-019"]
```

```yaml
task_card:
  task_id: T-021
  sub_agent: codex-native + attempted gpt5.3-codex-spark sidecar
  role: runtime-manager/evaluator
  objective: Launch XR-03C/D bounded geometry refinement from XR-03A best-center and preserve monitoring/provenance.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["XR-03A leader checkpoint", "XR-03 runner", "GPU availability", "XR-03A/B closeout metrics"]
  outputs: ["XR-03C/D tmux sessions", "launch logs", "updated progress and validation docs"]
  validation:
    - "raw event-count contract pass in both launch logs"
    - "resolved_device cuda:0 for XR-03C and cuda:1 for XR-03D"
    - "epoch 1/12 entry in both launch logs"
  dependencies: ["T-020"]
```

```yaml
task_card:
  task_id: T-022
  sub_agent: codex-native
  role: implementer/evaluator
  objective: Implement and validate XR-04 low-similarity-aware track-loss weighting candidate.
  file_ownership:
    - "src/hbtxr/loss/bundles/track.py"
    - "src/hbtxr/loss/stage2.py"
    - "tests/test_track_center_l2_loss.py"
    - "configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_lowsim_finetune_fullwidth.yaml"
    - "scripts/external/run_xr04_lowsim_ellipsestate_probe.sh"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["XR-04 failure bucket artifact", "similarity_target batch tensor", "current XR-03A/XR-03D leader checkpoints"]
  outputs: ["disabled-by-default loss weighting", "validated config", "bash runner", "active experiment command template"]
  validation:
    - "PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py"
    - "python3 -m py_compile src/hbtxr/loss/bundles/track.py src/hbtxr/loss/stage2.py"
    - "bash -n scripts/external/run_xr04_lowsim_ellipsestate_probe.sh"
    - "raw event-count contract for lowsim config"
  dependencies: ["T-021"]
```

```yaml
task_card:
  task_id: T-023
  sub_agent: codex-native
  role: evaluator/runtime-manager
  objective: Close out XR-03C/D, promote the correct gate, and launch XR-04 low-similarity probes on free GPUs.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["XR-03C/D train logs", "XR-03C/D eval summaries", "XR-04 runner", "GPU availability"]
  outputs: ["updated center/P10/P5 gate", "XR-04 GPU0/GPU1 tmux sessions", "updated tracking docs"]
  validation:
    - "all four XR-03C/D eval_summary.json files exist"
    - "XR-03D best-P10 center 20.433587510245186 is recorded as center-first leader"
    - "XR-04 GPU0/GPU1 logs show raw event-count contract pass and epoch 1/12 entry"
  dependencies: ["T-021", "T-022"]
```

```yaml
task_card:
  task_id: T-024
  sub_agent: "codex-native + gpt5.5 sidecar"
  role: implementer/evaluator/runtime-manager
  objective: Close out XR-04, implement direct track-state auxiliary head, and launch XR-05A probes.
  file_ownership:
    - "src/hbtxr/models/heads.py"
    - "src/hbtxr/models/tracker/head_factory.py"
    - "src/hbtxr/models/tracker/track_branch.py"
    - "src/hbtxr/models/hybrid_tracker.py"
    - "src/hbtxr/training/model_factory.py"
    - "src/hbtxr/loss/bundles/__init__.py"
    - "src/hbtxr/loss/bundles/track.py"
    - "src/hbtxr/loss/stage2.py"
    - "tests/test_track_center_l2_loss.py"
    - "configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_finetune_fullwidth.yaml"
    - "scripts/external/run_xr05a_trackstateaux_probe.sh"
    - "docs/Master-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs: ["XR-04 eval summaries", "XR-03D center leader checkpoint", "gpt5.5 sidecar next-experiment recommendation"]
  outputs: ["XR-04 no-promotion decision", "training-only track/state_aux implementation", "XR-05A GPU0/GPU1 tmux sessions", "updated tracking docs"]
  validation:
    - "all four XR-04 eval_summary.json files exist"
    - "XR-04 no-promotion compared to center 20.433587510245186 and balanced P10/P5 25.86862314088004/8.310374430247716"
    - "bash -n scripts/external/run_xr05a_trackstateaux_probe.sh"
    - "python3 -m py_compile on modified model/loss files"
    - "PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py"
    - "raw event-count readiness for XR-05A config"
    - "model build smoke creates TrackStateAuxHead"
    - "XR-05A GPU0/GPU1 logs show raw event-count contract pass and epoch 1/12 entry"
  dependencies: ["T-023"]
```

```yaml
task_card:
  task_id: T-025
  sub_agent: "codex-native + gpt5.5 evaluator"
  role: evaluator/runtime-manager/artifact-manager
  objective: Close out XR-05A, promote new leaders, and launch the next direct-aux refinements.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-05A eval summaries"
    - "gpt5.5 read-only evaluator report"
    - "GPU0/GPU1 availability"
  outputs:
    - "XR-05A promotion decision"
    - "updated center/P10/P5 gates"
    - "XR-05B GPU1 tmux session"
    - "XR-05C GPU0 tmux session"
    - "updated tracking docs"
  validation:
    - "all four XR-05A eval_summary.json files exist"
    - "XR-05A GPU0 LR 6e-6 best-P10 center 20.316973662376405 is recorded as center-first leader"
    - "XR-05A GPU1 LR 3e-6 best-P10 P10 26.57610618046352 is recorded as P10 leader"
    - "XR-05A GPU1 LR 3e-6 best-center P5 8.69557854788644 is recorded as P5/balanced secondary"
    - "XR-05B log shows raw event-count contract pass and epoch 1/12 entry"
    - "XR-05C log shows raw event-count contract pass and epoch 1/12 entry"
  dependencies: ["T-024"]
```

```yaml
task_card:
  task_id: T-026
  sub_agent: "codex-native"
  role: evaluator/runtime-manager/artifact-manager
  objective: Close out XR-05B/C, promote the updated second-goal center leader, and launch XR-05D/E refinements.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-05B eval summaries"
    - "XR-05C eval summaries"
    - "XR-05A and XR-05B leader checkpoints"
    - "GPU0/GPU1 availability"
  outputs:
    - "XR-05B promotion decision"
    - "XR-05C no-promotion decision"
    - "updated second-goal center/P10/P5 gates"
    - "XR-05D GPU1 tmux session"
    - "XR-05E GPU0 tmux session"
    - "updated tracking docs"
  validation:
    - "XR-05B best-P10 test center 20.29270099571773 is recorded as the center-first leader"
    - "XR-05A GPU1 LR 3e-6 best-P10 P10 26.57610618046352 remains the P10 gate"
    - "XR-05A GPU1 LR 3e-6 best-center P5 8.69557854788644 remains the P5/balanced gate"
    - "XR-05C best-center and best-P10 are recorded as no-promotion"
    - "XR-05B manual eval recovery records the HBTXR_DISABLE_CUDNN=1 requirement"
    - "XR-05D log shows raw event-count contract pass, XR-05B best-P10 init load, cuda:1 resolution, and epoch 1/12 entry"
    - "XR-05E log shows raw event-count contract pass, XR-05A best-center init load, cuda:0 resolution, and epoch 1/12 entry"
  dependencies: ["T-025"]
```

```yaml
task_card:
  task_id: T-027
  sub_agent: "gpt5.3-codex-spark attempted, gpt5.5 completed"
  role: evaluator/runtime-manager/artifact-manager
  objective: Close out XR-05D/E, update second-goal gates, and prepare bounded fallback experiments.
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_finetune_fullwidth.yaml"
    - "scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh"
    - "docs/Master-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-05D eval summaries"
    - "XR-05E eval summaries"
    - "gpt5.5 sidecar recommendation"
  outputs:
    - "XR-05D no-promotion decision"
    - "XR-05E P10 promotion decision"
    - "XR-06 weak-distill fallback config and runner"
    - "updated gates and tracking docs"
  validation:
    - "XR-05E best-center P10 26.653912333079745 recorded as new P10 gate"
    - "XR-05B best-P10 center 20.29270099571773 remains center gate"
    - "XR-05A best-center P5 8.69557854788644 remains P5/balanced gate"
    - "bash -n scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh"
    - "PYTHONPATH=src config-load smoke confirms XR-06 track_state_aux and distillation settings"
  dependencies: ["T-026"]
```

```yaml
task_card:
  task_id: T-028
  sub_agent: "codex-native"
  role: implementer/runtime-manager
  objective: Launch tiny center-hinge recovery probes from the XR-05B center leader while XR-02 dense trajectories remain unavailable.
  file_ownership:
    - "scripts/external/run_xr07_trackstateaux_centerhinge_probe.sh"
    - "runs/_logs/xr07a_trackstateaux_linearhinge_m10_w0p01_lr1e-6_gpu1_20260616.log"
    - "runs/_logs/xr07b_trackstateaux_squaredhinge_m10_w0p001_lr1e-6_gpu0_20260616.log"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-05B best-P10 checkpoint"
    - "XR-05D/E closeout gates"
    - "center-hinge loss implementation"
  outputs:
    - "XR-07 flexible hinge runner"
    - "XR-07A GPU1 linear-hinge tmux run"
    - "XR-07B GPU0 squared-hinge tmux run"
  validation:
    - "bash -n scripts/external/run_xr07_trackstateaux_centerhinge_probe.sh"
    - "XR-07A log shows raw event-count contract pass, XR-05B checkpoint load, cuda:1, and epoch 1/12 entry"
    - "XR-07B log shows raw event-count contract pass, XR-05B checkpoint load, cuda:0, and epoch 1/12 entry"
  dependencies: ["T-027"]
```

```yaml
task_card:
  task_id: T-029
  sub_agent: "gpt5.3-codex-spark attempted, codex-native"
  role: evaluator/runtime-manager/artifact-manager
  objective: Close out XR-07A/B, verify XR-02 dense-trajectory blocker, and launch the next bounded fallback.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
    - "runs/_logs/xr06_weakdistill_centerleader_lr1e-6_gpu1_20260616.log"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-07A/B train histories and logs"
    - "data/_internal/manifests/manifest1/test_manifest.jsonl"
    - "XR-06 prepared config and runner"
  outputs:
    - "XR-07 no-promotion decision"
    - "XR-02 sparse-gap blocker evidence"
    - "XR-06 GPU1 weak-distill launch"
    - "updated tracking docs"
  validation:
    - "XR-07A best val center/P10/P5 23.6261/21.8980/7.7943 recorded"
    - "XR-07B best val center/P10/P5 23.6286/21.8980/7.7943 recorded"
    - "test manifest median adjacent gap 4000003us recorded"
    - "XR-06 log shows raw event-count contract pass, XR-05B init/teacher checkpoint, cuda:1, and epoch 1/12 entry"
  dependencies: ["T-028"]
```

```yaml
task_card:
  task_id: T-030
  sub_agent: "gpt5.5 explorer + codex-native"
  role: research-worker/runtime-manager
  objective: Launch a complementary XR-06 weak-distill branch from the XR-05E P10 leader while XR-06A runs on GPU1.
  file_ownership:
    - "runs/_logs/xr06b_weakdistill_p10leader_lr1e-6_gpu0_20260616.log"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["cv-dl-expert", "agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-05E best-center P10 leader checkpoint"
    - "XR-06 weak-distill runner/config"
    - "Nash GPT5.5 explorer recommendation"
  outputs:
    - "XR-06B GPU0 weak-distill launch"
    - "updated tracking docs"
  validation:
    - "XR-06B log shows raw event-count contract pass"
    - "XR-06B log shows XR-05E best-center checkpoint used as init and teacher"
    - "XR-06B log shows resolved_device=cuda:0 and epoch 1/12 entry"
  dependencies: ["T-029"]
```

```yaml
task_card:
  task_id: T-031
  sub_agent: "gpt5.3-codex-spark attempted, codex-native"
  role: evaluator/runtime-manager/artifact-manager
  objective: Close out XR-06A/XR-06B and launch XR-05F no-distill control to isolate weak-distillation value.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
    - "runs/_logs/xr05f_nodistill_xr05e_lightaux_lr1e-6_gpu1_20260616.log"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-06A eval summaries"
    - "XR-06B eval summaries"
    - "XR-05E best-center P10 leader checkpoint"
    - "XR-05A direct track-state-aux runner"
  outputs:
    - "XR-06A center promotion decision"
    - "updated center/P10/P5 gates"
    - "XR-05F GPU1 no-distill control launch"
    - "updated tracking docs"
  validation:
    - "XR-06A best-center test center 20.283847980839866 recorded as new center gate"
    - "XR-06A best-P10 no-promotion recorded"
    - "XR-06B best-center and best-P10 no-promotion recorded"
    - "XR-05F log shows raw event-count contract pass, XR-05E checkpoint load, cuda:1, and epoch 1/12 entry"
  dependencies: ["T-030"]
```

```yaml
task_card:
  task_id: T-032
  sub_agent: "gpt5.3-codex-spark attempted, codex-native"
  role: evaluator/runtime-manager/artifact-manager
  objective: Close out XR-05F and launch XR-06C/XR-05G ultra-low-LR polish from the XR-06A center leader.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
    - "runs/_logs/xr06c_weakdistill_xr06a_lr5e-7_gpu0_20260616.log"
    - "runs/_logs/xr05g_nodistill_xr06a_lr5e-7_gpu1_20260616.log"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-05F eval summaries"
    - "XR-06A best-center checkpoint"
    - "XR-06 weak-distill runner"
    - "XR-05A no-distill track-state-aux runner"
  outputs:
    - "XR-05F no-promotion decision"
    - "XR-06C GPU0 weak-distill launch"
    - "XR-05G GPU1 no-distill launch"
    - "updated tracking docs"
  validation:
    - "XR-05F best-center and best-P10 no-promotion recorded"
    - "XR-06C log shows raw event-count contract pass, XR-06A init/teacher checkpoint, cuda:0, and epoch 1/12 entry"
    - "XR-05G log shows raw event-count contract pass, XR-06A init checkpoint, cuda:1, and epoch 1/12 entry"
  dependencies: ["T-031"]
```

```yaml
task_card:
  task_id: T-033
  sub_agent: "gpt5.5"
  role: evaluator/runtime-manager
  objective: Recommend the next bounded experiments after XR-06C/XR-05G if neither promotes.
  file_ownership:
    - "read-only:docs/Paper-Backed-Experiment-Plan.md"
    - "read-only:docs/track/PROGRESS.md"
    - "read-only:runs/eval_*"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-06A center leader"
    - "XR-05E P10 leader"
    - "XR-05A P5 leader"
    - "XR-06B and XR-05F no-promotion evidence"
  outputs:
    - "XR-06D center-preserve micro-polish proposal"
    - "XR-05H P10-preserve weak-distill proposal"
    - "XR-05I P5-preserve low-LR proposal"
  validation:
    - "Each proposal stays within existing track-state-aux or weak-distill runner surfaces"
    - "Each proposal has a single gate and checkpoint source"
    - "Stop-after-first-promotion policy recorded"
  dependencies: ["T-032", "XR-06C/XR-05G closeout"]
```

```yaml
task_card:
  task_id: T-034
  sub_agent: "codex-native"
  role: runtime-manager/evaluator/artifact-manager
  objective: Close out XR-06C/XR-05G and XR-06D/XR-06E promoted-leader follow-up.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
    - "runs/_logs/xr06d_nodistill_xr06c_center_lr2p5e-7_gpu0_20260616.log"
    - "runs/_logs/xr06e_weakdistill_xr06c_p10_lr2p5e-7_gpu1_20260616.log"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-05G eval summaries"
    - "XR-06C eval summaries"
    - "XR-06C best-center checkpoint"
    - "XR-06C best-P10 checkpoint"
  outputs:
    - "XR-05G no-promotion decision"
    - "XR-06C center and P10 promotion decision"
    - "XR-06D GPU0 launch"
    - "XR-06E GPU1 launch"
    - "XR-06D no-promotion decision"
    - "XR-06E no-promotion decision"
    - "updated tracking docs"
  validation:
    - "XR-05G best-center and best-P10 summaries parsed"
    - "XR-06C best-center and best-P10 summaries parsed"
    - "XR-06D and XR-06E logs show raw event-count contract pass and epoch 1/12 entry"
    - "XR-06D best-center and best-P10 summaries parsed"
    - "XR-06E best-center and best-P10 summaries parsed"
  dependencies: ["T-033"]
```

## Team Blueprint

```yaml
task_card:
  task_id: T-035
  sub_agent: "gpt5.3-codex-spark sidecar + codex-native implementer"
  role: evaluator/implementer/artifact-manager
  objective: Add and smoke-test XR-08 checkpoint interpolation between XR-06C center and P10 leaders.
  file_ownership:
    - "scripts/external/interpolate_hbtxr_checkpoints.py"
    - "scripts/external/run_xr08_checkpoint_interp_eval.sh"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-06C best-center checkpoint"
    - "XR-06C best-P10 checkpoint"
    - "scripts/external/eval_hbtxr.py"
    - "src/hbtxr/training/checkpoints.py"
  outputs:
    - "checkpoint interpolation utility"
    - "XR-08 eval runner"
    - "alpha 0.50 smoke result"
    - "updated tracking docs"
  validation:
    - "python compile passes"
    - "bash syntax passes"
    - "alpha 0.50 eval summary parsed"
    - "git diff --check passes"
  dependencies: ["T-034"]
```

```yaml
task_card:
  task_id: T-036
  sub_agent: "gpt5.3-codex-spark sidecar + codex-native evaluator"
  role: evaluator/artifact-manager
  objective: Close out XR-05I/XR-06F P5-preserve fallback and select next second-goal experiment priority.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-05I best-center and best-P10 eval summaries"
    - "XR-06F best-center and best-P10 eval summaries"
    - "XR-04 failure-bucket diagnostic"
    - "XR-02 dense trajectory status"
  outputs:
    - "XR-05I no-promotion decision"
    - "XR-06F no-promotion decision"
    - "unchanged active gates"
    - "next-priority recommendation: XR-04B targeted branch or XR-02 dense trajectories"
  validation:
    - "all four eval summaries parsed"
    - "gate comparison passes"
    - "tracking docs updated"
  dependencies: ["T-035"]
```

```yaml
task_card:
  task_id: T-037
  sub_agent: "gpt5.3-codex-spark sidecar + codex-native evaluator"
  role: evaluator/artifact-manager
  objective: Close out XR-04B/XR-04C focused low-similarity branches and select the next second-goal failure-bucket action.
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-04B best-center and best-P10 eval summaries"
    - "XR-04C best-center and best-P10 eval summaries"
    - "XR-04 failure-bucket diagnostic"
    - "XR-02 dense trajectory status"
  outputs:
    - "XR-04B no-promotion decision"
    - "XR-04C no-promotion decision"
    - "unchanged active gates"
    - "next-priority recommendation: manifest/sampler-level failure-bucket targeting or dense-trajectory XR-02"
  validation:
    - "all four eval summaries parsed"
    - "gate comparison passes"
    - "tracking docs updated"
  dependencies: ["T-036"]
```

```yaml
task_card:
  task_id: T-038
  sub_agent: "gpt5.3-codex-spark explorer + codex-native implementer"
  role: research/implementer/artifact-manager
  objective: Implement, launch, and close out manifest-subset failure-bucket probes after XR-04B/XR-04C no-promotion.
  file_ownership:
    - "scripts/external/build_failure_bucket_manifest.py"
    - "scripts/external/run_xr04d_failbucket_subset_probe.sh"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-04 failure-bucket diagnostic"
    - "manifest1 train/val jsonl"
    - "XR-03D best-P10 checkpoint"
  outputs:
    - "failure-bucket manifest builder"
    - "XR-04D/E subset runner"
    - "parallel GPU0/GPU1 launch"
    - "XR-04D/E no-promotion decision"
    - "updated tracking docs"
  validation:
    - "py_compile passes"
    - "bash syntax passes"
    - "subset counts recorded"
    - "startup logs show subset train/val sample counts and intended devices"
    - "four full-test eval summaries parsed"
    - "gate comparison passes"
  dependencies: ["T-037"]
```

```yaml
task_card:
  task_id: T-039
  sub_agent: "gpt5.3-codex-spark explorer + codex-native implementer/evaluator"
  role: implementer/evaluator/artifact-manager
  objective: Add full-manifest weighted sampler support and launch XR-09 weighted-sampler track-state-aux probes after hard-subset failure.
  file_ownership:
    - "src/hbtxr/data/loader.py"
    - "tests/test_weighted_sampler.py"
    - "configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_weightedsampler_fullwidth.yaml"
    - "scripts/external/run_xr09_weightedsampler_trackstateaux_probe.sh"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-04D/E hard-subset no-promotion evidence"
    - "manifest1 train rows"
    - "XR-06C center and P10 leaders"
  outputs:
    - "opt-in training.sampler weighted_failure_bucket implementation"
    - "weighted sampler unit tests"
    - "XR-09 runner/config"
    - "parallel GPU0/GPU1 launch or documented blocker"
  validation:
    - "py_compile passes"
    - "bash syntax passes"
    - "pytest weighted sampler passes"
    - "dataloader smoke with canonical root passes"
    - "weight distribution sanity recorded"
    - "startup logs show raw event-count contract and intended CUDA devices"
  dependencies: ["T-038"]
```

```yaml
task_card:
  task_id: T-040
  sub_agent: "gpt5.3-codex-spark explorer + codex-native implementer/evaluator"
  role: implementer/evaluator/artifact-manager
  objective: Add full-manifest loss-side failure-bucket weighting and launch XR-10 track-state-aux probes after XR-09 sampler-only no-promotion.
  file_ownership:
    - "src/hbtxr/data/components.py"
    - "src/hbtxr/loss/stage_common.py"
    - "src/hbtxr/loss/stage2.py"
    - "tests/test_loss_sample_weighting.py"
    - "configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_lossweight_fullwidth.yaml"
    - "scripts/external/run_xr10_lossweight_trackstateaux_probe.sh"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill: ["agent-hierarchy-runtime-manager", "caveman"]
  inputs:
    - "XR-09 sampler-only no-promotion evidence"
    - "manifest1 train rows"
    - "XR-06C center and P10 leaders"
  outputs:
    - "opt-in loss.track_sample_weight implementation"
    - "loss sample-weight unit tests"
    - "XR-10 runner/config"
    - "parallel GPU0/GPU1 launch or documented blocker"
  validation:
    - "py_compile passes"
    - "bash syntax passes"
    - "pytest loss weighting and related track tests pass"
    - "dataloader smoke confirms meta.session_key"
    - "weight distribution sanity recorded"
    - "startup logs show raw event-count contract and intended CUDA devices"
  dependencies: ["T-039"]
```

- Research team: literature and submission claims.
- Training team: Stage2 ablation matrix.
- QA team: command syntax, config load, result checker coverage.
- Artifact manager: docs/track updates and provenance.
- XR reference team: paper/codebase analysis and second-goal experiment integration.
## T-042 XR-11 Track-State SimDR Auxiliary

```yaml
task_card:
  task_id: T-042
  sub_agent: "gpt5.3-codex-spark"
  role: "implementer + evaluator"
  objective: "Add a paper-backed SimDR-style x/y coordinate auxiliary to the Stage2 track branch without changing runtime inference."
  file_ownership:
    - "src/hbtxr/models/heads.py"
    - "src/hbtxr/models/tracker/head_factory.py"
    - "src/hbtxr/models/tracker/track_branch.py"
    - "src/hbtxr/models/hybrid_tracker.py"
    - "src/hbtxr/training/model_factory.py"
    - "src/hbtxr/loss/bundles/track.py"
    - "src/hbtxr/loss/stage2.py"
    - "configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_simdr_fullwidth.yaml"
    - "scripts/external/run_xr11_trackstate_simdr_probe.sh"
    - "tests/test_track_center_l2_loss.py"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "caveman:caveman"
  inputs:
    - "XR-06C leader checkpoints"
    - "TDTracker/AIS SimDR/heatmap analysis"
    - "XR-10 no-promotion closeout"
  outputs:
    - "TrackStateSimDRHead"
    - "loss_track_state_simdr"
    - "XR-11 config and runner"
  validation:
    - "py_compile touched Python files"
    - "bash -n XR-11 runner"
    - "pytest track/loss/sampler regression tests"
    - "config model-build smoke with track/state_simdr shape [B,2,64]"
  dependencies:
    - "XR-10 closeout"
```

Status: implemented, locally validated, launched, and closed without promotion.

## T-044 XR-12 Weak-Distill SimDR Planning Sidecar

```yaml
task_card:
  task_id: T-044
  sub_agent: "gpt-5.3-codex-spark"
  role: "research + evaluator"
  objective: "Audit current configs/runners/docs and recommend the next safest P0 after XR-11 no-promotion."
  file_ownership:
    - "read-only:configs/external/**"
    - "read-only:scripts/external/**"
    - "read-only:docs/**"
    - "read-only:src/hbtxr/**"
  assigned_skill:
    - "caveman:caveman"
  inputs:
    - "XR-06C active gates"
    - "XR-11 no-promotion evidence"
    - "weak-distill and SimDR implementation files"
  outputs:
    - "XR-12 weak-distill light-SimDR recommendation"
    - "risk list and exact launch hyperparameters"
  validation:
    - "recommendation cites current files and active gates"
  dependencies:
    - "T-042 closeout"
```

Status: completed. Recommendation matched main integration decision: preserve XR-06C weak-distill init/teacher, add only low-weight SimDR (`0.0001`) first, then compare against active gates.

## T-045 XR-12 Weak-Distill Light-SimDR Execution

```yaml
task_card:
  task_id: T-045
  sub_agent: "codex-native"
  role: "implementer + runtime manager"
  objective: "Create and launch XR-12 weak-distill light-SimDR probes on both GPUs."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_fullwidth.yaml"
    - "scripts/external/run_xr12_weakdistill_simdr_probe.sh"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "caveman:caveman"
  inputs:
    - "XR-06C best-center and best-P10 checkpoints"
    - "XR-11 no-promotion closeout"
  outputs:
    - "XR-12 weak-distill SimDR config"
    - "XR-12 runner"
    - "parallel GPU0/GPU1 launch"
  validation:
    - "bash -n runner"
    - "config/model smoke"
    - "checkpoint existence checks"
    - "startup logs show raw event-count contract, checkpoint load, CUDA device, and epoch 1 entry"
  dependencies:
    - "T-044"
```

Status: completed. XR-12 train/eval closed without center/P10 promotion. Center-init best-P10 produced a P5-only scalar promotion (`8.7917`) but regressed center/P10, so it is not a new leader.

## T-048 XR-13 Head-Only Weak-Distill SimDR

```yaml
task_card:
  task_id: T-048
  sub_agent: "codex-native | gpt5.5 sidecar"
  role: "implementer + evaluator"
  objective: "Prepare a head-only weak-distill SimDR probe to reduce XR-11/XR-12 representation drift."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_headonly_fullwidth.yaml"
    - "scripts/external/run_xr13_headonly_weakdistill_simdr_probe.sh"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "caveman:caveman"
  inputs:
    - "XR-06C best-center and best-P10 checkpoints"
    - "XR-12 no-center/P10-promotion closeout"
  outputs:
    - "XR-13 head-only SimDR config"
    - "XR-13 runner"
    - "config/trainable-filter validation"
  validation:
    - "bash -n runner"
    - "config/model smoke"
    - "trainable-filter includes nonzero head parameters and freezes trunk"
    - "compare against center/P10/P5 gates"
  dependencies:
    - "T-045"
```

Status: completed. XR-13 train/eval closed without active gate promotion.

## T-050 XR-02 Dense Trajectory Preparation

```yaml
task_card:
  task_id: T-050
  sub_agent: "codex-native | gpt5.5 sidecar"
  role: "implementer + evaluator"
  objective: "Prepare dense/continuous prediction trajectories so the existing EyeLoRiN-style M2F refinement can be evaluated on valid time-contiguous sequences."
  file_ownership:
    - "scripts/external/eval_eyelorin_refinement.py"
    - "scripts/external/*trajectory*"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "caveman:caveman"
  inputs:
    - "XR-06C best-center and best-P10 checkpoints"
    - "Existing XR-02 EyeLoRiN-style refinement script"
    - "Sparse-manifest gap blocker evidence"
  outputs:
    - "Dense/continuous prediction trajectory generation path or documented blocker"
    - "XR-02 M2F window rerun on valid trajectories"
    - "Before/after center, P10, P5, and jitter metrics"
  validation:
    - "trajectory adjacent gaps satisfy the configured max-gap gate"
    - "all promoted claims compare against active gates"
    - "no smoothing across multi-second sparse manifest gaps"
  dependencies:
    - "T-048 closeout"
```

Status: completed as availability diagnostic. Current canonical1 labels have no dense segment under `50000us`; metric promotion is blocked until dense labels or a continuous split exists.

## T-051 XR-04 Diagnostics Refresh

```yaml
task_card:
  task_id: T-051
  sub_agent: "codex-native | gpt5.5 sidecar"
  role: "analyst + evaluator"
  objective: "Refresh failure diagnostics over blink/open-eye/low-event/fixation/saccade buckets before launching another temporal/head training branch."
  file_ownership:
    - "scripts/external/summarize_eval_failure_buckets.py"
    - "scripts/external/*diagnostic*"
    - "runs/diagnostics/**"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "caveman:caveman"
  inputs:
    - "XR-06C best-center and best-P10 eval rows"
    - "Current manifest metadata: closed_eye_flag, blink_candidate_score, similarity_target, session_key, event_window"
    - "XR-02 dense trajectory blocker artifact"
  outputs:
    - "diagnostic JSON/Markdown artifact"
    - "targetable failure-mode recommendation"
    - "next bounded training or no-train decision"
  validation:
    - "diagnostic bins sum to evaluated weighted rows"
    - "recommendation cites active gates and observed bucket errors"
  dependencies:
    - "T-050"
```

Status: completed. XR-04 diagnostics refresh confirms low `similarity_target`, `session_201`, and subjects `42/45/39` as the dominant weighted failures. Search/event fallback cannot be evaluated from the active track-only leader rows because those heads are disabled by normalized config.

## T-052 XR-04 Confidence/Relocalization Diagnostic

```yaml
task_card:
  task_id: T-052
  sub_agent: "codex-native | gpt5.3-codex-spark"
  role: "analyst + implementer"
  objective: "Collect track_pred confidence/quality from the active XR-06C leader and test whether it separates low-similarity failures enough to justify EX-Gaze-style gating or a confidence auxiliary loss."
  file_ownership:
    - "scripts/external/infer_hbtxr.py"
    - "scripts/external/*confidence*"
    - "runs/diagnostics/**"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "caveman:caveman"
    - "computer-vision-expert"
  inputs:
    - "XR-06C best-center checkpoint"
    - "XR-06C best-P10 checkpoint"
    - "XR-04 refreshed failure-bucket diagnostics"
    - "EX-Gaze similarity/relocalization reference"
  outputs:
    - "bounded inference rows with track_pred confidence/quality"
    - "confidence/quality bucket summary"
    - "go/no-go recommendation for confidence gate or auxiliary confidence loss"
  validation:
    - "diagnostic run has bounded row count in smoke mode"
    - "full run joins predictions to manifest metadata before any training is launched"
    - "recommendation cites active gates and low-sim bucket separation"
  dependencies:
    - "T-051"
```

Status: completed. Balanced low/high-sim probe joined `256/256` rows for both XR-06C best-center and best-P10 checkpoints. Existing `track_pred` confidence/quality is saturated and does not separate low-sim failures, so no-train confidence gating is rejected.

## T-053 XR-14 Fallback Diagnostics And Relocalization Branch

```yaml
task_card:
  task_id: T-053
  sub_agent: "codex-native | gpt5.3-codex-spark sidecar if quota allows"
  role: "expert + implementer"
  objective: "Close no-train fallback candidates and prepare the next relocalization-capable branch after current confidence logits failed as a no-train gate."
  file_ownership:
    - "configs/external/*relocalize*"
    - "scripts/external/run_xr14*"
    - "scripts/external/eval_similarity_fallback.py"
    - "scripts/external/infer_hbtxr.py"
    - "scripts/external/summarize_track_confidence_buckets.py"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "computer-vision-expert"
    - "deep-learning-framework-expert"
    - "caveman:caveman"
  inputs:
    - "XR-04 diagnostics refresh artifacts"
    - "XR-04 confidence/quality probe artifacts"
    - "EX-Gaze confidence/relocalization evidence"
    - "current active gates"
  outputs:
    - "no-train fallback diagnostic artifact"
    - "aux-state fallback diagnostic artifact"
    - "XR-14A all-head relocalization config/runner"
    - "explicit acceptance gate"
    - "launch/no-launch validation"
  validation:
    - "does not repeat low-sim reweighting only"
    - "does not use saturated confidence logits as a gate without calibration"
    - "preserves XR-06C active gates as comparison baseline"
  dependencies:
    - "T-052"
```

Status: completed for the fallback-gating branch. No-train previous-state/blend fallback is rejected; `track_state_aux` fallback is rejected; XR-14A all-head relocalization completed and did not promote. `search_state` and `event_state` are not usable fallback branches because full-test P10/P5 remain `0.0` with center error around `178px`.

Next execution:

1. Do not start calibrated search/event fallback training from XR-14A.
2. Promote the next 2차 목표 task to XR-15 support-adaptive fixed-count event-window training.
3. Keep XR-06C gates as active promotion criteria.

## T-054 XR-15 Support-Adaptive Event-Window Probe

```yaml
task_card:
  task_id: T-054
  sub_agent: "codex-native + gpt-5.5/gpt-5.3-codex-spark sidecars"
  role: "research + implementer + evaluator"
  objective: "Test whether support-adaptive fixed-count event windows improve low-sim/session/subject failures without changing the proven XR-06C track-only weak-distill/direct-aux contract."
  file_ownership:
    - "docs/XR15_SUPPORT_ADAPTIVE_PLAN.md"
    - "configs/external/*supportadaptive*"
    - "scripts/external/run_xr15*"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-06C best-center and best-P10 checkpoints"
    - "XR-14A closeout showing search/event fallback failure"
    - "EyeTrAES/EV-Eye event slicing and data-density evidence"
  outputs:
    - "XR-15 plan"
    - "support-adaptive Stage2 config"
    - "runner with center/P10 lane tags"
    - "preflight validation"
    - "post-eval full-test and failure-bucket comparison"
  validation:
    - "raw event-count manifest contract still passes"
    - "dataset smoke proves adaptive_count_resolved is active"
    - "train/eval exit 0 before any promotion claim"
    - "compare center/P10/P5 against XR-06C active gates"
  dependencies:
    - "T-053"
```

Status: completed for XR-15A center/P10 closeout. P10-lane best-center promoted the center gate to `20.225680075372967`; P10/P5 did not promote. Best-P5 helper eval remains incomplete because sandbox/logged timeout attempts produced only `hypers/*` and no summary.

## T-058 XR-15B Narrow Support-Adaptive Event-Window Probe

```yaml
task_card:
  task_id: T-058
  sub_agent: "codex-native + gpt-5.3-codex-spark evaluator"
  role: "implementer + evaluator"
  objective: "Run the narrower XR-15B adaptive event-window variant to test whether reduced min/max drift preserves XR-15A center gains while recovering P10/P5."
  file_ownership:
    - "scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh"
    - "docs/XR15_SUPPORT_ADAPTIVE_PLAN.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15A center leader checkpoint"
    - "XR-06C P10 leader checkpoint"
    - "XR-15A diagnostics"
  outputs:
    - "XR-15B train/eval logs"
    - "best-center/best-P10 full-test summaries"
    - "failure-bucket diagnostics"
    - "promotion decision"
  validation:
    - "min/base/max set to 224k/255k/288k"
    - "compare against center <20.225680075372967, P10 >26.74489871433803, P5 >8.69557854788644"
    - "do not treat P5 helper as blocker unless P5 checkpoint eval completes with summary"
  dependencies:
    - "T-054"
```

Status: completed for XR-15B best-center closeout. Center lane best-center promoted the center gate to `20.215672533852715`; P10/P5 did not promote. Best-P10 eval artifacts remain incomplete because the eval runs created only `hypers/*` and no summary.

## T-060 XR-15C Wider Support-Adaptive Event-Window Probe

```yaml
task_card:
  task_id: T-060
  sub_agent: "codex-native + gpt-5.3-codex-spark analyst sidecar"
  role: "research + evaluator"
  objective: "Run the wider XR-15C adaptive event-window variant to test whether larger event support recovers P10/P5 while preserving the new XR-15B center gain."
  file_ownership:
    - "runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_*xr15c*"
    - "runs/eval_fixed255k_xr15_supportadaptive_*xr15c*"
    - "runs/_logs/xr15c_supportadaptive_*"
    - "docs/XR15_SUPPORT_ADAPTIVE_PLAN.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15B center leader checkpoint"
    - "XR-06C P10 leader checkpoint"
    - "launch-time gates center <20.215672533852715, P10 >26.74489871433803, P5 >8.69557854788644"
  outputs:
    - "XR-15C train/eval logs"
    - "best-center/best-P10 full-test summaries"
    - "gate decision"
  validation:
    - "raw event-count contract passes"
    - "train/eval exit 0 before promotion claim"
    - "compare against updated center/P10/P5 gates"
  dependencies:
    - "T-058"
```

Status: completed for XR-15C best-center closeout. Center lane best-center promoted the center gate to `20.19088832650866`; P10/P5 did not promote. The next planned task is XR-15D bounded track-adapter/coordinate-head training.

## T-061 XR-15D Track-Adapter/Coordinate-Head Probe Recommendation

```yaml
task_card:
  task_id: T-061
  sub_agent: "gpt-5.3-codex-spark"
  role: "analyst"
  objective: "Recommend the next second-goal experiment after XR-15C using paper-backed constraints and the current leader gates."
  file_ownership:
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15C best-center closeout"
    - "XR-15C/XR-06C/XR-05A active gates"
    - "prior negative branches XR-04/XR-09/XR-10/XR-11/XR-12/XR-13/XR-14A"
  outputs:
    - "XR-15D next-experiment recommendation"
    - "gate and init checkpoint recommendation"
  validation:
    - "does not repeat exhausted low-sim weighting/sampling/fallback branches"
    - "keeps fixed255k contract unless explicitly testing a different event branch"
  dependencies:
    - "T-060"
```

Status: recommendation accepted. Use XR-15C center-best as the center seed, keep XR-06C best-P10 as the P10-preserve seed, and move to bounded track-adapter/coordinate-head training instead of another support-adaptive range sweep.

## T-063 XR-15D Track-Adapter/Coordinate-Head Launch

```yaml
task_card:
  task_id: T-063
  sub_agent: "codex-native + gpt-5.3-codex-spark evaluator sidecar"
  role: "implementer + evaluator"
  objective: "Create and launch the XR-15D bounded track-adapter/coordinate-head experiment on both GPUs."
  file_ownership:
    - "scripts/external/run_xr15d_trackadapters_probe.sh"
    - "runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackadapters_centerloss_*xr15d*"
    - "runs/eval_fixed255k_xr15d_trackadapters_*"
    - "runs/_logs/xr15d_trackadapters_*"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15C center leader checkpoint"
    - "XR-06C P10 leader checkpoint"
    - "active gates center <20.19088832650866, P10 >26.74489871433803, P5 >8.69557854788644"
  outputs:
    - "XR-15D runner"
    - "XR-15D center/P10 lane train/eval logs"
    - "gate decision after eval"
  validation:
    - "bash syntax passes"
    - "raw event-count contract passes"
    - "model-build smoke passes"
    - "startup logs show intended checkpoint, trainable filter, CUDA device, and epoch entry"
  dependencies:
    - "T-060"
    - "T-061"
```

Status: launched. Center lane is `hgtxr_xr15d_trackadapters_center_gpu0_20260616`; P10 lane is `hgtxr_xr15d_trackadapters_p10_gpu1_20260616`. Both reached raw contract pass, intended checkpoint load, trainable filter report, CUDA device resolution, and epoch `1/16` train steps.

Status update: completed with no promotion. All four XR-15D test eval summaries exist. Continue to XR-15E weak-distill low-LR adapter probe.

## T-064 XR-15E Weak-Distill Track-Adapter Follow-Up

```yaml
task_card:
  task_id: T-064
  sub_agent: "gpt-5.3-codex-spark requested; codex-native fallback used"
  role: "evaluator + implementer"
  objective: "Run a conservative weak-distill, low-LR track-adapter follow-up after XR-15D no-promotion."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackadapters_centerloss_finetune_fullwidth.yaml"
    - "scripts/external/run_xr15e_weakdistill_trackadapters_probe.sh"
    - "runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackadapters_centerloss_*xr15e*"
    - "runs/eval_fixed255k_xr15e_weakdistill_trackadapters_*"
    - "runs/_logs/xr15e_weakdistill_trackadapters_*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15D no-promotion results"
    - "XR-15C center leader checkpoint"
    - "XR-06C P10 leader checkpoint"
  outputs:
    - "XR-15E config and runner"
    - "XR-15E train/eval logs"
    - "gate decision after eval"
  validation:
    - "bash syntax passes"
    - "raw event-count contract passes"
    - "model+teacher smoke passes"
    - "startup logs show intended checkpoint, teacher, CUDA device, and epoch entry"
  dependencies:
    - "T-063"
```

Status: launched. GPT-5.3-Codex-Spark sub-agent request failed due quota, so Codex-native fallback validated and launched XR-15E. Both lanes reached epoch `1/12`.

Status update: completed with no promotion. All four XR-15E test eval summaries exist. Continue to XR-16A weak-distill event-path adapter isolation.

## T-065 XR-16 Candidate Sidecar Review

```yaml
task_card:
  task_id: T-065
  sub_agent: "gpt-5.5"
  role: "evaluator"
  objective: "Recommend the next isolated XR-16 experiment if XR-15E does not promote."
  file_ownership:
    - "read-only docs/scripts/runs"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15D/XR-15E branch evidence"
    - "active gates center <20.19088832650866, P10 >26.74489871433803, P5 >8.69557854788644"
  outputs:
    - "XR-16A event-path adapter recommendation"
    - "fallback and reject-list"
  validation:
    - "single-variable ablation"
    - "no repeat of exhausted branches"
  dependencies:
    - "T-064"
```

Status: completed. The sidecar recommended XR-16A event-only adapter weak-distill as the next isolated ablation, with XR-16B checkpoint interpolation as fallback.

## T-066 XR-16A Weak-Distill Event-Path Adapter Probe

```yaml
task_card:
  task_id: T-066
  sub_agent: "codex-native"
  role: "implementer + evaluator"
  objective: "Create, validate, and launch XR-16A event-path adapter-only weak-distill training on both GPUs."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackeventadapter_centerloss_finetune_fullwidth.yaml"
    - "scripts/external/run_xr16_weakdistill_trackeventadapter_probe.sh"
    - "runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackeventadapter_centerloss_*xr16a*"
    - "runs/eval_fixed255k_xr16_weakdistill_trackeventadapter_*"
    - "runs/_logs/xr16a_weakdistill_trackeventadapter_*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15E no-promotion results"
    - "XR-15C center leader checkpoint"
    - "XR-06C P10 leader checkpoint"
  outputs:
    - "XR-16A config and runner"
    - "XR-16A train/eval logs"
    - "gate decision after eval"
  validation:
    - "bash syntax passes"
    - "raw event-count contract passes"
    - "trainable-filter smoke shows event path plus track head only"
    - "startup logs show intended checkpoint, teacher, CUDA device, and epoch entry"
  dependencies:
    - "T-065"
```

Status: completed with no promotion. XR-16A early-stopped at epoch `8/12`; best result was center-lane best-center `20.281941199302672 / 26.2521265574864 / 8.633928898402623`, below the active gates.

## T-067 XR-16B Checkpoint Interpolation Diagnostic

```yaml
task_card:
  task_id: T-067
  sub_agent: "codex-native"
  role: "implementer + evaluator"
  objective: "Run a no-training checkpoint interpolation diagnostic between the XR-15C center leader and XR-06C P10 leader."
  file_ownership:
    - "scripts/external/run_xr16b_checkpoint_interp_eval.sh"
    - "runs/interpolated_checkpoints/xr16b_xr15c_center_xr06c_p10_alpha0p125.pt"
    - "runs/eval_fixed255k_xr16b_xr15c_center_xr06c_p10_interp_alpha0p125_*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15C center leader checkpoint"
    - "XR-06C P10 leader checkpoint"
  outputs:
    - "XR-16B interpolation runner"
    - "alpha 0.125 eval summary"
    - "gate decision"
  validation:
    - "runner bash syntax passes"
    - "endpoint checkpoints exist"
    - "eval_summary.json exists"
  dependencies:
    - "T-066"
```

Status: completed with no promotion. Alpha `0.125` produced `20.28155174595969 / 26.309524529320854 / 8.619047934668405`.

## T-068 XR-15C Best-P5 Artifact Gap Fill

```yaml
task_card:
  task_id: T-068
  sub_agent: "codex-native"
  role: "evaluator"
  objective: "Evaluate XR-15C best-P5 checkpoints to close the incomplete P5 artifact coverage and update active gates if warranted."
  file_ownership:
    - "runs/eval_fixed255k_xr15_supportadaptive_*xr15c*bestp5*"
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15C center-lane best_track_p5.pt"
    - "XR-15C P10-lane best_track_p5.pt"
    - "active gates before T-068"
  outputs:
    - "two XR-15C best-P5 eval summaries"
    - "updated P5/balanced secondary gate"
  validation:
    - "eval_summary.json exists for center and P10 lanes"
    - "gate comparison recorded"
  dependencies:
    - "T-060"
```

Status: completed. XR-15C P10-lane best-P5 promoted the P5 gate to `8.703231593540737`; active gates are now center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.703231593540737`.

## T-069 XR-17A Same-Branch Center-To-P5 Interpolation

```yaml
task_card:
  task_id: T-069
  sub_agent: "gpt-5.5 sidecar + codex-native"
  role: "research/evaluator + implementer"
  objective: "Run a no-training same-branch interpolation sweep between the XR-15C center leader and XR-15C P5 leader to test whether P5 can improve without another adapter-training branch."
  file_ownership:
    - "scripts/external/run_xr17a_xr15c_center_p5_interp_eval.sh"
    - "runs/interpolated_checkpoints/xr17a_xr15c_center_xr15c_p5_alpha*.pt"
    - "runs/eval_fixed255k_xr17a_xr15c_center_xr15c_p5_interp_alpha*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15C center-lane best-center checkpoint"
    - "XR-15C P10-lane best-P5 checkpoint"
    - "active gates center <20.19088832650866, P10 >26.74489871433803, P5 >8.703231593540737"
  outputs:
    - "XR-17A interpolation runner"
    - "alpha 0.25/0.50/0.625/0.75/0.875 eval summaries"
    - "updated P5 gate decision"
  validation:
    - "runner bash syntax passes"
    - "endpoint checkpoints exist"
    - "eval_summary.json exists for every alpha"
    - "no active XR-17A eval process remains"
  dependencies:
    - "T-068"
```

Status: completed. GPT-5.3-Codex-Spark sidecar was unavailable due quota, so GPT-5.5 sidecar reviewed the interpolation plan. XR-17A alpha `0.75` promoted P5 to `8.732993507385254`; active gates are now center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.732993507385254`.

## T-070 XR-17A Failure-Bucket Comparison

```yaml
task_card:
  task_id: T-070
  sub_agent: "codex-native"
  role: "analyst/evaluator"
  objective: "Compare XR-15C center leader and XR-17A alpha 0.75 failure buckets to decide whether XR-17A should become a training seed or only a bounded P5-direction anchor."
  file_ownership:
    - "runs/diagnostics/xr17a_compare_xr15c_center_failure_buckets_20260616.json"
    - "runs/diagnostics/xr17a_compare_xr17a_alpha0p75_failure_buckets_20260616.json"
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
    - "docs/track/log.md"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15C center eval_rows.json"
    - "XR-17A alpha 0.75 eval_rows.json"
    - "test manifest with fixed255k support-adaptive overrides"
  outputs:
    - "two failure-bucket diagnostic JSON files"
    - "bucket-level interpretation"
    - "next P0 recommendation"
  validation:
    - "diagnostic JSON files exist"
    - "row-weighted center/P10/P5 comparison recorded"
    - "next P0 preserves active center gate"
  dependencies:
    - "T-069"
```

Status: completed. XR-17A alpha `0.75` improves weighted P10/P5 from `26.1625/8.4824` to `26.5202/8.7890`, but worsens weighted center from `20.0095` to `20.0415`. Next P0 should preserve XR-15C center as primary seed/teacher and use XR-17A only as a bounded P5-direction anchor.

## T-071 XR-17B P5-Anchor Support-Adaptive Branch

```yaml
task_card:
  task_id: T-071
  sub_agent: "gpt-5.3-codex-spark attempted; gpt-5.5 read-only evaluator + codex-native implementer"
  role: "evaluator + implementer"
  objective: "Execute a bounded branch that uses XR-15C center as init and XR-17A alpha 0.75 as weak P5-direction teacher."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr2p5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_fullwidth.yaml"
    - "scripts/external/run_xr17b_p5anchor_supportadaptive_probe.sh"
    - "runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_centerinit_p5teacher_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_143310"
    - "runs/eval_fixed255k_xr17b_p5anchor_centerinit_p5teacher_adamw_lr2_5e_7_*"
    - "runs/diagnostics/xr17b_p5anchor_*_failure_buckets_20260616.json"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-15C center best-center checkpoint"
    - "XR-17A alpha 0.75 checkpoint"
    - "XR-17A failure-bucket comparison"
  outputs:
    - "XR-17B config and runner"
    - "best-center/best-P10/best-P5 test eval summaries"
    - "best-center and best-P5 failure-bucket diagnostics"
    - "updated active gates"
  validation:
    - "bash syntax passes"
    - "raw event-count contract passes"
    - "model build smoke passes"
    - "train exit 0"
    - "eval summaries exist"
  dependencies:
    - "T-070"
```

Status: completed. Spark sidecar hit quota; GPT-5.5 read-only evaluator confirmed the branch. XR-17B best-center promoted center to `20.182338142395018`; XR-17B best-P5 promoted P5 to `8.844813244683403`. Active gates are now center `<20.182338142395018`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.

## T-072 XR-18A P10 Interpolation Diagnostic

```yaml
task_card:
  task_id: T-072
  sub_agent: "codex-native"
  role: "evaluator"
  objective: "Run a no-training small-alpha interpolation sweep from XR-17B center leader toward XR-06C P10 leader before spending more GPU time on trainable P10 recovery."
  file_ownership:
    - "scripts/external/run_xr18a_xr17b_center_xr06c_p10_interp_eval.sh"
    - "runs/interpolated_checkpoints/xr18a_xr17b_center_xr06c_p10_alpha*.pt"
    - "runs/eval_fixed255k_xr18a_xr17b_center_xr06c_p10_interp_alpha*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-17B best-center checkpoint"
    - "XR-06C best-P10 checkpoint"
    - "active gates center <20.182338142395018, P10 >26.74489871433803, P5 >8.844813244683403"
  outputs:
    - "XR-18A interpolation runner"
    - "alpha 0.025/0.05/0.075/0.10 eval summaries"
    - "gate decision"
  validation:
    - "runner bash syntax passes"
    - "endpoint checkpoints exist"
    - "eval summaries exist"
  dependencies:
    - "T-071"
```

Status: completed with no promotion. Best P10 was alpha `0.10` at `20.193220179421562 / 26.20748372077942 / 8.629677173069545`, still below all active gates.

## T-073 XR-19A Trainable P10-Recovery Micro Polish

```yaml
task_card:
  task_id: T-073
  sub_agent: "gpt-5.5 read-only evaluator + codex-native implementer"
  role: "evaluator + implementer"
  objective: "Run the minimal trainable P10-recovery branch recommended after XR-18A: XR-17B best-P5 init, XR-06C best-P10 weak teacher, and micro LR 1.25e-7."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_fullwidth.yaml"
    - "scripts/external/run_xr19a_p10recovery_micro_polish.sh"
    - "runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_micro_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_152610"
    - "runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_micro_adamw_lr1_25e_7_*"
    - "runs/diagnostics/xr19a_p10recovery_bestp10_failure_buckets_20260616.json"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-17B best-P5 checkpoint"
    - "XR-06C best-P10 checkpoint"
    - "GPT-5.5 read-only evaluator recommendation"
    - "active gates center <20.182338142395018, P10 >26.74489871433803, P5 >8.844813244683403"
  outputs:
    - "XR-19A config and runner"
    - "best-center/best-P10/best-P5 test eval summaries"
    - "best-P10 failure-bucket diagnostic"
    - "updated active gates"
  validation:
    - "bash syntax passes"
    - "raw event-count contract passes"
    - "train exit 0"
    - "eval summaries exist"
    - "diagnostic JSON exists"
  dependencies:
    - "T-072"
```

Status: completed. XR-19A best-P5 promoted center slightly to `20.181213889803207`, but P10/P5 did not promote. Best-P10 reached `20.20784169435501 / 26.232143613270352 / 8.609694181169782`; weighted failure buckets still show low-similarity rows as the P10 bottleneck.

## T-074 XR-20A/XR-20B P10-Recovery LR Ladder

```yaml
task_card:
  task_id: T-074
  sub_agent: "gpt-5.5 read-only evaluator + codex-native implementer"
  role: "evaluator + implementer"
  objective: "Test whether a stronger LR ladder from XR-17B best-P5 init and XR-06C best-P10 teacher can recover P10 without losing the current center/P5 gates."
  file_ownership:
    - "runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_lr2p5e7_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_155340"
    - "runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_lr5e7_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_155656"
    - "runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_lr2p5e7_adamw_lr2_5e_7_*"
    - "runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_lr5e7_adamw_lr5e_7_*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-17B best-P5 checkpoint"
    - "XR-06C best-P10 checkpoint"
    - "Lovelace GPT-5.5 evaluator recommendation"
    - "active gates center <20.181213889803207, P10 >26.74489871433803, P5 >8.844813244683403"
  outputs:
    - "XR-20A LR 2.5e-7 eval summaries"
    - "XR-20B LR 5e-7 eval summaries"
    - "updated active gates"
    - "LR-ladder closeout decision"
  validation:
    - "train exits 0"
    - "eval summaries exist"
    - "no active train/eval process remains"
    - "GPU idle check passes"
  dependencies:
    - "T-073"
```

Status: completed. XR-20A best-center promoted center to `20.175542894431523`; XR-20B did not promote. P10/P5 remain below the XR-06C/XR-17B gates, so the P5-init/XR-06C-P10-teacher LR-only ladder is closed.

## T-075 XR-21 P10-Margin Mechanism Change

```yaml
task_card:
  task_id: T-075
  sub_agent: "gpt-5.5 read-only evaluator + codex-native implementer"
  role: "evaluator + implementer"
  objective: "Test a P10-specialized decoded-center squared hinge margin loss after the XR-20 LR-only ladder closed."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10margin_fullwidth.yaml"
    - "configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_margin_fullwidth.yaml"
    - "scripts/external/run_xr21_p10margin_supportadaptive_probe.sh"
    - "scripts/external/run_xr21_p10leader_margin_probe.sh"
    - "runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10margin_*"
    - "runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_margin_*"
    - "runs/eval_fixed255k_xr21_*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-20A best-center checkpoint"
    - "XR-06C best-P10 checkpoint"
    - "Erdos GPT-5.5 evaluator recommendation"
    - "active gates center <20.175542894431523, P10 >26.74489871433803, P5 >8.844813244683403"
  outputs:
    - "XR-21 secondary hinge-weight eval summaries"
    - "XR-21 primary P10-leader eval summaries"
    - "updated no-promotion closeout decision"
  validation:
    - "bash syntax passes"
    - "raw event-count contract passes"
    - "targeted loss pytest passes"
    - "train exits 0"
    - "eval summaries exist"
  dependencies:
    - "T-074"
```

Status: completed. Secondary XR-20A-init lanes with hinge weights `0.0015` and `0.003` did not promote; best was `20.187303059441703 / 26.126701450347902 / 8.791666977746146`. Primary XR-06C-init P10-leader lane with hinge weight `0.0005` reached `20.26367484842028 / 26.20110617365156 / 8.696003689084733`. Active gates remain unchanged.

## T-076 XR-22 Tri-Leader Checkpoint Soup

```yaml
task_card:
  task_id: T-076
  sub_agent: "gpt-5.5 read-only explorer + codex-native implementer"
  role: "explorer + implementer"
  objective: "Run a no-train tri-anchor checkpoint soup across center/P10/P5 leaders to test whether their tradeoffs share a useful basin."
  file_ownership:
    - "scripts/external/average_hbtxr_checkpoints.py"
    - "scripts/external/run_xr22_trileader_soup_eval.sh"
    - "runs/interpolated_checkpoints/xr22_trileader_soup_*.pt"
    - "runs/eval_fixed255k_xr22_trileader_soup_*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-20A best-center checkpoint"
    - "XR-06C best-P10 checkpoint"
    - "XR-17B best-P5 checkpoint"
    - "Wegener GPT-5.5 read-only explorer recommendation"
    - "active gates center <20.175542894431523, P10 >26.74489871433803, P5 >8.844813244683403"
  outputs:
    - "weighted N-way checkpoint soup utility"
    - "XR-22 eval runner"
    - "six no-train eval summaries"
    - "updated P5 gate"
  validation:
    - "py_compile passes"
    - "bash syntax passes"
    - "diff whitespace check passes"
    - "eval summaries exist"
    - "no active train/eval process remains"
  dependencies:
    - "T-075"
```

Status: completed. Six soup evals completed. `c34/p33/f33` promoted P5 to `8.869047941480364` with center `20.236038860252926` and P10 `26.4604599407741`. Center/P10 gates remain XR-20A/XR-06C.

## T-077 XR-23 P10-Preserving Optimizer Probe

```yaml
task_card:
  task_id: T-077
  sub_agent: "gpt-5.5 read-only optimizer explorer + codex-native implementer"
  role: "explorer + implementer"
  objective: "Test whether optimizer substitution can preserve or improve the XR-06C P10 leader without hinge replay."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_optprobe_fullwidth.yaml"
    - "scripts/external/run_xr23_p10_optimizer_probe.sh"
    - "runs/raw_mode1_stage2_count255000_adopt_lr2_5e_7_*"
    - "runs/raw_mode1_stage2_count255000_lion_lr1e_7_*"
    - "runs/eval_fixed255k_xr23_p10opt_*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-06C best-P10 checkpoint as init and teacher"
    - "Poincare GPT-5.5 read-only optimizer recommendation"
    - "active gates center <20.175542894431523, P10 >26.74489871433803, P5 >8.869047941480364"
  outputs:
    - "XR-23 opt-probe config"
    - "XR-23 opt-probe runner"
    - "ADOPT and Lion train/eval summaries"
    - "no-promotion closeout decision"
  validation:
    - "bash syntax passes"
    - "raw event-count contract passes"
    - "optimizer build smoke passes"
    - "train exits 0"
    - "eval summaries exist"
  dependencies:
    - "T-076"
```

Status: completed. ADOPT `2.5e-7` and Lion `1e-7` both completed without promoting center, P10, or P5. Best XR-23 test result was ADOPT best-P10 `20.261837770257678 / 26.318027945927213 / 8.681122745786395`; active gates remain XR-20A/XR-06C/XR-22.

## T-078 XR-24 XR-22-Anchor P10-Recovery

```yaml
task_card:
  task_id: T-078
  sub_agent: "gpt-5.5 read-only strategy explorer + codex-native implementer"
  role: "explorer + implementer"
  objective: "Use the XR-22 c34/p33/f33 soup as a P5/P10-friendly init anchor and test bounded AdamW P10 recovery with XR-06C best-P10 as teacher."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_fullwidth.yaml"
    - "scripts/external/run_xr24_xr22_anchor_p10recovery.sh"
    - "runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_*xr24_xr22p10recovery*"
    - "runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_*xr24_xr22p10recovery*"
    - "runs/eval_fixed255k_xr24_xr22p10recovery_*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-22 c34/p33/f33 soup checkpoint"
    - "XR-06C best-P10 checkpoint as teacher"
    - "Carson GPT-5.5 read-only recommendation"
    - "active gates center <20.175542894431523, P10 >26.74489871433803, P5 >8.869047941480364"
  outputs:
    - "XR-24 P10-recovery config"
    - "XR-24 P10-recovery runner"
    - "AdamW LR 1.25e-7 and 2.5e-7 train/eval summaries"
    - "no-promotion closeout decision"
  validation:
    - "bash syntax passes"
    - "config load smoke passes"
    - "raw event-count contract passes"
    - "input checkpoints exist"
    - "train exits 0"
    - "eval summaries exist"
  dependencies:
    - "T-077"
```

Status: completed. AdamW `1.25e-7` and `2.5e-7` both completed without promoting center, P10, or P5. Best XR-24 test P10 was LR `1.25e-7` best-P5 `26.415817070007325`; best XR-24 P5 was LR `2.5e-7` best-P5 `8.791666984558105`; active gates remain XR-20A/XR-06C/XR-22.

## T-079 XR-25 XR-22-Anchor ADOPT-Defaults Validation

```yaml
task_card:
  task_id: T-079
  sub_agent: "codex-native implementer"
  role: "implementer"
  objective: "Validate whether ADOPT's actual beta/eps defaults improve XR-22-anchor P10 recovery after XR-24 AdamW failed."
  file_ownership:
    - "configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_adoptdefaults_fullwidth.yaml"
    - "scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh"
    - "runs/raw_mode1_stage2_count255000_adopt_lr1_25e_7_*xr25_xr22p10recovery*"
    - "runs/raw_mode1_stage2_count255000_adopt_lr2_5e_7_*xr25_xr22p10recovery*"
    - "runs/eval_fixed255k_xr25_xr22p10recovery_*"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-22 c34/p33/f33 soup checkpoint"
    - "XR-06C best-P10 checkpoint as teacher"
    - "active gates center <20.175542894431523, P10 >26.74489871433803, P5 >8.869047941480364"
  outputs:
    - "XR-25 ADOPT-defaults config"
    - "XR-25 ADOPT-defaults runner"
    - "ADOPT LR 1.25e-7 and 2.5e-7 train/eval summaries"
    - "no-promotion closeout decision"
  validation:
    - "bash syntax passes"
    - "config load smoke passes"
    - "optimizer build smoke passes"
    - "raw event-count contract passes"
    - "input checkpoints exist"
    - "train exits 0"
    - "eval summaries exist"
  dependencies:
    - "T-078"
```

Status: completed. ADOPT-defaults `1.25e-7` and `2.5e-7` both completed without promoting center, P10, or P5. Best XR-25 test P10 was `26.46045993396214`; best P5 was `8.86607174192156`, narrowly below the active P5 gate.

## T-080 XR-26 P10-Boundary Aux Head-Only Probe

```yaml
task_card:
  task_id: T-080
  sub_agent: "gpt-5.5 read-only strategy explorer + codex-native implementer"
  role: "explorer + implementer"
  objective: "Add and test a P10-boundary differentiable loss on track/state and track/state_aux from XR-06C best-P10 init/teacher."
  file_ownership:
    - "src/hbtxr/loss/bundles/track.py"
    - "tests/test_track_center_l2_loss.py"
    - "configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10boundary_headonly_fullwidth.yaml"
    - "scripts/external/run_xr26_p10boundary_aux_probe.sh"
  assigned_skill:
    - "ablation-study-designer"
    - "caveman:caveman"
  inputs:
    - "XR-06C best-P10 checkpoint as init and teacher"
    - "Fermat GPT-5.5 read-only strategy recommendation"
    - "active gates center <20.175542894431523, P10 >26.74489871433803, P5 >8.869047941480364"
  outputs:
    - "P10-boundary loss implementation"
    - "unit tests"
    - "XR-26 config and runner"
    - "train/eval summaries if launched"
  validation:
    - "pytest for loss and trainable-filter tests"
    - "bash syntax passes"
    - "raw event-count contract passes"
    - "trainable-filter smoke confirms head-only scope"
  dependencies:
    - "T-079"
```

Status: completed. Spark sub-agent audit attempt failed due usage limit, so main agent performed integration/validation. XR-26 default and light boundary lanes both early-stopped at epoch `6/8` with train/eval exit `0`; neither promoted center, P10, or P5. Best test P10 stayed `26.45790890966143`, and best test P5 stayed `8.541666957310268`.

## T-081 XR-27 Coordinate-Representation Or Teacher-Quality Pivot

```yaml
task_card:
  task_id: T-081
  sub_agent: "gpt-5.5 strategy explorer or codex-native fallback"
  role: "analyst + implementer"
  objective: "Plan the next P0 after scalar P10-boundary polish failed: change coordinate representation or teacher quality while preserving HGTXR paper scope."
  file_ownership:
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "configs/external/*xr27*.yaml"
    - "scripts/external/*xr27*.sh"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
  inputs:
    - "XR-26 no-promotion results"
    - "XR-11/XR-12/XR-13 SimDR drift evidence"
    - "active gates center <20.175542894431523, P10 >26.74489871433803, P5 >8.869047941480364"
  outputs:
    - "bounded XR-27 config/runner"
    - "explicit gate expectations"
    - "validation plan"
  validation:
    - "runner syntax"
    - "config smoke"
    - "raw event-count contract"
    - "trainable-filter smoke"
  dependencies:
    - "T-080"
```

Status: completed. XR-27 implemented a track-center heatmap-state head trained from the XR-06C best-P10 checkpoint with head-only trainable scope. The initial state-distillation plan was audited and corrected because it would compare student heatmap-state outputs against a teacher with random heatmap weights. Clean runs disabled state distillation. XR-27A LR `1e-4` best-P10 promoted all active gates to center `17.274589475563594`, P10 `31.915391901561193`, and P5 `10.289966331209456`.

## T-082 XR-28 Track-Heatmap Consolidation Diagnostics

```yaml
task_card:
  task_id: T-082
  sub_agent: "gpt-5.5 evaluator or codex-native fallback"
  role: "analyst + evaluator"
  objective: "Consolidate the XR-27 promotion before longer training or broader sweeps."
  file_ownership:
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/resources/xr27_trackheatmap_results_2026_06_16.md"
    - "runs/diagnostics/*xr27*"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
  inputs:
    - "XR-27A best-P10 leader"
    - "XR-06C, XR-20A, and XR-22 prior leaders"
    - "active gates center <17.274589475563594, P10 >31.915391901561193, P5 >10.289966331209456"
  outputs:
    - "failure-bucket comparison"
    - "best-center checkpoint selection recommendation"
    - "bounded XR-28 LR/weight refinement plan"
  validation:
    - "diagnostic scripts run on identical test split"
    - "metric table compares center/P10/P5 and failure buckets"
    - "next training commands are bounded to one or two GPU lanes"
  dependencies:
    - "T-081"
```

Status: completed. Failure-bucket diagnostics are complete and recorded in `docs/resources/xr28_trackheatmap_consolidation_diagnostics_2026_06_16.md`. XR-27A strongly improves low/mid similarity buckets, but high-similarity `>0.9` P10 and subject `39` P10/P5 remain regression risks. Runner support for `XR27_BEST_METRIC_NAME` and `XR27_SCHEDULER_METRIC_NAME` was added, and the bounded LR `7e-5` / `1.5e-4` heatmap refinements were executed. LR `1.5e-4` promoted all active gates to `17.08485197339739/32.081208263124736/10.502551344462804`.

## T-083 XR-29 Post-XR-28 Consolidation

```yaml
task_card:
  task_id: T-083
  sub_agent: "gpt-5.5 evaluator or codex-native fallback"
  role: "analyst + experiment planner"
  objective: "Consolidate XR-28 promotion and design the next bounded accuracy-improvement step."
  file_ownership:
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/resources/xr28_heatmap_lr_refinement_results_2026_06_16.md"
    - "runs/diagnostics/*xr28*"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
  inputs:
    - "XR-28 LR 1.5e-4 promoted leader"
    - "XR-28 and XR-27 failure-bucket diagnostics"
    - "active gates center <17.04961508342198, P10 >32.56462665285383, P5 >11.50467722075326"
  outputs:
    - "post-XR-28 failure-bucket comparison"
    - "reproduction or longer-budget command"
    - "bounded LR-neighbor or heatmap-weight experiment plan"
  validation:
    - "diagnostic scripts run on identical test split"
    - "all next commands cite init checkpoint and gate metrics"
    - "no broad optimizer/loss replay without bucket evidence"
  dependencies:
    - "T-082"
```

Status: completed. Post-XR-28 failure-bucket comparison was generated at `runs/diagnostics/xr29_xr28_lr1p5e4_bestcenter_failure_buckets_20260616.json` with `2238` joined rows and `0` missing predictions. Weighted aggregate is `16.9868/32.3454/10.5774`; residual risks remain low similarity, subject `39`, subjects `42/45`, and session-heavy failures. LR-neighbor runs `1.25e-4` on GPU0 and `1.75e-4` on GPU1 completed with train/eval exit `0`. LR `1.25e-4` reached `17.179786903517588/32.04761978558132/10.53443912097386`; LR `1.75e-4` promoted all gates to `17.04961508342198/32.56462665285383/11.50467722075326`.

## T-084 XR-30 Heatmap LR Micro-Bracket

```yaml
task_card:
  task_id: T-084
  sub_agent: "gpt-5.5 evaluator or gpt5.3-codex-spark worker if quota available"
  role: "experiment runner + evaluator"
  objective: "Test bounded LR micro-neighbors around the XR-29 LR 1.75e-4 promoted heatmap-state leader before changing loss, optimizer, or teacher."
  file_ownership:
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/resources/xr29_lr_neighbor_results_2026_06_16.md"
    - "runs/_logs/xr30_*"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
  inputs:
    - "XR-29 LR 1.75e-4 promoted leader"
    - "active gates center <17.04961508342198, P10 >32.56462665285383, P5 >11.50467722075326"
    - "same support-adaptive fixed255k heatmap-state contract"
  outputs:
    - "LR 1.625e-4 and 1.875e-4 train/eval summaries"
    - "leader gate comparison"
    - "decision whether to test 2.0e-4 or change loss/teacher"
  validation:
    - "raw event-count contract passes"
    - "best-P10, best-P5, and best-center eval summaries generated"
    - "no active gate promotion claim without full test eval evidence"
  dependencies:
    - "T-083"
```

Status: completed. LR `1.625e-4` and LR `1.875e-4` both completed with train exit `0` and all six full-test eval summaries. LR `1.625e-4` reached `17.067689692974092/32.420068802152365/10.866496937615532` and did not promote. LR `1.875e-4` reached `17.035534060001375/32.42474567549569/11.266581957680838`, promoting center only. Active gates are now center `<17.035534060001375`, P10 `>32.56462665285383`, P5 `>11.50467722075326`.

## T-085 XR-31 XR-29/XR-30 Checkpoint Interpolation

```yaml
task_card:
  task_id: T-085
  sub_agent: "gpt-5.5 evaluator or gpt5.3-codex-spark worker if quota available"
  role: "experiment runner + evaluator"
  objective: "Evaluate no-train interpolation between the XR-29 unified leader and the XR-30 center leader to recover center without sacrificing P10/P5."
  file_ownership:
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/resources/xr30_lr_micro_bracket_results_2026_06_16.md"
    - "runs/eval_fixed255k_xr31_*"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
  inputs:
    - "XR-29 LR 1.75e-4 unified leader checkpoint"
    - "XR-30 LR 1.875e-4 center leader checkpoint"
    - "active gates center <17.035534060001375, P10 >32.56462665285383, P5 >11.50467722075326"
  outputs:
    - "interpolation eval summaries for small XR-30-direction alphas"
    - "decision whether to train LR 1.8125e-4"
  validation:
    - "all interpolation candidates evaluated on identical test manifest"
    - "no checkpoint promoted unless it passes the relevant active gate"
  dependencies:
    - "T-084"
```

Status: completed. Spark sub-agent execution was attempted but blocked by the GPT-5.3-Codex-Spark usage limit, so the main agent executed the task card. Added `scripts/external/run_xr31_xr29_xr30_heatmap_interp_eval.sh` and evaluated alpha `0.125/0.1875/0.25/0.50/0.75`. Alpha `0.25` promoted P10 to `32.57950758934021` with center `17.04659355367933`, but P5 `11.503826883860997` remained slightly below the XR-29 P5 gate. Active gates are now center `<17.035534060001375`, P10 `>32.57950758934021`, P5 `>11.50467722075326`; best unified checkpoint remains XR-29 LR `1.75e-4`.

## T-086 XR-32/XR-33 Heatmap LR Midpoint Probes

```yaml
task_card:
  task_id: T-086
  sub_agent: "main-agent fallback; gpt5.3-codex-spark unavailable due usage limit"
  role: "experiment runner + evaluator"
  objective: "Train the bounded LR midpoint lanes after XR-31 interpolation promoted P10 but missed P5."
  file_ownership:
    - "scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/resources/xr32_xr33_lr_midpoint_results_2026_06_16.md"
    - "runs/_logs/xr32_*"
    - "runs/_logs/xr33_*"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
  inputs:
    - "XR-29 LR 1.75e-4 unified leader"
    - "XR-30 LR 1.875e-4 center leader"
    - "XR-31 alpha 0.25 P10 leader"
    - "active gates center <17.035534060001375, P10 >32.57950758934021, P5 >11.50467722075326"
  outputs:
    - "XR-32 LR 1.8125e-4 train/eval summaries"
    - "XR-33 LR 1.84375e-4 train/eval summaries"
    - "updated active gate decision"
  validation:
    - "train completes epoch 10/10 with exit 0"
    - "best-P10, best-P5, and best-center eval summaries exist"
    - "no promotion claim without full-test eval evidence"
  dependencies:
    - "T-085"
```

Status: completed. XR-32 LR `1.8125e-4` on GPU0 and XR-33 LR `1.84375e-4` on GPU1 both completed epoch `10/10` with train exit `0`, and all best-P10/best-P5/best-center full-test eval summaries were produced. XR-32 reached `17.041867678506033/32.5463443006788/11.37500034059797` and did not promote. XR-33 reached `17.039179919447218/32.62074908529009/11.362245225906372`, promoting P10 only. Active gates are now center `<17.035534060001375`, P10 `>32.62074908529009`, P5 `>11.50467722075326`; best unified checkpoint remains XR-29 LR `1.75e-4`.

## T-087 XR-34 Heatmap Loss-Ratio Refinement

```yaml
task_card:
  task_id: T-087
  sub_agent: "gpt-5.5 sidecar evaluator + main-agent implementation"
  role: "experiment designer + runner + evaluator"
  objective: "Move beyond close LR-only replay by testing heatmap/offset/center loss-ratio changes anchored to the current center/P5 and P10 leaders."
  file_ownership:
    - "scripts/external/run_xr34_heatmap_lossratio_refinement.sh"
    - "docs/resources/xr34_heatmap_lossratio_plan_2026_06_16.md"
    - "runs/_logs/xr34a_heatmap_lossratio_gpu0_20260616.log"
    - "runs/_logs/xr34b_heatmap_lossratio_gpu1_20260616.log"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
    - "caveman"
  inputs:
    - "XR-29 LR 1.75e-4 best-center checkpoint"
    - "XR-33 LR 1.84375e-4 best-center checkpoint"
    - "active gates center <17.035534060001375, P10 >32.62074908529009, P5 >11.50467722075326"
  outputs:
    - "XR-34A center/P5-preserve lane"
    - "XR-34B P10-sharpen lane"
    - "full-test best-P10/best-P5/best-center eval summaries"
  validation:
    - "raw event-count contract passes"
    - "intended init checkpoint loaded"
    - "resolved CUDA devices are split across GPU0/GPU1"
    - "no promotion claim before eval_summary.json evidence"
  dependencies:
    - "T-086"
```

Status: completed. Added `scripts/external/run_xr34_heatmap_lossratio_refinement.sh` and `docs/resources/xr34_heatmap_lossratio_plan_2026_06_16.md`. XR-34A and XR-34B completed training with exit `0` and produced all best-P10, best-P5, and best-center full-test summaries. XR-34A reached `17.026217068944657/32.430698088237214/10.911139822006225`, promoting center only. XR-34B best-P10 reached `16.53321223940168/33.77168447630746/11.276786088943481`, promoting center and P10. P5 remains below the XR-29 gate, so XR-29 remains the strict P5/unified anchor.

## T-088 XR-35 XR-29/XR-34B Heatmap Interpolation

```yaml
task_card:
  task_id: T-088
  sub_agent: "gpt-5.3-codex-spark attempted; gpt-5.5 sidecar evaluator + main-agent implementation"
  role: "experiment designer + runner + evaluator"
  objective: "Recover XR-29 P5 while retaining part of XR-34B center/P10 gain via no-train checkpoint interpolation."
  file_ownership:
    - "scripts/external/run_xr35_xr29_xr34b_heatmap_interp_eval.sh"
    - "docs/resources/xr35_xr29_xr34b_interpolation_plan_2026_06_16.md"
    - "runs/interpolated_checkpoints/xr35_*"
    - "runs/_logs/xr35_*"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
    - "caveman"
  inputs:
    - "XR-29 LR 1.75e-4 best-center checkpoint"
    - "XR-34B best-P10 checkpoint"
    - "active gates center <16.53321223940168, P10 >33.77168447630746, P5 >11.50467722075326"
  outputs:
    - "small-alpha no-train interpolation eval summaries"
    - "promotion/no-promotion decision"
  validation:
    - "source checkpoints exist"
    - "interpolated checkpoint model keys match"
    - "full-test eval_summary.json exists for each alpha"
    - "no unified promotion claim unless P5 gate is restored"
  dependencies:
    - "T-087"
```

Status: completed. Added `scripts/external/run_xr35_xr29_xr34b_heatmap_interp_eval.sh` and `docs/resources/xr35_xr29_xr34b_interpolation_plan_2026_06_16.md`. Ran alpha `0.03125/0.0625/0.09375/0.125` on GPU1; results were `17.016330581051964/32.47534093856812/11.334609195164271`, `16.983979083810535/32.468963384628296/11.391156809670585`, `16.952843945366997/32.42432054110936/11.420918709891183`, and `16.923010180677686/32.352041605540684/11.255102368763515`. No active gate promoted; next task should be trainable P5-anchor continuation.

## T-089 XR-36 P5-Anchor Continuation

```yaml
task_card:
  task_id: T-089
  sub_agent: "main-agent implementation; sub-agent recommendation integrated from T-XR35-PLAN-REVIEW"
  role: "experiment designer + runner"
  objective: "Prepare a trainable P5-anchor continuation after XR-35 no-train interpolation failed to restore P5."
  file_ownership:
    - "scripts/external/run_xr36_p5_anchor_continuation.sh"
    - "docs/resources/xr36_p5_anchor_continuation_plan_2026_06_16.md"
    - "runs/_logs/xr36*"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
    - "caveman"
  inputs:
    - "XR-34B best-P10 checkpoint"
    - "XR-35 alpha 0.09375 checkpoint"
    - "XR-29 best-center P5 anchor checkpoint"
  outputs:
    - "XR-36A: XR-34B init, LR 5e-5, original heatmap loss ratio"
    - "XR-36B: XR-35 alpha 0.09375 init, LR 8.75e-5, original heatmap loss ratio"
  validation:
    - "bash -n runner passes"
    - "source checkpoints exist"
    - "full train/eval closeout before promotion decision"
  dependencies:
    - "T-088"
```

Status: completed. Added `scripts/external/run_xr36_p5_anchor_continuation.sh` and `docs/resources/xr36_p5_anchor_continuation_plan_2026_06_16.md`. Both lanes launched and completed with train exit `0`, early-stopping at epoch `7/10`. XR-36A best-P10 reached `16.59279990025929/34.39710958344596/11.738095617294311`, promoting P5. XR-36A best-P5 reached `16.576952314376832/34.74064704350063/11.415816688537598`, promoting P10. XR-36B best-P10 reached `16.53305721793856/34.19387831687927/11.502551344462804`, promoting center and P10. Active gates are now center `<16.53305721793856`, P10 `>34.74064704350063`, P5 `>11.738095617294311`.

## T-090 XR-37 XR-36B/XR-36A Interpolation

```yaml
task_card:
  task_id: T-090
  sub_agent: "main-agent implementation"
  role: "experiment designer + runner"
  objective: "Try to combine XR-36B center with XR-36A P5/P10 by no-train checkpoint interpolation."
  file_ownership:
    - "scripts/external/run_xr37_xr36b_xr36a_interp_eval.sh"
    - "docs/resources/xr37_xr36b_xr36a_interpolation_plan_2026_06_16.md"
    - "runs/interpolated_checkpoints/xr37_*"
    - "runs/_logs/xr37_*"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
    - "caveman"
  inputs:
    - "XR-36B best-P10 center leader checkpoint"
    - "XR-36A best-P10 P5/practical leader checkpoint"
  outputs:
    - "small alpha interpolation eval summaries"
    - "tradeoff and gate decision"
  validation:
    - "source checkpoints exist"
    - "bash -n runner passes"
    - "full-test eval_summary.json exists for each alpha"
  dependencies:
    - "T-089"
```

Status: completed. Added `scripts/external/run_xr37_xr36b_xr36a_interp_eval.sh` and `docs/resources/xr37_xr36b_xr36a_interpolation_plan_2026_06_16.md`. Static validation passed before launch. GPU1 interpolation alpha `0.10/0.20/0.35/0.50` completed and produced full-test summaries. Alpha `0.50` promoted center to `16.507612899371555` with P10 `34.33205857958112` and P5 `11.539116007941109`; P10/P5 gates remain XR-36A-owned.

## T-091 Post-XR-37 Center-Preserving P10/P5 Recovery

```yaml
task_card:
  task_id: T-091
  sub_agent: "main-agent or GPT5.5 evaluator before launch"
  role: "experiment designer"
  objective: "Design the next bounded experiment to preserve XR-37 alpha 0.50 center while recovering XR-36A P10/P5."
  file_ownership:
    - "docs/resources/xr38_*"
    - "scripts/external/run_xr38_*"
  assigned_skill:
    - "ablation-study-designer"
    - "computer-vision-expert"
    - "caveman"
  inputs:
    - "XR-37 alpha 0.50 center leader checkpoint"
    - "XR-36A best-P5 P10 leader checkpoint"
    - "XR-36A best-P10 P5 leader checkpoint"
  outputs:
    - "bounded XR-38 runner"
    - "promotion gate statement"
    - "validation plan"
  validation:
    - "source checkpoints exist"
    - "runner passes bash -n"
    - "full-test eval summaries exist after launch"
  dependencies:
    - "T-090"
```

Status: completed. Added `scripts/external/run_xr38_center_preserve_p10p5_recovery.sh` and `docs/resources/xr38_center_preserve_p10p5_recovery_plan_2026_06_17.md`. Static validation passed with `bash -n`, required XR-37 and XR-36A source checkpoints exist. Initial sandboxed launch failed because PyTorch could not initialize CUDA/NVML; reran unsandboxed. XR-38A and XR-38B completed with train exit `0`, both early-stopping at epoch `7/10`. XR-38A best-center/best-P10/best-P5 reached `16.556500560896737/33.96471167291914/11.519983339309693`, `16.57439456837518/33.92984774453299/11.455357497079032`, and `16.513289058208464/34.1815484387534/11.56462620326451`. XR-38B best-P10 reached `16.510086681161606/34.707483761651176/11.223214626312256`, narrowly missing P10. XR-38B best-P5 reached `16.513694180761064/33.93452457700457/11.843962955474854`, promoting P5 only. Active gates are now center `<16.507612899371555`, P10 `>34.74064704350063`, P5 `>11.843962955474854`.

## T-092 XR-39 Mixed-Leader No-Train Soup

```yaml
task_card:
  task_id: T-092
  sub_agent: gpt5.3-codex-spark attempted, gpt5.5 fallback evaluator
  role: runtime-manager/evaluator
  objective: "Evaluate no-train soups over XR-37 center, XR-36A P10, and XR-38B P5 leaders."
  file_ownership:
    - "scripts/external/run_xr39_mixed_leader_soup_eval.sh"
    - "docs/resources/xr39_mixed_leader_soup_plan_2026_06_17.md"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Spec.md"
    - "docs/Validation.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill: ["ablation-study-designer", "computer-vision-expert", "caveman"]
  inputs:
    - "XR-37 alpha 0.50 center checkpoint"
    - "XR-36A best-P5 P10 checkpoint"
    - "XR-38B best-P5 checkpoint"
  outputs:
    - "13 full-test soup eval summaries"
    - "updated active gates"
  validation:
    - "checkpoint existence"
    - "model-key compatibility"
    - "bash -n runner"
    - "full-test eval summaries"
  dependencies: ["T-091"]
```

Status: completed. Spark sidecar was unavailable due quota, and GPT5.5 sidecar returned a read-only audit. Added `scripts/external/run_xr39_mixed_leader_soup_eval.sh` and `docs/resources/xr39_mixed_leader_soup_plan_2026_06_17.md`. All three anchors existed and had matching `138` model keys. Ran 13 soups across GPU0/GPU1 with eval exit `0`. Best center was `c60p25f15` at `16.491779099191938/34.5306130204882/11.50000034059797`. Best P10 was `c25p45f30` at `16.503940873486656/35.02295998845781/11.460459525244577`. Best XR-39 P5 was `c70p20f10` at `11.744473137174333`, below XR-38B. Active gates are now center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.843962955474854`.

## T-093 XR-40 XR39-Leader/P5 No-Train Soup

```yaml
task_card:
  task_id: T-093
  sub_agent: gpt5.5 evaluator
  role: runtime-manager/evaluator
  objective: "Evaluate narrow no-train soups over XR-39 center, XR-39 P10, and XR-38B P5 leaders."
  file_ownership:
    - "scripts/external/run_xr40_xr39leader_p5_soup_eval.sh"
    - "docs/resources/xr40_xr39leader_p5_soup_plan_2026_06_17.md"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Spec.md"
    - "docs/Validation.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill: ["ablation-study-designer", "computer-vision-expert", "caveman"]
  inputs:
    - "XR-39 c60p25f15 center checkpoint"
    - "XR-39 c25p45f30 P10 checkpoint"
    - "XR-38B best-P5 checkpoint"
  outputs:
    - "7 full-test soup eval summaries"
    - "no-train follow-up decision"
  validation:
    - "checkpoint existence"
    - "model-key compatibility"
    - "bash -n runner"
    - "full-test eval summaries"
  dependencies: ["T-092"]
```

Status: completed with no promotion. Added `scripts/external/run_xr40_xr39leader_p5_soup_eval.sh` and `docs/resources/xr40_xr39leader_p5_soup_plan_2026_06_17.md`. All anchors existed and had matching `138` model keys. Ran 7 soups across GPU0/GPU1 with eval exit `0`. Best center was `c45p35f20` at `16.493494159834725`; best P10 was `c40p40f20` at `34.7750858102526`; best P5 was `c35p25f40` at `11.529762240818568`. All missed active gates. No-train soup follow-up is closed for now.

## T-094 XR-41 Loss-Ratio P5 Fallback

```yaml
task_card:
  task_id: T-094
  sub_agent: codex-native
  role: runtime-manager/evaluator
  objective: "Run trainable loss-ratio fallback to recover P5 after XR-40 no-train soup failure."
  file_ownership:
    - "scripts/external/run_xr41_lossratio_p5_fallback.sh"
    - "docs/resources/xr41_lossratio_p5_fallback_plan_2026_06_17.md"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Spec.md"
    - "docs/Validation.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill: ["ablation-study-designer", "computer-vision-expert", "caveman"]
  inputs:
    - "XR-38B best-P5 checkpoint"
    - "XR-39 c25p45f30 P10 checkpoint"
  outputs:
    - "XR-41 full-test eval summaries"
    - "updated P5 gate"
  validation:
    - "bash -n runner"
    - "train exit 0"
    - "full-test eval summaries"
  dependencies: ["T-093"]
```

Status: completed. Added `scripts/external/run_xr41_lossratio_p5_fallback.sh` and `docs/resources/xr41_lossratio_p5_fallback_plan_2026_06_17.md`. Both lanes completed train/eval with exit `0`, early-stopping at epoch `7/10`. XR-41A best-P5 reached `16.519514334201812/34.09821502821786/11.868197652271816`, promoting P5 only. XR-41B did not promote. Active gates are now center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.

## T-095 XR-42 P5-Preserve Low-Drift Branch

```yaml
task_card:
  task_id: T-095
  sub_agent: gpt5.5 evaluator
  role: runtime-manager/evaluator
  objective: "Preserve XR-39 center/P10 leaders while using XR-41A best-P5 as a bounded P5 teacher/reference."
  file_ownership:
    - "scripts/external/run_xr42_p5_preserve_lowdrift.sh"
    - "docs/resources/xr42_p5_preserve_lowdrift_plan_2026_06_17.md"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Spec.md"
    - "docs/Validation.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill: ["ablation-study-designer", "computer-vision-expert", "caveman"]
  inputs:
    - "XR-39 c60p25f15 center checkpoint"
    - "XR-39 c25p45f30 P10 checkpoint"
    - "XR-41A best-P5 checkpoint"
  outputs:
    - "XR-42 train/eval summaries"
    - "promotion/no-promotion decision"
  validation:
    - "bash -n runner"
    - "checkpoint existence"
    - "raw event-count contract"
    - "trainable filter report"
    - "full-test eval summaries"
  dependencies: ["T-094"]
```

Status: completed with no promotion. Added `scripts/external/run_xr42_p5_preserve_lowdrift.sh` and `docs/resources/xr42_p5_preserve_lowdrift_plan_2026_06_17.md`. Static validation passed. XR-42A launched on GPU0 from XR-39 center with XR-41A best-P5 teacher/reference, LR `3e-6`, best metric `metric_track_center_px`. XR-42B launched on GPU1 from XR-39 P10 with the same teacher/reference, LR `3e-6`, best metric `metric_track_p10_pct`. Both lanes passed startup validation: raw event-count contract, intended checkpoint load, resolved CUDA device, and trainable filter `6` tensors / `1,331,328` params. Both lanes completed train/eval with exit `0`, early-stopping at epoch `7/10`. XR-42A best-P10 reached `16.49802110535758/34.48299399103437/11.366922119685581`; XR-42A best-P5 and best-center reached `16.498888087272643/34.30739874839783/11.347789451054163`. XR-42B best-P10 reached `16.505801352432798/34.3171777180263/11.278911903926305`; XR-42B best-P5 reached `16.508924693720683/34.29506881577628/11.323554761069161`. No active gate promoted.

## T-096 XR-56 Direct P10 Soft-Threshold Supervision

```yaml
task_card:
  task_id: T-096
  sub_agent: "gpt5.5 explorer/evaluator"
  role: "expert/evaluator"
  objective: "Replace another checkpoint/scope P10 recovery attempt with a metric-aligned soft-threshold P10 loss and run two bounded GPU lanes."
  file_ownership:
    - "src/hbtxr/loss/bundles/track.py"
    - "src/hbtxr/loss/bundles/__init__.py"
    - "tests/test_track_center_l2_loss.py"
    - "scripts/external/run_xr56_soft_threshold_p10_supervision.sh"
    - "docs/resources/xr56_soft_threshold_p10_supervision_plan_2026_06_17.md"
    - "docs/Master-Plan.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
    - "docs/track/TODO.md"
  assigned_skill: ["ablation-study-designer", "cv-dl-expert", "caveman"]
  inputs:
    - "XR-39 P10 soup checkpoint c25p45f30"
    - "XR-52B best_track_p10 checkpoint"
    - "XR-55 no-promotion closeout"
  outputs:
    - "default-zero P10/P5 soft-threshold losses"
    - "XR-56 two-lane runner"
    - "XR-56 train/eval summaries"
  validation:
    - "py_compile"
    - "pytest tests/test_track_center_l2_loss.py"
    - "bash -n runner"
    - "lane A/B dry-run"
    - "full train/eval exit 0"
    - "compare center/P10/P5 against active gates"
  dependencies: ["T-095", "XR-55 closeout"]
```

Status: completed with partial promotion. GPT-5.5 explorer returned a read-only recommendation to prefer soft-threshold P10 loss over a new calibration head. Main integration added `center_soft_threshold_loss`, default-zero P10/P5 track and aux terms, unit coverage, and `scripts/external/run_xr56_soft_threshold_p10_supervision.sh`. Static validation and A/B dry-runs passed. XR-56A completed epoch `10/10`; XR-56B early-stopped at epoch `7/10`; train/eval exits were `0`. XR-56B best-P5 promoted center/P5 to `16.4701875601496/12.133503770828247`, but P10 remained below the XR-39 gate with best observed XR-56 P10 `34.62330012321472`.

## T-097 XR-57 Bounded P10 Center-Refine Calibration

```yaml
task_card:
  task_id: T-097
  sub_agent: "gpt5.5 explorer/evaluator attempted; fallback main-agent implementation"
  role: "expert/evaluator + implementer"
  objective: "Add a checkpoint-compatible bounded center-refinement head that routes into final track/state for P10 recovery."
  file_ownership:
    - "src/hbtxr/models/heads.py"
    - "src/hbtxr/models/tracker/head_factory.py"
    - "src/hbtxr/models/tracker/track_branch.py"
    - "src/hbtxr/models/hybrid_tracker.py"
    - "src/hbtxr/training/model_factory.py"
    - "src/hbtxr/loss/bundles/track.py"
    - "src/hbtxr/loss/stage2.py"
    - "tests/test_track_center_l2_loss.py"
    - "scripts/external/run_xr57_p10_center_refine_calibration.sh"
    - "docs/resources/xr57_p10_center_refine_calibration_plan_2026_06_17.md"
  assigned_skill:
    - "ablation-study-designer"
    - "cv-dl-expert"
    - "caveman"
  inputs:
    - "XR-56 active gates and no-P10 closeout"
    - "XR-39 P10 checkpoint"
    - "XR-56B center/P5 checkpoint"
  outputs:
    - "TrackCenterRefineHead"
    - "optional final track/state residual routing"
    - "XR-57 two-lane runner"
  validation:
    - "bash -n runner"
    - "py_compile"
    - "pytest tests/test_track_center_l2_loss.py"
    - "model-build smoke"
    - "lane A/B dry-run"
  dependencies: ["T-096"]
```

Status: completed with no promotion. Sub-agent spawn failed because the agent thread limit was reached, so the main agent performed the design review and implementation. Static validation passed, including target pytest `27 passed`. XR-57A early-stopped at epoch `7/12` and evaluated `best_track_p10` at `16.69362453562873/34.38307912690299/10.973214619500297` and `best_track_p5` at `16.715463175092424/34.849490649359566/10.925595603670393`. XR-57B completed epoch `12/12` and evaluated `best_track_p10` at `16.504537062985555/34.137755966186525/11.207483332497732` and `best_track_p5` at `16.4997801729611/34.190476996558054/11.476615987505232`. No active gate promoted. Next worker task should build XR-58 P10 teacher refresh.

## T-098 XR-58 P10 Teacher Refresh

```yaml
task_card:
  task_id: T-098
  sub_agent: "gpt5.3-codex-spark attempted; fallback main-agent implementation"
  role: "evaluator + implementer"
  objective: "Create a two-lane P10-specialized teacher-refresh runner seeded from XR-39 P10."
  file_ownership:
    - "scripts/external/run_xr58_p10_teacher_refresh.sh"
    - "docs/resources/xr58_p10_teacher_refresh_plan_2026_06_17.md"
    - "docs/track/PROGRESS.md"
  assigned_skill:
    - "ablation-study-designer"
    - "cv-dl-expert"
  inputs:
    - "XR-57 no-promotion closeout"
    - "XR-39 P10 checkpoint"
    - "fixed255k support-adaptive event-count contract"
  outputs:
    - "XR-58 two-lane runner"
    - "XR-58 plan"
  validation:
    - "bash -n runner"
    - "lane A/B dry-run"
    - "full train/eval exit 0"
    - "compare full-test P10 against 35.02295998845781"
  dependencies: ["T-097"]
```

Status: completed with no promotion. XR-58A/B train/eval completed with exit `0`; both lanes early-stopped at epoch `8/16`. XR-58A best-P10 reached `16.503614359242576/34.90306201662336/11.51275544847761`; XR-58A best-P5 reached `16.501813726765768/34.68920147078378/11.259779255730765`. XR-58B best-P10 reached `16.506438190596445/34.84226275852748/11.501275873184204`; XR-58B best-P5 reached `16.49833288192749/34.540391949244906/11.062075165339879`. Best XR-58 P10 improved over XR-57 but remained below XR-39 by about `0.1199`. Next worker task should run XR-59 as a narrow anchored-teacher bracket around XR-58A.

## T-099 XR-59 XR58A Teacher Bracket

```yaml
task_card:
  task_id: T-099
  sub_agent: "gpt5.3-codex-spark attempted; fallback main-agent implementation"
  role: "evaluator + implementer"
  objective: "Continue from XR-58A best-P10 with a narrow anchored-teacher bracket to close the remaining P10 gap."
  file_ownership:
    - "scripts/external/run_xr59_xr58a_teacher_bracket.sh"
    - "scripts/external/eval_xr59_completed_checkpoints.sh"
    - "docs/resources/xr59_xr58a_teacher_bracket_plan_2026_06_17.md"
    - "docs/track/PROGRESS.md"
  assigned_skill:
    - "ablation-study-designer"
    - "cv-dl-expert"
    - "caveman"
  inputs:
    - "XR-58A best-P10 checkpoint"
    - "XR-39 P10 teacher"
    - "fixed255k support-adaptive event-count contract"
  outputs:
    - "XR-59 two-lane runner"
    - "XR-59 plan"
  validation:
    - "bash -n runner"
    - "lane A/B dry-run"
    - "full train/eval exit 0"
    - "compare full-test P10 against 35.02295998845781"
  dependencies: ["T-098"]
```

Status: completed with no promotion. XR-59A/B both early-stopped at epoch `7/10`. Post-train full-test eval used `scripts/external/eval_xr59_completed_checkpoints.sh` because the train wrapper log stopped before the eval marker. XR-59A `best_track_p10` and `best_track_p5` both reached `16.520220368249074/34.300170864377705/11.376701021194458`; XR-59B `best_track_p10` and `best_track_p5` both reached `16.51065547806876/34.43409944261823/11.287415306908743`. No active gate promoted. Next worker task should close scalar P10-soft continuation and pivot to a different P10 representation/protocol.

## T-102 XR-60 P10 Candidate-Head Calibration

```yaml
task_card:
  task_id: T-102
  sub_agent: "codex-gpt5.5"
  role: "analyst/evaluator"
  objective: "Review runner patterns and define a two-lane XR-60 candidate-head calibration experiment after XR-59 no-promotion."
  file_ownership: []
  assigned_skill:
    - "ablation-study-designer"
    - "cv-dl-expert"
    - "caveman"
  inputs:
    - "XR-56 through XR-59 runner patterns"
    - "candidate head config/loss keys"
    - "active gates"
  outputs:
    - "XR-60 lane recommendations"
    - "runner override set"
    - "validation commands"
  validation:
    - "read-only review"
    - "exact key/path citations"
  dependencies: ["T-099"]
```

Status: completed read-only. Spark runtime was unavailable due usage limit, so GPT5.5 was used. Integrated recommendation into `scripts/external/run_xr60_p10_candidate_head.sh` and `docs/resources/xr60_p10_candidate_head_plan_2026_06_17.md`.

## T-103 XR-60 Implementation And Validation

```yaml
task_card:
  task_id: T-103
  sub_agent: "main-agent with GPT5.5 review input"
  role: "implementer/evaluator"
  objective: "Implement default-off candidate head/loss wiring and validate the XR-60 runner."
  file_ownership:
    - "src/hbtxr/models/heads.py"
    - "src/hbtxr/models/tracker/head_factory.py"
    - "src/hbtxr/models/tracker/track_branch.py"
    - "src/hbtxr/models/hybrid_tracker.py"
    - "src/hbtxr/training/model_factory.py"
    - "src/hbtxr/loss/bundles/track.py"
    - "src/hbtxr/loss/stage2.py"
    - "tests/test_track_center_l2_loss.py"
    - "tests/test_trainable_filter.py"
    - "scripts/external/run_xr60_p10_candidate_head.sh"
  assigned_skill:
    - "cv-dl-expert"
    - "ablation-study-designer"
  inputs:
    - "T-102 recommendation"
    - "XR-59B best-P10 checkpoint"
    - "XR-39 P10 teacher checkpoint"
  outputs:
    - "default-off candidate head"
    - "candidate losses"
    - "two-lane XR-60 runner"
  validation:
    - "py_compile"
    - "targeted pytest"
    - "model-build smoke"
    - "lane A/B dry-run"
  dependencies: ["T-102"]
```

Status: completed with no promotion. Static validation passed (`bash -n`, `py_compile`, targeted pytest `34 passed`, candidate model-build smoke, lane A/B dry-runs, and `git diff --check`). XR-60A/B train/eval completed with exit `0`; XR-60A early-stopped at epoch `7/12`, XR-60B early-stopped at epoch `11/12`. Best full-test metrics were XR-60A best-P10 `16.526124344553267/34.50255186898368/11.617772477013725`, XR-60A best-P5 `16.5058109828404/34.60331717899867/11.337160219464984`, XR-60B best-P10 `16.842585216249738/33.49277294022696/10.70663298198155`, and XR-60B best-P5 `16.717501049382346/33.87670159339905/10.971939107349941`.

## T-104 XR Eye-Tracking Reference Analysis Schema Review

```yaml
task_card:
  task_id: T-104
  sub_agent: "codex-gpt5.5"
  role: "analyst/evaluator"
  objective: "Review the requested hardware reference analysis style and define the required schema for XR eye-tracking codebase and paper analyses."
  file_ownership: []
  assigned_skill:
    - "code-analyzer"
    - "deep-research"
    - "documentation"
  inputs:
    - "hardware/analysis/vit-accel/codebases/FlexLLM/analysis.md"
    - "hardware/analysis/vit-accel/papers/EfficientViT-FPGA/analysis.md"
    - "existing anlaysis/xr-eye-tracking analysis files"
  outputs:
    - "codebase analysis heading/schema requirements"
    - "paper analysis heading/schema requirements"
    - "evidence and validation recommendations"
  validation:
    - "read-only review"
    - "schema mapped into generator and sample outputs"
  dependencies: []
```

Status: completed read-only. The schema was integrated into `scripts/external/write_xr_eye_tracking_detailed_analysis.py` and the regenerated analysis files.

## T-105 XR Eye-Tracking Detailed Analysis Regeneration

```yaml
task_card:
  task_id: T-105
  sub_agent: "main-agent with T-104 review input"
  role: "artifact manager/evaluator"
  objective: "Regenerate every XR eye-tracking codebase and paper analysis.md at the requested detailed reference-document level."
  file_ownership:
    - "scripts/external/write_xr_eye_tracking_detailed_analysis.py"
    - "anlaysis/xr-eye-tracking/codebases/*/analysis.md"
    - "anlaysis/xr-eye-tracking/papers/*/analysis.md"
    - "docs/track/PROGRESS.md"
    - "docs/Validation.md"
    - "docs/track/log.md"
    - "docs/track/CHANGELOG.md"
  assigned_skill:
    - "code-analyzer"
    - "deep-research"
    - "documentation"
  inputs:
    - "XR-Eye-Tracking Codebase tree"
    - "XR-Eye-Tracking Papers PDFs"
    - "HGTXR active gate context"
    - "T-104 schema"
  outputs:
    - "39 regenerated analysis.md files"
    - "reusable regeneration script"
    - "validation and progress records"
  validation:
    - "generator py_compile"
    - "required-token scan over all analysis.md files"
    - "wc -l audit"
    - "git diff --check"
  dependencies: ["T-104"]
```

Status: completed and reinforced on 2026-06-18. Regenerated `39` analysis files (`18` codebases and `21` papers), total `11574` lines. Validation passed with generator `py_compile`, full regeneration, required-section scans, FACET/EV-Eye spot checks, and `git diff --check`. Spark sub-agent attempts hit quota limits; GPT5.5 explorer outputs were integrated into the final schema.

## T-301 Next P0 Experiment Review

```yaml
task_card:
  task_id: T-301
  sub_agent: "codex-gpt5.5"
  role: "analyst"
  objective: "Review PAPER_REF-derived analysis, XR-60/XR-61 state, and active gates to rank the next P0 experiment."
  file_ownership: []
  assigned_skill:
    - "ablation-study-designer"
    - "cv-dl-expert"
  inputs:
    - "docs/track/PROGRESS.md"
    - "docs/Validation.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "anlaysis/xr-eye-tracking/papers/FACET/analysis.md"
    - "anlaysis/xr-eye-tracking/papers/EV-Eye/analysis.md"
    - "anlaysis/xr-eye-tracking/papers/Data-Scarcity/analysis.md"
  outputs:
    - "ranked next P0 candidates"
    - "XR-61 closeout priority"
    - "FACET/EV-Eye pseudo-ellipse diagnostic recommendation"
  validation:
    - "read-only"
    - "active gate citations"
  dependencies: []
```

Status: completed read-only. The main agent verified XR-61 eval summaries directly and used the recommendation to close XR-61 before launching a FACET-backed next branch.

## T-302 XR-62 FACET Geometry Auxiliary Refresh

```yaml
task_card:
  task_id: T-302
  sub_agent: "main-agent"
  role: "implementer/evaluator"
  objective: "Implement and run a FACET-style track-state auxiliary geometry refresh after XR-61 no-promotion."
  file_ownership:
    - "scripts/external/run_xr62_facet_geometry_aux_refresh.sh"
    - "docs/resources/xr62_facet_geometry_aux_refresh_plan_2026_06_17.md"
    - "docs/track/PROGRESS.md"
    - "docs/Validation.md"
    - "docs/track/log.md"
    - "docs/track/CHANGELOG.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
  assigned_skill:
    - "ablation-study-designer"
    - "cv-dl-expert"
  inputs:
    - "XR-56B best-P5 checkpoint"
    - "XR-58A best-P10 checkpoint"
    - "XR-39 P10 teacher checkpoint"
    - "FACET paper/codebase analyses"
  outputs:
    - "XR-62 runner"
    - "XR-62 plan"
    - "full-test closeout and active gate update"
  validation:
    - "bash -n"
    - "lane A/B DRY_RUN=1"
    - "GPU0/GPU1 train/eval exit 0"
    - "full-test eval summaries"
  dependencies: ["T-301"]
```

Status: completed. XR-62A promoted center to `16.468481131962367`; P10/P5 gates remain unchanged.

## T-404 XR-63 Input/Loader Verification

```yaml
task_card:
  task_id: T-404
  sub_agent: "gpt5.3-codex-spark"
  role: "evaluator"
  objective: "Locate current leader eval rows and verify target-loader keys for the XR-63 diagnostic."
  file_ownership: []
  assigned_skill:
    - "cv-dl-expert"
    - "caveman"
  inputs:
    - "runs/**/eval_rows.json"
    - "runs/**/eval_summary.json"
    - "src/hbtxr/data/loader.py"
    - "scripts/external/eval_hbtxr.py"
  outputs:
    - "leader eval_rows paths"
    - "loader target/weight key notes"
  validation:
    - "read-only"
  dependencies: ["T-302"]
```

Status: attempted but failed due GPT-5.3-Codex-Spark usage limit. Main-agent fallback verified loader keys directly: `cur_state`, `annotation_quality`, `mask_valid`, `closed_eye_flag`, and `valid_track`.

## T-405 XR-63 P10 Teacher-Target Oracle Diagnostic

```yaml
task_card:
  task_id: T-405
  sub_agent: "main-agent"
  role: "implementer/evaluator"
  objective: "Implement a no-train oracle diagnostic to decide whether P10 teacher-target construction has enough headroom after XR-62."
  file_ownership:
    - "scripts/external/analyze_p10_teacher_target_oracle.py"
    - "docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json"
    - "docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/Validation.md"
    - "docs/Sub-Plan.md"
  assigned_skill:
    - "ablation-study-designer"
    - "cv-dl-expert"
    - "caveman"
  inputs:
    - "XR-62A center leader eval rows"
    - "XR-39 P10 leader eval rows"
    - "XR-56B P5 leader eval rows"
    - "XR-58A P10-teacher eval rows"
    - "manifest1 test manifest"
  outputs:
    - "no-train oracle script"
    - "oracle JSON/Markdown evidence"
    - "next P0 decision"
  validation:
    - "py_compile"
    - "64-sample smoke"
    - "full-test diagnostic"
    - "eval_summary metric alignment"
  dependencies: ["T-404"]
```

Status: completed. XR-63 oracle upper bound reached `16.04023192701366/36.48596938775512/13.41751700680271`, exceeding all active gates. Decision: next P0 should be a leakage-safe P10 teacher-target construction branch.

## T-406 XR-64 Teacher-Target Construction Plan

```yaml
task_card:
  task_id: T-406
  sub_agent: "main-agent"
  role: "analyst"
  objective: "Define the next leakage-safe implementation path after XR-63 proves teacher-target headroom."
  file_ownership:
    - "docs/resources/xr64_teacher_target_construction_plan_2026_06_18.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/TODO.md"
  assigned_skill:
    - "ablation-study-designer"
    - "cv-dl-expert"
  inputs:
    - "XR-63 oracle JSON/Markdown"
    - "src/hbtxr/data/components.py"
    - "src/hbtxr/loss/stage2.py"
    - "src/hbtxr/loss/bundles/track.py"
  outputs:
    - "XR-64 construction plan"
    - "control variables"
    - "ablation matrix"
    - "validation plan"
  validation:
    - "read-only code inspection"
    - "tracking document links"
  dependencies: ["T-405"]
```

Status: completed. Current code has no direct sample-wise pseudo-target loading path, so XR-64 should add train/val-only target override generation plus dataset/loss support before training.

## T-407 XR-64 Leakage Review

```yaml
task_card:
  task_id: T-407
  sub_agent: "gpt-5.5"
  role: "evaluator"
  objective: "Review leakage risks and unit-test requirements for XR-64 target override implementation."
  file_ownership: []
  assigned_skill:
    - "cv-dl-expert"
    - "ablation-study-designer"
  inputs:
    - "src/hbtxr/data/**"
    - "src/hbtxr/loss/**"
    - "scripts/external/**"
  outputs:
    - "leakage risk checklist"
    - "test recommendations"
    - "runner validation commands"
  validation:
    - "read-only"
  dependencies: ["T-406"]
```

Status: completed read-only. Recommendations were integrated: do not overwrite `cur_state`, add separate target override tensors/losses, reject test manifest overrides by default, add builder/loss/dataset tests, and clear override config during test eval.

## T-408 XR-64 Target Override Implementation

```yaml
task_card:
  task_id: T-408
  sub_agent: "main-agent"
  role: "implementer"
  objective: "Implement XR-64 target override builder, dataset support, additive Stage2 loss, runner, and tests."
  file_ownership:
    - "scripts/external/build_xr64_teacher_target_overrides.py"
    - "scripts/external/run_xr64_teacher_target_construction.sh"
    - "src/hbtxr/config/runtime_config.py"
    - "src/hbtxr/data/dataset.py"
    - "src/hbtxr/loss/bundles/__init__.py"
    - "src/hbtxr/loss/bundles/track.py"
    - "src/hbtxr/loss/stage2.py"
    - "tests/test_track_target_override.py"
    - "tests/test_track_center_l2_loss.py"
    - "docs/resources/xr64_teacher_target_construction_implementation_2026_06_18.md"
  assigned_skill:
    - "cv-dl-expert"
    - "ablation-study-designer"
  inputs:
    - "XR-63 oracle result"
    - "T-407 leakage review"
  outputs:
    - "XR-64 implementation"
    - "targeted tests"
    - "runner dry-runs"
  validation:
    - "py_compile"
    - "bash -n"
    - "targeted pytest"
    - "builder smoke"
    - "A/B DRY_RUN"
  dependencies: ["T-407"]
```

Status: completed. Training launch is pending train split teacher eval rows and train-only override files.

## T-409 Stage1 Frame-Search Baseline Rebuild

```yaml
task_card:
  task_id: T-409
  sub_agent: "codex-native"
  role: "implementer"
  objective: "Improve Stage1 frame-based Search accuracy and define a promoted Stage1 baseline for later Stage2 initialization."
  file_ownership:
    - "scripts/external/run_stage1_frame_search_baseline_matrix.sh"
    - "docs/resources/stage1_frame_search_baseline_plan_2026_06_21.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "caveman:caveman"
  inputs:
    - "configs/external/mode1_stage1.yaml"
    - "runs/NON_XR/raw/raw_mode1_stage1_best_adamw255k_200ep_2gpu_20260611_222452/train/history.json"
    - "data/_internal/manifests/manifest1/train_manifest.jsonl"
    - "data/_internal/manifests/manifest1/val_manifest.jsonl"
  outputs:
    - "two-GPU detached Stage1 Search-only runner"
    - "Stage1 promotion gate and follow-up decision rule"
  validation:
    - "bash -n scripts/external/run_stage1_frame_search_baseline_matrix.sh"
    - "DRY_RUN=1 bash scripts/external/run_stage1_frame_search_baseline_matrix.sh"
    - "scripts/external/check_raw_event_count_training_readiness.py --require-cuda"
  dependencies: []
```

Status: ready to validate and launch. Real sub-agent spawn was attempted but blocked by `agent thread limit reached`; main agent owns integration and verification.
