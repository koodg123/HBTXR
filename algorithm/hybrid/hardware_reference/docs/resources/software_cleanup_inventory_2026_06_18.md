# Software Cleanup Inventory

Generated: `2026-06-18T10:13:04`

## Safety

- Destructive actions: `false`
- This report only classifies cleanup/refactor targets.

## Git Dirty State

- Changed/untracked entries under software scope: `236`
- Status counts: `{" M": 27, "??": 209}`

## Top-Level Size

| Path | Kind | Size |
|---|---|---:|
| `runs` | dir | 44.5 GiB |
| `.venv` | dir | 4.8 GiB |
| `data` | dir | 43.7 MiB |
| `src` | dir | 2.6 MiB |
| `scripts` | dir | 1.6 MiB |
| `anlaysis` | dir | 1.4 MiB |
| `docs` | dir | 1.3 MiB |
| `tests` | dir | 382.2 KiB |
| `configs` | dir | 140.4 KiB |
| `__pycache__` | dir | 53.5 KiB |
| `modules` | dir | 30.5 KiB |
| `tools` | dir | 10.2 KiB |
| `model.py` | file | 4.9 KiB |
| `runtime.py` | file | 4.0 KiB |
| `dataset.py` | file | 3.2 KiB |
| `scheduler.py` | file | 2.9 KiB |
| `io.py` | file | 2.8 KiB |
| `trainer.py` | file | 2.6 KiB |
| `losses.py` | file | 2.2 KiB |
| `geometry.py` | file | 2.1 KiB |
| `inference.py` | file | 1.7 KiB |
| `heads.py` | file | 1.7 KiB |
| `config.py` | file | 1.6 KiB |
| `README.md` | file | 1.5 KiB |
| `transforms.py` | file | 1.2 KiB |
| `evaluator.py` | file | 1.1 KiB |
| `event_repr.py` | file | 956.0 B |
| `metrics.py` | file | 881.0 B |
| `__init__.py` | file | 776.0 B |
| `visualization.py` | file | 561.0 B |
| `.agents` | dir | 40.0 B |
| `.codex` | dir | 40.0 B |
| `.git` | dir | 40.0 B |
| `32.56462665285383` | file | 0.0 B |

## Root File Findings

- Legacy shim candidates: `config.py, dataset.py, evaluator.py, event_repr.py, geometry.py, heads.py, inference.py, io.py, losses.py, metrics.py, model.py, runtime.py, scheduler.py, trainer.py, transforms.py, visualization.py`
- Stray nonstandard files: `32.56462665285383`

## Runs Classification

- Runs directory exists: `true`
- Run directories: `849`
- Referenced run names: `475`
- Label counts: `{"archive_candidate_closed_experiment": 130, "archive_candidate_duplicate": 6, "archive_candidate_eval": 315, "cleanup_candidate_smoke": 8, "preserve_or_review": 390}`

### Largest Run Directories

| Run | Label | Size | Reasons |
|---|---|---:|---|
| `interpolated_checkpoints` | `preserve_or_review` | 2.4 GiB | referenced_by_docs_or_code, shared_support_artifact_dir |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr0p0001_b64_s1p5_p10_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_075427` | `preserve_or_review` | 228.4 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr0p0001_b64_s1p5_center_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_075416` | `preserve_or_review` | 228.4 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ellipseaux_fullwidth_20260610_233657` | `preserve_or_review` | 225.8 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707` | `preserve_or_review` | 221.0 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709` | `preserve_or_review` | 221.0 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerloss_lite_fullwidth_20260610_224120` | `preserve_or_review` | 220.8 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_lite_fullwidth_20260610_223243` | `preserve_or_review` | 220.8 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerckpt_fullwidth_20260610_224355` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_selfreg_fullwidth_20260610_234009` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_recency2_fullwidth_20260610_222436` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_231556` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ema996_fullwidth_20260610_225302` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230041` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr2e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234649` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerloss_fullwidth_20260611_010010` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_center_trackpreserve_fullwidth_20260616_094218` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_strong_fullwidth_20260610_231233` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_teachercenter_fullwidth_20260610_233212` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230218` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234952` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth_20260611_004943` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_p10_trackpreserve_fullwidth_20260616_094220` | `preserve_or_review` | 220.7 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_schedulefree_fullwidth_20260610_232114` | `preserve_or_review` | 220.5 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111512` | `preserve_or_review` | 215.6 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819` | `preserve_or_review` | 215.6 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103148` | `preserve_or_review` | 215.6 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113816` | `preserve_or_review` | 215.6 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111515` | `preserve_or_review` | 215.6 MiB | referenced_by_docs_or_code |
| `raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205` | `preserve_or_review` | 215.6 MiB | referenced_by_docs_or_code |

## Recommended Next Actions

1. Review `preserve_or_review` rows before any archive/delete action.
2. Move no files until path-reference checks and compatibility wrappers are prepared.
3. Start with pycache/empty/stray-file cleanup after explicit approval.
4. Archive unreferenced `smoke` and duplicate eval runs only after leader checkpoint list is frozen.
