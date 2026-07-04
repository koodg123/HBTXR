# Software Path Reference Audit

Generated: `2026-06-18T10:14:14`

## Safety

- Destructive actions: `false`
- This report identifies path-coupling risks before directory refactors.

## Summary

- Findings: `1166`
- Kind counts: `{"canonical_root": 254, "home_absolute": 130, "mnt_absolute": 1, "paths_config": 51, "project_root": 411, "venv_python": 318, "windows_drive": 1}`

## Highest-Coupled Files

| File | Findings |
|---|---:|
| `scripts/external/run_prepare_and_train.sh` | 34 |
| `configs/external/paths/reference_path.json` | 20 |
| `scripts/external/run_followup_queue_20260611.sh` | 20 |
| `scripts/external/_config.py` | 17 |
| `scripts/external/run_mode2_end_to_end.sh` | 16 |
| `scripts/external/check_raw_event_count_training_result.py` | 11 |
| `scripts/external/run_xr17b_p5anchor_supportadaptive_probe.sh` | 11 |
| `scripts/external/run_xr19a_p10recovery_micro_polish.sh` | 11 |
| `scripts/external/run_xr21_p10margin_supportadaptive_probe.sh` | 11 |
| `configs/external/paths/ev_eye_groundedsam_paths.json` | 10 |
| `docs/track/PROGRESS.md` | 10 |
| `docs/track/log.md` | 10 |
| `scripts/external/run_xr04d_failbucket_subset_probe.sh` | 10 |
| `scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh` | 10 |
| `configs/external/paths/ev_eye_raw_paths.json` | 9 |
| `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 9 |
| `scripts/external/check_raw_event_count_training_readiness.py` | 9 |
| `scripts/external/report_second_goal_status.py` | 9 |
| `scripts/external/run_best_adamw255k_stage1_stage2_200ep_2gpu.sh` | 9 |
| `scripts/external/run_centerhinge_probe_queue_20260611.sh` | 9 |
| `scripts/external/run_centerhingesq_probe_queue_20260611.sh` | 9 |
| `scripts/external/run_count_extension_queue_20260611.sh` | 9 |
| `scripts/external/run_ellipsestate_probe_queue_20260611.sh` | 9 |
| `scripts/external/run_optimizer_probe_queue_20260611.sh` | 9 |
| `scripts/external/run_prevpupilcrop_probe_queue_20260611.sh` | 9 |

## First Findings

| Kind | File | Line | Snippet |
|---|---|---:|---|
| `canonical_root` | `configs/external/base.yaml` | 83 | `canonical_root: null                                                          # workspace root containing canonical bundles` |
| `home_absolute` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 2 | `"project_root": "/home/kjm26/project/PRJXR/external_hybrid_package",` |
| `project_root` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 2 | `"project_root": "/home/kjm26/project/PRJXR/external_hybrid_package",` |
| `home_absolute` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 3 | `"raw_root": "/home/kjm26/project/dataset/EV_Eye/raw_data/Data_davis",` |
| `home_absolute` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 4 | `"groundedsam_root": "/home/kjm26/project/PRJXR/grounded-segment-anything",` |
| `home_absolute` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 5 | `"annotation_root": "/home/kjm26/project/PRJXR/external_hybrid_package/workspace/groundedsam_annotations",` |
| `home_absolute` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 6 | `"canonical_root": "/home/kjm26/project/PRJXR/external_hybrid_package/workspace/canonical",` |
| `canonical_root` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 6 | `"canonical_root": "/home/kjm26/project/PRJXR/external_hybrid_package/workspace/canonical",` |
| `home_absolute` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 7 | `"indexes_root": "/home/kjm26/project/PRJXR/external_hybrid_package/workspace/canonical/indexes",` |
| `home_absolute` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 8 | `"manifests_root": "/home/kjm26/project/PRJXR/external_hybrid_package/workspace/manifests",` |
| `home_absolute` | `configs/external/paths/ev_eye_groundedsam_paths.json` | 9 | `"preview_root": "/home/kjm26/project/PRJXR/external_hybrid_package/workspace/previews"` |
| `home_absolute` | `configs/external/paths/ev_eye_raw_paths.json` | 2 | `"project_root": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software",` |
| `project_root` | `configs/external/paths/ev_eye_raw_paths.json` | 2 | `"project_root": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software",` |
| `home_absolute` | `configs/external/paths/ev_eye_raw_paths.json` | 3 | `"raw_root": "/home/kjm26/project/dataset/EV_Eye/raw_data/Data_davis",` |
| `home_absolute` | `configs/external/paths/ev_eye_raw_paths.json` | 4 | `"canonical_root": "/home/kjm26/project/dataset/EV_Eye/canonical",` |
| `canonical_root` | `configs/external/paths/ev_eye_raw_paths.json` | 4 | `"canonical_root": "/home/kjm26/project/dataset/EV_Eye/canonical",` |
| `home_absolute` | `configs/external/paths/ev_eye_raw_paths.json` | 5 | `"indexes_root": "/home/kjm26/project/dataset/EV_Eye/canonical/indexes",` |
| `home_absolute` | `configs/external/paths/ev_eye_raw_paths.json` | 6 | `"manifests_root": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/data/_internal/manifests",` |
| `home_absolute` | `configs/external/paths/ev_eye_raw_paths.json` | 7 | `"annotation_root": "/home/kjm26/project/dataset/EV_Eye/raw_data/Data_davis",` |
| `home_absolute` | `configs/external/paths/ev_eye_raw_paths.json` | 9 | `"preview_root": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/data/_internal/previews"` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 4 | `"project_root": "/home/user/project/PRJXR/external_hybrid_package",` |
| `project_root` | `configs/external/paths/reference_path.json` | 4 | `"project_root": "/home/user/project/PRJXR/external_hybrid_package",` |
| `mnt_absolute` | `configs/external/paths/reference_path.json` | 5 | `"raw_root": "/mnt/d/dataset/EV_Eye/raw_data/Data_davis",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 6 | `"groundedsam_root": "/home/user/project/PRJXR/grounded-segment-anything",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 7 | `"annotation_root": "/home/user/project/PRJXR/external_hybrid_package/dataset/groundedsam_annotations",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 8 | `"canonical_root": "/home/user/project/PRJXR/external_hybrid_package/dataset/canonical",` |
| `canonical_root` | `configs/external/paths/reference_path.json` | 8 | `"canonical_root": "/home/user/project/PRJXR/external_hybrid_package/dataset/canonical",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 9 | `"indexes_root": "/home/user/project/PRJXR/external_hybrid_package/dataset/canonical/indexes",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 10 | `"manifests_root": "/home/user/project/PRJXR/external_hybrid_package/dataset/manifests",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 11 | `"preview_root": "/home/user/project/PRJXR/external_hybrid_package/dataset/previews"` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 17 | `"project_root": "/home/kjm26/project/PRJXR/external_hybrid_package",` |
| `project_root` | `configs/external/paths/reference_path.json` | 17 | `"project_root": "/home/kjm26/project/PRJXR/external_hybrid_package",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 18 | `"raw_root": "/home/kjm26/project/dataset/EV_Eye/raw_data/Data_davis",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 19 | `"groundedsam_root": "/home/kjm26/project/PRJXR/grounded-segment-anything",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 20 | `"annotation_root": "/home/kjm26/project/PRJXR/external_hybrid_package/dataset/groundedsam_annotations",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 21 | `"canonical_root": "/home/kjm26/project/PRJXR/external_hybrid_package/dataset/canonical",` |
| `canonical_root` | `configs/external/paths/reference_path.json` | 21 | `"canonical_root": "/home/kjm26/project/PRJXR/external_hybrid_package/dataset/canonical",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 22 | `"indexes_root": "/home/kjm26/project/PRJXR/external_hybrid_package/dataset/canonical/indexes",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 23 | `"manifests_root": "/home/kjm26/project/PRJXR/external_hybrid_package/dataset/manifests",` |
| `home_absolute` | `configs/external/paths/reference_path.json` | 24 | `"preview_root": "/home/kjm26/project/PRJXR/external_hybrid_package/dataset/previews"` |
| `project_root` | `configs/external/paths.example.json` | 2 | `"project_root": "..",` |
| `windows_drive` | `configs/external/paths.example.json` | 3 | `"raw_root": "E:/WSL/Shared/dataset/Eye/EV_Eye/raw_data/Data_davis",` |
| `canonical_root` | `configs/external/paths.example.json` | 11 | `"canonical_root": "./workspace",` |
| `home_absolute` | `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md` | 65 | `- `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF`` |
| `venv_python` | `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md` | 328 | `.venv/bin/python scripts/external/build_xr64_teacher_target_overrides.py \` |
| `venv_python` | `docs/resources/experiment_pause_documentation_2026_06_18.md` | 160 | `5. Run strict checker: `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py`.` |
| `venv_python` | `docs/resources/second_goal_artifact_index_2026_06_18.json` | 78 | `".venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary",` |
| `venv_python` | `docs/resources/second_goal_artifact_index_2026_06_18.json` | 79 | `".venv/bin/python scripts/external/report_second_goal_status.py --format summary",` |
| `venv_python` | `docs/resources/second_goal_artifact_index_2026_06_18.json` | 82 | `".venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_xr64_resume_artifacts.py tests/test_track_target_override.py",` |
| `venv_python` | `docs/resources/second_goal_artifact_index_2026_06_18.json` | 86 | `"strict_checker": ".venv/bin/python scripts/external/check_xr64_resume_artifacts.py --format summary",` |
| `venv_python` | `docs/resources/second_goal_artifact_index_2026_06_18.md` | 118 | `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary` |
| `venv_python` | `docs/resources/second_goal_artifact_index_2026_06_18.md` | 119 | `.venv/bin/python scripts/external/report_second_goal_status.py --format summary` |
| `venv_python` | `docs/resources/second_goal_artifact_index_2026_06_18.md` | 122 | `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_xr64_resume_artifacts.py tests/test_track_target_override.py` |
| `venv_python` | `docs/resources/second_goal_artifact_index_2026_06_18.md` | 146 | `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --format summary` |
| `home_absolute` | `docs/resources/second_goal_completion_audit_2026_06_18.md` | 9 | `1. Analyze papers under `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF`.` |
| `home_absolute` | `docs/resources/second_goal_completion_audit_2026_06_18.md` | 11 | `3. Use the planned experiments to maximize accuracy toward `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial`.` |
| `venv_python` | `docs/resources/second_goal_completion_audit_2026_06_18.md` | 110 | `\| 5 \| Run strict XR-64 artifact checker \| `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py` returns `0` \|` |
| `home_absolute` | `docs/resources/software_cleanup_inventory_2026_06_18.json` | 3 | `"project_root": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software",` |
| `project_root` | `docs/resources/software_cleanup_inventory_2026_06_18.json` | 3 | `"project_root": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software",` |
| `venv_python` | `docs/resources/xr27_trackheatmap_results_2026_06_16.md` | 59 | `- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_trainable_filter.py` passed: `22 passed, 1 warning`.` |
| `venv_python` | `docs/resources/xr29_post_xr28_failure_bucket_diagnostics_2026_06_16.md` | 30 | `.venv/bin/python scripts/external/summarize_eval_failure_buckets.py \` |
| `home_absolute` | `docs/resources/xr29_post_xr28_failure_bucket_diagnostics_2026_06_16.md` | 40 | `--override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \` |
| `canonical_root` | `docs/resources/xr29_post_xr28_failure_bucket_diagnostics_2026_06_16.md` | 40 | `--override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \` |
| `home_absolute` | `docs/resources/xr36_p5_anchor_continuation_plan_2026_06_16.md` | 28 | `'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr36_p5_anchor_continuation.sh a > runs/_logs/xr36a_p5anchor_gpu0_20260616.log 2>&1'` |
| `home_absolute` | `docs/resources/xr36_p5_anchor_continuation_plan_2026_06_16.md` | 33 | `'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr36_p5_anchor_continuation.sh b > runs/_logs/xr36b_p5anchor_gpu1_20260616.log 2>&1'` |
| `venv_python` | `docs/resources/xr56_soft_threshold_p10_supervision_plan_2026_06_17.md` | 60 | `- [x] `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py`` |
| `venv_python` | `docs/resources/xr57_p10_center_refine_calibration_plan_2026_06_17.md` | 61 | `- [x] `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py`` |
| `venv_python` | `docs/resources/xr57_p10_center_refine_calibration_plan_2026_06_17.md` | 62 | `- [x] `PYTHONPATH=src .venv/bin/python -c ... build_model(...)`` |
| `venv_python` | `docs/resources/xr60_p10_candidate_head_plan_2026_06_17.md` | 81 | `- [ ] `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_trainable_filter.py`` |
| `home_absolute` | `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` | 6 | `"reference_config": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/runs/eval_fixed255k_xr62_a_xr56b_facetaux_p10_preserve_adamw_lr3e_7_g32_hm0_004_off0_0015_c0_0025_p10soft0_004_p5soft0_0020_auxc0_0006_auxa0_0125_auxt0_005_best_track_p10_` |
| `home_absolute` | `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` | 7 | `"manifest": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/data/_internal/manifests/manifest1/test_manifest.jsonl",` |
| `home_absolute` | `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` | 9 | `"xr62a_center": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/runs/eval_fixed255k_xr62_a_xr56b_facetaux_p10_preserve_adamw_lr3e_7_g32_hm0_004_off0_0015_c0_0025_p10soft0_004_p5soft0_0020_auxc0_0006_auxa0_0125_auxt0_005_best_track_p10_test` |
| `home_absolute` | `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` | 10 | `"xr39_p10": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/runs/eval_fixed255k_xr39_mixedleader_soup_c25p45f30_gpu1_w0_20260617_010154/eval/test/eval_rows.json",` |
| `home_absolute` | `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` | 11 | `"xr56b_p5": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/runs/eval_fixed255k_xr56_b_xr52seed_xr39teacher_directp10p5soft_adamw_lr5e_7_g32_hm0_004_off0_0015_c0_0030_p10soft0_008_p5soft0_0020_state0_00035_best_track_p5_test_gpu1_w0_202606` |
| `home_absolute` | `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` | 12 | `"xr58a_p10teacher": "/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/runs/eval_fixed255k_xr58_a_xr39self_p10teacher_anchored_adamw_lr1e_6_g32_hm0_004_off0_0015_c0_0005_p10soft0_014_p5soft0_0_disttrue_state0_00010_best_track_p10_test_gpu0_w0` |
| `venv_python` | `docs/resources/xr64_resume_runbook_2026_06_18.md` | 69 | `.venv/bin/python scripts/external/check_raw_event_count_training_readiness.py --require-cuda` |
| `venv_python` | `docs/resources/xr64_resume_runbook_2026_06_18.md` | 76 | `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated` |
| `venv_python` | `docs/resources/xr64_resume_runbook_2026_06_18.md` | 77 | `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary` |
| `venv_python` | `docs/resources/xr64_resume_runbook_2026_06_18.md` | 86 | `.venv/bin/python - <<'PY'` |
| `venv_python` | `docs/resources/xr64_resume_runbook_2026_06_18.md` | 217 | `.venv/bin/python -c '` |
| `venv_python` | `docs/resources/xr64_resume_runbook_2026_06_18.md` | 247 | `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py` |
| `venv_python` | `docs/resources/xr64_resume_runbook_2026_06_18.md` | 248 | `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --format summary` |
| `venv_python` | `docs/resources/xr64_teacher_target_construction_implementation_2026_06_18.md` | 40 | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_target_override.py tests/test_track_center_l2_loss.py` |
| `venv_python` | `docs/resources/xr64_teacher_target_construction_implementation_2026_06_18.md` | 48 | `.venv/bin/python scripts/external/build_xr64_teacher_target_overrides.py \` |
| `home_absolute` | `docs/track/CONTRIBUTING.md` | 3 | `- Keep software work scoped to `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software` unless user explicitly expands scope.` |
| `home_absolute` | `docs/track/CONVERSATION.md` | 7 | `1. Analyze papers under `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF`.` |
| `home_absolute` | `docs/track/CONVERSATION.md` | 9 | `3. Push HGTXR software accuracy toward `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial` quality.` |
| `home_absolute` | `docs/track/CONVERSATION.md` | 13 | `- Use `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software` as hard working directory.` |
| `home_absolute` | `docs/track/EXTERNAL_HYBRID_PACKAGE_PAST_EXPERIMENT_RESULTS.md` | 7 | `- `/home/kjm26/project/PRJXR/XR-VIT/external_hybrid_package/docs`` |
| `home_absolute` | `docs/track/EXTERNAL_HYBRID_PACKAGE_PAST_EXPERIMENT_RESULTS.md` | 11 | `- `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software`` |
| `home_absolute` | `docs/track/EXTERNAL_HYBRID_PACKAGE_PAST_EXPERIMENT_RESULTS.md` | 465 | ``/home/kjm26/project/PRJXR/XR-VIT/external_hybrid_package/docs` and cross-checking against` |
| `home_absolute` | `docs/track/PROGRESS.md` | 24 | `- [x] Analyzed `/home/kjm26/project/PRJXR/References/XR-Eye-Tracking` codebase and paper inventory with main-agent reads plus GPT5.5 read-only sub-agents.` |
| `home_absolute` | `docs/track/PROGRESS.md` | 86 | `- [x] Main working directory set: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software`` |
| `venv_python` | `docs/track/PROGRESS.md` | 193 | `- Adaptive-count validation passed: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_event_builder.py tests/test_raw_event_count_contract.py` produced `6 passed`; config load and raw event-count contract passed. Pytest emitted a rea` |
| `venv_python` | `docs/track/PROGRESS.md` | 262 | `Implemented a decoded-center threshold hinge loss for the next loss-axis probe. Code path: `src/hbtxr/loss/bundles/track.py` now emits `loss_track_center_hinge`, controlled by `loss.track_center_hinge_weight` and `loss.track_center_hinge_ma` |
| `venv_python` | `docs/track/PROGRESS.md` | 274 | `Validation passed for the local-anchor crop axis: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_adaptive_roi_resolver.py tests/test_event_builder.py` produced `6 passed` with the known read-only parent `.pytest_cache` warning; co` |
| `venv_python` | `docs/track/PROGRESS.md` | 280 | `Prepared a checkpoint-compatible FACET-style ellipse-state loss axis while fixed105k/fixed110k continue running. `src/hbtxr/loss/bundles/track.py` now exposes disabled-by-default `loss.track_axis_log_weight` and `loss.track_angle_cos_weight` |
| `venv_python` | `docs/track/PROGRESS.md` | 410 | `Added squared decoded-center threshold hinge loss for the next P10/P5 loss-axis probe. Code path: `src/hbtxr/loss/bundles/track.py` now emits disabled-by-default `loss_track_center_hinge_sq`, controlled by `loss.track_center_hinge_sq_weight` |
| `venv_python` | `docs/track/PROGRESS.md` | 817 | `- [x] Validation passed: `bash -n`, `py_compile`, `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` with `23 passed`, lane A/B `DRY_RUN=1`, and `git diff --check`.` |
| `venv_python` | `docs/track/PROGRESS.md` | 832 | `- [x] Validation passed: `bash -n`, `py_compile`, `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` with `27 passed`, model-build smoke, and lane A/B `DRY_RUN=1`.` |
| `venv_python` | `docs/track/PROGRESS.md` | 936 | `- [x] Validation passed: `python3 -m py_compile scripts/external/report_second_goal_status.py`, `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_xr64_resume_artifacts.py` with `5 passed`, and `.venv/bin/python scripts/external` |
| `home_absolute` | `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 7 | `Train the paper-model software stack on `/home/kjm26/project/dataset/EV_Eye` without Frame/Event interpolation:` |
| `paths_config` | `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 15 | `- Path config: `configs/external/paths/ev_eye_raw_paths.json`` |
| `venv_python` | `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 77 | `.venv/bin/python scripts/external/check_raw_event_count_contract.py` |
| `venv_python` | `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 87 | `.venv/bin/python scripts/external/check_raw_event_count_training_readiness.py --require-cuda` |
| `venv_python` | `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 93 | `.venv/bin/python scripts/external/diagnose_cuda_nvml.py` |
| `venv_python` | `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 106 | `.venv/bin/python scripts/external/check_raw_event_count_training_result.py` |
| `venv_python` | `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 112 | `.venv/bin/python scripts/external/check_raw_event_count_training_result.py --allow-limited \` |
| `venv_python` | `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 175 | `.venv/bin/python scripts/external/check_raw_event_count_training_result.py \` |
| `home_absolute` | `docs/track/RAW_EVENT_COUNT_TRAINING.md` | 176 | `--project-root /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software \` |
| `venv_python` | `docs/track/TODO.md` | 7 | `- P0: Use `.venv/bin/python scripts/external/report_second_goal_status.py --format summary` for read-only current-state reporting while experiments remain paused.` |
| `venv_python` | `docs/track/TODO.md` | 9 | `- P0: Before XR-64A/B execution after resume, run `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py`; it must return `0`.` |
| `venv_python` | `docs/track/log.md` | 119 | `- Implemented adaptive-count event slicing in `src/hbtxr/data/event_builder.py` with tests for enabled scaling and disabled compatibility. Added `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_fi` |
| `venv_python` | `docs/track/log.md` | 138 | `- Validation for center-L2 path passed: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` produced `2 passed` with only the known parent `.pytest_cache` read-only warning; `python3 -m py_compile src/hbtxr/los` |
| `home_absolute` | `docs/track/log.md` | 280 | `- Source root analyzed: `/home/kjm26/project/PRJXR/XR-VIT/external_hybrid_package/docs`.` |
| `venv_python` | `docs/track/log.md` | 358 | `- Validation passed: `bash -n`, `python3 -m py_compile` on touched Python files, `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_loss_sample_weighting.py tests/test_weighted_sampler.py` with `19 p` |
| `venv_python` | `docs/track/log.md` | 1308 | `- `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` -> `27 passed`` |
| `venv_python` | `docs/track/log.md` | 1443 | `- `.venv/bin/python -m pytest -q tests/test_xr64_resume_artifacts.py tests/test_track_target_override.py` -> `9 passed`` |
| `venv_python` | `docs/track/log.md` | 1444 | `- `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated` -> `ready=true`, `generated_complete=false`` |
| `venv_python` | `docs/track/log.md` | 1445 | `- `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary` -> `resume_status=generated_incomplete`` |

## Refactor Policy

1. Keep existing script basenames until wrapper compatibility exists.
2. Replace direct `.venv/bin/python` with `PYTHON_BIN` only in active scripts first.
3. Move absolute dataset paths into paths config overlays before relocating scripts/configs.
4. Treat docs as historical evidence; do not rewrite old commands unless current authority docs supersede them.
