# HGTXR-SW Architecture Decision Records

## ADR-001: Defer Stage3 Slimming Until Full-Width Accuracy Works

Date: 2026-06-10

Decision: run full-width Stage2 ablations before self-supervised slimming.

Reason: current Stage2 accuracy is too low. Distillation and structural width reduction can hide basic target/alignment defects. Submission pipeline uses slimming after Stage1/Stage2 accuracy is established.

Consequence: first ablations disable pruning or set effective width to 1.0. Hardware-aligned slim model is deferred.

## ADR-002: Use Stage2-Only Ablations First

Date: 2026-06-10

Decision: reuse current Stage1 checkpoint for initial experiments.

Reason: Stage2 is the immediate failure point and can be rerun faster than full Stage1+Stage2. Stage1 rerun is only justified after Stage2 ablation evidence identifies teacher/anchor weakness.

Consequence: experiment runner defaults to `--skip-prepare --skip-stage1 --stage1-checkpoint`.

## ADR-003: Separate Validation, Test, Center, And P10 Leaders

Date: 2026-06-12

Decision: track validation-center leader, test-center leader, P10 leader, and P5/tail behavior as separate leaderboards.

Reason: both legacy `external_hybrid_package` experiments and current `HGTXR/software` runs show that validation loss, validation center, test center, and P10/P5 can diverge. Legacy Mode0 Stage2 often peaked at epoch 1 while `loss_total` moved in a different direction. Current fixed-count and AdamW runs also show that best-center and best-P10 checkpoints are not always interchangeable.

Consequence: no experiment is promoted from validation alone. Each promoted candidate must name the checkpoint used, test split metrics, actual resolved hyperparameters, and stop/promotion reason. Queue thresholds must be refreshed from current eval summaries to avoid stale promotion gates.

## ADR-004: Do Not Train Calibrated Search/Event Fallback From XR-14A

Date: 2026-06-16

Decision: close the current calibrated search/event fallback path.

Reason: no-train confidence gating, previous-state/blend fallback, and `track_state_aux` fallback were negative. XR-14A restored all heads but did not beat the active XR-06C gates, and `search_state`/`event_state` joined all test rows while scoring P10/P5 `0.0` with center error around `178px`.

Consequence: the next 2차 목표 experiment should change representation or data density for the low `similarity_target`, `session_201`, and subject `42/45/39` failure mode. Search/event fallback can be reopened only after another branch produces a valid alternate state.

## ADR-005: Pivot From Scalar P10 Loss To Candidate Center Representation

Date: 2026-06-17

Decision: after XR-59 no-promotion, stop scalar P10-soft teacher continuation and test a default-off multi-candidate center head.

Reason: XR-56 soft-threshold supervision improved center/P5 but not P10; XR-57 single residual correction failed; XR-58 teacher refresh narrowed but did not clear the P10 gate; XR-59 scalar continuation regressed. A candidate head changes the coordinate representation and directly optimizes "at least one candidate within P10" rather than applying more scalar pressure to one center.

Consequence: XR-60 adds checkpoint-compatible optional parameters and loss keys, with all weights disabled by default. Promotion still requires full-test gate evidence.
