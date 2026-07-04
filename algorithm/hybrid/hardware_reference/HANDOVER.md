# HGTXR Software Handover

Current dated handover:

- `docs/HANDOVER_2026_06_27.md`

Current comprehensive synthesis:

- `docs/resources/current_project_synthesis_2026_06_27.md`

Short version:

- Active priority is Stage1 frame-based Search accuracy.
- Active baseline is `stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999`.
- Baseline metrics are P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- The next experiment is S1-OC integrated Search candidate branch.
- S1-OC is implemented and smoke-validated, but not yet trained.
- No `*s1oc*` run/log artifacts currently exist.

Launch command after GPU check:

```bash
DRY_RUN=0 DETACH=1 RUN_TAG=stage1_s1oc_search_candidate_$(date +%Y%m%d_%H%M%S) \
  bash scripts/external/run_stage1_s1oc_search_candidate.sh
```

Promotion requires at least `50` epochs, P10 above `28.27717937613433`, and center `<= 17.257583906065744`.
