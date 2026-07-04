# Runs Compatibility Symlink Removal

- generated_at: `2026-06-18T11:07:04`
- execute: `true`
- include_shared: `false`
- removable_symlinks: `846`
- removed_symlinks: `846`
- retained_symlinks: `3`
- catalog_removed: `true`
- catalog_removed_entries: `2092`

## Policy

- Remove only top-level compatibility symlinks whose target already points into organized `runs/XR-*` or `runs/NON_XR/{eval,raw,other}`.
- Keep shared aliases by default: `runs/_logs`, `runs/diagnostics`, `runs/interpolated_checkpoints`.
- Remove `_catalog` only when it is a generated symlink-only tree, because it otherwise becomes stale after top-level symlink removal.

## First Removed

- `eval_adamw255k_200ep_stage_split_bestcenter_test_gpu0_w0_20260612_062656` -> `NON_XR/eval/eval_adamw255k_200ep_stage_split_bestcenter_test_gpu0_w0_20260612_062656`
- `eval_adamw255k_200ep_stage_split_bestp10_test_gpu0_w0_20260612_062949` -> `NON_XR/eval/eval_adamw255k_200ep_stage_split_bestp10_test_gpu0_w0_20260612_062949`
- `eval_adaptivecount_lr5e6_bestcenter_test_gpu0_w0_20260611_022233` -> `NON_XR/eval/eval_adaptivecount_lr5e6_bestcenter_test_gpu0_w0_20260611_022233`
- `eval_adaptivecount_sqrt_bestp10_test_gpu1_w0_20260611_022812` -> `NON_XR/eval/eval_adaptivecount_sqrt_bestp10_test_gpu1_w0_20260611_022812`
- `eval_centerckpt_bestcenter_test_20260610_220650` -> `NON_XR/eval/eval_centerckpt_bestcenter_test_20260610_220650`
- `eval_centerckpt_bestcenter_test_20260610_220703` -> `NON_XR/eval/eval_centerckpt_bestcenter_test_20260610_220703`
- `eval_centerckpt_bestcenter_test_20260610_221035` -> `NON_XR/eval/eval_centerckpt_bestcenter_test_20260610_221035`
- `eval_centerckpt_bestcenter_test_gpu0_20260610_222633` -> `NON_XR/eval/eval_centerckpt_bestcenter_test_gpu0_20260610_222633`
- `eval_centerckpt_bestcenter_test_gpu0_retry_20260610_222803` -> `NON_XR/eval/eval_centerckpt_bestcenter_test_gpu0_retry_20260610_222803`
- `eval_centerckpt_bestcenter_test_w0_20260610_221218` -> `NON_XR/eval/eval_centerckpt_bestcenter_test_w0_20260610_221218`
- `eval_centerckpt_bestp10_test_gpu0_w0_20260611_001755` -> `NON_XR/eval/eval_centerckpt_bestp10_test_gpu0_w0_20260611_001755`
- `eval_centerl2_30k_bestcenter_test_gpu1_w0_20260611_032232` -> `NON_XR/eval/eval_centerl2_30k_bestcenter_test_gpu1_w0_20260611_032232`
- `eval_centerl2_30k_bestp10_test_gpu1_w0_20260611_032421` -> `NON_XR/eval/eval_centerl2_30k_bestp10_test_gpu1_w0_20260611_032421`
- `eval_centerl2_35k_bestcenter_test_gpu1_w0_20260611_033529` -> `NON_XR/eval/eval_centerl2_35k_bestcenter_test_gpu1_w0_20260611_033529`
- `eval_centerl2_35k_bestp10_test_gpu1_w0_20260611_033623` -> `NON_XR/eval/eval_centerl2_35k_bestp10_test_gpu1_w0_20260611_033623`
- `eval_centerl2_fixed20k_bestcenter_test_gpu0_w0_20260611_030932` -> `NON_XR/eval/eval_centerl2_fixed20k_bestcenter_test_gpu0_w0_20260611_030932`
- `eval_centerl2_fixed20k_bestp10_test_gpu0_w0_20260611_031011` -> `NON_XR/eval/eval_centerl2_fixed20k_bestp10_test_gpu0_w0_20260611_031011`
- `eval_centerloss_bestcenter_test_gpu1_20260610_230859` -> `NON_XR/eval/eval_centerloss_bestcenter_test_gpu1_20260610_230859`
- `eval_centerloss_bestp10_test_20260610_220650` -> `NON_XR/eval/eval_centerloss_bestp10_test_20260610_220650`
- `eval_centerloss_bestp10_test_20260610_220703` -> `NON_XR/eval/eval_centerloss_bestp10_test_20260610_220703`
- `eval_centerloss_bestp10_test_20260610_221035` -> `NON_XR/eval/eval_centerloss_bestp10_test_20260610_221035`
- `eval_centerloss_bestp10_test_gpu0_retry_20260610_222812` -> `NON_XR/eval/eval_centerloss_bestp10_test_gpu0_retry_20260610_222812`
- `eval_centerloss_bestp10_test_w0_20260610_221218` -> `NON_XR/eval/eval_centerloss_bestp10_test_w0_20260610_221218`
- `eval_eyelorin_fixed255k_adamw_bestcenter_smoke64_gap50k_gpu0_20260616_002102` -> `NON_XR/eval/eval_eyelorin_fixed255k_adamw_bestcenter_smoke64_gap50k_gpu0_20260616_002102`
- `eval_eyelorin_fixed255k_adamw_bestcenter_smoke64_gpu0_20260616_002001` -> `NON_XR/eval/eval_eyelorin_fixed255k_adamw_bestcenter_smoke64_gpu0_20260616_002001`
- `eval_eyelorin_fixed255k_adamw_bestcenter_test_gpu0_20260616_001703` -> `NON_XR/eval/eval_eyelorin_fixed255k_adamw_bestcenter_test_gpu0_20260616_001703`
- `eval_fixed100k_bestcenter_test_gpu0_w0_20260611_052137` -> `NON_XR/eval/eval_fixed100k_bestcenter_test_gpu0_w0_20260611_052137`
- `eval_fixed100k_bestp10_test_gpu0_w0_20260611_052348` -> `NON_XR/eval/eval_fixed100k_bestp10_test_gpu0_w0_20260611_052348`
- `eval_fixed105k_bestcenter_test_gpu1_w0_20260611_053723` -> `NON_XR/eval/eval_fixed105k_bestcenter_test_gpu1_w0_20260611_053723`
- `eval_fixed105k_bestp10_test_gpu1_w0_20260611_053936` -> `NON_XR/eval/eval_fixed105k_bestp10_test_gpu1_w0_20260611_053936`
