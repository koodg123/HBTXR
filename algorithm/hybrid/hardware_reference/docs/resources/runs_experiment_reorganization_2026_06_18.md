# Runs Experiment Reorganization Report

Generated: `2026-06-18T10:26:19`

## Safety

- Original run directory contents were moved under experiment/category directories.
- Compatibility symlinks were left at original `runs/<run_id>` paths.
- No run directory contents were deleted.
- `runs/_catalog` was excluded because it is generated catalog metadata.

## Summary

- Dry run: `false`
- Source dirs considered: `849`
- Action counts: `{"moved_with_compat_symlink": 849}`

## Category Counts

| Category | Moved dirs |
|---|---:|
| `runs/NON_XR/eval` | 183 |
| `runs/NON_XR/raw` | 189 |
| `runs/NON_XR/shared` | 3 |
| `runs/XR-10` | 4 |
| `runs/XR-11` | 4 |
| `runs/XR-12` | 4 |
| `runs/XR-13` | 4 |
| `runs/XR-14` | 5 |
| `runs/XR-15` | 34 |
| `runs/XR-16` | 7 |
| `runs/XR-17` | 8 |
| `runs/XR-18` | 6 |
| `runs/XR-19` | 9 |
| `runs/XR-20` | 2 |
| `runs/XR-21` | 8 |
| `runs/XR-22` | 6 |
| `runs/XR-23` | 4 |
| `runs/XR-24` | 6 |
| `runs/XR-25` | 6 |
| `runs/XR-26` | 8 |
| `runs/XR-27` | 54 |
| `runs/XR-28` | 2 |
| `runs/XR-29` | 2 |
| `runs/XR-3` | 12 |
| `runs/XR-30` | 2 |
| `runs/XR-31` | 7 |
| `runs/XR-32` | 1 |
| `runs/XR-33` | 1 |
| `runs/XR-34` | 2 |
| `runs/XR-35` | 4 |
| `runs/XR-36` | 2 |
| `runs/XR-37` | 4 |
| `runs/XR-38` | 4 |
| `runs/XR-39` | 13 |
| `runs/XR-4` | 12 |
| `runs/XR-40` | 7 |
| `runs/XR-41` | 2 |
| `runs/XR-42` | 11 |
| `runs/XR-43` | 10 |
| `runs/XR-44` | 8 |
| `runs/XR-45` | 8 |
| `runs/XR-46` | 10 |
| `runs/XR-47` | 6 |
| `runs/XR-48` | 8 |
| `runs/XR-49` | 7 |
| `runs/XR-5` | 24 |
| `runs/XR-50` | 6 |
| `runs/XR-51` | 12 |
| `runs/XR-52` | 6 |
| `runs/XR-53` | 12 |
| `runs/XR-55` | 6 |
| `runs/XR-56` | 6 |
| `runs/XR-57` | 8 |
| `runs/XR-58` | 6 |
| `runs/XR-59` | 6 |
| `runs/XR-6` | 21 |
| `runs/XR-60` | 12 |
| `runs/XR-61` | 4 |
| `runs/XR-62` | 6 |
| `runs/XR-64` | 2 |
| `runs/XR-7` | 5 |
| `runs/XR-8` | 4 |
| `runs/XR-9` | 4 |

## Compatibility

Existing paths like `runs/<run_id>` now resolve through symlinks to
`runs/XR-<n>/<run_id>` or `runs/NON_XR/<group>/<run_id>`.

## First Actions

| Source | Destination | Action |
|---|---|---|
| `runs/_logs` | `runs/NON_XR/shared/_logs` | `moved_with_compat_symlink` |
| `runs/diagnostics` | `runs/NON_XR/shared/diagnostics` | `moved_with_compat_symlink` |
| `runs/eval_adamw255k_200ep_stage_split_bestcenter_test_gpu0_w0_20260612_062656` | `runs/NON_XR/eval/eval_adamw255k_200ep_stage_split_bestcenter_test_gpu0_w0_20260612_062656` | `moved_with_compat_symlink` |
| `runs/eval_adamw255k_200ep_stage_split_bestp10_test_gpu0_w0_20260612_062949` | `runs/NON_XR/eval/eval_adamw255k_200ep_stage_split_bestp10_test_gpu0_w0_20260612_062949` | `moved_with_compat_symlink` |
| `runs/eval_adaptivecount_lr5e6_bestcenter_test_gpu0_w0_20260611_022233` | `runs/NON_XR/eval/eval_adaptivecount_lr5e6_bestcenter_test_gpu0_w0_20260611_022233` | `moved_with_compat_symlink` |
| `runs/eval_adaptivecount_sqrt_bestp10_test_gpu1_w0_20260611_022812` | `runs/NON_XR/eval/eval_adaptivecount_sqrt_bestp10_test_gpu1_w0_20260611_022812` | `moved_with_compat_symlink` |
| `runs/eval_centerckpt_bestcenter_test_20260610_220650` | `runs/NON_XR/eval/eval_centerckpt_bestcenter_test_20260610_220650` | `moved_with_compat_symlink` |
| `runs/eval_centerckpt_bestcenter_test_20260610_220703` | `runs/NON_XR/eval/eval_centerckpt_bestcenter_test_20260610_220703` | `moved_with_compat_symlink` |
| `runs/eval_centerckpt_bestcenter_test_20260610_221035` | `runs/NON_XR/eval/eval_centerckpt_bestcenter_test_20260610_221035` | `moved_with_compat_symlink` |
| `runs/eval_centerckpt_bestcenter_test_gpu0_20260610_222633` | `runs/NON_XR/eval/eval_centerckpt_bestcenter_test_gpu0_20260610_222633` | `moved_with_compat_symlink` |
| `runs/eval_centerckpt_bestcenter_test_gpu0_retry_20260610_222803` | `runs/NON_XR/eval/eval_centerckpt_bestcenter_test_gpu0_retry_20260610_222803` | `moved_with_compat_symlink` |
| `runs/eval_centerckpt_bestcenter_test_w0_20260610_221218` | `runs/NON_XR/eval/eval_centerckpt_bestcenter_test_w0_20260610_221218` | `moved_with_compat_symlink` |
| `runs/eval_centerckpt_bestp10_test_gpu0_w0_20260611_001755` | `runs/NON_XR/eval/eval_centerckpt_bestp10_test_gpu0_w0_20260611_001755` | `moved_with_compat_symlink` |
| `runs/eval_centerl2_30k_bestcenter_test_gpu1_w0_20260611_032232` | `runs/NON_XR/eval/eval_centerl2_30k_bestcenter_test_gpu1_w0_20260611_032232` | `moved_with_compat_symlink` |
| `runs/eval_centerl2_30k_bestp10_test_gpu1_w0_20260611_032421` | `runs/NON_XR/eval/eval_centerl2_30k_bestp10_test_gpu1_w0_20260611_032421` | `moved_with_compat_symlink` |
| `runs/eval_centerl2_35k_bestcenter_test_gpu1_w0_20260611_033529` | `runs/NON_XR/eval/eval_centerl2_35k_bestcenter_test_gpu1_w0_20260611_033529` | `moved_with_compat_symlink` |
| `runs/eval_centerl2_35k_bestp10_test_gpu1_w0_20260611_033623` | `runs/NON_XR/eval/eval_centerl2_35k_bestp10_test_gpu1_w0_20260611_033623` | `moved_with_compat_symlink` |
| `runs/eval_centerl2_fixed20k_bestcenter_test_gpu0_w0_20260611_030932` | `runs/NON_XR/eval/eval_centerl2_fixed20k_bestcenter_test_gpu0_w0_20260611_030932` | `moved_with_compat_symlink` |
| `runs/eval_centerl2_fixed20k_bestp10_test_gpu0_w0_20260611_031011` | `runs/NON_XR/eval/eval_centerl2_fixed20k_bestp10_test_gpu0_w0_20260611_031011` | `moved_with_compat_symlink` |
| `runs/eval_centerloss_bestcenter_test_gpu1_20260610_230859` | `runs/NON_XR/eval/eval_centerloss_bestcenter_test_gpu1_20260610_230859` | `moved_with_compat_symlink` |
| `runs/eval_centerloss_bestp10_test_20260610_220650` | `runs/NON_XR/eval/eval_centerloss_bestp10_test_20260610_220650` | `moved_with_compat_symlink` |
| `runs/eval_centerloss_bestp10_test_20260610_220703` | `runs/NON_XR/eval/eval_centerloss_bestp10_test_20260610_220703` | `moved_with_compat_symlink` |
| `runs/eval_centerloss_bestp10_test_20260610_221035` | `runs/NON_XR/eval/eval_centerloss_bestp10_test_20260610_221035` | `moved_with_compat_symlink` |
| `runs/eval_centerloss_bestp10_test_gpu0_retry_20260610_222812` | `runs/NON_XR/eval/eval_centerloss_bestp10_test_gpu0_retry_20260610_222812` | `moved_with_compat_symlink` |
| `runs/eval_centerloss_bestp10_test_w0_20260610_221218` | `runs/NON_XR/eval/eval_centerloss_bestp10_test_w0_20260610_221218` | `moved_with_compat_symlink` |
| `runs/eval_eyelorin_fixed255k_adamw_bestcenter_smoke64_gap50k_gpu0_20260616_002102` | `runs/NON_XR/eval/eval_eyelorin_fixed255k_adamw_bestcenter_smoke64_gap50k_gpu0_20260616_002102` | `moved_with_compat_symlink` |
| `runs/eval_eyelorin_fixed255k_adamw_bestcenter_smoke64_gpu0_20260616_002001` | `runs/NON_XR/eval/eval_eyelorin_fixed255k_adamw_bestcenter_smoke64_gpu0_20260616_002001` | `moved_with_compat_symlink` |
| `runs/eval_eyelorin_fixed255k_adamw_bestcenter_test_gpu0_20260616_001703` | `runs/NON_XR/eval/eval_eyelorin_fixed255k_adamw_bestcenter_test_gpu0_20260616_001703` | `moved_with_compat_symlink` |
| `runs/eval_fixed100k_bestcenter_test_gpu0_w0_20260611_052137` | `runs/NON_XR/eval/eval_fixed100k_bestcenter_test_gpu0_w0_20260611_052137` | `moved_with_compat_symlink` |
| `runs/eval_fixed100k_bestp10_test_gpu0_w0_20260611_052348` | `runs/NON_XR/eval/eval_fixed100k_bestp10_test_gpu0_w0_20260611_052348` | `moved_with_compat_symlink` |
| `runs/eval_fixed105k_bestcenter_test_gpu1_w0_20260611_053723` | `runs/NON_XR/eval/eval_fixed105k_bestcenter_test_gpu1_w0_20260611_053723` | `moved_with_compat_symlink` |
| `runs/eval_fixed105k_bestp10_test_gpu1_w0_20260611_053936` | `runs/NON_XR/eval/eval_fixed105k_bestp10_test_gpu1_w0_20260611_053936` | `moved_with_compat_symlink` |
| `runs/eval_fixed110k_bestcenter_test_gpu0_w0_20260611_053743` | `runs/NON_XR/eval/eval_fixed110k_bestcenter_test_gpu0_w0_20260611_053743` | `moved_with_compat_symlink` |
| `runs/eval_fixed110k_bestp10_test_gpu0_w0_20260611_053959` | `runs/NON_XR/eval/eval_fixed110k_bestp10_test_gpu0_w0_20260611_053959` | `moved_with_compat_symlink` |
| `runs/eval_fixed115k_bestcenter_test_gpu1_w0_20260611_055330` | `runs/NON_XR/eval/eval_fixed115k_bestcenter_test_gpu1_w0_20260611_055330` | `moved_with_compat_symlink` |
| `runs/eval_fixed115k_bestp10_test_gpu1_w0_20260611_055541` | `runs/NON_XR/eval/eval_fixed115k_bestp10_test_gpu1_w0_20260611_055541` | `moved_with_compat_symlink` |
| `runs/eval_fixed120k_bestcenter_test_gpu0_w0_20260611_055546` | `runs/NON_XR/eval/eval_fixed120k_bestcenter_test_gpu0_w0_20260611_055546` | `moved_with_compat_symlink` |
| `runs/eval_fixed120k_bestp10_test_gpu0_w0_20260611_055800` | `runs/NON_XR/eval/eval_fixed120k_bestp10_test_gpu0_w0_20260611_055800` | `moved_with_compat_symlink` |
| `runs/eval_fixed125k_bestcenter_test_gpu1_w0_20260611_061215` | `runs/NON_XR/eval/eval_fixed125k_bestcenter_test_gpu1_w0_20260611_061215` | `moved_with_compat_symlink` |
| `runs/eval_fixed125k_bestp10_test_gpu1_w0_20260611_061433` | `runs/NON_XR/eval/eval_fixed125k_bestp10_test_gpu1_w0_20260611_061433` | `moved_with_compat_symlink` |
| `runs/eval_fixed130k_bestcenter_test_gpu0_w0_20260611_061202` | `runs/NON_XR/eval/eval_fixed130k_bestcenter_test_gpu0_w0_20260611_061202` | `moved_with_compat_symlink` |
| `runs/eval_fixed130k_bestp10_test_gpu0_w0_20260611_061420` | `runs/NON_XR/eval/eval_fixed130k_bestp10_test_gpu0_w0_20260611_061420` | `moved_with_compat_symlink` |
| `runs/eval_fixed135k_bestcenter_test_gpu1_w0_20260611_063112` | `runs/NON_XR/eval/eval_fixed135k_bestcenter_test_gpu1_w0_20260611_063112` | `moved_with_compat_symlink` |
| `runs/eval_fixed135k_bestp10_test_gpu1_w0_20260611_063333` | `runs/NON_XR/eval/eval_fixed135k_bestp10_test_gpu1_w0_20260611_063333` | `moved_with_compat_symlink` |
| `runs/eval_fixed140k_bestcenter_test_gpu0_w0_20260611_063107` | `runs/NON_XR/eval/eval_fixed140k_bestcenter_test_gpu0_w0_20260611_063107` | `moved_with_compat_symlink` |
| `runs/eval_fixed140k_bestp10_test_gpu0_w0_20260611_063329` | `runs/NON_XR/eval/eval_fixed140k_bestp10_test_gpu0_w0_20260611_063329` | `moved_with_compat_symlink` |
| `runs/eval_fixed145k_bestcenter_test_gpu1_w0_20260611_064953` | `runs/NON_XR/eval/eval_fixed145k_bestcenter_test_gpu1_w0_20260611_064953` | `moved_with_compat_symlink` |
| `runs/eval_fixed145k_bestp10_test_gpu1_w0_20260611_065216` | `runs/NON_XR/eval/eval_fixed145k_bestp10_test_gpu1_w0_20260611_065216` | `moved_with_compat_symlink` |
| `runs/eval_fixed150k_bestcenter_test_gpu0_w0_20260611_065009` | `runs/NON_XR/eval/eval_fixed150k_bestcenter_test_gpu0_w0_20260611_065009` | `moved_with_compat_symlink` |
| `runs/eval_fixed150k_bestp10_test_gpu0_w0_20260611_065233` | `runs/NON_XR/eval/eval_fixed150k_bestp10_test_gpu0_w0_20260611_065233` | `moved_with_compat_symlink` |
| `runs/eval_fixed155k_bestcenter_test_gpu1_w0_20260611_070836` | `runs/NON_XR/eval/eval_fixed155k_bestcenter_test_gpu1_w0_20260611_070836` | `moved_with_compat_symlink` |
| `runs/eval_fixed155k_bestp10_test_gpu1_w0_20260611_071104` | `runs/NON_XR/eval/eval_fixed155k_bestp10_test_gpu1_w0_20260611_071104` | `moved_with_compat_symlink` |
| `runs/eval_fixed160k_bestcenter_test_gpu0_w0_20260611_070835` | `runs/NON_XR/eval/eval_fixed160k_bestcenter_test_gpu0_w0_20260611_070835` | `moved_with_compat_symlink` |
| `runs/eval_fixed160k_bestp10_test_gpu0_w0_20260611_071103` | `runs/NON_XR/eval/eval_fixed160k_bestp10_test_gpu0_w0_20260611_071103` | `moved_with_compat_symlink` |
| `runs/eval_fixed165k_bestcenter_test_gpu1_w0_20260611_072734` | `runs/NON_XR/eval/eval_fixed165k_bestcenter_test_gpu1_w0_20260611_072734` | `moved_with_compat_symlink` |
| `runs/eval_fixed165k_bestp10_test_gpu1_w0_20260611_073005` | `runs/NON_XR/eval/eval_fixed165k_bestp10_test_gpu1_w0_20260611_073005` | `moved_with_compat_symlink` |
| `runs/eval_fixed170k_bestcenter_test_gpu0_w0_20260611_072745` | `runs/NON_XR/eval/eval_fixed170k_bestcenter_test_gpu0_w0_20260611_072745` | `moved_with_compat_symlink` |
| `runs/eval_fixed170k_bestp10_test_gpu0_w0_20260611_073018` | `runs/NON_XR/eval/eval_fixed170k_bestp10_test_gpu0_w0_20260611_073018` | `moved_with_compat_symlink` |
| `runs/eval_fixed175k_bestcenter_test_gpu1_w0_20260611_074834` | `runs/NON_XR/eval/eval_fixed175k_bestcenter_test_gpu1_w0_20260611_074834` | `moved_with_compat_symlink` |
| `runs/eval_fixed175k_bestp10_test_gpu1_w0_20260611_075108` | `runs/NON_XR/eval/eval_fixed175k_bestp10_test_gpu1_w0_20260611_075108` | `moved_with_compat_symlink` |
| `runs/eval_fixed180k_bestcenter_test_gpu0_w0_20260611_074847` | `runs/NON_XR/eval/eval_fixed180k_bestcenter_test_gpu0_w0_20260611_074847` | `moved_with_compat_symlink` |
| `runs/eval_fixed180k_bestp10_test_gpu0_w0_20260611_075125` | `runs/NON_XR/eval/eval_fixed180k_bestp10_test_gpu0_w0_20260611_075125` | `moved_with_compat_symlink` |
| `runs/eval_fixed185k_bestcenter_test_gpu1_w0_20260611_080940` | `runs/NON_XR/eval/eval_fixed185k_bestcenter_test_gpu1_w0_20260611_080940` | `moved_with_compat_symlink` |
| `runs/eval_fixed185k_bestp10_test_gpu1_w0_20260611_081219` | `runs/NON_XR/eval/eval_fixed185k_bestp10_test_gpu1_w0_20260611_081219` | `moved_with_compat_symlink` |
| `runs/eval_fixed18k_bestcenter_test_gpu1_w0_20260611_025644` | `runs/NON_XR/eval/eval_fixed18k_bestcenter_test_gpu1_w0_20260611_025644` | `moved_with_compat_symlink` |
| `runs/eval_fixed18k_bestp10_test_gpu1_w0_20260611_025812` | `runs/NON_XR/eval/eval_fixed18k_bestp10_test_gpu1_w0_20260611_025812` | `moved_with_compat_symlink` |
| `runs/eval_fixed190k_bestcenter_test_gpu0_w0_20260611_080924` | `runs/NON_XR/eval/eval_fixed190k_bestcenter_test_gpu0_w0_20260611_080924` | `moved_with_compat_symlink` |
| `runs/eval_fixed190k_bestp10_test_gpu0_w0_20260611_081205` | `runs/NON_XR/eval/eval_fixed190k_bestp10_test_gpu0_w0_20260611_081205` | `moved_with_compat_symlink` |
| `runs/eval_fixed195k_bestcenter_test_gpu1_w0_20260611_083117` | `runs/NON_XR/eval/eval_fixed195k_bestcenter_test_gpu1_w0_20260611_083117` | `moved_with_compat_symlink` |
| `runs/eval_fixed195k_bestp10_test_gpu1_w0_20260611_083355` | `runs/NON_XR/eval/eval_fixed195k_bestp10_test_gpu1_w0_20260611_083355` | `moved_with_compat_symlink` |
| `runs/eval_fixed200k_bestcenter_test_gpu0_w0_20260611_083126` | `runs/NON_XR/eval/eval_fixed200k_bestcenter_test_gpu0_w0_20260611_083126` | `moved_with_compat_symlink` |
| `runs/eval_fixed200k_bestp10_test_gpu0_w0_20260611_083407` | `runs/NON_XR/eval/eval_fixed200k_bestp10_test_gpu0_w0_20260611_083407` | `moved_with_compat_symlink` |
| `runs/eval_fixed205k_bestcenter_test_gpu1_w0_20260611_085245` | `runs/NON_XR/eval/eval_fixed205k_bestcenter_test_gpu1_w0_20260611_085245` | `moved_with_compat_symlink` |
| `runs/eval_fixed205k_bestp10_test_gpu1_w0_20260611_085529` | `runs/NON_XR/eval/eval_fixed205k_bestp10_test_gpu1_w0_20260611_085529` | `moved_with_compat_symlink` |
| `runs/eval_fixed20k_bestcenter_test_gpu1_w0_20260611_023553` | `runs/NON_XR/eval/eval_fixed20k_bestcenter_test_gpu1_w0_20260611_023553` | `moved_with_compat_symlink` |
| `runs/eval_fixed20k_bestp10_test_gpu1_w0_20260611_023632` | `runs/NON_XR/eval/eval_fixed20k_bestp10_test_gpu1_w0_20260611_023632` | `moved_with_compat_symlink` |
| `runs/eval_fixed210k_bestcenter_test_gpu0_w0_20260611_085248` | `runs/NON_XR/eval/eval_fixed210k_bestcenter_test_gpu0_w0_20260611_085248` | `moved_with_compat_symlink` |
| `runs/eval_fixed210k_bestp10_test_gpu0_w0_20260611_085538` | `runs/NON_XR/eval/eval_fixed210k_bestp10_test_gpu0_w0_20260611_085538` | `moved_with_compat_symlink` |
| `runs/eval_fixed215k_bestcenter_test_gpu1_w0_20260611_091557` | `runs/NON_XR/eval/eval_fixed215k_bestcenter_test_gpu1_w0_20260611_091557` | `moved_with_compat_symlink` |
| `runs/eval_fixed215k_bestp10_test_gpu1_w0_20260611_091846` | `runs/NON_XR/eval/eval_fixed215k_bestp10_test_gpu1_w0_20260611_091846` | `moved_with_compat_symlink` |
| `runs/eval_fixed220k_bestcenter_test_gpu0_w0_20260611_091547` | `runs/NON_XR/eval/eval_fixed220k_bestcenter_test_gpu0_w0_20260611_091547` | `moved_with_compat_symlink` |
| `runs/eval_fixed220k_bestp10_test_gpu0_w0_20260611_091835` | `runs/NON_XR/eval/eval_fixed220k_bestp10_test_gpu0_w0_20260611_091835` | `moved_with_compat_symlink` |
| `runs/eval_fixed225k_bestcenter_test_gpu1_w0_20260611_093808` | `runs/NON_XR/eval/eval_fixed225k_bestcenter_test_gpu1_w0_20260611_093808` | `moved_with_compat_symlink` |
| `runs/eval_fixed225k_bestp10_test_gpu1_w0_20260611_094100` | `runs/NON_XR/eval/eval_fixed225k_bestp10_test_gpu1_w0_20260611_094100` | `moved_with_compat_symlink` |
| `runs/eval_fixed22k_bestcenter_test_gpu0_w0_20260611_030731` | `runs/NON_XR/eval/eval_fixed22k_bestcenter_test_gpu0_w0_20260611_030731` | `moved_with_compat_symlink` |
| `runs/eval_fixed22k_bestp10_test_gpu0_w0_20260611_030849` | `runs/NON_XR/eval/eval_fixed22k_bestp10_test_gpu0_w0_20260611_030849` | `moved_with_compat_symlink` |
| `runs/eval_fixed230k_bestcenter_test_gpu0_w0_20260611_093824` | `runs/NON_XR/eval/eval_fixed230k_bestcenter_test_gpu0_w0_20260611_093824` | `moved_with_compat_symlink` |
| `runs/eval_fixed230k_bestp10_test_gpu0_w0_20260611_094116` | `runs/NON_XR/eval/eval_fixed230k_bestp10_test_gpu0_w0_20260611_094116` | `moved_with_compat_symlink` |
| `runs/eval_fixed235k_bestcenter_test_gpu1_w0_20260611_100104` | `runs/NON_XR/eval/eval_fixed235k_bestcenter_test_gpu1_w0_20260611_100104` | `moved_with_compat_symlink` |
| `runs/eval_fixed235k_bestp10_test_gpu1_w0_20260611_100356` | `runs/NON_XR/eval/eval_fixed235k_bestp10_test_gpu1_w0_20260611_100356` | `moved_with_compat_symlink` |
| `runs/eval_fixed240k_bestcenter_test_gpu0_w0_20260611_100105` | `runs/NON_XR/eval/eval_fixed240k_bestcenter_test_gpu0_w0_20260611_100105` | `moved_with_compat_symlink` |
| `runs/eval_fixed240k_bestp10_test_gpu0_w0_20260611_100404` | `runs/NON_XR/eval/eval_fixed240k_bestp10_test_gpu0_w0_20260611_100404` | `moved_with_compat_symlink` |
| `runs/eval_fixed245k_bestcenter_test_gpu1_w0_20260611_102421` | `runs/NON_XR/eval/eval_fixed245k_bestcenter_test_gpu1_w0_20260611_102421` | `moved_with_compat_symlink` |
| `runs/eval_fixed245k_bestp10_test_gpu1_w0_20260611_102719` | `runs/NON_XR/eval/eval_fixed245k_bestp10_test_gpu1_w0_20260611_102719` | `moved_with_compat_symlink` |
| `runs/eval_fixed250k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260616_002657` | `runs/NON_XR/eval/eval_fixed250k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260616_002657` | `moved_with_compat_symlink` |
| `runs/eval_fixed250k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260616_004544` | `runs/NON_XR/eval/eval_fixed250k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260616_004544` | `moved_with_compat_symlink` |
| `runs/eval_fixed250k_adamw_lr8e_6_bestp10_test_gpu0_w0_20260616_002658` | `runs/NON_XR/eval/eval_fixed250k_adamw_lr8e_6_bestp10_test_gpu0_w0_20260616_002658` | `moved_with_compat_symlink` |
| `runs/eval_fixed250k_adamw_lr8e_6_bestp10_test_gpu0_w0_20260616_004850` | `runs/NON_XR/eval/eval_fixed250k_adamw_lr8e_6_bestp10_test_gpu0_w0_20260616_004850` | `moved_with_compat_symlink` |
| `runs/eval_fixed250k_bestcenter_test_gpu0_w0_20260611_102523` | `runs/NON_XR/eval/eval_fixed250k_bestcenter_test_gpu0_w0_20260611_102523` | `moved_with_compat_symlink` |
| `runs/eval_fixed250k_bestp10_test_gpu0_w0_20260611_102822` | `runs/NON_XR/eval/eval_fixed250k_bestp10_test_gpu0_w0_20260611_102822` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_adamw_lr1e_5_bestcenter_test_gpu1_w0_20260616_011048` | `runs/NON_XR/eval/eval_fixed255k_adamw_lr1e_5_bestcenter_test_gpu1_w0_20260616_011048` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_adamw_lr1e_5_bestp10_test_gpu1_w0_20260616_011358` | `runs/NON_XR/eval/eval_fixed255k_adamw_lr1e_5_bestp10_test_gpu1_w0_20260616_011358` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_011043` | `runs/NON_XR/eval/eval_fixed255k_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_011043` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_011356` | `runs/NON_XR/eval/eval_fixed255k_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_011356` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260611_113913` | `runs/NON_XR/eval/eval_fixed255k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260611_113913` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_adamw_lr8e_6_bestp10_test_gpu0_w0_20260611_114133` | `runs/NON_XR/eval/eval_fixed255k_adamw_lr8e_6_bestp10_test_gpu0_w0_20260611_114133` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_bestcenter_test_gpu1_w0_20260611_104936` | `runs/NON_XR/eval/eval_fixed255k_bestcenter_test_gpu1_w0_20260611_104936` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_bestp10_test_gpu1_w0_20260611_105246` | `runs/NON_XR/eval/eval_fixed255k_bestp10_test_gpu1_w0_20260611_105246` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_prevpupilcrop_bestcenter_test_cuda_1_w0_20260611_133617` | `runs/NON_XR/eval/eval_fixed255k_prevpupilcrop_bestcenter_test_cuda_1_w0_20260611_133617` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_prevpupilcrop_bestp10_test_cuda_1_w0_20260611_133859` | `runs/NON_XR/eval/eval_fixed255k_prevpupilcrop_bestp10_test_cuda_1_w0_20260611_133859` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p025_angle0p01_adamw_lr1e_5_bestcenter_test_gpu0_w0_20260616_013647` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p025_angle0p01_adamw_lr1e_5_bestcenter_test_gpu0_w0_20260616_013647` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p025_angle0p01_adamw_lr1e_5_bestp10_test_gpu0_w0_20260616_013954` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p025_angle0p01_adamw_lr1e_5_bestp10_test_gpu0_w0_20260616_013954` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p025_angle0p01_adamw_lr6e_6_bestcenter_test_gpu1_w0_20260616_013846` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p025_angle0p01_adamw_lr6e_6_bestcenter_test_gpu1_w0_20260616_013846` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p025_angle0p01_adamw_lr6e_6_bestp10_test_gpu1_w0_20260616_014148` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p025_angle0p01_adamw_lr6e_6_bestp10_test_gpu1_w0_20260616_014148` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr1e_5_bestcenter_test_gpu0_w0_20260616_020507` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr1e_5_bestcenter_test_gpu0_w0_20260616_020507` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr1e_5_bestp10_test_gpu0_w0_20260616_020747` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr1e_5_bestp10_test_gpu0_w0_20260616_020747` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr6e_6_bestcenter_test_gpu1_w0_20260616_020002` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr6e_6_bestcenter_test_gpu1_w0_20260616_020002` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr6e_6_bestp10_test_gpu1_w0_20260616_020310` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr6e_6_bestp10_test_gpu1_w0_20260616_020310` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p075_angle0p03_adamw_lr1e_5_bestcenter_test_gpu0_w0_20260616_022634` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p075_angle0p03_adamw_lr1e_5_bestcenter_test_gpu0_w0_20260616_022634` | `moved_with_compat_symlink` |
| `runs/eval_fixed255k_xr03_ellipsestate_axis0p075_angle0p03_adamw_lr1e_5_bestp10_test_gpu0_w0_20260616_022942` | `runs/XR-3/eval_fixed255k_xr03_ellipsestate_axis0p075_angle0p03_adamw_lr1e_5_bestp10_test_gpu0_w0_20260616_022942` | `moved_with_compat_symlink` |
