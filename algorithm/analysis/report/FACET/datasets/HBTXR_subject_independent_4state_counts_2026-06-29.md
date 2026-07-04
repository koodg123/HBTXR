# HBTXR Subject-Independent Dataset 4-State Counts

Source CSV: `references/report/FACET/HBTXR_subject_independent_dataset_motion_counts_2026-06-29.csv`

Dataset root: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`

## 4-State Count Criteria

The table uses all cached valid samples in `DeanDataset_full_unet_subject_independent`.

- `Sample Counts`: number of valid cached samples for the subject in the split.
- `Saccade`: sample is assigned to Saccade when pseudo-label ellipse-center velocity is greater than `493 px/s`.
- `Fixation`: sample is not Saccade and belongs to fixation sessions, i.e. session code `101` or `201`.
- `Smooth`: sample is not Saccade and belongs to smooth-pursuit sessions, i.e. session code `102` or `202`.
- `Blink`: sample belongs to a session whose `seg_blink` field contains `y` in `subject-motion-analysis/cache/inventory.csv` (`y`, `y(n)`, or `n(y)`).

Important interpretation: `Fixation`, `Saccade`, and `Smooth` are mutually exclusive motion classes. `Blink` is a session-level blink flag count, not a frame-local blink annotation, so it can overlap with the three motion classes.

## Subject-Wise 4-State Counts

| Split | Subject | Sample Counts | Fixation | Saccade | Smooth | Blink |
|---|---:|---:|---:|---:|---:|---:|
| train | 1 | 30,487 | 19,848 | 6 | 10,633 | 20,533 |
| train | 2 | 30,822 | 19,661 | 0 | 11,161 | 15,432 |
| train | 3 | 29,848 | 19,224 | 16 | 10,608 | 20,253 |
| train | 4 | 30,092 | 19,597 | 8 | 10,487 | 15,021 |
| train | 5 | 30,227 | 19,698 | 1 | 10,528 | 15,260 |
| train | 6 | 30,669 | 20,172 | 20 | 10,477 | 15,536 |
| train | 7 | 30,493 | 19,918 | 1 | 10,574 | 15,215 |
| train | 8 | 29,726 | 19,607 | 2 | 10,117 | 14,946 |
| train | 9 | 30,942 | 19,790 | 27 | 11,125 | 15,399 |
| train | 10 | 30,636 | 19,634 | 4 | 10,998 | 15,363 |
| train | 11 | 30,707 | 19,505 | 2 | 11,200 | 15,395 |
| train | 12 | 30,587 | 19,569 | 9 | 11,009 | 20,819 |
| train | 13 | 30,811 | 19,726 | 11 | 11,074 | 15,370 |
| train | 14 | 30,664 | 19,499 | 1 | 11,164 | 20,930 |
| train | 15 | 30,501 | 19,561 | 10 | 10,930 | 15,382 |
| train | 16 | 29,644 | 19,404 | 0 | 10,240 | 14,924 |
| train | 17 | 28,837 | 18,902 | 0 | 9,935 | 19,607 |
| train | 18 | 30,291 | 19,930 | 0 | 10,361 | 15,165 |
| train | 19 | 31,233 | 20,107 | 0 | 11,126 | 21,139 |
| train | 20 | 29,201 | 18,462 | 34 | 10,705 | 14,672 |
| train | 21 | 30,826 | 19,652 | 0 | 11,174 | 15,327 |
| train | 22 | 30,883 | 19,811 | 7 | 11,065 | 15,425 |
| train | 23 | 30,260 | 19,727 | 0 | 10,533 | 15,171 |
| train | 24 | 30,482 | 19,969 | 0 | 10,513 | 15,386 |
| train | 25 | 30,322 | 19,926 | 13 | 10,383 | 15,141 |
| train | 26 | 29,587 | 19,331 | 2 | 10,254 | 14,760 |
| train | 27 | 30,272 | 19,762 | 15 | 10,495 | 15,140 |
| train | 28 | 30,168 | 19,754 | 3 | 10,411 | 20,290 |
| train | 29 | 28,809 | 18,715 | 11 | 10,083 | 14,391 |
| train | 30 | 30,099 | 19,567 | 20 | 10,512 | 14,939 |
| train | 31 | 30,656 | 19,616 | 0 | 11,040 | 15,259 |
| train | 32 | 30,091 | 19,417 | 0 | 10,674 | 14,818 |
| val | 33 | 30,620 | 19,845 | 44 | 10,731 | 15,323 |
| val | 34 | 30,359 | 19,610 | 140 | 10,609 | 15,206 |
| val | 35 | 30,736 | 19,602 | 24 | 11,110 | 15,204 |
| val | 36 | 31,061 | 19,811 | 16 | 11,234 | 15,519 |
| test | 37 | 30,953 | 19,908 | 1 | 11,044 | 15,390 |
| test | 38 | 29,760 | 19,377 | 3 | 10,380 | 14,833 |
| test | 39 | 30,372 | 19,840 | 0 | 10,532 | 15,159 |
| test | 40 | 30,757 | 19,751 | 1 | 11,005 | 15,217 |
| test | 41 | 30,684 | 19,534 | 40 | 11,110 | 15,296 |
| test | 42 | 29,739 | 18,820 | 0 | 10,919 | 20,342 |
| test | 43 | 30,738 | 19,558 | 3 | 11,177 | 15,252 |
| test | 44 | 30,766 | 19,742 | 0 | 11,024 | 20,874 |
| test | 45 | 30,607 | 19,421 | 3 | 11,183 | 15,278 |
| test | 46 | 30,709 | 19,696 | 0 | 11,013 | 15,269 |
| test | 47 | 30,715 | 19,720 | 4 | 10,991 | 15,312 |
| test | 48 | 30,371 | 19,432 | 0 | 10,939 | 15,059 |
| all | all | 1,457,820 | 940,728 | 502 | 516,590 | 776,941 |

## Split Totals

| Split | Sample Counts | Fixation | Saccade | Smooth | Blink |
|---|---:|---:|---:|---:|---:|
| train | 968,873 | 627,061 | 223 | 341,589 | 522,408 |
| val | 122,776 | 78,868 | 224 | 43,684 | 61,252 |
| test | 366,171 | 234,799 | 55 | 131,317 | 193,281 |
| all | 1,457,820 | 940,728 | 502 | 516,590 | 776,941 |
