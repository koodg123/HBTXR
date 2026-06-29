# HBTXR Subject-Independent Frame/Event 4-State Counts

Source count CSV: `references/report/FACET/HBTXR_subject_independent_dataset_motion_counts_2026-06-29.csv`

Dataset root: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`

Generated CSVs:

- `HBTXR_subject_independent_frame_event_4state_counts_2026-06-29.csv`
- `HBTXR_subject_independent_frame_event_coverage_2026-06-29.csv`

## Interpretation

This generated HBTXR dataset does not store independent frame-only and event-only sample sets. It stores valid frame-event pairs:

- `Frame`: the valid frame-aligned sample that provides the pseudo-label ellipse and motion class context.
- `Event`: the event window paired to the same frame timestamp and used as the model input.

Therefore the 4-state counts for `Frame` and `Event` are intentionally identical at the valid dataset-sample level. Raw source frames that were skipped because no ellipse label was available, or because no event window was available, are not assigned to the 4-state table; they are reported in the coverage table.

## 4-State Count Criteria

- `Sample Counts`: number of valid frame-event paired samples for the subject in the split.
- `Saccade`: pseudo-label ellipse-center velocity is greater than `493 px/s`.
- `Fixation`: sample is not Saccade and belongs to fixation sessions, i.e. session code `101` or `201`.
- `Smooth`: sample is not Saccade and belongs to smooth-pursuit sessions, i.e. session code `102` or `202`.
- `Blink`: sample belongs to a session whose `seg_blink` field contains `y` in `subject-motion-analysis/cache/inventory.csv` (`y`, `y(n)`, or `n(y)`).

Important: `Fixation`, `Saccade`, and `Smooth` are mutually exclusive motion classes. `Blink` is a session-level blink flag count, not a frame-local blink annotation, so it can overlap with the three motion classes.

## Subject-Wise Frame/Event 4-State Counts

| Data Type | Split | Subject | Sample Counts | Fixation | Saccade | Smooth | Blink |
|---|---|---:|---:|---:|---:|---:|---:|
| Frame | train | 1 | 30,487 | 19,848 | 6 | 10,633 | 20,533 |
| Frame | train | 2 | 30,822 | 19,661 | 0 | 11,161 | 15,432 |
| Frame | train | 3 | 29,848 | 19,224 | 16 | 10,608 | 20,253 |
| Frame | train | 4 | 30,092 | 19,597 | 8 | 10,487 | 15,021 |
| Frame | train | 5 | 30,227 | 19,698 | 1 | 10,528 | 15,260 |
| Frame | train | 6 | 30,669 | 20,172 | 20 | 10,477 | 15,536 |
| Frame | train | 7 | 30,493 | 19,918 | 1 | 10,574 | 15,215 |
| Frame | train | 8 | 29,726 | 19,607 | 2 | 10,117 | 14,946 |
| Frame | train | 9 | 30,942 | 19,790 | 27 | 11,125 | 15,399 |
| Frame | train | 10 | 30,636 | 19,634 | 4 | 10,998 | 15,363 |
| Frame | train | 11 | 30,707 | 19,505 | 2 | 11,200 | 15,395 |
| Frame | train | 12 | 30,587 | 19,569 | 9 | 11,009 | 20,819 |
| Frame | train | 13 | 30,811 | 19,726 | 11 | 11,074 | 15,370 |
| Frame | train | 14 | 30,664 | 19,499 | 1 | 11,164 | 20,930 |
| Frame | train | 15 | 30,501 | 19,561 | 10 | 10,930 | 15,382 |
| Frame | train | 16 | 29,644 | 19,404 | 0 | 10,240 | 14,924 |
| Frame | train | 17 | 28,837 | 18,902 | 0 | 9,935 | 19,607 |
| Frame | train | 18 | 30,291 | 19,930 | 0 | 10,361 | 15,165 |
| Frame | train | 19 | 31,233 | 20,107 | 0 | 11,126 | 21,139 |
| Frame | train | 20 | 29,201 | 18,462 | 34 | 10,705 | 14,672 |
| Frame | train | 21 | 30,826 | 19,652 | 0 | 11,174 | 15,327 |
| Frame | train | 22 | 30,883 | 19,811 | 7 | 11,065 | 15,425 |
| Frame | train | 23 | 30,260 | 19,727 | 0 | 10,533 | 15,171 |
| Frame | train | 24 | 30,482 | 19,969 | 0 | 10,513 | 15,386 |
| Frame | train | 25 | 30,322 | 19,926 | 13 | 10,383 | 15,141 |
| Frame | train | 26 | 29,587 | 19,331 | 2 | 10,254 | 14,760 |
| Frame | train | 27 | 30,272 | 19,762 | 15 | 10,495 | 15,140 |
| Frame | train | 28 | 30,168 | 19,754 | 3 | 10,411 | 20,290 |
| Frame | train | 29 | 28,809 | 18,715 | 11 | 10,083 | 14,391 |
| Frame | train | 30 | 30,099 | 19,567 | 20 | 10,512 | 14,939 |
| Frame | train | 31 | 30,656 | 19,616 | 0 | 11,040 | 15,259 |
| Frame | train | 32 | 30,091 | 19,417 | 0 | 10,674 | 14,818 |
| Frame | val | 33 | 30,620 | 19,845 | 44 | 10,731 | 15,323 |
| Frame | val | 34 | 30,359 | 19,610 | 140 | 10,609 | 15,206 |
| Frame | val | 35 | 30,736 | 19,602 | 24 | 11,110 | 15,204 |
| Frame | val | 36 | 31,061 | 19,811 | 16 | 11,234 | 15,519 |
| Frame | test | 37 | 30,953 | 19,908 | 1 | 11,044 | 15,390 |
| Frame | test | 38 | 29,760 | 19,377 | 3 | 10,380 | 14,833 |
| Frame | test | 39 | 30,372 | 19,840 | 0 | 10,532 | 15,159 |
| Frame | test | 40 | 30,757 | 19,751 | 1 | 11,005 | 15,217 |
| Frame | test | 41 | 30,684 | 19,534 | 40 | 11,110 | 15,296 |
| Frame | test | 42 | 29,739 | 18,820 | 0 | 10,919 | 20,342 |
| Frame | test | 43 | 30,738 | 19,558 | 3 | 11,177 | 15,252 |
| Frame | test | 44 | 30,766 | 19,742 | 0 | 11,024 | 20,874 |
| Frame | test | 45 | 30,607 | 19,421 | 3 | 11,183 | 15,278 |
| Frame | test | 46 | 30,709 | 19,696 | 0 | 11,013 | 15,269 |
| Frame | test | 47 | 30,715 | 19,720 | 4 | 10,991 | 15,312 |
| Frame | test | 48 | 30,371 | 19,432 | 0 | 10,939 | 15,059 |
| Event | train | 1 | 30,487 | 19,848 | 6 | 10,633 | 20,533 |
| Event | train | 2 | 30,822 | 19,661 | 0 | 11,161 | 15,432 |
| Event | train | 3 | 29,848 | 19,224 | 16 | 10,608 | 20,253 |
| Event | train | 4 | 30,092 | 19,597 | 8 | 10,487 | 15,021 |
| Event | train | 5 | 30,227 | 19,698 | 1 | 10,528 | 15,260 |
| Event | train | 6 | 30,669 | 20,172 | 20 | 10,477 | 15,536 |
| Event | train | 7 | 30,493 | 19,918 | 1 | 10,574 | 15,215 |
| Event | train | 8 | 29,726 | 19,607 | 2 | 10,117 | 14,946 |
| Event | train | 9 | 30,942 | 19,790 | 27 | 11,125 | 15,399 |
| Event | train | 10 | 30,636 | 19,634 | 4 | 10,998 | 15,363 |
| Event | train | 11 | 30,707 | 19,505 | 2 | 11,200 | 15,395 |
| Event | train | 12 | 30,587 | 19,569 | 9 | 11,009 | 20,819 |
| Event | train | 13 | 30,811 | 19,726 | 11 | 11,074 | 15,370 |
| Event | train | 14 | 30,664 | 19,499 | 1 | 11,164 | 20,930 |
| Event | train | 15 | 30,501 | 19,561 | 10 | 10,930 | 15,382 |
| Event | train | 16 | 29,644 | 19,404 | 0 | 10,240 | 14,924 |
| Event | train | 17 | 28,837 | 18,902 | 0 | 9,935 | 19,607 |
| Event | train | 18 | 30,291 | 19,930 | 0 | 10,361 | 15,165 |
| Event | train | 19 | 31,233 | 20,107 | 0 | 11,126 | 21,139 |
| Event | train | 20 | 29,201 | 18,462 | 34 | 10,705 | 14,672 |
| Event | train | 21 | 30,826 | 19,652 | 0 | 11,174 | 15,327 |
| Event | train | 22 | 30,883 | 19,811 | 7 | 11,065 | 15,425 |
| Event | train | 23 | 30,260 | 19,727 | 0 | 10,533 | 15,171 |
| Event | train | 24 | 30,482 | 19,969 | 0 | 10,513 | 15,386 |
| Event | train | 25 | 30,322 | 19,926 | 13 | 10,383 | 15,141 |
| Event | train | 26 | 29,587 | 19,331 | 2 | 10,254 | 14,760 |
| Event | train | 27 | 30,272 | 19,762 | 15 | 10,495 | 15,140 |
| Event | train | 28 | 30,168 | 19,754 | 3 | 10,411 | 20,290 |
| Event | train | 29 | 28,809 | 18,715 | 11 | 10,083 | 14,391 |
| Event | train | 30 | 30,099 | 19,567 | 20 | 10,512 | 14,939 |
| Event | train | 31 | 30,656 | 19,616 | 0 | 11,040 | 15,259 |
| Event | train | 32 | 30,091 | 19,417 | 0 | 10,674 | 14,818 |
| Event | val | 33 | 30,620 | 19,845 | 44 | 10,731 | 15,323 |
| Event | val | 34 | 30,359 | 19,610 | 140 | 10,609 | 15,206 |
| Event | val | 35 | 30,736 | 19,602 | 24 | 11,110 | 15,204 |
| Event | val | 36 | 31,061 | 19,811 | 16 | 11,234 | 15,519 |
| Event | test | 37 | 30,953 | 19,908 | 1 | 11,044 | 15,390 |
| Event | test | 38 | 29,760 | 19,377 | 3 | 10,380 | 14,833 |
| Event | test | 39 | 30,372 | 19,840 | 0 | 10,532 | 15,159 |
| Event | test | 40 | 30,757 | 19,751 | 1 | 11,005 | 15,217 |
| Event | test | 41 | 30,684 | 19,534 | 40 | 11,110 | 15,296 |
| Event | test | 42 | 29,739 | 18,820 | 0 | 10,919 | 20,342 |
| Event | test | 43 | 30,738 | 19,558 | 3 | 11,177 | 15,252 |
| Event | test | 44 | 30,766 | 19,742 | 0 | 11,024 | 20,874 |
| Event | test | 45 | 30,607 | 19,421 | 3 | 11,183 | 15,278 |
| Event | test | 46 | 30,709 | 19,696 | 0 | 11,013 | 15,269 |
| Event | test | 47 | 30,715 | 19,720 | 4 | 10,991 | 15,312 |
| Event | test | 48 | 30,371 | 19,432 | 0 | 10,939 | 15,059 |

## Split Totals

| Data Type | Split | Subject | Sample Counts | Fixation | Saccade | Smooth | Blink |
|---|---|---:|---:|---:|---:|---:|---:|
| Frame | train | all | 968,873 | 627,061 | 223 | 341,589 | 522,408 |
| Frame | val | all | 122,776 | 78,868 | 224 | 43,684 | 61,252 |
| Frame | test | all | 366,171 | 234,799 | 55 | 131,317 | 193,281 |
| Frame | all | all | 1,457,820 | 940,728 | 502 | 516,590 | 776,941 |
| Event | train | all | 968,873 | 627,061 | 223 | 341,589 | 522,408 |
| Event | val | all | 122,776 | 78,868 | 224 | 43,684 | 61,252 |
| Event | test | all | 366,171 | 234,799 | 55 | 131,317 | 193,281 |
| Event | all | all | 1,457,820 | 940,728 | 502 | 516,590 | 776,941 |

## Raw Frame Coverage

| Split | Subject | Raw Frames | Valid Frame-Event Pairs | Skipped | No Ellipse | No Events |
|---|---:|---:|---:|---:|---:|---:|
| train | 1 | 30,981 | 30,487 | 494 | 493 | 1 |
| train | 2 | 31,345 | 30,822 | 523 | 523 | 0 |
| train | 3 | 31,432 | 29,848 | 1,584 | 1,583 | 1 |
| train | 4 | 30,538 | 30,092 | 446 | 446 | 0 |
| train | 5 | 30,728 | 30,227 | 501 | 501 | 0 |
| train | 6 | 31,118 | 30,669 | 449 | 449 | 0 |
| train | 7 | 30,695 | 30,493 | 202 | 202 | 0 |
| train | 8 | 31,204 | 29,726 | 1,478 | 1,478 | 0 |
| train | 9 | 31,151 | 30,942 | 209 | 209 | 0 |
| train | 10 | 31,471 | 30,636 | 835 | 835 | 0 |
| train | 11 | 31,512 | 30,707 | 805 | 805 | 0 |
| train | 12 | 30,981 | 30,587 | 394 | 391 | 3 |
| train | 13 | 31,059 | 30,811 | 248 | 248 | 0 |
| train | 14 | 31,570 | 30,664 | 906 | 906 | 0 |
| train | 15 | 31,188 | 30,501 | 687 | 686 | 1 |
| train | 16 | 30,432 | 29,644 | 788 | 788 | 0 |
| train | 17 | 31,143 | 28,837 | 2,306 | 2,306 | 0 |
| train | 18 | 30,768 | 30,291 | 477 | 477 | 0 |
| train | 19 | 31,506 | 31,233 | 273 | 273 | 0 |
| train | 20 | 30,929 | 29,201 | 1,728 | 1,726 | 2 |
| train | 21 | 31,129 | 30,826 | 303 | 303 | 0 |
| train | 22 | 31,057 | 30,883 | 174 | 173 | 1 |
| train | 23 | 30,654 | 30,260 | 394 | 393 | 1 |
| train | 24 | 31,312 | 30,482 | 830 | 830 | 0 |
| train | 25 | 30,813 | 30,322 | 491 | 491 | 0 |
| train | 26 | 30,817 | 29,587 | 1,230 | 1,230 | 0 |
| train | 27 | 30,697 | 30,272 | 425 | 423 | 2 |
| train | 28 | 30,655 | 30,168 | 487 | 487 | 0 |
| train | 29 | 30,571 | 28,809 | 1,762 | 1,761 | 1 |
| train | 30 | 30,850 | 30,099 | 751 | 751 | 0 |
| train | 31 | 30,993 | 30,656 | 337 | 337 | 0 |
| train | 32 | 31,211 | 30,091 | 1,120 | 1,114 | 6 |
| val | 33 | 30,842 | 30,620 | 222 | 219 | 3 |
| val | 34 | 30,908 | 30,359 | 549 | 548 | 1 |
| val | 35 | 31,561 | 30,736 | 825 | 822 | 3 |
| val | 36 | 31,264 | 31,061 | 203 | 201 | 2 |
| test | 37 | 31,237 | 30,953 | 284 | 281 | 3 |
| test | 38 | 30,554 | 29,760 | 794 | 791 | 3 |
| test | 39 | 30,497 | 30,372 | 125 | 122 | 3 |
| test | 40 | 31,421 | 30,757 | 664 | 664 | 0 |
| test | 41 | 31,332 | 30,684 | 648 | 647 | 1 |
| test | 42 | 31,267 | 29,739 | 1,528 | 1,527 | 1 |
| test | 43 | 31,273 | 30,738 | 535 | 533 | 2 |
| test | 44 | 30,985 | 30,766 | 219 | 217 | 2 |
| test | 45 | 31,240 | 30,607 | 633 | 633 | 0 |
| test | 46 | 31,318 | 30,709 | 609 | 607 | 2 |
| test | 47 | 31,088 | 30,715 | 373 | 372 | 1 |
| test | 48 | 31,251 | 30,371 | 880 | 879 | 1 |
| all | all | 1,490,548 | 1,457,820 | 32,728 | 32,681 | 47 |
