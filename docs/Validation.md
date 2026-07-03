# Validation: Retina/ERVT Subject37-48 Package

Date: 2026-07-03

## Syntax

The following scripts passed `py_compile`:

- `analysis/scripts/generate_retina_ervt_error_distribution.py`
- `analysis/scripts/rebuild_retina_ervt_with_hbtxr_motion_labels.py`
- `analysis/scripts/create_retina_ervt_jetcas_tables.py`

## Label Map Validation

HBTXR joined label map:

- Rows: 360,495
- Metadata rows: 366,171
- Missing joined labels in metadata sample space: 5,676
- Labels changed versus raw metadata `motion_state`: 3,234

The raw metadata had zero Saccade labels for subjects 39, 42, 44, 46, and 48. After rebuilding from the HBTXR joined label map, Retina and ERVT have nonzero Saccade counts for all subjects 37-48.

## Final Motion Counts

Retina:

| Subject | Fixation | Saccade | Smooth |
|---:|---:|---:|---:|
| 37 | 19583 | 275 | 10938 |
| 38 | 18738 | 214 | 10168 |
| 39 | 19698 | 95 | 10491 |
| 40 | 19196 | 144 | 10910 |
| 41 | 18720 | 656 | 10841 |
| 42 | 17440 | 341 | 10576 |
| 43 | 18744 | 434 | 11035 |
| 44 | 19507 | 158 | 10915 |
| 45 | 18772 | 334 | 11040 |
| 46 | 19308 | 100 | 10843 |
| 47 | 19371 | 284 | 10947 |
| 48 | 18876 | 145 | 10658 |

ERVT:

| Subject | Fixation | Saccade | Smooth |
|---:|---:|---:|---:|
| 37 | 19566 | 275 | 10924 |
| 38 | 18738 | 214 | 10168 |
| 39 | 19698 | 95 | 10456 |
| 40 | 19196 | 144 | 10910 |
| 41 | 18720 | 656 | 10841 |
| 42 | 17440 | 341 | 10576 |
| 43 | 18744 | 434 | 11035 |
| 44 | 19498 | 158 | 10910 |
| 45 | 18772 | 334 | 11040 |
| 46 | 19308 | 100 | 10843 |
| 47 | 19371 | 284 | 10916 |
| 48 | 18876 | 145 | 10658 |

## Overall Metrics

Retina joined rows:

- Rows: 360,495
- Mean error: 1.3434778992680325 input64 px
- Median error: 0.6244472367567486 input64 px
- P95 error: 4.822659651102031 input64 px
- P99 error: 12.551940315896012 input64 px

ERVT joined rows:

- Rows: 360,384
- Mean error: 7.821229081128637 input64 px
- Median error: 7.469624873676994 input64 px
- P95 error: 14.833466110022504 input64 px
- P99 error: 18.91460331944571 input64 px

## Excel Validation

Generated workbooks have the expected JETCAS table shape:

- 15 rows
- 18 columns
- Subject rows 37-48
- Nonempty Saccade cells for all subject rows
