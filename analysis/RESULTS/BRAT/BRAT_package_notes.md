# BRAT Package Notes

- Source prediction file: `submission_test.csv` with columns `row_id,x,y`.
- Source test file list has 96 unique eye-session entries with explicit `L`/`R` eye tokens.
- This package maps BRAT predictions to the same FACET subject-independent test metadata used by EIDet.
- Left-eye and right-eye test rows are both represented.
- Pixel error is reported in 64x64 input coordinates.
- IoU is a center proxy using ground-truth ellipse axes/angle shifted to the predicted center.
