# XR-63 P10 Teacher Target Oracle Diagnostic

## Summary

- sample_count: `2238`
- active_gate_center: `<16.468481131962367`
- active_gate_p10: `>35.02295998845781`
- active_gate_p5: `>12.133503770828247`
- decision: `GO_teacher_target_construction`

## Model Metrics

| Model | Center px | P10 % | P5 % | Samples |
|---|---:|---:|---:|---:|
| `xr62a_center` | 16.468481 | 34.307823 | 12.052721 | 2238 |
| `xr39_p10` | 16.503941 | 35.022959 | 11.460459 | 2238 |
| `xr56b_p5` | 16.470188 | 34.292942 | 12.133503 | 2238 |
| `xr58a_p10teacher` | 16.503614 | 34.903061 | 11.512755 | 2238 |
| `oracle_best_center` | 16.040232 | 36.485969 | 13.417517 | 2238 |

## Oracle Selection

| Model | Selected samples | Effective weight |
|---|---:|---:|
| `xr39_p10` | 856 | 762.000000 |
| `xr56b_p5` | 408 | 347.000000 |
| `xr58a_p10teacher` | 595 | 509.000000 |
| `xr62a_center` | 379 | 339.000000 |

## Baseline Miss Recovery

- baseline_model: `xr62a_center`
- baseline_p10_miss_count: `1486`
- p10_miss_recoverable_by_any_weight_pct: `3.426791`
- baseline_p5_miss_count: `1981`
- p5_miss_recoverable_by_any_weight_pct: `1.570681`

## Top Buckets By Oracle Gain

| Bucket | Samples | Baseline center | Oracle center | Gain px |
|---|---:|---:|---:|---:|
| `session:user42/right/session_202` | 25 | 21.585378 | 20.720778 | 0.864600 |
| `session:user46/right/session_201` | 33 | 17.064133 | 16.256189 | 0.807944 |
| `session:user47/left/session_202` | 28 | 15.828502 | 15.166941 | 0.661561 |
| `session:user42/right/session_102` | 27 | 16.207378 | 15.574134 | 0.633244 |
| `session:user44/right/session_201` | 50 | 15.286209 | 14.678436 | 0.607773 |
| `session:user44/right/session_102` | 27 | 11.590991 | 11.016398 | 0.574593 |
| `session:user45/right/session_201` | 31 | 24.602613 | 24.035175 | 0.567438 |
| `session:user40/left/session_201` | 47 | 16.899624 | 16.338046 | 0.561578 |
| `session:user39/left/session_102` | 14 | 21.093262 | 20.539540 | 0.553721 |
| `session:user42/right/session_201` | 46 | 23.543568 | 22.994632 | 0.548936 |
| `session:user45/left/session_102` | 28 | 16.897319 | 16.349185 | 0.548134 |
| `session:user41/left/session_201` | 46 | 19.356641 | 18.811876 | 0.544765 |
| `session:user40/right/session_102` | 28 | 10.579867 | 10.036384 | 0.543483 |
| `subject:user42` | 195 | 21.568916 | 21.038739 | 0.530177 |
| `session:user41/left/session_202` | 25 | 19.007891 | 18.481958 | 0.525933 |
| `session:user40/right/session_202` | 26 | 11.458232 | 10.951354 | 0.506878 |
| `session:user44/left/session_201` | 50 | 13.869314 | 13.364324 | 0.504990 |
| `session:user39/left/session_202` | 20 | 19.628893 | 19.125503 | 0.503390 |
| `subject:user40` | 204 | 14.418081 | 13.918347 | 0.499733 |
| `session:user40/right/session_201` | 47 | 14.774842 | 14.276568 | 0.498274 |

## Interpretation

- This is a no-train oracle, not deployable model performance.
- Positive signal means teacher-target construction has headroom only if a selector/pseudo-label rule can be trained without split leakage.
- Negative signal means another auxiliary-only head/loss replay is unlikely to recover P10/P5.
