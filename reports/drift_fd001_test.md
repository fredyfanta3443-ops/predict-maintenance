# Drift report: FD001 test (same conditions)

**Overall: OK** — Inputs look like the training data. Keep using the model.

- Fleet: 100 engines, 13,096 flights
- Sensors shifted (PSI > 0.2): **0 of 14**
- Fleet classifier AUC on healthy early life: **0.51** (OK; 0.5 = identical, 1.0 = completely different)

| Sensor | PSI (age-matched) | Training mean | Fleet mean | Status |
|---|---|---|---|---|
| s4 | 0.060 | 1406.258 | 1404.735 | OK |
| s11 | 0.060 | 47.460 | 47.416 | OK |
| s12 | 0.052 | 521.628 | 521.748 | OK |
| s15 | 0.048 | 8.432 | 8.426 | OK |
| s7 | 0.048 | 553.619 | 553.758 | OK |
| s21 | 0.047 | 23.320 | 23.336 | OK |
| s14 | 0.044 | 8140.266 | 8138.948 | OK |
| s20 | 0.043 | 38.867 | 38.893 | OK |
| s2 | 0.043 | 642.550 | 642.475 | OK |
| s13 | 0.038 | 2388.080 | 2388.071 | OK |
| s9 | 0.036 | 9060.432 | 9058.407 | OK |
| s8 | 0.033 | 2388.081 | 2388.071 | OK |
| s17 | 0.032 | 392.790 | 392.572 | OK |
| s3 | 0.031 | 1588.931 | 1588.099 | OK |

PSI: < 0.1 stable, 0.1–0.2 watch, > 0.2 shifted.
