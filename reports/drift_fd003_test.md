# Drift report: FD003 test (new fault mode)

**Overall: ALERT** — Fleet no longer looks like the training data; RUL predictions are not trustworthy. Fall back to the one-sensor rule, collect inspection findings, and retrain before trusting the model again.

- Fleet: 100 engines, 16,596 flights
- Sensors shifted (PSI > 0.2): **13 of 14**
- Fleet classifier AUC on healthy early life: **0.75** (ALERT; 0.5 = identical, 1.0 = completely different)

| Sensor | PSI (age-matched) | Training mean | Fleet mean | Status |
|---|---|---|---|---|
| s15 | 1.212 | 8.440 | 8.395 | ALERT |
| s4 | 0.942 | 1408.502 | 1400.568 | ALERT |
| s11 | 0.922 | 47.528 | 47.291 | ALERT |
| s21 | 0.914 | 23.295 | 23.399 | ALERT |
| s20 | 0.896 | 38.825 | 38.997 | ALERT |
| s2 | 0.718 | 642.659 | 642.268 | ALERT |
| s17 | 0.688 | 393.137 | 391.900 | ALERT |
| s3 | 0.606 | 1590.242 | 1585.608 | ALERT |
| s8 | 0.511 | 2388.094 | 2388.021 | ALERT |
| s13 | 0.492 | 2388.094 | 2388.021 | ALERT |
| s7 | 0.482 | 553.407 | 554.555 | ALERT |
| s12 | 0.471 | 521.449 | 522.467 | ALERT |
| s9 | 0.222 | 9064.123 | 9056.274 | ALERT |
| s14 | 0.141 | 8142.887 | 8138.327 | WATCH |

PSI: < 0.1 stable, 0.1–0.2 watch, > 0.2 shifted.
