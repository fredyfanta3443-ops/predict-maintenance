# Model Card: Engine Remaining-Useful-Life (RUL) Model

| | |
|---|---|
| **Model** | LightGBM regressor predicting flights left before failure, capped at 125 (`models/lightgbm.joblib`) |
| **Decision it drives** | Nightly: open a service work order when predicted flights left < **k = 15** |
| **Trained on** | NASA C-MAPSS FD001: 100 simulated turbofan engines, 20,631 flights, one operating condition, one fault mode (compressor) |
| **Status** | Prototype on simulated data. **Not validated on real client engines.** |

## Bottom line

1. **Condition-based maintenance saves money.** Moving from the fixed schedule to servicing on engine condition cuts total maintenance cost by **~37–39%** (95% interval 21–51%), with no increase in failures.
2. **This model is not yet proven better than a simple one-sensor rule.** Service when sensor s11 (5-flight average) exceeds ~47.96. The rule captures most of the saving, and the model's extra 3–9% is inside the noise.
3. **The model breaks when conditions change.** Deployed unchanged on engines with a new fault mode, it let **44 of 100 engines fail**. The one-sensor rule let 1 fail.

**Recommended rollout:** run the one-sensor rule for decisions, run this model in *shadow mode* alongside it, and switch only once the model beats the rule on the client's own engines.

## Intended use

- **For:** ranking which engines should be inspected or serviced next, as input to a maintenance planner.
- **Not for:** replacing mandated airworthiness inspections, making decisions without a human, or fleets whose operating conditions or fault types differ from training (see *When not to trust it*).

## How it works

1. **Label.** Each flight is labelled with flights left until failure. Anything above 125 is treated as 125, because early wear doesn't show in the sensors.
2. **Features (141).** From 14 sensors that actually change (7 flat sensors dropped), plus the flight count. For each sensor: average, variability and trend over the last 5, 10 and 30 flights, looking backwards only.
3. **Model.** LightGBM, compared against a linear baseline. Every run is logged in MLflow.
4. **Policy.** Service when predicted flights left < k. Costs have a cliff at k = 8, where engines start failing, so we run at **k = 15**: about 3% more cost for a safety margin.

## Performance

**Prediction error.** Every test engine was unseen during training. Splits are by engine, never by row.

| Trained on → tested on | Typical error (RMSE, flights) | PHM08 score (lower is better; late predictions penalised more) |
|---|---|---|
| Linear baseline, FD001 → FD001 | 15.9 | 376 |
| **LightGBM, FD001 → FD001** | **11.9** | **210** |
| LightGBM, FD003 → FD003 (retrained) | 12.3 | 264 |
| LightGBM, FD001 → FD003 (**not retrained**) | **23.8** | **2,435**; too late on 67% of engines |

A row-level split would have reported 6.5 flights of error. That figure is false: the same engine appears in both training and testing.

**Cost.** Per 1,000 flights, with an unplanned failure = 10 × a planned service. Each policy's setting is chosen on some engines and scored on others.

| Policy | FD001 cost | FD001 failures / 100 | FD003 cost | FD003 failures / 100 |
|---|---|---|---|---|
| Never service (run to failure) | 48.47 | 100 | — | — |
| Fixed schedule (today) | 8.66 | 1.2 | 7.66 | 1.2 |
| One-sensor rule (s11) | 6.02 | 1.7 | 4.82 | 1.4 |
| **LightGBM policy** | **5.48** | 1.0 | **4.67** | 1.1 |
| Perfect foresight (floor) | 4.87 | 0 | — | — |

**Savings of the LightGBM policy.** 95% bootstrap intervals over engines.

| Compared with | FD001 | FD003 |
|---|---|---|
| Fixed schedule | **36.7%** (21.1 to 49.1) ✅ | **39.1%** (23.7 to 51.1) ✅ |
| One-sensor rule (s11) | 8.9% (−16.2 to 27.5) ⚠️ | 3.3% (−18.6 to 22.4) ⚠️ |

**Sensitivity.** If a failure costs 3× to 50× a planned service instead of 10×, the saving against the fixed schedule stays between **32% and 40%**. The headline doesn't depend on the exact figure.

## When not to trust it

| Risk | What happens | Safeguard in place |
|---|---|---|
| **New operating conditions or fault types** | Predictions become late and dangerous (44/100 failures on FD003) | Nightly **drift report** (`src/drift.py`): FD001 test = OK, FD003 = ALERT (13/14 sensors shifted). On ALERT, fall back to the one-sensor rule and retrain |
| **Young engines** (fewer than 30 flights) | Trend features use fewer flights and are noisier | Same feature code in training and the nightly job, so no train-serve skew |
| **Engines far from failure** | Predicts about 125 ("healthy") and can't tell 150 from 300 flights left | Acceptable: service decisions happen below about 15 flights |
| **Censored labels after rollout** | Engines serviced on the model's advice never fail, so real failures stop being observed | Record **technician inspection findings** (wear found or not) at every service and use them as the new labels |

## Assumptions the client must confirm

1. **Cost ratio.** An unplanned failure, including downtime, costs about **10×** a planned service.
2. **Timing.** A work order raised overnight is actioned before the engine's next flight. Any slower and k must rise.
3. **Data.** The client's sensors match the 21 C-MAPSS channels, arrive at least once per flight, and can be labelled per engine.
4. **Relevance.** Results come from NASA simulations. Real engines will be noisier. **Every number above must be re-measured on the client's own history** before any saving is promised.

## Operations

- **Serving:** nightly batch (`src/batch.py`). It scores every engine and posts work orders to the FastAPI service (`src/api.py`), with one open order per engine and safe reruns. Online serving adds cost with no benefit, because sensor data arrives in batches.
- **Monitoring:** run the drift report nightly. Once inspection findings accumulate, track the error on real outcomes, which takes weeks to months.
- **Retraining triggers:** a drift ALERT, a new engine type or operating condition, or the monthly error on inspected engines rising above about 15 flights.
- **Reproduce:** `train.py` → `policy.py` → `generalize.py` → `drift.py` (all in `src/`). Experiments are in MLflow: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.
