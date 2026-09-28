"""Collect every result into one JSON file for the React dashboard.

Run after train.py, policy.py (nothing to save), generalize.py and batch.py:
    uv run python src/export_dashboard.py
Writes frontend/public/data.json.
"""
import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import pandas as pd

from data import load_test, load_train
from drift import drift_report
from data import COLUMNS, RAW_DIR
from features import select_sensors
from policy import FAILURE_COST, cost_per_1000, cv_outcomes, rate, run_all, saving, bootstrap_savings
from train import ROOT

OUT = ROOT / "frontend" / "public" / "data.json"
r2 = lambda x: round(float(x), 2)


def load_fd001_frame():
    oof = pd.read_csv(ROOT / "data" / "processed" / "oof_fd001.csv")
    raw = load_train("FD001")[["unit", "cycle", "s11"]]
    raw["s11_mean_5"] = raw.groupby("unit")["s11"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    return oof.merge(raw, on=["unit", "cycle"]).sort_values(["unit", "cycle"])


def engines_section(df):
    """Per-engine life story: true flights left vs what the model predicted (blind, out-of-fold)."""
    out = {}
    for unit, g in df.groupby("unit"):
        T = int(g["cycle"].max())
        out[int(unit)] = {
            "cycle": g["cycle"].astype(int).tolist(),
            "true": (T - g["cycle"]).astype(int).tolist(),
            "pred": g["pred_lightgbm"].round(1).tolist(),
            "s11": g["s11_mean_5"].round(3).tolist(),
        }
    return out


def cost_section(df, life):
    sweeps = run_all(df, life)
    honest = {n: cv_outcomes(sw) for n, sw in sweeps.items()}

    lgbm, lin = sweeps["Predictive (LightGBM)"], sweeps["Predictive (linear)"]
    rng = np.random.default_rng(42)
    n = len(lgbm["flown"].iloc[0])
    idx = rng.integers(0, n, (500, n))
    band = np.array([[cost_per_1000(r["flown"][i], r["failed"][i]) for i in idx] for _, r in lgbm.iterrows()])
    lo, hi = np.percentile(band, [2.5, 97.5], axis=1)

    curve = [{"k": int(k), "lgbm": r2(c), "lo": r2(l), "hi": r2(h), "linear": r2(cl),
              "failures": int(f), "avg_flights": r2(a)}
             for k, c, l, h, cl, f, a in zip(lgbm["knob"], lgbm["cost_per_1000"], lo, hi,
                                             lin["cost_per_1000"], lgbm["failures"], lgbm["avg_flights"])]

    levels = {n: {"cost": r2(rate(h["flown"], h["cost"])), "failures": r2(h["fail_rate"].sum()),
                  "avg_flights": r2(h["flown"].mean()), "knob": str(pd.Series(h["knobs"]).median())}
              for n, h in honest.items()}
    levels["Run to failure"] = {"cost": r2(cost_per_1000(life.to_numpy(float), np.ones(len(life), bool))),
                                "failures": 100, "avg_flights": r2(life.mean()), "knob": "-"}
    levels["Perfect foresight"] = {"cost": r2(cost_per_1000((life - 1).to_numpy(float), np.zeros(len(life), bool))),
                                   "failures": 0, "avg_flights": r2(life.mean() - 1), "knob": "-"}

    sens = []
    for fc in [3, 5, 10, 20, 50]:
        sw = run_all(df, life, fc)
        h = {k: cv_outcomes(sw[k], fc) for k in ["Predictive (LightGBM)", "Fixed schedule", "One-sensor rule (s11)"]}
        m = h["Predictive (LightGBM)"]
        sens.append({"failure_cost": fc, "vs_fixed": r2(saving(m, h["Fixed schedule"])),
                     "vs_sensor": r2(saving(m, h["One-sensor rule (s11)"]))})

    b = bootstrap_savings(honest["Predictive (LightGBM)"], honest["Fixed schedule"])
    headline = {"saving": r2(saving(honest["Predictive (LightGBM)"], honest["Fixed schedule"])),
                "lo": r2(np.percentile(b, 2.5)), "hi": r2(np.percentile(b, 97.5))}
    return curve, levels, sens, headline


def accuracy_step4():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    runs = mlflow.search_runs(experiment_names=["fd001-rul"], order_by=["start_time DESC"])
    rows, seen = [], set()
    for _, r in runs.iterrows():
        name = r["tags.mlflow.runName"]
        if name in seen:
            continue
        seen.add(name)
        rows.append({"model": name, "cv_rmse": r2(r["metrics.cv_rmse"]),
                     "test_rmse": None if pd.isna(r.get("metrics.test_rmse")) else r2(r["metrics.test_rmse"]),
                     "test_phm08": None if pd.isna(r.get("metrics.test_phm08")) else r2(r["metrics.test_phm08"])})
    return rows


def drift_section():
    ref = load_train("FD001")
    sensors = select_sensors(ref)
    out = {}
    for name, f in {"FD001 test (same conditions)": "test_FD001.txt",
                    "FD003 test (new fault mode)": "test_FD003.txt"}.items():
        cur = pd.read_csv(RAW_DIR / f, sep=r"\s+", header=None, names=COLUMNS)
        rep = drift_report(ref, cur, sensors)
        out[name] = {"overall": rep["overall"], "auc": r2(rep["auc"]), "n_alert": rep["n_alert"],
                     "sensors": [{"sensor": r.sensor, "psi": round(r.psi, 3), "status": r.status}
                                 for r in rep["per_sensor"].itertuples()]}
    return out


def fleet_section():
    files = sorted((ROOT / "reports").glob("scores_*.csv"))
    if not files:
        return None
    s = pd.read_csv(files[-1])
    _, rul = load_test("FD001")
    s["true_rul"] = s["unit"].map(rul)
    return {"run_date": str(s["run_date"].iloc[0]),
            "engines": [{"unit": int(r.unit), "last_cycle": int(r.last_cycle), "pred": float(r.predicted_rul),
                         "true": int(r.true_rul), "due": bool(r.due)} for r in s.itertuples()]}


def main():
    df = load_fd001_frame()
    life = df.groupby("unit")["cycle"].max()
    curve, levels, sens, headline = cost_section(df, life)
    data = {
        "failure_cost": FAILURE_COST,
        "deployed_k": 15,
        "headline": headline,
        "life": life.astype(int).tolist(),
        "engines": engines_section(df),
        "accuracy": accuracy_step4(),
        "cost_curve": curve,
        "policy_levels": levels,
        "sensitivity": sens,
        "generalize": json.loads((ROOT / "data" / "processed" / "generalize.json").read_text()),
        "drift": drift_section(),
        "fleet": fleet_section(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print("Headline:", headline)


if __name__ == "__main__":
    main()
