"""Step 6a: does the approach survive a change in conditions? FD001 vs FD003.

FD001: one operating condition, one fault mode (HPC degradation).
FD003: one operating condition, TWO fault modes (HPC or fan degradation).

Three questions:
  1. Shift: a model trained on FD001 and deployed unchanged on FD003 - how bad?
  2. Retrain: trained on FD003 itself - does it recover?
  3. Money: on each dataset, does ML beat the fixed schedule AND the best one-sensor rule?
"""
import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from batch import THRESHOLD as DEPLOYED_K
from data import load_test, load_train
from features import build_features, feature_columns, select_sensors
from labels import add_test_rul, add_train_rul
from policy import (bootstrap_savings, cv_outcomes, outcomes_fixed, outcomes_predictive, rate, saving, sweep,
                    to_arrays)
from train import N_FOLDS, ROOT, cross_validate, make_lgbm, phm08_score, rmse

REPORTS = ROOT / "reports"


def prepare(subset):
    """Labelled train/test for one subset, raw RUL kept for the policy simulation."""
    train = add_train_rul(load_train(subset))
    test_raw, rul_end = load_test(subset)
    test = add_test_rul(test_raw, rul_end)
    return train, test


def sensor_rule_sweep(df, sensors, n_thresholds=120):
    """Every 'service when smoothed sensor X crosses T' rule, both directions.

    Fast version of policy.outcomes_sensor: the first flight a sensor exceeds T is
    where its running max first exceeds T, so one searchsorted per engine.
    """
    groups = [g for _, g in df.groupby("unit")]
    life = np.array([g["cycle"].iloc[-1] for g in groups], dtype=float)
    rows = []
    for s in sensors:
        col = f"{s}_mean_5"
        for direction, sign in [("up", 1), ("down", -1)]:
            # running extreme over all flights except the failure flight (acting then is too late)
            runs = [np.maximum.accumulate(sign * g[col].to_numpy()[:-1]) for g in groups]
            cycles = [g["cycle"].to_numpy() for g in groups]
            lo, hi = np.quantile(np.concatenate(runs), [0.3, 0.9999])
            for t in np.linspace(lo, hi, n_thresholds):
                idx = [np.searchsorted(r, t, side="right") for r in runs]
                hit = np.array([i < len(r) for i, r in zip(idx, runs)])
                flown = np.where(hit, [c[min(i, len(c) - 1)] for i, c in zip(idx, cycles)], life).astype(float)
                rows.append({"policy": "sensor", "knob": f"{s} {direction} {sign * t:.3f}",
                             "flown": flown, "failed": ~hit})
    out = pd.DataFrame(rows)
    out["cost_per_1000"] = [rate(f, np.where(x, 10.0, 1.0)) for f, x in zip(out["flown"], out["failed"])]
    return out


def policy_table(df, pred_cols, sensors):
    """Honestly tuned cost of each policy on run-to-failure engines in df."""
    life = df.groupby("unit")["cycle"].max()
    ks = np.arange(1, 81)
    sweeps = {label: sweep(col, ks, lambda k, c=col: outcomes_predictive(df, c, k)) for label, col in pred_cols.items()}
    sweeps["Fixed schedule"] = sweep("fixed", np.arange(50, 401), lambda n: outcomes_fixed(life, n))
    all_sensors = sensor_rule_sweep(df, sensors)
    sweeps["One-sensor rule (s11 up)"] = all_sensors[all_sensors["knob"].str.startswith("s11 up")].reset_index(drop=True)
    sweeps["Best one-sensor rule"] = all_sensors
    return {name: cv_outcomes(sw) for name, sw in sweeps.items()}


def at_fixed_k(df, col, k):
    """What happens if we deploy with k fixed in advance (no re-tuning on this fleet)."""
    flown, failed = to_arrays(outcomes_predictive(df, col, k))
    return {"flown": flown, "cost": np.where(failed, 10.0, 1.0), "fail_rate": failed.astype(float)}


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("fd003-generalization")

    data = {s: prepare(s) for s in ["FD001", "FD003"]}
    sensors = {s: select_sensors(data[s][0]) for s in data}
    feats = {s: (build_features(tr, sensors[s]).reset_index(drop=True), build_features(te, sensors[s]))
             for s, (tr, te) in data.items()}
    # FD001's model only knows FD001's sensors, so FD003 also needs features built with that list.
    fd003_as_fd001 = (build_features(data["FD003"][0], sensors["FD001"]).reset_index(drop=True),
                      build_features(data["FD003"][1], sensors["FD001"]))

    # ---------- 1 & 2: prediction accuracy ----------
    print("Prediction error (flights). Test = each NASA test engine's last observed flight.\n")
    acc_rows, preds = [], {}
    models = {}
    for s in ["FD001", "FD003"]:
        tr, te = feats[s]
        cols = feature_columns(sensors[s])
        oof = cross_validate(make_lgbm, tr[cols], tr["rul"], tr["unit"], GroupKFold(N_FOLDS))
        models[s] = make_lgbm().fit(tr[cols], tr["rul"])
        last = te.groupby("unit").tail(1)
        p = np.clip(models[s].predict(last[cols]), 0, None)
        preds[(s, s)] = oof
        acc_rows.append({"trained on": s, "tested on": s, "cv_rmse": rmse(tr["rul"], oof),
                         "test_rmse": rmse(last["rul"], p), "test_phm08": phm08_score(last["rul"], p)})

    cols01 = feature_columns(sensors["FD001"])
    tr3, te3 = fd003_as_fd001
    last3 = te3.groupby("unit").tail(1)
    p_shift_test = np.clip(models["FD001"].predict(last3[cols01]), 0, None)
    p_shift_train = np.clip(models["FD001"].predict(tr3[cols01]), 0, None)
    preds[("FD001", "FD003")] = p_shift_train
    acc_rows.insert(1, {"trained on": "FD001", "tested on": "FD003 (shift)", "cv_rmse": rmse(tr3["rul"], p_shift_train),
                        "test_rmse": rmse(last3["rul"], p_shift_test),
                        "test_phm08": phm08_score(last3["rul"], p_shift_test)})
    acc = pd.DataFrame(acc_rows)
    print(acc.round(2).to_string(index=False))
    print("(cv_rmse for the shift row = error on all FD003 train rows; none were seen in training)")

    late = (p_shift_test - last3["rul"].to_numpy())
    print(f"\nShifted model on FD003 test: predicts too LATE (dangerous) on {(late > 0).mean():.0%} of engines, "
          f"by >20 flights on {(late > 20).mean():.0%}")

    for r in acc_rows:
        with mlflow.start_run(run_name=f"lgbm {r['trained on']} -> {r['tested on']}"):
            mlflow.log_params({"trained_on": r["trained on"], "tested_on": r["tested_on"] if "tested_on" in r else r["tested on"]})
            mlflow.log_metrics({k: float(v) for k, v in r.items() if k.endswith(("rmse", "phm08"))})

    summary = {"accuracy": acc.round(3).to_dict("records"),
               "late_share": float((late > 0).mean()), "late_20_share": float((late > 20).mean()),
               "policies": {}, "as_is": [], "savings": {}}

    # ---------- 3: money ----------
    results = {}
    for s in ["FD001", "FD003"]:
        tr = feats[s][0]
        df = tr[["unit", "cycle"] + [f"{x}_mean_5" for x in sensors[s]]].copy()
        df["pred_lightgbm"] = preds[(s, s)]
        pred_cols = {"LightGBM (trained on same data)": "pred_lightgbm"}
        if s == "FD003":
            df["pred_shift"] = preds[("FD001", "FD003")]
            pred_cols["LightGBM (FD001 model, k re-tuned on FD003)"] = "pred_shift"
        results[s] = (df, policy_table(df, pred_cols, sensors[s]))

    for s, (df, honest) in results.items():
        print(f"\n===== {s}: cost per 1,000 flights (failure = 10x planned), knob tuned honestly =====")
        print(f"{'Policy':<40}{'typical knob':>18}{'cost':>8}{'failures/100':>14}")
        summary["policies"][s] = []
        for name, h in honest.items():
            knob = pd.Series(h["knobs"]).mode().iloc[0]
            print(f"{name:<40}{str(knob):>18}{rate(h['flown'], h['cost']):>8.2f}{h['fail_rate'].sum():>14.1f}")
            summary["policies"][s].append({"policy": name, "knob": str(knob), "cost": rate(h["flown"], h["cost"]),
                                           "failures": float(h["fail_rate"].sum()), "avg_flights": float(h["flown"].mean())})
        if s == "FD003":
            # Carry every policy over from FD001 unchanged: what a client would actually run on day one.
            fd001 = results["FD001"][1]
            s11_t = float(pd.Series(fd001["One-sensor rule (s11 up)"]["knobs"]).mode().iloc[0].split()[-1])
            fixed_n = int(pd.Series(fd001["Fixed schedule"]["knobs"]).mode().iloc[0])
            life = df.groupby("unit")["cycle"].max()
            as_is = {
                f"FD001 model as-is, k={DEPLOYED_K}": at_fixed_k(df, "pred_shift", DEPLOYED_K),
                f"FD001 s11 rule as-is, >{s11_t:.2f}": dict(zip(["flown", "failed"], to_arrays(
                    outcomes_predictive(df.assign(neg=-df["s11_mean_5"]), "neg", -s11_t)))),
                f"FD001 fixed schedule as-is, {fixed_n}": dict(zip(["flown", "failed"], to_arrays(
                    outcomes_fixed(life, fixed_n)))),
            }
            print(f"  -- deployed unchanged from FD001 (no FD003 failure data used) --")
            for name, d in as_is.items():
                failed = d.get("failed", d.get("fail_rate", 0) > 0)
                cost = rate(d['flown'], np.where(failed, 10.0, 1.0))
                print(f"{name:<40}{'':>18}{cost:>8.2f}{failed.sum():>14.1f}")
                summary["as_is"].append({"policy": name, "cost": cost, "failures": float(failed.sum())})

        model = honest["LightGBM (trained on same data)"]
        print(f"\n  LightGBM (trained on {s}) savings, 95% bootstrap interval:")
        for ref in ["Fixed schedule", "One-sensor rule (s11 up)", "Best one-sensor rule"]:
            b = bootstrap_savings(model, honest[ref])
            lo, hi = np.percentile(b, [2.5, 97.5])
            print(f"    vs {ref:<26}{saving(model, honest[ref]):6.1f}%  [{lo:6.1f}%, {hi:6.1f}%]  P(>0) = {(b > 0).mean():.2f}")
            summary["savings"].setdefault(s, []).append({"vs": ref, "saving": saving(model, honest[ref]),
                                                          "lo": float(lo), "hi": float(hi), "p_positive": float((b > 0).mean())})

    out = ROOT / "data" / "processed" / "generalize.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nSaved {out}")
    plot(results, acc)


def plot(results, acc):
    REPORTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = ["Fixed schedule", "One-sensor rule (s11 up)", "Best one-sensor rule",
             "LightGBM (FD001 model, k re-tuned on FD003)", "LightGBM (trained on same data)"]
    colors = ["C3", "C2", "C8", "C7", "C0"]
    for ax, (s, (_, honest)) in zip(axes, results.items()):
        names = [n for n in order if n in honest]
        vals = [rate(honest[n]["flown"], honest[n]["cost"]) for n in names]
        fails = [honest[n]["fail_rate"].sum() for n in names]
        bars = ax.barh(names, vals, color=[colors[order.index(n)] for n in names])
        for b, v, f in zip(bars, vals, fails):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.2f}  ({f:.1f} failures/100)", va="center", fontsize=8)
        ax.set_xlim(0, max(vals) * 1.75)
        ax.set_title(f"{s}: cost per 1,000 flights (lower is better)")
        ax.invert_yaxis()
    fig.suptitle("Honestly tuned maintenance policies, failure = 10x planned service")
    fig.tight_layout()
    fig.savefig(REPORTS / "fd003_generalization.png", dpi=100)
    print(f"\nSaved {REPORTS / 'fd003_generalization.png'}")


if __name__ == "__main__":
    main()
