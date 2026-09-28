"""Step 5: turn RUL predictions into a maintenance policy and price it.

Cost model (units = cost of one planned service):
  - planned service         = 1
  - unplanned failure       = FAILURE_COST (default 10; the client must confirm this)
Metric: cost per 1,000 flights. Servicing early looks cheap per engine but
wastes life, so the engine needs servicing more often; dividing by flights flown
captures that. Downtime is assumed to be folded into FAILURE_COST.

Timing: data for flight c arrives, the nightly job decides, and service happens
before flight c+1. An engine's last recorded flight T is the one it fails on,
so acting at any c < T is a planned service; not acting by T is a failure.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_train

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FAILURE_COST = 10.0
PLANNED_COST = 1.0
N_BOOT = 2000
SEED = 42


# ---------- per-engine outcomes: (flights flown, failed?) ----------

def act_when(signal: np.ndarray, cycles: np.ndarray, trigger) -> tuple[int, bool]:
    """Service at the first flight where trigger(signal) is true, if before failure."""
    T = cycles[-1]
    hits = np.flatnonzero(trigger(signal[:-1]))  # acting on the failure flight is too late
    if hits.size:
        return int(cycles[hits[0]]), False
    return int(T), True


def outcomes_predictive(df, pred_col, k):
    return [act_when(g[pred_col].to_numpy(), g["cycle"].to_numpy(), lambda p: p < k)
            for _, g in df.groupby("unit")]


def outcomes_fixed(life: pd.Series, interval: int):
    return [(interval, False) if T > interval else (int(T), True) for T in life]


def outcomes_sensor(df, col, threshold):
    return [act_when(g[col].to_numpy(), g["cycle"].to_numpy(), lambda s: s > threshold)
            for _, g in df.groupby("unit")]


def to_arrays(outcomes):
    flown = np.array([o[0] for o in outcomes], dtype=float)
    failed = np.array([o[1] for o in outcomes], dtype=bool)
    return flown, failed


def engine_costs(failed, failure_cost=FAILURE_COST):
    return np.where(failed, failure_cost, PLANNED_COST)


def cost_per_1000(flown, failed, failure_cost=FAILURE_COST):
    return rate(flown, engine_costs(failed, failure_cost))


def rate(flown, cost):
    return 1000 * cost.sum() / flown.sum()


# ---------- sweep each policy's knob, keep the best ----------

def sweep(name, knob_values, outcome_fn, failure_cost=FAILURE_COST):
    rows = []
    for v in knob_values:
        flown, failed = to_arrays(outcome_fn(v))
        rows.append({"policy": name, "knob": v, "cost_per_1000": cost_per_1000(flown, failed, failure_cost),
                     "failures": int(failed.sum()), "avg_flights": flown.mean(), "flown": flown, "failed": failed})
    return pd.DataFrame(rows)


def run_all(df, life, failure_cost=FAILURE_COST):
    ks = np.arange(1, 81)
    s11_grid = np.round(np.arange(47.3, 48.3, 0.01), 2)
    return {
        "Predictive (LightGBM)": sweep("lightgbm", ks, lambda k: outcomes_predictive(df, "pred_lightgbm", k), failure_cost),
        "Predictive (linear)": sweep("linear", ks, lambda k: outcomes_predictive(df, "pred_linear", k), failure_cost),
        "Fixed schedule": sweep("fixed", np.arange(50, 301), lambda n: outcomes_fixed(life, n), failure_cost),
        "One-sensor rule (s11)": sweep("sensor", s11_grid, lambda t: outcomes_sensor(df, "s11_mean_5", t), failure_cost),
    }


def best(sw):
    return sw.loc[sw["cost_per_1000"].idxmin()]


# ---------- honest tuning: pick the knob on some engines, score on others ----------

def cv_outcomes(sw, failure_cost=FAILURE_COST, n_folds=5, repeats=20, seed=SEED):
    """Per-engine expected (flights, cost, failure rate) when the knob is chosen WITHOUT that engine.

    Tuning a knob on the same engines you score it on finds the edge of a cliff
    (e.g. a fixed interval of 127 because the shortest engine lived 128).
    Repeated 5-fold over engines averages out the luck of one split.
    """
    flown = np.stack(sw["flown"].to_numpy())             # (n_knobs, n_engines)
    cost = engine_costs(np.stack(sw["failed"].to_numpy()), failure_cost)
    n = flown.shape[1]
    rng = np.random.default_rng(seed)
    acc_f, acc_c, acc_x = np.zeros(n), np.zeros(n), np.zeros(n)
    chosen = []
    for _ in range(repeats):
        folds = np.array_split(rng.permutation(n), n_folds)
        for held in folds:
            fit = np.setdiff1d(np.arange(n), held)
            k = np.argmin(cost[:, fit].sum(1) / flown[:, fit].sum(1))
            chosen.append(sw["knob"].iloc[k])
            acc_f[held] += flown[k, held]
            acc_c[held] += cost[k, held]
            acc_x[held] += cost[k, held] == failure_cost
    return {"flown": acc_f / repeats, "cost": acc_c / repeats, "fail_rate": acc_x / repeats,
            "knobs": np.array(chosen)}


def bootstrap_savings(a, b, n_boot=N_BOOT, seed=SEED):
    """% saving of policy a vs policy b, resampling engines with replacement."""
    rng = np.random.default_rng(seed)
    n = len(a["flown"])
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ca, cb = rate(a["flown"][idx], a["cost"][idx]), rate(b["flown"][idx], b["cost"][idx])
        out[i] = 100 * (cb - ca) / cb
    return out


def saving(a, b):
    ca, cb = rate(a["flown"], a["cost"]), rate(b["flown"], b["cost"])
    return 100 * (cb - ca) / cb


def main():
    oof = pd.read_csv(ROOT / "data" / "processed" / "oof_fd001.csv")
    raw = load_train("FD001")[["unit", "cycle", "s11"]]
    raw["s11_mean_5"] = raw.groupby("unit")["s11"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df = oof.merge(raw, on=["unit", "cycle"]).sort_values(["unit", "cycle"])
    life = df.groupby("unit")["cycle"].max()

    sweeps = run_all(df, life)
    bests = {name: best(sw) for name, sw in sweeps.items()}
    honest = {name: cv_outcomes(sw) for name, sw in sweeps.items()}

    run_to_fail = cost_per_1000(life.to_numpy(float), np.ones(len(life), bool))
    oracle = cost_per_1000((life - 1).to_numpy(float), np.zeros(len(life), bool))

    print(f"Cost assumption: failure = {FAILURE_COST:g} x planned service")
    print("Knob chosen on 80 engines, scored on the other 20 (5-fold x 20 repeats).\n")
    print(f"{'Policy':<24}{'knob (typical)':>15}{'cost/1000 flights':>19}{'failures /100':>15}{'avg flights':>13}")
    print(f"{'Run to failure':<24}{'-':>15}{run_to_fail:>19.2f}{100:>15.1f}{life.mean():>13.1f}")
    for name, h in honest.items():
        knob = f"{np.median(h['knobs']):g}"
        print(f"{name:<24}{knob:>15}{rate(h['flown'], h['cost']):>19.2f}"
              f"{h['fail_rate'].sum():>15.1f}{h['flown'].mean():>13.1f}")
    print(f"{'Perfect foresight':<24}{'-':>15}{oracle:>19.2f}{0:>15.1f}{life.mean() - 1:>13.1f}")

    model = honest["Predictive (LightGBM)"]
    print("\nLightGBM savings, 95% bootstrap interval over engines:")
    boots = {}
    for ref in ["Fixed schedule", "One-sensor rule (s11)"]:
        s = bootstrap_savings(model, honest[ref])
        boots[ref] = s
        lo, hi = np.percentile(s, [2.5, 97.5])
        print(f"  vs {ref:<22} {saving(model, honest[ref]):5.1f}%  [{lo:5.1f}%, {hi:5.1f}%]"
              f"   P(saving > 0) = {(s > 0).mean():.3f}")

    # Sensitivity: the 10x number is an assumption. How do savings move if it's wrong?
    print("\nSensitivity to the failure-cost assumption (each policy re-tuned honestly):")
    print(f"{'failure cost':>13}{'k':>6}{'vs fixed':>10}{'vs sensor':>11}{'model failures/100':>20}")
    for fc in [3, 5, 10, 20, 50]:
        sw = run_all(df, life, fc)
        h = {n: cv_outcomes(sw[n], fc) for n in ["Predictive (LightGBM)", "Fixed schedule", "One-sensor rule (s11)"]}
        m = h["Predictive (LightGBM)"]
        print(f"{fc:>12}x{np.median(m['knobs']):>6g}{saving(m, h['Fixed schedule']):>9.1f}%"
              f"{saving(m, h['One-sensor rule (s11)']):>10.1f}%{m['fail_rate'].sum():>20.1f}")

    plot(sweeps, bests, boots, oracle, honest)


def plot(sweeps, bests, boots, oracle, honest):
    REPORTS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: cost curve by threshold k, with a bootstrap band.
    lgbm = sweeps["Predictive (LightGBM)"]
    rng = np.random.default_rng(SEED)
    n = len(lgbm["flown"].iloc[0])
    idx = rng.integers(0, n, (500, n))
    band = np.array([[cost_per_1000(r["flown"][i], r["failed"][i]) for i in idx] for _, r in lgbm.iterrows()])
    lo, hi = np.percentile(band, [2.5, 97.5], axis=1)

    ax1.fill_between(lgbm["knob"], lo, hi, color="C0", alpha=0.2, label="LightGBM 95% interval")
    ax1.plot(lgbm["knob"], lgbm["cost_per_1000"], color="C0", lw=2, label="Predictive (LightGBM)")
    lin = sweeps["Predictive (linear)"]
    ax1.plot(lin["knob"], lin["cost_per_1000"], color="C1", lw=1.2, ls="--", label="Predictive (linear)")
    for name, color in [("Fixed schedule", "C3"), ("One-sensor rule (s11)", "C2")]:
        h = honest[name]
        ax1.axhline(rate(h["flown"], h["cost"]), color=color, ls=":", lw=1.5, label=f"{name} (honestly tuned)")
    ax1.axhline(oracle, color="grey", lw=1, label="Perfect foresight")
    h = honest["Predictive (LightGBM)"]
    ax1.scatter([np.median(h["knobs"])], [rate(h["flown"], h["cost"])], color="C0", marker="D", zorder=5,
                label="LightGBM (honestly tuned)")
    ax1.axvspan(0, 8, color="red", alpha=0.06)
    ax1.text(1, oracle * 0.95, "k too low: engines fail", fontsize=8, color="darkred")
    ax1.set_ylim(oracle * 0.9, rate(honest["Fixed schedule"]["flown"], honest["Fixed schedule"]["cost"]) * 1.4)
    ax1.set_xlabel("Service when predicted flights left < k")
    ax1.set_ylabel(f"Cost per 1,000 flights (planned service = 1, failure = {FAILURE_COST:g})")
    ax1.set_title("Total maintenance cost by threshold")
    ax1.legend(fontsize=8, loc="upper right")

    # Right: bootstrap distribution of % savings.
    for (name, s), color in zip(boots.items(), ["C3", "C2"]):
        ax2.hist(s, bins=50, alpha=0.5, color=color, label=f"vs {name.lower()}")
        ax2.axvline(np.median(s), color=color, lw=1.5)
    ax2.axvline(0, color="black", lw=1)
    ax2.set_xlabel("% cost saving of LightGBM policy")
    ax2.set_ylabel("bootstrap resamples")
    ax2.set_title("How sure are we? (2,000 resamples of engines)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(REPORTS / "cost_curve.png", dpi=100)
    print(f"\nSaved {REPORTS / 'cost_curve.png'}")


if __name__ == "__main__":
    main()
