"""Step 6c: drift report. Does today's fleet still look like the training data?

Real failure labels take weeks to months to arrive, so input drift is the early
warning that the model may be wrong. Two checks, both comparing like with like:

1. Age-matched PSI per sensor. A young fleet naturally reads "healthier" than
   the run-to-failure training data, so the reference is reweighted to the
   current fleet's age mix first. Without this, even FD001's own test engines
   look drifted (max PSI 0.35 -> 0.06 after matching).
2. Fleet classifier on healthy early life (first 30 flights). Train a model to
   tell reference engines from current engines. AUC ~0.5 = indistinguishable;
   high AUC = the fleets differ in a way no single sensor may show. Split by
   engine, so it can't just memorise engines.

Run:  uv run python src/drift.py                        # demo: FD001 test vs FD003 test
      uv run python src/drift.py --fleet path/to/fleet.txt --name tonight
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from data import COLUMNS, RAW_DIR, load_train
from features import select_sensors

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

# Industry rule of thumb for PSI: < 0.1 stable, 0.1-0.2 watch, > 0.2 shifted.
PSI_WATCH, PSI_ALERT = 0.1, 0.2
AUC_WATCH, AUC_ALERT = 0.6, 0.7
AGE_BAND = 25       # flights per age band when matching age mix
EARLY_LIFE = 30     # flights treated as "healthy baseline"


def psi(ref: np.ndarray, cur: np.ndarray, ref_weights: np.ndarray | None = None, bins: int = 10) -> float:
    """Population Stability Index, bins set by reference deciles."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, edges, weights=ref_weights)[0].astype(float)
    p_cur = np.histogram(cur, edges)[0].astype(float)
    p_ref, p_cur = np.clip(p_ref / p_ref.sum(), 1e-4, None), np.clip(p_cur / p_cur.sum(), 1e-4, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def age_weights(ref: pd.DataFrame, cur: pd.DataFrame) -> np.ndarray:
    """Reweight reference rows so their age (cycle) mix matches the current fleet."""
    band = lambda d: (d["cycle"] // AGE_BAND).clip(upper=12)
    rb, cb = band(ref), band(cur)
    ratio = cb.value_counts(normalize=True) / rb.value_counts(normalize=True)
    return rb.map(ratio).fillna(0).to_numpy()


def fleet_classifier_auc(ref: pd.DataFrame, cur: pd.DataFrame, sensors: list[str]) -> float:
    r, c = ref[ref["cycle"] <= EARLY_LIFE], cur[cur["cycle"] <= EARLY_LIFE]
    X = pd.concat([r[sensors], c[sensors]])
    y = np.r_[np.zeros(len(r)), np.ones(len(c))]
    groups = np.r_[r["unit"].to_numpy(), c["unit"].to_numpy() + 100_000]
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    p = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def level(value, watch, alert):
    return "ALERT" if value > alert else "WATCH" if value > watch else "OK"


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, sensors: list[str]) -> dict:
    w = age_weights(ref, cur)
    per_sensor = pd.DataFrame({
        "sensor": sensors,
        "psi": [psi(ref[s].to_numpy(), cur[s].to_numpy(), w) for s in sensors],
        "ref_mean": [np.average(ref[s], weights=w) for s in sensors],
        "cur_mean": [cur[s].mean() for s in sensors],
    })
    per_sensor["status"] = [level(v, PSI_WATCH, PSI_ALERT) for v in per_sensor["psi"]]
    per_sensor = per_sensor.sort_values("psi", ascending=False).reset_index(drop=True)
    auc = fleet_classifier_auc(ref, cur, sensors)
    n_alert = int((per_sensor["status"] == "ALERT").sum())
    overall = "ALERT" if n_alert >= 2 or auc > AUC_ALERT else \
              "WATCH" if n_alert or (per_sensor["status"] == "WATCH").any() or auc > AUC_WATCH else "OK"
    return {"per_sensor": per_sensor, "auc": auc, "overall": overall, "n_alert": n_alert,
            "n_engines": cur["unit"].nunique(), "n_rows": len(cur)}


ACTIONS = {
    "OK": "Inputs look like the training data. Keep using the model.",
    "WATCH": "Some movement. Keep using the model, check again tomorrow, and ask the fleet team if anything changed.",
    "ALERT": ("Fleet no longer looks like the training data; RUL predictions are not trustworthy. "
              "Fall back to the one-sensor rule, collect inspection findings, and retrain before trusting the model again."),
}


def write_markdown(name: str, rep: dict, path: Path):
    ps = rep["per_sensor"]
    lines = [
        f"# Drift report: {name}",
        "",
        f"**Overall: {rep['overall']}** — {ACTIONS[rep['overall']]}",
        "",
        f"- Fleet: {rep['n_engines']} engines, {rep['n_rows']:,} flights",
        f"- Sensors shifted (PSI > {PSI_ALERT}): **{rep['n_alert']} of {len(ps)}**",
        f"- Fleet classifier AUC on healthy early life: **{rep['auc']:.2f}** "
        f"({level(rep['auc'], AUC_WATCH, AUC_ALERT)}; 0.5 = identical, 1.0 = completely different)",
        "",
        "| Sensor | PSI (age-matched) | Training mean | Fleet mean | Status |",
        "|---|---|---|---|---|",
        *[f"| {r.sensor} | {r.psi:.3f} | {r.ref_mean:.3f} | {r.cur_mean:.3f} | {r.status} |" for r in ps.itertuples()],
        "",
        f"PSI: < {PSI_WATCH} stable, {PSI_WATCH}–{PSI_ALERT} watch, > {PSI_ALERT} shifted.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot(reports: dict, path: Path):
    fig, axes = plt.subplots(1, len(reports), figsize=(6 * len(reports), 5), sharey=True, squeeze=False)
    colors = {"OK": "C2", "WATCH": "C1", "ALERT": "C3"}
    top = max(r["per_sensor"]["psi"].max() for r in reports.values()) * 1.1
    for ax, (name, rep) in zip(axes[0], reports.items()):
        ps = rep["per_sensor"].sort_values("sensor", key=lambda s: s.str[1:].astype(int))
        ax.bar(ps["sensor"], ps["psi"], color=[colors[s] for s in ps["status"]])
        ax.axhline(PSI_WATCH, color="C1", ls=":", lw=1)
        ax.axhline(PSI_ALERT, color="C3", ls="--", lw=1, label=f"alert ({PSI_ALERT})")
        ax.set_ylim(0, max(top, 0.3))
        ax.set_title(f"{name}\nOverall {rep['overall']} | fleet classifier AUC {rep['auc']:.2f}")
        ax.set_xlabel("sensor")
        ax.tick_params(axis="x", rotation=45)
    axes[0][0].set_ylabel("PSI vs FD001 training data (age-matched)")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", type=Path, help="fleet sensor file (C-MAPSS format); omit for the demo")
    ap.add_argument("--name", default="fleet")
    args = ap.parse_args()

    ref = load_train("FD001")
    sensors = select_sensors(ref)
    fleets = ({args.name: args.fleet} if args.fleet else
              {"FD001 test (same conditions)": RAW_DIR / "test_FD001.txt",
               "FD003 test (new fault mode)": RAW_DIR / "test_FD003.txt"})

    REPORTS.mkdir(exist_ok=True)
    reports = {}
    for name, path in fleets.items():
        cur = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
        rep = drift_report(ref, cur, sensors)
        reports[name] = rep
        slug = name.split()[0].lower() + "_" + name.split()[1].lower() if not args.fleet else args.name
        md = REPORTS / f"drift_{slug}.md"
        write_markdown(name, rep, md)
        print(f"{name:<32} overall {rep['overall']:<6} sensors shifted {rep['n_alert']:>2}/{len(sensors)}"
              f"   max PSI {rep['per_sensor']['psi'].max():.3f}   classifier AUC {rep['auc']:.2f}   -> {md.name}")
    plot(reports, REPORTS / "drift.png")
    print(f"Saved {REPORTS / 'drift.png'}")


if __name__ == "__main__":
    main()
