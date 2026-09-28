"""Step 6b: nightly batch job. Score every engine, post work orders for the ones due.

Run:  uv run python src/batch.py                 # posts to the API
      uv run python src/batch.py --dry-run       # score only, post nothing

The fleet snapshot defaults to NASA's test_FD001: 100 engines part-way through
their lives, which is exactly what "today" looks like in production.
"""
import argparse
import os
from datetime import date
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

from data import RAW_DIR, COLUMNS
from features import build_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lightgbm.joblib"
REPORTS = ROOT / "reports"
API_URL = os.environ.get("WORKORDER_API", "http://127.0.0.1:8000")
# Step 5 found the cost cliff at k=8; 15 buys a safety margin for ~3% extra cost.
THRESHOLD = 15


def load_fleet(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def score_fleet(fleet: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Predict RUL at each engine's latest flight.

    Features come from the same build_features() used in training, over each
    engine's full history, so training and serving can't drift apart
    (including young engines with fewer flights than the 30-flight window).
    """
    feats = build_features(fleet, bundle["sensors"])
    latest = feats.groupby("unit").tail(1)
    pred = np.clip(bundle["model"].predict(latest[bundle["features"]]), 0, None)
    return pd.DataFrame({"unit": latest["unit"].to_numpy(), "last_cycle": latest["cycle"].to_numpy(),
                         "predicted_rul": pred.round(1)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", type=Path, default=RAW_DIR / "test_FD001.txt")
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    ap.add_argument("--run-date", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bundle = joblib.load(MODEL_PATH)
    model_version = f"lightgbm@{int(MODEL_PATH.stat().st_mtime)}"

    scores = score_fleet(load_fleet(args.fleet), bundle)
    scores["due"] = scores["predicted_rul"] < args.threshold
    scores["run_date"] = args.run_date

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"scores_{args.run_date}.csv"
    scores.to_csv(out, index=False)

    due = scores[scores["due"]].sort_values("predicted_rul")
    print(f"Scored {len(scores)} engines -> {out.name}")
    print(f"{len(due)} due for service (predicted flights left < {args.threshold})")

    if args.dry_run or due.empty:
        print(due.to_string(index=False) if not due.empty else "")
        return

    created = existing = 0
    with httpx.Client(base_url=API_URL, timeout=10) as client:
        for r in due.itertuples():
            resp = client.post("/work-orders", json={
                "unit": int(r.unit), "predicted_rul": float(r.predicted_rul), "last_cycle": int(r.last_cycle),
                "threshold": args.threshold, "model_version": model_version, "run_date": args.run_date,
            })
            resp.raise_for_status()
            created += resp.status_code == 201
            existing += resp.status_code == 200
            print(f"  engine {r.unit:>3}: {r.predicted_rul:5.1f} flights left -> order #{resp.json()['id']}"
                  f" ({'new' if resp.status_code == 201 else 'already open'})")
    print(f"Work orders: {created} created, {existing} already open")


if __name__ == "__main__":
    main()
