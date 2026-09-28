"""Step 4: train RUL models, compare them, log everything to MLflow."""
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_test, load_train
from features import build_features, feature_columns, select_sensors
from labels import RUL_CAP, add_test_rul, add_train_rul

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
SEED = 42
N_FOLDS = 5


# ---------- metrics ----------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def phm08_score(y_true, y_pred) -> float:
    """NASA PHM08 score: lower is better. Late predictions (pred > true) cost more.

    Predicting 10 cycles too late costs e^(10/10)-1 = 1.72;
    predicting 10 cycles too early costs e^(10/13)-1 = 1.16.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# ---------- models ----------

def make_linear():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=SEED, verbose=-1,
)


def make_lgbm():
    return lgb.LGBMRegressor(**LGB_PARAMS)


MODELS = {"linear": make_linear, "lightgbm": make_lgbm}


# ---------- evaluation ----------

def cross_validate(make_model, X, y, groups, splitter):
    """Out-of-fold predictions: every row is predicted by a model that never saw it."""
    oof = np.zeros(len(y))
    for tr_idx, va_idx in splitter.split(X, y, groups):
        model = make_model().fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict(X.iloc[va_idx])
    return np.clip(oof, 0, None)


def main():
    train = add_train_rul(load_train("FD001"))
    test_raw, rul_end = load_test("FD001")
    test = add_test_rul(test_raw, rul_end)

    sensors = select_sensors(train)
    cols = feature_columns(sensors)
    train = build_features(train, sensors).reset_index(drop=True)
    test = build_features(test, sensors)
    # NASA's benchmark scores only each test engine's last observed cycle.
    test_last = test.groupby("unit").tail(1)

    X, y, groups = train[cols], train["rul"], train["unit"]

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("fd001-rul")

    oof_all = train[["unit", "cycle", "rul"]].copy()
    results = []
    for name, make_model in MODELS.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "rul_cap": RUL_CAP, "n_features": len(cols),
                               "sensors": ",".join(sensors), "cv": f"GroupKFold({N_FOLDS}) by unit"})
            if name == "lightgbm":
                mlflow.log_params(LGB_PARAMS)

            # Honest CV: whole engines held out.
            oof = cross_validate(make_model, X, y, groups, GroupKFold(N_FOLDS))
            oof_all[f"pred_{name}"] = oof

            # Final model on all 100 train engines, scored on NASA's unseen test engines.
            model = make_model().fit(X, y)
            pred_test = np.clip(model.predict(test_last[cols]), 0, None)

            metrics = {
                "cv_rmse": rmse(y, oof),
                "test_rmse": rmse(test_last["rul"], pred_test),
                "test_phm08": phm08_score(test_last["rul"], pred_test),
            }
            mlflow.log_metrics(metrics)
            MODELS_DIR.mkdir(exist_ok=True)
            joblib.dump({"model": model, "features": cols, "sensors": sensors}, MODELS_DIR / f"{name}.joblib")
            results.append({"model": name, **metrics})

    # Leakage demo: shuffle rows so the same engine lands in train and validation.
    with mlflow.start_run(run_name="lightgbm-LEAKY-row-split"):
        leaky = cross_validate(make_lgbm, X, y, groups, KFold(N_FOLDS, shuffle=True, random_state=SEED))
        mlflow.log_params({"model": "lightgbm", "cv": f"KFold({N_FOLDS}) by ROW (leaky, do not use)"})
        mlflow.log_metric("cv_rmse", rmse(y, leaky))
        results.append({"model": "lightgbm (LEAKY row split)", "cv_rmse": rmse(y, leaky)})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    oof_all.to_csv(PROCESSED_DIR / "oof_fd001.csv", index=False)

    print(pd.DataFrame(results).set_index("model").round(2).to_string())
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
        print("\nTop 10 LightGBM features:\n" + imp.head(10).to_string())


if __name__ == "__main__":
    main()
