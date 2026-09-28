"""Step 3: turn raw sensor readings into model inputs."""
import numpy as np
import pandas as pd

from data import SENSOR_COLS

WINDOWS = (5, 10, 30)
# Sensors with less spread than this are flat lines and carry no signal.
MIN_STD = 0.01


def select_sensors(train: pd.DataFrame, min_std: float = MIN_STD) -> list[str]:
    """Pick sensors that actually move. Decided on training data only."""
    stds = train[SENSOR_COLS].std()
    return [c for c in SENSOR_COLS if stds[c] > min_std]


def _rolling_slope(x: pd.Series, window: int) -> pd.Series:
    """Least-squares slope of x over the last `window` cycles (per-cycle change).

    Uses running sums so it's fast, and works for windows shorter than `window`
    (young engines) instead of returning NaN.
    """
    t = pd.Series(np.arange(len(x), dtype=float), index=x.index)
    roll = lambda s: s.rolling(window, min_periods=1).sum()
    n = x.rolling(window, min_periods=1).count()
    st, sx, stt, stx = roll(t), roll(x), roll(t * t), roll(t * x)
    denom = n * stt - st**2
    slope = (n * stx - st * sx) / denom.replace(0, np.nan)
    return slope.fillna(0.0)  # a single point has no trend


def build_features(df: pd.DataFrame, sensors: list[str], windows=WINDOWS) -> pd.DataFrame:
    """Add rolling mean / std / slope per sensor, computed within each engine.

    Rolling windows only look backwards, so a row never sees its own future.
    min_periods=1 means an engine with 3 cycles still gets features, exactly
    as it would in the nightly job.
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit", group_keys=False)
    new_cols = {}
    for s in sensors:
        for w in windows:
            new_cols[f"{s}_mean_{w}"] = g[s].transform(lambda x: x.rolling(w, min_periods=1).mean())
            new_cols[f"{s}_std_{w}"] = g[s].transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0.0)
            new_cols[f"{s}_slope_{w}"] = g[s].transform(lambda x: _rolling_slope(x, w))
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def feature_columns(sensors: list[str], windows=WINDOWS) -> list[str]:
    """The exact input columns the model sees, in a fixed order."""
    cols = ["cycle"] + list(sensors)
    for s in sensors:
        for w in windows:
            cols += [f"{s}_mean_{w}", f"{s}_std_{w}", f"{s}_slope_{w}"]
    return cols
