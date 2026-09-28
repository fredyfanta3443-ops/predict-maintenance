"""Step 2: label every engine-cycle with remaining useful life (RUL)."""
import pandas as pd

# Far from failure the sensors look healthy and flat, so the model can't tell
# 300 cycles left from 150. Anything above the cap is treated as "healthy".
RUL_CAP = 125


def add_train_rul(train: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Train engines run to failure: RUL = failure cycle - current cycle."""
    df = train.copy()
    failure_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = failure_cycle - df["cycle"]
    if cap is not None:
        df["rul"] = df["rul"].clip(upper=cap)
    return df


def add_test_rul(test: pd.DataFrame, rul_at_end: pd.Series, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Test engines are cut off early; NASA gives RUL at the last observed cycle.

    RUL at an earlier cycle = RUL at the end + cycles remaining until that end.
    """
    df = test.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = df["unit"].map(rul_at_end) + (last_cycle - df["cycle"])
    if cap is not None:
        df["rul"] = df["rul"].clip(upper=cap)
    return df
