"""Load NASA C-MAPSS files into pandas DataFrames."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# The raw files have no header. Column order comes from the dataset readme:
# engine id, cycle number, 3 operating settings, 21 sensors.
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS


def load_train(subset: str = "FD001") -> pd.DataFrame:
    """Full run-to-failure histories: each engine's last cycle is its failure."""
    return pd.read_csv(RAW_DIR / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)


def load_test(subset: str = "FD001") -> tuple[pd.DataFrame, pd.Series]:
    """Histories cut off before failure, plus the true RUL at each engine's last cycle."""
    test = pd.read_csv(RAW_DIR / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(RAW_DIR / f"RUL_{subset}.txt", header=None, names=["rul"])["rul"]
    rul.index = range(1, len(rul) + 1)  # index = engine unit id
    return test, rul
