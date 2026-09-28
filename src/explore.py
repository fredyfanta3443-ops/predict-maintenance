"""Step 1: first look at FD001."""
from pathlib import Path

import matplotlib.pyplot as plt

from data import SENSOR_COLS, load_test, load_train

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"
OUT_DIR.mkdir(exist_ok=True)

train = load_train("FD001")
test, test_rul = load_test("FD001")

print(f"Train: {len(train):,} rows, {train['unit'].nunique()} engines")
print(f"Test:  {len(test):,} rows, {test['unit'].nunique()} engines\n")

life = train.groupby("unit")["cycle"].max()
print("Engine lifetime in cycles (train):")
print(life.describe().round(1).to_string(), "\n")

# A sensor whose std is ~0 never changes, so it can't tell us anything.
stds = train[SENSOR_COLS].std().sort_values()
print("Sensor standard deviation (lowest first):")
print(stds.round(4).to_string())

# Plot every sensor for 5 engines, aligned so x=0 is the failure point.
fig, axes = plt.subplots(7, 3, figsize=(15, 20), sharex=True)
for ax, col in zip(axes.flat, SENSOR_COLS):
    for unit in [1, 2, 3, 4, 5]:
        eng = train[train["unit"] == unit]
        ax.plot(eng["cycle"] - eng["cycle"].max(), eng[col], lw=0.8)
    ax.set_title(f"{col} (std={stds[col]:.3f})")
for ax in axes[-1]:
    ax.set_xlabel("cycles before failure")
fig.tight_layout()
fig.savefig(OUT_DIR / "fd001_sensors.png", dpi=80)
print(f"\nSaved plot: {OUT_DIR / 'fd001_sensors.png'}")
