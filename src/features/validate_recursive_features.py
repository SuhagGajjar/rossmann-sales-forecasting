
from pathlib import Path

import numpy as np
import pandas as pd

from recursive_forecasting import RecursiveFeatureBuilder


ROOT = Path(__file__).resolve().parents[2]

history = pd.read_pickle(
    ROOT / "data/processed/train_preprocessed.pkl"
)

validation = pd.read_pickle(
    ROOT / "data/processed/validation_base.pkl"
)

history["Date"] = pd.to_datetime(history["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])

# Strict validation forecast boundary.
cutoff = pd.Timestamp("2015-06-19")

history = history.loc[
    history["Date"] <= cutoff
].copy()

first_day = validation.loc[
    validation["Date"] == pd.Timestamp("2015-06-20")
].copy()

builder = RecursiveFeatureBuilder(history)

generated = builder.build_day_features(first_day)

# Independent pandas reference calculation.
combined = pd.concat(
    [
        history[["Store", "Date", "DayOfWeek", "Sales"]],
        first_day[["Store", "Date", "DayOfWeek"]].assign(Sales=np.nan),
    ],
    ignore_index=True,
).sort_values(["Store", "Date"])

grouped = combined.groupby("Store")["Sales"]

expected = pd.DataFrame(index=combined.index)

for lag in [1, 7, 14, 28]:
    expected[f"Sales_Lag_{lag}"] = grouped.shift(lag)

for window in [7, 14, 28]:
    expected[f"Sales_RollingMean_{window}"] = grouped.transform(
        lambda s: s.shift(1).rolling(
            window,
            min_periods=1
        ).mean()
    )

expected["Store_HistoricalMean"] = grouped.transform(
    lambda s: s.shift(1).expanding(min_periods=1).mean()
)

expected["Store_Weekday_HistoricalMean"] = (
    combined.groupby(["Store", "DayOfWeek"])["Sales"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
)

expected = expected.loc[
    combined["Date"] == pd.Timestamp("2015-06-20")
]

expected.index = first_day.index

print("Validation date:", first_day["Date"].iloc[0])
print("Stores checked:", len(first_day))

all_passed = True

for column in generated.columns:

    actual = generated[column]
    reference = expected[column]

    passed = np.allclose(
        actual.to_numpy(dtype=float),
        reference.to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-8,
        equal_nan=True
    )

    print(column, "PASS" if passed else "FAIL")
    all_passed = all_passed and passed

print("\nOverall:", "PASS" if all_passed else "FAIL")

if not all_passed:
    raise AssertionError(
        "Recursive feature generation differs from reference."
    )
