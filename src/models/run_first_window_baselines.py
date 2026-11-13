from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)

ROOT = Path(__file__).resolve().parents[2]

from src.features.recursive_forecasting import RecursiveFeatureBuilder
from src.models.recursive_forecaster import recursive_forecast
from src.models.baselines import LagBaseline, MovingAverageBaseline


# ---------------------------------------------------------
# Configuration: first 42-day forecast window
# ---------------------------------------------------------
CUTOFF_DATE = pd.Timestamp("2015-05-08")
VALIDATION_START = pd.Timestamp("2015-05-09")
VALIDATION_END = pd.Timestamp("2015-06-19")

EXPECTED_DAYS = 42
EXPECTED_ROWS = 46_830

# Load the original preprocessed training data.
train = pd.read_pickle(
    ROOT / "data" / "processed" / "train_preprocessed.pkl"
)

train["Date"] = pd.to_datetime(train["Date"])

# Strict forecast origin: history contains actual observations
# only through 8 May 2015.
history = train.loc[
    train["Date"] <= CUTOFF_DATE
].copy()

# First-window rows are taken from the preprocessed training data.
# Their actual Sales values are retained only for evaluation.
validation = train.loc[
    train["Date"].between(VALIDATION_START, VALIDATION_END)
].copy()

# ---------------------------------------------------------
# Data-window checks
# ---------------------------------------------------------
assert not history.empty, "Training history is empty"
assert not validation.empty, "First-window validation data is empty"

assert history["Date"].max() == CUTOFF_DATE
assert validation["Date"].min() == VALIDATION_START
assert validation["Date"].max() == VALIDATION_END
assert validation["Date"].nunique() == EXPECTED_DAYS
assert len(validation) == EXPECTED_ROWS

assert not validation.duplicated(["Store", "Date"]).any()
assert not history.duplicated(["Store", "Date"]).any()

print("Forecast origin:", CUTOFF_DATE.date())
print("Validation window:", VALIDATION_START.date(), "to", VALIDATION_END.date())
print("History rows:", len(history))
print("Validation rows:", len(validation))
print("Validation dates:", validation["Date"].nunique())

# ---------------------------------------------------------
# Recursive baseline models
# ---------------------------------------------------------
baselines = {
    "Naive_Lag1": (
        LagBaseline("Sales_Lag_1"),
        ["Sales_Lag_1"],
    ),
    "SeasonalNaive_Lag7": (
        LagBaseline("Sales_Lag_7"),
        ["Sales_Lag_7"],
    ),
    "MovingAverage_7": (
        MovingAverageBaseline(),
        ["Sales_RollingMean_7"],
    ),
}

all_predictions = []
metrics = []

validation_actual = validation[
    ["Store", "Date", "Sales"]
].rename(columns={"Sales": "ActualSales"})

for name, (model, features) in baselines.items():

    print(f"\nRunning {name}...")

    # Each model receives an independent copy of the same
    # historical information and recursively builds its own history.
    builder = RecursiveFeatureBuilder(history.copy())

    forecast = recursive_forecast(
        base_data=validation.copy(),
        feature_builder=builder,
        model=model,
        feature_columns=features,
    )

    forecast_eval = forecast[
        ["Store", "Date", "Sales", "PredictedSales"]
    ].rename(columns={"Sales": "ForecastAttachedSales"})

    # Independently align predictions and actuals by Store-Date.
    evaluation = forecast_eval.merge(
        validation_actual,
        on=["Store", "Date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    # Verify complete one-to-one coverage.
    assert len(evaluation) == EXPECTED_ROWS
    assert (evaluation["_merge"] == "both").all()

    # Verify actual sales remain attached to the correct dates/stores.
    assert np.allclose(
        evaluation["ForecastAttachedSales"],
        evaluation["ActualSales"],
    ), "Actual sales alignment failed"

    actual = evaluation["ActualSales"].to_numpy(dtype=float)
    predictions = evaluation["PredictedSales"].to_numpy(dtype=float)

    assert np.isfinite(actual).all()
    assert np.isfinite(predictions).all()
    assert (actual >= 0).all()
    assert (predictions >= 0).all()
    assert actual.sum() > 0

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    wmape = np.abs(actual - predictions).sum() / actual.sum()
    rmsle = np.sqrt(mean_squared_log_error(actual, predictions))

    print("Store-Date alignment: PASS")
    print("Rows evaluated:", len(evaluation))

    metrics.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "WMAPE": wmape,
        "RMSLE": rmsle,
    })

    all_predictions.append(
        forecast[
            ["Store", "Date", "Sales", "PredictedSales"]
        ].assign(Model=name)
    )

    print(
        f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
        f"WMAPE={wmape:.4%}, RMSLE={rmsle:.4f}"
    )

# ---------------------------------------------------------
# Save results separately from the existing later window
# ---------------------------------------------------------
metrics_df = pd.DataFrame(metrics).sort_values("WMAPE")
predictions_df = pd.concat(all_predictions, ignore_index=True)

reports_dir = ROOT / "reports"
processed_dir = ROOT / "data" / "processed"

reports_dir.mkdir(exist_ok=True)
processed_dir.mkdir(exist_ok=True)

metrics_path = reports_dir / "first_window_baseline_results.csv"
predictions_path = (
    processed_dir / "first_window_baseline_predictions.csv"
)

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

print("\nFirst-window baseline comparison:")
print(metrics_df.to_string(index=False))

print("\nSaved:")
print(metrics_path)
print(predictions_path)
