
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))

from recursive_forecasting import RecursiveFeatureBuilder
from recursive_forecaster import recursive_forecast
from baselines import LagBaseline, MovingAverageBaseline


# Load data
train = pd.read_pickle(
    ROOT / "data" / "processed" / "train_preprocessed.pkl"
)
validation = pd.read_pickle(
    ROOT / "data" / "processed" / "validation_base.pkl"
)

train["Date"] = pd.to_datetime(train["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])

# Strict forecast origin: actual history through June 19 only.
cutoff = pd.Timestamp("2015-06-19")
history = train.loc[train["Date"] <= cutoff].copy()

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

actual = validation["Sales"].to_numpy(dtype=float)


for name, (model, features) in baselines.items():

    print(f"Running {name}...")

    builder = RecursiveFeatureBuilder(history)

    forecast = recursive_forecast(
        base_data=validation,
        feature_builder=builder,
        model=model,
        feature_columns=features,
    )

    # Independently align predictions and actuals by Store-Date.
    forecast_eval = forecast[
        ["Store", "Date", "Sales", "PredictedSales"]
    ].rename(columns={"Sales": "ForecastAttachedSales"})

    validation_actual = validation[
        ["Store", "Date", "Sales"]
    ].rename(columns={"Sales": "ActualSales"})

    evaluation = forecast_eval.merge(
        validation_actual,
        on=["Store", "Date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    # Verify complete one-to-one coverage.
    assert len(evaluation) == len(validation), "Row count mismatch"
    assert (evaluation["_merge"] == "both").all(), "Store-Date mismatch"

    # Verify actual sales remained attached to the correct forecast rows.
    assert np.allclose(
        evaluation["ForecastAttachedSales"],
        evaluation["ActualSales"],
    ), "Actual sales alignment failed"

    actual = evaluation["ActualSales"].to_numpy(dtype=float)
    predictions = evaluation["PredictedSales"].to_numpy(dtype=float)

    # Calculate metrics.
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    wmape = np.abs(actual - predictions).sum() / actual.sum()
    rmsle = np.sqrt(
        mean_squared_log_error(actual, predictions)
    )

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
        forecast[["Store", "Date", "Sales", "PredictedSales"]]
        .assign(Model=name)
    )

    print(
        f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
        f"WMAPE={wmape:.4f}, RMSLE={rmsle:.4f}"
    )


# Save results
metrics_df = pd.DataFrame(metrics).sort_values("WMAPE")
predictions_df = pd.concat(all_predictions, ignore_index=True)

reports_dir = ROOT / "reports"
processed_dir = ROOT / "data" / "processed"

reports_dir.mkdir(exist_ok=True)

metrics_df.to_csv(
    reports_dir / "baseline_results.csv",
    index=False
)

predictions_df.to_csv(
    processed_dir / "validation_baseline_predictions.csv",
    index=False
)

print("\nBaseline comparison:")
print(metrics_df.to_string(index=False))
print("\nSaved:")
print("reports/baseline_results.csv")
print("data/processed/validation_baseline_predictions.csv")
