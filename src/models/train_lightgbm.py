
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src" / "features"))

from recursive_forecasting import RecursiveFeatureBuilder
from recursive_forecaster import recursive_forecast
from feature_config import get_feature_columns, CATEGORICAL_FEATURES


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------

train = pd.read_pickle(
    ROOT / "data" / "processed" / "train_features.pkl"
)

history = pd.read_pickle(
    ROOT / "data" / "processed" / "train_preprocessed.pkl"
)

validation = pd.read_pickle(
    ROOT / "data" / "processed" / "validation_base.pkl"
)

for df in (train, history, validation):
    df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------------------------
# 2. Prepare training data
# ---------------------------------------------------------

# Train only on observations available before forecast origin.
cutoff = pd.Timestamp("2015-06-19")

train = train.loc[
    (train["Date"] <= cutoff) &
    (train["Open"] == 1)
].copy()

history = history.loc[
    history["Date"] <= cutoff
].copy()

feature_columns = get_feature_columns(train)

categorical_features = [
    col for col in CATEGORICAL_FEATURES
    if col in feature_columns
]

X_train = train[feature_columns].copy()
y_train = train["Sales"].astype(float)

# Establish categorical levels from training data.
category_levels = {}

for col in categorical_features:
    category_levels[col] = pd.Index(
        X_train[col].dropna().unique()
    )

    X_train[col] = pd.Categorical(
        X_train[col],
        categories=category_levels[col]
    )

print("Training rows:", len(X_train))
print("Training features:", len(feature_columns))
print("Categorical features:", categorical_features)


# ---------------------------------------------------------
# 3. Train global LightGBM model
# ---------------------------------------------------------

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

print("\nTraining LightGBM...")

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features,
)

print("LightGBM training completed.")


# ---------------------------------------------------------
# 4. Prepare validation categorical columns
# ---------------------------------------------------------

for col in categorical_features:
    validation[col] = pd.Categorical(
        validation[col],
        categories=category_levels[col]
    )


# ---------------------------------------------------------
# 5. Recursive validation forecasting
# ---------------------------------------------------------

print("\nStarting recursive validation forecast...")

builder = RecursiveFeatureBuilder(history)

forecast = recursive_forecast(
    base_data=validation,
    feature_builder=builder,
    model=model,
    feature_columns=feature_columns,
)

# Align evaluation by Store-Date.
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

assert len(evaluation) == len(validation), "Row count mismatch"
assert (evaluation["_merge"] == "both").all(), "Store-Date mismatch"

assert np.allclose(
    evaluation["ForecastAttachedSales"],
    evaluation["ActualSales"],
), "Actual sales alignment failed"

actual = evaluation["ActualSales"].to_numpy(dtype=float)
predictions = evaluation["PredictedSales"].to_numpy(dtype=float)

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
wmape = np.abs(actual - predictions).sum() / actual.sum()
rmsle = np.sqrt(
    mean_squared_log_error(actual, predictions)
)

print("\nLightGBM validation results")
print("---------------------------")
print("Store-Date alignment: PASS")
print("Rows evaluated:", len(evaluation))
print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"WMAPE: {wmape:.4%}")
print(f"RMSLE: {rmsle:.4f}")


# ---------------------------------------------------------
# 6. Save model, predictions and feature importance
# ---------------------------------------------------------

processed_dir = ROOT / "data" / "processed"
reports_dir = ROOT / "reports"

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "category_levels": category_levels,
    },
    processed_dir / "lightgbm_model.joblib",
)

forecast[
    ["Store", "Date", "Sales", "PredictedSales"]
].to_csv(
    processed_dir / "validation_lightgbm_predictions.csv",
    index=False,
)

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)

importance.to_csv(
    reports_dir / "lightgbm_feature_importance.csv",
    index=False,
)

metrics = pd.DataFrame([{
    "Model": "LightGBM",
    "MAE": mae,
    "RMSE": rmse,
    "WMAPE": wmape,
    "RMSLE": rmsle,
}])

metrics.to_csv(
    reports_dir / "lightgbm_results.csv",
    index=False,
)

print("\nSaved:")
print("data/processed/lightgbm_model.joblib")
print("data/processed/validation_lightgbm_predictions.csv")
print("reports/lightgbm_feature_importance.csv")
print("reports/lightgbm_results.csv")

print("\nTop 15 features:")
print(importance.head(15).to_string(index=False))
