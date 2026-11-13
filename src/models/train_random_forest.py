from sklearn.impute import SimpleImputer
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
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


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

train = pd.read_pickle(ROOT / "data/processed/train_features.pkl")
history = pd.read_pickle(ROOT / "data/processed/train_preprocessed.pkl")
validation = pd.read_pickle(ROOT / "data/processed/validation_base.pkl")

for df in (train, history, validation):
    df["Date"] = pd.to_datetime(df["Date"])

cutoff = pd.Timestamp("2015-06-19")

train = train[
    (train["Date"] <= cutoff) &
    (train["Open"] == 1)
].copy()

history = history[history["Date"] <= cutoff].copy()

feature_columns = get_feature_columns(train)

categorical_features = [
    col for col in CATEGORICAL_FEATURES
    if col in feature_columns
]

numeric_features = [
    col for col in feature_columns
    if col not in categorical_features
]

# Convert categorical columns consistently for OneHotEncoder.
for col in categorical_features:
    train[col] = train[col].astype(str)
    validation[col] = validation[col].astype(str)

print(f"Training rows available: {len(train)}")
print(f"Training features: {len(feature_columns)}")
print(f"Categorical features: {categorical_features}")

# --------------------------------------------------
# 2. Reproducible training sample
# --------------------------------------------------

X_train = train[feature_columns]
y_train = train["Sales"].astype(float)

print(f"Random Forest training rows: {len(train)}")

# --------------------------------------------------
# 3. Preprocessing + Random Forest
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                dtype=np.float32,
            ),
            categorical_features,
        ),
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
            ]),
            numeric_features,
        ),
    ],
    remainder="drop",
)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=10,
    max_features=0.7,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", rf),
])

# --------------------------------------------------
# 4. Train
# --------------------------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Random Forest training completed.")

# --------------------------------------------------
# 5. Recursive validation
# --------------------------------------------------

print("\nStarting recursive validation forecast...")

builder = RecursiveFeatureBuilder(history)

forecast = recursive_forecast(
    base_data=validation,
    feature_builder=builder,
    model=model,
    feature_columns=feature_columns,
)

# --------------------------------------------------
# 6. Independent Store-Date alignment check
# --------------------------------------------------

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

assert len(evaluation) == len(validation)
assert evaluation["_merge"].eq("both").all()

assert np.allclose(
    evaluation["ForecastAttachedSales"],
    evaluation["ActualSales"],
)

print("Store-Date alignment: PASS")
print(f"Rows evaluated: {len(evaluation)}")

# --------------------------------------------------
# 7. Evaluation metrics
# --------------------------------------------------

actual = evaluation["ActualSales"].to_numpy(dtype=float)
predictions = evaluation["PredictedSales"].to_numpy(dtype=float)

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
wmape = np.abs(actual - predictions).sum() / np.abs(actual).sum()
rmsle = np.sqrt(
    mean_squared_log_error(
        actual,
        np.maximum(predictions, 0),
    )
)

print("\nRandom Forest validation results")
print("--------------------------------")
print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"WMAPE: {wmape:.4%}")
print(f"RMSLE: {rmsle:.4f}")

# --------------------------------------------------
# 8. Save artifacts
# --------------------------------------------------

model_path = ROOT / "data/processed/random_forest_model.joblib"

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "training_rows": len(train),
        "random_state": 42,
    },
    model_path,
)

forecast.to_csv(
    ROOT / "data/processed/validation_random_forest_predictions.csv",
    index=False,
)

metrics = pd.DataFrame([{
    "Model": "RandomForest",
    "MAE": mae,
    "RMSE": rmse,
    "WMAPE": wmape,
    "RMSLE": rmsle,
    "RowsEvaluated": len(evaluation),
    "StoreDateAlignment": "PASS",
    "ValidationType": "Recursive",
    "ValidationStart": "2015-06-20",
    "ValidationEnd": "2015-07-31",
    "TrainingRowsAvailable": len(train),
    "TrainingRows": len(train),
}])

metrics.to_csv(
    ROOT / "reports/random_forest_results.csv",
    index=False,
)

# --------------------------------------------------
# 9. Feature importance
# --------------------------------------------------

feature_names = (
    model.named_steps["preprocessor"].get_feature_names_out()
)

importance = model.named_steps[
    "random_forest"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance,
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    ROOT / "reports/random_forest_feature_importance.csv",
    index=False,
)

print("\nTop 15 Random Forest features:")
print(importance_df.head(15).to_string(index=False))

print("\nSaved:")
print("data/processed/random_forest_model.joblib")
print("data/processed/validation_random_forest_predictions.csv")
print("reports/random_forest_results.csv")
print("reports/random_forest_feature_importance.csv")
