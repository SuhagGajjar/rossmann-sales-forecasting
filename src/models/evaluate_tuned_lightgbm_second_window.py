
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)

from src.features.build_features import (
    ROOT,
    load_data,
    build_base_features,
    add_training_history_features,
)
from src.features.recursive_forecasting import RecursiveFeatureBuilder
from src.models.recursive_forecaster import recursive_forecast
from src.models.feature_config import (
    CATEGORICAL_FEATURES,
    get_feature_columns,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

TRAIN_CUTOFF = pd.Timestamp("2015-06-19")
VALIDATION_START = pd.Timestamp("2015-06-20")
VALIDATION_END = pd.Timestamp("2015-07-31")

RANDOM_STATE = 42

TUNING_DIR = (
    ROOT / "models" / "experiments" / "lightgbm_tuning"
)

BEST_PARAMS_PATH = (
    TUNING_DIR / "lightgbm_tuning_best_params.json"
)

OUTPUT_DIR = (
    ROOT / "models" / "experiments" / "lightgbm_second_window"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0,
    )

    denominator = np.sum(np.abs(actual))

    wmape = (
        np.sum(np.abs(actual - predicted)) / denominator * 100
        if denominator > 0
        else np.nan
    )

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "WMAPE": float(wmape),
        "RMSLE": float(
            np.sqrt(mean_squared_log_error(actual, predicted))
        ),
    }


def evaluate_predictions(predictions):
    actual = predictions["Sales"].to_numpy(dtype=float)
    predicted = predictions["PredictedSales"].to_numpy(dtype=float)

    all_store_metrics = calculate_metrics(actual, predicted)

    open_mask = predictions["Open"].to_numpy() == 1

    open_store_metrics = calculate_metrics(
        actual[open_mask],
        predicted[open_mask],
    )

    return {
        "all_store": all_store_metrics,
        "open_store": open_store_metrics,
    }


# ============================================================
# 3. PREPARE SECOND-WINDOW DATA
# ============================================================

def prepare_datasets():
    print("\nLoading source data...")

    train_raw, _ = load_data()

    train_raw["Date"] = pd.to_datetime(train_raw["Date"])

    training_raw = train_raw.loc[
        train_raw["Date"] <= TRAIN_CUTOFF
    ].copy()

    validation_raw = train_raw.loc[
        train_raw["Date"].between(
            VALIDATION_START,
            VALIDATION_END,
        )
    ].copy()

    expected_days = (
        VALIDATION_END - VALIDATION_START
    ).days + 1

    actual_days = validation_raw["Date"].nunique()

    if actual_days != expected_days:
        raise ValueError(
            f"Expected {expected_days} validation days, "
            f"found {actual_days}."
        )

    if training_raw["Date"].max() != TRAIN_CUTOFF:
        raise ValueError("Training cutoff mismatch.")

    if validation_raw["Date"].min() != VALIDATION_START:
        raise ValueError("Validation start mismatch.")

    expected_rows = 1115 * expected_days

    if len(validation_raw) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} validation rows, "
            f"found {len(validation_raw):,}."
        )

    print(f"Training cutoff: {TRAIN_CUTOFF.date()}")
    print(
        f"Training rows: {len(training_raw):,}; "
        f"validation rows: {len(validation_raw):,}"
    )
    print(f"Validation days: {actual_days}")

    # Build features independently for the chronological splits.
    training = build_base_features(training_raw)
    validation_base = build_base_features(validation_raw)

    # Historical features use training observations only.
    training = add_training_history_features(training)

    # Train only on open-store observations.
    training_open = training.loc[
        training["Open"] == 1
    ].copy()

    feature_columns = get_feature_columns(training_open)

    categorical_columns = [
        col for col in CATEGORICAL_FEATURES
        if col in feature_columns
    ]

    # Preserve training category levels at inference.
    category_levels = {}

    for col in categorical_columns:
        categories = (
            training_open[col]
            .astype("category")
            .cat.categories
        )

        category_levels[col] = categories.tolist()

        training_open[col] = pd.Categorical(
            training_open[col],
            categories=categories,
        )

        validation_base[col] = pd.Categorical(
            validation_base[col],
            categories=categories,
        )

    # Recursive history must stop at the forecast origin.
    history = training_raw[
        ["Store", "Date", "DayOfWeek", "Sales"]
    ].copy()

    print(f"Open-store training observations: {len(training_open):,}")
    print(f"Feature count: {len(feature_columns)}")
    print("Categorical features:", categorical_columns)

    return (
        training_open,
        validation_base,
        history,
        feature_columns,
        categorical_columns,
        category_levels,
    )


# ============================================================
# 4. LOAD FIXED OPTUNA PARAMETERS
# ============================================================

def load_best_params():
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Best-parameters file not found: {BEST_PARAMS_PATH}"
        )

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as file:
        best_params = json.load(file)

    # Fixed training/runtime parameters.
    final_params = {
        **best_params,
        "objective": "regression",
        "subsample_freq": 1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    print("\nLoaded frozen Optuna parameters:")
    print(json.dumps(final_params, indent=4))

    return final_params


# ============================================================
# 5. RECURSIVE SECOND-WINDOW EVALUATION
# ============================================================

def run_recursive_evaluation(
    model,
    validation_base,
    history,
    feature_columns,
):
    # Fresh builder; future actual sales are never fed back.
    feature_builder = RecursiveFeatureBuilder(history.copy())
    feature_capture = []
    predictions = recursive_forecast(
        base_data=validation_base.copy(),
        feature_builder=feature_builder,
        model=model,
        feature_columns=feature_columns,
        prediction_col="PredictedSales",
        feature_capture=feature_capture,
    )

    predictions["Date"] = pd.to_datetime(predictions["Date"])

    if predictions.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate Store-Date predictions.")

    if predictions["PredictedSales"].isna().any():
        raise ValueError("Predictions contain missing values.")

    # Explicit one-to-one alignment check.
    expected_keys = validation_base[
        ["Store", "Date"]
    ].copy()

    predicted_keys = predictions[
        ["Store", "Date"]
    ].copy()

    alignment = expected_keys.merge(
        predicted_keys,
        on=["Store", "Date"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not (alignment["_merge"] == "both").all():
        raise ValueError("Store-Date alignment failed.")

    if len(predictions) != len(validation_base):
        raise ValueError("Prediction row count mismatch.")

    print("\nStore-Date alignment: PASS")
    print(f"Rows evaluated: {len(predictions):,}")

    metrics = evaluate_predictions(predictions)

    return predictions, metrics, pd.concat(
        feature_capture,
        ignore_index=True
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():
    (
        training,
        validation_base,
        history,
        feature_columns,
        categorical_columns,
        category_levels,
    ) = prepare_datasets()

    X_train = training[feature_columns]
    y_train = training["Sales"].astype(float)

    final_params = load_best_params()

    print("\nRefitting tuned LightGBM on expanded training window...")

    model = lgb.LGBMRegressor(**final_params)

    model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_columns,
    )

    print("\nRunning recursive 42-day forecast...")

    predictions, metrics, explanation_features = run_recursive_evaluation(
        model=model,
        validation_base=validation_base,
        history=history,
        feature_columns=feature_columns,
    )

    # ========================================================
    # 7. SAVE PREDICTIONS AND MODEL
    # ========================================================

    prediction_path = (
        OUTPUT_DIR / "second_window_tuned_lightgbm_predictions.csv"
    )
    predictions.to_csv(prediction_path, index=False)

    model_path = (
        OUTPUT_DIR / "second_window_tuned_lightgbm_model.joblib"
    )
    joblib.dump(model, model_path)

    explanation_path = OUTPUT_DIR / "recursive_explanation_features.joblib"
    joblib.dump(explanation_features, explanation_path)

    print(f"Recursive explanation features saved: {explanation_path}")

    # ========================================================
    # 8. SAVE FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance_Gain": model.booster_.feature_importance(
            importance_type="gain"
        ),
        "Importance_Split": model.booster_.feature_importance(
            importance_type="split"
        ),
    }).sort_values(
        "Importance_Gain",
        ascending=False,
    )

    importance_path = (
        OUTPUT_DIR / "second_window_tuned_lightgbm_feature_importance.csv"
    )
    importance_df.to_csv(importance_path, index=False)

    # ========================================================
    # 9. SAVE METRICS AND RUN CONFIGURATION
    # ========================================================

    metrics_path = (
        REPORT_DIR / "second_window_tuned_lightgbm_results.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    run_config = {
        "model": "Tuned LightGBM",
        "training_cutoff": str(TRAIN_CUTOFF.date()),
        "validation_start": str(VALIDATION_START.date()),
        "validation_end": str(VALIDATION_END.date()),
        "validation_days": 42,
        "training_rows_open_stores": len(training),
        "validation_rows": len(validation_base),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "categorical_features": categorical_columns,
        "category_levels": {
            key: [str(value) for value in values]
            for key, values in category_levels.items()
        },
        "parameters_source": str(BEST_PARAMS_PATH),
        "parameters": final_params,
        "forecast_method": "Recursive multi-step forecasting",
        "future_actual_sales_used_for_updates": False,
        "metrics": metrics,
    }

    config_path = (
        OUTPUT_DIR / "second_window_tuned_lightgbm_run_config.json"
    )

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=4)

    # ========================================================
    # 10. RESULTS
    # ========================================================

    print("\n========================================")
    print("TUNED LIGHTGBM — SECOND WINDOW COMPLETE")
    print("========================================")

    print("\nAll-store metrics:")
    print(json.dumps(metrics["all_store"], indent=4))

    print("\nOpen-store metrics:")
    print(json.dumps(metrics["open_store"], indent=4))

    print("\nTop 15 features by gain:")
    print(importance_df.head(15).to_string(index=False))

    print("\nOutputs saved to:")
    print(OUTPUT_DIR)
    print(REPORT_DIR)


if __name__ == "__main__":
    main()
