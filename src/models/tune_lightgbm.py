
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)

from src.features.build_features import (
    ROOT,
    TRAIN_PATH,
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

TRAIN_CUTOFF = pd.Timestamp("2015-05-08")
VALIDATION_START = pd.Timestamp("2015-05-09")
VALIDATION_END = pd.Timestamp("2015-06-19")

N_TRIALS = 20
RANDOM_STATE = 42

OUTPUT_DIR = (
    ROOT / "models" / "experiments" / "lightgbm_tuning"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = OUTPUT_DIR / "lightgbm_tuning.db"
STUDY_NAME = "lightgbm_42day_tuning_seed42"


# ============================================================
# 2. METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    predicted = np.maximum(predicted, 0)

    denominator = np.sum(np.abs(actual))

    wmape = (
        np.sum(np.abs(actual - predicted)) / denominator * 100
        if denominator > 0 else np.nan
    )

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(
            mean_squared_error(actual, predicted)
        )),
        "WMAPE": float(wmape),
        "RMSLE": float(np.sqrt(
            mean_squared_log_error(actual, predicted)
        )),
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
# 3. PREPARE THE CHRONOLOGICAL DATA SPLIT
# ============================================================

def prepare_datasets():
    print("\nLoading preprocessed data...")

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
        raise ValueError("Training cutoff does not match configuration.")

    if validation_raw["Date"].min() != VALIDATION_START:
        raise ValueError("Validation start does not match configuration.")

    print(f"Training cutoff: {TRAIN_CUTOFF.date()}")
    print(
        f"Training rows: {len(training_raw):,}; "
        f"validation rows: {len(validation_raw):,}"
    )
    print(
        f"Validation dates: {actual_days}; "
        f"validation store-days: {len(validation_raw):,}"
    )

    # Base features are created independently for each split.
    training = build_base_features(training_raw)
    validation_base = build_base_features(validation_raw)

    # Historical training features use only earlier rows.
    training = add_training_history_features(training)

    # Train only on open stores, matching the existing model.
    training_open = training.loc[
        training["Open"] == 1
    ].copy()

    # Determine model features from training data only.
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

    # Historical observations must stop at the forecast origin.
    history = training_raw[
        ["Store", "Date", "DayOfWeek", "Sales"]
    ].copy()

    return (
        training_open,
        validation_base,
        history,
        feature_columns,
        categorical_columns,
        category_levels,
    )


# ============================================================
# 4. RECURSIVE VALIDATION
# ============================================================

def forecast_and_evaluate(
    model,
    validation_base,
    history,
    feature_columns,
):
    # A fresh builder is essential for every trial.
    # Each recursive forecast mutates its historical state.
    feature_builder = RecursiveFeatureBuilder(history.copy())

    predictions = recursive_forecast(
        base_data=validation_base.copy(),
        feature_builder=feature_builder,
        model=model,
        feature_columns=feature_columns,
        prediction_col="PredictedSales",
    )

    predictions["Date"] = pd.to_datetime(predictions["Date"])

    # Validate one-to-one Store-Date coverage.
    if predictions.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate Store-Date predictions.")

    if len(predictions) != len(validation_base):
        raise ValueError("Prediction row count mismatch.")

    if predictions["PredictedSales"].isna().any():
        raise ValueError("Predictions contain missing values.")

    metrics = evaluate_predictions(predictions)

    return predictions, metrics


# ============================================================
# 5. MAIN OPTUNA TUNING
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

    print("\nTraining observations:", f"{len(training):,}")
    print("Feature count:", len(feature_columns))
    print("Categorical features:", categorical_columns)

    def objective(trial):
        params = {
            "objective": "regression",
            "n_estimators": trial.suggest_int(
                "n_estimators", 300, 800, step=100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.08, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 15, 63
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", [-1, 8, 10, 12]
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 100, step=10
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.7, 1.0
            ),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.1, 10.0, log=True
            ),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }

        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_columns,
        )

        predictions, metrics = forecast_and_evaluate(
            model=model,
            validation_base=validation_base,
            history=history,
            feature_columns=feature_columns,
        )

        all_metrics = metrics["all_store"]
        open_metrics = metrics["open_store"]

        # Preserve all metrics for every trial.
        for metric_name, metric_value in all_metrics.items():
            trial.set_user_attr(
                f"all_store_{metric_name}",
                metric_value,
            )

        for metric_name, metric_value in open_metrics.items():
            trial.set_user_attr(
                f"open_store_{metric_name}",
                metric_value,
            )

        print(
            f"\nTrial {trial.number}: "
            f"WMAPE={all_metrics['WMAPE']:.4f}% | "
            f"MAE={all_metrics['MAE']:.2f} | "
            f"RMSE={all_metrics['RMSE']:.2f} | "
            f"RMSLE={all_metrics['RMSLE']:.4f}"
        )

        # WMAPE is the single optimisation objective.
        return all_metrics["WMAPE"]

    storage_url = f"sqlite:///{DB_PATH.as_posix()}"

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
        load_if_exists=True,
    )

    completed_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )

    print("\nStarting Optuna tuning...")
    print("Completed trials already in study:", completed_before)

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        gc_after_trial=True,
    )

    best_trial = study.best_trial
    best_params = study.best_params

    print("\nBest trial:", best_trial.number)
    print("Best WMAPE:", round(study.best_value, 4))
    print("Best parameters:")
    print(json.dumps(best_params, indent=4))

    # ========================================================
    # 6. EXPORT TRIAL RESULTS
    # ========================================================

    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    )

    trials_path = OUTPUT_DIR / "lightgbm_tuning_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    # ========================================================
    # 7. REFIT THE BEST TUNING MODEL
    # ========================================================

    final_params = {
        **best_params,
        "objective": "regression",
        "subsample_freq": 1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    best_model = lgb.LGBMRegressor(**final_params)

    best_model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_columns,
    )

    best_predictions, best_metrics = forecast_and_evaluate(
        model=best_model,
        validation_base=validation_base,
        history=history,
        feature_columns=feature_columns,
    )

    # Save tuning-window predictions separately.
    prediction_path = (
        OUTPUT_DIR / "tuned_lightgbm_tuning_predictions.csv"
    )
    best_predictions.to_csv(prediction_path, index=False)

    model_path = OUTPUT_DIR / "tuned_lightgbm_tuning_model.joblib"
    joblib.dump(best_model, model_path)

    # ========================================================
    # 8. SAVE CONFIGURATION AND METRICS
    # ========================================================

    run_config = {
        "study_name": STUDY_NAME,
        "random_state": RANDOM_STATE,
        "n_trials_requested_this_run": N_TRIALS,
        "training_cutoff": str(TRAIN_CUTOFF.date()),
        "validation_start": str(VALIDATION_START.date()),
        "validation_end": str(VALIDATION_END.date()),
        "validation_days": 42,
        "validation_rows": len(validation_base),
        "training_rows_open_stores": len(training),
        "feature_columns": feature_columns,
        "categorical_features": categorical_columns,
        "category_levels": {
            key: [str(value) for value in values]
            for key, values in category_levels.items()
        },
        "best_trial_number": best_trial.number,
        "best_params": best_params,
        "optimisation_metric": "WMAPE",
        "all_metrics": best_metrics["all_store"],
        "open_store_metrics": best_metrics["open_store"],
    }

    config_path = OUTPUT_DIR / "lightgbm_tuning_run_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=4)

    metrics_path = OUTPUT_DIR / "lightgbm_tuning_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=4)

    params_path = OUTPUT_DIR / "lightgbm_tuning_best_params.json"

    with open(params_path, "w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=4)

    print("\n========================================")
    print("LIGHTGBM TUNING COMPLETE")
    print("========================================")
    print("Best trial:", best_trial.number)
    print("Best WMAPE:", round(study.best_value, 4))
    print("\nFinal tuning-window metrics:")
    print(json.dumps(best_metrics, indent=4))
    print("\nOutputs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
