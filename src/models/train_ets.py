
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train_preprocessed.pkl"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation_base.pkl"

PREDICTION_PATH = ROOT / "data" / "processed" / "validation_ets_predictions.csv"
RESULTS_PATH = ROOT / "reports" / "ets_results.csv"
STATUS_PATH = ROOT / "reports" / "ets_store_fit_status.csv"

CUTOFF_DATE = pd.Timestamp("2015-06-19")

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WMAPE": (
            np.abs(actual - predicted).sum() / np.abs(actual).sum()
            if np.abs(actual).sum() > 0 else np.nan
        ),
        "RMSLE": np.sqrt(
            np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)
        ),
    }

def seasonal_naive_fallback(history, horizon):
    """Repeat the most recent seven daily observations."""
    last_week = history[-7:]

    if len(last_week) < 7:
        raise ValueError("Insufficient history for seasonal-naive fallback.")

    return np.tile(last_week, int(np.ceil(horizon / 7)))[:horizon]

def main():
    train = pd.read_pickle(TRAIN_PATH)
    validation = pd.read_pickle(VALIDATION_PATH)

    train["Date"] = pd.to_datetime(train["Date"])
    validation["Date"] = pd.to_datetime(validation["Date"])

    required_train = {"Store", "Date", "Sales"}
    required_validation = {"Store", "Date", "Sales", "Open"}

    if not required_train.issubset(train.columns):
        raise ValueError(
            f"Training data missing: {required_train - set(train.columns)}"
        )

    if not required_validation.issubset(validation.columns):
        raise ValueError(
            f"Validation data missing: "
            f"{required_validation - set(validation.columns)}"
        )

    history_data = train.loc[train["Date"] <= CUTOFF_DATE].copy()

    if history_data["Date"].max() != CUTOFF_DATE:
        raise ValueError("Training history does not reach the cutoff date.")

    forecast_dates = np.sort(validation["Date"].unique())
    horizon = len(forecast_dates)

    if horizon != 42:
        raise ValueError(f"Expected 42 validation dates, found {horizon}.")

    if validation.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate Store-Date keys in validation data.")

    validation_stores = set(validation["Store"].unique())
    history_stores = set(history_data["Store"].unique())

    if validation_stores != history_stores:
        raise ValueError("Training and validation store coverage differs.")

    print(f"Training history rows: {len(history_data):,}")
    print(f"Stores: {len(history_stores):,}")
    print(f"Forecast horizon: {horizon} days")
    print(f"Cutoff: {CUTOFF_DATE.date()}")

    predictions = []
    statuses = []

    for store, group in history_data.groupby("Store", sort=True):
        group = group.sort_values("Date")

        # ETS seasonal periods assume consecutive daily observations.
        date_differences = group["Date"].diff().dropna().dt.days

        if not date_differences.eq(1).all():
            raise ValueError(
                f"Non-consecutive daily history found for Store {store}. "
                "Investigate missing dates before fitting ETS."
            )

        y = group["Sales"].astype(float).to_numpy()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ExponentialSmoothing(
                    y,
                    trend="add",
                    damped_trend=True,
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated",
                )

                fitted_model = model.fit(optimized=True)
                forecast = np.asarray(
                    fitted_model.forecast(horizon), dtype=float
                )

            if len(forecast) != horizon or not np.isfinite(forecast).all():
                raise ValueError("ETS returned invalid forecasts.")

            status = "ETS"

        except Exception as exc:
            forecast = seasonal_naive_fallback(y, horizon)
            status = f"SeasonalNaive_Fallback: {type(exc).__name__}"

        forecast = np.maximum(forecast, 0)

        store_validation = (
            validation.loc[validation["Store"] == store]
            .sort_values("Date")
            .copy()
        )

        if len(store_validation) != horizon:
            raise ValueError(
                f"Store {store} has {len(store_validation)} validation rows; "
                f"expected {horizon}."
            )

        # Apply the known operational closure rule.
        forecast[store_validation["Open"].to_numpy() == 0] = 0

        predictions.append(
            pd.DataFrame({
                "Store": store,
                "Date": forecast_dates,
                "PredictedSales": forecast,
            })
        )

        statuses.append({
            "Store": store,
            "FitStatus": status,
            "TrainingObservations": len(y),
        })

    forecast_df = pd.concat(predictions, ignore_index=True)
    status_df = pd.DataFrame(statuses)

    # Independently align predictions to actuals using Store-Date.
    evaluation = validation[
        ["Store", "Date", "Sales", "Open"]
    ].merge(
        forecast_df,
        on=["Store", "Date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not evaluation["_merge"].eq("both").all():
        raise AssertionError("ETS Store-Date alignment failed.")

    if len(evaluation) != len(validation):
        raise AssertionError("ETS evaluation row count mismatch.")

    evaluation = evaluation.drop(columns="_merge")

    if not np.isfinite(evaluation["PredictedSales"]).all():
        raise AssertionError("ETS predictions contain non-finite values.")

    closed_mask = evaluation["Open"].eq(0)

    if not evaluation.loc[closed_mask, "PredictedSales"].eq(0).all():
        raise AssertionError("Closed-store predictions are not all zero.")

    print("\nStore-Date alignment: PASS")
    print(f"Rows evaluated: {len(evaluation):,}")
    print(f"Stores fitted: {status_df['Store'].nunique():,}")
    print(f"ETS fits: {status_df['FitStatus'].eq('ETS').sum():,}")
    print(
        "Fallback fits: "
        f"{status_df['FitStatus'].str.startswith('SeasonalNaive_Fallback').sum():,}"
    )

    # Report both the full operational population and open-store accuracy.
    overall_metrics = calculate_metrics(
        evaluation["Sales"], evaluation["PredictedSales"]
    )

    open_evaluation = evaluation.loc[evaluation["Open"].eq(1)]

    open_metrics = calculate_metrics(
        open_evaluation["Sales"], open_evaluation["PredictedSales"]
    )

    results = pd.DataFrame([
        {
            "Model": "ETS_AdditiveTrend_WeeklySeasonality",
            "EvaluationScope": "All store-days",
            "TrainingRows": len(history_data),
            "ValidationRows": len(evaluation),
            "MAE": overall_metrics["MAE"],
            "RMSE": overall_metrics["RMSE"],
            "WMAPE": overall_metrics["WMAPE"],
            "RMSLE": overall_metrics["RMSLE"],
            "StoreDateAlignment": "PASS",
            "FallbackStores": int(
                status_df["FitStatus"].str.startswith(
                    "SeasonalNaive_Fallback"
                ).sum()
            ),
        },
        {
            "Model": "ETS_AdditiveTrend_WeeklySeasonality",
            "EvaluationScope": "Open store-days only",
            "TrainingRows": len(history_data),
            "ValidationRows": len(open_evaluation),
            "MAE": open_metrics["MAE"],
            "RMSE": open_metrics["RMSE"],
            "WMAPE": open_metrics["WMAPE"],
            "RMSLE": open_metrics["RMSLE"],
            "StoreDateAlignment": "PASS",
            "FallbackStores": int(
                status_df["FitStatus"].str.startswith(
                    "SeasonalNaive_Fallback"
                ).sum()
            ),
        },
    ])

    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    evaluation.to_csv(PREDICTION_PATH, index=False)
    results.to_csv(RESULTS_PATH, index=False)
    status_df.to_csv(STATUS_PATH, index=False)

    print("\nETS evaluation:")
    print(results.to_string(index=False))

    print("\nSaved:")
    print(PREDICTION_PATH)
    print(RESULTS_PATH)
    print(STATUS_PATH)

if __name__ == "__main__":
    main()