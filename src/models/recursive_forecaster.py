
import numpy as np
import pandas as pd


def recursive_forecast(
    base_data: pd.DataFrame,
    feature_builder,
    model,
    feature_columns: list,
    prediction_col: str = "PredictedSales",
    feature_capture: list | None = None,
) -> pd.DataFrame:
    """
    Generate multi-day recursive forecasts.

    Historical features are generated from the feature builder.
    Model predictions are appended to history after each forecast day.
    Actual future Sales are never used as model inputs.

    If feature_capture is provided, capture the exact feature rows
    passed to model.predict(), together with Store and Date metadata.
    """

    required = {"Store", "Date", "DayOfWeek", "Open"}
    missing = required - set(base_data.columns)

    if missing:
        raise ValueError(
            f"Forecast data missing columns: {sorted(missing)}"
        )

    if not feature_columns:
        raise ValueError("feature_columns cannot be empty.")

    data = base_data.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    if data.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate Store-Date rows in forecast data.")

    missing_features = set(feature_columns) - (
        set(data.columns) | {
            "Sales_Lag_1", "Sales_Lag_7",
            "Sales_Lag_14", "Sales_Lag_28",
            "Sales_RollingMean_7", "Sales_RollingMean_14",
            "Sales_RollingMean_28",
            "Store_HistoricalMean",
            "Store_Weekday_HistoricalMean",
        }
    )

    if missing_features:
        raise ValueError(
            f"Missing model features: {sorted(missing_features)}"
        )

    results = []

    for forecast_date in sorted(data["Date"].unique()):

        day = (
            data.loc[data["Date"] == forecast_date]
            .sort_values("Store")
            .copy()
        )

        historical_features = feature_builder.build_day_features(day)

        model_data = pd.concat(
            [day, historical_features],
            axis=1
        )

        predictions = np.zeros(len(day), dtype=float)
        open_mask = day["Open"].to_numpy() == 1

        if open_mask.any():
            X = model_data.loc[open_mask, feature_columns]

            if X.isna().all(axis=1).any():
                raise ValueError(
                    f"All model features missing for some stores on "
                    f"{pd.Timestamp(forecast_date).date()}."
                )

            # Capture precisely the features used by the model.
            if feature_capture is not None:
                captured = X.copy()
                captured.insert(0, "Date", pd.Timestamp(forecast_date))
                feature_capture.append(captured)

            predictions[open_mask] = model.predict(X)

        predictions = np.maximum(predictions, 0)

        day[prediction_col] = predictions

        # Only predictions (including zero for closed stores)
        # are appended to recursive history.
        feature_builder.update_history(day, predictions)

        results.append(day)

    return pd.concat(results, ignore_index=True)
