
from collections import defaultdict

import numpy as np
import pandas as pd


LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


class RecursiveFeatureBuilder:
    """
    Builds leakage-safe historical features for recursive forecasting.

    Historical observations are actual sales before the forecast origin.
    During forecasting, predicted sales are appended to the history.
    """

    def __init__(self, historical_data: pd.DataFrame):

        required = {"Store", "Date", "DayOfWeek", "Sales"}
        missing = required - set(historical_data.columns)

        if missing:
            raise ValueError(
                f"Historical data missing columns: {sorted(missing)}"
            )

        history = historical_data[
            ["Store", "Date", "DayOfWeek", "Sales"]
        ].copy()

        history["Date"] = pd.to_datetime(history["Date"])
        history = history.sort_values(["Store", "Date"])

        if history.duplicated(["Store", "Date"]).any():
            raise ValueError("Duplicate Store-Date in historical data.")

        self.state = {}

        for store, group in history.groupby("Store", sort=False):

            sales = group["Sales"].astype(float).tolist()
            weekdays = group["DayOfWeek"].astype(int).tolist()

            weekday_sums = defaultdict(float)
            weekday_counts = defaultdict(int)

            for weekday, value in zip(weekdays, sales):
                weekday_sums[weekday] += value
                weekday_counts[weekday] += 1

            self.state[int(store)] = {
                "sales": sales,
                "weekday_sums": weekday_sums,
                "weekday_counts": weekday_counts,
                "sales_sum": sum(sales),
                "sales_count": len(sales),
            }

    def build_day_features(
        self,
        current_rows: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate historical features for one forecast date.

        Current-day Sales is never accessed.
        """

        if current_rows["Date"].nunique() != 1:
            raise ValueError(
                "Pass rows for exactly one forecast date."
            )

        records = []

        for row in current_rows.itertuples(index=False):

            store = int(row.Store)
            weekday = int(row.DayOfWeek)

            if store not in self.state:
                raise ValueError(
                    f"No historical data available for Store {store}."
                )

            state = self.state[store]
            sales = state["sales"]

            features = {
                "Store": store,
            }

            # Historical lags.
            for lag in LAGS:
                features[f"Sales_Lag_{lag}"] = (
                    sales[-lag] if len(sales) >= lag else np.nan
                )

            # Historical rolling means.
            for window in ROLLING_WINDOWS:
                previous_sales = sales[-window:]

                features[f"Sales_RollingMean_{window}"] = (
                    float(np.mean(previous_sales))
                    if previous_sales else np.nan
                )

            # Store-wide historical mean.
            features["Store_HistoricalMean"] = (
                state["sales_sum"] / state["sales_count"]
                if state["sales_count"] > 0 else np.nan
            )

            # Same-store, same-weekday historical mean.
            count = state["weekday_counts"][weekday]

            features["Store_Weekday_HistoricalMean"] = (
                state["weekday_sums"][weekday] / count
                if count > 0 else np.nan
            )

            records.append(features)

        return pd.DataFrame(records, index=current_rows.index).drop(
            columns=["Store"]
        )

    def update_history(
        self,
        current_rows: pd.DataFrame,
        sales_values
    ):
        """
        Append actual or predicted sales to history.

        During recursive validation/test, sales_values must be
        model predictions (or zero for closed stores).
        """

        sales_values = np.asarray(sales_values, dtype=float)

        if len(sales_values) != len(current_rows):
            raise ValueError(
                "Prediction count does not match current forecast rows."
            )

        if not np.isfinite(sales_values).all():
            raise ValueError("Predictions contain non-finite values.")

        sales_values = np.maximum(sales_values, 0)

        for row, prediction in zip(
            current_rows.itertuples(index=False),
            sales_values
        ):

            store = int(row.Store)
            weekday = int(row.DayOfWeek)

            if store not in self.state:
                raise ValueError(
                    f"No historical state for Store {store}."
                )

            state = self.state[store]

            state["sales"].append(float(prediction))
            state["sales_sum"] += float(prediction)
            state["sales_count"] += 1

            state["weekday_sums"][weekday] += float(prediction)
            state["weekday_counts"][weekday] += 1
