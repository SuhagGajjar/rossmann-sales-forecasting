
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_PATH = Path(
    "models/experiments/lightgbm_second_window/"
    "second_window_tuned_lightgbm_predictions.csv"
)
OUTPUT_DIR = Path("src/evaluation/results")

REQUIRED_COLUMNS = [
    "Store",
    "Date",
    "Sales",
    "PredictedSales",
    "Open",
    "StateHoliday",
    "SchoolHoliday",
]


def validate_data(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[["Store", "Date"]].duplicated().any():
        raise ValueError("Duplicate Store-Date combinations found.")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    for col in ["Sales", "PredictedSales", "Open", "SchoolHoliday"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not df["Open"].isin([0, 1]).all():
        raise ValueError("Open must contain only 0 and 1.")

    if not df["SchoolHoliday"].isin([0, 1]).all():
        raise ValueError("SchoolHoliday must contain only 0 and 1.")

    if not np.isfinite(df[["Sales", "PredictedSales"]]).all().all():
        raise ValueError("Non-finite actual or predicted sales found.")

    if (df[["Sales", "PredictedSales"]] < 0).any().any():
        raise ValueError("Negative sales or predictions found.")

    # Normalise holiday categories read from CSV.
    df["StateHoliday"] = (
        df["StateHoliday"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "0.0": "0",
            "nan": "0",
            "none": "0",
        })
    )

    return df


def calculate_metrics(group):
    actual = group["Sales"].to_numpy(dtype=float)
    predicted = group["PredictedSales"].to_numpy(dtype=float)

    error = predicted - actual
    absolute_error = np.abs(error)

    under = np.maximum(actual - predicted, 0)
    over = np.maximum(predicted - actual, 0)

    n = len(group)
    actual_total = actual.sum()

    under_count = int((error < 0).sum())
    over_count = int((error > 0).sum())

    return {
        "Observations": n,
        "Stores": group["Store"].nunique(),
        "ActualSales": actual_total,
        "PredictedSales": predicted.sum(),
        "MeanActualSales": actual.mean(),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WMAPE_Pct": (
            absolute_error.sum() / actual_total * 100
            if actual_total > 0 else np.nan
        ),
        "RMSLE": np.sqrt(
            np.mean(
                (np.log1p(predicted) - np.log1p(actual)) ** 2
            )
        ),
        "MeanBias": error.mean(),
        "TotalBias": error.sum(),
        "Underforecast_Count": under_count,
        "Underforecast_Rate_Pct": under_count / n * 100,
        "Overforecast_Count": over_count,
        "Overforecast_Rate_Pct": over_count / n * 100,
        "Average_Underforecast": (
            under[error < 0].mean() if under_count else 0.0
        ),
        "Average_Overforecast": (
            over[error > 0].mean() if over_count else 0.0
        ),
    }


def append_group_results(rows, df, analysis, column, mapping=None):
    for value in sorted(df[column].dropna().unique()):
        group = df.loc[df[column] == value].copy()

        if group.empty:
            continue

        metrics = calculate_metrics(group)
        metrics["Analysis"] = analysis
        metrics["Segment"] = (
            mapping.get(value, str(value))
            if mapping else str(value)
        )
        rows.append(metrics)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = validate_data(df)

    # Evaluate demand only on open-store observations.
    open_df = df.loc[df["Open"] == 1].copy()

    if open_df.empty:
        raise ValueError("No open-store observations found.")

    print("Prediction data validation: PASS")
    print(f"Total validation rows: {len(df):,}")
    print(f"Open-store observations: {len(open_df):,}")

    # Derive calendar attributes from actual dates.
    # pandas Monday=0 through Sunday=6.
    open_df["WeekdayNumber"] = open_df["Date"].dt.dayofweek
    open_df["WeekdayName"] = open_df["Date"].dt.day_name()
    open_df["MonthNumber"] = open_df["Date"].dt.month
    open_df["MonthName"] = open_df["Date"].dt.strftime("%B")

    open_df["WeekPart"] = np.where(
        open_df["WeekdayNumber"] >= 5,
        "Weekend",
        "Weekday",
    )

    rows = []

    # 1. Day of week.
    weekday_mapping = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    append_group_results(
        rows,
        open_df,
        "Day of Week",
        "WeekdayNumber",
        weekday_mapping,
    )

    # 2. Weekday versus weekend.
    append_group_results(
        rows,
        open_df,
        "Weekday vs Weekend",
        "WeekPart",
    )

    # 3. Month.
    month_mapping = {
        1: "January", 2: "February", 3: "March",
        4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September",
        10: "October", 11: "November", 12: "December",
    }

    append_group_results(
        rows,
        open_df,
        "Month",
        "MonthNumber",
        month_mapping,
    )

    # 4. State holiday.
    state_holiday_mapping = {
        "0": "No State Holiday",
        "a": "Public Holiday",
        "b": "Easter Holiday",
        "c": "Christmas Holiday",
    }

    unknown_state_values = (
        set(open_df["StateHoliday"].unique())
        - set(state_holiday_mapping.keys())
    )

    if unknown_state_values:
        raise ValueError(
            f"Unexpected StateHoliday values: {unknown_state_values}"
        )

    append_group_results(
        rows,
        open_df,
        "State Holiday",
        "StateHoliday",
        state_holiday_mapping,
    )

    # 5. School holiday.
    school_holiday_mapping = {
        0: "No School Holiday",
        1: "School Holiday",
    }

    append_group_results(
        rows,
        open_df,
        "School Holiday",
        "SchoolHoliday",
        school_holiday_mapping,
    )

    results_df = pd.DataFrame(rows)

    column_order = [
        "Analysis",
        "Segment",
        "Observations",
        "Stores",
        "ActualSales",
        "PredictedSales",
        "MeanActualSales",
        "MAE",
        "RMSE",
        "WMAPE_Pct",
        "RMSLE",
        "MeanBias",
        "TotalBias",
        "Underforecast_Count",
        "Underforecast_Rate_Pct",
        "Overforecast_Count",
        "Overforecast_Rate_Pct",
        "Average_Underforecast",
        "Average_Overforecast",
    ]

    results_df = results_df[column_order]

    output_path = OUTPUT_DIR / "calendar_performance.csv"
    results_df.to_csv(output_path, index=False)

    print("\nCalendar performance:")
    display_columns = [
        "Analysis",
        "Segment",
        "Observations",
        "Stores",
        "MeanActualSales",
        "MAE",
        "RMSE",
        "WMAPE_Pct",
        "RMSLE",
        "MeanBias",
        "Underforecast_Rate_Pct",
        "Overforecast_Rate_Pct",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            formatters={
                "MeanActualSales": "{:,.2f}".format,
                "MAE": "{:,.2f}".format,
                "RMSE": "{:,.2f}".format,
                "WMAPE_Pct": "{:.2f}%".format,
                "RMSLE": "{:.4f}".format,
                "MeanBias": "{:,.2f}".format,
                "Underforecast_Rate_Pct": "{:.2f}%".format,
                "Overforecast_Rate_Pct": "{:.2f}%".format,
            },
        )
    )

    print("\n========================================")
    print("CALENDAR EVALUATION COMPLETE")
    print("========================================")
    print(f"Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
