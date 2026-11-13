
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
    "Store", "Date", "Sales", "PredictedSales",
    "Open", "Promo", "Promo2_Active"
]


def validate_data(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            "Check whether Promo and Promo2_Active are retained "
            "in the prediction output."
        )

    if df[["Store", "Date"]].duplicated().any():
        raise ValueError("Duplicate Store-Date combinations found.")

    for col in ["Sales", "PredictedSales", "Open", "Promo", "Promo2_Active"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    for col in ["Open", "Promo", "Promo2_Active"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 and 1.")

    if not np.isfinite(df[["Sales", "PredictedSales"]]).all().all():
        raise ValueError("Non-finite actual or predicted sales found.")

    if (df[["Sales", "PredictedSales"]] < 0).any().any():
        raise ValueError("Negative sales or predictions found.")

    return df


def calculate_metrics(group):
    actual = group["Sales"].to_numpy(dtype=float)
    predicted = group["PredictedSales"].to_numpy(dtype=float)

    error = predicted - actual
    abs_error = np.abs(error)

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
            abs_error.sum() / actual_total * 100
            if actual_total > 0 else np.nan
        ),
        "RMSLE": np.sqrt(
            np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)
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


def analyse_grouping(df, grouping_column, grouping_name):
    rows = []

    for value, label in [(0, "Inactive"), (1, "Active")]:
        group = df.loc[df[grouping_column] == value].copy()

        if group.empty:
            continue

        metrics = calculate_metrics(group)
        metrics["Analysis"] = grouping_name
        metrics["Segment"] = label
        rows.append(metrics)

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = validate_data(df)

    print("Prediction data validation: PASS")
    print(f"Total validation rows: {len(df):,}")

    # Focus on open stores for meaningful demand-error analysis.
    open_df = df.loc[df["Open"] == 1].copy()

    print(f"Open-store observations: {len(open_df):,}")
    print(
        f"Daily Promo active observations: "
        f"{(open_df['Promo'] == 1).sum():,}"
    )
    print(
        f"Promo2 active observations: "
        f"{(open_df['Promo2_Active'] == 1).sum():,}"
    )

    results = []

    # 1. Daily promotions.
    results.extend(
        analyse_grouping(open_df, "Promo", "Daily Promo")
    )

    # 2. Continuous/recurring Promo2 programme.
    results.extend(
        analyse_grouping(open_df, "Promo2_Active", "Promo2 Active")
    )

    # 3. Combined interaction: both promotion mechanisms.
    open_df["PromotionCombination"] = np.select(
        [
            (open_df["Promo"] == 0) & (open_df["Promo2_Active"] == 0),
            (open_df["Promo"] == 1) & (open_df["Promo2_Active"] == 0),
            (open_df["Promo"] == 0) & (open_df["Promo2_Active"] == 1),
            (open_df["Promo"] == 1) & (open_df["Promo2_Active"] == 1),
        ],
        [
            "Neither Active",
            "Daily Promo Only",
            "Promo2 Only",
            "Both Active",
        ],
        default="Unknown",
    )

    for segment in [
        "Neither Active",
        "Daily Promo Only",
        "Promo2 Only",
        "Both Active",
    ]:
        group = open_df.loc[
            open_df["PromotionCombination"] == segment
        ]

        if group.empty:
            continue

        metrics = calculate_metrics(group)
        metrics["Analysis"] = "Combined Promotion"
        metrics["Segment"] = segment
        results.append(metrics)

    results_df = pd.DataFrame(results)

    column_order = [
        "Analysis", "Segment", "Observations", "Stores",
        "ActualSales", "PredictedSales", "MeanActualSales",
        "MAE", "RMSE", "WMAPE_Pct", "RMSLE",
        "MeanBias", "TotalBias",
        "Underforecast_Count", "Underforecast_Rate_Pct",
        "Overforecast_Count", "Overforecast_Rate_Pct",
        "Average_Underforecast", "Average_Overforecast",
    ]

    results_df = results_df[column_order]

    output_path = OUTPUT_DIR / "promotion_performance.csv"
    results_df.to_csv(output_path, index=False)

    display_columns = [
        "Analysis", "Segment", "Observations", "Stores",
        "MeanActualSales", "MAE", "RMSE", "WMAPE_Pct",
        "RMSLE", "MeanBias",
        "Underforecast_Rate_Pct", "Overforecast_Rate_Pct",
    ]

    print("\nPromotion performance:")
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
    print("PROMOTION EVALUATION COMPLETE")
    print("========================================")
    print(f"Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
