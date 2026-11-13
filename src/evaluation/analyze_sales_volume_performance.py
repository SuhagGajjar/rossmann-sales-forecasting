
from pathlib import Path
import json

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
]

SEGMENT_LABELS = {
    4: ["Low", "Lower-Middle", "Upper-Middle", "High"],
    3: ["Low", "Medium", "High"],
    2: ["Low", "High"],
    1: ["All"],
}


def validate_predictions(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[["Store", "Date"]].duplicated().any():
        raise ValueError("Duplicate Store-Date combinations found.")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    for col in ["Sales", "PredictedSales", "Open"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not df["Open"].isin([0, 1]).all():
        raise ValueError("Open must contain only 0 and 1.")

    if not np.isfinite(df[["Sales", "PredictedSales"]]).all().all():
        raise ValueError("Non-finite sales or prediction values found.")

    if (df[["Sales", "PredictedSales"]] < 0).any().any():
        raise ValueError("Negative sales or predictions found.")

    if df.empty:
        raise ValueError("Prediction dataset is empty.")

    return df


def calculate_metrics(group, total_actual_sales):
    actual = group["Sales"].to_numpy(dtype=float)
    predicted = group["PredictedSales"].to_numpy(dtype=float)

    error = predicted - actual
    absolute_error = np.abs(error)

    underforecast = np.maximum(actual - predicted, 0)
    overforecast = np.maximum(predicted - actual, 0)

    n = len(group)
    actual_sum = actual.sum()

    wmape = (
        absolute_error.sum() / actual_sum * 100
        if actual_sum > 0
        else np.nan
    )

    rmsle = np.sqrt(
        np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)
    )

    under_count = int((error < 0).sum())
    over_count = int((error > 0).sum())

    return {
        "Observations": n,
        "ActualSales": actual_sum,
        "PredictedSales": predicted.sum(),
        "MeanActualSales": actual.mean(),
        "MedianActualSales": np.median(actual),
        "MinActualSales": actual.min(),
        "MaxActualSales": actual.max(),
        "ActualSalesShare_Pct": (
            actual_sum / total_actual_sales * 100
            if total_actual_sales > 0 else np.nan
        ),
        "ObservationShare_Pct": np.nan,  # populated after aggregation
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WMAPE_Pct": wmape,
        "RMSLE": rmsle,
        "MeanBias": error.mean(),
        "TotalBias": error.sum(),
        "Underforecast_Count": under_count,
        "Underforecast_Rate_Pct": under_count / n * 100,
        "Overforecast_Count": over_count,
        "Overforecast_Rate_Pct": over_count / n * 100,
        "Average_Underforecast": (
            underforecast[error < 0].mean()
            if under_count else 0.0
        ),
        "Average_Overforecast": (
            overforecast[error > 0].mean()
            if over_count else 0.0
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = validate_predictions(df)

    print("Prediction data validation: PASS")
    print(f"Total rows: {len(df):,}")
    print(f"Open observations: {(df['Open'] == 1).sum():,}")

    # Sales-volume analysis is performed on open-store observations only.
    open_df = df.loc[df["Open"] == 1].copy()

    if open_df.empty:
        raise ValueError("No open-store observations available.")

    # Calculate quartile boundaries from actual validation sales.
    quantiles = open_df["Sales"].quantile([0.25, 0.50, 0.75])

    # Keep unique internal boundaries strictly inside the observed range.
    min_sales = float(open_df["Sales"].min())
    max_sales = float(open_df["Sales"].max())

    internal_edges = sorted({
        float(value)
        for value in quantiles.values
        if min_sales < value < max_sales
    })

    edges = [-np.inf] + internal_edges + [np.inf]

    number_of_segments = len(edges) - 1
    labels = SEGMENT_LABELS.get(
        number_of_segments,
        [f"Segment_{i + 1}" for i in range(number_of_segments)]
    )

    open_df["SalesVolumeSegment"] = pd.cut(
        open_df["Sales"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    if open_df["SalesVolumeSegment"].isna().any():
        raise ValueError("Some observations could not be assigned a volume segment.")

    total_actual_sales = open_df["Sales"].sum()

    results = []

    for segment in labels:
        group = open_df.loc[
            open_df["SalesVolumeSegment"] == segment
        ].copy()

        if group.empty:
            continue

        metrics = calculate_metrics(group, total_actual_sales)
        metrics["SalesVolumeSegment"] = segment
        metrics["ActualSalesLowerBound"] = group["Sales"].min()
        metrics["ActualSalesUpperBound"] = group["Sales"].max()
        metrics["ObservationShare_Pct"] = (
            len(group) / len(open_df) * 100
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    column_order = [
        "SalesVolumeSegment",
        "Observations",
        "ObservationShare_Pct",
        "ActualSalesLowerBound",
        "ActualSalesUpperBound",
        "MeanActualSales",
        "MedianActualSales",
        "MinActualSales",
        "MaxActualSales",
        "ActualSales",
        "ActualSalesShare_Pct",
        "PredictedSales",
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

    output_csv = OUTPUT_DIR / "sales_volume_performance.csv"
    results_df.to_csv(output_csv, index=False)

    # Save the exact boundaries and scope for reproducibility/documentation.
    metadata = {
        "analysis": "Sales-volume segment performance",
        "input_file": str(INPUT_PATH),
        "evaluation_scope": "Open-store observations only",
        "total_open_observations": int(len(open_df)),
        "total_actual_sales": float(total_actual_sales),
        "quantile_boundaries": {
            "25th_percentile": float(quantiles.loc[0.25]),
            "50th_percentile": float(quantiles.loc[0.50]),
            "75th_percentile": float(quantiles.loc[0.75]),
        },
        "segment_edges": [
            None if np.isinf(edge) else float(edge)
            for edge in edges
        ],
        "segment_labels": labels,
        "segmentation_note": (
            "Segments are diagnostic groups defined using actual sales "
            "within the validation window. Actual sales are not model inputs."
        ),
    }

    output_json = OUTPUT_DIR / "sales_volume_performance_metadata.json"
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nSales-volume boundaries:")
    print(f"25th percentile: {quantiles.loc[0.25]:,.2f}")
    print(f"50th percentile: {quantiles.loc[0.50]:,.2f}")
    print(f"75th percentile: {quantiles.loc[0.75]:,.2f}")

    print("\nSales-volume segment performance:")
    display_columns = [
        "SalesVolumeSegment",
        "Observations",
        "MeanActualSales",
        "MAE",
        "RMSE",
        "WMAPE_Pct",
        "RMSLE",
        "MeanBias",
        "Underforecast_Rate_Pct",
        "Overforecast_Rate_Pct",
        "ActualSalesShare_Pct",
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
                "ActualSalesShare_Pct": "{:.2f}%".format,
            }
        )
    )

    print("\n========================================")
    print("SALES-VOLUME EVALUATION COMPLETE")
    print("========================================")
    print(f"Results: {output_csv.resolve()}")
    print(f"Metadata: {output_json.resolve()}")


if __name__ == "__main__":
    main()
