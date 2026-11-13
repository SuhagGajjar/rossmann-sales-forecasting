
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.build_features import ROOT


# ============================================================
# 1. CONFIGURATION
# ============================================================

PREDICTION_PATH = (
    ROOT
    / "models"
    / "experiments"
    / "lightgbm_second_window"
    / "second_window_tuned_lightgbm_predictions.csv"
)

OUTPUT_DIR = ROOT / "src" / "evaluation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_START = pd.Timestamp("2015-06-20")
EXPECTED_END = pd.Timestamp("2015-07-31")


# ============================================================
# 2. VALIDATION
# ============================================================

def load_predictions():

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(PREDICTION_PATH)

    df = pd.read_csv(PREDICTION_PATH)

    required_columns = {
        "Store",
        "Date",
        "Sales",
        "PredictedSales",
        "Open",
        "StoreType",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    numeric_columns = [
        "Sales",
        "PredictedSales",
        "Open",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[["Sales", "PredictedSales"]].isna().any().any():
        raise ValueError("Missing actual or predicted sales.")

    if not np.isfinite(
        df[["Sales", "PredictedSales"]].to_numpy()
    ).all():
        raise ValueError("Non-finite sales values found.")

    if (df["Sales"] < 0).any():
        raise ValueError("Negative actual sales found.")

    if (df["PredictedSales"] < 0).any():
        raise ValueError("Negative predictions found.")

    if df.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate Store-Date observations.")

    if df["Date"].min() != EXPECTED_START:
        raise ValueError("Unexpected validation start.")

    if df["Date"].max() != EXPECTED_END:
        raise ValueError("Unexpected validation end.")

    expected_rows = 1115 * 42

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows; "
            f"found {len(df):,}."
        )

    if not df["Open"].isin([0, 1]).all():
        raise ValueError("Unexpected Open values.")

    # Forecast error: positive means overforecast.
    df["Error"] = df["PredictedSales"] - df["Sales"]
    df["AbsoluteError"] = df["Error"].abs()

    # Directional errors.
    df["Underforecast"] = (
        df["Sales"] - df["PredictedSales"]
    ).clip(lower=0)

    df["Overforecast"] = (
        df["PredictedSales"] - df["Sales"]
    ).clip(lower=0)

    print("Prediction data validation: PASS")
    print(f"Rows: {len(df):,}")
    print(f"Stores: {df['Store'].nunique():,}")
    print(f"Dates: {df['Date'].nunique():,}")
    print(f"Open observations: {(df['Open'] == 1).sum():,}")

    return df


# ============================================================
# 3. METRIC CALCULATION
# ============================================================

def calculate_group_metrics(group):

    actual = group["Sales"].to_numpy(dtype=float)
    predicted = group["PredictedSales"].to_numpy(dtype=float)
    error = predicted - actual

    absolute_error = np.abs(error)

    total_actual = np.abs(actual).sum()

    wmape = (
        absolute_error.sum() / total_actual * 100
        if total_actual > 0
        else np.nan
    )

    under_mask = error < 0
    over_mask = error > 0

    under_count = int(under_mask.sum())
    over_count = int(over_mask.sum())
    total_count = len(group)

    return {
        "Observations": total_count,
        "ActualSales": float(actual.sum()),
        "PredictedSales": float(predicted.sum()),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "WMAPE_Pct": float(wmape),
        "MeanBias": float(error.mean()),
        "TotalBias": float(error.sum()),
        "Underforecast_Count": under_count,
        "Underforecast_Rate_Pct": (
            under_count / total_count * 100
        ),
        "Overforecast_Count": over_count,
        "Overforecast_Rate_Pct": (
            over_count / total_count * 100
        ),
        "Average_Underforecast": (
            float(group.loc[under_mask, "Underforecast"].mean())
            if under_count > 0 else 0.0
        ),
        "Average_Overforecast": (
            float(group.loc[over_mask, "Overforecast"].mean())
            if over_count > 0 else 0.0
        ),
    }


# ============================================================
# 4. STORE-LEVEL PERFORMANCE
# ============================================================

def analyze_store_performance(df):

    # Open-store observations are primary for operational
    # error direction and magnitude.
    open_df = df.loc[df["Open"] == 1].copy()

    records = []

    for store, group in open_df.groupby("Store"):

        metrics = calculate_group_metrics(group)

        store_type = group["StoreType"].mode().iloc[0]

        records.append({
            "Store": store,
            "StoreType": store_type,
            **metrics,
        })

    result = pd.DataFrame(records)

    result = result.sort_values(
        "WMAPE_Pct",
        ascending=False,
    )

    result.to_csv(
        OUTPUT_DIR / "store_level_performance.csv",
        index=False,
    )

    print("\nStore-level performance calculated.")
    print(f"Stores evaluated: {len(result):,}")

    return result


# ============================================================
# 5. STORES WITH BEST AND WORST PERFORMANCE
# ============================================================

def summarize_store_rankings(store_results):

    worst = store_results.head(20).copy()

    best = store_results.sort_values(
        "WMAPE_Pct",
        ascending=True,
    ).head(20).copy()

    worst.to_csv(
        OUTPUT_DIR / "worst_20_stores.csv",
        index=False,
    )

    best.to_csv(
        OUTPUT_DIR / "best_20_stores.csv",
        index=False,
    )

    print("\nWorst 10 stores by WMAPE:")
    print(
        worst.head(10)[
            [
                "Store",
                "StoreType",
                "Observations",
                "WMAPE_Pct",
                "MAE",
                "MeanBias",
                "Underforecast_Rate_Pct",
            ]
        ].to_string(index=False)
    )

    print("\nBest 10 stores by WMAPE:")
    print(
        best.head(10)[
            [
                "Store",
                "StoreType",
                "Observations",
                "WMAPE_Pct",
                "MAE",
                "MeanBias",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 6. STORETYPE-LEVEL PERFORMANCE
# ============================================================

def analyze_storetype_performance(df, store_results):

    open_df = df.loc[df["Open"] == 1].copy()

    records = []

    for store_type, group in open_df.groupby("StoreType"):

        pooled_metrics = calculate_group_metrics(group)

        store_subset = store_results.loc[
            store_results["StoreType"] == store_type
        ]

        records.append({
            "StoreType": store_type,
            "NumberOfStores": store_subset["Store"].nunique(),
            "Mean_Store_WMAPE": store_subset["WMAPE_Pct"].mean(),
            "Median_Store_WMAPE": store_subset["WMAPE_Pct"].median(),
            "Min_Store_WMAPE": store_subset["WMAPE_Pct"].min(),
            "Max_Store_WMAPE": store_subset["WMAPE_Pct"].max(),
            **pooled_metrics,
        })

    result = pd.DataFrame(records).sort_values(
        "WMAPE_Pct",
        ascending=True,
    )

    result.to_csv(
        OUTPUT_DIR / "storetype_level_performance.csv",
        index=False,
    )

    print("\nStoreType-level performance:")
    print(result.to_string(index=False))

    return result


# ============================================================
# 7. ALL-STORE VS OPEN-STORE CHECK
# ============================================================

def analyze_all_vs_open(df):

    all_metrics = calculate_group_metrics(df)

    open_df = df.loc[df["Open"] == 1].copy()
    open_metrics = calculate_group_metrics(open_df)

    comparison = pd.DataFrame([
        {"EvaluationScope": "All Store-Dates", **all_metrics},
        {"EvaluationScope": "Open Store-Dates", **open_metrics},
    ])

    comparison.to_csv(
        OUTPUT_DIR / "all_vs_open_performance.csv",
        index=False,
    )

    print("\nAll-store vs open-store performance:")
    print(comparison.to_string(index=False))

    return comparison


# ============================================================
# 8. MAIN
# ============================================================

def main():

    df = load_predictions()

    store_results = analyze_store_performance(df)

    summarize_store_rankings(store_results)

    storetype_results = analyze_storetype_performance(
        df,
        store_results,
    )

    scope_results = analyze_all_vs_open(df)

    print("\n========================================")
    print("STORE-LEVEL EVALUATION COMPLETE")
    print("========================================")

    print("Outputs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
