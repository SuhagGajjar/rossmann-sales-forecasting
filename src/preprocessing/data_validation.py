
from pathlib import Path

import pandas as pd


def load_datasets(data_dir="data/raw"):
    """Load original Rossmann competition datasets."""
    data_path = Path(data_dir)

    train = pd.read_csv(
        data_path / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False
    )

    test = pd.read_csv(
        data_path / "test.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False
    )

    store = pd.read_csv(data_path / "store.csv")

    return train, test, store


def build_audit_reports(train, test, store):
    """Create structured audit tables for Excel export."""

    datasets = {
        "train": train,
        "test": test,
        "store": store
    }

    overview_rows = []
    missing_rows = []
    duplicate_rows = []

    for name, df in datasets.items():
        overview_rows.append({
            "Dataset": name,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Unique Stores": df["Store"].nunique() if "Store" in df else None,
            "Unique Dates": df["Date"].nunique() if "Date" in df else None,
            "Start Date": df["Date"].min() if "Date" in df else None,
            "End Date": df["Date"].max() if "Date" in df else None
        })

        for column in df.columns:
            missing_rows.append({
                "Dataset": name,
                "Column": column,
                "Data Type": str(df[column].dtype),
                "Missing Count": df[column].isna().sum(),
                "Missing Percentage": round(
                    df[column].isna().mean() * 100, 3
                )
            })

        duplicate_rows.append({
            "Dataset": name,
            "Exact Duplicate Rows": df.duplicated().sum()
        })

    overview = pd.DataFrame(overview_rows)
    missing = pd.DataFrame(missing_rows)
    duplicates = pd.DataFrame(duplicate_rows)

    sales_summary = train["Sales"].describe(
        percentiles=[0.01, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).rename_axis("Statistic").reset_index(name="Sales Value")

    sales_by_open = (
        train.groupby("Open")["Sales"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    zero_sales = pd.DataFrame([
        {
            "Category": "All zero-sales rows",
            "Count": (train["Sales"] == 0).sum()
        },
        {
            "Category": "Closed store rows",
            "Count": (train["Open"] == 0).sum()
        },
        {
            "Category": "Zero sales despite Open=1",
            "Count": ((train["Sales"] == 0) & (train["Open"] == 1)).sum()
        },
        {
            "Category": "Negative sales rows",
            "Count": (train["Sales"] < 0).sum()
        }
    ])

    train_stores = set(train["Store"].unique())
    test_stores = set(test["Store"].unique())
    store_master = set(store["Store"].unique())

    coverage = pd.DataFrame([
        {"Check": "Unique train stores", "Count": len(train_stores)},
        {"Check": "Unique test stores", "Count": len(test_stores)},
        {"Check": "Store master records", "Count": len(store_master)},
        {"Check": "Train stores missing from store master",
         "Count": len(train_stores - store_master)},
        {"Check": "Test stores missing from store master",
         "Count": len(test_stores - store_master)},
        {"Check": "Train/test date overlap",
         "Count": len(set(train["Date"]) & set(test["Date"]))}
    ])

    risks = pd.DataFrame([
        {
            "Risk": "Customers unavailable in test",
            "Impact": "Training-inference mismatch and leakage risk",
            "Detection": "Compare train and test feature availability",
            "Treatment": "Exclude actual future Customers from production features"
        },
        {
            "Risk": "Missing Open values in test",
            "Impact": "Potential incorrect forecasts for closed stores",
            "Detection": "Inspect all 11 missing Open records",
            "Treatment": "Investigate and document a closed-store assumption if supported"
        },
        {
            "Risk": "Missing Promo2 timing attributes",
            "Impact": "Incorrect promotion-related features",
            "Detection": "Cross-check missingness against Promo2 status",
            "Treatment": "Treat structural absence separately from unknown values"
        },
        {
            "Risk": "Missing CompetitionDistance",
            "Impact": "Incomplete competition context",
            "Detection": "Inspect the three affected stores",
            "Treatment": "Use justified imputation and retain missingness indicator if useful"
        },
        {
            "Risk": "Zero sales while Open=1",
            "Impact": "Potentially misleading demand signal",
            "Detection": "Inspect store, date, promotion and customer context",
            "Treatment": "Investigate before deciding whether to retain or flag"
        },
        {
            "Risk": "Extreme sales values",
            "Impact": "Potential model sensitivity",
            "Detection": "Inspect high quantiles and store/date context",
            "Treatment": "Retain genuine demand peaks; correct only verified errors"
        },
        {
            "Risk": "Mixed StateHoliday data types",
            "Impact": "Inconsistent preprocessing and feature encoding",
            "Detection": "Inspect inferred column types",
            "Treatment": "Standardise StateHoliday as string during loading"
        },
        {
            "Risk": "Test horizon differs from six-week brief",
            "Impact": "Potential mismatch between evaluation and business requirement",
            "Detection": "Compare actual test date range with 42-day horizon",
            "Treatment": "Document both and define the evaluation window explicitly"
        }
    ])

    return {
        "Dataset Overview": overview,
        "Missing Values": missing,
        "Duplicates": duplicates,
        "Sales Summary": sales_summary,
        "Sales by Open": sales_by_open,
        "Zero Sales": zero_sales,
        "Store Coverage": coverage,
        "Risk Register": risks
    }


def export_audit_to_excel(audit_tables, output_path="reports/data_audit.xlsx"):
    """Save all audit tables into a multi-sheet Excel workbook."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in audit_tables.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(max_length + 3, 55)

    print(f"\nExcel audit report saved to: {output.resolve()}")


def audit_datasets(train, test, store):
    """Print a concise audit summary."""

    for name, df in {
        "TRAIN": train,
        "TEST": test,
        "STORE": store
    }.items():
        print(f"\n{'=' * 55}")
        print(f"{name} DATASET")
        print(f"{'=' * 55}")
        print(f"Shape: {df.shape}")
        print(f"Missing values: {df.isna().sum().sum()}")
        print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nSales summary:")
    print(train["Sales"].describe())

    print("\nZero-sales observations:")
    print((train["Sales"] == 0).sum())

    print("\nZero sales despite Open=1:")
    print(((train["Sales"] == 0) & (train["Open"] == 1)).sum())


if __name__ == "__main__":
    train, test, store = load_datasets()

    audit_datasets(train, test, store)

    audit_tables = build_audit_reports(train, test, store)

    export_audit_to_excel(audit_tables)
