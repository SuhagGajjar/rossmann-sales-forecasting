
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")
REPORT_DIR = Path("reports")


def load_data():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False
    )

    test = pd.read_csv(
        DATA_DIR / "test.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False
    )

    store = pd.read_csv(DATA_DIR / "store.csv")

    return train, test, store


def investigate(train, test, store):
    findings = []
    sheets = {}

    def add_finding(title, observation, implication):
        findings.append({
            "Finding": title,
            "Observation": observation,
            "Implication": implication
        })

    # --------------------------------------------------
    # 1. Dataset coverage and integrity
    # --------------------------------------------------
    duplicate_keys = train[
        train.duplicated(["Store", "Date"], keep=False)
    ]

    train_stores = set(train["Store"].unique())
    test_stores = set(test["Store"].unique())
    master_stores = set(store["Store"].unique())

    train_dates = sorted(train["Date"].unique())
    test_dates = sorted(test["Date"].unique())

    store_coverage = (
        train.groupby("Store")["Date"]
        .agg(
            First_Date="min",
            Last_Date="max",
            Rows="count",
            Unique_Dates="nunique"
        )
        .reset_index()
    )

    sheets["Store Coverage"] = store_coverage

    add_finding(
        "Dataset size and coverage",
        (
            f"Train: {len(train):,} rows, {len(train_stores):,} stores, "
            f"{len(train_dates)} dates ({train['Date'].min().date()} to "
            f"{train['Date'].max().date()}). "
            f"Test: {len(test):,} rows, {len(test_stores):,} stores, "
            f"{len(test_dates)} dates ({test['Date'].min().date()} to "
            f"{test['Date'].max().date()}). "
            f"Store master: {len(master_stores):,} stores."
        ),
        (
            f"Store-date duplicate rows: {len(duplicate_keys):,}. "
            f"Train stores absent from master: "
            f"{len(train_stores - master_stores)}. "
            f"Test stores absent from master: "
            f"{len(test_stores - master_stores)}. "
            f"Test stores absent from train: {len(test_stores - train_stores)}."
        )
    )

    sheets["Duplicate Store Dates"] = duplicate_keys

    # --------------------------------------------------
    # 2. Missing Open values in test
    # --------------------------------------------------
    missing_open = test[test["Open"].isna()].copy()

    historical_open_rate = (
        train.groupby(["Store", "DayOfWeek"])["Open"]
        .mean()
        .rename("Historical_Store_Weekday_Open_Rate")
        .reset_index()
    )

    missing_open = missing_open.merge(
        historical_open_rate,
        on=["Store", "DayOfWeek"],
        how="left"
    )

    store_open_rate = (
        train.groupby("Store")["Open"]
        .mean()
        .rename("Historical_Store_Open_Rate")
        .reset_index()
    )

    missing_open = missing_open.merge(
        store_open_rate, on="Store", how="left"
    )

    sheets["Test Missing Open"] = missing_open

    add_finding(
        "Missing Open values in test",
        (
            f"{len(missing_open)} test records have missing Open values. "
            f"Historical store/weekday opening rates are attached for review."
        ),
        (
            "Do not automatically impute these as open or closed. "
            "Investigate the affected records and define an explicit "
            "production fallback."
        )
    )

    # --------------------------------------------------
    # 3. Zero sales despite Open=1
    # --------------------------------------------------
    zero_open = train[
        (train["Open"] == 1) & (train["Sales"] == 0)
    ].copy()

    zero_open["Customers_Zero"] = zero_open["Customers"].eq(0)

    zero_summary = pd.DataFrame({
        "Measure": [
            "Open and zero sales",
            "Of which customers are zero",
            "Of which customers are positive",
            "Share with zero customers (%)"
        ],
        "Value": [
            len(zero_open),
            int(zero_open["Customers"].eq(0).sum()),
            int(zero_open["Customers"].gt(0).sum()),
            round(zero_open["Customers"].eq(0).mean() * 100, 2)
            if len(zero_open) else 0
        ]
    })

    sheets["Zero Sales Open"] = zero_open
    sheets["Zero Sales Summary"] = zero_summary

    add_finding(
        "Zero sales while store is open",
        (
            f"{len(zero_open):,} records have Open=1 and Sales=0. "
            f"{zero_open['Customers'].eq(0).sum():,} also have zero "
            f"Customers; {zero_open['Customers'].gt(0).sum():,} "
            "have positive Customers."
        ),
        (
            "Retain these records pending investigation. Zero sales may "
            "represent genuine zero demand, operational issues, or data "
            "quality problems. Do not blindly remove them."
        )
    )

    # --------------------------------------------------
    # 4. Promo2 metadata missingness
    # --------------------------------------------------
    promo2_audit = (
        store.groupby("Promo2")
        .agg(
            Stores=("Store", "count"),
            Missing_Start_Week=("Promo2SinceWeek", lambda x: x.isna().sum()),
            Missing_Start_Year=("Promo2SinceYear", lambda x: x.isna().sum()),
            Missing_Interval=("PromoInterval", lambda x: x.isna().sum())
        )
        .reset_index()
    )

    sheets["Promo2 Missingness"] = promo2_audit

    promo2_off = store[store["Promo2"] == 0]
    promo2_on = store[store["Promo2"] == 1]

    add_finding(
        "Promo2 missing metadata",
        (
            f"Promo2=0 stores: {len(promo2_off)}. "
            f"Promo2=1 stores: {len(promo2_on)}. "
            f"Missing PromoInterval among Promo2=0: "
            f"{promo2_off['PromoInterval'].isna().sum()}. "
            f"Missing PromoInterval among Promo2=1: "
            f"{promo2_on['PromoInterval'].isna().sum()}."
        ),
        (
            "If missingness is confined to Promo2=0, it is likely "
            "structural rather than a data defect. Encode promotion "
            "participation separately from promotion timing."
        )
    )

    # --------------------------------------------------
    # 5. Competition metadata
    # --------------------------------------------------
    missing_competition = store[
        store["CompetitionDistance"].isna()
    ].copy()

    sheets["Missing Competition"] = missing_competition

    competition_missingness = pd.DataFrame({
        "Column": [
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear"
        ],
        "Missing_Count": [
            store["CompetitionDistance"].isna().sum(),
            store["CompetitionOpenSinceMonth"].isna().sum(),
            store["CompetitionOpenSinceYear"].isna().sum()
        ]
    })

    sheets["Competition Missingness"] = competition_missingness

    add_finding(
        "Competition information",
        (
            f"Missing CompetitionDistance: "
            f"{store['CompetitionDistance'].isna().sum()}. "
            f"Missing competition opening month/year: "
            f"{store['CompetitionOpenSinceMonth'].isna().sum()} / "
            f"{store['CompetitionOpenSinceYear'].isna().sum()}."
        ),
        (
            "Investigate whether missing values mean no known competitor "
            "or unknown information. Consider imputation plus a missingness "
            "indicator rather than assuming all missing values are equivalent."
        )
    )

    # --------------------------------------------------
    # 6. Sales distribution and extreme observations
    # --------------------------------------------------
    sales_quantiles = train["Sales"].quantile(
        [0, 0.5, 0.90, 0.95, 0.99, 0.995, 0.999, 1]
    ).rename_axis("Quantile").reset_index(name="Sales")

    sheets["Sales Quantiles"] = sales_quantiles

    top_sales = (
        train.nlargest(100, "Sales")
        .merge(store, on="Store", how="left")
    )

    sheets["Top 100 Sales"] = top_sales

    add_finding(
        "Sales distribution and extremes",
        (
            f"Median sales: {train['Sales'].median():,.0f}; "
            f"95th percentile: {train['Sales'].quantile(0.95):,.0f}; "
            f"99th percentile: {train['Sales'].quantile(0.99):,.0f}; "
            f"99.9th percentile: {train['Sales'].quantile(0.999):,.0f}; "
            f"maximum: {train['Sales'].max():,.0f}."
        ),
        (
            "Investigate extreme observations by store, promotion, "
            "holiday and customer count. Preserve genuine high-demand "
            "events; assess robust losses or transformations if needed."
        )
    )

    # --------------------------------------------------
    # 7. Promotion and opening patterns
    # --------------------------------------------------
    sales_open_promo = (
        train.groupby(["Open", "Promo"])
        .agg(
            Rows=("Sales", "size"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean")
        )
        .reset_index()
    )

    sheets["Sales Open Promo"] = sales_open_promo

    open_train = train[train["Open"] == 1].copy()

    promo_effect = (
        open_train.groupby("Promo")
        .agg(
            Observations=("Sales", "size"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean")
        )
        .reset_index()
    )

    sheets["Promotion Comparison"] = promo_effect

    promo_means = promo_effect.set_index("Promo")["Mean_Sales"]

    if 0 in promo_means.index and 1 in promo_means.index:
        promo_change = (
            promo_means.loc[1] / promo_means.loc[0] - 1
        ) * 100
        promo_text = f"Raw mean sales difference: {promo_change:+.1f}%."
    else:
        promo_text = "Promotion comparison could not be calculated."

    add_finding(
        "Promotion association",
        promo_text,
        (
            "This is an unadjusted association, not a causal promotion "
            "uplift estimate. Store, weekday, seasonality and selection "
            "effects must be considered before interpreting uplift."
        )
    )

    # --------------------------------------------------
    # 8. Weekday, month and holiday patterns
    # --------------------------------------------------
    sales_by_weekday = (
        open_train.groupby("DayOfWeek")
        .agg(
            Observations=("Sales", "size"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean")
        )
        .reset_index()
    )

    sheets["Sales by Weekday"] = sales_by_weekday

    sales_by_month = (
        open_train.assign(Month=open_train["Date"].dt.month)
        .groupby("Month")
        .agg(
            Observations=("Sales", "size"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean")
        )
        .reset_index()
    )

    sheets["Sales by Month"] = sales_by_month

    holiday_sales = (
        open_train.groupby(["StateHoliday", "SchoolHoliday"])
        .agg(
            Observations=("Sales", "size"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median")
        )
        .reset_index()
    )

    sheets["Holiday Sales"] = holiday_sales

    highest_day = sales_by_weekday.loc[
        sales_by_weekday["Mean_Sales"].idxmax()
    ]
    lowest_day = sales_by_weekday.loc[
        sales_by_weekday["Mean_Sales"].idxmin()
    ]

    add_finding(
        "Weekday sales pattern",
        (
            f"Highest mean sales: weekday {int(highest_day['DayOfWeek'])} "
            f"({highest_day['Mean_Sales']:,.0f}); "
            f"lowest: weekday {int(lowest_day['DayOfWeek'])} "
            f"({lowest_day['Mean_Sales']:,.0f})."
        ),
        (
            "Weekday is likely an important calendar feature. "
            "Check store-level variation before assuming a single "
            "weekday pattern applies to every store."
        )
    )

    highest_month = sales_by_month.loc[
        sales_by_month["Mean_Sales"].idxmax()
    ]
    lowest_month = sales_by_month.loc[
        sales_by_month["Mean_Sales"].idxmin()
    ]

    add_finding(
        "Monthly sales pattern",
        (
            f"Highest mean month: {int(highest_month['Month'])} "
            f"({highest_month['Mean_Sales']:,.0f}); "
            f"lowest: {int(lowest_month['Month'])} "
            f"({lowest_month['Mean_Sales']:,.0f})."
        ),
        (
            "Month and seasonal effects may be useful. Interpret carefully "
            "because this dataset covers fewer than three full years and "
            "store composition can affect monthly averages."
        )
    )

    # --------------------------------------------------
    # 9. Store profiles
    # --------------------------------------------------
    train_with_store = train.merge(
        store, on="Store", how="left"
    )

    store_profile = (
        train_with_store[train_with_store["Open"] == 1]
        .groupby(["StoreType", "Assortment"])
        .agg(
            Observations=("Sales", "size"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean")
        )
        .reset_index()
    )

    sheets["Store Profile"] = store_profile

    # --------------------------------------------------
    # 10. Customers vs sales
    # --------------------------------------------------
    customer_corr = open_train[["Sales", "Customers"]].corr().iloc[0, 1]

    customer_summary = (
        open_train.groupby("Store")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Customers=("Customers", "sum"),
            Average_Sales=("Sales", "mean"),
            Average_Customers=("Customers", "mean")
        )
        .reset_index()
    )

    customer_summary["Sales_per_Customer"] = (
        customer_summary["Total_Sales"]
        / customer_summary["Total_Customers"].replace(0, pd.NA)
    )

    sheets["Customer Sales"] = customer_summary

    add_finding(
        "Customers and Sales relationship",
        (
            f"Correlation between daily Customers and Sales on open days: "
            f"{customer_corr:.3f}. Customers is present in train but absent "
            "from test."
        ),
        (
            "Do not use actual future Customers as a production feature. "
            "It is unavailable at inference time and may encode information "
            "observed during the sales event. Consider it for EDA and "
            "diagnostic analysis, not as a default model input."
        )
    )

    # --------------------------------------------------
    # 11. Forecast horizon and feature availability
    # --------------------------------------------------
    horizon_days = len(test_dates)

    add_finding(
        "Forecast horizon",
        (
            f"The assignment describes six weeks (42 days), while the "
            f"Kaggle test dataset contains {horizon_days} distinct dates."
        ),
        (
            "Align primary validation and evaluation to the 42-day "
            "assignment horizon. Document the additional test dates and "
            "decide how to handle them in final inference."
        )
    )

    add_finding(
        "Feature availability",
        (
            f"Customers exists in train: {'Customers' in train.columns}; "
            f"Customers exists in test: {'Customers' in test.columns}. "
            f"Train target exists: {'Sales' in train.columns}; "
            f"test target exists: {'Sales' in test.columns}."
        ),
        (
            "Separate features available at forecast origin from variables "
            "observed only after the target date. All validation must mimic "
            "the intended forecast origin."
        )
    )

    return findings, sheets


def write_outputs(findings, sheets):
    REPORT_DIR.mkdir(exist_ok=True)

    findings_df = pd.DataFrame(findings)

    # Concise Markdown report
    report_path = REPORT_DIR / "data_findings.md"

    lines = [
        "# Rossmann — Automated Data Investigation",
        "",
        "This report is generated from the train, test and store datasets.",
        "Observations are separated from modelling implications.",
        ""
    ]

    for i, row in findings_df.iterrows():
        lines.extend([
            f"## {i + 1}. {row['Finding']}",
            "",
            f"**Observation:** {row['Observation']}",
            "",
            f"**Implication:** {row['Implication']}",
            ""
        ])

    lines.extend([
        "## Initial modelling principles",
        "",
        "- Use a global pooled model as the starting point, then evaluate "
        "performance by store and segment.",
        "- Do not use actual future Customers as an inference feature.",
        "- Do not blindly remove zero-sales or extreme-sales observations.",
        "- Treat missing metadata according to its meaning, not just "
        "because it is missing.",
        "- Use time-aware validation that simulates the complete "
        "forecast horizon.",
        "- Keep the 42-day assignment horizon distinct from the "
        "48-day Kaggle test period.",
        ""
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Supporting workbook
    workbook_path = REPORT_DIR / "data_investigation_details.xlsx"

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        findings_df.to_excel(
            writer, sheet_name="Findings Summary", index=False
        )

        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for column_cells in ws.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                ws.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(max_length + 3, 45)

    print("\n" + "=" * 65)
    print("ROSSMANN — AUTOMATED INVESTIGATION FINDINGS")
    print("=" * 65)

    for i, row in findings_df.iterrows():
        print(f"\n{i + 1}. {row['Finding']}")
        print(f"   Observation: {row['Observation']}")
        print(f"   Implication: {row['Implication']}")

    print("\n" + "=" * 65)
    print(f"Markdown report: {report_path.resolve()}")
    print(f"Supporting workbook: {workbook_path.resolve()}")
    print("=" * 65)


def main():
    train, test, store = load_data()
    findings, sheets = investigate(train, test, store)
    write_outputs(findings, sheets)


if __name__ == "__main__":
    main()
