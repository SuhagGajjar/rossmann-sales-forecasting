
from pathlib import Path
from datetime import date
import calendar
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpyxl.drawing.image import Image as ExcelImage


DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
CHART_DIR = REPORT_DIR / "customer_eda_charts"

CHART_DPI = 160
SCATTER_SAMPLE_SIZE = 20000


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False
    )

    store = pd.read_csv(DATA_DIR / "store.csv")

    return train, store


# ============================================================
# 2. PROMO2 DATE HANDLING
# ============================================================

def get_promo2_start_date(year, week):
    """Convert Promo2 ISO year/week into its Monday start date."""

    if pd.isna(year) or pd.isna(week):
        return pd.NaT

    try:
        return pd.Timestamp(
            date.fromisocalendar(int(year), int(week), 1)
        )
    except (ValueError, TypeError):
        return pd.NaT


def get_active_months(interval):
    month_lookup = {
        calendar.month_abbr[i]: i
        for i in range(1, 13)
    }

    if pd.isna(interval):
        return set()

    return {
        month_lookup[m.strip()[:3].title()]
        for m in str(interval).split(",")
        if m.strip()[:3].title() in month_lookup
    }


# ============================================================
# 3. BUILD ENRICHED CUSTOMER EDA SOURCE
# ============================================================

def build_customer_source(train, store):

    df = train.merge(
        store,
        on="Store",
        how="left",
        validate="many_to_one"
    )

    # Calendar
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Is_Weekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    # Daily promotion
    df["Promo"] = df["Promo"].fillna(0).astype(int)

    # Recurring promotion participation
    df["Promo2"] = df["Promo2"].fillna(0).astype(int)

    df["Promo2_Start_Date"] = [
        get_promo2_start_date(year, week)
        for year, week in zip(
            df["Promo2SinceYear"],
            df["Promo2SinceWeek"]
        )
    ]

    active_months = {
        store_id: get_active_months(interval)
        for store_id, interval in zip(
            store["Store"],
            store["PromoInterval"]
        )
    }

    df["Promo2_Active_Month"] = [
        month in active_months.get(store_id, set())
        for store_id, month in zip(df["Store"], df["Month"])
    ]

    valid_promo2_start = (
        df["Promo2_Start_Date"].notna()
        & (df["Date"] >= df["Promo2_Start_Date"])
    )

    df["Promo2_Metadata_Missing"] = (
        (df["Promo2"] == 1)
        & (
            df["Promo2_Start_Date"].isna()
            | df["PromoInterval"].isna()
        )
    ).astype(int)

    df["Promo2_Active"] = (
        (df["Promo2"] == 1)
        & valid_promo2_start
        & df["Promo2_Active_Month"]
        & (df["Promo2_Metadata_Missing"] == 0)
    ).astype(int)

    # Four promotion combinations
    conditions = [
        (df["Promo"] == 0) & (df["Promo2_Active"] == 0),
        (df["Promo"] == 1) & (df["Promo2_Active"] == 0),
        (df["Promo"] == 0) & (df["Promo2_Active"] == 1),
        (df["Promo"] == 1) & (df["Promo2_Active"] == 1)
    ]

    labels = [
        "Neither active",
        "Promo only",
        "Promo2 only",
        "Both active"
    ]

    df["Promotion_Combination"] = np.select(
        conditions,
        labels,
        default="Other / Unknown"
    )

    # Competition context
    df["CompetitionAgeMonths"] = (
        (df["Date"].dt.year - df["CompetitionOpenSinceYear"]) * 12
        + df["Date"].dt.month
        - df["CompetitionOpenSinceMonth"]
    )

    df["Competition_Metadata_Missing"] = (
        df["CompetitionOpenSinceYear"].isna()
        | df["CompetitionOpenSinceMonth"].isna()
    ).astype(int)

    df["Competition_Distance_Band"] = pd.cut(
        df["CompetitionDistance"],
        bins=[0, 500, 1000, 2000, 5000, 10000, np.inf],
        labels=[
            "0-500 m",
            "501-1,000 m",
            "1,001-2,000 m",
            "2,001-5,000 m",
            "5,001-10,000 m",
            ">10,000 m"
        ],
        include_lowest=True
    )

    # Customer spending measure
    df["Sales_per_Customer"] = np.where(
        df["Customers"] > 0,
        df["Sales"] / df["Customers"],
        np.nan
    )

    return df


# ============================================================
# 4. COMMON AGGREGATION
# ============================================================

def summarise(data, group_columns):

    result = (
        data.groupby(
            group_columns,
            observed=True,
            dropna=False
        )
        .agg(
            Observations=("Sales", "size"),
            Unique_Stores=("Store", "nunique"),
            Total_Customers=("Customers", "sum"),
            Average_Customers=("Customers", "mean"),
            Median_Customers=("Customers", "median"),
            Total_Sales=("Sales", "sum"),
            Average_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Average_Daily_Sales_per_Customer=(
                "Sales_per_Customer", "mean"
            )
        )
        .reset_index()
    )

    result["Weighted_Sales_per_Customer"] = (
        result["Total_Sales"]
        / result["Total_Customers"].replace(0, np.nan)
    )

    return result


# ============================================================
# 5. BUILD ANALYSIS TABLES
# ============================================================

def build_eda_tables(df):

    open_df = df[
        (df["Open"] == 1)
        & df["Customers"].notna()
    ].copy()

    tables = {}

    tables["By Store Type"] = summarise(
        open_df, ["StoreType"]
    )

    tables["By Assortment"] = summarise(
        open_df, ["Assortment"]
    )

    tables["By Competition"] = summarise(
        open_df, ["Competition_Distance_Band"]
    )

    tables["Daily Promo"] = summarise(
        open_df, ["Promo"]
    )

    tables["Promo2 Participation"] = summarise(
        open_df, ["Promo2"]
    )

    tables["Promo2 Activity"] = summarise(
        open_df, ["Promo2", "Promo2_Active"]
    )

    tables["Promotion Combinations"] = summarise(
        open_df, ["Promo", "Promo2_Active"]
    )

    tables["Promotion by Store Type"] = summarise(
        open_df, ["StoreType", "Promo"]
    )

    tables["Promotion by Assortment"] = summarise(
        open_df, ["Assortment", "Promo"]
    )

    tables["Promotion by Weekday"] = summarise(
        open_df, ["DayOfWeek", "Promo"]
    )

    tables["Promo2 by Store Type"] = summarise(
        open_df, ["StoreType", "Promo2_Active"]
    )

    tables["Combined Promotion Detail"] = summarise(
        open_df,
        ["StoreType", "Promo", "Promo2_Active"]
    )

    store_summary = (
        open_df.groupby(
            ["Store", "StoreType", "Assortment", "Promo2"],
            observed=True
        )
        .agg(
            Open_Days=("Sales", "size"),
            Average_Customers=("Customers", "mean"),
            Average_Sales=("Sales", "mean"),
            Total_Customers=("Customers", "sum"),
            Total_Sales=("Sales", "sum")
        )
        .reset_index()
    )

    store_summary["Sales_per_Customer"] = (
        store_summary["Total_Sales"]
        / store_summary["Total_Customers"].replace(0, np.nan)
    )

    tables["Store Level Summary"] = store_summary

    correlation = pd.DataFrame([
        {
            "Analysis": "Daily open store-day correlation",
            "Correlation": open_df["Customers"].corr(
                open_df["Sales"], method="pearson"
            ),
            "Observations": len(open_df)
        },
        {
            "Analysis": "Store-level average customer-sales correlation",
            "Correlation": store_summary["Average_Customers"].corr(
                store_summary["Average_Sales"], method="pearson"
            ),
            "Observations": len(store_summary)
        }
    ])

    tables["Customer Sales Correlation"] = correlation

    return tables, open_df


# ============================================================
# 6. GENERATE FINDINGS
# ============================================================

def generate_findings(tables):

    findings = []

    def add(title, observation, caveat):
        findings.append({
            "Finding": title,
            "Observation": observation,
            "Interpretation / Caveat": caveat
        })

    store_type = tables["By Store Type"].sort_values(
        "Average_Customers", ascending=False
    )

    top = store_type.iloc[0]

    add(
        "Customer volume by store type",
        (
            f"StoreType {top['StoreType']} has the highest average "
            f"daily customers ({top['Average_Customers']:,.1f}), "
            f"across {int(top['Unique_Stores'])} stores."
        ),
        "Store groups differ in size and composition; the result is descriptive."
    )

    promo = tables["Daily Promo"].set_index("Promo")

    if 0 in promo.index and 1 in promo.index:

        customer_change = (
            promo.loc[1, "Average_Customers"]
            / promo.loc[0, "Average_Customers"] - 1
        ) * 100

        sales_change = (
            promo.loc[1, "Average_Sales"]
            / promo.loc[0, "Average_Sales"] - 1
        ) * 100

        spend_change = (
            promo.loc[1, "Weighted_Sales_per_Customer"]
            / promo.loc[0, "Weighted_Sales_per_Customer"] - 1
        ) * 100

        add(
            "Daily promotion association",
            (
                f"Promo=1 versus Promo=0: average customers "
                f"{customer_change:+.1f}%; average sales "
                f"{sales_change:+.1f}%; weighted sales per customer "
                f"{spend_change:+.1f}%."
            ),
            "Unadjusted comparison; this is not a causal estimate of promotional uplift."
        )

    promo2 = tables["Promo2 Activity"].set_index(
        ["Promo2", "Promo2_Active"]
    )

    if (1, 0) in promo2.index and (1, 1) in promo2.index:

        inactive = promo2.loc[(1, 0)]
        active = promo2.loc[(1, 1)]

        customer_change = (
            active["Average_Customers"]
            / inactive["Average_Customers"] - 1
        ) * 100

        sales_change = (
            active["Average_Sales"]
            / inactive["Average_Sales"] - 1
        ) * 100

        add(
            "Recurring promotion activity",
            (
                f"Among participating stores, active versus inactive "
                f"Promo2 periods show average customer difference "
                f"{customer_change:+.1f}% and average sales difference "
                f"{sales_change:+.1f}%."
            ),
            "Seasonality, store mix and promotion timing may influence this comparison."
        )

    corr = tables["Customer Sales Correlation"]

    add(
        "Customer-sales relationship",
        (
            f"Daily Pearson correlation: {corr.iloc[0]['Correlation']:.3f}; "
            f"store-level Pearson correlation: {corr.iloc[1]['Correlation']:.3f}."
        ),
        (
            "Correlation is not causation. Actual Customers is unavailable "
            "in test and will not be used as a forecasting input."
        )
    )

    return pd.DataFrame(findings)


# ============================================================
# 7. CHART UTILITIES
# ============================================================

def save_chart(fig, filename):
    path = CHART_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def annotate_bars(ax, bars, format_string="{:,.0f}"):

    for bar in bars:
        height = bar.get_height()

        if pd.notna(height):
            ax.annotate(
                format_string.format(height),
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )


# ============================================================
# 8. GENERATE FIVE CUSTOMER EDA CHARTS
# ============================================================

def generate_charts(tables, open_df):

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    chart_paths = {}

    # --------------------------------------------------------
    # Chart 1: Average customers by store type
    # --------------------------------------------------------

    data = tables["By Store Type"].sort_values("StoreType")

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        data["StoreType"].astype(str),
        data["Average_Customers"]
    )

    annotate_bars(ax, bars, "{:,.0f}")

    for bar, stores in zip(bars, data["Unique_Stores"]):
        ax.annotate(
            f"{int(stores)} stores",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

    ax.set_title("Average Daily Customers by Store Type")
    ax.set_xlabel("Store Type")
    ax.set_ylabel("Average Customers per Open Store-Day")
    ax.grid(axis="y", alpha=0.25)

    chart_paths["store_type"] = save_chart(
        fig, "01_average_customers_by_store_type.png"
    )

    # --------------------------------------------------------
    # Chart 2: Daily Promo comparison
    # --------------------------------------------------------

    data = tables["Daily Promo"].set_index("Promo").sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("Average_Customers", "Average Customers", "{:,.0f}"),
        ("Average_Sales", "Average Sales", "{:,.0f}"),
        (
            "Weighted_Sales_per_Customer",
            "Weighted Sales per Customer",
            "{:.2f}"
        )
    ]

    for ax, (column, title, fmt) in zip(axes, metrics):

        values = [
            data.loc[0, column],
            data.loc[1, column]
        ]

        bars = ax.bar(
            ["No Daily Promo", "Daily Promo"],
            values
        )

        annotate_bars(ax, bars, fmt)

        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Daily Promotion: Customer and Sales Comparison")

    chart_paths["daily_promo"] = save_chart(
        fig, "02_daily_promo_comparison.png"
    )

    # --------------------------------------------------------
    # Chart 3: Promo2 activity among participating stores only
    # --------------------------------------------------------

    data = tables["Promo2 Activity"]

    data = data[data["Promo2"] == 1].set_index("Promo2_Active")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (column, title, fmt) in zip(axes, metrics):

        values = [
            data.loc[0, column],
            data.loc[1, column]
        ]

        bars = ax.bar(
            ["Inactive period", "Active period"],
            values
        )

        annotate_bars(ax, bars, fmt)

        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Promo2 Activity: Participating Stores Only"
    )

    chart_paths["promo2_activity"] = save_chart(
        fig, "03_promo2_active_vs_inactive.png"
    )

    # --------------------------------------------------------
    # Chart 4: Four promotion combinations
    # --------------------------------------------------------

    data = tables["Promotion Combinations"].copy()

    combination_order = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1)
    ]

    combination_labels = [
        "Neither active",
        "Promo only",
        "Promo2 only",
        "Both active"
    ]

    indexed = data.set_index(["Promo", "Promo2_Active"])

    values = [
        indexed.loc[key]
        for key in combination_order
    ]

    avg_customers = [
        row["Average_Customers"] for row in values
    ]

    avg_sales = [
        row["Average_Sales"] for row in values
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(combination_labels, avg_customers)
    annotate_bars(axes[0], bars, "{:,.0f}")
    axes[0].set_title("Average Customers")
    axes[0].set_ylabel("Customers per Open Store-Day")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    bars = axes[1].bar(combination_labels, avg_sales)
    annotate_bars(axes[1], bars, "{:,.0f}")
    axes[1].set_title("Average Sales")
    axes[1].set_ylabel("Sales per Open Store-Day")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Daily Promo and Recurring Promo2: Four Combinations"
    )

    chart_paths["promotion_combinations"] = save_chart(
        fig, "04_four_promotion_combinations.png"
    )

    # --------------------------------------------------------
    # Chart 5: Customers vs Sales
    # --------------------------------------------------------

    plot_data = open_df[
        ["Customers", "Sales"]
    ].dropna()

    sample = plot_data.sample(
        n=min(SCATTER_SAMPLE_SIZE, len(plot_data)),
        random_state=42
    )

    correlation = plot_data["Customers"].corr(
        plot_data["Sales"], method="pearson"
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        sample["Customers"],
        sample["Sales"],
        alpha=0.22,
        s=9
    )

    # Regression line calculated from the plotted sample.
    slope, intercept = np.polyfit(
        sample["Customers"],
        sample["Sales"],
        1
    )

    x_values = np.linspace(
        sample["Customers"].min(),
        sample["Customers"].max(),
        100
    )

    ax.plot(
        x_values,
        slope * x_values + intercept,
        linewidth=2,
        label="Sample trendline"
    )

    ax.set_title(
        f"Customers vs Sales | Full-data Pearson correlation = {correlation:.3f}"
    )
    ax.set_xlabel("Customers per Open Store-Day")
    ax.set_ylabel("Daily Sales")
    ax.legend()
    ax.grid(alpha=0.2)

    ax.text(
        0.98,
        0.03,
        f"Displayed sample: {len(sample):,} of {len(plot_data):,} observations",
        transform=ax.transAxes,
        ha="right",
        fontsize=8
    )

    chart_paths["customer_sales"] = save_chart(
        fig, "05_customers_vs_sales.png"
    )

    return chart_paths


# ============================================================
# 9. DOCUMENTATION: PLAIN-ENGLISH METHODOLOGY
# ============================================================

def build_methodology_report(df, open_df, tables, findings, chart_paths):

    correlation = tables["Customer Sales Correlation"]
    chart_rel = lambda key: (
        Path("customer_eda_charts") / chart_paths[key].name
    ).as_posix()

    store_count = df["Store"].nunique()
    source_rows = len(df)
    open_rows = len(open_df)

    promo_counts = tables["Daily Promo"].set_index("Promo")
    promo2_counts = tables["Promo2 Activity"]

    active_promo2 = promo2_counts[
        (promo2_counts["Promo2"] == 1)
        & (promo2_counts["Promo2_Active"] == 1)
    ]

    inactive_promo2 = promo2_counts[
        (promo2_counts["Promo2"] == 1)
        & (promo2_counts["Promo2_Active"] == 0)
    ]

    def md_table(table):
        return table.to_markdown(index=False, floatfmt=".2f")

    lines = [
        "# Rossmann Customer and Promotion EDA",
        "",
        "## 1. Objective and scope",
        "",
        "This analysis investigates how customer footfall and sales vary "
        "across store characteristics and promotion conditions. "
        "It supplements the overall sales EDA with customer-level insight.",
        "",
        "The analysis is descriptive. It identifies patterns and associations; "
        "it does not establish that a promotion caused a change in sales or customers.",
        "",
        "## 2. Data sources and analysis population",
        "",
        "- Training data: `data/raw/train.csv`.",
        "- Store metadata: `data/raw/store.csv`.",
        "- Join: both datasets are joined using the `Store` column.",
        "- Join validation: many training store-day records to one store metadata record.",
        "- Analysis grain: one record per store per date.",
        f"- Enriched source records: {source_rows:,}.",
        f"- Distinct stores: {store_count:,}.",
        f"- Eligible open store-day observations: {open_rows:,}.",
        "- Customer and sales comparisons include records where Open = 1 "
        "and Customers is available.",
        "- Closed store-days are excluded from customer behaviour comparisons.",
        "- No customer values are imputed.",
        "",
        "The source CSV retains all original training observations, including "
        "closed store-days. The open-store filter is applied only when calculating "
        "customer and sales comparisons.",
        "",
        "## 3. Columns and derived fields",
        "",
        "| Field | Source / derivation | Purpose |",
        "|---|---|---|",
        "| Store | train.csv and store.csv | Store identifier and join key |",
        "| Date | train.csv | Calendar and promotion timing |",
        "| Open | train.csv | Identifies open store-days |",
        "| Customers | train.csv | Daily customer count |",
        "| Sales | train.csv | Daily sales outcome |",
        "| StoreType | store.csv | Store format comparison |",
        "| Assortment | store.csv | Assortment comparison |",
        "| Promo | train.csv | Daily promotion flag |",
        "| Promo2 | store.csv | Recurring programme participation |",
        "| Promo2SinceYear / Week | store.csv | Recurring programme start date |",
        "| PromoInterval | store.csv | Months associated with recurring programme |",
        "| Promo2_Start_Date | Derived | Monday of the specified ISO week |",
        "| Promo2_Active | Derived | Participation, start-date and active-month conditions |",
        "| Promotion_Combination | Derived | Daily Promo and Promo2_Active combination |",
        "| CompetitionDistance | store.csv | Distance to nearest competitor |",
        "| Competition_Distance_Band | Derived | Groups competition distance into bands |",
        "| CompetitionAgeMonths | Derived | Months since competitor opening, where metadata exists |",
        "| Sales_per_Customer | Derived | Store-day sales divided by customers when customers > 0 |",
        "",
        "## 4. Calculation methodology",
        "",
        "### 4.1 Common aggregation rules",
        "",
        "- **Observations:** number of eligible store-day records in each group.",
        "- **Unique stores:** number of distinct Store identifiers represented in each group.",
        "- **Total customers:** sum of Customers across eligible records.",
        "- **Average customers:** total customer count divided by eligible store-day observations.",
        "- **Median customers:** middle customer count across eligible store-day observations.",
        "- **Total sales:** sum of Sales across eligible records.",
        "- **Average sales:** total sales divided by eligible store-day observations.",
        "- **Median sales:** middle sales value across eligible store-day observations.",
        "",
        "These averages are store-day averages. They do not give every store equal "
        "weight: stores with more eligible observations contribute more records.",
        "",
        "### 4.2 Sales per customer",
        "",
        "**Average daily sales per customer** uses Sales and Customers for each "
        "eligible store-day. It calculates the sales-to-customer ratio for each "
        "store-day where Customers is greater than zero, then averages those ratios. "
        "Store-days with zero customers are excluded from this particular measure "
        "because division is not meaningful.",
        "",
        "**Weighted sales per customer** divides the total Sales of the group by "
        "the total Customers of the same group. It represents the overall sales "
        "per customer across all included store-days and gives more weight to "
        "store-days with more customers.",
        "",
        "These two measures are different and should not be used interchangeably. "
        "Weighted sales per customer is the primary group-level spending measure "
        "in the promotion comparisons.",
        "",
        "### 4.3 Customer volume by store type",
        "",
        "- Columns: StoreType, Open, Customers, Store.",
        "- Filter: open store-days with a recorded customer count.",
        "- Grouping: StoreType.",
        "- Calculation: average Customers across eligible store-days in each type.",
        "- Supporting context: distinct store count and observation count are reported.",
        "",
        "### 4.4 Customer volume by assortment",
        "",
        "- Columns: Assortment, Open, Customers, Store.",
        "- Filter: same eligible open store-day population.",
        "- Grouping: Assortment.",
        "- Calculation: average and median Customers, with totals and store counts.",
        "",
        "### 4.5 Daily promotion comparison",
        "",
        "- Columns: Promo, Open, Customers, Sales.",
        "- Filter: eligible open store-days.",
        "- Grouping: Promo = 0 and Promo = 1.",
        "- Calculation: average customers and average sales per eligible store-day, "
        "plus weighted sales per customer.",
        "- Percentage difference: the Promo = 1 group value is compared with "
        "the Promo = 0 group value; the difference is divided by the Promo = 0 "
        "value and expressed as a percentage.",
        "",
        "This is an unadjusted comparison. Promotion days may differ from "
        "non-promotion days in calendar timing, store mix and other conditions.",
        "",
        "### 4.6 Promo2 participation comparison",
        "",
        "- Columns: Promo2, Open, Customers, Sales.",
        "- Filter: eligible open store-days.",
        "- Grouping: Promo2 = 0 versus Promo2 = 1.",
        "- Calculation: average customers, average sales and weighted sales per customer.",
        "",
        "This compares stores participating in the recurring programme with "
        "stores not participating. It is not an active-versus-inactive comparison "
        "within the same participating stores.",
        "",
        "### 4.7 Promo2 active versus inactive comparison",
        "",
        "- Columns: Promo2, Promo2_Active, Open, Customers, Sales.",
        "- Filter for the active/inactive comparison: Promo2 = 1 only.",
        "- Grouping: Promo2_Active = 0 versus Promo2_Active = 1.",
        "- Calculation: average customers and sales per eligible store-day, "
        "plus weighted sales per customer.",
        "- Percentage difference: active-period average compared with "
        "inactive-period average, using the inactive-period average as the reference.",
        "",
        "Non-participating stores are not treated as participating stores with "
        "an inactive promotion. The participation flag is retained in the analysis.",
        "",
        "### 4.8 Four promotion combinations",
        "",
        "- Columns: Promo and Promo2_Active, with Open, Customers and Sales.",
        "- Filter: eligible open store-days.",
        "- Grouping: both promotion flags together.",
        "- Categories: neither active, daily Promo only, Promo2 only, and both active.",
        "- Calculation: average customers and average sales for each combination.",
        "",
        "The 'Promo2 not active' groups can contain both non-participating stores "
        "and participating stores in inactive periods. They must not be interpreted "
        "as only participating stores with inactive promotions.",
        "",
        "### 4.9 Promotion by store type, assortment and weekday",
        "",
        "- Columns: StoreType or Assortment or DayOfWeek, Promo, Open, Customers and Sales.",
        "- Filter: eligible open store-days.",
        "- Grouping: each selected store characteristic together with daily Promo.",
        "- Calculation: the same common aggregation measures are calculated "
        "within each group.",
        "",
        "These breakdowns help identify whether the overall daily promotion "
        "pattern is consistent across different store categories and weekdays.",
        "",
        "### 4.10 Promo2 by store type and combined promotion detail",
        "",
        "- Columns: StoreType, Promo, Promo2_Active, Open, Customers and Sales.",
        "- Filter: eligible open store-days.",
        "- Grouping: store type and the relevant promotion flags.",
        "- Calculation: average customers, average sales, totals, medians "
        "and weighted sales per customer.",
        "",
        "These tables provide more detailed context for differences in store "
        "composition and promotion conditions.",
        "",
        "### 4.11 Competition distance",
        "",
        "- Columns: CompetitionDistance, Competition_Distance_Band, Open, Customers and Sales.",
        "- Filter: eligible open store-days.",
        "- Grouping: competition distance band.",
        "- Calculation: customer and sales aggregation within each distance band.",
        "",
        "Distance bands are descriptive groupings. Missing competition distances "
        "remain a separate missing group rather than being assigned an invented distance.",
        "",
        "### 4.12 Store-level customer-sales relationship",
        "",
        "- Columns: Store, StoreType, Assortment, Promo2, Open, Customers and Sales.",
        "- Filter: eligible open store-days.",
        "- Grouping: Store.",
        "- Calculation: average customers and average sales across each store's "
        "eligible open days, alongside totals and number of open days.",
        "",
        "This creates one summary record per store. It is used to examine whether "
        "stores with higher average customer volume also tend to have higher average sales.",
        "",
        "### 4.13 Customer-sales correlation",
        "",
        "- Daily correlation: Pearson correlation between Customers and Sales "
        "across all eligible open store-day observations.",
        "- Store-level correlation: Pearson correlation between each store's "
        "average Customers and average Sales.",
        "- Correlation ranges from -1 to +1. Positive values indicate that higher "
        "customer counts tend to occur alongside higher sales; values closer to +1 "
        "indicate a stronger positive linear relationship.",
        "",
        "The two correlations answer different questions. The first is based on "
        "individual store-days; the second compares store averages.",
        "",
        "## 5. Promo2 date interpretation and assumptions",
        "",
        "The Promo2 start year and week are converted to the Monday of the "
        "corresponding ISO calendar week. A participating store is marked active "
        "only when the observation date is on or after that start date and its "
        "calendar month appears in PromoInterval.",
        "",
        "The active-month list is interpreted as recurring in subsequent years "
        "after the start date. This is an analytical assumption that should be "
        "confirmed with the business owner or dataset documentation before "
        "operational use.",
        "",
        "A missing or invalid start date or missing PromoInterval for a participating "
        "store prevents that record from being classified as active. Such records "
        "are flagged through Promo2_Metadata_Missing.",
        "",
        "## 6. Visual findings",
        "",
        "### Chart 1: Average customers by store type",
        "",
        "Shows average customers per open store-day for each store type. "
        "Store counts are annotated to make the size of each group visible. "
        "A high average for a small store group should not automatically be "
        "generalised to the full store population.",
        "",
        f"![Average customers by store type]({chart_rel('store_type')})",
        "",
        "### Chart 2: Daily promotion comparison",
        "",
        "Compares average customers, average sales and weighted sales per customer "
        "between daily promotion and non-promotion store-days.",
        "",
        f"![Daily promotion comparison]({chart_rel('daily_promo')})",
        "",
        "### Chart 3: Promo2 active versus inactive",
        "",
        "Compares participating stores only, separating their active and inactive "
        "recurring-promotion periods.",
        "",
        f"![Promo2 active versus inactive]({chart_rel('promo2_activity')})",
        "",
        "### Chart 4: Four promotion combinations",
        "",
        "Compares average customers and average sales across the four combinations "
        "of daily Promo and Promo2 activity.",
        "",
        f"![Four promotion combinations]({chart_rel('promotion_combinations')})",
        "",
        "### Chart 5: Customers versus sales",
        "",
        "Displays a reproducible random sample of up to 20,000 open store-day "
        "observations for readability. The reported Pearson correlation is "
        "calculated using the full eligible open store-day dataset, not just "
        "the displayed sample. The trendline is fitted to the displayed sample.",
        "",
        f"![Customers versus sales]({chart_rel('customer_sales')})",
        "",
        "## 7. Findings",
        ""
    ]

    for _, row in findings.iterrows():
        lines.extend([
            f"### {row['Finding']}",
            "",
            f"**Observation:** {row['Observation']}",
            "",
            f"**Interpretation / limitation:** {row['Interpretation / Caveat']}",
            ""
        ])

    lines.extend([
        "## 8. Customer-sales relationship and forecasting implications",
        "",
        f"- Daily open store-day Pearson correlation: "
        f"{correlation.iloc[0]['Correlation']:.3f} "
        f"({int(correlation.iloc[0]['Observations']):,} observations).",
        f"- Store-level average Pearson correlation: "
        f"{correlation.iloc[1]['Correlation']:.3f} "
        f"({int(correlation.iloc[1]['Observations']):,} stores).",
        "",
        "The positive relationship indicates that customer volume is informative "
        "when explaining historical sales. However, Customers is absent from "
        "the competition test dataset and is not available at the time the "
        "forecast must be generated. It is therefore excluded from model inputs. "
        "It is used here for retrospective analysis and interpretation only.",
        "",
        "## 9. Limitations and interpretation safeguards",
        "",
        "- These are descriptive comparisons, not controlled experiments.",
        "- Promotion timing, seasonality, weekday effects and store characteristics "
        "may influence the observed results.",
        "- Store-day averages give more weight to stores with more eligible observations.",
        "- Store groups differ substantially in their number of stores.",
        "- Correlation does not demonstrate causation.",
        "- Promo2 activity depends on the stated ISO-week and recurring-month assumptions.",
        "- Sales per customer is undefined for store-days with zero customers.",
        "- Actual Customers is unavailable in the test data and is excluded from forecasting features.",
        "",
        "## 10. Reproducibility and output files",
        "",
        "- Source dataset: `data/processed/customer_eda_source.csv`.",
        "- Numerical summaries: `reports/customer_eda_summary.xlsx`.",
        "- Charts: `reports/customer_eda_charts/`.",
        "- Script: `src/preprocessing/customer_eda.py`.",
        "",
        "All reported tables and charts are generated from the same enriched "
        "training source and the same open-store-day analysis population.",
        ""
    ])

    return "\n".join(lines)


# ============================================================
# 10. EXPORT SOURCE, WORKBOOK, CHARTS AND METHODOLOGY
# ============================================================

def export_outputs(df, tables, findings, open_df, chart_paths):

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    source_path = PROCESSED_DIR / "customer_eda_source.csv"
    workbook_path = REPORT_DIR / "customer_eda_summary.xlsx"
    report_path = REPORT_DIR / "customer_eda_findings.md"

    # Enriched source
    df.to_csv(source_path, index=False)

    # Workbook with all numeric tables
    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl"
    ) as writer:

        findings.to_excel(
            writer,
            sheet_name="Findings",
            index=False
        )

        for name, table in tables.items():
            table.to_excel(
                writer,
                sheet_name=name[:31],
                index=False
            )

        # Embed generated charts in the workbook
        chart_sheet = writer.book.create_sheet("Charts")

        chart_sheet["A1"] = "Rossmann Customer EDA Charts"
        chart_sheet["A1"].font = chart_sheet["A1"].font.copy(
            bold=True,
            size=16
        )

        positions = [
            ("A3", "store_type"),
            ("A30", "daily_promo"),
            ("A57", "promo2_activity"),
            ("A84", "promotion_combinations"),
            ("A111", "customer_sales")
        ]

        for cell, key in positions:
            image = ExcelImage(str(chart_paths[key]))
            image.width = 720
            image.height = 390
            chart_sheet.add_image(image, cell)

        for ws in writer.book.worksheets:

            if ws.title == "Charts":
                continue

            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for column_cells in ws.columns:

                max_length = max(
                    len(str(cell.value))
                    if cell.value is not None else 0
                    for cell in column_cells
                )

                ws.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(max_length + 3, 45)

    # Detailed Markdown methodology and findings
    report_text = build_methodology_report(
        df,
        open_df,
        tables,
        findings,
        chart_paths
    )

    report_path.write_text(
        report_text,
        encoding="utf-8"
    )

    print("\nCUSTOMER EDA COMPLETED")
    print(f"Source dataset: {source_path.resolve()}")
    print(f"Excel summary: {workbook_path.resolve()}")
    print(f"Findings report: {report_path.resolve()}")
    print(f"Charts folder: {CHART_DIR.resolve()}")
    print(f"Source rows: {len(df):,}")
    print(f"Source columns: {len(df.columns):,}")
    print(f"Eligible open store-days: {len(open_df):,}")
    print(f"Charts generated: {len(chart_paths)}")


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("Loading training and store data...")
    train, store = load_data()

    print("Building enriched customer EDA source...")
    source = build_customer_source(train, store)

    print("Generating customer and promotion analysis...")
    tables, open_df = build_eda_tables(source)

    print("Generating findings...")
    findings = generate_findings(tables)

    print("Generating five customer EDA charts...")
    chart_paths = generate_charts(tables, open_df)

    print("Exporting source, Excel workbook and methodology report...")
    export_outputs(
        source,
        tables,
        findings,
        open_df,
        chart_paths
    )

    print("\nGENERATED FINDINGS")
    print(findings.to_string(index=False))

    print("\nPROMOTION COMBINATION COUNTS")
    print(
        tables["Promotion Combinations"][
            [
                "Promo",
                "Promo2_Active",
                "Observations",
                "Unique_Stores",
                "Average_Customers",
                "Average_Sales"
            ]
        ].to_string(index=False)
    )

    print("\nCUSTOMER-SALES CORRELATION")
    print(tables["Customer Sales Correlation"].to_string(index=False))


if __name__ == "__main__":
    main()
