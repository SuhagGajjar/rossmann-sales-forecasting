
from copy import copy
from datetime import date
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 1. PROJECT PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
STORE_PATH = ROOT / "data" / "raw" / "store.csv"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

OUTPUT_EXCEL = REPORTS_DIR / "eda_summary.xlsx"
OUTPUT_REPORT = REPORTS_DIR / "eda_findings.md"

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "font.size": 10,
})


# =========================================================
# 2. PROMO2 DATE-LEVEL FEATURE ENGINEERING
# =========================================================

PROMO_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def get_promo2_start_date(row):
    """
    Convert Promo2SinceYear and Promo2SinceWeek into
    the Monday of the corresponding ISO calendar week.

    Returns NaT if the metadata is missing or invalid.
    """
    year = row["Promo2SinceYear"]
    week = row["Promo2SinceWeek"]

    if pd.isna(year) or pd.isna(week):
        return pd.NaT

    try:
        return pd.Timestamp(
            date.fromisocalendar(int(year), int(week), 1)
        )
    except (ValueError, TypeError, OverflowError):
        return pd.NaT

def parse_promo_months(value):
    if pd.isna(value):
        return set()

    return {
        PROMO_MONTHS[part.strip().lower()]
        for part in str(value).split(",")
        if part.strip().lower() in PROMO_MONTHS
    }

def get_promo2_active_flag(row):
    """
    Promo2 is active when:
    1. Store participates in Promo2.
    2. Observation date is on/after the programme start date.
    3. Observation month is in the recurring PromoInterval.

    The recurring months are applied in every subsequent year.
    """
    if row["Promo2"] != 1:
        return 0

    if pd.isna(row["Promo2_Start_Date"]):
        return 0

    if pd.isna(row["Date"]):
        return 0

    if row["Date"] < row["Promo2_Start_Date"]:
        return 0

    if pd.isna(row["PromoInterval"]):
        return 0

    months = {
        PROMO_MONTHS.get(part.strip().lower())
        for part in str(row["PromoInterval"]).split(",")
    }

    months.discard(None)

    return int(row["Date"].month in months)


# =========================================================
# 3. LOAD AND PREPARE DATA
# =========================================================

def load_data():
    print("Loading Rossmann datasets...")

    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False
    )

    store = pd.read_csv(STORE_PATH, low_memory=False)

    if store["Store"].duplicated().any():
        raise ValueError("Duplicate Store IDs found in store.csv.")

    calculated_day = train["Date"].dt.day_of_week + 1

    mismatch_count = int(
        (calculated_day != train["DayOfWeek"]).sum()
    )

    if mismatch_count:
        raise ValueError(
            f"DayOfWeek validation failed: {mismatch_count} mismatches."
        )

    # Calculate programme start date only once per store.
    store["Promo2_Start_Date"] = store.apply(
        get_promo2_start_date,
        axis=1
    )

    # Parse recurring months only once per store.
    store["Promo2_Months"] = store["PromoInterval"].apply(
        parse_promo_months
    )

    df = train.merge(
        store,
        on="Store",
        how="left",
        validate="many_to_one"
    )

    if df["StoreType"].isna().any():
        raise ValueError("Some training stores have no store metadata.")

    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    # Vectorised recurring Promo2 feature.
    month_values = df["Date"].dt.month.to_numpy()

    interval_month_match = np.fromiter(
        (
            month in months
            for month, months in zip(
                month_values,
                df["Promo2_Months"]
            )
        ),
        dtype=bool,
        count=len(df)
    )

    df["Promo2_Active"] = (
        (df["Promo2"] == 1)
        & df["Promo2_Start_Date"].notna()
        & (df["Date"] >= df["Promo2_Start_Date"])
        & interval_month_match
    ).astype("int8")

    # Competition features.
    comp_month = pd.to_numeric(
        df["CompetitionOpenSinceMonth"],
        errors="coerce"
    )

    comp_year = pd.to_numeric(
        df["CompetitionOpenSinceYear"],
        errors="coerce"
    )

    df["Competition_Info_Missing"] = (
        comp_month.isna() | comp_year.isna()
    ).astype("int8")

    df["CompetitionDistance_Missing"] = (
        df["CompetitionDistance"].isna()
    ).astype("int8")

    df["CompetitionAgeMonths"] = (
        (df["Date"].dt.year - comp_year) * 12
        + (df["Date"].dt.month - comp_month)
    )

    print(f"Loaded {len(df):,} training records.")
    print(f"Stores: {df['Store'].nunique():,}")
    print(
        f"Period: {df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
    )
    print("Calendar weekday validation: passed.")
    print("Promo2 recurring date-level feature: created.")
    print("Competition features: created.")

    return df

# =========================================================
# 4. CHART HELPERS
# =========================================================

def save_chart(filename):
    path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"Created chart: {filename}")


def annotate_bars(ax, fmt="{:,.0f}", rotation=0):
    """Display values above every bar."""
    for container in ax.containers:
        labels = []

        for bar in container:
            height = bar.get_height()

            if pd.isna(height):
                labels.append("")
            else:
                labels.append(fmt.format(height))

        ax.bar_label(
            container,
            labels=labels,
            padding=4,
            fontsize=8,
            rotation=rotation
        )

    # Leave room above the bars for labels.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.12 if ymax > 0 else 1)



def plot_annotated_bars(
    data,
    x,
    y,
    title,
    xlabel,
    ylabel,
    filename,
    fmt="{:,.0f}",
    rotation=0,
    count_column=None
):
    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        data[x].astype(str),
        data[y],
        color=sns.color_palette("deep", n_colors=len(data))
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Annotate average values above each bar.
    ax.bar_label(
        bars,
        labels=[
            fmt.format(v) if pd.notna(v) else ""
            for v in data[y]
        ],
        padding=5,
        fontsize=9
    )

    # Annotate the number of stores below each bar.
    if count_column is not None:
        for bar, count in zip(bars, data[count_column]):
            ax.annotate(
                f"Stores: {int(count)}",
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, -30),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8
            )

        ax.margins(y=0.18)
        fig.subplots_adjust(bottom=0.20, top=0.88)
    else:
        fig.tight_layout()

    ax.tick_params(axis="x", rotation=rotation)

    save_chart(filename)


def md_table(data, columns=None):
    """Convert a dataframe into a Markdown table."""
    if columns is not None:
        data = data[columns]

    headers = list(data.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    for row in data.itertuples(index=False, name=None):
        values = []

        for value in row:
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:,.2f}")
            else:
                values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# =========================================================
# 5. BUILD ANALYTICAL TABLES
# =========================================================

def build_tables(df):
    open_df = df[df["Open"] == 1].copy()

    # -----------------------------------------------------
    # 1. Daily sales trend
    # -----------------------------------------------------

    daily = (
        open_df.groupby("Date")
        .agg(
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Open_Stores=("Store", "nunique")
        )
        .reset_index()
        .sort_values("Date")
    )

    daily["Rolling_28D_Mean"] = (
        daily["Mean_Sales"].rolling(28, min_periods=1).mean()
    )

    # -----------------------------------------------------
    # 2. Weekday analysis
    # -----------------------------------------------------

    weekday = (
        open_df.groupby("DayOfWeek")
        .agg(
            Observations=("Sales", "size"),
            Stores=("Store", "nunique"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean")
        )
        .reset_index()
        .sort_values("DayOfWeek")
    )

    weekday["Day"] = weekday["DayOfWeek"].map({
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday"
    })

    # -----------------------------------------------------
    # 3. Monthly analysis
    # -----------------------------------------------------

    monthly = (
        open_df.groupby("Month")
        .agg(
            Observations=("Sales", "size"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median")
        )
        .reset_index()
        .sort_values("Month")
    )

    monthly["Month_Name"] = monthly["Month"].apply(
        lambda x: pd.Timestamp(2000, int(x), 1).strftime("%B")
    )

    # -----------------------------------------------------
    # 4. Daily Promo analysis
    # -----------------------------------------------------

    promo = (
        open_df.groupby("Promo")
        .agg(
            Observations=("Sales", "size"),
            Stores=("Store", "nunique"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean")
        )
        .reset_index()
        .sort_values("Promo")
    )

    promo["Promotion"] = promo["Promo"].map({
        0: "No",
        1: "Yes"
    })

    # -----------------------------------------------------
    # 5. Four promotion combinations
    # -----------------------------------------------------

    promotion_groups = open_df.copy()

    promotion_groups["Promotion_Group"] = np.select(
        [
            (promotion_groups["Promo"] == 0)
            & (promotion_groups["Promo2_Active"] == 0),

            (promotion_groups["Promo"] == 0)
            & (promotion_groups["Promo2_Active"] == 1),

            (promotion_groups["Promo"] == 1)
            & (promotion_groups["Promo2_Active"] == 0),

            (promotion_groups["Promo"] == 1)
            & (promotion_groups["Promo2_Active"] == 1)
        ],
        [
            "Neither",
            "Promo2 only",
            "Daily Promo only",
            "Both active"
        ],
        default="Unknown"
    )

    group_order = [
        "Neither",
        "Promo2 only",
        "Daily Promo only",
        "Both active"
    ]

    promotion_combinations = (
        promotion_groups.groupby("Promotion_Group")
        .agg(
            Observations=("Sales", "size"),
            Stores=("Store", "nunique"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean")
        )
        .reindex(group_order)
        .reset_index()
    )

    # Add difference from neither-promotion baseline.
    neither_mean = promotion_combinations.loc[
        promotion_combinations["Promotion_Group"] == "Neither",
        "Mean_Sales"
    ].iloc[0]

    promotion_combinations["Mean_Sales_Difference_Percent"] = (
        (
            promotion_combinations["Mean_Sales"]
            / neither_mean
        ) - 1
    ) * 100

    # -----------------------------------------------------
    # 6. Store type and assortment
    # -----------------------------------------------------

    store_type = (
        open_df.groupby("StoreType")
        .agg(
            Observations=("Sales", "size"),
            Stores=("Store", "nunique"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean")
        )
        .reset_index()
    )

    assortment = (
        open_df.groupby("Assortment")
        .agg(
            Observations=("Sales", "size"),
            Stores=("Store", "nunique"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median")
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # 7. Holiday analysis
    # -----------------------------------------------------

    state_holiday = (
        open_df.groupby("StateHoliday", dropna=False)
        .agg(
            Observations=("Sales", "size"),
            Stores=("Store", "nunique"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median")
        )
        .reset_index()
    )

    school_holiday = (
        open_df.groupby("SchoolHoliday")
        .agg(
            Observations=("Sales", "size"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median")
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # 8. Store-level summary
    # -----------------------------------------------------

    store_summary = (
        open_df.groupby("Store")
        .agg(
            Observations=("Sales", "size"),
            Mean_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Mean_Customers=("Customers", "mean"),
            Sales_Std=("Sales", "std")
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # 9. Sales distribution and customer/zero-sales samples
    # -----------------------------------------------------

    sales_quantiles = (
        open_df["Sales"]
        .quantile([
            0, .01, .05, .25, .50, .75,
            .90, .95, .99, .999, 1
        ])
        .rename_axis("Quantile")
        .reset_index(name="Sales")
    )

    customer_sample = open_df.sample(
        n=min(5000, len(open_df)),
        random_state=42
    )[
        ["Store", "Date", "Sales", "Customers", "Promo"]
    ].copy()

    zero_sales = df[
        (df["Open"] == 1) & (df["Sales"] == 0)
    ][
        ["Store", "Date", "Sales", "Customers", "Open", "Promo"]
    ].copy()

    # -----------------------------------------------------
    # 10. Sunday coverage
    # -----------------------------------------------------

    sunday_open = open_df[open_df["DayOfWeek"] == 7]

    sunday_store_detail = (
        sunday_open.groupby("Store")
        .agg(Sunday_Open_Days=("Date", "nunique"))
        .reset_index()
    )

    sunday_coverage = pd.DataFrame({
        "Metric": [
            "Stores with at least one open Sunday",
            "Mean open Sundays per participating store",
            "Median open Sundays per participating store",
            "Minimum open Sundays",
            "Maximum open Sundays"
        ],
        "Value": [
            sunday_store_detail["Store"].nunique(),
            sunday_store_detail["Sunday_Open_Days"].mean(),
            sunday_store_detail["Sunday_Open_Days"].median(),
            sunday_store_detail["Sunday_Open_Days"].min(),
            sunday_store_detail["Sunday_Open_Days"].max()
        ]
    })

    # -----------------------------------------------------
    # 11. Promo2 participation and interval schedules
    # -----------------------------------------------------

    store_master = df.drop_duplicates("Store").copy()

    promo2_participation = (
        store_master.groupby("Promo2", dropna=False)
        .agg(Store_Count=("Store", "nunique"))
        .reset_index()
    )

    promo2_participation["Percentage_of_Stores"] = (
        100 * promo2_participation["Store_Count"]
        / promo2_participation["Store_Count"].sum()
    )

    promo2_stores = store_master[
        store_master["Promo2"] == 1
    ].copy()

    promo2_intervals = (
        promo2_stores.groupby("PromoInterval", dropna=False)
        .agg(Store_Count=("Store", "nunique"))
        .reset_index()
        .sort_values("Store_Count", ascending=False)
    )

    promo2_intervals["Percentage_of_Promo2_Stores"] = (
        100 * promo2_intervals["Store_Count"]
        / max(promo2_intervals["Store_Count"].sum(), 1)
    )

    promo2_data_quality = (
        store_master.groupby("Promo2", dropna=False)
        .agg(
            Stores=("Store", "nunique"),
            Missing_Start_Year=(
                "Promo2SinceYear",
                lambda x: x.isna().sum()
            ),
            Missing_Start_Week=(
                "Promo2SinceWeek",
                lambda x: x.isna().sum()
            ),
            Missing_Interval=(
                "PromoInterval",
                lambda x: x.isna().sum()
            )
        )
        .reset_index()
    )

    # Check whether active flags have valid programme metadata.
    promo2_flag_quality = pd.DataFrame({
        "Metric": [
            "Open observations",
            "Open observations with Promo2 active",
            "Open observations with Promo2 inactive",
            "Participating stores with missing start date",
            "Participating stores with missing interval"
        ],
        "Value": [
            len(open_df),
            int(open_df["Promo2_Active"].sum()),
            int((open_df["Promo2_Active"] == 0).sum()),
            int(
                promo2_stores["Promo2_Start_Date"].isna().sum()
            ),
            int(
                promo2_stores["PromoInterval"].isna().sum()
            )
        ]
    })


    # -----------------------------------------------------
    # 12. Competition analysis
    # -----------------------------------------------------

    distance = store_master["CompetitionDistance"].dropna()

    competition_distance = pd.DataFrame({
        "Metric": [
            "Total stores",
            "Stores with known competition distance",
            "Stores with missing competition distance",
            "Minimum distance (metres)",
            "25th percentile (metres)",
            "Median distance (metres)",
            "75th percentile (metres)",
            "90th percentile (metres)",
            "95th percentile (metres)",
            "Maximum distance (metres)"
        ],
        "Value": [
            store_master["Store"].nunique(),
            distance.notna().sum(),
            store_master["CompetitionDistance"].isna().sum(),
            distance.min(),
            distance.quantile(.25),
            distance.median(),
            distance.quantile(.75),
            distance.quantile(.90),
            distance.quantile(.95),
            distance.max()
        ]
    })

    # Competition metadata quality.
    competition_metadata = store_master.copy()

    competition_metadata["Competition_Info_Missing"] = (
        competition_metadata["CompetitionOpenSinceMonth"].isna()
        | competition_metadata["CompetitionOpenSinceYear"].isna()
    ).astype("int8")

    competition_metadata["CompetitionDistance_Missing"] = (
        competition_metadata["CompetitionDistance"].isna()
    ).astype("int8")

    competition_metadata_summary = pd.DataFrame({
        "Metric": [
            "Stores with missing competition opening month/year",
            "Stores with known competition opening information",
            "Stores with missing competition distance",
            "Stores with known competition distance"
        ],
        "Value": [
            int(
                competition_metadata[
                    "Competition_Info_Missing"
                ].sum()
            ),
            int(
                (
                    competition_metadata[
                        "Competition_Info_Missing"
                    ] == 0
                ).sum()
            ),
            int(
                competition_metadata[
                    "CompetitionDistance_Missing"
                ].sum()
            ),
            int(
                (
                    competition_metadata[
                        "CompetitionDistance_Missing"
                    ] == 0
                ).sum()
            )
        ]
    })

    # Store-level mean daily sales.
    # Each store contributes one mean value, calculated across
    # its available open-day training observations.
    store_sales = (
        open_df.groupby("Store")
        .agg(
            Store_Mean_Sales=("Sales", "mean"),
            Store_Median_Sales=("Sales", "median")
        )
        .reset_index()
    )

    competition_sales = store_master[
        [
            "Store",
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear"
        ]
    ].merge(
        store_sales,
        on="Store",
        how="left",
        validate="one_to_one"
    )

    valid_competition = competition_sales.dropna(
        subset=["CompetitionDistance", "Store_Mean_Sales"]
    )

    # Store-level Spearman correlation.
    # Each store contributes one observation.
    spearman_corr = valid_competition[
        ["CompetitionDistance", "Store_Mean_Sales"]
    ].corr(method="spearman").iloc[0, 1]

    competition_sales_summary = pd.DataFrame({
        "Metric": [
            "Stores with both distance and sales available",
            "Spearman correlation: distance vs store mean sales"
        ],
        "Value": [
            len(valid_competition),
            spearman_corr
        ]
    })

    # Fixed competition-distance bands, in metres.
    distance_bins = [
        0, 500, 1000, 2000, 5000, 10000, np.inf
    ]

    distance_labels = [
        "0-500 m",
        "501-1,000 m",
        "1,001-2,000 m",
        "2,001-5,000 m",
        "5,001-10,000 m",
        ">10,000 m"
    ]

    competition_sales["Competition_Distance_Band"] = pd.cut(
        competition_sales["CompetitionDistance"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True
    )

    # Calculate the average of store-level mean daily sales.
    # Every store receives equal weight within its distance band.
    competition_distance_bands = (
        competition_sales.dropna(
            subset=[
                "Competition_Distance_Band",
                "Store_Mean_Sales"
            ]
        )
        .groupby(
            "Competition_Distance_Band",
            observed=False
        )
        .agg(
            Number_of_Stores=("Store", "nunique"),
            Average_Store_Mean_Daily_Sales=(
                "Store_Mean_Sales", "mean"
            ),
            Median_Store_Mean_Daily_Sales=(
                "Store_Mean_Sales", "median"
            ),
            Mean_Competition_Distance_Metres=(
                "CompetitionDistance", "mean"
            )
        )
        .reset_index()
    )

    # Competition opening-age summary, by store.
    store_competition_age = (
        df.groupby("Store")
        .agg(
            CompetitionAgeMonths=(
                "CompetitionAgeMonths", "first"
            ),
            Competition_Info_Missing=(
                "Competition_Info_Missing", "first"
            )
        )
        .reset_index()
    )

    competition_age_summary = pd.DataFrame({
        "Metric": [
            "Stores with known opening month and year",
            "Stores with missing opening month or year",
            "Minimum competition age in historical data (months)",
            "Median competition age in historical data (months)",
            "Maximum competition age in historical data (months)"
        ],
        "Value": [
            int(
                store_competition_age[
                    "CompetitionAgeMonths"
                ].notna().sum()
            ),
            int(
                store_competition_age[
                    "CompetitionAgeMonths"
                ].isna().sum()
            ),
            store_competition_age[
                "CompetitionAgeMonths"
            ].min(),
            store_competition_age[
                "CompetitionAgeMonths"
            ].median(),
            store_competition_age[
                "CompetitionAgeMonths"
            ].max()
        ]
    })

    # -----------------------------------------------------
    # 13. Return all analytical tables
    # -----------------------------------------------------

    return {
        "Daily Trend": daily,
        "Weekday Analysis": weekday,
        "Monthly Analysis": monthly,
        "Promotion Analysis": promo,
        "Promotion Combinations": promotion_combinations,
        "Store Type": store_type,
        "Assortment": assortment,
        "State Holiday": state_holiday,
        "School Holiday": school_holiday,
        "Store Summary": store_summary,
        "Sales Quantiles": sales_quantiles,
        "Customer Sample": customer_sample,
        "Zero Sales": zero_sales,
        "Sunday Coverage": sunday_coverage,
        "Promo2 Participation": promo2_participation,
        "Promo2 Intervals": promo2_intervals,
        "Promo2 Data Quality": promo2_data_quality,
        "Promo2 Flag Quality": promo2_flag_quality,
        "Competition Distance": competition_distance,
        "Competition Metadata": competition_metadata_summary,
        "Competition Sales Summary": competition_sales_summary,
        "Competition Distance Bands": competition_distance_bands,
        "Competition Age Summary": competition_age_summary
    }


# =========================================================
# 6. GENERATE EDA CHARTS
# =========================================================

def generate_charts(df, tables):
    open_df = df[df["Open"] == 1].copy()

    # -----------------------------------------------------
    # Chart 1: Daily sales trend
    # -----------------------------------------------------

    daily = tables["Daily Trend"]

    fig, ax = plt.subplots()

    ax.plot(
        daily["Date"],
        daily["Mean_Sales"],
        alpha=0.30,
        linewidth=0.8,
        label="Daily mean"
    )

    ax.plot(
        daily["Date"],
        daily["Rolling_28D_Mean"],
        linewidth=2,
        label="28-day rolling mean"
    )

    # Label rolling mean at approximately 90-day intervals.
    label_indices = np.unique(
        np.linspace(
            0,
            len(daily) - 1,
            min(12, len(daily)),
            dtype=int
        )
    )

    for i in label_indices:
        ax.annotate(
            f"{daily.iloc[i]['Rolling_28D_Mean']:,.0f}",
            (
                daily.iloc[i]["Date"],
                daily.iloc[i]["Rolling_28D_Mean"]
            ),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=7,
            ha="center"
        )

    ax.set_title("Daily Average Sales per Open Store")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Sales")
    ax.legend()

    save_chart("01_daily_sales_trend.png")

    # -----------------------------------------------------
    # Chart 2: Weekday sales
    # -----------------------------------------------------

    weekday = tables["Weekday Analysis"]

    plot_annotated_bars(
        weekday,
        "Day",
        "Mean_Sales",
        "Average Sales by Day of Week",
        "Day of Week",
        "Average Sales on Open Days",
        "02_sales_by_weekday.png",
        fmt="{:,.0f}",
        rotation=0
    )

    # -----------------------------------------------------
    # Chart 3: Monthly seasonality
    # -----------------------------------------------------

    monthly = tables["Monthly Analysis"]

    fig, ax = plt.subplots()

    ax.plot(
        monthly["Month_Name"],
        monthly["Mean_Sales"],
        marker="o",
        linewidth=2
    )

    for _, row in monthly.iterrows():
        ax.annotate(
            f"{row['Mean_Sales']:,.0f}",
            (row["Month_Name"], row["Mean_Sales"]),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=8,
            ha="center"
        )

    ax.set_title("Average Sales by Calendar Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales on Open Days")
    ax.tick_params(axis="x", rotation=30)

    save_chart("03_sales_by_month.png")

    # -----------------------------------------------------
    # Chart 4: Sales distribution
    # -----------------------------------------------------

    fig, ax = plt.subplots()

    sns.histplot(
        open_df["Sales"],
        bins=40,
        kde=True,
        ax=ax
    )

    p99 = open_df["Sales"].quantile(0.99)

    ax.axvline(
        p99,
        linestyle="--",
        label=f"99th percentile: {p99:,.0f}"
    )

    ax.set_title("Distribution of Sales on Open Store-Days")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Number of Store-Days")
    ax.legend()

    save_chart("04_sales_distribution.png")

    # -----------------------------------------------------
    # Chart 5: Daily Promo comparison
    # -----------------------------------------------------

    promo = tables["Promotion Analysis"].copy()

    plot_annotated_bars(
        promo,
        "Promotion",
        "Mean_Sales",
        "Average Sales: Daily Promo vs No Daily Promo",
        "Daily Promo",
        "Average Sales on Open Days",
        "05_promotion_comparison.png",
        fmt="{:,.0f}"
    )

    # -----------------------------------------------------
    # Chart 6: Customers versus sales
    # -----------------------------------------------------

    sample = open_df.sample(
        n=min(15000, len(open_df)),
        random_state=42
    )

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=sample,
        x="Customers",
        y="Sales",
        alpha=0.25,
        s=15,
        ax=ax
    )

    corr = open_df[["Customers", "Sales"]].corr().iloc[0, 1]

    ax.set_title(
        f"Customers vs Sales (Pearson correlation = {corr:.3f})"
    )
    ax.set_xlabel("Customers")
    ax.set_ylabel("Sales")

    save_chart("06_customers_vs_sales.png")

    # -----------------------------------------------------
    # Chart 7: Store type
    # -----------------------------------------------------

    plot_annotated_bars(
        tables["Store Type"],
        "StoreType",
        "Mean_Sales",
        "Average Sales by Store Type",
        "Store Type",
        "Average Sales on Open Days",
        "07_sales_by_store_type.png",
        fmt="{:,.0f}"
    )

    # -----------------------------------------------------
    # Chart 8: State holiday
    # -----------------------------------------------------

    holiday = tables["State Holiday"].copy()

    holiday["StateHoliday"] = (
        holiday["StateHoliday"].fillna("Missing").astype(str)
    )

    plot_annotated_bars(
        holiday,
        "StateHoliday",
        "Mean_Sales",
        "Average Sales by State Holiday Category",
        "State Holiday Category",
        "Average Sales on Open Days",
        "08_sales_by_state_holiday.png",
        fmt="{:,.0f}"
    )

    # -----------------------------------------------------
    # Chart 9: Store-level variation
    # -----------------------------------------------------

    fig, ax = plt.subplots()

    sns.histplot(
        tables["Store Summary"]["Mean_Sales"],
        bins=35,
        kde=True,
        ax=ax
    )

    median_store_sales = (
        tables["Store Summary"]["Mean_Sales"].median()
    )

    ax.axvline(
        median_store_sales,
        linestyle="--",
        label=f"Median store mean: {median_store_sales:,.0f}"
    )

    ax.set_title("Variation in Average Sales Across Stores")
    ax.set_xlabel("Store-Level Mean Sales")
    ax.set_ylabel("Number of Stores")
    ax.legend()

    save_chart("09_store_sales_variation.png")

    # -----------------------------------------------------
    # Chart 10: Four promotion combinations
    # -----------------------------------------------------

    promotion_combinations = tables[
        "Promotion Combinations"
    ].copy()

    plot_annotated_bars(
        promotion_combinations,
        "Promotion_Group",
        "Mean_Sales",
        "Average Sales Across Four Promotion Combinations",
        "Promotion Combination",
        "Average Sales on Open Days",
        "10_four_promotion_combinations.png",
        fmt="{:,.0f}",
        rotation=15
    )

    # -----------------------------------------------------
    # Chart 11: Competition distance bands
    # -----------------------------------------------------




    competition_bands = tables[
        "Competition Distance Bands"
    ].copy()

    plot_annotated_bars(
        competition_bands,
        "Competition_Distance_Band",
        "Average_Store_Mean_Daily_Sales",
        "Average of Store-Level Mean Daily Sales by Competition Distance",
        "Distance to Nearest Competitor",
        "Average Store-Level Mean Daily Sales",
        "11_sales_by_competition_distance.png",
        fmt="{:,.0f}",
        rotation=20,
        count_column="Number_of_Stores"
    )


# =========================================================
# 7. GENERATE MARKDOWN REPORT
# =========================================================

def generate_report(df, tables):
    open_df = df[df["Open"] == 1].copy()

    weekday = tables["Weekday Analysis"]
    monthly = tables["Monthly Analysis"]
    promo = tables["Promotion Analysis"]
    combinations = tables["Promotion Combinations"]
    store_type = tables["Store Type"]
    assortment = tables["Assortment"]
    holiday = tables["State Holiday"]
    school_holiday = tables["School Holiday"]
    store_summary = tables["Store Summary"]
    quantiles = tables["Sales Quantiles"]
    daily = tables["Daily Trend"]

    corr = open_df[["Sales", "Customers"]].corr().iloc[0, 1]

    promo_means = promo.set_index("Promo")["Mean_Sales"]

    no_promo_mean = promo_means.get(0, np.nan)
    promo_mean = promo_means.get(1, np.nan)

    promo_change = (
        (promo_mean / no_promo_mean) - 1
    ) * 100 if no_promo_mean else np.nan

    weekday_high = weekday.loc[weekday["Mean_Sales"].idxmax()]
    weekday_low = weekday.loc[weekday["Mean_Sales"].idxmin()]

    month_high = monthly.loc[monthly["Mean_Sales"].idxmax()]
    month_low = monthly.loc[monthly["Mean_Sales"].idxmin()]

    type_high = store_type.loc[store_type["Mean_Sales"].idxmax()]
    type_low = store_type.loc[store_type["Mean_Sales"].idxmin()]

    highest_daily = daily.loc[daily["Mean_Sales"].idxmax()]
    lowest_daily = daily.loc[daily["Mean_Sales"].idxmin()]

    closed_rows = int((df["Open"] == 0).sum())

    closed_nonzero = int(
        ((df["Open"] == 0) & (df["Sales"] != 0)).sum()
    )

    open_zero_sales = int(
        ((df["Open"] == 1) & (df["Sales"] == 0)).sum()
    )

    negative_sales = int((df["Sales"] < 0).sum())

    sunday_stores = int(
        open_df.loc[
            open_df["DayOfWeek"] == 7,
            "Store"
        ].nunique()
    )

    # Four promotion group results.
    neither = combinations.loc[
        combinations["Promotion_Group"] == "Neither",
        "Mean_Sales"
    ].iloc[0]

    promo2_only = combinations.loc[
        combinations["Promotion_Group"] == "Promo2 only",
        "Mean_Sales"
    ].iloc[0]

    promo2_difference = (
        (promo2_only / neither) - 1
    ) * 100 if neither else np.nan

    # Competition results.
    competition_corr = (
        tables["Competition Sales Summary"].iloc[1]["Value"]
    )

    store_min = store_summary["Mean_Sales"].min()
    store_median = store_summary["Mean_Sales"].median()
    store_max = store_summary["Mean_Sales"].max()

    report = f"""# Rossmann Sales Forecasting — Exploratory Data Analysis

## Executive overview

This analysis examines historical store-level sales, calendar patterns,
daily promotions, the recurring Promo2 programme, customer counts,
store characteristics, holidays and competition attributes.

The objective is to identify useful forecasting signals, understand
store-level heterogeneity, assess data behaviour and establish
candidate features for a global forecasting model.

All sales comparisons use open store-day observations unless
otherwise stated. Findings are descriptive, not causal.

## 1. Dataset scope and validation

| Metric | Value |
|---|---:|
| Training records | {len(df):,} |
| Stores | {df['Store'].nunique():,} |
| Historical start | {df['Date'].min().date()} |
| Historical end | {df['Date'].max().date()} |
| Open store-day observations | {len(open_df):,} |
| Closed store-day observations | {closed_rows:,} |
| Open-store zero-sales observations | {open_zero_sales:,} |
| Negative sales observations | {negative_sales:,} |

The calendar weekday field was checked against the actual date.
No weekday mismatches were found.

## 2. Daily average sales trend

- Highest daily average: {highest_daily['Mean_Sales']:,.2f}
  on {highest_daily['Date'].date()}.
- Lowest daily average: {lowest_daily['Mean_Sales']:,.2f}
  on {lowest_daily['Date'].date()}.

The 28-day rolling mean helps distinguish medium-term movement
from individual-day fluctuations.

**Modelling implication:** Retain calendar features and evaluate
temporal patterns through future-period validation. Aggregate
movement should not be interpreted as a change in every store.

## 3. Sales by weekday

{md_table(weekday, [
    "Day", "Observations", "Stores",
    "Mean_Sales", "Median_Sales", "Mean_Customers"
])}

- Highest mean: {weekday_high['Mean_Sales']:,.2f}
  on {weekday_high['Day']}.
- Lowest mean: {weekday_low['Mean_Sales']:,.2f}
  on {weekday_low['Day']}.
- Stores with at least one open Sunday: {sunday_stores}.

Sunday observations have a much narrower store population.
The Sunday aggregate is therefore not directly comparable with
weekday averages as though the same stores contributed.

**Modelling implication:** Include weekday as a candidate feature,
but evaluate store-level and weekday-level performance.

## 4. Monthly seasonality

{md_table(monthly, [
    "Month_Name", "Observations", "Mean_Sales", "Median_Sales"
])}

- Highest mean: {month_high['Mean_Sales']:,.2f}
  in {month_high['Month_Name']}.
- Lowest mean: {month_low['Mean_Sales']:,.2f}
  in {month_low['Month_Name']}.

The historical period is approximately two and a half years.
Long-term seasonal conclusions should therefore remain cautious.

**Modelling implication:** Test month, week-of-year and other
calendar features through time-based validation.

## 5. Sales distribution and extreme values

{md_table(quantiles)}

- Mean sales on open days: {open_df['Sales'].mean():,.2f}.
- Median sales on open days: {open_df['Sales'].median():,.2f}.
- Minimum: {open_df['Sales'].min():,.0f}.
- Maximum: {open_df['Sales'].max():,.0f}.

The upper tail may include genuine high-demand, seasonal and
promotional observations.

**Modelling implication:** Do not remove extreme values solely
because they are high. Evaluate MAE, RMSE, WMAPE and RMSLE.

## 6. Daily Promo analysis

{md_table(promo, [
    "Promotion", "Observations", "Stores",
    "Mean_Sales", "Median_Sales", "Mean_Customers"
])}

- Mean sales without daily Promo: {no_promo_mean:,.2f}.
- Mean sales with daily Promo: {promo_mean:,.2f}.
- Raw mean difference: {promo_change:+.2f}%.

This is a descriptive comparison, not a causal estimate.
Store mix, calendar effects and promotion selection may influence it.

**Modelling implication:** Include daily Promo as a candidate
feature and separately evaluate forecast accuracy on promotion days.

## 7. Four-way promotion analysis: Promo and Promo2

### Promotion definitions

- `Promo` is the daily promotion indicator in train.csv and test.csv.
- `Promo2` is store-level participation in the recurring programme.
- `Promo2SinceYear` and `Promo2SinceWeek` indicate when participation began.
- `PromoInterval` specifies recurring promotion-round months.

### Four combinations

{md_table(combinations, [
    "Promotion_Group", "Observations", "Stores",
    "Mean_Sales", "Median_Sales",
    "Mean_Customers", "Mean_Sales_Difference_Percent"
])}

**Primary comparison: Promo2 only versus neither promotion active.**

- Neither active mean sales: {neither:,.2f}.
- Promo2-only mean sales: {promo2_only:,.2f}.
- Raw mean difference: {promo2_difference:+.2f}%.

**Interpretation:** The four combinations allow us to compare
observations with neither promotion, Promo2 only, daily Promo only,
and both active. These differences are descriptive, not causal.
Store characteristics, seasonality and the composition of each
group can influence observed sales.

**Modelling implication:** Include daily `Promo` and the derived
date-level `Promo2_Active` as separate candidate features.
Their interaction can also be tested by a tree-based global model.
Evaluate performance separately for all four combinations.

## 8. Customer counts versus sales

- Pearson correlation on open store-days: {corr:.4f}.
- Mean customers: {open_df['Customers'].mean():,.2f}.
- Median customers: {open_df['Customers'].median():,.2f}.

Customer count is historically associated with sales, but
Customers is absent from the test dataset and is not known
for the future forecast horizon.

**Modelling implication:** Do not use actual future customer
counts as predictors. This would create a training-serving
availability mismatch and potential leakage.

## 9. Store type

{md_table(store_type, [
    "StoreType", "Stores", "Observations",
    "Mean_Sales", "Median_Sales", "Mean_Customers"
])}

- Highest mean: Type {type_high['StoreType']}
  ({type_high['Mean_Sales']:,.2f}).
- Lowest mean: Type {type_low['StoreType']}
  ({type_low['Mean_Sales']:,.2f}).

**Modelling implication:** Include store type in the global model
and report performance by store type.

## 10. Assortment

{md_table(assortment, [
    "Assortment", "Stores", "Observations",
    "Mean_Sales", "Median_Sales"
])}

Assortment may capture differences in product range and should
be tested alongside store type and store identity.

## 11. Holiday effects

### State holidays

{md_table(holiday, [
    "StateHoliday", "Observations", "Mean_Sales", "Median_Sales"
])}

### School holidays

{md_table(school_holiday, [
    "SchoolHoliday", "Observations", "Mean_Sales", "Median_Sales"
])}

Some holiday categories may have relatively few observations.
Aggregate averages can also be influenced by the stores and dates
represented in each category.

**Modelling implication:** Retain known holiday indicators as
candidate features and evaluate holiday-period performance.

## 12. Store-level variation

- Lowest store mean: {store_min:,.2f}.
- Median store mean: {store_median:,.2f}.
- Highest store mean: {store_max:,.2f}.
- Number of stores: {len(store_summary):,}.

Store-level sales heterogeneity supports starting with a pooled
global model that learns across stores while incorporating store
identity and store attributes.

## 13. Promo2 programme metadata and feature engineering

### Store participation

{md_table(tables["Promo2 Participation"])}

### Recurring interval schedules

{md_table(tables["Promo2 Intervals"])}

### Metadata completeness

{md_table(tables["Promo2 Data Quality"])}

### Derived flag quality

{md_table(tables["Promo2 Flag Quality"])}

The date-level `Promo2_Active` feature is constructed using:

1. `Promo2 == 1`.
2. Observation date on or after the recorded ISO-week start date.
3. Observation month present in the store's recurring
   `PromoInterval`.

The recurring month schedule is applied in subsequent calendar
years. Dates before the actual programme start date remain inactive.

**Important interpretation:** The listed months are treated as
active promotion months, not as continuous activity throughout
the three-month interval. This is the agreed modelling assumption
for this analysis and should be documented if the programme's
operational definition is later clarified.

The same feature-engineering logic must be applied to training
and test dates.

## 14. Competition analysis

### Competition distance distribution

{md_table(tables["Competition Distance"])}

### Competition metadata completeness

{md_table(tables["Competition Metadata"])}

### Sales by competition-distance band

{md_table(tables["Competition Distance Bands"])}

### Competition age

{md_table(tables["Competition Age Summary"])}

### Distance and store sales relationship

{md_table(tables["Competition Sales Summary"])}

**Interpretation and modelling implications:**

- `CompetitionDistance` is a static store-level feature.
- `CompetitionAgeMonths` represents the number of calendar
  months since the recorded competitor opening date, relative
  to each observation date.
- Negative age values indicate that the recorded opening date
  is later than the observation month.
- Three stores have missing competition distance, requiring
  a documented missing-value treatment.
- Missing competition opening month/year must not be treated
  as a known opening date.
- The Spearman correlation and distance-band averages are
  descriptive; they do not establish causality.

**Candidate global-model features:**

- `CompetitionDistance`
- `CompetitionDistance_Missing`
- `CompetitionAgeMonths`
- `Competition_Info_Missing`

The same metadata and date-based feature logic should be
available for test records.

## 15. Zero-sales and closure behaviour

- Closed-store observations: {closed_rows:,}.
- Closed observations with non-zero sales: {closed_nonzero:,}.
- Open-store observations with zero sales: {open_zero_sales:,}.
- Negative sales observations: {negative_sales:,}.

Closed-store observations with zero sales are consistent with
closure. Open-store zero-sales observations should not be
automatically removed or imputed.

## 16. Business impact considerations

| Forecast behaviour | Business consequence | KPI impact | Potential action |
|---|---|---|---|
| Underforecast | Risk of stock out | Availability falls | Increase safety stock |
| Overforecast | Excess inventory | Inventory efficiency falls | Reduce replenishment |
| Promotion underforecast | Too little stock for a promotion | Promotion revenue falls | Adjust promotion planning |

The final solution should consider the asymmetric operational
consequences of underforecasting and overforecasting.

Actual inventory costs and stockout penalties are not supplied
in this dataset. Any asymmetric loss weights must be labelled
as business assumptions and tested through sensitivity analysis.

## 17. Consolidated modelling decisions

| Finding | Modelling decision |
|---|---|
| Weekday/month variation | Calendar features |
| Daily Promo | Include as candidate feature |
| Recurring Promo2 schedule | Derive date-level Promo2_Active |
| Four promotion combinations | Evaluate separately |
| Competition distance | Include numeric feature |
| Competition opening date | Derive CompetitionAgeMonths |
| Missing competition metadata | Retain explicit missingness flags |
| Customer-sales association | Exclude actual future Customers |
| Store-type and assortment variation | Include store metadata |
| Store-level heterogeneity | Start with pooled global model |
| High-sales tail | Retain and evaluate multiple metrics |
| Holiday variation | Include known holiday indicators |
| Sunday coverage | Avoid universal Sunday uplift assumptions |
| Closed-store observations | Apply known closure constraints |
| Open-store zero sales | Retain and investigate |

## 18. Limitations and next steps

1. EDA comparisons are descriptive, not causal.
2. Customer counts are unavailable at inference time.
3. Store composition can affect aggregate comparisons.
4. Historical seasonal coverage is limited.
5. The Promo2 active-month interpretation is an explicit
   feature-engineering assumption.
6. Feature usefulness must be assessed through time-aware
   validation and holdout evaluation.

**Next step:** Build leakage-safe features and compare operational
baselines with at least three different forecasting approaches
using a six-week validation horizon.
"""

    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Created report: {OUTPUT_REPORT.name}")


# =========================================================
# 8. EXPORT SUPPORTING TABLES TO EXCEL
# =========================================================

def export_excel(tables):
    print("Exporting supporting EDA tables...")

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        for sheet_name, table in tables.items():
            table.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False
            )

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for column_cells in ws.columns:
                max_length = max(
                    len(str(cell.value))
                    if cell.value is not None else 0
                    for cell in column_cells[:200]
                )

                column_letter = column_cells[0].column_letter

                ws.column_dimensions[column_letter].width = min(
                    max(max_length + 2, 12),
                    36
                )

            for cell in ws[1]:
                new_font = copy(cell.font)
                new_font.bold = True
                cell.font = new_font

    print(f"Created workbook: {OUTPUT_EXCEL.name}")


# =========================================================
# 9. MAIN EXECUTION
# =========================================================

def run_eda():
    df = load_data()

    print("\nBuilding analytical tables...")
    tables = build_tables(df)

    print("\nGenerating charts...")
    generate_charts(df, tables)

    print("\nGenerating interpretation report...")
    generate_report(df, tables)

    print("\nExporting Excel workbook...")
    export_excel(tables)

    print("\nEDA completed successfully.")
    print(f"Charts: {FIGURES_DIR}")
    print(f"Report: {OUTPUT_REPORT}")
    print(f"Workbook: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    run_eda()
