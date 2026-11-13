
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS AND FORECAST CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

TRAIN_PATH = PROCESSED_DIR / "train_preprocessed.pkl"
TEST_PATH = PROCESSED_DIR / "test_preprocessed.pkl"

TRAIN_OUTPUT = PROCESSED_DIR / "train_features.pkl"
VALIDATION_OUTPUT = PROCESSED_DIR / "validation_base.pkl"
TEST_OUTPUT = PROCESSED_DIR / "test_base.pkl"

CUTOFF_DATE = pd.Timestamp("2015-06-19")
VALIDATION_START = pd.Timestamp("2015-06-20")
VALIDATION_END = pd.Timestamp("2015-07-31")


# ============================================================
# 2. LOAD PREPROCESSED DATA
# ============================================================

def load_data():
    print("Loading preprocessed datasets...")

    train = pd.read_pickle(TRAIN_PATH)
    test = pd.read_pickle(TEST_PATH)

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    train = train.sort_values(["Store", "Date"]).reset_index(drop=True)
    test = test.sort_values(["Store", "Date"]).reset_index(drop=True)

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    return train, test


# ============================================================
# 3. PROMO2 START DATE
# ============================================================

def get_promo2_start_date(row):
    year = row.get("Promo2SinceYear")
    week = row.get("Promo2SinceWeek")

    if pd.isna(year) or pd.isna(week):
        return pd.NaT

    try:
        return pd.Timestamp(
            date.fromisocalendar(int(year), int(week), 1)
        )
    except (ValueError, TypeError):
        return pd.NaT


# ============================================================
# 4. BASE FEATURE ENGINEERING
# ============================================================

def build_base_features(df):
    print("Creating calendar, holiday, store and promotion features...")

    df = df.copy()

    # -------------------------
    # Calendar features
    # -------------------------

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    df["IsWeekend"] = (
        df["DayOfWeek"].isin([6, 7]).astype("int8")
    )

    # Cyclical calendar representation
    df["DayOfWeek_sin"] = np.sin(
        2 * np.pi * df["DayOfWeek"] / 7
    )
    df["DayOfWeek_cos"] = np.cos(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    df["Month_sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )
    df["Month_cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    # -------------------------
    # Holiday features
    # -------------------------

    # Preserve StateHoliday as a categorical value.
    df["StateHoliday"] = (
        df["StateHoliday"]
        .astype("string")
        .str.strip()
        .replace({"0.0": "0", "nan": "0"})
        .fillna("0")
    )

    df["SchoolHoliday"] = (
        df["SchoolHoliday"].fillna(0).astype("int8")
    )


    # -------------------------
    # Competition features
    # -------------------------

    if {
        "CompetitionOpenSinceYear",
        "CompetitionOpenSinceMonth",
    }.issubset(df.columns):

        competition_year = pd.to_numeric(
            df["CompetitionOpenSinceYear"], errors="coerce"
        )

        competition_month = pd.to_numeric(
            df["CompetitionOpenSinceMonth"], errors="coerce"
        )

        # Flag implausible source years.
        # 1900 is specifically observed for Store 815.
        df["CompetitionOpenSince_invalid"] = (
            competition_year.eq(1900)
        ).astype("int8")

        # Exclude invalid years from age calculation.
        valid_competition_date = (
            competition_year.notna()
            & competition_month.between(1, 12)
            & competition_year.ne(1900)
        )

        df["CompetitionAgeMonths"] = (
            (df["Year"] - competition_year) * 12
            + (df["Month"] - competition_month)
        ).where(valid_competition_date)

        # A negative age means the competitor has
        # not yet opened as of the observation date.
        df["Competition_NotYetOpen"] = (
            df["CompetitionAgeMonths"].lt(0)
        ).astype("int8")

    else:
        df["CompetitionOpenSince_invalid"] = 0
        df["CompetitionAgeMonths"] = np.nan
        df["Competition_NotYetOpen"] = 0


    # -------------------------
    # Promo2 features
    # -------------------------

    if "Promo2" in df.columns:
        df["Promo2"] = df["Promo2"].fillna(0).astype("int8")
    else:
        df["Promo2"] = 0

    if "PromoInterval" in df.columns:
        intervals = (
            df["PromoInterval"]
            .fillna("")
            .astype(str)
            .str.upper()
        )
    else:
        intervals = pd.Series("", index=df.index)

    if {
        "Promo2SinceYear",
        "Promo2SinceWeek",
    }.issubset(df.columns):

        df["Promo2StartDate"] = df.apply(
            get_promo2_start_date, axis=1
        )
    else:
        df["Promo2StartDate"] = pd.NaT

    month_number = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "SEPT": 9, "OCT": 10,
        "NOV": 11, "DEC": 12,
    }

    active_months = intervals.apply(
        lambda value: {
            month_number[item.strip()]
            for item in value.split(",")
            if item.strip() in month_number
        }
    )

    df["Promo2_Active"] = [
        int(
            participation == 1
            and pd.notna(start_date)
            and current_date >= start_date
            and current_date.month in months
        )
        for participation, start_date, current_date, months
        in zip(
            df["Promo2"],
            df["Promo2StartDate"],
            df["Date"],
            active_months,
        )
    ]

    df["Promo2_Active"] = df["Promo2_Active"].astype("int8")

    # Promo2 metadata completeness indicator.
    if "Promo2_metadata_missing" in df.columns:
        df["Promo2_metadata_missing"] = (
            df["Promo2_metadata_missing"].fillna(0).astype("int8")
        )
    else:
        df["Promo2_metadata_missing"] = 0

    # Remove the intermediate date; retain source metadata for audit.
    df = df.drop(columns=["Promo2StartDate"])

    return df


# ============================================================
# 5. HISTORICAL SALES FEATURES
# ============================================================

def add_training_history_features(df):
    """
    Creates historical features using only earlier dates.

    The shift(1) is essential:
    today's Sales must never be used to calculate
    today's lag or rolling features.
    """

    print("Creating leakage-safe historical sales features...")

    df = df.sort_values(["Store", "Date"]).copy()

    store_sales = df.groupby("Store", sort=False)["Sales"]

    # -------------------------
    # Lag features
    # -------------------------

    for lag in [1, 7, 14, 28]:
        df[f"Sales_Lag_{lag}"] = store_sales.shift(lag)

    # -------------------------
    # Rolling averages
    # -------------------------

    for window in [7, 14, 28]:
        df[f"Sales_RollingMean_{window}"] = (
            df.groupby("Store", sort=False)["Sales"]
            .transform(
                lambda values: (
                    values.shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            )
        )

    # -------------------------
    # Expanding historical averages
    # -------------------------

    df["Store_HistoricalMean"] = (
        df.groupby("Store", sort=False)["Sales"]
        .transform(
            lambda values: (
                values.shift(1).expanding(min_periods=1).mean()
            )
        )
    )

    df["Store_Weekday_HistoricalMean"] = (
        df.groupby(
            ["Store", "DayOfWeek"], sort=False
        )["Sales"]
        .transform(
            lambda values: (
                values.shift(1).expanding(min_periods=1).mean()
            )
        )
    )

    return df


# ============================================================
# 6. MAIN FEATURE GENERATION
# ============================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_data()

    # Split before generating historical features.
    # Validation sales must not enter training history.
    training_period = train[
        train["Date"] <= CUTOFF_DATE
    ].copy()

    validation_period = train[
        (train["Date"] >= VALIDATION_START)
        & (train["Date"] <= VALIDATION_END)
    ].copy()

    if training_period.empty or validation_period.empty:
        raise ValueError("Training or validation split is empty.")

    if training_period["Date"].max() >= validation_period["Date"].min():
        raise ValueError("Training and validation dates overlap.")

    if validation_period["Date"].min() != VALIDATION_START:
        raise ValueError("Validation period does not start on expected date.")

    if validation_period["Date"].max() != VALIDATION_END:
        raise ValueError("Validation period does not end on expected date.")

    print("\nApplying base feature engineering...")

    training_period = build_base_features(training_period)
    validation_period = build_base_features(validation_period)
    test = build_base_features(test)

    # Historical features are calculated only on the
    # training period. No validation/test actual Sales
    # values are used to construct their features.
    training_features = add_training_history_features(
        training_period
    )

    # Validation and test will receive historical features
    # recursively during model forecasting.
    training_features.to_pickle(TRAIN_OUTPUT)
    validation_period.to_pickle(VALIDATION_OUTPUT)
    test.to_pickle(TEST_OUTPUT)

    # -------------------------
    # Validation checks
    # -------------------------

    assert training_features["Date"].max() == CUTOFF_DATE
    assert validation_period["Date"].min() == VALIDATION_START
    assert validation_period["Date"].max() == VALIDATION_END

    assert "Customers" in training_features.columns
    assert "Id" in test.columns
    assert "Sales" in training_features.columns

    # Check that historical features are not all missing.
    lag_columns = [
        "Sales_Lag_1",
        "Sales_Lag_7",
        "Sales_Lag_14",
        "Sales_Lag_28",
    ]

    print("\nHistorical feature missingness:")
    print(
        training_features[lag_columns]
        .isna()
        .sum()
        .to_string()
    )

    print("\nFEATURE ENGINEERING COMPLETED")
    print(f"Training feature rows: {len(training_features):,}")
    print(f"Validation base rows: {len(validation_period):,}")
    print(f"Test base rows: {len(test):,}")
    print(
        "Training date range:",
        training_features["Date"].min().date(),
        "to",
        training_features["Date"].max().date(),
    )
    print(
        "Validation date range:",
        validation_period["Date"].min().date(),
        "to",
        validation_period["Date"].max().date(),
    )
    print(
        "Validation unique stores:",
        validation_period["Store"].nunique(),
    )

    print("\nSaved files:")
    print(TRAIN_OUTPUT)
    print(VALIDATION_OUTPUT)
    print(TEST_OUTPUT)


if __name__ == "__main__":
    main()
