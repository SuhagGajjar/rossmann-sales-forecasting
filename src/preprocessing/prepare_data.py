
from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    store = pd.read_csv(RAW_DIR / "store.csv")

    return train, test, store


def validate_inputs(train, test, store):
    required_train = {
        "Store", "DayOfWeek", "Date", "Sales", "Customers",
        "Open", "Promo", "StateHoliday", "SchoolHoliday"
    }

    required_test = {
        "Id", "Store", "DayOfWeek", "Date",
        "Open", "Promo", "StateHoliday", "SchoolHoliday"
    }

    required_store = {
        "Store", "StoreType", "Assortment", "CompetitionDistance",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"
    }

    assert required_train.issubset(train.columns), (
        f"Missing train columns: {required_train - set(train.columns)}"
    )
    assert required_test.issubset(test.columns), (
        f"Missing test columns: {required_test - set(test.columns)}"
    )
    assert required_store.issubset(store.columns), (
        f"Missing store columns: {required_store - set(store.columns)}"
    )

    assert not store["Store"].duplicated().any(), (
        "Store master contains duplicate Store IDs."
    )

    assert not train.duplicated(["Store", "Date"]).any(), (
        "Duplicate Store-Date combinations in train."
    )

    assert not test.duplicated(["Store", "Date"]).any(), (
        "Duplicate Store-Date combinations in test."
    )

    assert train["Store"].isin(store["Store"]).all()
    assert test["Store"].isin(store["Store"]).all()

    assert test["Id"].is_unique, "Test Id is not unique."

    train["Date"] = pd.to_datetime(
        train["Date"], errors="raise"
    )
    test["Date"] = pd.to_datetime(
        test["Date"], errors="raise"
    )

    return train, test, store


def infer_missing_open(train, test):
    """
    Infer missing test Open values using the historical
    store-specific weekday opening pattern.

    The historical mode is calculated exclusively from
    training data, avoiding test-target leakage.
    """

    train["Open_was_missing"] = (
        train["Open"].isna().astype("int8")
    )
    test["Open_was_missing"] = (
        test["Open"].isna().astype("int8")
    )

    train["Open_imputation_method"] = "not_missing"
    test["Open_imputation_method"] = "not_missing"

    if train["Open"].isna().any():
        raise ValueError(
            "Unexpected missing Open values in training data. "
            "Review before proceeding."
        )

    historical_pattern = (
        train.groupby(["Store", "DayOfWeek"])["Open"]
        .agg(lambda x: x.mode().iloc[0])
    )

    missing_rows = test["Open"].isna()

    for idx in test.index[missing_rows]:
        store_id = test.at[idx, "Store"]
        weekday = test.at[idx, "DayOfWeek"]

        key = (store_id, weekday)

        if key not in historical_pattern.index:
            raise ValueError(
                f"No historical opening pattern for {key}"
            )

        test.at[idx, "Open"] = historical_pattern.loc[key]
        test.at[idx, "Open_imputation_method"] = (
            "historical_store_weekday_mode"
        )

    train["Open"] = train["Open"].astype("int8")
    test["Open"] = test["Open"].astype("int8")

    return train, test


def clean_store_master(store):
    store = store.copy()

    # Competition distance: three missing stores.
    store["CompetitionDistance_was_missing"] = (
        store["CompetitionDistance"].isna().astype("int8")
    )

    distance_median = store["CompetitionDistance"].median()

    store["CompetitionDistance"] = (
        store["CompetitionDistance"].fillna(distance_median)
    )

    # Competition opening metadata may be incomplete.
    store["CompetitionOpenSince_was_missing"] = (
        store[
            [
                "CompetitionOpenSinceMonth",
                "CompetitionOpenSinceYear"
            ]
        ].isna().any(axis=1).astype("int8")
    )

    # Promo2 missing metadata is structural for nonparticipants.
    # Do not fill these fields with invented dates or intervals.
    promo2_metadata = [
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval"
    ]

    store["Promo2_metadata_missing"] = (
        (
            store["Promo2"].eq(1)
            & store[promo2_metadata].isna().any(axis=1)
        ).astype("int8")
    )

    return store


def merge_store_data(data, store):
    original_rows = len(data)

    merged = data.merge(
        store,
        on="Store",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    unmatched = int((merged["_merge"] != "both").sum())

    if unmatched:
        raise ValueError(
            f"{unmatched} rows could not be matched to store master."
        )

    merged.drop(columns="_merge", inplace=True)

    if len(merged) != original_rows:
        raise ValueError("Row count changed during store merge.")

    return merged


def validate_prepared_data(train, test):
    if train["Sales"].isna().any():
        raise ValueError("Training Sales contains missing values.")

    if (train["Sales"] < 0).any():
        raise ValueError("Negative Sales values detected.")

    if not train["Open"].isin([0, 1]).all():
        raise ValueError("Invalid Open values in train.")

    if not test["Open"].isin([0, 1]).all():
        raise ValueError("Invalid Open values in test.")

    # Closed stores should have zero sales.
    closed_nonzero = int(
        ((train["Open"] == 0) & (train["Sales"] != 0)).sum()
    )

    return closed_nonzero


def save_outputs(train, test, store, closed_nonzero):
    train_path = PROCESSED_DIR / "train_preprocessed.pkl"
    test_path = PROCESSED_DIR / "test_preprocessed.pkl"
    store_path = PROCESSED_DIR / "store_preprocessed.pkl"

    train.to_pickle(train_path)
    test.to_pickle(test_path)
    store.to_pickle(store_path)

    missing_open_rows = test[
        test["Open_was_missing"] == 1
    ]

    report = f"""
# Preprocessing Methodology and Findings

## 1. Outputs

- Training rows: {len(train):,}
- Test rows: {len(test):,}
- Store master rows: {len(store):,}

Prepared files:
- data/processed/train_preprocessed.pkl
- data/processed/test_preprocessed.pkl
- data/processed/store_preprocessed.pkl

## 2. Store master integration

Store attributes were joined using Store as the key.
A many-to-one merge validation was applied to prevent
accidental row multiplication.

All training and test stores matched the store master.

## 3. Missing Open values

- Missing Open values in test: {len(missing_open_rows)}
- Imputation: historical store-specific weekday mode.
- Indicator retained: Open_was_missing.
- Method recorded in: Open_imputation_method.

Training Open values were not imputed.

Store 622's historical weekday pattern supports
Monday-Saturday opening and Sunday closure for the
11 affected test observations.

This is an inference, not confirmed ground truth.

## 4. Competition metadata

- Missing CompetitionDistance values were filled using
  the median across stores.
- CompetitionDistance_was_missing preserves the
  original missingness information.
- CompetitionOpenSinceMonth and Year were retained
  without invented values.
- CompetitionOpenSince_was_missing identifies
  incomplete metadata.

## 5. Promo2 metadata

Missing Promo2SinceWeek, Promo2SinceYear and
PromoInterval values for non-participating stores
are structural and were not artificially imputed.

Promo2_metadata_missing flags participating stores
with incomplete programme metadata.

## 6. Target preservation

- Sales was not imputed.
- Zero-sales observations were retained.
- High-sales observations were retained.
- Closed-store nonzero-sales rows: {closed_nonzero}

No blanket outlier removal was performed.

## 7. Leakage safeguards

- Customers is retained for historical analysis but
  must not be used as a forecasting feature because
  it is unavailable in test.
- Id is retained for test prediction mapping but
  must not be used as a model feature.
- No future Sales values were used to impute test data.

## 8. Remaining work

Calendar, competition-age, Promo2 activity,
historical lag and rolling features will be handled
in the separate feature-engineering stage.

Time-aware validation and forecasting baselines
will be implemented after feature engineering.
"""

    report_path = REPORT_DIR / "preprocessing_findings.md"
    report_path.write_text(report.strip(), encoding="utf-8")

    print("\nPREPROCESSING COMPLETED")
    print("-----------------------")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Store rows: {len(store):,}")
    print(
        "Test Open values imputed:",
        len(missing_open_rows)
    )
    print(
        "CompetitionDistance missing before imputation:",
        int(store["CompetitionDistance_was_missing"].sum())
    )
    print(
        "Closed stores with nonzero training sales:",
        closed_nonzero
    )
    print(f"\nSaved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {store_path}")
    print(f"Saved: {report_path}")


def main():
    print("Loading raw data...")

    train, test, store = load_data()

    print("Validating input data...")
    train, test, store = validate_inputs(
        train, test, store
    )

    print("Handling missing Open values...")
    train, test = infer_missing_open(train, test)

    print("Cleaning store master...")
    store = clean_store_master(store)

    print("Merging store attributes...")
    train = merge_store_data(train, store)
    test = merge_store_data(test, store)

    print("Validating prepared datasets...")
    closed_nonzero = validate_prepared_data(
        train, test
    )

    print("Saving prepared datasets and report...")
    save_outputs(
        train, test, store, closed_nonzero
    )


if __name__ == "__main__":
    main()
