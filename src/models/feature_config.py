
CANDIDATE_FEATURES = [
    # Store and calendar
    "Store",
    "DayOfWeek",
    "StoreType",
    "Assortment",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "IsWeekend",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "Month_sin",
    "Month_cos",

    # Promotions and holidays
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "Promo2",
    "Promo2_Active",
    "Promo2_metadata_missing",

    # Competition
    "CompetitionDistance",
    "CompetitionDistance_was_missing",
    "CompetitionOpenSince_was_missing",
    "CompetitionOpenSince_invalid",
    "CompetitionAgeMonths",
    "Competition_NotYetOpen",

    # Historical sales
    "Sales_Lag_1",
    "Sales_Lag_7",
    "Sales_Lag_14",
    "Sales_Lag_28",
    "Sales_RollingMean_7",
    "Sales_RollingMean_14",
    "Sales_RollingMean_28",
    "Store_HistoricalMean",
    "Store_Weekday_HistoricalMean",
]

CATEGORICAL_FEATURES = [
    "Store",
    "DayOfWeek",
    "StoreType",
    "Assortment",
    "StateHoliday",
]


def get_feature_columns(train_data):
    missing = set(CANDIDATE_FEATURES) - set(train_data.columns)

    if missing:
        raise ValueError(
            f"Training data missing features: {sorted(missing)}"
        )

    # Remove constant features based on training data only.
    # The resulting list must remain unchanged at inference.
    features = [
        col for col in CANDIDATE_FEATURES
        if train_data[col].nunique(dropna=False) > 1
    ]

    print("Model feature count:", len(features))
    print("Categorical features:", [
        col for col in CATEGORICAL_FEATURES if col in features
    ])

    return features
