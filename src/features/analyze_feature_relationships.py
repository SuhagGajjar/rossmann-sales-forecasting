
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/train_features.pkl"
REPORT_DIR = ROOT / "reports"
CHART_DIR = REPORT_DIR / "feature_analysis_charts"

REPORT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

CORRELATION_THRESHOLD = 0.85
VIF_SAMPLE_SIZE = 30000
RANDOM_STATE = 42


# ---------------------------------------------------------
# 1. Load training data
# ---------------------------------------------------------

df = pd.read_pickle(DATA_PATH)

# Analyse open store-days only.
df = df.loc[df["Open"].eq(1)].copy()


# ---------------------------------------------------------
# 2. Candidate numerical predictors
# ---------------------------------------------------------

candidate_features = [
    "Open", "Promo", "SchoolHoliday", "Open_was_missing",
    "CompetitionDistance", "CompetitionDistance_was_missing",
    "CompetitionOpenSince_was_missing",
    "Promo2", "Promo2_metadata_missing",
    "Year", "Month", "Day", "WeekOfYear", "IsWeekend",
    "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos",
    "CompetitionAgeMonths", "CompetitionOpenSince_invalid",
    "Competition_NotYetOpen", "Promo2_Active",
    "Sales_Lag_1", "Sales_Lag_7", "Sales_Lag_14", "Sales_Lag_28",
    "Sales_RollingMean_7", "Sales_RollingMean_14",
    "Sales_RollingMean_28",
    "Store_HistoricalMean", "Store_Weekday_HistoricalMean",
]

features = [c for c in candidate_features if c in df.columns]

# Exclude constant predictors.
constant_features = [
    c for c in features if df[c].nunique(dropna=True) <= 1
]

features = [
    c for c in features if c not in constant_features
]


# ---------------------------------------------------------
# 3. Full-data correlation analysis
# ---------------------------------------------------------

X = df[features].replace([np.inf, -np.inf], np.nan)
y = df["Sales"]

# Median imputation is only for correlation diagnostics.
# It does not modify the original modelling dataset.
X = X.fillna(X.median())

correlation_data = X.copy()
correlation_data["Sales"] = y

corr = correlation_data.corr(method="pearson")

corr.to_csv(
    REPORT_DIR / "feature_correlation_matrix.csv"
)

# Feature-to-target correlations.
target_corr = (
    corr["Sales"]
    .drop("Sales")
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

target_corr.to_csv(
    REPORT_DIR / "feature_target_correlations.csv"
)


# ---------------------------------------------------------
# 4. Highly correlated feature pairs
# ---------------------------------------------------------

feature_corr = X.corr()

upper = feature_corr.where(
    np.triu(
        np.ones(feature_corr.shape),
        k=1
    ).astype(bool)
)

pairs = (
    upper.stack()
    .rename("correlation")
    .reset_index()
    .rename(
        columns={
            "level_0": "feature_1",
            "level_1": "feature_2"
        }
    )
)

high_corr = pairs[
    pairs["correlation"].abs() >= CORRELATION_THRESHOLD
].sort_values(
    "correlation",
    key=lambda s: s.abs(),
    ascending=False
)

high_corr.to_csv(
    REPORT_DIR / "high_correlation_pairs.csv",
    index=False
)


# ---------------------------------------------------------
# 5. VIF analysis on reproducible sample
# ---------------------------------------------------------

vif_sample = df.sample(
    n=min(VIF_SAMPLE_SIZE, len(df)),
    random_state=RANDOM_STATE
)

vif_X = (
    vif_sample[features]
    .replace([np.inf, -np.inf], np.nan)
)

# Use full-data medians for consistent diagnostic imputation.
vif_X = vif_X.fillna(X.median())

vif_results = []

for i, column in enumerate(vif_X.columns):
    try:
        vif = variance_inflation_factor(
            vif_X.values,
            i
        )
    except Exception:
        vif = np.nan

    vif_results.append({
        "feature": column,
        "VIF": vif
    })

vif_df = (
    pd.DataFrame(vif_results)
    .sort_values("VIF", ascending=False)
)

vif_df.to_csv(
    REPORT_DIR / "feature_vif.csv",
    index=False
)


# ---------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(18, 15))

image = ax.imshow(
    feature_corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(feature_corr.columns)))
ax.set_yticks(range(len(feature_corr.columns)))

ax.set_xticklabels(
    feature_corr.columns,
    rotation=90,
    fontsize=7
)

ax.set_yticklabels(
    feature_corr.columns,
    fontsize=7
)

fig.colorbar(
    image,
    ax=ax,
    label="Pearson correlation"
)

ax.set_title(
    "Training Feature Correlation Matrix (Full Dataset)"
)

fig.tight_layout()

fig.savefig(
    CHART_DIR / "feature_correlation_heatmap.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------

print("Open store-day observations available:", len(df))
print("Rows used for correlation:", len(X))
print("Rows used for VIF:", len(vif_X))
print("Features analysed:", len(features))
print("Constant features excluded:", constant_features)

print("\nTop feature-to-Sales correlations:")
print(target_corr.head(15).to_string())

print("\nHighly correlated feature pairs:")
if len(high_corr):
    print(high_corr.to_string(index=False))
else:
    print("No pairs above threshold.")

print("\nTop VIF values:")
print(vif_df.head(15).to_string(index=False))

print("\nReports saved under:", REPORT_DIR)
print("Heatmap saved under:", CHART_DIR)
