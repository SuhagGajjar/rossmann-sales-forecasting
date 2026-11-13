# Preprocessing and Feature Engineering Methodology

> **Purpose.** This document is the consolidated technical record of the Rossmann Store Sales preprocessing and feature-engineering decisions. It explains what is transformed, why each decision was made, how temporal leakage is prevented, and which checks support the resulting datasets and features.
>
> **Implementation sources:** `src/preprocessing/data_validation.py`, `src/preprocessing/prepare_data.py`, `src/features/build_features.py`, `src/features/recursive_forecasting.py`, `src/features/validate_recursive_features.py`, and `src/models/feature_config.py`.

---

## 1. Scope and design principles

The objective is to forecast daily sales for each Rossmann store using information available at the forecast origin. The supplied competition data contains:

- **Training data:** 1,017,209 store-date observations, from 1 January 2013 to 31 July 2015, with observed `Sales` and `Customers`.
- **Test data:** 48 days, from 1 August to 17 September 2015, with no observed `Sales`.
- **Store master:** store-level attributes such as store type, assortment, competition, and Promo2 programme metadata.
- **Forecasting requirement:** a six-week (42-day) forecast horizon. The supplied test file is 48 days long, so the assignment evaluation uses explicit 42-day chronological windows rather than silently treating the entire test file as a six-week horizon.

The pipeline follows these principles:

1. Preserve the original target and source information; do not remove unusual observations without evidence of data error.
2. Distinguish true structural absence from unknown or incomplete metadata.
3. Make training and inference features consistent with information actually available at prediction time.
4. Generate historical sales features using past information only.
5. Evaluate chronologically and recursively: after the forecast origin, use model predictions—not future actual sales—to update history.
6. Keep audit fields and identifiers available for traceability while excluding them from model predictors where appropriate.

## 2. Data loading and integrity checks

### 2.1 Input contracts

The preparation pipeline expects the following key fields:

| Dataset | Required fields (summary) | Role |
|---|---|---|
| Train | `Store`, `DayOfWeek`, `Date`, `Sales`, `Customers`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` | Historical observations and target |
| Test | `Id`, `Store`, `DayOfWeek`, `Date`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` | Future prediction rows |
| Store | `Store`, `StoreType`, `Assortment`, competition fields, Promo2 fields | Store-level reference data |

Dates are parsed as datetimes. `StateHoliday` is loaded/standardised as a string so numeric-looking values and letter categories are not accidentally mixed by type inference.

### 2.2 Structural and relational validation

The implementation checks:

- Required columns exist in each input.
- Store master has one row per `Store`.
- Train and test contain no duplicate `Store`–`Date` combinations.
- Every train/test store ID exists in the store master.
- Test `Id` values are unique.
- Dates can be parsed without silent coercion.
- Store-master joins use `validate="many_to_one"`, an indicator column, and a row-count check. An unmatched row or unexpected row multiplication raises an error.

The separate audit workflow also reports dataset dimensions, missingness by column, duplicate rows, store coverage, date overlap, sales distribution, zero-sales patterns, and known data risks. These checks are intended to make data problems visible before feature construction.

### 2.3 Target and observation treatment

The observed target is `Sales`. The audit found no negative sales values. Zero-sales observations are retained, including zero sales on open-store days; they are not automatically treated as missing values. Closed-store rows are retained in the prepared data and are handled explicitly in forecasting (closed stores receive zero forecasts).

High sales values are also retained. No blanket outlier clipping or removal is applied: genuine demand peaks are commercially meaningful and should not be erased simply because they are large. Any correction would require evidence that a value is erroneous.

`Customers` is retained for historical analysis and exploratory analysis, but is excluded from the production feature set because it is not supplied in the competition test data and would not be known when a future sales forecast is made.

## 3. Preprocessing decisions

### 3.1 Missing `Open` values

The 11 missing `Open` values are all in **Store 622**, for 5–17 September 2015. Training `Open` values are not imputed; an unexpected missing training value causes the preparation step to stop.

For the 11 test rows, the pipeline infers opening status from the historical mode of `Open` for the same `Store` and `DayOfWeek`, calculated using training data only. Store 622's historical weekday pattern supports Monday–Saturday opening and Sunday closure for these dates.

Two traceability fields are retained:

- `Open_was_missing`: 1 if the original value was missing, otherwise 0.
- `Open_imputation_method`: records `historical_store_weekday_mode` for imputed rows and `not_missing` otherwise.

The imputation is a documented inference, not confirmed ground truth. `Open_imputation_method` is audit metadata and must not be used as a model predictor. The missingness indicator can be retained for audit and considered as a feature only if the feature configuration explicitly includes it; it is not part of the current candidate feature list.

### 3.2 Competition distance

Three stores have missing `CompetitionDistance`. The missing values are filled with the median competition distance across the store master. The original missingness is preserved in `CompetitionDistance_was_missing`.

This provides a usable numeric input without pretending that the imputed distance was observed. The indicator allows a model to distinguish an imputed value from a measured one.

### 3.3 Competition opening metadata and age

`CompetitionOpenSinceMonth` and `CompetitionOpenSinceYear` are retained as source metadata. Incomplete metadata is not filled with invented dates. `CompetitionOpenSince_was_missing` flags stores for which either component is missing.

A source year of **1900** is specifically identified for Store 815 as implausible. It is flagged through `CompetitionOpenSince_invalid` and excluded from age calculation. It is not silently interpreted as a genuine opening date.

For valid records, competition age is calculated for each observation date:

\[
\text{CompetitionAgeMonths}
= 12(\text{ObservationYear}-\text{CompetitionOpenYear})
+ (\text{ObservationMonth}-\text{CompetitionOpenMonth})
\]

The value is left missing when the source opening year/month is incomplete or invalid. Negative values are deliberately retained: they indicate a competitor scheduled to open in the future relative to the observation date, rather than a calculation error. The accompanying `Competition_NotYetOpen` flag is 1 when the calculated age is negative. The investigation found negative ages in 191 stores, with 11 stores still not yet opened by the end of the training period. Clipping negative values to zero would discard potentially useful future-competition information.

### 3.4 Promo2 metadata

Promo2 participation and its timing/interval fields have a different missingness meaning from ordinary numeric missingness. For stores that do not participate in Promo2, missing `Promo2SinceWeek`, `Promo2SinceYear`, and `PromoInterval` are structural; they are not filled with invented dates or intervals.

For participating stores, `Promo2_metadata_missing` flags incomplete programme metadata. `Promo2` is represented as an integer indicator, with missing values defaulted to 0 in the feature-building stage.

### 3.5 Holiday and categorical values

The dataset's `StateHoliday` semantics are taken from the competition documentation:

| Value | Meaning |
|---|---|
| `0` | No state holiday |
| `a` | Public holiday |
| `b` | Easter holiday |
| `c` | Christmas |

The value is kept categorical; it is not treated as an ordinal scale. For models requiring one-hot encoding, the agreed representation uses three indicators (`a`, `b`, `c`) with `0` as the reference category (n−1 encoding). The final LightGBM configuration treats `StateHoliday` as categorical.

`SchoolHoliday` is filled with 0 if missing and cast to an integer indicator. This is the current implementation's explicit treatment; any unexpected missingness should remain visible in the audit output.

## 4. Base feature engineering

Base features are derived from each row's date, store attributes, holiday information, and promotion metadata. They do not require future target values and can therefore be constructed consistently for training, validation, and test rows.

### 4.1 Calendar features

| Feature | Definition | Rationale |
|---|---|---|
| `Year` | Calendar year from `Date` | Captures broad temporal change |
| `Month` | Calendar month (1–12) | Captures monthly/seasonal patterns |
| `Day` | Day of month | Captures within-month effects |
| `WeekOfYear` | ISO week number | Captures weekly/annual seasonality |
| `DayOfWeek` | Supplied weekday (1–7) | Strong store trading pattern |
| `IsWeekend` | 1 when weekday is 6 or 7 | Distinguishes weekend trading |
| `DayOfWeek_sin`, `DayOfWeek_cos` | Sine/cosine encoding of weekday | Represents cyclic adjacency (Sunday and Monday are near each other) |
| `Month_sin`, `Month_cos` | Sine/cosine encoding of month | Represents cyclic annual seasonality |

The cyclical encodings are:

\[
\text{sin}(x)=\sin\left(\frac{2\pi x}{P}\right),\qquad
\text{cos}(x)=\cos\left(\frac{2\pi x}{P}\right)
\]

where (x) is the weekday or month index and (P) is its cycle length (7 for weekday, 12 for month). Using both sine and cosine avoids treating the beginning and end of a cycle as maximally distant.

### 4.2 Store attributes

The store master is merged onto each daily row using `Store`. The candidate model features include:

- `Store`: store identity, allowing the model to learn persistent store-level differences.
- `StoreType`: store format/type.
- `Assortment`: assortment category.

These are categorical identifiers/categories, not continuous measurements. `Store`, `DayOfWeek`, `StoreType`, `Assortment`, and `StateHoliday` are declared categorical in `src/models/feature_config.py`.

### 4.3 Promotion features

- `Promo`: daily promotion indicator from the source data.
- `Promo2`: store's participation in the continuing promotion programme.
- `Promo2_Active`: whether the store's Promo2 programme is active on the observation date.
- `Promo2_metadata_missing`: identifies incomplete timing metadata for a participating store.

To calculate `Promo2_Active`, the pipeline reconstructs the programme start date from `Promo2SinceYear` and `Promo2SinceWeek` using ISO calendar week semantics (the Monday of that ISO week). It parses `PromoInterval` into month numbers and sets activity to 1 only when all of the following hold:

1. The store participates in Promo2 (`Promo2 == 1`).
2. A valid programme start date is available.
3. The observation date is on or after the programme start date.
4. The observation month is included in the programme's listed active months.

Otherwise, activity is 0. This prevents a store from being marked active in a listed month before its programme has started. The intermediate `Promo2StartDate` is dropped after the flag is derived; source metadata remains available for audit.

### 4.4 Competition features

The candidate set includes:

- `CompetitionDistance` and `CompetitionDistance_was_missing`.
- `CompetitionOpenSince_was_missing` and `CompetitionOpenSince_invalid`.
- `CompetitionAgeMonths`.
- `Competition_NotYetOpen`.

This separates distance, elapsed time since opening, future opening status, and source-data quality. The model is not forced to interpret an invalid or missing opening date as a real numeric age.

## 5. Historical sales features

Historical sales features are often among the strongest predictors in store-level demand forecasting, but they are also the main source of temporal leakage risk. Every feature must represent information that would genuinely have been known at the time the forecast was issued.

### 5.1 Lag features

For each store, rows are ordered by date and the following prior observations are used:

| Feature | Definition |
|---|---|
| `Sales_Lag_1` | Sales on the previous store observation/day |
| `Sales_Lag_7` | Sales seven observations/days earlier (weekly seasonal lag) |
| `Sales_Lag_14` | Sales fourteen observations/days earlier |
| `Sales_Lag_28` | Sales twenty-eight observations/days earlier |

The implementation uses `groupby("Store")["Sales"].shift(lag)`. It relies on the daily store panel being correctly sorted and complete for the relevant store-date sequence. Missing lag values at the beginning of a store's history are left missing rather than filled with future information.

### 5.2 Rolling means

For each store, the model receives:

- `Sales_RollingMean_7`
- `Sales_RollingMean_14`
- `Sales_RollingMean_28`

For a window of length (w), the feature for date (t) is:

\[
\text{RollingMean}_{w,t}
=\frac{1}{n}\sum_{i=1}^{n} Sales_{t-i},
\qquad n=\min(w,\text{available prior observations})
\]

The implementation explicitly applies `shift(1)` before the rolling operation and uses `min_periods=1`. Therefore, the current date's target is excluded, and early-history rows can use the available prior observations rather than requiring a full window. The feature is a mean of the preceding observations, not a centred rolling statistic.

### 5.3 Expanding historical averages

Two cumulative historical statistics are created:

- `Store_HistoricalMean`: mean of all prior sales observations for that store.
- `Store_Weekday_HistoricalMean`: mean of all prior sales observations for that store on the same weekday.

For both, the implementation uses `shift(1).expanding(min_periods=1).mean()`. The shift ensures the current target is not included. The weekday-specific mean is calculated within each `Store`–`DayOfWeek` group.

These features provide a longer-run store baseline and a weekday-adjusted store baseline, complementing the shorter lag and rolling windows.

## 6. Leakage prevention and forecast-origin protocol

### 6.1 What is leakage in this task?

Leakage occurs if a feature uses information that would not have been available when the forecast was made. In particular, using actual validation/test sales to create later validation/test lags or rolling means would make the offline evaluation unrealistically easy: a real forecasting system does not know those future actuals.

The following fields are therefore excluded from model predictors:

| Field | Reason for exclusion |
|---|---|
| `Sales` | It is the target, never a same-row predictor |
| `Customers` | Not available in the competition test set / future forecast origin |
| `Id` | Submission-row identifier only; no demand meaning |
| `Open_imputation_method` | Audit/process metadata, not a business predictor |

The target `Sales` remains in the training dataset for fitting and for constructing historical features from permitted past observations. Its presence in a dataframe does not make it an eligible same-row feature.

`DaysAhead` was not included in the feature set: a horizon-index feature should only be introduced if training examples and validation are constructed with horizon-matched semantics. Adding it without that design would create a train/inference mismatch.

### 6.2 Chronological splits

The project uses two chronological, recursive 42-day validation windows:

| Window | Historical training cutoff | Validation dates | Purpose |
|---|---|---|---|
| Window 1 | 8 May 2015 | 9 May–19 June 2015 | Candidate comparison and tuning |
| Window 2 | 19 June 2015 | 20 June–31 July 2015 | Later holdout with tuned parameters frozen |

The feature-building script's primary split is Window 2: training history ends on 19 June 2015; validation spans 20 June–31 July 2015. This gives 42 calendar days and 46,830 validation rows (1,115 stores × 42 dates). Window 1 is handled by its corresponding first-window modelling workflow.

The supplied test set runs from 1 August to 17 September 2015 (48 days). The assignment's six-week evaluation horizon is 42 days; the extra six supplied test days are a distinct property of the competition file and should not be confused with the validation-window length.

### 6.3 Recursive validation and test feature generation

At a forecast origin:

1. Initialise each store's historical state using actual sales strictly before the origin.
2. Build the current forecast date's lag, rolling, and historical-mean features from that state.
3. Predict sales for all relevant stores for that date.
4. Apply the closed-store rule (zero forecast where `Open == 0`) and the forecast post-processing used by the modelling workflow.
5. Append the day’s predictions to each store's history.
6. Move to the next date and repeat.

The `RecursiveFeatureBuilder` stores per-store sales history, weekday sums/counts, and total sums/counts. Its `build_day_features` method does not read current-day `Sales`. Its `update_history` method accepts the supplied forecast values, checks their length and finiteness, clips negative forecasts to zero, and appends them to history. In validation/test forecasting, the values appended after the origin must be predictions, not actual future sales.

This is the key distinction between:
- **Training feature construction:** shifted historical features can be computed from the chronologically ordered training target series.
- **Validation/test feature construction:** the history must be advanced recursively with model outputs after the forecast origin.

### 6.4 Other leakage controls

- Store-master attributes are joined by store ID; no target-derived aggregate is computed across future validation/test rows.
- Competition age and Promo2 activity use date and store metadata, not future sales.
- Feature selection removes constant columns using training data only. The resulting feature list is held fixed at inference.
- Validation-period sales are used only as evaluation truth after predictions have been generated, not to populate the recursive feature history.

## 7. Candidate model feature configuration

`src/models/feature_config.py` defines 33 candidate features across four groups:

| Group | Features |
|---|---|
| Store and calendar | `Store`, `DayOfWeek`, `StoreType`, `Assortment`, `Year`, `Month`, `Day`, `WeekOfYear`, `IsWeekend`, `DayOfWeek_sin`, `DayOfWeek_cos`, `Month_sin`, `Month_cos` |
| Promotions and holidays | `Promo`, `StateHoliday`, `SchoolHoliday`, `Promo2`, `Promo2_Active`, `Promo2_metadata_missing` |
| Competition | `CompetitionDistance`, `CompetitionDistance_was_missing`, `CompetitionOpenSince_was_missing`, `CompetitionOpenSince_invalid`, `CompetitionAgeMonths`, `Competition_NotYetOpen` |
| Historical sales | `Sales_Lag_1`, `Sales_Lag_7`, `Sales_Lag_14`, `Sales_Lag_28`, `Sales_RollingMean_7`, `Sales_RollingMean_14`, `Sales_RollingMean_28`, `Store_HistoricalMean`, `Store_Weekday_HistoricalMean` |

The code removes features that are constant in the training data (`nunique(dropna=False) <= 1`). This decision is based on training data only; the selected feature list must remain unchanged for validation and inference. The final modelling run used 33 features after the training-only constant-feature check.

The declared categorical fields are `Store`, `DayOfWeek`, `StoreType`, `Assortment`, and `StateHoliday`. Model-specific encoding is applied in the training pipeline; the feature configuration itself does not turn these categories into ordinal numeric measurements.

## 8. Validation checks and evidence

The checks below describe what the implementation verifies. A check marked **PASS by construction/assertion** means the code contains a guard that raises an error if the condition fails; it is not a claim that every possible runtime environment has been independently rerun while preparing this document.

| Area | Validation / expected result | Observed or recorded evidence | Status |
|---|---|---|---|
| Required schema | Required train/test/store columns exist | Explicit required-column assertions in `validate_inputs` | PASS by assertion |
| Store master key | One row per Store | Duplicate Store IDs cause an error | PASS by assertion |
| Store-date grain | No duplicate Store-Date in train or test | Duplicate keys cause an error | PASS by assertion |
| Store coverage | Every train/test store maps to store master | Left join indicator; unmatched rows cause an error | PASS by assertion |
| Join cardinality | Merge does not multiply or drop rows | Many-to-one validation and row-count equality check | PASS by assertion |
| Test identifier | `Id` is unique | Uniqueness assertion | PASS by assertion |
| Date parsing | All dates parse successfully | `errors="raise"` / datetime conversion | PASS by assertion |
| Target integrity | No missing or negative training Sales | Preparation validation raises for either condition; audit found no negative Sales | PASS for recorded audit |
| Open domain | Open is binary after imputation | Values checked against {0,1}; 11 missing test values handled | PASS by assertion |
| Missing Open traceability | 11 affected test rows identified and method retained | All 11 are Store 622, 5–17 Sep 2015 | DOCUMENTED INFERENCE |
| Competition distance | Three missing store distances imputed; missingness retained | Median imputation and indicator in preparation code | IMPLEMENTED |
| Competition opening metadata | Incomplete/invalid values not converted into invented ages | Missing and invalid indicators; 1900 for Store 815 flagged | IMPLEMENTED |
| Promo2 structural missingness | Non-participant missing metadata is not filled with invented dates | Structural absence retained; participating incomplete metadata flagged | IMPLEMENTED |
| Split boundary | Window 2 training ends 19 Jun; validation is 20 Jun–31 Jul | Date assertions in `build_features.py`; 46,830 validation rows | PASS by assertion / recorded output |
| Historical feature shift | Current-day Sales excluded from lag/rolling/expanding features | `shift(1)` in all training historical statistics | PASS by implementation |
| Recursive feature parity | Recursive builder agrees with independent pandas reference for first validation day | `validate_recursive_features.py` compares all generated columns with `np.allclose` | Executable check; run to confirm |
| Inference feature consistency | Same candidate feature names and training-only constant filter used | Central `CANDIDATE_FEATURES` and `get_feature_columns` | IMPLEMENTED |
| Future-target isolation | Validation/test future actual Sales never appended to recursive history | Recursive builder API appends caller-supplied predictions; forecasting workflow must pass predictions | PROTOCOL REQUIREMENT |

### 8.1 Recursive feature parity test

`src/features/validate_recursive_features.py` provides an independent implementation check for the first validation date (20 June 2015):

- It restricts history to dates on or before 19 June 2015.
- It creates the first validation date's features with `RecursiveFeatureBuilder`.
- It independently calculates expected lags, shifted rolling means, store expanding mean, and store-weekday expanding mean with pandas.
- It compares actual and reference values with `np.allclose` (`rtol=1e-9`, `atol=1e-8`, `equal_nan=True`).
- Any mismatch raises an assertion error.

This test checks implementation parity for the initial forecast date. The recursive forecasting protocol remains responsible for ensuring that later dates are updated with predictions only.

## 9. Reproducibility and execution order

Run commands from the repository root. Raw Rossmann files are not committed and must be placed in `data/raw/` as `train.csv`, `test.csv`, and `store.csv`.

Recommended sequence:

1. **Audit inputs:** run `src/preprocessing/data_validation.py` and the associated data investigation workflow.
2. **Prepare datasets:** run `src/preprocessing/prepare_data.py`. This validates inputs, imputes the missing test `Open` values, cleans store metadata, joins store attributes, and writes preprocessed pickle files plus `reports/preprocessing_findings.md`.
3. **Build base and training features:** run `src/features/build_features.py`. It creates `train_features.pkl`, `validation_base.pkl`, and `test_base.pkl`.
4. **Validate recursive features:** run `src/features/validate_recursive_features.py` from the execution context expected by its import (`src/features`), after the required processed datasets have been created.
5. **Train/evaluate:** use the model scripts under `src/models/`, keeping the chronological split and recursive forecast protocol intact.

Key generated files:

| Output | Contents |
|---|---|
| `data/processed/train_preprocessed.pkl` | Training rows after preparation and store metadata join |
| `data/processed/test_preprocessed.pkl` | Test rows after preparation and store metadata join |
| `data/processed/store_preprocessed.pkl` | Cleaned store master |
| `reports/preprocessing_findings.md` | Preparation summary and missing-value decisions |
| `data/processed/train_features.pkl` | Window 2 training rows with base and shifted historical features |
| `data/processed/validation_base.pkl` | Window 2 validation rows with base features; recursive history is added during forecasting |
| `data/processed/test_base.pkl` | Test rows with base features; recursive history is added during forecasting |

## 10. Limitations and interpretation

- The Store 622 `Open` inference is based on historical weekday behaviour; it is not verified future ground truth.
- Median competition-distance imputation reduces missingness but does not recover the true distance for the three affected stores. The indicator preserves this distinction.
- Competition opening metadata may be missing or invalid; missing age should not be interpreted as zero age.
- Promo2 activity depends on the correctness and completeness of the source programme start week/year and month interval.
- The historical lags assume the per-store daily sequence is correctly sorted and aligned. Data-grain and date checks are therefore important prerequisites.
- A recursive 42-day forecast can accumulate error because each prediction influences subsequent historical features. This is why validation must be recursive and why one-step-ahead evaluation would be an optimistic substitute.
- The 48-day supplied competition test period and the 42-day assignment validation horizon are different. Reports must state which horizon is being evaluated.
- Feature importance or SHAP attribution describes model associations, not causal effects of promotions, competition, or store attributes.

## 11. Final feature-engineering contract

A valid model input row must satisfy all of the following:

1. It represents one store on one forecast date.
2. Static/calendar/promotion/competition features are derived from information available for that date.
3. Historical sales features use only information available before that date.
4. For dates after a forecast origin, historical state is advanced using model predictions, never actual future sales.
5. `Sales`, `Customers`, `Id`, and audit-only fields are not included as predictors.
6. The model uses the same training-selected feature columns and categorical semantics at inference.
7. Closed stores are handled explicitly by the forecasting workflow rather than by treating their sales as ordinary open-store demand.

These rules define the boundary between the data-preparation layer, deterministic feature construction, and the recursive forecasting process. Any future feature addition should be reviewed against this contract and tested at the forecast origin before it is accepted.
