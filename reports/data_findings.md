# Rossmann — Data Findings and Investigation

This report consolidates observations from the raw training, test and store datasets, supported by the data audit and subsequent investigations.

Observations are distinguished from modelling implications and decisions. Descriptive associations are not interpreted as causal effects.

## 1. Dataset size and coverage

**Observations**

| Dataset | Rows | Stores | Date coverage |
|---|---:|---:|---|
| Train | 1,017,209 | 1,115 | 2013-01-01 to 2015-07-31 |
| Test | 41,088 | 856 | 2015-08-01 to 2015-09-17 |
| Store master | 1,115 | 1,115 | Store-level metadata |

- Training data contains 942 distinct calendar dates.
- No duplicate `Store-Date` records were found.
- All training and test stores exist in the store master.
- All test stores are represented in the training data.
- No overlap exists between training and test dates.
- All test `Id` values are unique.

**Modelling implications**

- Store and date provide the common keys for joining and validating datasets.
- Preserve the test `Id` mapping for submission generation, but exclude it from model features.
- Use time-aware validation that respects forecast origins and the complete forecast horizon.
- The apparent date coverage does not mean that every store has an observation for every date. The source-data coverage gap is detailed in Section 13.

## 2. Missing `Open` values in test

**Observations**

- There are 11 missing `Open` values in the test dataset.
- All 11 records belong to Store 622 and cover 5–17 September 2015.
- The affected dates are Monday through Saturday.
- Historical weekday opening behaviour was investigated for this store.

**Modelling and preprocessing decisions**

- Do not automatically assume missing `Open` means either open or closed.
- The 11 values were imputed using Store 622's historical weekday opening mode.
- An `Open_was_missing` indicator was retained to distinguish imputed values from observed values.
- These imputed values are estimates, not confirmed operational truth.
- In production, missing operational status should trigger a documented fallback or data-quality alert.

The model uses `Open` to determine whether a store is forecast to generate sales. Closed stores receive zero predicted sales; the model is applied only to open-store records.

## 3. Zero sales while a store is open

**Observations**

- 172,871 training records have zero sales.
- 172,817 records have `Open = 0`; all have zero sales.
- 54 records have `Open = 1` and `Sales = 0`.
  - 52 also have zero customers.
  - 2 have positive customer counts.

**Modelling implications**

- Retain all 54 open-store, zero-sales observations.
- Do not automatically classify them as errors or remove them.
- They may represent genuine zero demand, unusual operational circumstances or data-quality issues.
- Closed-store sales are consistently zero in the historical data, supporting the explicit zero-prediction rule for closed stores.
- Evaluate model performance both across all store-days and across open-store observations where useful.

## 4. Promo2 participation and missing metadata

**Observations**

| Promo2 status | Number of stores | Missing PromoInterval |
|---|---:|---:|
| Promo2 = 0 | 544 | 544 |
| Promo2 = 1 | 571 | 0 |

- Missing `PromoInterval` values are confined to stores that do not participate in Promo2.
- Promo2 is a store-level recurring promotion programme, whereas `Promo` represents a daily promotion indicator.

**Modelling and preprocessing decisions**

- Treat missing Promo2 metadata for nonparticipants as structural rather than automatically classifying it as a data defect.
- Preserve `Promo2` participation separately from date-level promotion activity.
- Derive `Promo2_Active` using participation, programme start information and recurring active months.
- Normalize the `Sept`/`Sep` month abbreviation inconsistency when interpreting recurring promotion months.
- Retain an indicator for incomplete Promo2 metadata among participating stores where relevant.
- Do not interpret Promo2 participation or observed sales differences as causal promotion effects.

## 5. Competition information

**Observations**

- `CompetitionDistance` is missing for 3 stores.
- `CompetitionOpenSinceMonth` and `CompetitionOpenSinceYear` are each missing for 354 stores.
- Store 815 contains a 1900 competition-opening-year placeholder, treated as invalid.
- After accounting for invalid and missing opening metadata, 191 stores have negative `CompetitionAgeMonths`.
- These negative values represent competitors whose opening dates are still in the future relative to the observation date.
- 180 stores transition from a future competitor opening to an opened competitor during the training period.
- 11 stores remain not-yet-open at the end of the training period.
- The minimum observed competition age is -31 months; a valid maximum of 644 months was retained.

**Modelling and preprocessing decisions**

- Preserve valid negative competition ages: they carry information about future competitor openings.
- Flag invalid placeholder years and avoid treating them as genuine opening dates.
- Set invalid competition ages to missing and retain an invalid-metadata indicator.
- Impute missing `CompetitionDistance` using the training median and retain a missingness indicator.
- Retain a missingness indicator for competition opening metadata.
- Do not assume that missing competition information necessarily means no competitor exists.

## 6. Sales distribution and extremes

**Observations**

| Statistic | Sales |
|---|---:|
| Mean | 5,773.82 |
| Standard deviation | 3,849.93 |
| Median | 5,744 |
| 95th percentile | 12,137 |
| 99th percentile | 17,160 |
| 99.9th percentile | 23,980 |
| Maximum | 41,551 |

- Sales are non-negative.
- No negative sales values were found.
- The distribution contains high-demand observations and a substantial spread between typical and extreme sales.

**Modelling implications**

- Investigate high-sales observations by store, promotion, holiday, weekday and customer count.
- Preserve genuine high-demand events rather than removing observations solely because they are statistically extreme.
- Evaluate metrics that reflect different error characteristics, including MAE, RMSE, WMAPE and RMSLE.
- Consider business-weighted or asymmetric loss functions when translating forecast errors into operational consequences.

## 7. Promotion association

**Observations**

Comparing daily promotion and non-promotion observations:

- Average customers are approximately 21.2% higher on `Promo = 1` days.
- Average sales are approximately 38.8% higher on `Promo = 1` days.
- Weighted sales per customer are approximately 14.5% higher on promotion days.

These are descriptive comparisons.

**Modelling implications**

- Promotion indicators are potentially useful forecasting features.
- These differences are not causal estimates of promotion uplift.
- Store composition, weekday, seasonality, holidays and promotion-selection effects may influence the observed differences.
- Do not use these raw percentage differences as direct estimates of incremental promotion revenue.

## 8. Weekday and calendar patterns

**Observations**

- DayOfWeek uses 1 for Monday through 7 for Sunday.
- The original EDA found the highest average sales on weekday 7 (Sunday), approximately 8,225, and the lowest on weekday 6 (Saturday), approximately 5,875.
- Only 33 stores were observed open on at least one Sunday, so aggregate Sunday behaviour is not representative of all stores.
- Average sales also vary by month.
- December has the highest average sales (approximately 8,609), while September has the lowest (approximately 6,546).

**Modelling implications**

- Weekday, month and other calendar features are useful candidates.
- Store-level operating patterns must be considered; a single aggregate weekday pattern should not be assumed to apply uniformly.
- Monthly patterns should be interpreted cautiously because the dataset covers fewer than three full years and store composition can influence aggregates.

## 9. Customers and Sales relationship

**Observations**

- Customers is available in the training dataset but absent from the test dataset.
- Pearson correlation between Customers and Sales on open-store observations is approximately 0.824 (844,392 observations).
- Store-level average Customers and Sales also have a strong positive correlation (approximately 0.830 across 1,115 stores).

**Modelling implications**

- Customers is useful for exploratory analysis and retrospective diagnostics.
- Do not use actual future Customers as a production forecasting feature because it is unavailable at inference time and is observed alongside the sales outcome.
- Any feature used in validation must be available at the forecast origin or be independently forecast using information available at that time.

## 10. Forecast horizon and feature availability

**Observations**

- The assignment specifies a six-week, 42-day forecasting horizon.
- The Kaggle test dataset contains 48 distinct dates, from 1 August to 17 September 2015.
- Customers is available in training but not in test.
- Sales is the training target and is unavailable in test.

**Modelling implications**

- Use a 42-day horizon for the primary assignment validation and model comparison.
- Keep the additional six test dates distinct from the assignment's primary 42-day evaluation horizon.
- Maintain a strict separation between forecast-origin history and future validation observations.
- Simulate the full forecast horizon recursively rather than using actual future sales to construct lag or rolling features.
- Exclude variables unavailable at inference time, including actual future Customers.

## 11. Store and date integrity

**Observations**

- Store master contains 1,115 unique stores.
- No duplicate store identifiers were found.
- All training and test stores have matching store metadata.
- Test `Id` values are unique.
- There are no duplicate `Store-Date` combinations in the training data.

**Modelling implications**

- Validate joins using Store and Date, with one-to-one validation where appropriate.
- Retain test identifiers separately for final prediction output.
- Check complete Store-Date coverage and alignment after every forecast and evaluation run.
- Do not rely on row position alone to align actual and predicted values.

## 12. Initial modelling principles

The investigation supports the following principles:

1. Start with a global pooled forecasting model and evaluate performance across stores and relevant business segments.
2. Do not use actual future Customers as a model feature.
3. Do not blindly remove zero-sales or extreme-sales observations.
4. Interpret missing values according to their underlying meaning, distinguishing structural missingness from unknown information.
5. Use a strict time-based forecast origin and recursively generate predictions across the complete forecast horizon.
6. Ensure that future actual sales are used only for evaluation, never to update forecast history.
7. Evaluate both statistical accuracy and operational implications of underforecasting and overforecasting.
8. Preserve Store-Date alignment and test identifier mapping throughout the pipeline.
9. Treat the 42-day assignment horizon separately from the 48-day Kaggle test period.
10. Document source-data limitations and their implications for time-series modelling.

## 13. Source-data date-coverage gap

**Investigation**

A separate daily Store-Date coverage audit was conducted against the original `data/raw/train.csv`, not only the preprocessed dataset. This confirmed that the missing observations originate in the supplied source data.

The raw training data contains:

- 1,017,209 observed rows.
- 1,115 unique stores.
- 942 distinct calendar dates, from 1 January 2013 to 31 July 2015.
- 1,114 stores on 1 January 2013, indicating one missing store-date observation.
- Only 935 stores per day from 1 July 2014 through 31 December 2014, a continuous period of 184 days.
- The same 180 stores are absent throughout that 184-day period.
- All 1,115 stores are represented on the other dates.

For a complete panel of 1,115 stores over 942 dates, the expected number of store-date observations is:

`1,115 × 942 = 1,050,330`

The difference between expected and observed rows is:

`1,050,330 − 1,017,209 = 33,121`

This consists of:

- `180 × 184 = 33,120` observations missing during July–December 2014.
- One additional missing store-date on 1 January 2013.

**Modelling implications**

- The missing rows are absent from the original source data; they were not introduced by preprocessing.
- Missing Store-Date records must not automatically be interpreted as store closures or zero sales.
- Sales and Open status are unknown for these absent records.
- Do not reconstruct these rows by assigning zero sales or an assumed opening status without independent evidence.
- The gap affects historical daily time-series continuity for the 180 affected stores.

**Impact on lag and rolling features**

The historical feature pipeline uses store-grouped, row-based operations such as `groupby("Store").shift(7)` and rolling calculations over prior observations.

Because 180 stores have a 184-day gap, row-based lags immediately after the gap do not necessarily represent exact calendar-day lags. For example, the seventh previous observed row may be substantially more than seven calendar days earlier.

The same concern applies to rolling windows and other historical features based on consecutive observations.

However, the gap ends on 31 December 2014. By the first forecast origin used in the assignment's model experiments (8 May 2015) and the later forecast origin (19 June 2015), more than 28 consecutive calendar days of observations are available after the gap. Therefore, the recent 1-, 7-, 14- and 28-observation lag features at these forecast origins are based on post-gap observations.

Historical training rows immediately after the gap still require caution when interpreting their row-based lag and rolling features.

**Impact on ETS and statistical time-series models**

A per-store daily ETS model requires a suitable regular daily series. The missing source observations prevent a complete regular daily history from being constructed for all affected stores without additional assumptions.

An ETS implementation was attempted, but fitting failed on a store with non-consecutive daily observations (Store 13).

We therefore do not claim a complete ETS benchmark across all stores. The limitation is documented rather than hidden through artificial zero-filling or unsupported reconstruction.

**Recommended treatment**

- Preserve the raw source data.
- Retain the documented missing-coverage findings.
- Avoid fabricating zero-sales or closed-store observations.
- Validate calendar alignment when creating historical features.
- Consider calendar-aware reindexing and explicit missing-history indicators in a future production implementation, provided the business can supply or approve a defensible treatment for absent observations.
- Disclose the source-data limitation when interpreting historical features and model comparisons.

## 14. Overall investigation conclusion

The data is sufficiently structured to support global store-level sales forecasting, but several characteristics require explicit treatment:

- Missing operational status for 11 test observations.
- Structural missingness in Promo2 metadata.
- Missing and invalid competition metadata.
- Open-store zero-sales observations.
- A substantial original-source Store-Date coverage gap affecting 180 stores for 184 days.
- A 42-day assignment forecast horizon versus a 48-day Kaggle test period.
- Customers being unavailable at inference time.

The modelling pipeline should preserve valid observations, avoid unsupported assumptions about missing records, prevent future-information leakage and evaluate the complete recursive forecast horizon.

These findings should be considered alongside the preprocessing, feature-engineering, model-evaluation and business-impact reports.