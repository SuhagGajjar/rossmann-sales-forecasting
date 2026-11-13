# Rossmann Sales Forecasting — Exploratory Data Analysis

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
| Training records | 1,017,209 |
| Stores | 1,115 |
| Historical start | 2013-01-01 |
| Historical end | 2015-07-31 |
| Open store-day observations | 844,392 |
| Closed store-day observations | 172,817 |
| Open-store zero-sales observations | 54 |
| Negative sales observations | 0 |

The calendar weekday field was checked against the actual date.
No weekday mismatches were found.

## 2. Daily average sales trend

- Highest daily average: 14,263.12
  on 2015-04-03.
- Lowest daily average: 4,067.14
  on 2013-12-31.

The 28-day rolling mean helps distinguish medium-term movement
from individual-day fluctuations.

**Modelling implication:** Retain calendar features and evaluate
temporal patterns through future-period validation. Aggregate
movement should not be interpreted as a change in every store.

## 3. Sales by weekday

| Day | Observations | Stores | Mean_Sales | Median_Sales | Mean_Customers |
| --- | --- | --- | --- | --- | --- |
| Monday | 137560 | 1115 | 8,216.07 | 7,539.00 | 855.44 |
| Tuesday | 143961 | 1115 | 7,088.11 | 6,502.00 | 769.99 |
| Wednesday | 141936 | 1115 | 6,728.12 | 6,210.00 | 740.60 |
| Thursday | 134644 | 1115 | 6,767.31 | 6,246.00 | 755.57 |
| Friday | 138640 | 1115 | 7,072.68 | 6,580.00 | 781.77 |
| Saturday | 144058 | 1115 | 5,874.84 | 5,425.00 | 660.18 |
| Sunday | 3593 | 33 | 8,224.72 | 6,876.00 | 1,441.53 |

- Highest mean: 8,224.72
  on Sunday.
- Lowest mean: 5,874.84
  on Saturday.
- Stores with at least one open Sunday: 33.

Sunday observations have a much narrower store population.
The Sunday aggregate is therefore not directly comparable with
weekday averages as though the same stores contributed.

**Modelling implication:** Include weekday as a candidate feature,
but evaluate store-level and weekday-level performance.

## 4. Monthly seasonality

| Month_Name | Observations | Mean_Sales | Median_Sales |
| --- | --- | --- | --- |
| January | 86343 | 6,563.69 | 6,062.00 |
| February | 80243 | 6,589.17 | 6,078.00 |
| March | 85980 | 6,976.41 | 6,385.00 |
| April | 81731 | 7,046.23 | 6,431.00 |
| May | 80103 | 7,106.45 | 6,592.00 |
| June | 82576 | 7,000.98 | 6,365.00 |
| July | 85587 | 6,952.68 | 6,407.00 |
| August | 54413 | 6,648.98 | 6,124.00 |
| September | 52330 | 6,546.34 | 5,999.00 |
| October | 53292 | 6,602.84 | 6,105.00 |
| November | 51401 | 7,188.55 | 6,691.00 |
| December | 50393 | 8,608.96 | 7,858.00 |

- Highest mean: 8,608.96
  in December.
- Lowest mean: 6,546.34
  in September.

The historical period is approximately two and a half years.
Long-term seasonal conclusions should therefore remain cautious.

**Modelling implication:** Test month, week-of-year and other
calendar features through time-based validation.

## 5. Sales distribution and extreme values

| Quantile | Sales |
| --- | --- |
| 0.00 | 0.00 |
| 0.01 | 2,185.00 |
| 0.05 | 3,173.00 |
| 0.25 | 4,859.00 |
| 0.50 | 6,369.00 |
| 0.75 | 8,360.00 |
| 0.90 | 10,771.00 |
| 0.95 | 12,668.00 |
| 0.99 | 17,789.00 |
| 1.00 | 24,668.87 |
| 1.00 | 41,551.00 |

- Mean sales on open days: 6,955.51.
- Median sales on open days: 6,369.00.
- Minimum: 0.
- Maximum: 41,551.

The upper tail may include genuine high-demand, seasonal and
promotional observations.

**Modelling implication:** Do not remove extreme values solely
because they are high. Evaluate MAE, RMSE, WMAPE and RMSLE.

## 6. Daily Promo analysis

| Promotion | Observations | Stores | Mean_Sales | Median_Sales | Mean_Customers |
| --- | --- | --- | --- | --- | --- |
| No | 467496 | 1115 | 5,929.41 | 5,459.00 | 696.86 |
| Yes | 376896 | 1115 | 8,228.28 | 7,649.00 | 844.43 |

- Mean sales without daily Promo: 5,929.41.
- Mean sales with daily Promo: 8,228.28.
- Raw mean difference: +38.77%.

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

| Promotion_Group | Observations | Stores | Mean_Sales | Median_Sales | Mean_Customers | Mean_Sales_Difference_Percent |
| --- | --- | --- | --- | --- | --- | --- |
| Neither | 398286 | 1115 | 6,018.04 | 5,527.00 | 712.22 | 0.00 |
| Promo2 only | 69210 | 571 | 5,419.36 | 5,108.00 | 608.45 | -9.95 |
| Daily Promo only | 319726 | 1115 | 8,315.15 | 7,714.00 | 861.55 | 38.17 |
| Both active | 57170 | 571 | 7,742.48 | 7,317.00 | 748.71 | 28.65 |

**Primary comparison: Promo2 only versus neither promotion active.**

- Neither active mean sales: 6,018.04.
- Promo2-only mean sales: 5,419.36.
- Raw mean difference: -9.95%.

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

- Pearson correlation on open store-days: 0.8236.
- Mean customers: 762.73.
- Median customers: 676.00.

Customer count is historically associated with sales, but
Customers is absent from the test dataset and is not known
for the future forecast horizon.

**Modelling implication:** Do not use actual future customer
counts as predictors. This would create a training-serving
availability mismatch and potential leakage.

## 9. Store type

| StoreType | Stores | Observations | Mean_Sales | Median_Sales | Mean_Customers |
| --- | --- | --- | --- | --- | --- |
| a | 602 | 457077 | 6,925.17 | 6,285.00 | 795.36 |
| b | 17 | 15563 | 10,231.41 | 9,130.00 | 2,021.82 |
| c | 148 | 112978 | 6,932.51 | 6,407.00 | 815.47 |
| d | 348 | 258774 | 6,822.14 | 6,395.00 | 606.34 |

- Highest mean: Type b
  (10,231.41).
- Lowest mean: Type d
  (6,822.14).

**Modelling implication:** Include store type in the global model
and report performance by store type.

## 10. Assortment

| Assortment | Stores | Observations | Mean_Sales | Median_Sales |
| --- | --- | --- | --- | --- |
| a | 593 | 444909 | 6,621.02 | 6,082.00 |
| b | 9 | 8212 | 8,639.35 | 8,081.00 |
| c | 513 | 391271 | 7,300.53 | 6,675.00 |

Assortment may capture differences in product range and should
be tested alongside store type and store identity.

## 11. Holiday effects

### State holidays

| StateHoliday | Observations | Mean_Sales | Median_Sales |
| --- | --- | --- | --- |
| 0 | 843482 | 6,953.52 | 6,368.00 |
| a | 694 | 8,487.47 | 7,556.00 |
| b | 145 | 9,887.89 | 8,423.00 |
| c | 71 | 9,743.75 | 8,397.00 |

### School holidays

| SchoolHoliday | Observations | Mean_Sales | Median_Sales |
| --- | --- | --- | --- |
| 0 | 680935 | 6,896.78 | 6,326.00 |
| 1 | 163457 | 7,200.18 | 6,562.00 |

Some holiday categories may have relatively few observations.
Aggregate averages can also be influenced by the stores and dates
represented in each category.

**Modelling implication:** Retain known holiday indicators as
candidate features and evaluate holiday-period performance.

## 12. Store-level variation

- Lowest store mean: 2,703.74.
- Median store mean: 6,589.95.
- Highest store mean: 21,757.48.
- Number of stores: 1,115.

Store-level sales heterogeneity supports starting with a pooled
global model that learns across stores while incorporating store
identity and store attributes.

## 13. Promo2 programme metadata and feature engineering

### Store participation

| Promo2 | Store_Count | Percentage_of_Stores |
| --- | --- | --- |
| 0 | 544 | 48.79 |
| 1 | 571 | 51.21 |

### Recurring interval schedules

| PromoInterval | Store_Count | Percentage_of_Promo2_Stores |
| --- | --- | --- |
| Jan,Apr,Jul,Oct | 335 | 58.67 |
| Feb,May,Aug,Nov | 130 | 22.77 |
| Mar,Jun,Sept,Dec | 106 | 18.56 |

### Metadata completeness

| Promo2 | Stores | Missing_Start_Year | Missing_Start_Week | Missing_Interval |
| --- | --- | --- | --- | --- |
| 0 | 544 | 544 | 544 | 544 |
| 1 | 571 | 0 | 0 | 0 |

### Derived flag quality

| Metric | Value |
| --- | --- |
| Open observations | 844392 |
| Open observations with Promo2 active | 126380 |
| Open observations with Promo2 inactive | 718012 |
| Participating stores with missing start date | 0 |
| Participating stores with missing interval | 0 |

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

| Metric | Value |
| --- | --- |
| Total stores | 1,115.00 |
| Stores with known competition distance | 1,112.00 |
| Stores with missing competition distance | 3.00 |
| Minimum distance (metres) | 20.00 |
| 25th percentile (metres) | 717.50 |
| Median distance (metres) | 2,325.00 |
| 75th percentile (metres) | 6,882.50 |
| 90th percentile (metres) | 15,679.00 |
| 95th percentile (metres) | 20,144.50 |
| Maximum distance (metres) | 75,860.00 |

### Competition metadata completeness

| Metric | Value |
| --- | --- |
| Stores with missing competition opening month/year | 354 |
| Stores with known competition opening information | 761 |
| Stores with missing competition distance | 3 |
| Stores with known competition distance | 1112 |

### Sales by competition-distance band

| Competition_Distance_Band | Number_of_Stores | Average_Store_Mean_Daily_Sales | Median_Store_Mean_Daily_Sales | Mean_Competition_Distance_Metres |
| --- | --- | --- | --- | --- |
| 0-500 m | 220 | 7,581.49 | 7,016.61 | 245.50 |
| 501-1,000 m | 110 | 6,642.81 | 5,910.00 | 721.55 |
| 1,001-2,000 m | 172 | 7,020.61 | 6,679.16 | 1,439.42 |
| 2,001-5,000 m | 261 | 6,675.97 | 6,464.92 | 3,194.48 |
| 5,001-10,000 m | 162 | 6,724.84 | 6,597.95 | 7,283.09 |
| >10,000 m | 187 | 6,831.57 | 6,675.29 | 19,335.13 |

### Competition age

| Metric | Value |
| --- | --- |
| Stores with known opening month and year | 761.00 |
| Stores with missing opening month or year | 354.00 |
| Minimum competition age in historical data (months) | -1.00 |
| Median competition age in historical data (months) | 65.00 |
| Maximum competition age in historical data (months) | 1,386.00 |

### Distance and store sales relationship

| Metric | Value |
| --- | --- |
| Stores with both distance and sales available | 1,112.00 |
| Spearman correlation: distance vs store mean sales | -0.03 |

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

- Closed-store observations: 172,817.
- Closed observations with non-zero sales: 0.
- Open-store observations with zero sales: 54.
- Negative sales observations: 0.

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
