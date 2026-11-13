# Rossmann Store Sales Forecasting — Model Evaluation Report

**Scope:** 1,115 stores; daily sales; 42-day recursive forecasting  
**Primary selected candidate:** Tuned LightGBM  
**Evidence:** committed validation predictions, metrics, segment analyses and SHAP outputs in this repository.

## 1. Executive evaluation

Tuned LightGBM is the strongest evaluated candidate across both chronological windows. It achieved **7.8206% WMAPE** in the model-selection window and **8.5643% WMAPE** in the later fixed-parameter holdout. In Window 2 it outperformed untuned LightGBM (8.7480%) and Random Forest (9.2387%) on pooled WMAPE. The model is recommended for production-readiness validation, not described as production-certified.

## 2. Problem framing and evaluation design

The objective is to forecast daily sales for each Rossmann store over a six-week horizon to support replenishment, inventory and promotion planning. The modelling population covers 1,115 stores and the forecast horizon is 42 calendar days.

Two chronological windows were used:

| Design element | Window 1: model selection | Window 2: later holdout |
|---|---:|---:|
| Training cutoff | 8 May 2015 | 19 June 2015 |
| Validation period | 9 May–19 June 2015 | 20 June–31 July 2015 |
| Horizon | 42 days | 42 days |
| Store-date rows | 46,830 | 46,830 |
| Open-store training rows | 766,637 | 804,110 |
| Purpose | Baselines, model comparison, Optuna tuning | Evaluate frozen hyperparameters on later period |

Window 1 was used for candidate comparison and hyperparameter selection. Window 2 refit the selected configuration on data through 19 June and evaluated it without further tuning. This is stronger evidence than reusing the tuning window, but it is not a completely untouched test of the entire modelling process and does not cover a full annual cycle.

Forecasts were generated recursively. At each forecast date, historical features were computed from information available before that date; model predictions (and zero for closed stores) were then appended to the history. Actual future validation sales were never fed back into the feature history. Store-date alignment was checked one-to-one.

## 3. Data audit, leakage controls and EDA

The training data contains **1,017,209 rows**, **1,115 stores**, and dates from 1 January 2013 through 31 July 2015. There are 844,392 open store-days and 172,817 closed store-days. The audit found 54 open-store zero-sales observations and no negative sales. Weekday values were checked against dates; no mismatches were found.

Key data and feature decisions:
- Sales is the target and is not imputed. Zero sales and high-sales observations are retained; no blanket outlier removal is applied.
- Closed-store rows are retained for evaluation and assigned zero forecast sales; model fitting uses open-store observations.
- `Customers` is excluded from model predictors because it is absent from the test data and unavailable at forecast time, despite its historical association with sales (Pearson correlation 0.8236 on open store-days).
- `Id` is an identifier for prediction mapping, not a predictor.
- No future sales are used to impute test values or construct historical features.
- StateHoliday semantics follow the competition data dictionary: 0 = no holiday, a = public holiday, b = Easter, c = Christmas.
- Competition data includes invalid placeholder information for Store 815. Invalid values are flagged; negative competition age is retained where it represents a future competitor opening.
- The 11 missing `Open` values in test belong to Store 622 (5–17 September 2015). They are resolved using Store 622's weekday-specific historical opening behaviour, not target sales.

EDA identified weekday, month, promotion, store type, assortment, holidays, competition and store-level heterogeneity as relevant descriptive patterns. Daily Promo observations had higher mean sales than non-promotion observations (8,228.28 vs 5,929.41); this is descriptive, not causal. Promo2-only observations had lower mean sales than neither-promotion observations in the historical aggregate. Sunday data is sparse: only 33 stores have any open Sunday observations, so Sunday averages are not network-representative. December had the highest monthly mean sales (8,608.96), while September had the lowest (6,546.34); the short history limits long-term seasonal claims.

## 4. Feature engineering and leakage prevention

The configured candidate set contains 34 features. Constant-feature removal is calculated on training data only, yielding **33 selected features** for the final fitted model; that exact selected feature list is persisted and held fixed at inference.

Feature groups:
- Store and calendar: Store, DayOfWeek, StoreType, Assortment, year/month/day/week, weekend flag and cyclical calendar encodings.
- Promotion and holidays: Promo, StateHoliday, SchoolHoliday, Promo2 and date-level Promo2 activity.
- Competition: distance, missing/invalid flags, competition age and not-yet-open indicator.
- Historical sales: lags 1/7/14/28, rolling means 7/14/28, store historical mean and store-weekday historical mean.

The recursive feature builder rejects duplicate Store-Date history, requires a single forecast date per call, and never reads current-day Sales while building features. Predictions are clipped at zero before being appended to history. These safeguards are essential to prevent target leakage and training-serving mismatch.

## 5. Baselines and candidate model comparison

All metrics below use the committed evaluation results. WMAPE is pooled absolute error divided by pooled actual sales, expressed as a percentage.

| Window | Candidate | MAE | RMSE | WMAPE | RMSLE |
|---|---|---:|---:|---:|---:|
| 1 | Seasonal naive (lag 7) | 2,168.95 | 3,745.19 | 37.23% | 3.6797 |
| 1 | Moving average (7) | 3,823.84 | 5,046.76 | 65.64% | 1.3806 |
| 1 | Naive (lag 1) | 5,515.36 | 6,785.97 | 94.67% | 7.6310 |
| 1 | Random Forest | 511.64 | 811.90 | 8.7826% | 0.1185 |
| 1 | Tuned LightGBM | 455.60 | 708.49 | **7.8206%** | **0.1085** |
| 2 | Random Forest | 554.59 | 866.16 | 9.2387% | 0.1155 |
| 2 | Untuned LightGBM | 525.13 | 817.43 | 8.7480% | 0.1102 |
| 2 | Tuned LightGBM | 514.10 | 807.69 | **8.5643%** | **0.1077** |

The tuned LightGBM Window 1 result is a model-selection result, not an independent estimate. The later Window 2 result is the primary holdout evidence. The simple baselines are materially weaker, confirming that store, calendar, promotion and historical sales signals add substantial predictive value.

## 6. Hyperparameter tuning

Optuna tuning minimised WMAPE over the complete recursive 42-day Window 1 horizon. The selected configuration is:
- n_estimators = 800
- learning_rate = 0.066245
- num_leaves = 48
- max_depth = 10
- min_child_samples = 30
- subsample = 0.92117
- colsample_bytree = 0.72884
- reg_lambda = 0.46922

These parameters were frozen before Window 2. Window 2 used the expanded training history through 19 June 2015 and no additional hyperparameter search.

## 7. Window 2 detailed error analysis

The final holdout contains 46,830 store-date rows, of which 40,282 are open-store observations. On all rows, MAE = 514.10, RMSE = 807.69, WMAPE = 8.5643%, RMSLE = 0.1077. On open rows, MAE = 597.67, RMSE = 870.87, WMAPE = 8.5643%, RMSLE = 0.1161. The WMAPE is identical because closed rows have zero actual and predicted sales and therefore contribute neither numerator nor denominator.

### Store and store-type variation

Store-level WMAPE varies materially. The worst examples include Store 292 (46.53%, 18 open observations), Store 909 (46.34%, 22 observations), and Store 876 (37.39%, 22 observations). These short coverage counts require caution. Store 909 and Store 530 show frequent underforecasting; Stores 550 and 722 were overforecast on every evaluated open observation. The best-performing examples include Store 817 (3.68%) and Store 682 (4.03%).

| StoreType | Stores | Mean store WMAPE | Pooled WMAPE | MAE | RMSE | Mean bias |
|---|---:|---:|---:|---:|---:|---:|
| B | 17 | 6.93% | 6.51% | 703.46 | 958.48 | -91.07 |
| A | 602 | 8.74% | 8.51% | 585.06 | 906.66 | +170.99 |
| C | 148 | 8.91% | 8.62% | 593.85 | 792.68 | +178.53 |
| D | 348 | 8.92% | 8.82% | 615.08 | 833.37 | +290.33 |

StoreType B has the lowest pooled WMAPE but only 17 stores. Types A, C and D are comparatively close. Mean store WMAPE and pooled WMAPE are distinct: the former weights stores equally, while the latter weights observations by actual sales.

### Sales-volume segments

Open observations were retrospectively divided into actual-sales quartiles (diagnostic only; not a model input). WMAPE decreases from 10.74% in the low-volume quartile to 7.55% in the high-volume quartile. MAE rises from 418.99 to 835.20 as sales scale increases. The low-volume group has a 69.20% overforecast rate; the high-volume group is nearly balanced (51.62% overforecast). This indicates relatively greater proportional difficulty for low-volume store-days.

### Promotion and calendar segments

Daily Promo-active observations have WMAPE 9.06% versus 8.06% when inactive, with mean bias +397.52 versus +67.81. The combined state with both daily Promo and Promo2 active has the highest segment WMAPE (10.02%) and positive mean bias (+495.68). These are descriptive segment comparisons; store mix and sales scale may contribute.

Thursday has the highest weekday WMAPE (10.27%) and Friday the lowest (7.62%). Sunday has only 193 open observations in Window 2 and must not be generalised. July WMAPE (8.99%) exceeds the partial June period (7.24%), but the periods differ in length. All Window 2 open observations have StateHoliday = 0, so holiday-category performance cannot be assessed in this holdout.

### Directional errors and business implications

Across 40,282 open-store observations:
- Underforecast: 15,505 (38.49%); average magnitude 510.72.
- Overforecast: 24,777 (61.51%); average magnitude 652.09.
- Mean bias: +204.51 per open store-day.
- Total bias: approximately +8.24 million; predicted sales approximately 289.35 million versus actual sales 281.11 million (net +2.93%).

The model has a positive aggregate bias. Overforecasting can increase holding costs and waste; underforecasting can increase stockout risk and lost sales. The correct trade-off depends on product-level margin, holding cost, replenishment lead time and service-level objectives, none of which are encoded in this competition metric.

## 8. Explainability

Native LightGBM TreeSHAP contributions were calculated on 40,282 open-store observations using 33 features. Additivity validation passed with maximum error 1.24e-10. The leading global mean absolute SHAP drivers were:
1. Store_Weekday_HistoricalMean (1,123.67)
2. Promo (829.98)
3. Sales_Lag_1 (317.59)
4. Day (250.32)
5. Sales_Lag_14 (224.75)
6. Sales_RollingMean_28 (181.86)

This supports the importance of store-specific weekday history, promotion status and recent sales history. SHAP describes model attribution, not causal effect.

## 9. Model selection and recommendation

**Selected candidate: Tuned LightGBM.** It leads the evaluated candidates across MAE, RMSE, WMAPE and RMSLE in both windows, supports recursive multi-step forecasting, and provides feature and prediction-level explanations. Proceed to production-readiness validation with:
1. automated input/schema/date/store validation;
2. store- and segment-level WMAPE, MAE, RMSE and bias monitoring;
3. promotion-specific monitoring and investigation of low-volume stores;
4. review of positive bias and business-specific asymmetric costs;
5. uncertainty estimates (e.g. quantile forecasts or prediction intervals);
6. drift monitoring, retraining policy and reproducible model/version management;
7. shadow or controlled rollout before replenishment decisions depend on forecasts.

## 10. Limitations

The validation windows are two 42-day periods, not a full annual cycle. Window 1 informed tuning; Window 2 is a later holdout but not a fully untouched external test. Point forecasts do not quantify uncertainty. External drivers such as weather, local events, stockouts, competitor actions and promotion mechanics are not fully represented. Aggregate metrics can conceal store-level failures. Feature attribution and segment comparisons are not causal. State-holiday performance is not represented in Window 2.

## 11. Metric definitions and aggregation

For observations i = 1,...,N, actual sales are y_i and predictions are ŷ_i.

| Metric | Formula | Meaning / aggregation |
|---|---|---|
| MAE | (1/N) Σ |y_i − ŷ_i| | Mean absolute store-day error; each row equal weight; sales units. |
| RMSE | sqrt[(1/N) Σ(y_i − ŷ_i)²] | Penalises large errors more; pooled across rows; sales units. |
| WMAPE | [Σ |y_i − ŷ_i| / Σ |y_i|] × 100 | Pooled absolute error relative to total actual sales; higher-volume observations carry more denominator weight. |
| RMSLE | sqrt[(1/N) Σ(ln(1+ŷ_i) − ln(1+y_i))²] | Log-scale error, useful across different sales magnitudes; zero-safe due to +1. |
| Mean bias | (1/N) Σ(ŷ_i − y_i) | Positive = overforecast; negative = underforecast. |

For store s or segment g, replace the full population with its corresponding index set I_s or I_g and denominator n_s or n_g. Pooled segment WMAPE is computed from summed absolute errors divided by summed absolute actual sales; it is not the arithmetic mean of individual store WMAPE. Underforecast rate is count(ŷ_i < y_i)/N × 100; overforecast rate is count(ŷ_i > y_i)/N × 100. Their denominators are the observations in the stated segment; exact predictions belong to neither group.
