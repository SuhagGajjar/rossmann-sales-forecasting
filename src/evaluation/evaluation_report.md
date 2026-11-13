# Rossmann Store Sales Forecasting — Model Evaluation

## 1. Evaluation Overview

### 1.1 Objective

The objective of this evaluation is to assess the predictive accuracy, reliability, and business suitability of candidate models for forecasting daily sales across Rossmann stores.

The evaluation considers:

- Forecast accuracy across the complete store network.
- Performance across different stores and store characteristics.
- Performance under different sales volumes and promotional conditions.
- Forecast errors, including underforecasting and overforecasting.
- Model performance over a multi-step forecasting horizon.
- Generalisation across chronological validation periods.
- Practical suitability for a production forecasting workflow.

The final model should provide reliable forecasts that can support replenishment planning, inventory management, store availability, and promotional planning.

### 1.2 Evaluation principles

The evaluation follows these principles:

1. **Chronological validation:** Training observations always precede the corresponding validation period.
2. **Leakage-safe feature engineering:** Historical features use only information available before the forecast date.
3. **Recursive forecasting:** Predictions are generated over the complete 42-day horizon. Predicted sales, rather than actual future sales, are used to update forecasting history.
4. **Consistent comparison:** Models are compared on identical Store-Date validation observations within each validation window.
5. **Open-store modelling:** Models are trained on open-store observations. Closed stores receive zero predicted sales.
6. **Multiple evaluation metrics:** MAE, RMSE, WMAPE, and RMSLE are used to assess different aspects of forecast error.
7. **Business interpretation:** Model selection considers operational usefulness in addition to statistical accuracy.
8. **Separate model-selection and holdout periods:** Window 1 is used for model comparison and hyperparameter selection. Window 2 evaluates the selected configuration on a later chronological period without further tuning.

## 2. Chronological Validation Design

Two chronological 42-day validation windows were used. Both windows cover all 1,115 stores and 42 calendar dates, giving 46,830 expected Store-Date observations per window.

| Parameter | Window 1: Tuning and Model Selection | Window 2: Later Chronological Holdout |
|---|---|---|
| Training cutoff | 8 May 2015 | 19 June 2015 |
| Validation start | 9 May 2015 | 20 June 2015 |
| Validation end | 19 June 2015 | 31 July 2015 |
| Forecast horizon | 42 days | 42 days |
| Validation Store-Date rows | 46,830 | 46,830 |
| Training observations (open stores) | 766,637 | 804,110 |
| Forecast method | Recursive | Recursive |
| Purpose | Model comparison and hyperparameter selection | Later evaluation of the selected configuration |

### 2.1 Window 1: Tuning and model selection

**Window 1 covers 9 May to 19 June 2015.** The training data was restricted to observations available through 8 May 2015.

This window was used to:

- Compare baseline forecasting approaches.
- Evaluate Random Forest and LightGBM.
- Conduct Optuna hyperparameter optimisation for LightGBM.
- Select the final LightGBM hyperparameter configuration.

The tuning objective was to minimise WMAPE over the complete recursive 42-day forecast horizon.

Because this window was used for model and hyperparameter selection, its final performance is treated as a model-selection result rather than an independent estimate of generalisation.

### 2.2 Window 2: Later chronological holdout

**Window 2 covers 20 June to 31 July 2015.** The training data was expanded to include observations available through 19 June 2015.

The selected LightGBM hyperparameters from Window 1 were kept fixed. The model was refitted using the expanded training dataset and evaluated recursively over the later 42-day period.

No further hyperparameter optimisation was performed using Window 2.

Window 2 therefore provides a later chronological holdout assessment of the selected configuration. It is the primary window used for the detailed error analysis in this report.

### 2.3 Interpretation of the two windows

The two windows serve different purposes and should not be treated as interchangeable:

- **Window 1 (9 May–19 June):** Model development, comparison, and hyperparameter selection.
- **Window 2 (20 June–31 July):** Later-period evaluation of the selected configuration with fixed hyperparameters.

The Window 1 results demonstrate the performance achieved during model selection. Window 2 provides the stronger evidence of how the selected configuration performs on a later period.

Both windows use the same 42-day recursive forecasting approach and the same expected number of Store-Date observations.

## 3. Evaluation Metrics

Four complementary metrics are used throughout the evaluation: MAE, RMSE, WMAPE, and RMSLE.

Unless otherwise stated, the evaluation population consists of the Store-Date observations in the relevant validation window. Metrics are calculated across the complete validation population, including closed-store observations assigned zero actual and predicted sales. Open-store-only metrics are additionally reported where relevant.

Let:

- \(y_i\) = actual sales for observation \(i\).
- \(\hat{y}_i\) = predicted sales for observation \(i\).
- \(e_i = \hat{y}_i-y_i\) = signed forecast error.
- \(N\) = number of evaluated Store-Date observations.

### 3.1 Mean Absolute Error (MAE)

**What it measures**

MAE measures the average absolute difference between actual and predicted sales. It expresses the typical magnitude of forecast error in the original sales units.

**Formula**

\[
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
\]

**Definition and aggregation**

- The numerator is the sum of absolute forecast errors across all evaluated observations.
- The denominator is the total number of evaluated Store-Date observations.
- Each observation receives equal weight, regardless of its actual sales volume.

MAE is useful because it provides an intuitive estimate of the average absolute sales deviation per store-day.

**Interpretation**

A lower MAE indicates smaller average absolute errors. However, MAE does not distinguish between underforecasting and overforecasting, and it does not give additional weight to unusually large errors.

### 3.2 Root Mean Squared Error (RMSE)

**What it measures**

RMSE measures the square root of the average squared forecast error. Squaring errors gives larger errors disproportionately more influence.

**Formula**

\[
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}
\]

**Definition and aggregation**

- Each forecast error is squared.
- Squared errors are summed across the evaluated observations.
- The sum is divided by the number of observations.
- The square root converts the result back into the original sales units.

All observations are included in the aggregation, and larger errors contribute more heavily than smaller errors.

**Interpretation**

A lower RMSE indicates better performance. Comparing RMSE with MAE helps identify the influence of large errors. A relatively high RMSE compared with MAE suggests that some observations have substantially larger errors than the typical observation.

### 3.3 Weighted Mean Absolute Percentage Error (WMAPE)

**What it measures**

WMAPE measures total absolute forecast error relative to total actual sales.

**Formula**

\[
WMAPE =
\frac{\sum_{i=1}^{N}|y_i-\hat{y}_i|}
{\sum_{i=1}^{N}|y_i|}
\times 100
\]

**Definition and aggregation**

- The numerator is the total absolute error across the evaluation population.
- The denominator is the sum of absolute actual sales across the same population.
- The metric is expressed as a percentage.

This is a pooled metric: errors and actual sales are aggregated across all observations before the ratio is calculated.

For non-negative sales, higher-volume observations contribute more to the denominator and generally have greater influence on the aggregate measure.

WMAPE is useful for a multi-store retail forecasting problem because it evaluates total error relative to the total sales volume being forecast.

**Interpretation**

A WMAPE of 8.56% means that the sum of absolute forecast errors is equivalent to approximately 8.56% of total actual sales across the evaluated population.

WMAPE is not the same as the arithmetic mean of individual store WMAPE values. The mean of store-level WMAPE gives every store equal weight, while pooled WMAPE weights stores according to their actual sales volume.

### 3.4 Root Mean Squared Logarithmic Error (RMSLE)

**What it measures**

RMSLE measures the root mean squared difference between the logarithms of actual and predicted sales. It gives greater relative importance to errors at lower sales magnitudes than raw-scale metrics do.

**Formula**

\[
RMSLE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left[
\ln(1+\hat{y}_i)-\ln(1+y_i)
\right]^2
}
\]

**Definition and aggregation**

- The logarithm is applied to actual and predicted sales after adding 1.
- The difference between the two logarithms is squared.
- Squared logarithmic errors are averaged across the evaluation population.
- The square root is taken to obtain the final metric.

The addition of 1 allows zero sales values to be included in the calculation.

**Interpretation**

A lower RMSLE indicates better performance in logarithmic sales space. RMSLE is useful alongside MAE and RMSE because it provides a complementary view of relative error behaviour across different sales magnitudes.

### 3.5 Bias and directional error

In addition to absolute error metrics, signed forecast error is used to measure systematic underforecasting or overforecasting.

**Mean bias**

\[
MeanBias =
\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)
\]

A positive mean bias indicates overforecasting on average. A negative mean bias indicates underforecasting on average.

**Total bias**

\[
TotalBias = \sum_{i=1}^{N}(\hat{y}_i-y_i)
\]

Total bias measures the aggregate difference between predicted and actual sales.

**Underforecast rate**

\[
UnderforecastRate =
\frac{\#\{i:\hat{y}_i<y_i\}}{N}\times100
\]

**Overforecast rate**

\[
OverforecastRate =
\frac{\#\{i:\hat{y}_i>y_i\}}{N}\times100
\]

The denominator is the number of observations in the relevant evaluation segment. Observations with exactly correct predictions are neither underforecasts nor overforecasts.

The average underforecast and overforecast magnitudes are calculated separately using the observations belonging to each respective category.

These measures help identify systematic directional patterns that MAE, RMSE, WMAPE, and RMSLE cannot reveal on their own.

### 3.6 Metric interpretation

The metrics complement one another:

| Metric | Primary interpretation |
|---|---|
| MAE | Typical absolute error in sales units |
| RMSE | Sensitivity to larger forecast errors |
| WMAPE | Total absolute error relative to total actual sales |
| RMSLE | Relative error behaviour across different sales magnitudes |
| Mean bias | Average direction and magnitude of forecast bias |
| Underforecast rate | Frequency of predictions below actual sales |
| Overforecast rate | Frequency of predictions above actual sales |

No single metric fully captures the business cost of forecasting errors. Underforecasting and overforecasting are therefore analysed separately.

## 4. Baseline Model Evaluation

Three simple forecasting approaches were evaluated as reference benchmarks.

| Baseline | Description |
|---|---|
| Naive Lag 1 | Uses the previous day's sales |
| Seasonal Naive Lag 7 | Uses sales from the corresponding weekday in the previous week |
| Moving Average 7 | Uses the trailing seven-day average |

All baselines were evaluated using recursive forecasting over the same 42-day validation windows as the machine-learning models.

### 4.1 Window 1 baseline results (9 May–19 June 2015)

| Model | MAE | RMSE | WMAPE | RMSLE |
|---|---:|---:|---:|---:|
| Naive Lag 1 | 5,515.36 | 6,785.97 | 94.6743% | 7.6310 |
| Seasonal Naive Lag 7 | 2,168.95 | 3,745.19 | 37.2313% | 3.6797 |
| Moving Average 7 | 3,823.84 | 5,046.76 | 65.6385% | 1.3806 |

Seasonal Naive Lag 7 was the strongest of the three baselines on MAE, RMSE, and WMAPE. This indicates that the previous week's corresponding weekday provides a more useful simple reference than either the previous day or the trailing seven-day average.

### 4.2 Window 2 baseline results (20 June–31 July 2015)

| Model | MAE | RMSE | WMAPE | RMSLE |
|---|---:|---:|---:|---:|
| Naive Lag 1 | 5,709.89 | 6,805.64 | 95.1196% | 7.8862 |
| Seasonal Naive Lag 7 | 1,282.25 | 1,953.55 | 21.3607% | 0.3715 |
| Moving Average 7 | 3,028.78 | 4,039.92 | 50.4557% | 0.8693 |

Seasonal Naive Lag 7 again outperformed the other simple baselines across all four metrics.

This establishes a meaningful benchmark for assessing whether machine-learning models can improve on recurring weekly sales patterns.

## 5. Random Forest Evaluation

A Random Forest regression model was developed using the engineered store, calendar, promotion, competition, and historical sales features.

The model was trained on open-store observations and evaluated using recursive multi-step forecasting.

### 5.1 Window 1 (9 May–19 June 2015)

| Metric | Result |
|---|---:|
| MAE | 511.64 |
| RMSE | 811.90 |
| WMAPE | 8.7826% |
| RMSLE | 0.1185 |

### 5.2 Window 2 (20 June–31 July 2015)

| Metric | Result |
|---|---:|
| MAE | 554.59 |
| RMSE | 866.16 |
| WMAPE | 9.2387% |
| RMSLE | 0.1155 |

The second-window evaluation covered 46,830 Store-Date observations, of which 40,282 were open-store observations.

Open-store-only WMAPE was also 9.2387%. WMAPE remains unchanged because closed-store observations have zero actual and predicted sales, contributing neither absolute error nor actual sales to the WMAPE calculation.

Random Forest substantially outperformed the simple baselines but was less accurate than the LightGBM models.

## 6. LightGBM Evaluation

An initial LightGBM regression model was trained using the engineered feature set and evaluated recursively.

### 6.1 Window 2 results (20 June–31 July 2015)

| Metric | Result |
|---|---:|
| MAE | 525.13 |
| RMSE | 817.43 |
| WMAPE | 8.7480% |
| RMSLE | 0.1102 |

The untuned LightGBM model outperformed Random Forest and the simple baselines on the second validation window.

## 7. LightGBM Hyperparameter Tuning

### 7.1 Tuning approach

Optuna was used to optimise LightGBM hyperparameters.

The tuning study used:

- Study name: `lightgbm_42day_tuning_seed42`
- Number of completed trials: 20
- Random seed: 42
- Optimisation objective: WMAPE minimisation
- Validation approach: recursive 42-day forecasting
- Training cutoff: 8 May 2015
- Validation period: 9 May–19 June 2015 (Window 1)

Each trial trained a model and evaluated the complete recursive validation horizon. The optimisation objective was WMAPE, while MAE, RMSE, and RMSLE were also recorded.

### 7.2 Selected hyperparameters

| Hyperparameter | Selected value |
|---|---:|
| n_estimators | 800 |
| learning_rate | 0.0662449763 |
| num_leaves | 48 |
| max_depth | 10 |
| min_child_samples | 30 |
| subsample | 0.9211711911 |
| colsample_bytree | 0.7288364633 |
| reg_lambda | 0.4692159054 |
| random_state | 42 |

The selected configuration was frozen after the tuning study. No further hyperparameter optimisation was performed using Window 2.

### 7.3 Tuning-window performance (Window 1)

| Metric | Result |
|---|---:|
| MAE | 455.60 |
| RMSE | 708.49 |
| WMAPE | 7.8206% |
| RMSLE | 0.1085 |

These values represent performance on the window used for hyperparameter selection and should not be interpreted as an independent generalisation estimate.

## 8. Cross-Window Model Comparison

### 8.1 Window 1 comparison (9 May–19 June 2015)

| Model | MAE | RMSE | WMAPE | RMSLE |
|---|---:|---:|---:|---:|
| Tuned LightGBM | 455.60 | 708.49 | 7.8206% | 0.1085 |
| Random Forest | 511.64 | 811.90 | 8.7826% | 0.1185 |
| Seasonal Naive Lag 7 | 2,168.95 | 3,745.19 | 37.2313% | 3.6797 |
| Moving Average 7 | 3,823.84 | 5,046.76 | 65.6385% | 1.3806 |
| Naive Lag 1 | 5,515.36 | 6,785.97 | 94.6743% | 7.6310 |

Tuned LightGBM achieved the lowest error across all four metrics in the first window.

This result is useful for comparing the candidate models and selecting hyperparameters, but it is not an independent generalisation estimate because Window 1 was used during model selection.

### 8.2 Window 2 comparison (20 June–31 July 2015)

| Model | MAE | RMSE | WMAPE | RMSLE |
|---|---:|---:|---:|---:|
| Tuned LightGBM | 514.10 | 807.69 | 8.5643% | 0.1077 |
| Untuned LightGBM | 525.13 | 817.43 | 8.7480% | 0.1102 |
| Random Forest | 554.59 | 866.16 | 9.2387% | 0.1155 |
| Seasonal Naive Lag 7 | 1,282.25 | 1,953.55 | 21.3607% | 0.3715 |
| Moving Average 7 | 3,028.78 | 4,039.92 | 50.4557% | 0.8693 |
| Naive Lag 1 | 5,709.89 | 6,805.64 | 95.1196% | 7.8862 |

Tuned LightGBM achieved the lowest error across all four metrics in the later chronological holdout window.

Its WMAPE was approximately 2.10% lower relative to the untuned LightGBM model and approximately 7.30% lower relative to Random Forest.

### 8.3 Cross-window interpretation

The selected LightGBM configuration performs consistently well across both chronological windows:

- It achieves the lowest error among the evaluated models in Window 1.
- It retains the best performance across all four reported metrics in Window 2.
- It outperforms the untuned LightGBM model in the later window.
- It substantially improves on the simple seasonal and non-seasonal baselines.

The Window 2 result is particularly important because the selected hyperparameters were frozen before this later evaluation period.

The change in WMAPE from 7.82% in Window 1 to 8.56% in Window 2 is not, by itself, evidence of model deterioration. The windows cover different dates and use different amounts of training data. The results should be interpreted as performance on two separate chronological periods.

## 9. Detailed Error Analysis — Window 2

### 9.1 Evaluation population and metric scope

The detailed error analysis uses the tuned LightGBM prediction file:

`models/experiments/lightgbm_second_window/second_window_tuned_lightgbm_predictions.csv`

The evaluation window is **20 June to 31 July 2015**, covering 42 calendar days.

| Evaluation item | Result |
|---|---:|
| Stores | 1,115 |
| Calendar days | 42 |
| Total Store-Date observations | 46,830 |
| Open-store observations | 40,282 |
| Closed-store observations | 6,548 |
| Total actual sales | 281,113,498 |
| Total predicted sales | Approximately 289,351,548 |

The model assigns zero predicted sales to closed stores. For this reason, two metric populations are useful:

1. **All Store-Date observations:** Includes open and closed stores and measures complete network-level forecasting performance.
2. **Open-store observations:** Focuses on demand forecasting where stores are open.

The reported network-level WMAPE is unchanged when closed-store observations are excluded because both actual and predicted sales are zero for those observations. MAE and RMSE do change because the denominator includes fewer observations when calculated on open-store rows only.

### 9.2 Overall Window 2 performance

| Metric | All Store-Date observations | Open-store observations |
|---|---:|---:|
| MAE | 514.10 | 597.67 |
| RMSE | 807.69 | 870.87 |
| WMAPE | 8.5643% | 8.5643% |
| RMSLE | 0.1077 | 0.1161 |

Open-store-only WMAPE of 8.5643% means that total absolute forecast error is approximately 8.56% of total actual sales across open-store observations.

The difference between all-row and open-only MAE/RMSE is a denominator effect, not an improvement in the predictions for open stores.

### 9.3 Store-level performance

Store-level metrics were calculated independently for each store using its open-store observations in the 42-day window.

For a store \(s\), let \(I_s\) represent the set of open-store observations belonging to that store, and let \(n_s\) be the number of those observations.

**Store-level MAE**

\[
MAE_s =
\frac{1}{n_s}
\sum_{i\in I_s}|y_i-\hat{y}_i|
\]

This measures the average absolute daily sales error for that store.

**Store-level WMAPE**

\[
WMAPE_s =
\frac{\sum_{i\in I_s}|y_i-\hat{y}_i|}
{\sum_{i\in I_s}|y_i|}
\times100
\]

This measures the store's total absolute error relative to its actual sales over the evaluation period.

**Store-level mean bias**

\[
MeanBias_s =
\frac{1}{n_s}
\sum_{i\in I_s}(\hat{y}_i-y_i)
\]

This measures whether the model systematically overforecasts or underforecasts for that store.

**Aggregation and interpretation**

Each store is evaluated using its own open-store observations. The arithmetic mean of store WMAPE values gives every store equal weight. It is therefore different from network-level pooled WMAPE, which gives more influence to stores with higher actual sales.

#### Worst-performing stores

The following stores had the highest store-level WMAPE values.

| Store | StoreType | Open observations | WMAPE | MAE | Mean bias | Underforecast rate |
|---|---|---:|---:|---:|---:|---:|
| 292 | A | 18 | 46.53% | 3,409.97 | -1,427.12 | 38.89% |
| 909 | A | 22 | 46.34% | 7,334.31 | -4,385.99 | 81.82% |
| 876 | A | 22 | 37.39% | 4,208.99 | -1,627.37 | 36.36% |
| 550 | D | 36 | 34.21% | 1,768.43 | +1,768.43 | 0.00% |
| 722 | A | 36 | 28.66% | 2,167.40 | +2,167.40 | 0.00% |
| 269 | A | 36 | 23.39% | 1,804.04 | +1,761.39 | 5.56% |
| 530 | A | 42 | 23.08% | 1,602.94 | -1,515.41 | 88.10% |
| 636 | C | 36 | 20.97% | 1,383.47 | -84.51 | 41.67% |
| 274 | B | 42 | 20.20% | 1,191.86 | -1,191.86 | 100.00% |
| 931 | A | 42 | 19.41% | 895.40 | -895.40 | 100.00% |

The worst-performing stores show different error patterns:

- Stores 292, 909, and 876 have particularly high WMAPE, but their evaluation coverage is only 18–22 open observations. Their rankings should therefore be interpreted cautiously.
- Stores 909 and 530 show substantial underforecasting frequency and negative mean bias.
- Stores 550 and 722 were overforecast on every evaluated open observation. For these stores, mean bias equals MAE because every signed error was positive.
- Store 274 and Store 931 were underforecast on every evaluated open observation.

These patterns suggest that store-specific investigation is important. A high WMAPE may reflect persistent directional bias, unusually variable demand, limited historical information, or a small actual-sales denominator.

#### Best-performing stores

The following stores had the lowest store-level WMAPE values.

| Store | StoreType | Open observations | WMAPE | MAE | Mean bias |
|---|---|---:|---:|---:|---:|
| 817 | A | 36 | 3.68% | 764.62 | -341.88 |
| 682 | B | 42 | 4.03% | 452.16 | +39.43 |
| 974 | A | 36 | 4.21% | 513.40 | +180.32 |
| 733 | B | 42 | 4.32% | 630.71 | +457.93 |
| 1097 | B | 42 | 4.32% | 461.64 | -188.17 |
| 416 | A | 36 | 4.48% | 509.77 | -7.46 |
| 1075 | A | 36 | 4.56% | 528.17 | +168.84 |
| 498 | A | 36 | 4.57% | 319.12 | +139.34 |
| 270 | A | 36 | 4.58% | 457.01 | +57.66 |
| 632 | A | 36 | 4.80% | 363.85 | +130.95 |

The best-performing stores show that the model can achieve relatively low proportional errors across multiple stores and store types. However, low WMAPE does not necessarily imply low absolute error, as stores with different sales volumes have different error scales.

### 9.4 StoreType-level performance

StoreType-level pooled metrics were calculated by aggregating all open-store observations belonging to each StoreType.

For StoreType \(t\), let \(I_t\) represent its open-store observations.

**Pooled StoreType WMAPE**

\[
WMAPE_t =
\frac{\sum_{i\in I_t}|y_i-\hat{y}_i|}
{\sum_{i\in I_t}|y_i|}
\times100
\]

**Mean store WMAPE within a StoreType**

\[
MeanStoreWMAPE_t =
\frac{1}{S_t}\sum_{s=1}^{S_t}WMAPE_s
\]

where \(S_t\) is the number of stores in StoreType \(t\).

Pooled StoreType WMAPE weights observations according to actual sales volume. Mean store WMAPE gives each store equal weight. The two measures answer different questions and should not be treated as interchangeable.

| StoreType | Stores | Mean store WMAPE | Median store WMAPE | Pooled WMAPE | MAE | RMSE | Mean bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 17 | 6.93% | 6.44% | 6.51% | 703.46 | 958.48 | -91.07 |
| A | 602 | 8.74% | 8.15% | 8.51% | 585.06 | 906.66 | +170.99 |
| C | 148 | 8.91% | 8.49% | 8.62% | 593.85 | 792.68 | +178.53 |
| D | 348 | 8.92% | 8.39% | 8.82% | 615.08 | 833.37 | +290.33 |

**Key findings**

- StoreType B has the lowest pooled WMAPE at 6.51%. However, it contains only 17 stores, so the result should be interpreted with the smaller group size in mind.
- StoreTypes A, C, and D have relatively similar pooled WMAPE values, ranging from approximately 8.51% to 8.82%.
- StoreType D has the highest pooled WMAPE among the four groups, although the difference from A and C is modest.
- StoreTypes A, C, and D have positive mean bias, indicating aggregate overforecasting.
- StoreType B has a small negative mean bias, indicating slight aggregate underforecasting.
- The mean store WMAPE is higher than pooled WMAPE for every StoreType, demonstrating the influence of store-level weighting and differences in store sales volume.

StoreType B's lower error is an observed association. It does not establish that the store type itself causes better forecasting performance.

### 9.5 Sales-volume segment analysis

Open-store observations were divided into four sales-volume segments using actual daily sales values.

The segmentation is based on **actual Sales**, not Customers. Customer counts were not used because they are unavailable at forecast inference time and are excluded from the model features.

The quartile boundaries were:

| Boundary | Actual daily sales |
|---|---:|
| 25th percentile | 4,919.25 |
| 50th percentile | 6,384.00 |
| 75th percentile | 8,367.00 |

Each segment contains approximately one-quarter of the open-store observations.

For a segment \(g\), pooled WMAPE is:

\[
WMAPE_g =
\frac{\sum_{i\in I_g}|y_i-\hat{y}_i|}
{\sum_{i\in I_g}|y_i|}
\times100
\]

where \(I_g\) represents the observations assigned to that actual-sales segment.

The actual-sales share is:

\[
ActualSalesShare_g =
\frac{\sum_{i\in I_g}y_i}
{\sum_{i=1}^{N}y_i}
\times100
\]

This measures the contribution of each segment to total actual sales in the evaluation population.

| Sales-volume segment | Observations | Mean actual sales | MAE | RMSE | WMAPE | RMSLE | Mean bias | Underforecast rate | Overforecast rate | Actual sales share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Low | 10,071 | 3,899.51 | 418.99 | 585.69 | 10.74% | 0.1368 | +259.03 | 30.80% | 69.20% | 13.97% |
| Lower-Middle | 10,071 | 5,649.41 | 512.85 | 691.79 | 9.08% | 0.1138 | +253.79 | 35.68% | 64.32% | 20.24% |
| Upper-Middle | 10,070 | 7,298.70 | 623.67 | 836.52 | 8.54% | 0.1082 | +252.55 | 39.11% | 60.89% | 26.15% |
| High | 10,070 | 11,067.38 | 835.20 | 1,229.78 | 7.55% | 0.1029 | +52.66 | 48.38% | 51.62% | 39.65% |

**Key findings**

- The low sales-volume segment has the highest proportional error, with WMAPE of 10.74% and RMSLE of 0.1368.
- Relative accuracy improves as sales volume increases. WMAPE decreases from 10.74% in the low segment to 7.55% in the high segment.
- Absolute errors increase with sales volume: MAE increases from approximately 419 to 835, and RMSE increases from approximately 586 to 1,230. This reflects the larger scale of actual sales and does not necessarily indicate poorer relative performance.
- The low-volume segment has an overforecast rate of 69.20%, compared with 51.62% in the high-volume segment.
- Mean bias is positive in all four segments, although the high-volume segment has a substantially smaller mean bias.
- The high-volume segment contributes 39.65% of total actual sales, while the low-volume segment contributes 13.97%.

The segment boundaries are based on actual validation-period sales and are used only for retrospective diagnostics. They are not model inputs or operational future-known segment labels.

### 9.6 Promotion performance analysis

Promotion performance was assessed across three views:

1. Daily promotion (`Promo`) active versus inactive.
2. Continuing promotion programme (`Promo2_Active`) active versus inactive.
3. Combined promotion states.

The metrics are calculated using open-store observations within each segment.

For a promotion segment \(p\), WMAPE is:

\[
WMAPE_p =
\frac{\sum_{i\in I_p}|y_i-\hat{y}_i|}
{\sum_{i\in I_p}|y_i|}
\times100
\]

where \(I_p\) is the set of open-store observations belonging to that promotion segment.

The number of observations, store coverage, and mean actual sales are reported to provide context for comparisons between segments.

#### Daily Promo status

| Daily Promo status | Observations | Stores | Mean actual sales | MAE | RMSE | WMAPE | RMSLE | Mean bias | Underforecast rate | Overforecast rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Inactive | 23,581 | 1,115 | 5,905.37 | 475.98 | 733.52 | 8.06% | 0.1123 | +67.81 | 44.16% | 55.84% |
| Active | 16,701 | 1,115 | 8,494.04 | 769.50 | 1,034.20 | 9.06% | 0.1214 | +397.52 | 30.49% | 69.51% |

Daily promotion-active observations have higher MAE, RMSE, WMAPE, and RMSLE than inactive observations.

The mean bias is also substantially higher on active promotion observations. The overforecast rate increases from 55.84% to 69.51%.

This indicates that the model has greater difficulty with the observed daily-promotion segment and tends to overforecast more frequently in that segment.

#### Promo2 status

| Promo2 status | Observations | Stores | Mean actual sales | MAE | RMSE | WMAPE | RMSLE | Mean bias | Underforecast rate | Overforecast rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Inactive | 30,255 | 1,115 | 7,078.83 | 584.84 | 873.37 | 8.26% | 0.1118 | +181.91 | 39.04% | 60.96% |
| Active | 10,027 | 441 | 6,676.33 | 636.39 | 863.27 | 9.53% | 0.1282 | +272.69 | 36.82% | 63.18% |

Promo2-active observations have higher MAE, WMAPE, and RMSLE than inactive observations, although RMSE is slightly lower.

Promo2-active observations cover 441 stores rather than the full 1,115-store network. The difference in store coverage and store mix should be considered when interpreting the results.

Promo2 represents a continuing promotion programme status and should not be interpreted as a new independent promotion event on every active day.

#### Combined promotion states

| Combined promotion state | Observations | Stores | Mean actual sales | MAE | RMSE | WMAPE | RMSLE | Mean bias | Underforecast rate | Overforecast rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Neither active | 18,121 | 1,115 | 5,985.47 | 467.25 | 748.47 | 7.81% | 0.1089 | +62.28 | 43.98% | 56.02% |
| Daily Promo only | 12,134 | 1,115 | 8,711.65 | 760.44 | 1,032.13 | 8.73% | 0.1160 | +360.57 | 31.68% | 68.32% |
| Promo2 only | 5,460 | 441 | 5,639.53 | 504.94 | 681.53 | 8.95% | 0.1227 | +86.16 | 44.76% | 55.24% |
| Both active | 4,567 | 441 | 7,915.87 | 793.54 | 1,039.67 | 10.02% | 0.1346 | +495.68 | 27.33% | 72.67% |

**Key findings**

- The combined segment with both daily Promo and Promo2 active has the highest WMAPE at 10.02%.
- This segment also has the highest RMSLE at 0.1346 and the highest mean positive bias at approximately 495.68.
- Overforecasting occurs in 72.67% of observations where both promotion indicators are active.
- Daily Promo-active observations show higher error than inactive observations across all four principal metrics.
- Promo2-active observations have higher MAE, WMAPE, and RMSLE, although their RMSE is slightly lower.

These findings identify overlapping promotion conditions as an area for further investigation. They do not establish that promotions cause the higher errors. Store mix, actual sales volume, calendar effects, and other factors may contribute to the observed differences.

### 9.7 Calendar performance analysis

Calendar performance was analysed by day of week, weekday versus weekend, month, StateHoliday, and SchoolHoliday.

For a calendar segment \(c\), the pooled WMAPE is:

\[
WMAPE_c =
\frac{\sum_{i\in I_c}|y_i-\hat{y}_i|}
{\sum_{i\in I_c}|y_i|}
\times100
\]

where \(I_c\) is the set of open-store observations in that calendar segment.

#### Day-of-week performance

| Day | Observations | Mean actual sales | MAE | RMSE | WMAPE | RMSLE | Mean bias | Underforecast rate | Overforecast rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Monday | 6,682 | 8,215.74 | 639.98 | 1,011.83 | 7.79% | 0.1091 | +110.12 | 42.14% | 57.86% |
| Tuesday | 6,682 | 7,330.33 | 580.63 | 852.32 | 7.92% | 0.1080 | +191.09 | 35.33% | 64.67% |
| Wednesday | 6,682 | 6,813.71 | 624.95 | 892.34 | 9.17% | 0.1183 | +323.65 | 32.89% | 67.11% |
| Thursday | 6,680 | 6,804.80 | 698.81 | 962.86 | 10.27% | 0.1273 | +312.95 | 38.88% | 61.12% |
| Friday | 6,681 | 7,018.47 | 534.77 | 723.37 | 7.62% | 0.1037 | +161.86 | 40.28% | 59.72% |
| Saturday | 6,682 | 5,644.41 | 501.42 | 732.09 | 8.88% | 0.1248 | +148.61 | 40.56% | 59.44% |
| Sunday | 193 | 8,513.45 | 787.26 | 1,170.10 | 9.25% | 0.2065 | -529.40 | 68.39% | 31.61% |

**Key findings**

- Thursday has the highest WMAPE among the weekdays at 10.27%.
- Friday has the lowest weekday WMAPE at 7.62%.
- Wednesday also has relatively high WMAPE at 9.17%, with a positive mean bias of approximately 323.65.
- Sunday has a relatively high RMSLE of 0.2065 and the highest MAE and RMSE among the listed days. However, it has only 193 open-store observations, compared with approximately 6,680 observations on each other day.
- Sunday results therefore reflect a limited subset of stores and should not be generalised to the entire store network.

#### Weekday versus weekend

| Segment | Observations | MAE | RMSE | WMAPE | RMSLE | Mean bias | Underforecast rate | Overforecast rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weekday | 33,407 | 615.83 | 894.08 | 8.51% | 0.1136 | +219.93 | 37.91% | 62.09% |
| Weekend | 6,875 | 509.45 | 747.89 | 8.90% | 0.1278 | +129.58 | 41.34% | 58.66% |

Weekend WMAPE is slightly higher than weekday WMAPE, although weekend MAE and RMSE are lower.

This illustrates why multiple metrics are useful: lower absolute error does not necessarily mean lower proportional error.

#### Month-level performance

Window 2 covers only the final 11 days of June and the full month of July.

| Month | Open observations | MAE | RMSE | WMAPE | RMSLE | Mean bias |
|---|---:|---:|---:|---:|---:|---:|
| June (20–30 June) | 10,094 | 493.32 | 832.65 | 7.24% | 0.1012 | +101.74 |
| July (1–31 July) | 30,188 | 632.56 | 883.28 | 8.99% | 0.1207 | +238.87 |

July has higher error across all four principal metrics.

However, June contains only 11 calendar days, while July contains 31. These segments also differ in calendar coverage and observation volume. The results should not be interpreted as proof of a seasonal or month-specific effect.

#### StateHoliday performance

All 40,282 open-store observations in Window 2 have `StateHoliday = 0`.

Consequently, the Window 2 evaluation does not contain open-store observations for public, Easter, or Christmas holidays.

The model's performance on those holiday categories cannot be assessed using this validation window.

#### SchoolHoliday performance

| SchoolHoliday status | Observations | MAE | RMSE | WMAPE | RMSLE | Mean bias | Underforecast rate | Overforecast rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No School Holiday | 26,978 | 579.37 | 872.12 | 8.69% | 0.1193 | +180.23 | 39.40% | 60.60% |
| School Holiday | 13,304 | 634.78 | 868.32 | 8.35% | 0.1094 | +253.75 | 36.64% | 63.36% |

School-holiday observations have slightly lower WMAPE and RMSLE but higher MAE and mean bias.

These differences are descriptive. Store composition, sales volume, and calendar interactions may influence the observed results.

### 9.8 Directional error analysis

Directional error was analysed across all 40,282 open-store observations.

The underforecast and overforecast rates are calculated as the number of observations in each category divided by the total number of open-store observations.

| Measure | Result |
|---|---:|
| Open-store observations | 40,282 |
| Underforecast observations | 15,505 |
| Overforecast observations | 24,777 |
| Underforecast rate | 38.49% |
| Overforecast rate | 61.51% |
| Average underforecast magnitude | 510.72 |
| Average overforecast magnitude | 652.09 |
| Mean bias per open-store observation | +204.51 |
| Total bias | Approximately +8,238,050 |

The model overforecasts more frequently than it underforecasts.

The average overforecast magnitude is also larger than the average underforecast magnitude. This results in a positive aggregate bias.

The total predicted sales are approximately 289.35 million, compared with actual sales of approximately 281.11 million. The net difference is approximately 8.24 million, or 2.93% of actual sales.

The positive aggregate bias is important for inventory planning because persistent overforecasting may lead to excess replenishment and inventory holding costs.

However, aggregate overforecasting does not mean every store or segment is overforecast. Store-level and segment-level analysis identifies examples of substantial underforecasting as well.

### 9.9 Error categories and potential explanations

The Window 2 analysis identifies several categories requiring attention.

| Error category | Observed evidence | Potential explanation or investigation |
|---|---|---|
| Store-specific errors | Store WMAPE ranges from 3.68% to 46.53% among the reported best and worst stores | Different demand patterns, limited historical observations, store-specific events, or incomplete historical data |
| Low-volume errors | Low-volume WMAPE is 10.74%, compared with 7.55% for high-volume observations | Smaller sales denominators make proportional errors more sensitive; low-volume patterns may also be more variable |
| Promotion-related errors | Both promotion indicators active has WMAPE of 10.02% and mean bias of +495.68 | Promotion intensity, interactions, campaign timing, and demand response may not be fully represented |
| Calendar-related errors | Thursday has WMAPE of 10.27%; Sunday has limited coverage and high RMSLE | Day-specific demand variation and uneven store opening patterns |
| Directional bias | 61.51% of open-store predictions are overforecasts | Model calibration or segment-specific prediction bias may require investigation |
| Historical-data limitations | Historical gaps and competition-data quality issues exist in the source dataset | Some stores may have less reliable lag, rolling, or competition-related information |

These are potential explanations and investigation areas, not proven causal attributions. Further controlled analysis would be required to establish whether a particular data-quality issue or feature limitation directly explains the observed prediction errors.

## 10. Key Findings and Business Interpretation

### 10.1 Forecast accuracy

Tuned LightGBM achieved a Window 2 WMAPE of 8.5643%, with MAE of 514.10 and RMSE of 807.69 across all Store-Date observations.

On open-store observations, MAE is 597.67 and WMAPE remains 8.5643%.

The model outperforms the evaluated Random Forest, untuned LightGBM, and simple baseline models on the later chronological holdout.

### 10.2 Store-level consistency

The model performs well across many stores, but performance is not uniform.

Some stores have WMAPE below 5%, while a small number have WMAPE above 30%. These high-error stores should be prioritised for operational monitoring and targeted investigation.

Store-level reporting is important because a good network-level WMAPE can conceal individual stores with substantial forecasting problems.

### 10.3 Sales-volume differences

Relative forecasting accuracy improves with sales volume in the evaluated window.

Low-volume observations have the highest WMAPE and RMSLE, while high-volume observations have higher absolute MAE and RMSE but lower proportional error.

This distinction matters operationally: a high absolute error at a high-volume store may be more important in units, while a smaller absolute error at a low-volume store may represent a larger proportion of demand.

### 10.4 Promotion-related risk

Promotion-active observations have higher errors than non-promotion observations in several comparisons.

The combined segment with both daily Promo and Promo2 active has the highest WMAPE and mean positive bias.

This suggests that promotion-heavy periods deserve additional monitoring and potentially more detailed promotion-related features.

### 10.5 Overforecasting and underforecasting

The model overforecasts in approximately 61.51% of open-store observations and has a positive total bias of approximately 8.24 million sales units across the evaluation window.

This is a relevant inventory-planning consideration.

However, the model also underforecasts some stores and segments. An operational decision should therefore not be based solely on the aggregate network bias.

### 10.6 Business implications

Forecasting errors have different operational consequences.

| Forecast outcome | Potential business consequence | Operational consideration |
|---|---|---|
| Underforecasting | Insufficient replenishment, stockouts, reduced product availability, and potentially lost sales | Monitor underforecasting frequency and magnitude, especially for high-demand and promotion periods |
| Overforecasting | Excess inventory, higher holding costs, waste, and reduced inventory efficiency | Monitor persistent positive bias and avoid unnecessary replenishment |
| Promotion underforecasting | Insufficient stock during promotional demand peaks | Review promotion-specific forecast errors and replenishment buffers |
| Store-specific errors | Uneven service levels across the store network | Monitor individual store performance rather than relying only on network-level metrics |

The appropriate balance between underforecasting and overforecasting depends on product characteristics, replenishment lead times, holding costs, stockout costs, and service-level requirements.

### 10.7 Asymmetric business loss

MAE and WMAPE treat underforecasting and overforecasting symmetrically. In retail operations, their costs may be different.

A simple asymmetric loss function can be defined as:

\[
L(y,\hat{y}) =
c_u\max(y-\hat{y},0)
+
c_o\max(\hat{y}-y,0)
\]

where:

- \(y\) is actual sales.
- \(\hat{y}\) is predicted sales.
- \(c_u\) is the business cost per unit of underforecasting.
- \(c_o\) is the business cost per unit of overforecasting.
- \(c_u\) and \(c_o\) represent the relative operational costs of the two error types.

When stockout and lost-sales costs are greater than excess-inventory costs, \(c_u\) may be greater than \(c_o\). Where inventory waste or holding costs dominate, the relationship may be reversed.

The actual cost weights should be determined using business inputs rather than assumed from forecast metrics.

A future model enhancement could evaluate cost-weighted forecasting or quantile regression to support different inventory service-level objectives.

## 11. Final Model Selection

### 11.1 Selected candidate: Tuned LightGBM

Tuned LightGBM is selected as the preferred forecasting candidate based on the available validation evidence.

The selection is supported by:

- Best performance across MAE, RMSE, WMAPE, and RMSLE in Window 1.
- Best performance across the same four metrics in the later Window 2 holdout.
- Better Window 2 performance than untuned LightGBM.
- Better Window 2 performance than Random Forest.
- Substantial improvement over the simple forecasting baselines.
- Ability to model nonlinear relationships and interactions across store, calendar, promotion, competition, and historical sales features.
- Suitability for a scalable tabular forecasting workflow.

### 11.2 Model comparison summary

| Evaluation criterion | Tuned LightGBM assessment |
|---|---|
| Forecast accuracy | Best among evaluated models on both validation windows |
| Multi-step forecasting | Evaluated using recursive 42-day forecasting |
| Store-level performance | Good for many stores, with a subset requiring investigation |
| Promotion performance | Higher error in some promotion-active segments; requires monitoring |
| Bias | Positive aggregate bias in Window 2 |
| Interpretability | Supports feature importance and prediction-contribution analysis |
| Operational suitability | Suitable candidate for batch forecasting, subject to production validation |
| Remaining risks | Historical data quality, promotion-related error, store-level variation, and recursive error propagation |

### 11.3 Selection caveat

The selected model should be considered the preferred candidate for production validation, not a fully production-certified forecasting system.

Before deployment, the model should undergo further checks covering:

- Reproducibility of training and inference.
- Prediction and input-data validation.
- Store-level monitoring and bias checks.
- Explainability review for representative predictions.
- Business-specific cost evaluation for underforecasting and overforecasting.
- Production pipeline and monitoring validation.

## 12. Limitations and Evaluation Caveats

### 12.1 Validation limitations

- Window 1 was used for model comparison and hyperparameter selection and is not an independent generalisation estimate.
- Window 2 is a later chronological holdout with fixed selected hyperparameters, but it is not a completely untouched test of the entire modelling process.
- Both windows cover 42 days and do not establish performance across a complete annual seasonal cycle.
- State-holiday performance cannot be assessed using Window 2 because all open-store observations in that window have `StateHoliday = 0`.
- Sunday results have limited coverage and should not be generalised across the full store network.

### 12.2 Data-quality limitations

- Historical source-data gaps exist for some stores and may affect lag and rolling feature availability or representativeness.
- Competition data contains invalid placeholder information for Store 815. Invalid competition records are flagged rather than treated as reliable dates.
- Negative competition age values are retained where they represent future competitor-opening information.
- Promotion indicators do not capture every aspect of promotion intensity, mechanics, or customer response.
- Missing or structurally unavailable source information may limit the model's ability to represent certain store characteristics.

### 12.3 Forecasting limitations

- Both validation windows use recursive forecasting. Errors in earlier predictions can influence lag and rolling features used for later predictions.
- The model produces point forecasts rather than prediction intervals, so it does not directly quantify uncertainty.
- External demand drivers such as weather, local events, competitor activity, stock availability, and stockouts are not fully represented.
- Customers is excluded from the model features because it is not available at forecast inference time.
- A high network-level score can conceal poor performance at individual stores or in specific operational segments.

### 12.4 Metric and interpretation limitations

- WMAPE is an aggregate metric and does not give every store equal weight.
- Mean store WMAPE and pooled network WMAPE measure different things and should not be interchanged.
- MAE and RMSE are expressed in sales units and are affected by the scale of actual sales.
- WMAPE does not distinguish between underforecasting and overforecasting.
- Feature importance and prediction contributions describe model behaviour; they do not establish causal relationships.
- Promotion and calendar segment comparisons are descriptive and may be influenced by store mix, sales volume, and other confounding factors.

### 12.5 Production-readiness considerations

Before production deployment, the following should be addressed:

1. **Data validation:** Validate store identifiers, dates, opening status, promotion information, competition attributes, and historical sales completeness.
2. **Forecast monitoring:** Monitor WMAPE, MAE, RMSE, and directional bias at both network and store levels.
3. **Segment monitoring:** Track low-volume stores, promotion-active periods, and stores with historically high errors.
4. **Bias calibration:** Investigate the positive aggregate bias and determine whether business-specific calibration is appropriate.
5. **Uncertainty estimation:** Consider prediction intervals or quantile forecasts for replenishment decisions.
6. **Model monitoring:** Monitor changes in input distributions, store behaviour, and forecast errors over time.
7. **Retraining strategy:** Establish a retraining schedule and trigger additional reviews when sustained performance deterioration is detected.
8. **Business alignment:** Set error thresholds and asymmetric cost weights in consultation with replenishment and inventory stakeholders.

## 13. Overall Conclusion

Tuned LightGBM is the strongest candidate among the evaluated models for Rossmann daily store sales forecasting.

It achieved a WMAPE of 7.8206% in Window 1 and 8.5643% in the later Window 2 holdout, outperforming the evaluated Random Forest, untuned LightGBM, and simple forecasting baselines.

The detailed Window 2 analysis shows that overall forecasting accuracy is accompanied by meaningful differences across stores, sales-volume segments, promotional conditions, and calendar periods. The model also exhibits a positive aggregate bias, with overforecasting more frequent than underforecasting.

These findings do not undermine the model selection, but they identify the operational controls required before deployment. Store-level monitoring, promotion-related analysis, data-quality checks, and business-aware treatment of underforecasting and overforecasting should form part of the production forecasting workflow.

**Final recommendation:** Proceed with tuned LightGBM as the preferred forecasting candidate for the next stage of validation and production-readiness assessment. Its deployment should be accompanied by store-level performance monitoring, bias tracking, data-quality validation, and ongoing evaluation against actual replenishment and inventory outcomes.