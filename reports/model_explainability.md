# Rossmann Store Sales Forecasting — Model Explainability Report

**Model explained:** Tuned LightGBM, Window 2 (later holdout)  
**Explanation method:** Native LightGBM TreeSHAP contributions  
**Scope:** 40,282 open-store validation observations and 33 fitted model features  
**Implementation:** `src/evaluation/explain_lightgbm_model.py`  
**Evidence:** committed SHAP result files under `src/evaluation/results/`

## 1. Purpose and scope

This report documents the explainability work actually implemented for the selected tuned LightGBM model. It covers:

- global feature attribution across the Window 2 open-store validation population;
- local feature attribution for three selected validation cases;
- numerical validation that the SHAP contributions reconcile to the model's raw predictions;
- interpretation, appropriate business use, and limitations.

The explanations are generated from the fitted Window 2 tuned LightGBM model and the feature matrix captured during recursive validation. They are not explanations of a separately retrained model or of the competition's unseen test predictions.

The underlying validation design is documented in [the model evaluation report](model_evaluation_report.md). Window 2 covers 20 June–31 July 2015, a 42-day recursive horizon. The complete holdout has 46,830 store-date rows; 40,282 are open-store observations. SHAP is calculated for those 40,282 open-store observations.

## 2. Implementation and method

### 2.1 Reproducible implementation

The implementation is in `src/evaluation/explain_lightgbm_model.py`. It:

1. loads `second_window_tuned_lightgbm_model.joblib` from `models/experiments/lightgbm_second_window/`;
2. loads the captured recursive explanation feature data from `recursive_explanation_features.joblib`;
3. obtains the fitted model's feature names and selects those columns, in model order;
4. calculates native LightGBM contribution values with `booster.predict(X, pred_contrib=True)`;
5. separately obtains raw model predictions with `booster.predict(X, raw_score=True)`;
6. validates SHAP additivity;
7. writes global feature importance, local contributions, selected case metadata and validation metadata to `src/evaluation/results/`.

The script explicitly checks that the fitted model has 33 features and fails if the expected feature count is not met. It also checks for rows where every model feature is missing.

### 2.2 TreeSHAP contributions

For observation (i) and feature (j), let (phi_{ij}) denote the SHAP contribution of feature (j), and let (phi_{i0}) denote the base value for that observation. The implementation uses LightGBM's native TreeSHAP contribution calculation.

The local additive explanation is:

\[
f(x_i) = \phi_{i0} + \sum_{j=1}^{p}\phi_{ij}
\]

where:

- (f(x_i)) is the model's raw prediction for observation (i);
- (phi_{i0}) is the base value;
- (phi_{ij}) is the contribution assigned to feature (j);
- (p=33) is the number of fitted features.

A positive SHAP value moves the model output above the base value; a negative value moves it below the base value. The magnitude indicates the size of that contribution in the model-output scale. For this sales regression model, the contributions are interpreted in predicted-sales units.

SHAP contributions are an additive attribution of the fitted model's output. They are not coefficients, standalone feature effects, or causal effects.

### 2.3 Global importance definitions

The global results file contains four model-importance fields alongside each feature:

| Field | Definition | Interpretation |
|---|---|---|
| `MeanAbsSHAP` | Mean of the absolute SHAP contribution for that feature across the explanation observations | Average magnitude of attribution; used to rank global importance |
| `MeanSHAP` | Mean signed SHAP contribution across the observations | Average direction of attribution over this population; positive and negative cases can cancel |
| `GainImportance` | LightGBM split gain importance | Training-model measure of the total gain attributed to splits using the feature |
| `SplitImportance` | Number of splits using the feature | Frequency of use in the fitted trees |

Mean absolute SHAP is the primary global ranking in this report. Gain and split importance are retained as complementary model diagnostics; they are not interchangeable with SHAP.

## 3. Global feature attribution

The following are the ten leading features ranked by mean absolute SHAP in the committed `shap_global_importance.csv`.

| Rank | Feature | Mean absolute SHAP | Mean signed SHAP |
|---:|---|---:|---:|
| 1 | Store_Weekday_HistoricalMean | 1,123.6721 | +100.3530 |
| 2 | Promo | 829.9783 | -18.4236 |
| 3 | Sales_Lag_1 | 317.5855 | -11.5766 |
| 4 | Day | 250.3212 | +33.9049 |
| 5 | Sales_Lag_14 | 224.7545 | +30.1733 |
| 6 | Sales_RollingMean_28 | 181.8590 | +45.1722 |
| 7 | Store | 136.3896 | -22.5092 |
| 8 | Sales_RollingMean_7 | 114.0743 | +36.1585 |
| 9 | Sales_Lag_28 | 103.4892 | +27.9774 |
| 10 | DayOfWeek_sin | 95.1932 | -13.5047 |

Values are in model-output / predicted-sales units. The displayed values are rounded from the committed CSV; the CSV remains the source of full precision.

### 3.1 Interpretation

**Store- and weekday-specific historical demand is the leading attribution.** `Store_Weekday_HistoricalMean` has the largest mean absolute SHAP value (1,123.6721). This is consistent with substantial variation in demand patterns by store and day of week. It does not mean that changing this feature by one unit would cause sales to change by the reported amount.

**Promotion status is the second-largest attribution.** `Promo` has mean absolute SHAP of 829.9783. Its signed mean is slightly negative (-18.4236), despite its large absolute contribution. This is a useful example of why mean signed SHAP is not a feature-importance ranking: positive and negative local contributions can offset one another when averaged.

**Recent sales history is important.** `Sales_Lag_1`, `Sales_Lag_14`, `Sales_RollingMean_28`, `Sales_RollingMean_7`, and `Sales_Lag_28` all appear among the leading features. The model relies materially on recent and recurring historical sales patterns, alongside store and calendar context.

**Calendar features contribute to the forecasts.** `Day` and `DayOfWeek_sin` appear in the top ten. The model can use calendar position and cyclical weekday representation in combination with store-level historical features.

These rankings describe attribution over the Window 2 explanation population. They should not be interpreted as universal rankings for every store, forecast horizon, season, or future model version.

## 4. Local explanations

Three cases are selected by the implementation from the Window 2 open-store observations:

- **HighestPrediction:** observation with the highest predicted sales;
- **LargestOverforecast:** observation with the largest positive prediction error, where error is prediction minus actual;
- **LargestUnderforecast:** observation with the most negative prediction error.

If two selection rules identify the same observation, the script avoids selecting that row twice. The selected case metadata and all feature contributions are stored in `shap_explanation_cases.csv` and `shap_local_explanations.csv`.

### 4.1 Case outcomes

| Case | Date | Store | Actual sales | Predicted sales | Error (predicted − actual) | Base value |
|---|---|---:|---:|---:|---:|---:|
| HighestPrediction | 2015-07-05 | 262 | 30,255.00 | 29,624.97 | -630.03 | 6,954.18 |
| LargestOverforecast | 2015-07-01 | 909 | 3,547.00 | 16,176.79 | +12,629.79 | 6,954.18 |
| LargestUnderforecast | 2015-06-22 | 909 | 41,551.00 | 9,089.94 | -32,461.06 | 6,954.18 |

The error sign convention is explicit: positive means overforecast; negative means underforecast. The base value is the model's TreeSHAP baseline contribution for the selected observations. It is not the actual sales, a store average, or a separate benchmark forecast.

### 4.2 Highest-prediction case — Store 262, 5 July 2015

The model predicted 29,624.97 against actual sales of 30,255, an underforecast of 630.03.

| Leading feature contribution | Feature value | SHAP contribution |
|---|---:|---:|
| Store_Weekday_HistoricalMean | 28,358.4953 | +9,119.53 |
| DayOfWeek_cos | 1.0000 | +3,259.76 |
| Sales_Lag_1 | 18,003.4449 | +2,021.10 |
| Sales_Lag_14 | 29,053.0253 | +1,921.75 |
| Store | 262 | +1,875.25 |
| Sales_RollingMean_28 | 20,798.9461 | +1,208.36 |
| Sales_RollingMean_14 | 21,311.1773 | +994.06 |
| Sales_RollingMean_7 | 23,253.1183 | +819.29 |
| Sales_Lag_7 | 28,949.3613 | +745.05 |
| Promo | 0 | -546.60 |

The leading positive contributions are the store-weekday historical mean, weekday-cycle value, and recent sales lags and rolling means. The non-promotion value contributes negatively in this case. The model's final prediction is the base value plus all 33 feature contributions, not just the subset displayed here.

### 4.3 Largest-overforecast case — Store 909, 1 July 2015

The model predicted 16,176.79 against actual sales of 3,547, an overforecast of 12,629.79.

| Leading feature contribution | Feature value | SHAP contribution |
|---|---:|---:|
| Store_Weekday_HistoricalMean | 10,208.2181 | +2,298.48 |
| Sales_Lag_1 | 17,327.8803 | +1,903.03 |
| Promo | 1 | +1,475.95 |
| Day | 1 | +735.03 |
| Sales_Lag_14 | 12,244.0000 | +536.49 |
| Year | 2015 | +383.92 |
| Sales_RollingMean_28 | 9,049.4225 | +379.34 |
| Sales_RollingMean_14 | 9,327.8450 | +307.48 |
| DayOfWeek | 3 | +289.92 |
| Sales_Lag_28 | 13,209.0000 | +287.42 |

The selected row has a high prior-day sales value and an active daily promotion; both receive positive attribution. The store-weekday historical mean and other lag/rolling features also push the prediction upward. The actual outcome is much lower than the model prediction. This is a concrete diagnostic example of how a model can over-rely on recent history and promotion context when the realised sales outcome differs sharply from those signals. The SHAP output explains the model's prediction; it does not establish why actual sales were low.

### 4.4 Largest-underforecast case — Store 909, 22 June 2015

The model predicted 9,089.94 against actual sales of 41,551, an underforecast of 32,461.06.

| Leading feature contribution | Feature value | SHAP contribution |
|---|---:|---:|
| Store_Weekday_HistoricalMean | 11,568.9608 | +2,611.72 |
| Promo | 0 | -1,399.59 |
| Sales_RollingMean_7 | 10,676.3661 | +523.41 |
| Sales_Lag_1 | 0 | +436.65 |
| DayOfWeek_sin | 0.7818 | -334.79 |
| Sales_RollingMean_28 | 8,825.2344 | +316.01 |
| Day | 22 | -266.18 |
| Sales_RollingMean_14 | 9,233.9688 | +236.08 |
| Sales_Lag_14 | 9,352.0000 | +224.56 |
| DayOfWeek | 1 | -184.36 |

The store-weekday historical mean and rolling sales history move the prediction upward, while the non-promotion value and several calendar contributions move it downward. The observed actual sales are far above the prediction. This highlights a large residual case for investigation, but the attribution alone cannot identify an unobserved event or prove the cause of the sales spike.

## 5. Additivity and numerical validation

The implementation validates the TreeSHAP additive identity against LightGBM's raw model predictions for every explanation observation.

For each row (i), it calculates:

\[
\text{reconstructed}_i = \phi_{i0} + \sum_{j=1}^{33}\phi_{ij}
\]

\[
\text{additivity error}_i =
\left|\text{reconstructed}_i - f_{\text{raw}}(x_i)\right|
\]

The maximum additivity error is then:

\[
\max_i\left(\text{additivity error}_i\right)
\]

**Committed validation result:**

| Validation item | Result |
|---|---:|
| Explanation observations | 40,282 |
| Model features | 33 |
| Method | LightGBM native TreeSHAP contributions |
| Maximum additivity error | 1.2369127944111824 × 10⁻¹⁰ |
| Threshold in implementation | 1 × 10⁻⁵ |
| Validation status | PASS |
| Local cases exported | 3 |

The maximum observed error is substantially below the implementation's failure threshold of (10^{-5}). The script raises an error if the threshold is exceeded. This verifies numerical additivity of the exported contribution values against the model's raw predictions; it does not validate predictive accuracy, data quality, causal validity, or generalisation.

The validation metadata is persisted in `src/evaluation/results/shap_contribution_validation.json`.

## 6. How to use these explanations

The current explainability outputs can support:

1. **Model review:** understand which feature groups contribute most across the evaluated population.
2. **Forecast review:** inspect the strongest positive and negative attributions for a specific prediction.
3. **Error investigation:** prioritise large overforecast and underforecast cases for operational review.
4. **Feature governance:** compare attribution patterns across later model versions or monitoring windows, provided the same explanation population and calculation approach are used.
5. **Stakeholder communication:** explain that store-specific weekday history, promotion status, and recent sales history are prominent inputs to this fitted model.

For operational investigation, local explanations should be considered together with the underlying store calendar, promotion execution, availability/stockout information, local events and data-quality checks where those records are available. Those external facts are not inferred by SHAP.

## 7. Important interpretation limits

- **Attribution is not causation.** A positive Promo contribution does not prove that a promotion caused a given amount of incremental sales. Features can be correlated and the model learns predictive associations.
- **Global rankings are population-dependent.** The rankings are calculated on the Window 2 open-store explanation observations. They can change across periods, store mixes, retraining runs and model versions.
- **Local contributions are conditional on this fitted model and feature representation.** They do not describe what would happen under a real-world intervention that changes one feature while holding everything else fixed.
- **Correlated features require care.** Historical lags, rolling means and historical averages contain related demand information. Attribution among related predictors should not be read as a clean separation of independent business effects.
- **Three local cases are diagnostic examples, not a representative sample.** They were selected by extreme prediction/error rules and should not be used to estimate typical model behaviour.
- **No causal or counterfactual analysis is implemented here.** The current code produces additive feature contributions; it does not calculate intervention effects, confidence intervals for SHAP values, or causal impact.
- **No SHAP explanation is a forecast guarantee.** Additivity passing confirms the arithmetic decomposition, not that the prediction is correct.
- **Explanations cover open-store validation rows.** Closed-store rows are assigned zero forecasts by the forecasting workflow and are not included in this SHAP explanation population.

## 8. Repository artefacts and reproduction

| Artefact | Purpose |
|---|---|
| `src/evaluation/explain_lightgbm_model.py` | Source implementation for global/local TreeSHAP and additivity validation |
| `src/evaluation/results/shap_global_importance.csv` | Global feature attribution and LightGBM gain/split importance |
| `src/evaluation/results/shap_local_explanations.csv` | Feature values and signed SHAP contributions for each selected case |
| `src/evaluation/results/shap_explanation_cases.csv` | Actual, predicted, error and base-value metadata for the selected cases |
| `src/evaluation/results/shap_contribution_validation.json` | Observation/feature counts, method, maximum additivity error and validation status |

Run the explanation script from the repository environment after the Window 2 tuned model and recursive explanation feature artifact have been generated:

```bash
python src/evaluation/explain_lightgbm_model.py
```

The script expects the model and captured feature artifact at the paths defined near the top of the source file. It writes the four result files to `src/evaluation/results/`. Reproduction therefore depends on the corresponding model and feature artifacts being available locally; the script does not train the model or rebuild the recursive validation feature capture.

## 9. Summary

The repository contains an implemented and numerically validated explainability workflow for the selected Window 2 tuned LightGBM model. Across 40,282 open-store observations and 33 features, store-weekday historical demand, promotion status, and recent sales history are the leading global attribution drivers. Three local cases demonstrate how the same fitted model can produce a near-accurate high prediction, a material overforecast, and a material underforecast for different feature contexts.

The maximum TreeSHAP additivity error is (1.2369127944111824 \times 10^{-10}), and the committed validation status is **PASS**. These results improve model transparency and support error investigation, while remaining strictly model-attribution evidence—not causal explanation or proof of business impact.
