# Executive Summary — Rossmann Store Sales Forecasting

## Recommendation

**Advance tuned LightGBM as the preferred candidate for production-readiness validation.** Across two chronological 42-day evaluation windows, it outperformed the tested seasonal-naive, moving-average, naive, Random Forest and untuned LightGBM alternatives on the reported core metrics. It is not yet a production-certified system.

## Business problem

Forecast daily sales for 1,115 Rossmann stores over a six-week horizon. Reliable store-level forecasts can support replenishment planning, inventory availability, promotion planning and working-capital efficiency. Underforecasting risks stockouts and lost sales; overforecasting risks excess inventory, holding costs and waste.

## What was built

A global LightGBM forecasting approach uses store attributes, calendar patterns, promotions, holidays, competition information and historical sales. A leakage-safe recursive feature builder generates forecasts across the 42-day horizon, feeding predictions—not actual future sales—back into subsequent lag and rolling features. Closed stores receive zero predicted sales; model training uses open-store observations.

## Results

| Evaluation window | Selected model | MAE | RMSE | WMAPE | RMSLE |
|---|---|---:|---:|---:|---:|
| 9 May–19 June 2015 (selection) | Tuned LightGBM | 455.60 | 708.49 | **7.8206%** | **0.1085** |
| 20 June–31 July 2015 (later holdout) | Tuned LightGBM | 514.10 | 807.69 | **8.5643%** | **0.1077** |

In the later holdout, tuned LightGBM improved WMAPE over Random Forest (9.2387%) and untuned LightGBM (8.7480%). The simple seasonal-naive baseline was materially weaker in the model-selection window (37.23% WMAPE). Window 1 informed model selection; Window 2 used the frozen selected hyperparameters and is the primary later-period evidence.

## Operational findings

- Forecast error varies substantially by store. A few stores have WMAPE above 35%, often with limited open-day coverage; store-level monitoring is essential.
- Low-volume store-days have higher proportional error (10.74% WMAPE) than high-volume store-days (7.55%).
- Daily promotion-active observations have higher WMAPE (9.06%) than inactive observations (8.06%). The combined daily Promo + Promo2-active segment has 10.02% WMAPE.
- Aggregate bias is positive: 61.51% of open-store predictions are overforecasts, versus 38.49% underforecasts. Mean bias is +204.51 sales units per open store-day; net total bias is approximately +8.24 million (2.93% of actual sales).
- TreeSHAP indicates store-weekday historical mean, daily Promo and recent sales lags are the strongest global contributors. Additivity validation passed.

## Delivery and controls

The repository contains modular preprocessing, feature engineering, recursive forecasting, model training, tuning, evaluation and explainability code, alongside saved metrics, predictions and diagnostics. Before deployment, add automated data contracts, reproducible scheduled training/inference, model/version registry, drift and accuracy monitoring, store/promotion alerting, business-cost calibration, uncertainty estimates and shadow rollout.

## Caveats

The evidence is based on two 42-day periods, not a full annual cycle. Window 1 was used for tuning; Window 2 is a later holdout but not a fully untouched external test. Forecasts are point estimates. Weather, stockouts, local events and detailed promotion mechanics are not fully represented. Segment comparisons and SHAP attributions are descriptive, not causal.

**Decision:** proceed with tuned LightGBM to a controlled production validation stage, with store-level and promotion-aware monitoring and explicit review of positive forecast bias.
