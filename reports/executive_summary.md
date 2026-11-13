# Executive Summary — Rossmann Store Sales Forecasting

**To:** Chief Commercial Officer  
**Decision requested:** Approve tuned LightGBM for a controlled production-validation pilot, not an immediate network-wide autonomous rollout.

## Recommendation

Select **tuned LightGBM** to produce daily store sales forecasts for the next 42 days. It is the strongest candidate tested and is materially better than the simple seasonal baseline. In the first validation period, it reduced WMAPE from **37.23% to 7.82% — a 79.0% reduction in forecast error**. It also performed best among the tested machine-learning candidates in the later validation period, with **8.56% WMAPE**.

## What improved

The model-selection comparison covered the same 42-day period and store-date population:

| Forecast-quality measure | Seasonal baseline | Tuned LightGBM | Improvement |
|---|---:|---:|---:|
| WMAPE (lower is better) | 37.23% | **7.82%** | **79.0% lower error** |
| MAE (sales units; lower is better) | 2,168.95 | **455.60** | **79.0% lower error** |
| RMSE (sales units; lower is better) | 3,745.19 | **708.49** | **81.1% lower error** |

These are improvements in forecast accuracy, not measured improvements in stock availability, waste, sales, margin or working capital. Those commercial KPIs should be measured during the pilot before claiming financial benefit.

In the later 42-day holdout, tuned LightGBM achieved 8.56% WMAPE, compared with 8.75% for untuned LightGBM and 9.24% for Random Forest. The selected model therefore retained its lead on a later period using frozen tuning choices.

## Where it works well — and where it needs attention

The model performs consistently across the network, but results are not uniform:

- **Relatively strong:** high-sales store-days had 7.55% WMAPE. StoreType B had 6.51% pooled WMAPE, although this result covers only 17 stores and should be treated cautiously.
- **Needs attention:** low-sales store-days had 10.74% WMAPE. Promotion-active days had 9.06%, compared with 8.06% when daily promotions were inactive. Days with both promotion programmes active reached 10.02%.
- **Store-specific exceptions:** some stores had much higher error than the network average; a few worst cases exceeded 35–46% WMAPE, with limited validation observations for some stores. These stores need individual review rather than a blanket network-wide assumption.

The model also tends to forecast high: 61.5% of open-store predictions were above actual sales in the later holdout. Average bias was +204.51 sales units per open store-day, equivalent to a net positive difference of approximately 2.93% of actual sales across that period. This could lead to excess stock if forecasts are translated directly into replenishment orders.

## Main risks and actions before production

1. **Commercial cost is not yet optimised.** Forecast accuracy alone does not balance the cost of excess inventory against lost sales. Calibrate decisions using product margins, lead times, holding costs and service-level targets.
2. **Review high-risk segments.** Investigate low-volume stores, promotion periods and stores with persistent over- or underforecasting. Set store-level and promotion-level alert thresholds.
3. **Run a controlled pilot.** Generate forecasts in shadow mode alongside the current planning process. Compare forecast-driven recommendations with actual orders, availability, waste and sales outcomes before changing replenishment decisions.
4. **Establish operating controls.** Validate incoming data, monitor forecast error and bias as actual sales arrive, track model versions, define retraining triggers and provide a fallback when data checks fail.
5. **Broaden validation.** The evidence covers two 42-day periods, not a complete annual cycle. Test additional seasonal periods and operational conditions before a full rollout.

**Bottom line:** tuned LightGBM is the clear forecasting candidate from this evaluation and substantially reduces forecast error versus the seasonal baseline. Approve it for controlled production validation, with special attention to positive bias, promotion periods and store-level exceptions. Measure commercial KPIs in the pilot before scaling.
