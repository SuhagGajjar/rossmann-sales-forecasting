# Executive Summary — Rossmann Store Sales Forecasting

**For:** Chief Commercial Officer  
**Decision:** Approve tuned LightGBM for a controlled production-validation pilot.

## Recommendation

Use **tuned LightGBM** as the preferred model for daily sales forecasts across 1,115 stores over a 42-day horizon. In the first 42-day validation period, it reduced forecast error (WMAPE) from **37.23% for the seasonal baseline to 7.82% — a 79.0% reduction**. It also led the tested machine-learning candidates in the later validation period, achieving 8.56% WMAPE versus 8.75% for untuned LightGBM and 9.24% for Random Forest.

## What improves

In the model-selection period, tuned LightGBM reduced:
- **WMAPE:** 37.23% to 7.82% (79.0% lower).
- **Average absolute error (MAE):** 2,168.95 to 455.60 sales units (79.0% lower).
- **Large-error measure (RMSE):** 3,745.19 to 708.49 sales units (81.1% lower).

These are forecast-quality measures. The assignment did not measure changes in sales, margin, stock availability, waste or working capital; commercial benefits must be established in a pilot.

## Where it works — and where it struggles

The model is strongest proportionally on high-sales store-days (7.55% WMAPE). StoreType B recorded 6.51% pooled WMAPE, but this represents only 17 stores. Error is higher on low-sales days (10.74%), promotion-active days (9.06% versus 8.06% when daily promotions are inactive), and days when both promotion programmes are active (10.02%). A small number of individual stores also have much higher error than the network average and need local review.

There is a material overforecast tendency: 61.5% of open-store predictions were above actual sales in the later validation period. Average bias was +204.51 sales units per open store-day, with total predicted sales approximately 2.93% above actual sales. If used without adjustment, this could encourage excess inventory.

## Risks and actions before production

1. **Balance commercial costs:** calibrate replenishment decisions against product margin, holding cost, lead time, stockout cost and service-level targets; forecast accuracy alone does not make the ordering decision.
2. **Monitor exceptions:** track error and over/underforecast bias by store, sales volume and promotion status. Investigate persistent outliers and promotion periods.
3. **Pilot before scaling:** run forecasts in shadow mode alongside current planning. Compare recommendations with actual availability, waste, sales and inventory outcomes.
4. **Put controls in place:** validate incoming data, version models, monitor performance as actuals arrive, define retraining triggers and retain a fallback process.
5. **Broaden validation:** the evidence covers two 42-day periods, not a full annual cycle. Test additional seasons before network-wide rollout.

**Bottom line:** tuned LightGBM is the clear forecasting candidate and substantially outperforms the seasonal baseline. Advance it to controlled production validation, with explicit safeguards for positive bias and store/promotion exceptions. Do not claim commercial KPI uplift until the pilot measures it.
