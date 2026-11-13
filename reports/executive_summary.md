# Executive Summary — Rossmann Store Sales Forecasting

**For:** Chief Commercial Officer  
**Decision requested:** Approve a controlled pilot of the tuned LightGBM sales-forecasting model—not a full network rollout.

## Recommendation

Select **tuned LightGBM** as the preferred model for forecasting daily sales across the 1,115 stores over the next 42 days. It was the strongest of the models tested in both evaluation periods and performed substantially better than the simple forecasting approach currently used as the comparison point.

In the first 42-day test, the model reduced the gap between forecast and actual sales from **37.23% to 7.82%**—a **79.0% reduction in forecast error**. In a later 42-day test, it achieved **8.56% forecast error**, ahead of the other machine-learning models tested. This later result is the more relevant check that performance held up on a subsequent period.

## What improved

The benefits demonstrated so far are improvements in forecast accuracy—not proven improvements in commercial results.

- **Overall forecast error:** reduced from 37.23% to 7.82% in the first test period (79.0% lower). This compares the total size of forecast misses with total actual sales.
- **Typical size of a daily miss:** reduced from about 2,169 to 456 sales units per store-day (79.0% lower).
- **Impact of large misses:** the measure that gives greater weight to bigger errors fell from about 3,745 to 708 sales units (81.1% lower).

These are the forecasting measures that improved. The work did **not** measure an increase in sales or margin, fewer stockouts, less waste, lower inventory, or improved working capital. Those outcomes must be tested in a business pilot before any financial benefit is claimed.

## Where it performs well—and where attention is needed

- **Higher-sales days:** forecast error was 7.55%, the strongest result across the sales-volume groups examined.
- **Lower-sales days:** error was higher at 10.74%. These forecasts need closer review, especially where small absolute misses represent a large share of sales.
- **Promotions:** error was 9.06% on days with a daily promotion, compared with 8.06% on days without one. When both daily and longer-running promotions were active, error rose to 10.02%.
- **Differences between stores:** results varied considerably. A few stores had much higher errors than the network average and should be reviewed individually. One store grouping showed 6.51% error, but it covered only 17 stores, so it is not enough evidence to assume the same result across a wider group.

There is also a tendency to forecast above actual sales. In the later test, **61.5% of open-store forecasts were higher than actual sales**. Across those store-days, forecasts were on average about **205 sales units too high per day**, and total forecast sales were about **2.93% above actual sales**. If replenishment decisions follow these forecasts without checks, some stores could receive more stock than they need.

## Main risks

1. **Excess stock from overforecasting:** forecasts are not orders. Ordering rules must account for product shelf life, margin, holding costs and the cost of running out.
2. **Promotion uncertainty:** forecast errors were higher during promotion periods. Promotion plans and execution should be checked closely during the pilot.
3. **Uneven store performance:** network averages can hide stores where forecasts are unreliable. Store-level exceptions need to be visible to planners.
4. **Limited test coverage:** the evaluation covers two consecutive 42-day test periods, not a full year of seasonal trading. Performance across all seasons is not yet established.
5. **Benefits not yet proven:** better forecasts do not automatically translate into higher sales, better availability or less waste. Those outcomes depend on how the business uses the forecasts.

## What to do before production

1. **Run a controlled pilot:** initially show forecasts alongside existing planning, without allowing them to trigger orders automatically.
2. **Measure commercial outcomes:** compare stock availability, lost sales where measurable, waste, inventory levels and gross margin against the current process. Agree the measurement method and success targets before the pilot starts.
3. **Set safeguards for ordering:** use product-level stock rules, lead times, shelf life and service targets. Apply human review or limits where forecasts are unusually high or where the model has a history of errors.
4. **Give planners clear exception reports:** highlight stores with repeated over- or underforecasting, low-sales days and promotion periods.
5. **Check data and performance regularly:** validate incoming store, sales and promotion information; compare forecasts with actual sales as they arrive; assign owners for investigating problems and deciding when the model needs updating.
6. **Expand testing across seasons:** evaluate additional trading periods, especially major seasonal and holiday periods, before considering a wider rollout.

**Bottom line:** tuned LightGBM is the strongest forecasting option tested and cut forecast error by 79.0% against the simple baseline in the first test. The evidence supports moving to a controlled pilot—not immediate full deployment. The pilot should prove whether better forecasts translate into better availability, less waste and stronger commercial performance, while actively managing the model's tendency to forecast too high.
