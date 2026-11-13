# Executive Summary — Rossmann Store Sales Forecasting

**For:** Chief Commercial Officer  
**Decision requested:** Approve a controlled pilot of the tuned LightGBM sales-forecasting model, while using the findings to shape the next commercial analytics deep-dive—not a full network rollout.

## Recommendation

Select **tuned LightGBM** as the preferred model for forecasting daily sales across the 1,115 stores over the next 42 days. It was the strongest of the models tested in both evaluation periods and performed substantially better than the simple forecasting approach used as the comparison point.

In the first 42-day test, the model reduced the gap between forecast and actual sales from **37.23% to 7.82%**—a **79.0% reduction in forecast error**. In a later 42-day test, it achieved **8.56% forecast error**, ahead of the other machine-learning models tested. This later result is the more relevant check that performance held up on a subsequent period.

## What improved

The work established a materially stronger store-level demand signal:

- **Overall forecast error:** reduced from 37.23% to 7.82% in the first test period (79.0% lower). This compares the total size of forecast misses with total actual sales.
- **Typical size of a daily miss:** reduced from about 2,169 to 456 sales units per store-day (79.0% lower).
- **Impact of large misses:** the measure that gives greater weight to bigger errors fell from about 3,745 to 708 sales units (81.1% lower).

This is a meaningful forecasting improvement and a foundation for better commercial and operational decisions. The analysis also explored historical customer traffic and promotion patterns, providing additional signals about where deeper investigation may create value.

## Where it performs well—and what the analysis reveals

- **Higher-sales days:** forecast error was 7.55%, the strongest result across the sales-volume groups examined.
- **Lower-sales days:** error was higher at 10.74%. These forecasts need closer review, especially where small absolute misses represent a large share of sales.
- **Promotions:** forecast error was 9.06% on days with a daily promotion, compared with 8.06% on days without one. When both daily and longer-running promotions were active, error rose to 10.02%.
- **Differences between stores:** results varied considerably. A few stores had much higher errors than the network average and should be reviewed individually. StoreType B showed 6.51% error, but it covered only 17 stores, so it is not enough evidence to assume the same result across a wider group.
- **Forecast bias:** there is a tendency to forecast above actual sales. In the later test, **61.5% of open-store forecasts were higher than actual sales**. Across those store-days, forecasts were on average about **205 sales units too high per day**, and total forecast sales were about **2.93% above actual sales**. This should be considered when translating forecasts into replenishment decisions.

The customer and promotion exploratory analysis adds useful commercial context:

- **Customer traffic and sales move together:** the correlation between daily customer counts and sales was **0.824** across 844,392 open-store observations (and 0.830 at store level across 1,115 stores). Customer traffic is therefore an important lens for understanding store demand, although correlation alone does not establish causation.
- **Store formats behave differently:** StoreType B had the highest average traffic at **2,021.8 customers per day** (17 stores), while StoreType D had the highest weighted sales per customer at **11.25**. StoreType B had the lowest at **5.06**. This points to differences in traffic and customer spend across store formats that merit more granular investigation.
- **Daily promotions coincide with higher activity:** promotion days had **21.2% more customers**, **38.8% higher sales**, and **14.5% higher weighted sales per customer** than non-promotion days in the descriptive analysis. This is an observed association, not a causal estimate of promotion lift.
- **Longer-running promotions need closer examination:** among stores participating in Promo2, active periods showed approximately **2.0% lower customers and 1.9% lower sales** than inactive periods in the descriptive comparison. Store mix, timing and seasonality may influence this result; it should not be interpreted as evidence that Promo2 reduces demand.

Customer counts were available in the historical training data but not in the competition test data, so they were used for exploratory analysis and not as an input to the deployed forecasting feature set. These findings are not an endpoint: they identify questions that richer customer and promotion data can help answer.

## The next opportunity: turn better demand insight into commercial value

The forecasting work and exploratory findings create a platform for a broader, joined-up commercial deep-dive. The next step is to connect the store-level demand signal with the operational and customer data that influence how stores serve that demand.

1. **Customer understanding and segmentation:** enrich the current store-day customer counts with relevant customer attributes such as customer segments, loyalty status, visit frequency, purchase history, basket composition and product preferences. This can help explain *who is visiting, what they buy, how much they spend and how behaviour differs by store format or trading occasion*. It can support more relevant assortment, targeted engagement and demand planning.
2. **Promotion effectiveness:** add granular promotion attributes—offer type, discount depth, eligible products, campaign duration, customer exposure and redemption, where available. This would allow the business to investigate which customer groups and products respond to which promotions, distinguish traffic growth from increased basket value, and assess incremental effectiveness rather than relying only on broad Promo/Promo2 flags.
3. **Inventory and replenishment optimisation:** connect forecasts to SKU-level inventory-on-hand, stock movements, deliveries, supplier lead times, replenishment orders, stockouts and waste or expiry data. This can help assess whether demand is being met, identify availability gaps, set more appropriate replenishment quantities and balance service levels against excess stock and holding costs.
4. **In-store workforce planning:** combine expected store demand with staffing rosters, shift hours, employee roles, labour costs and store activity measures. This can help investigate how staffing levels and shift allocation could better match expected customer traffic and operational workload, while monitoring service and labour-efficiency outcomes.

Together, these data streams can move the organisation from **forecasting store sales more accurately** towards **understanding demand drivers and optimising the decisions around customers, promotions, inventory and people**. The commercial value—such as improved availability, reduced waste, more effective promotions, better labour utilisation and stronger margin—must be quantified through those deeper analyses and a measured business pilot; it has not been claimed as an outcome of this forecasting exercise.

## Main risks and considerations

1. **Forecasts are not orders:** the model's tendency to forecast above actual sales means replenishment decisions need product-level stock rules, lead times, shelf life, margin and service targets.
2. **Promotion effects are not yet explained:** forecast errors were higher during promotion periods, and the exploratory comparisons are descriptive. More detailed campaign and customer data are needed to understand promotion response and improve decisions.
3. **Uneven store performance:** network averages can hide stores where forecasts are less reliable. Store-level exceptions need to be visible to planners.
4. **Limited test coverage:** the evaluation covers two consecutive 42-day test periods, not a full year of seasonal trading. Performance across all seasons is not yet established.
5. **Operational value depends on integration:** the forecast becomes more actionable when connected to inventory, replenishment and workforce information, with appropriate decision rules and ownership.

## What to do next

1. **Run a controlled forecasting pilot:** initially show forecasts alongside existing planning, without allowing them to trigger orders automatically. Track forecast accuracy, bias and store-level exceptions.
2. **Scope the commercial data deep-dive:** assess availability and quality of customer, promotion, SKU-level inventory, replenishment and workforce data. Prioritise the analyses that can most directly inform commercial decisions.
3. **Define outcome measures:** agree how to measure availability, lost sales where measurable, waste, inventory, promotion effectiveness, labour utilisation, customer experience and gross margin. Set baselines and success criteria before testing interventions.
4. **Develop decision support with business teams:** use the combined analysis to identify where assortment, targeted promotions, replenishment policies or staffing schedules could be improved. Keep human review and operational constraints in the decision process.
5. **Expand model validation:** evaluate additional trading periods, especially major seasonal and holiday periods, and monitor performance as actual sales arrive.

**Bottom line:** tuned LightGBM is the strongest forecasting option tested and cut forecast error by 79.0% against the simple baseline in the first test. The customer and promotion EDA also reveals meaningful differences in traffic, spend and observed promotion patterns across stores and trading conditions. Together, these findings provide a stronger basis for a controlled forecasting pilot and a focused next-stage deep-dive using richer customer, promotion, inventory and workforce data—to identify and quantify how better demand insight can translate into commercial value.
