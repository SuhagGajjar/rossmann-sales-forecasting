# Preprocessing Methodology and Findings

## 1. Outputs

- Training rows: 1,017,209
- Test rows: 41,088
- Store master rows: 1,115

Prepared files:
- data/processed/train_preprocessed.pkl
- data/processed/test_preprocessed.pkl
- data/processed/store_preprocessed.pkl

## 2. Store master integration

Store attributes were joined using Store as the key.
A many-to-one merge validation was applied to prevent
accidental row multiplication.

All training and test stores matched the store master.

## 3. Missing Open values

- Missing Open values in test: 11
- Imputation: historical store-specific weekday mode.
- Indicator retained: Open_was_missing.
- Method recorded in: Open_imputation_method.

Training Open values were not imputed.

Store 622's historical weekday pattern supports
Monday-Saturday opening and Sunday closure for the
11 affected test observations.

This is an inference, not confirmed ground truth.

## 4. Competition metadata

- Missing CompetitionDistance values were filled using
  the median across stores.
- CompetitionDistance_was_missing preserves the
  original missingness information.
- CompetitionOpenSinceMonth and Year were retained
  without invented values.
- CompetitionOpenSince_was_missing identifies
  incomplete metadata.

## 5. Promo2 metadata

Missing Promo2SinceWeek, Promo2SinceYear and
PromoInterval values for non-participating stores
are structural and were not artificially imputed.

Promo2_metadata_missing flags participating stores
with incomplete programme metadata.

## 6. Target preservation

- Sales was not imputed.
- Zero-sales observations were retained.
- High-sales observations were retained.
- Closed-store nonzero-sales rows: 0

No blanket outlier removal was performed.

## 7. Leakage safeguards

- Customers is retained for historical analysis but
  must not be used as a forecasting feature because
  it is unavailable in test.
- Id is retained for test prediction mapping but
  must not be used as a model feature.
- No future Sales values were used to impute test data.

## 8. Remaining work

Calendar, competition-age, Promo2 activity,
historical lag and rolling features will be handled
in the separate feature-engineering stage.

Time-aware validation and forecasting baselines
will be implemented after feature engineering.