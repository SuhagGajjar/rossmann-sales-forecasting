
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = ROOT / "data/processed/validation_base.pkl"
PREDICTION_PATH = ROOT / "data/processed/validation_random_forest_predictions.csv"
RESULTS_PATH = ROOT / "reports/random_forest_results.csv"


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WMAPE": np.abs(actual - predicted).sum() / np.abs(actual).sum(),
        "RMSLE": np.sqrt(
            np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)
        ),
    }


def main():
    validation = pd.read_pickle(VALIDATION_PATH)
    predictions = pd.read_csv(PREDICTION_PATH, parse_dates=["Date"])

    validation["Date"] = pd.to_datetime(validation["Date"])

    evaluation = validation[
        ["Store", "Date", "Sales", "Open"]
    ].merge(
        predictions[["Store", "Date", "PredictedSales"]],
        on=["Store", "Date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    assert evaluation["_merge"].eq("both").all()
    assert len(evaluation) == len(validation)
    assert np.isfinite(evaluation["PredictedSales"]).all()

    evaluation = evaluation.drop(columns="_merge")

    overall = metrics(evaluation["Sales"], evaluation["PredictedSales"])
    open_rows = evaluation[evaluation["Open"].eq(1)]
    open_metrics = metrics(open_rows["Sales"], open_rows["PredictedSales"])

    results = pd.DataFrame([
        {
            "Model": "RandomForest",
            "TrainingRows": 804110,
            "EvaluationScope": "All store-days",
            "ValidationRows": len(evaluation),
            "StoreDateAlignment": "PASS",
            **overall,
        },
        {
            "Model": "RandomForest",
            "TrainingRows": 804110,
            "EvaluationScope": "Open store-days only",
            "ValidationRows": len(open_rows),
            "StoreDateAlignment": "PASS",
            **open_metrics,
        },
    ])

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    print("Store-Date alignment: PASS")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
