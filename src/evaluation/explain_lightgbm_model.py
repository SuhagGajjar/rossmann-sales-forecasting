
from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = (
    ROOT / "models" / "experiments" / "lightgbm_second_window"
)

RESULT_DIR = ROOT / "src" / "evaluation" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading model and recursive explanation features...")

    model = joblib.load(
        ARTIFACT_DIR / "second_window_tuned_lightgbm_model.joblib"
    )

    captured = joblib.load(
        ARTIFACT_DIR / "recursive_explanation_features.joblib"
    )

    feature_columns = model.feature_name_

    X = captured[feature_columns].copy()

    print(f"Explanation observations: {len(X):,}")
    print(f"Model features: {len(feature_columns)}")

    if len(feature_columns) != 33:
        raise ValueError("Expected 33 model features.")

    if X.isna().all(axis=1).any():
        raise ValueError("Found rows with all model features missing.")

    print("\nCalculating native LightGBM TreeSHAP contributions...")

    booster = model.booster_

    contributions = booster.predict(
        X,
        pred_contrib=True,
    )

    raw_predictions = booster.predict(
        X,
        raw_score=True,
    )

    shap_values = contributions[:, :-1]
    base_values = contributions[:, -1]

    # Verify SHAP additivity against the model's raw predictions.
    reconstructed = shap_values.sum(axis=1) + base_values

    max_additivity_error = float(
        np.max(np.abs(reconstructed - raw_predictions))
    )

    print(f"Maximum SHAP additivity error: {max_additivity_error:.12f}")

    if max_additivity_error > 1e-5:
        raise ValueError("SHAP additivity validation failed.")

    print("SHAP additivity validation: PASS")

    # Global feature importance.
    global_importance = pd.DataFrame({
        "Feature": feature_columns,
        "MeanAbsSHAP": np.mean(np.abs(shap_values), axis=0),
        "MeanSHAP": np.mean(shap_values, axis=0),
        "GainImportance": booster.feature_importance(
            importance_type="gain"
        ),
        "SplitImportance": booster.feature_importance(
            importance_type="split"
        ),
    })

    global_importance = global_importance.sort_values(
        "MeanAbsSHAP",
        ascending=False,
    )

    global_importance.to_csv(
        RESULT_DIR / "shap_global_importance.csv",
        index=False,
    )

    print("\nTop 15 features by mean absolute SHAP:")
    print(global_importance.head(15).to_string(index=False))

    # Load actual and predicted validation results for local cases.
    predictions = pd.read_csv(
        ARTIFACT_DIR / "second_window_tuned_lightgbm_predictions.csv"
    )

    captured_metadata = captured[["Date", "Store"]].copy()
    captured_metadata["Date"] = pd.to_datetime(
        captured_metadata["Date"]
    )
    captured_metadata["Store"] = captured_metadata["Store"].astype(int)

    predictions["Date"] = pd.to_datetime(predictions["Date"])
    predictions["Store"] = predictions["Store"].astype(int)

    cases = captured_metadata.merge(
        predictions[["Date", "Store", "Sales", "PredictedSales"]],
        on=["Date", "Store"],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if cases[["Sales", "PredictedSales"]].isna().any().any():
        raise ValueError("Missing actual or predicted values after alignment.")

    cases["Error"] = cases["PredictedSales"] - cases["Sales"]

    # Select three representative cases.
    selected = []

    candidates = [
        ("HighestPrediction", cases["PredictedSales"].idxmax()),
        ("LargestOverforecast", cases["Error"].idxmax()),
        ("LargestUnderforecast", cases["Error"].idxmin()),
    ]

    used_indices = set()

    for label, idx in candidates:
        if idx not in used_indices:
            selected.append((label, idx))
            used_indices.add(idx)

    local_records = []
    case_records = []

    for label, idx in selected:
        row = cases.loc[idx]

        case_records.append({
            "Case": label,
            "Date": row["Date"],
            "Store": row["Store"],
            "ActualSales": row["Sales"],
            "PredictedSales": row["PredictedSales"],
            "Error": row["Error"],
            "BaseValue": float(base_values[idx]),
        })

        contributions_for_case = pd.DataFrame({
            "Feature": feature_columns,
            "FeatureValue": X.iloc[idx].astype(str).to_numpy(),
            "SHAPValue": shap_values[idx],
        })

        contributions_for_case["AbsoluteSHAP"] = (
            contributions_for_case["SHAPValue"].abs()
        )

        contributions_for_case = contributions_for_case.sort_values(
            "AbsoluteSHAP",
            ascending=False,
        )

        for _, feature_row in contributions_for_case.iterrows():
            local_records.append({
                "Case": label,
                "Date": row["Date"],
                "Store": row["Store"],
                "ActualSales": row["Sales"],
                "PredictedSales": row["PredictedSales"],
                "Error": row["Error"],
                "Feature": feature_row["Feature"],
                "FeatureValue": feature_row["FeatureValue"],
                "SHAPValue": feature_row["SHAPValue"],
                "AbsoluteSHAP": feature_row["AbsoluteSHAP"],
            })

    pd.DataFrame(case_records).to_csv(
        RESULT_DIR / "shap_explanation_cases.csv",
        index=False,
    )

    pd.DataFrame(local_records).to_csv(
        RESULT_DIR / "shap_local_explanations.csv",
        index=False,
    )

    validation = {
        "observations": int(len(X)),
        "features": int(len(feature_columns)),
        "shap_method": "LightGBM native TreeSHAP contributions",
        "max_additivity_error": max_additivity_error,
        "additivity_validation": "PASS",
        "local_cases": len(selected),
    }

    with open(
        RESULT_DIR / "shap_contribution_validation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(validation, file, indent=4)

    print("\nLocal explanation cases:")
    print(pd.DataFrame(case_records).to_string(index=False))

    print("\nSHAP outputs saved to:")
    print(RESULT_DIR)

    print("\nMODEL EXPLAINABILITY GENERATION COMPLETE")


if __name__ == "__main__":
    main()
