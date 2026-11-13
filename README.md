# Rossmann Store Sales Forecasting

A reproducible, production-oriented forecasting case study for daily sales across 1,115 Rossmann stores over a 42-day horizon.

## Submission deliverables

1. **End-to-end notebook:** [notebooks/01_end_to_end_forecasting.ipynb](notebooks/01_end_to_end_forecasting.ipynb)
2. **Modular Python source:** `src/` (preprocessing, feature engineering, recursive forecasting, models, evaluation and explainability).
3. **Model evaluation report:** [reports/model_evaluation_report.md](reports/model_evaluation_report.md)
4. **Executive summary:** [reports/executive_summary.md](reports/executive_summary.md)
5. **Production architecture:** [architecture/production_architecture.md](architecture/production_architecture.md)

Supporting evidence (metrics, predictions, SHAP outputs, segment analysis and charts) is retained under `reports/`, `src/evaluation/results/`, `data/processed/` and `models/experiments/`.

## Project structure

- `data/raw/`: source competition files (not committed; obtain from the Rossmann Store Sales competition).
- `data/processed/`: derived datasets and validation predictions.
- `notebooks/`: analytical walkthrough.
- `src/preprocessing/`: validation, audit, data preparation and EDA.
- `src/features/`: leakage-safe feature engineering and recursive feature state.
- `src/models/`: baselines, Random Forest, LightGBM, tuning and recursive forecasting.
- `src/evaluation/`: model evaluation, segment analysis, explainability and saved diagnostics.
- `models/experiments/`: model artifacts, tuning trials, configurations and predictions.
- `reports/`: findings, metrics, charts and submission reports.
- `architecture/`: production architecture diagram.

## Environment

Python 3.10+ is recommended. Create an isolated environment and install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data setup

Download the Rossmann Store Sales competition files and place the original CSVs under `data/raw/` using the filenames expected by `src/features/build_features.py` and `src/preprocessing/prepare_data.py` (typically `train.csv`, `test.csv`, and `store.csv`). Raw competition data is intentionally not included in this repository.

## Analytical workflow

The source modules are the executable implementation. The notebook explains the end-to-end workflow and points to the corresponding modules and saved evidence. Run scripts from the repository root so package imports and relative paths resolve.

1. Data validation and audit: `src/preprocessing/data_validation.py`, `data_investigation.py`, `prepare_data.py`.
2. EDA: `src/preprocessing/eda.py`, `customer_eda.py`.
3. Feature relationships and leakage-safe features: `src/features/analyze_feature_relationships.py`, `build_features.py`, `recursive_forecasting.py`, `validate_recursive_features.py`.
4. Baselines and models: `src/models/run_baselines.py`, `train_random_forest.py`, `train_lightgbm.py`.
5. Hyperparameter optimisation: `src/models/tune_lightgbm.py`.
6. Later chronological holdout: `src/models/evaluate_tuned_lightgbm_second_window.py`.
7. Error analysis and explainability: scripts under `src/evaluation/`.

The experiment outputs already committed in this repository are the evidence used in the submission reports. Re-running training or tuning may overwrite experiment artifacts; preserve the committed results if reproducing individual stages.

## Evaluation design and selected model

Two chronological, recursive 42-day windows were evaluated:
- Window 1: train through 8 May 2015; validate 9 May–19 June 2015. Used for model comparison and Optuna tuning.
- Window 2: train through 19 June 2015; validate 20 June–31 July 2015. Frozen selected hyperparameters; no further tuning.

Tuned LightGBM is the preferred candidate. It achieved 7.8206% WMAPE in Window 1 and 8.5643% in Window 2. See the evaluation report for metrics, segment findings, error analysis, SHAP interpretation and caveats.

## Important interpretation

This is a forecasting assignment and a production-oriented prototype, not a deployed production service. Architecture components such as orchestration, registry, monitoring and downstream integrations are proposed production controls, not implemented services.
