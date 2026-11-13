# Rossmann Store Sales Forecasting

A reproducible forecasting case study for daily sales across 1,115 Rossmann stores over a 42-day horizon.

## Submission guide

This repository contains the five requested deliverables. **The notebook is the analytical walkthrough; the modular Python packages are the implementation.** The same code, charts and findings are intentionally not copied into the notebook. Instead, each notebook section points to its authoritative source files and saved evidence.

| Requirement | Primary deliverable | Where the detailed work lives |
|---|---|---|
| 1. Jupyter notebook: framing through model selection | [End-to-end analytical walkthrough](notebooks/01_end_to_end_forecasting.ipynb) | The notebook maps every requested topic to the relevant source modules, findings, charts and experiment outputs. |
| 2. Modular Python source, project structure and README | `src/`, this README, `requirements.txt` | Reusable implementation is organized by preprocessing, features, models and evaluation. |
| 3. Model evaluation report (8–10 pages maximum) | [Model evaluation report](reports/model_evaluation_report.md) | Detailed metrics, validation design, segment/error analysis, explainability and model selection. |
| 4. Executive summary (up to two pages) | [Chief Commercial Officer executive summary](reports/executive_summary.md) | One-page, plain-language decision brief. |
| 5. Production architecture diagram | [Production architecture](architecture/production_architecture.md) | Mermaid diagram and component responsibilities. |

## How Deliverable 1 maps to the repository

The notebook follows the requested sequence. Use this map to navigate to the full implementation and evidence without duplicating large code blocks or charts.

| Notebook topic | Implementation / evidence |
|---|---|
| Problem framing | `reports/model_evaluation_report.md` |
| Data audit | `src/preprocessing/data_validation.py`, `data_investigation.py`; `reports/data_findings.md`, `reports/data_audit.xlsx`, `reports/data_investigation.xlsx` |
| EDA | `src/preprocessing/eda.py`, `customer_eda.py`; `reports/eda_findings.md`, `customer_eda_findings.md`, `reports/figures/`, `reports/customer_eda_charts/` |
| Leakage assessment | `src/features/recursive_forecasting.py`, `validate_recursive_features.py`; leakage and feature decisions in the evaluation report |
| Feature engineering | `src/features/build_features.py`, `src/models/feature_config.py`, `src/features/recursive_forecasting.py` |
| Baselines | `src/models/baselines.py`, `run_baselines.py`, `run_first_window_baselines.py`; `reports/first_window_baseline_results.csv` |
| Models | `src/models/train_random_forest.py`, `train_lightgbm.py`, `train_first_window_random_forest.py` |
| Tuning | `src/models/tune_lightgbm.py`; `models/experiments/lightgbm_tuning/` |
| Evaluation | `src/models/evaluate_tuned_lightgbm_second_window.py`; `reports/second_window_tuned_lightgbm_results.json` |
| Error analysis | `src/evaluation/analyze_*.py`; `src/evaluation/results/` |
| Explainability | `src/evaluation/explain_lightgbm_model.py`; SHAP outputs in `src/evaluation/results/` |
| Model selection | `reports/model_evaluation_report.md`, `reports/executive_summary.md` |

**Notebook role:** provide the end-to-end story, summarize decisions and results, and direct reviewers to the executable implementation and supporting evidence. It is not intended to be a second copy of every source script.

## Project structure

- `data/raw/`: original Rossmann competition files (not committed; download separately).
- `data/processed/`: prepared datasets, validation predictions and intermediate artifacts.
- `notebooks/`: end-to-end analytical walkthrough.
- `src/preprocessing/`: data validation, investigation, preparation and EDA.
- `src/features/`: feature relationships, feature construction and recursive feature state.
- `src/models/`: baselines, model training, hyperparameter tuning and recursive forecasting.
- `src/evaluation/`: metric evaluation, segment/error analysis and explainability.
- `src/evaluation/results/`: saved store-, segment- and SHAP-level results.
- `models/experiments/`: model artifacts, tuning trials, run configurations and predictions.
- `reports/`: audit/EDA findings, charts, metric outputs and submission reports.
- `architecture/`: production architecture diagram.

## Environment and installation

Python 3.10+ is recommended. Create an isolated environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data setup

Download the Rossmann Store Sales competition data and place the original CSVs under `data/raw/` using the filenames expected by `src/features/build_features.py` and `src/preprocessing/prepare_data.py` (typically `train.csv`, `test.csv` and `store.csv`). Raw competition data is intentionally not included in this repository.

## Running and reproducing the workflow

Run commands from the repository root so package imports and relative paths resolve. The notebook can be opened and reviewed using the committed findings and outputs; rerunning the full data/model workflow requires the raw competition files.

1. **Audit and preparation:** `src/preprocessing/data_validation.py`, `data_investigation.py`, `prepare_data.py`.
2. **EDA:** `src/preprocessing/eda.py`, `customer_eda.py`.
3. **Feature engineering and leakage checks:** `src/features/build_features.py`, `recursive_forecasting.py`, `validate_recursive_features.py`.
4. **Baselines and candidate models:** scripts under `src/models/`, including `run_first_window_baselines.py`, `train_random_forest.py` and `train_lightgbm.py`.
5. **Tuning:** `src/models/tune_lightgbm.py`.
6. **Later chronological holdout:** `src/models/evaluate_tuned_lightgbm_second_window.py`.
7. **Error analysis and explainability:** scripts under `src/evaluation/`.

The committed experiment outputs are the evidence used in the reports. Re-running training or tuning may overwrite experiment artifacts; preserve the committed results when reproducing individual stages.

## Evaluation design and selected candidate

Two chronological, recursive 42-day windows were evaluated:

- Window 1: train through 8 May 2015; validate 9 May–19 June 2015. Used for model comparison and Optuna tuning.
- Window 2: train through 19 June 2015; validate 20 June–31 July 2015. The selected hyperparameters were frozen; no further tuning was performed.

Tuned LightGBM is the preferred candidate. It achieved 7.8206% WMAPE in Window 1 and 8.5643% in the later Window 2 holdout. The evaluation report explains the comparison, limitations and production-readiness recommendation.

## Important interpretation

This is a forecasting assignment and a production-oriented prototype, not a deployed production service. The architecture's orchestration, registry, monitoring and downstream integrations are proposed production controls, not claims of implemented or live services.
