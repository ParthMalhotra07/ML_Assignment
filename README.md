# Polynomial Regression Assignment

Roll number: BT2024197

Repository: https://github.com/ParthMalhotra07/ML_Assignment

This project trains separate polynomial regression models for both parts of the assignment:

- **var1:** Net Power Score prediction using six inputs.
- **var2:** Thermal Anomaly Score prediction using three inputs.

## Setup

Use Python 3.12 and install the dependencies:

```text
python -m pip install -r requirements.txt
```

The four personalized datasets are included in `BT2024197/`:

```text
BT2024197/
    BT2024197_train_var1.csv
    BT2024197_test_var1.csv
    BT2024197_train_var2.csv
    BT2024197_test_var2.csv
```

The supplied `sample_submission.csv` is included in the repository root. Use only these BT2024197 datasets when reproducing this assignment.

CSV files retain their original line endings so the saved file hashes remain reproducible across operating systems.

## Training and prediction

Run these commands from the repository root:

```text
python src/train.py --search-only
python src/search_sparse.py
python src/finalize.py --sample sample_submission.csv
python src/verify.py
```

The searches compare polynomial degrees and regularization strengths. `finalize.py` selects each model using the cross-validation results, evaluates it on held-out training rows, fits it again using all training rows, and generates both prediction files.

The main generated files are:

```text
submission/BT2024197_pred_var1.csv
submission/BT2024197_pred_var2.csv
results/final_model_var1.npz
results/final_model_var2.npz
results/final_metrics.json
results/figures/
```

Each prediction file contains 1,000 values in the original test-row order, with one column named `y` and no index column.

Predictions can be regenerated from the saved models with:

```text
python src/predict.py --variant 1 --sample sample_submission.csv
python src/predict.py --variant 2 --sample sample_submission.csv
```

On Windows, `./verify_and_predict.ps1` runs both inference commands and the numerical checks using Python from your environment.

`search_sparse.py` reuses completed degree results if its output files already exist. Start with an empty `results` folder when changing the data or search settings.

## Method

Each dataset is split into 800 development rows and 200 holdout rows using seed 197. Model selection uses five shuffled folds within the development set, with seed 4197. Polynomial feature means and scales are fitted separately inside each training fold.

Ridge regression is tested at every degree from 1 to 10 for var1 and 1 to 20 for var2, using monomial and Legendre bases. Its penalty grid contains 13 powers of ten from 1e-8 to 1e4. The Lasso search tests degrees 3, 4, 5, 6 for var1 and 6, 8, 10, 12 for var2, with penalties 0.1, 0.03, 0.01, and 0.003.

Selection uses mean cross-validation MSE. Lasso candidates with a maximum fold duality gap above 0.001 are excluded. Holdout scores are not used to choose the model. Both final models are fitted on all 1,000 training rows.

All polynomial terms have total degree no greater than the selected degree, including interaction terms. The intercept is not penalized.

## Results

| Dataset | Model | Degree | Alpha | Holdout MSE | Holdout R-squared |
| --- | --- | --- | --- | --- | --- |
| var1 | Polynomial Lasso | 5 | 0.01 | 0.318792 | 0.970073 |
| var2 | Polynomial Ridge | 10 | 1 | 0.234471 | 0.996000 |

These scores were measured on 200 held-out training rows per dataset. The test targets are hidden, so test scores are not available.

## Files

| File | Purpose |
| --- | --- |
| `src/polynomial.py` | Polynomial features, Ridge fitting, and shared prediction functions |
| `src/train.py` | Ridge degree and penalty search |
| `src/search_sparse.py` | Lasso fitting and cross-validation search |
| `src/finalize.py` | Model selection, holdout evaluation, final fitting, and CSV export |
| `src/predict.py` | Inference using saved models |
| `src/verify.py` | Numerical and output checks |
| `src/recheck_models.py` | Additional development-only checks using three fold seeds |
| `src/summarize_review.py` | Summaries of completed degree-recheck folds |

## Saved results and figures

`results/` contains the original Ridge and Lasso search tables, selected model parameters, holdout predictions, final metrics, and numerical check results. `results/figures/` contains the six figures used in the report: training correlations, input boundary comparisons, degree-selection curves, degree/penalty heatmaps, holdout diagnostics, and fitted response slices.

`results/model_review/` contains the additional degree-recheck results. Its summary includes only configurations with all 15 folds completed across seeds 4197, 4207, and 4217. Some degree-6 Lasso fold records are incomplete and are excluded from that summary.

The optional checks can be rerun with:

```text
python src/recheck_models.py 1
python src/recheck_models.py 2
python src/summarize_review.py
```

Each recheck replaces the stored fold records for its selected variant. These extra checks do not change the selected models or submission CSVs.

Response heatmaps are labelled slices of the fitted functions, not hidden-test ground truth.

## Course submission

Upload these three files to the course portal:

```text
submission/BT2024197_report.pdf
submission/BT2024197_pred_var1.csv
submission/BT2024197_pred_var2.csv
```

The report has five pages and includes the GitHub repository link. The datasets, sample CSV, prediction CSVs, saved results, and source code are tracked here. The report PDF is kept locally for the separate course upload. Publishing this repository does not submit the assignment to the course portal.
