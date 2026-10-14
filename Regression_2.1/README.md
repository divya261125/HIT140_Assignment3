# HIT140 Assessment 3 — Regression 2.1

## Project overview

This project uses Linear Regression to predict goal difference in FIFA World Cup 2026 matches.

Goal difference is calculated as the team's goals minus the opponent's goals.

## Dataset

The dataset contains 104 matches and eight explanatory variables:

1. Team FIFA ranking
2. Opponent FIFA ranking
3. Team Elo rating
4. Opponent Elo rating
5. Host advantage
6. Team's recent average goal difference
7. Opponent's recent average goals conceded
8. Team's recent win percentage

## How to run

Install Python and the required libraries:

```bash
python -m pip install pandas numpy scikit-learn
```

Run the regression model:

```bash
python 02_regression_model.py
```

The script reads `regression21_final_dataset.csv` and creates a `Results` folder containing predictions, coefficients and model evaluation results.

## Model evaluation

The dataset is divided into 80% training and 20% testing using `random_state=42`.

| Metric | Result |
|---|---|
| MAE | 1.487 |
| RMSE | 1.875 |
| R² | 0.363 |
| Baseline MAE | 1.839 |

The model performs better than the mean-prediction baseline on the held-out test matches.

## Files

- `01_prepare_dataset.py` — prepares the model dataset using supporting historical data.
- `02_regression_model.py` — trains and evaluates the Linear Regression model.
- `regression21_final_dataset.csv` — final dataset.
- `regression21_predictions.csv` — saved predictions.
- `regression21_coefficients.csv` — saved regression coefficients.
- `regression21_model_results.csv` — saved evaluation metrics.

**Note:** Re-running `01_prepare_dataset.py` requires additional source data files that are not yet included in this folder.
