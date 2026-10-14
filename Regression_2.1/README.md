# Regression 2.1 — FIFA World Cup 2026 Goal Difference Prediction

## Project overview

This section uses multiple linear regression to predict goal difference in 104 FIFA World Cup 2026 matches.

The model uses eight explanatory variables, including FIFA rankings, Elo ratings, host advantage and recent international match performance.

## Files and folders

| File or folder | Description |
|---|---|
| `01_prepare_dataset.py` | Prepares the dataset and calculates recent-form variables. |
| `02_regression_model.py` | Trains and evaluates the linear regression model. |
| `regression21_final_dataset.csv` | Final dataset containing 104 matches. |
| `Data/raw/` | Local historical international match results. |
| `Data/processed/` | Local World Cup match data with FIFA rankings and Elo ratings. |
| `Results/` | Model predictions, evaluation metrics and regression coefficients. |

## Explanatory variables

1. `team_fifa_rank` — FIFA ranking of the first-listed team.
2. `opponent_fifa_rank` — FIFA ranking of the opposing team.
3. `team_elo` — Elo rating of the first-listed team.
4. `opponent_elo` — Elo rating of the opposing team.
5. `host_advantage` — Indicator for selected host-country matches.
6. `team_recent_goal_difference_avg` — Average goal difference over the team's five most recent international matches.
7. `opponent_recent_goals_conceded_avg` — Average goals conceded by the opponent over its five most recent international matches.
8. `team_recent_win_percentage` — Win percentage over the team's five most recent international matches.

The dependent variable is match goal difference, calculated as the first-listed team's goals minus the opponent's goals.

## How to run

Install Python and the required libraries:

```bash
python -m pip install pandas numpy scikit-learn
```

The final prepared dataset is included in this folder.

To run the regression model:

```bash
python 02_regression_model.py
```

To recreate the dataset from the source files, place `results.csv` inside `Data/raw/` and `worldcup2026_regression21_with_fifa_and_elo.csv` inside `Data/processed/`, then run:

```bash
python 01_prepare_dataset.py
```

The source input files are not included in this public folder pending confirmation of their redistribution permissions.

## Model evaluation

The dataset was split into 83 training matches and 21 testing matches.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 1.487 |
| Root Mean Squared Error (RMSE) | 1.875 |
| R-squared | 0.363 |
| Baseline MAE | 1.839 |

The model achieved a lower MAE than the baseline on the test set. However, prediction errors remain, and these results do not establish performance on genuinely unseen future World Cup matches.

## Limitations

- FIFA rankings and Elo ratings are strongly related, which may affect coefficient interpretation.
- The model uses only 104 matches.
- Host advantage is represented by a simplified indicator.
- The match dataset includes recorded outcomes, so the evaluation should be interpreted as a retrospective modelling exercise rather than a verified pre-tournament forecast.
- The original data sources and redistribution permissions still require documentation and verification.
