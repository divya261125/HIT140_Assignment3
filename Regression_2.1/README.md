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

## Exploratory Data Analysis (EDA)

`03_eda_visualisations.py` examines the distribution of the eight explanatory variables, their correlations with goal difference, host advantage, match stages and multicollinearity.

The analysis found strong negative correlations between FIFA rankings and Elo ratings (approximately −0.89), indicating multicollinearity. Variance Inflation Factors (VIFs) were also calculated to assess relationships between explanatory variables.

EDA charts are saved in `Figures/`, and supporting tables are saved in `Results/`.

## Cross-validation and model optimisation

`04_cross_validation_optimisation.py` evaluates the original linear regression using 5-fold cross-validation repeated 10 times, giving 50 validation results.

The script also compares transformations of the explanatory variables and several alternative regression algorithms.

The selected model is Ridge regression with log-transformed FIFA rankings.

### Cross-validation results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 1.592 | 2.019 | −0.072 |
| Original linear regression | 1.306 | 1.629 | 0.275 |
| Ridge regression + log FIFA ranks | 1.268 | 1.600 | 0.312 |

The Ridge model achieved the lowest cross-validation MAE among the compared models. However, its improvement over the original linear regression was modest and was not statistically significant according to the reported corrected t-test (p = 0.442).

### Holdout test results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Original linear regression | 1.487 | 1.875 | 0.363 |
| Ridge regression + log FIFA ranks | 1.327 | 1.730 | 0.457 |

## Running all four scripts

Install the required libraries:

```bash
python -m pip install pandas numpy matplotlib scipy scikit-learn statsmodels
```

From the `Regression_2.1` folder, run:

```bash
python 01_prepare_dataset.py
python 02_regression_model.py
python 03_eda_visualisations.py
python 04_cross_validation_optimisation.py
```

The first script requires the two source CSV files in their documented local folders. If those source files are unavailable, the included `regression21_final_dataset.csv` can still be used to run scripts 02, 03 and 04.

Figures are saved in `Figures/` and evaluation tables in `Results/`.
