
# Regression 2.1 — FIFA World Cup 2026 Goal Difference Prediction

## Project overview

This section uses multiple linear regression to predict goal difference in 104 FIFA World Cup 2026 matches.

The dataset contains exactly eight explanatory variables, selected using information available before each match. These include FIFA rankings, Elo ratings, host advantage and recent international match performance.

The analysis includes exploratory data analysis (EDA), ordinary least squares (OLS) regression, cross-validation, feature engineering, model optimisation and comparisons with alternative algorithms.

## Files and folders

| File or folder | Description |
|---|---|
| `01_prepare_dataset.py` | Prepares the dataset and calculates recent-form variables. |
| `02_regression_model.py` | Trains and evaluates the original linear regression model. |
| `03_eda_visualisations.py` | Performs EDA and generates visualisations. |
| `04_cross_validation_optimisation.py` | Performs repeated cross-validation and compares alternative models. |
| `regression21_final_dataset.csv` | Final dataset containing 104 matches and eight original explanatory variables. |
| `Data/raw/` | Local historical international match results. |
| `Data/processed/` | Local World Cup match data with FIFA rankings and Elo ratings. |
| `Results/` | Model predictions, evaluation metrics and regression coefficients. |
| `Figures/` | EDA visualisations. |
| `../Report/regression_2_1_supplement.py` | Supplementary feature-engineering and strength-gap model evaluation. |
| `../Report/results_2_1/` | Outputs from the supplementary analysis. |

## Explanatory variables

The original dataset uses the following eight explanatory variables:

1. `team_fifa_rank` — FIFA ranking of the first-listed team.
2. `opponent_fifa_rank` — FIFA ranking of the opposing team.
3. `team_elo` — Elo rating of the first-listed team.
4. `opponent_elo` — Elo rating of the opposing team.
5. `host_advantage` — Indicator for selected host-country matches.
6. `team_recent_goal_difference_avg` — Average goal difference over the team's five most recent international matches.
7. `opponent_recent_goals_conceded_avg` — Average goals conceded by the opponent over its five most recent international matches.
8. `team_recent_win_percentage` — Win percentage over the team's five most recent international matches.

The dependent variable is match goal difference, calculated as the first-listed team's goals minus the opponent's goals.

All eight explanatory variables are based on information available before kickoff. Recent-form variables are calculated from historical international matches played before 11 June 2026.

The original dataset retains these eight variables. Supplementary experiments transform or combine existing predictors rather than requiring additional match-outcome information.

## How to run

Install Python and the required libraries:

```bash
python -m pip install pandas numpy matplotlib scipy scikit-learn statsmodels
```

The final prepared dataset is included in the `Regression_2.1` folder.

To run the original regression model, navigate to that folder and execute:

```bash
python 02_regression_model.py
```

To recreate the dataset from the original source files, place:

- `results.csv` inside `Data/raw/`
- `worldcup2026_regression21_with_fifa_and_elo.csv` inside `Data/processed/`

Then run:

```bash
python 01_prepare_dataset.py
```

These two source input files are not included in the public folder pending confirmation of redistribution permissions.

The prepared dataset can still be used for the subsequent regression, EDA and optimisation scripts.

## Model evaluation

The original dataset was split into 83 training matches and 21 testing matches.

### Original linear regression holdout results

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 1.487 |
| Root Mean Squared Error (RMSE) | 1.875 |
| R-squared | 0.363 |
| Baseline MAE | 1.839 |

The original linear regression achieved a lower MAE than the baseline on the test set.

However, prediction errors remain substantial, and these results do not establish performance on genuinely unseen future World Cup matches.

## Exploratory Data Analysis (EDA)

`03_eda_visualisations.py` examines:

- The distribution of the explanatory variables.
- Relationships between explanatory variables and goal difference.
- Host advantage and match stages.
- Correlations among predictors.
- Multicollinearity using Variance Inflation Factors (VIFs).

The analysis identified strong negative correlations between FIFA rankings and Elo ratings, approximately −0.89.

This indicates that some strength-related predictors contain overlapping information, which can affect the stability and interpretation of regression coefficients.

EDA charts are saved in `Figures/`, and supporting tables are saved in `Results/`.

## Cross-validation and model optimisation

`04_cross_validation_optimisation.py` evaluates the original linear regression using five-fold cross-validation repeated ten times, producing 50 validation results.

The script also compares transformations of the explanatory variables and alternative regression algorithms.

### Original optimisation results

The earlier optimisation selected Ridge regression with log-transformed FIFA rankings.

| Model | CV MAE | CV RMSE | CV R² |
|---|---:|---:|---:|
| Baseline | 1.592 | 2.019 | −0.072 |
| Original linear regression | 1.306 | 1.629 | 0.275 |
| Ridge regression + log FIFA ranks | 1.268 | 1.600 | 0.312 |

Ridge regression with log-transformed FIFA rankings achieved the lowest cross-validation MAE among the models compared in the original optimisation script.

However, its improvement over the original linear regression was modest and was not statistically significant according to the reported corrected t-test (p = 0.442).

### Original optimisation holdout results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Original linear regression | 1.487 | 1.875 | 0.363 |
| Ridge regression + log FIFA ranks | 1.327 | 1.730 | 0.457 |

## Supplementary strength-gap experiment

A further feature-engineering experiment was conducted using:

`Report/regression_2_1_supplement.py`

This experiment examines whether representing the relative strengths of the opposing teams improves predictive performance.

Two engineered predictors are derived from the original strength variables:

- `elo_gap` = `team_elo` − `opponent_elo`
- `log_rank_ratio` = log(`opponent_fifa_rank`) − log(`team_fifa_rank`)

Positive values of these contrasts indicate that the first-listed team is stronger according to the corresponding rating measure.

These engineered variables are calculated from the original pre-match explanatory variables.

The required dataset continues to contain exactly eight original explanatory variables. The supplementary experiment changes the model's input representation rather than adding new information collected after a match.

### Supplementary cross-validation results

| Model | CV MAE |
|---|---:|
| Original linear regression | 1.306 |
| Ridge regression + log FIFA ranks | 1.268 |
| Ridge regression + strength-gap features | 1.242 |

The strength-gap Ridge model achieved a cross-validation MAE of 1.242, compared with 1.306 for the original linear regression.

The corresponding supplementary analysis reported:

- CV RMSE: 1.586
- CV R²: 0.323
- Corrected statistical test: p = 0.241

The improvement over the original linear regression was not statistically significant at the 5% level.

Therefore, the strength-gap Ridge model should be interpreted as a promising exploratory alternative rather than conclusive evidence of superior predictive performance.

Because several models were compared using the available validation data, the selected model's reported performance may also be optimistic.

The original eight-variable linear regression remains the reference model for the required task, while the strength-gap experiment provides additional evidence about feature engineering and model optimisation.

## Reproducing the analysis

### Original Regression 2.1 scripts

From the `Regression_2.1` folder, run:

```bash
python 02_regression_model.py
python 03_eda_visualisations.py
python 04_cross_validation_optimisation.py
```

These scripts use the included prepared dataset.

To regenerate the prepared dataset from source files, run:

```bash
python 01_prepare_dataset.py
```

This preparation step requires the two source CSV files described earlier.

### Supplementary strength-gap analysis

From the repository root, run:

```bash
python Report/regression_2_1_supplement.py
```

The supplementary script reads:

`Regression_2.1/regression21_final_dataset.csv`

It writes its analysis outputs to:

`Report/results_2_1/`

The supplementary analysis can be run without recreating the original dataset, provided the prepared dataset is available.

## Data limitations and interpretation

- The analysis uses only 104 World Cup matches from a single tournament.
- FIFA rankings and Elo ratings are strongly related, creating multicollinearity concerns.
- Host advantage is represented using a simplified indicator.
- Recent-form variables are calculated from historical matches before the tournament rather than continuously updated during the tournament.
- The evaluation is retrospective because the match outcomes are already recorded.
- The original source inputs required to rebuild the prepared dataset are not currently included in the public repository.
- The improvements achieved by the supplementary strength-gap model were not statistically significant.
- Selecting a model after comparing several candidates on the same validation folds can produce optimistic performance estimates.
- Results from one tournament should not be assumed to generalise to future international competitions.

## Conclusion

The original linear regression provides a baseline for predicting World Cup goal differences using eight pre-match explanatory variables.

Ridge regression and feature engineering were investigated to address overlapping strength-related predictors and improve predictive accuracy.

The supplementary strength-gap Ridge model achieved the lowest reported cross-validation MAE of 1.242 among the models discussed here.

However, the statistical evidence does not establish a reliable improvement over the original linear regression.

These findings support cautious use of strength-gap features and further validation before applying the model to future matches.
