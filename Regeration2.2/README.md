# HIT140 Assessment 3: Regression 2.2 

Predicts the goals scored by one team in one FIFA World Cup 2026 match, using eight
predictors that are all known before kickoff.

## Repository layout

```
Data/Regression_2_2_Dataset.csv            208 rows: 104 matches x 2 teams
Regeration2.2/Regression_2_2_Divya_S398598.py
Regeration2.2/Regression_2_2_outputs/       created when the script runs
requirements.txt
```

The script reads the dataset from `../Data/` relative to its own folder, so keep
this layout.

## How to reproduce the report

```bash
pip install -r requirements.txt
python Regeration2.2/Regression_2_2_Divya_S398598.py
```

Runtime is about a minute. All models use `random_state=42`. The script prints
the data checks, the model comparison table and the robustness checks, and writes
these files to `Regression_2_2_outputs/`:

| File | Used for |
|---|---|
| `model_comparison.csv` | Model comparison table (CV and holdout metrics) |
| `ols_inference.csv` | OLS coefficients, robust 95% CIs, p-values and VIFs |
| `eda_correlations.csv`, `eda_goals_by_rank_band.csv` | EDA numbers behind Figure 1 |
| `holdout_bootstrap.csv` | Bootstrap intervals for holdout RMSE gains |
| `linear_model_standardised_coefficients.csv` | Coefficients of the reported Ridge model |
| `tuning_all_settings.csv` | Every hyperparameter setting tested for every model |
| `boosting_wide_grid.csv` | Wider 60-setting Gradient Boosting check (CV only) |
| `feature_set_ablation.csv` | All 8 vs shared 4 vs Regression 2.2 four |
| `permutation_importance.csv` | Holdout permutation importance (reported linear model) |
| `holdout_predictions.csv` | Predictions for the final 21 matches |
| `summary.json` | Headline numbers and tests |
| `report_fig1_eda.png` | Report Figure 1 (EDA) |
| `report_fig2_model_comparison.png` | Report Figure 2 |
| `report_fig3_importance_and_fit.png` | Report Figure 3 |

## Models

- **Linear regression pathway (the task's required technique):** OLS, Ridge,
  Ridge with log FIFA ranks (reported model), Ridge with log ranks plus clipping.
- **Competing algorithms:** Poisson regression with log ranks, Random Forest,
  Gradient Boosting.
- **Reference:** mean baseline.

## Validation design

- The final 21 matches (2 to 19 July 2026) are a chronological holdout, used once
  for reporting only.
- All tuning and model selection use four expanding-window folds on the first 83
  matches. Both rows of a match always stay in the same split.
- Imputation and scaling are fitted inside each training fold.

## Data sources

| Columns | Source |
|---|---|
| `Team_FIFA_Rank`, `Opponent_FIFA_Rank` | FIFA/Coca-Cola Men's World Ranking released 11 June 2026 (before the first match) |
| `Team_Elo_PreMatch`, `Opponent_Elo_PreMatch` | World Football Elo Ratings, https://www.eloratings.net, ratings before the first kickoff (as at 10 June 2026); accessed 15 October 2026 |
| `Goals_Scored`, dates, fixtures | Jürisoo, M., international_results, https://github.com/martj42/international_results (commit a16ff3e, 20 July 2026). Goals include extra time and exclude penalty shoot-outs |
| Rest days, matches played | Derived from match dates; only earlier dates are used |

Rank and Elo are fixed pre-tournament values, so they contain no information
from any World Cup match.

## Data provenance check

```bash
python Regeration2.2/verify_data_provenance.py
```

Compares all 208 team rows with the independent results record above (pinned
commit), recomputes rest days and matches played from match dates, and checks
FIFA ranks and a stratified sample of Elo ratings against their published
sources. Writes `Regression_2_2_outputs/provenance_checks.csv`; all 10 checks
passed on 15 October 2026. Needs internet access to raw.githubusercontent.com.

## Reproducibility record

15 October 2026: fresh clone, new Python 3.13 virtual environment built only from
`requirements.txt`, script run from the repository root in about 35 seconds.
All 15 output files were regenerated; every numeric value matched the committed
outputs to within floating-point rounding (largest difference 4e-13).

