"""
Regression 2.1 - Stage 3: feature engineering (supplement to Regression_2.1 script 04).

Lives in Report/ so the Regression_2.1 folder is left unchanged: it only reads
Regression_2.1/regression21_final_dataset.csv and writes to Report/results_2_1/.

Adds pre-match contrast features built only from the eight collected variables:
  elo_gap        = team_elo - opponent_elo
  log_rank_ratio = log(opponent_fifa_rank) - log(team_fifa_rank)
Both are positive when the first-listed team is stronger. The dataset still
contains exactly eight explanatory variables; only the model inputs change.

Uses the same 5-fold x 10 RepeatedKFold (random_state=42) and corrected
resampled t-test as script 04, so results are directly comparable.
Also writes the inferential EDA tests used in the report.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent / "Regression_2.1"
OUT = Path(__file__).resolve().parent / "results_2_1"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(BASE_DIR / "regression21_final_dataset.csv", encoding="utf-8-sig")
variables = [
    "team_fifa_rank", "opponent_fifa_rank", "team_elo", "opponent_elo",
    "host_advantage", "team_recent_goal_difference_avg",
    "opponent_recent_goals_conceded_avg", "team_recent_win_percentage",
]
X, y = df[variables], df["goal_difference"]
assert len(X) == 104 and X.shape[1] == 8


class LogRanks(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in ["team_fifa_rank", "opponent_fifa_rank"]:
            X[c] = np.log(X[c])
        return X


class Contrasts(BaseEstimator, TransformerMixin):
    """Replace the four strength columns with two team-minus-opponent gaps."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["elo_gap"] = X["team_elo"] - X["opponent_elo"]
        X["log_rank_ratio"] = np.log(X["opponent_fifa_rank"]) - np.log(X["team_fifa_rank"])
        return X.drop(columns=["team_fifa_rank", "opponent_fifa_rank",
                               "team_elo", "opponent_elo"])


alphas = np.logspace(-3, 3, 61)
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

models = {
    "Baseline (predict the mean)": DummyRegressor(strategy="mean"),
    "Linear regression: original variables": LinearRegression(),
    "Ridge regression + log FIFA ranks (script 04 final)":
        make_pipeline(LogRanks(), StandardScaler(), RidgeCV(alphas=alphas)),
    "Linear regression: strength contrasts":
        make_pipeline(Contrasts(), LinearRegression()),
    "Ridge regression: strength contrasts":
        make_pipeline(Contrasts(), StandardScaler(), RidgeCV(alphas=alphas)),
}

folds = {}
for name, m in models.items():
    s = cross_validate(m, X, y, cv=cv, scoring=scoring)
    folds[name] = pd.DataFrame({"MAE": -s["test_MAE"], "RMSE": -s["test_RMSE"], "R2": s["test_R2"]})

n_test = len(X) / 5
n_train = len(X) - n_test


def corrected_t(a, b, return_ci=False):
    d = (a - b).values
    k = len(d)
    se = np.sqrt((1 / k + n_test / n_train) * d.var(ddof=1))
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), k - 1)
    if return_ci:
        m = stats.t.ppf(0.975, k - 1) * se
        return d.mean(), d.mean() - m, d.mean() + m, p
    return d.mean(), p


ref = folds["Linear regression: original variables"]["MAE"]
rows = []
for name, f in folds.items():
    diff, lo, hi, p = (corrected_t(f["MAE"], ref, True)
                       if name != "Linear regression: original variables" else (0.0, 0.0, 0.0, 1.0))
    rows.append({"Model": name, "CV MAE": f["MAE"].mean(), "CV MAE SD": f["MAE"].std(),
                 "CV RMSE": f["RMSE"].mean(), "CV R2": f["R2"].mean(),
                 "MAE change vs original": diff, "95% CI low": lo, "95% CI high": hi, "p (corrected t-test)": p,
                 "% folds better than original": 100 * (f["MAE"] < ref).mean()})
res = pd.DataFrame(rows)
res.to_csv(OUT / "stage3_feature_engineering.csv", index=False)
pd.concat(folds, names=["Model", "Fold"]).to_csv(OUT / "stage3_fold_scores.csv")
print(res.round(4).to_string(index=False))

# Contrast-model test against the script 04 final model
best = "Ridge regression: strength contrasts"
d, p = corrected_t(folds[best]["MAE"], folds["Ridge regression + log FIFA ranks (script 04 final)"]["MAE"])
print(f"\n{best} vs Ridge+log ranks: MAE diff {d:.4f}, p = {p:.3f}")

# Full-sample coefficients of the contrast Ridge model (standardised)
final = models[best].fit(X, y)
names = final[0].transform(X).columns
coef = pd.Series(final[-1].coef_, index=names)
print("alpha =", final[-1].alpha_)
print(coef.round(3))
coef.to_csv(OUT / "stage3_contrast_ridge_coefficients.csv", header=["standardised_coef"])

# ------------------------------------------------------------------
# Inferential EDA used in the report
# ------------------------------------------------------------------
gd = y
m, s, n = gd.mean(), gd.std(ddof=1), len(gd)
ci = stats.t.interval(0.95, n - 1, loc=m, scale=s / np.sqrt(n))
one = stats.ttest_1samp(gd, 0)
host = stats.ttest_ind(gd[df.host_advantage == 1], gd[df.host_advantage == 0], equal_var=False)
ko = df["round"].str.contains("Group") == False
stage = stats.ttest_ind(gd[~ko], gd[ko], equal_var=False)
eg = df.team_elo - df.opponent_elo
r_gap = stats.pearsonr(eg, gd)
lrr = np.log(df.opponent_fifa_rank) - np.log(df.team_fifa_rank)
r_lrr = stats.pearsonr(lrr, gd)
inf = pd.DataFrame([
    {"Test": "Mean goal difference, 95% CI", "Value": m, "CI low": ci[0], "CI high": ci[1], "p": one.pvalue},
    {"Test": "Host vs non-host (Welch)", "Value": gd[df.host_advantage == 1].mean() - gd[df.host_advantage == 0].mean(),
     "CI low": np.nan, "CI high": np.nan, "p": host.pvalue},
    {"Test": "Group vs knockout (Welch)", "Value": gd[~ko].mean() - gd[ko].mean(),
     "CI low": np.nan, "CI high": np.nan, "p": stage.pvalue},
    {"Test": "r(elo_gap, goal difference)", "Value": r_gap.statistic, "CI low": np.nan, "CI high": np.nan, "p": r_gap.pvalue},
    {"Test": "r(log_rank_ratio, goal difference)", "Value": r_lrr.statistic, "CI low": np.nan, "CI high": np.nan, "p": r_lrr.pvalue},
])
inf.to_csv(OUT / "inferential_eda.csv", index=False)
print(inf.round(4).to_string(index=False))
