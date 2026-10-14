"""
Regression 2.2 - feature-engineering supplement to Regression_2_2_Divya_S398598.py

Uses the same data checks, chronological holdout (final 21 matches) and four
expanding-window CV folds as the main script. Candidate inputs are built only
from the eight collected pre-match variables:
  elo_gap        = Team_Elo_PreMatch - Opponent_Elo_PreMatch
  log_rank_ratio = log(Opponent_FIFA_Rank) - log(Team_FIFA_Rank)
  rest_gap       = Team rest days - Opponent rest days (0 for opening matches,
                   when neither side has played)
  knockout       = 1 if matches already played >= 3 (replaces the two identical
                   matches-played columns, so holdout values 4-7 are no longer
                   extrapolated beyond the training range 0-3)
Models are selected on CV RMSE only; the holdout is scored once afterwards.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder.parent / "Data" / "Regression_2_2_Dataset.csv")
target = "Goals_Scored"
features = ["Team_FIFA_Rank", "Opponent_FIFA_Rank", "Team_Elo_PreMatch", "Opponent_Elo_PreMatch",
            "Team_PreMatch_Rest_Days", "Opponent_PreMatch_Rest_Days",
            "Team_PreMatch_Matches_Played", "Opponent_PreMatch_Matches_Played"]
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Date", "Match_ID", "Team"]).reset_index(drop=True)
assert len(data) == 208 and data.Match_ID.nunique() == 104

# Same holdout and folds as the main script
match_ids = data[["Match_ID", "Date"]].drop_duplicates("Match_ID").sort_values(["Date", "Match_ID"])["Match_ID"].to_numpy()
test_ids = set(match_ids[-int(np.ceil(len(match_ids) * 0.2)):])
train = data[~data.Match_ID.isin(test_ids)].reset_index(drop=True)
test = data[data.Match_ID.isin(test_ids)].reset_index(drop=True)
tr_ids = train[["Match_ID", "Date"]].drop_duplicates("Match_ID").sort_values(["Date", "Match_ID"])["Match_ID"].to_numpy()
first = min(max(30, len(tr_ids) // 2), len(tr_ids) - 8)
folds, count = [], first
for block in [b for b in np.array_split(tr_ids[first:], 4) if len(b)]:
    folds.append((np.flatnonzero(train.Match_ID.isin(tr_ids[:count]).to_numpy()),
                  np.flatnonzero(train.Match_ID.isin(block).to_numpy())))
    count += len(block)


class LogRank(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in ["Team_FIFA_Rank", "Opponent_FIFA_Rank"]:
            X[c] = np.log(X[c])
        return X


class Contrasts(BaseEstimator, TransformerMixin):
    def __init__(self, keep_team_level=False):
        self.keep_team_level = keep_team_level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        out["elo_gap"] = X["Team_Elo_PreMatch"] - X["Opponent_Elo_PreMatch"]
        out["log_rank_ratio"] = np.log(X["Opponent_FIFA_Rank"]) - np.log(X["Team_FIFA_Rank"])
        out["rest_gap"] = (X["Team_PreMatch_Rest_Days"] - X["Opponent_PreMatch_Rest_Days"]).fillna(0)
        out["knockout"] = (X["Team_PreMatch_Matches_Played"] >= 3).astype(float)
        if self.keep_team_level:
            out["log_team_rank"] = np.log(X["Team_FIFA_Rank"])
        return out


def pipe(model, transform):
    return Pipeline([("t", transform), ("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("model", model)])


ridge = {"model__alpha": [1, 3, 10, 30, 100, 300, 1000]}
pois = {"model__alpha": [0.01, 0.1, 0.3, 1, 3, 10]}
cands = {
    "Ridge + log FIFA rank (main script)": (pipe(Ridge(), LogRank()), ridge),
    "Ridge + contrasts": (pipe(Ridge(), Contrasts()), ridge),
    "Ridge + contrasts + team level": (pipe(Ridge(), Contrasts(True)), ridge),
    "Poisson + contrasts": (pipe(PoissonRegressor(max_iter=3000), Contrasts()), pois),
    "Poisson + contrasts + team level": (pipe(PoissonRegressor(max_iter=3000), Contrasts(True)), pois),
}

rng = np.random.default_rng(42)
mids = test.Match_ID.unique()
base_pred = np.full(len(test), train[target].mean())
rows, preds = [], {}
for name, (m, grid) in cands.items():
    gs = GridSearchCV(m, grid, scoring="neg_root_mean_squared_error", cv=folds, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(train[features], train[target])
    p = gs.predict(test[features])
    preds[name] = p
    i = gs.best_index_
    rows.append({"Model": name, "CV_RMSE": -gs.cv_results_["mean_test_score"][i],
                 "CV_RMSE_SD": gs.cv_results_["std_test_score"][i],
                 "Test_MAE": mean_absolute_error(test[target], p),
                 "Test_RMSE": np.sqrt(mean_squared_error(test[target], p)),
                 "Test_R2": r2_score(test[target], p), "Min_prediction": p.min(),
                 "Best_params": gs.best_params_})

res = pd.DataFrame(rows)
out = folder / "Regression_2_2_outputs"
res.to_csv(out / "feature_engineering_supplement.csv", index=False)
print(res.round(4).to_string(index=False))

# Match-level bootstrap of holdout RMSE gain vs mean baseline (both rows of a match resampled together)
y_t = test[target].to_numpy()
idx_by_match = [np.flatnonzero(test.Match_ID.to_numpy() == m) for m in mids]
boot = []
for name, p in preds.items():
    gains = []
    for _ in range(5000):
        pick = np.concatenate([idx_by_match[j] for j in rng.integers(0, len(mids), len(mids))])
        r_b = np.sqrt(np.mean((y_t[pick] - base_pred[pick]) ** 2))
        r_m = np.sqrt(np.mean((y_t[pick] - p[pick]) ** 2))
        gains.append(r_b - r_m)
    boot.append({"Model": name, "RMSE_gain_vs_baseline": np.sqrt(np.mean((y_t - base_pred) ** 2)) - np.sqrt(np.mean((y_t - p) ** 2)),
                 "CI_2.5": np.percentile(gains, 2.5), "CI_97.5": np.percentile(gains, 97.5)})
boot = pd.DataFrame(boot)
boot.to_csv(out / "feature_engineering_bootstrap.csv", index=False)
print(boot.round(4).to_string(index=False))
