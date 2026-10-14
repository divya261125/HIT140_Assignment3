# HIT140 Assessment 3 - Regression 2.2
# Predict the goals scored by a team in one World Cup match.
#
# Structure
#   1. Load and validate the dataset
#   2. Chronological holdout (final 20% of matches) + four expanding CV folds
#   3. Exploratory data analysis (EDA)
#   4. Linear regression pathway: OLS -> Ridge -> Ridge with log-rank
#   5. Competing algorithms: Poisson GLM, Random Forest, Gradient Boosting
#   6. Inferential analysis (OLS with robust standard errors)
#   7. Robustness checks, figures and saved outputs
#
# All choices are made with the chronological CV folds. The holdout is only
# used to report final results.

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


folder = Path(__file__).resolve().parent
data_file = folder.parent / "Data" / "Regression_2_2_Dataset.csv"
out_folder = folder / "Regression_2_2_outputs"

# Columns used in the analysis.
target_col = "Goals_Scored"
id_cols = ["Match_ID", "Date", "Team", "Opponent"]

# These four variables are shared with Regression 2.1.
shared_feats = [
    "Team_FIFA_Rank",
    "Opponent_FIFA_Rank",
    "Team_Elo_PreMatch",
    "Opponent_Elo_PreMatch",
]

# These are the four extra variables for Regression 2.2.
extra_feats = [
    "Team_PreMatch_Rest_Days",
    "Opponent_PreMatch_Rest_Days",
    "Team_PreMatch_Matches_Played",
    "Opponent_PreMatch_Matches_Played",
]

features = shared_feats + extra_feats
required_cols = id_cols + [target_col] + features


# Read and check the dataset.
if not data_file.exists():
    raise FileNotFoundError(
        "Dataset not found at ../Data/Regression_2_2_Dataset.csv."
    )

data = pd.read_csv(data_file)

missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

data = data[required_cols].copy()
data["Date"] = pd.to_datetime(data["Date"], errors="raise")

for col in features + [target_col]:
    data[col] = pd.to_numeric(data[col], errors="raise")

data = data.sort_values(["Date", "Match_ID", "Team"]).reset_index(drop=True)

if len(data) != 208:
    raise ValueError(f"Expected 208 rows, found {len(data)}")
if data["Match_ID"].nunique() != 104:
    raise ValueError("Expected 104 unique matches")
if data["Team"].nunique() != 48:
    raise ValueError("Expected 48 teams")
if data[id_cols + [target_col]].isna().any().any():
    raise ValueError("ID columns and the target must not contain missing values")
if data.duplicated(["Match_ID", "Team"]).any():
    raise ValueError("Duplicate team rows found for a match")
if not (data.groupby("Match_ID").size() == 2).all():
    raise ValueError("Each match must have two team rows")
if (data[target_col] < 0).any() or not np.allclose(data[target_col], data[target_col].round()):
    raise ValueError("Goals must be non-negative whole numbers")
if len(features) != 8 or set(shared_feats) & set(extra_feats):
    raise ValueError("The model must have eight different predictors")

# Check that each row's opponent is the other team in that match.
for match_id, match_rows in data.groupby("Match_ID"):
    teams = match_rows["Team"].tolist()
    opponents = match_rows["Opponent"].tolist()
    if len(teams) != 2 or teams[0] != opponents[1] or teams[1] != opponents[0]:
        raise ValueError(f"Team/opponent mismatch in match {match_id}")

# FIFA ranking and Elo values should be present. First-match rest days can be blank.
if data[shared_feats].isna().any().any():
    raise ValueError("Some FIFA ranking or Elo values are missing")

out_folder.mkdir(exist_ok=True)


# Keep the latest 20% of matches for the final test.
match_ids = (
    data[["Match_ID", "Date"]]
    .drop_duplicates("Match_ID")
    .sort_values(["Date", "Match_ID"])["Match_ID"]
    .to_numpy()
)
test_count = int(np.ceil(len(match_ids) * 0.20))
test_ids = set(match_ids[-test_count:])

train_data = data[~data["Match_ID"].isin(test_ids)].reset_index(drop=True)
test_data = data[data["Match_ID"].isin(test_ids)].reset_index(drop=True)

# Make four expanding time-based validation folds.
# Both rows from one match stay in the same fold.
train_match_ids = (
    train_data[["Match_ID", "Date"]]
    .drop_duplicates("Match_ID")
    .sort_values(["Date", "Match_ID"])["Match_ID"]
    .to_numpy()
)

first_part = max(30, len(train_match_ids) // 2)
first_part = min(first_part, len(train_match_ids) - 4 * 2)
if first_part < 1 or first_part >= len(train_match_ids):
    raise ValueError("Not enough matches for chronological cross-validation")

later_ids = train_match_ids[first_part:]
validation_blocks = [
    block for block in np.array_split(later_ids, 4) if len(block)
]

cv_folds = []
train_count = first_part
for valid_ids in validation_blocks:
    fold_train_ids = train_match_ids[:train_count]
    fold_train_rows = np.flatnonzero(
        train_data["Match_ID"].isin(fold_train_ids).to_numpy()
    )
    fold_valid_rows = np.flatnonzero(
        train_data["Match_ID"].isin(valid_ids).to_numpy()
    )
    cv_folds.append((fold_train_rows, fold_valid_rows))
    train_count += len(valid_ids)


# ---------------------------------------------------------------------------
# Feature transformation used by the linear models.
# ---------------------------------------------------------------------------
class PreMatchTransform(BaseEstimator, TransformerMixin):
    """Optional log of FIFA ranks and optional clipping of the tournament
    variables to the range seen in the training fold."""

    rank_cols = ["Team_FIFA_Rank", "Opponent_FIFA_Rank"]
    clip_cols = extra_feats

    def __init__(self, log_rank=False, clip=False):
        self.log_rank = log_rank
        self.clip = clip

    def fit(self, X, y=None):
        cols = [c for c in self.clip_cols if c in X.columns]
        self.low_ = X[cols].min()
        self.high_ = X[cols].max()
        return self

    def transform(self, X):
        X = X.copy()
        if self.clip:
            for col in self.low_.index:
                X[col] = X[col].clip(self.low_[col], self.high_[col])
        if self.log_rank:
            for col in self.rank_cols:
                if col in X.columns:
                    X[col] = np.log(X[col])
        return X


def linear_pipe(model, log_rank=False, clip=False):
    return Pipeline([
        ("transform", PreMatchTransform(log_rank=log_rank, clip=clip)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def tree_pipe(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


ridge_alphas = {"model__alpha": [1, 3, 10, 30, 100, 300, 1000]}

# Family "Linear regression" = the task's required technique.
models = {
    "Mean baseline": ("Baseline", tree_pipe(DummyRegressor(strategy="mean")), [{}]),
    "OLS linear regression": ("Linear regression", linear_pipe(LinearRegression()), [{}]),
    "Ridge regression": ("Linear regression", linear_pipe(Ridge()), [ridge_alphas]),
    "Ridge + log FIFA rank": ("Linear regression", linear_pipe(Ridge(), log_rank=True), [ridge_alphas]),
    "Ridge + log rank + clipping": (
        "Linear regression", linear_pipe(Ridge(), log_rank=True, clip=True), [ridge_alphas]),
    "Poisson regression + log rank": (
        "Competing algorithm",
        linear_pipe(PoissonRegressor(max_iter=3000), log_rank=True),
        [{"model__alpha": [0.01, 0.1, 0.3, 1, 3, 10]}],
    ),
    "Random forest": (
        "Competing algorithm",
        tree_pipe(RandomForestRegressor(random_state=42, n_jobs=1)),
        [{
            "model__n_estimators": [100, 200],
            "model__max_depth": [2, 4, None],
            "model__min_samples_leaf": [3, 5],
            "model__max_features": [0.7, 1.0],
        }],
    ),
    "Gradient boosting": (
        "Competing algorithm",
        tree_pipe(GradientBoostingRegressor(random_state=42)),
        [{
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.03, 0.08],
            "model__max_depth": [1, 2],
        }],
    ),
}


def fit_search(model, grid, columns=features):
    search = GridSearchCV(
        estimator=model, param_grid=grid, scoring="neg_root_mean_squared_error",
        cv=cv_folds, n_jobs=-1, refit=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(train_data[columns], train_data[target_col])
    return search


# ---------------------------------------------------------------------------
# Exploratory data analysis
# ---------------------------------------------------------------------------
goals = data[target_col]
eda = {
    "mean": goals.mean(), "variance": goals.var(), "skew": goals.skew(),
    "zero_share": (goals == 0).mean(),
}
correlations = data[features + [target_col]].corr()[target_col].drop(target_col)
log_rank_corr = np.log(data["Team_FIFA_Rank"]).corr(goals)
rank_bands = pd.cut(data["Team_FIFA_Rank"], [0, 10, 20, 40, 90],
                    labels=["1–10", "11–20", "21–40", "41–85"])
goals_by_band = goals.groupby(rank_bands, observed=True).agg(["mean", "size"])
same_matches_played = bool(
    (data["Team_PreMatch_Matches_Played"] == data["Opponent_PreMatch_Matches_Played"]).all()
)
rest_day_corr = data["Team_PreMatch_Rest_Days"].corr(data["Opponent_PreMatch_Rest_Days"])
pd.DataFrame({"correlation_with_goals": correlations}).to_csv(out_folder / "eda_correlations.csv")
goals_by_band.to_csv(out_folder / "eda_goals_by_rank_band.csv")

# Train and compare the models.
all_results = []
model_results = {}
tuning_tables = []

for model_name, (family, model, param_grid) in models.items():
    search = fit_search(model, param_grid)
    predicted = search.predict(test_data[features])
    all_results.append({
        "Model": model_name,
        "Family": family,
        "CV_RMSE": -search.best_score_,
        "CV_RMSE_SD": search.cv_results_["std_test_score"][search.best_index_],
        "Test_MAE": mean_absolute_error(test_data[target_col], predicted),
        "Test_RMSE": rmse(test_data[target_col], predicted),
        "Test_R2": r2_score(test_data[target_col], predicted),
        "Best_Params": str(search.best_params_),
    })
    model_results[model_name] = (search, predicted)

    tested = pd.DataFrame(search.cv_results_["params"])
    tested.insert(0, "Model", model_name)
    tested["CV_RMSE"] = -search.cv_results_["mean_test_score"]
    tested["CV_RMSE_SD"] = search.cv_results_["std_test_score"]
    tuning_tables.append(tested)

results = pd.DataFrame(all_results).sort_values("CV_RMSE").reset_index(drop=True)
results.to_csv(out_folder / "model_comparison.csv", index=False)

tuning = pd.concat(tuning_tables, ignore_index=True)
tuning.columns = [c.replace("model__", "") for c in tuning.columns]
tuning.sort_values(["Model", "CV_RMSE"]).to_csv(out_folder / "tuning_all_settings.csv", index=False)

# The reported linear regression model is the linear-family model with the
# lowest CV RMSE. The best overall model is reported as a benchmark.
linear_rows = results[results["Family"] == "Linear regression"]
best_linear = linear_rows.iloc[0]["Model"]
best_overall = results.iloc[0]["Model"]
linear_search, linear_pred = model_results[best_linear]

pred_data = test_data[id_cols + [target_col]].copy()
pred_data["Predicted_Goals_Linear"] = linear_pred
pred_data["Predicted_Goals_Best_Overall"] = model_results[best_overall][1]
pred_data.to_csv(out_folder / "holdout_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# Inferential analysis: OLS on the development period with HC3 robust SEs.
# Team and opponent match counts are identical in every row, so one is
# dropped to avoid perfect collinearity. FIFA ranks are logged.
# ---------------------------------------------------------------------------
X_inf = train_data[features].copy()
X_inf = X_inf.fillna(X_inf.median())
X_inf = X_inf.drop(columns=["Opponent_PreMatch_Matches_Played"])
for col in ["Team_FIFA_Rank", "Opponent_FIFA_Rank"]:
    X_inf[col] = np.log(X_inf[col])
X_inf = X_inf.rename(columns={"Team_FIFA_Rank": "log_Team_FIFA_Rank",
                              "Opponent_FIFA_Rank": "log_Opponent_FIFA_Rank"})
X_const = sm.add_constant(X_inf)
ols_robust = sm.OLS(train_data[target_col], X_const).fit(cov_type="HC3")
ols_plain = sm.OLS(train_data[target_col], X_const).fit()
strength_cols = ["log_Team_FIFA_Rank", "log_Opponent_FIFA_Rank",
                 "Team_Elo_PreMatch", "Opponent_Elo_PreMatch"]
ols_reduced = sm.OLS(train_data[target_col], sm.add_constant(X_inf[strength_cols])).fit()
strength_test = ols_robust.f_test(", ".join(f"{c} = 0" for c in strength_cols))
extra_F, extra_p, extra_df = ols_plain.compare_f_test(ols_reduced)
vif = {
    col: variance_inflation_factor(X_const.values, i)
    for i, col in enumerate(X_const.columns) if col != "const"
}
coef_table = ols_robust.summary2().tables[1]
coef_table["VIF"] = pd.Series(vif)
coef_table.to_csv(out_folder / "ols_inference.csv")

# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------
# Check 1: wider Gradient Boosting grid (the first choice sat on a grid edge).
wide_search = fit_search(models["Gradient boosting"][1], {
    "model__n_estimators": [25, 50, 100, 200, 400],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__max_depth": [1, 2, 3],
})
wide_table = pd.DataFrame(wide_search.cv_results_["params"])
wide_table.columns = [c.replace("model__", "") for c in wide_table.columns]
wide_table["CV_RMSE"] = -wide_search.cv_results_["mean_test_score"]
wide_table.sort_values("CV_RMSE").to_csv(out_folder / "boosting_wide_grid.csv", index=False)

# Check 2: do the four Regression 2.2 variables add anything?
ablation_rows = []
for model_name in [best_linear, "Gradient boosting"]:
    _, base_model, grid = models[model_name]
    for set_name, columns in [("All eight", features),
                              ("Shared four only", shared_feats),
                              ("Regression 2.2 four only", extra_feats)]:
        s = fit_search(base_model, grid, columns)
        p = s.predict(test_data[columns])
        ablation_rows.append({
            "Model": model_name, "Feature_set": set_name,
            "CV_RMSE": -s.best_score_,
            "Holdout_RMSE": rmse(test_data[target_col], p),
            "Holdout_R2": r2_score(test_data[target_col], p),
        })
ablation = pd.DataFrame(ablation_rows)
ablation.to_csv(out_folder / "feature_set_ablation.csv", index=False)

# Check 3: bootstrap the holdout by match (both team rows stay together).
actual = test_data[target_col].to_numpy()
baseline_pred = model_results["Mean baseline"][1]
best_pred = model_results[best_overall][1]
match_ids = test_data["Match_ID"].to_numpy()
rows_by_match = {m: np.flatnonzero(match_ids == m) for m in np.unique(match_ids)}
rng = np.random.default_rng(42)
boot = {"linear_vs_baseline": [], "best_vs_baseline": [], "best_vs_linear": []}
for _ in range(5000):
    sample = rng.choice(list(rows_by_match), len(rows_by_match), replace=True)
    idx = np.concatenate([rows_by_match[m] for m in sample])
    r_base = rmse(actual[idx], baseline_pred[idx])
    r_lin = rmse(actual[idx], linear_pred[idx])
    r_best = rmse(actual[idx], best_pred[idx])
    boot["linear_vs_baseline"].append(r_base - r_lin)
    boot["best_vs_baseline"].append(r_base - r_best)
    boot["best_vs_linear"].append(r_lin - r_best)
point = {
    "linear_vs_baseline": rmse(actual, baseline_pred) - rmse(actual, linear_pred),
    "best_vs_baseline": rmse(actual, baseline_pred) - rmse(actual, best_pred),
    "best_vs_linear": rmse(actual, linear_pred) - rmse(actual, best_pred),
}
bootstrap = pd.DataFrame({
    "Comparison": list(point),
    "RMSE_gain": list(point.values()),
    "CI_2.5": [np.percentile(boot[k], 2.5) for k in point],
    "CI_97.5": [np.percentile(boot[k], 97.5) for k in point],
})
bootstrap.to_csv(out_folder / "holdout_bootstrap.csv", index=False)

# Check 4: permutation importance of the reported linear model on the holdout.
perm = permutation_importance(
    linear_search.best_estimator_, test_data[features], test_data[target_col],
    scoring="neg_root_mean_squared_error", n_repeats=200, random_state=42,
)
perm_table = pd.DataFrame({
    "Feature": features,
    "Group": ["Shared"] * 4 + ["Regression 2.2 only"] * 4,
    "RMSE_increase": perm.importances_mean,
    "SD": perm.importances_std,
}).sort_values("RMSE_increase", ascending=False)
perm_table.to_csv(out_folder / "permutation_importance.csv", index=False)
ridge_coef = pd.Series(
    linear_search.best_estimator_.named_steps["model"].coef_, index=features
)
ridge_coef.to_csv(out_folder / "linear_model_standardised_coefficients.csv")

# ---------------------------------------------------------------------------
# Figures (blue = shared / CV, orange = Regression 2.2 / holdout, grey = reference)
# ---------------------------------------------------------------------------
BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Figure 1: EDA.
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.4, 2.5),
                                 gridspec_kw={"width_ratios": [1, 1.1, 1.2]})
counts = goals.value_counts().sort_index()
a1.set_ylim(0, counts.max() * 1.35)
a1.bar(counts.index, counts.values, color=BLUE, width=0.75)
a1.set_xlabel("Goals scored")
a1.set_ylabel("Team-match rows")
a1.set_title("(a) Goals per team-match", loc="left", fontsize=8.5, color=INK)
a1.text(0.97, 0.97, f"mean {eda['mean']:.2f}\nvariance {eda['variance']:.2f}",
        transform=a1.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)

a2.bar(range(len(goals_by_band)), goals_by_band["mean"], color=BLUE, width=0.6)
a2.set_xticks(range(len(goals_by_band)))
a2.set_xticklabels(goals_by_band.index, fontsize=7)
for i, (m, n) in enumerate(zip(goals_by_band["mean"], goals_by_band["size"])):
    a2.text(i, m + 0.05, f"{m:.2f}", ha="center", fontsize=7.5, color=INK)
a2.set_xlabel("Team FIFA rank band")
a2.set_ylabel("Mean goals")
a2.set_ylim(0, goals_by_band["mean"].max() * 1.2)
a2.set_title("(b) Goals fall with rank", loc="left", fontsize=8.5, color=INK)

short_names = {
    "Team_FIFA_Rank": "Team FIFA rank", "Opponent_FIFA_Rank": "Opp. FIFA rank",
    "Team_Elo_PreMatch": "Team Elo", "Opponent_Elo_PreMatch": "Opp. Elo",
    "Team_PreMatch_Rest_Days": "Team rest days", "Opponent_PreMatch_Rest_Days": "Opp. rest days",
    "Team_PreMatch_Matches_Played": "Team matches played",
    "Opponent_PreMatch_Matches_Played": "Opp. matches played",
}
c_plot = correlations.reindex(features[::-1])
a3.barh([short_names[c] for c in c_plot.index], c_plot.values,
        color=[ORANGE if c in extra_feats else BLUE for c in c_plot.index], height=0.6)
a3.axvline(0, color=GREY, linewidth=0.8)
a3.set_xlim(-0.62, 0.42)
a3.set_xlabel("Correlation with goals (r)")
a3.tick_params(axis="y", labelsize=7)
a3.set_title("(c) Correlation with goals", loc="left", fontsize=8.5, color=INK)
a3.scatter([], [], marker="s", color=BLUE, label="Shared")
a3.scatter([], [], marker="s", color=ORANGE, label="Reg. 2.2 only")
a3.legend(loc="lower left", frameon=False, fontsize=6.5)
for ax in (a1, a2, a3):
    ax.grid(axis="x" if ax is a3 else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out_folder / "report_fig1_eda.png", dpi=220)
plt.close(fig)

# Figure 2: model comparison on one axis.
order = results.sort_values("CV_RMSE", ascending=False).reset_index(drop=True)
ypos = np.arange(len(order))
baseline_cv = float(results.loc[results["Model"] == "Mean baseline", "CV_RMSE"].iloc[0])
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.axvline(baseline_cv, color=GREY, linestyle="--", linewidth=1, zorder=1)
ax.text(baseline_cv + 0.006, len(order) - 0.45, "mean baseline CV RMSE",
        color=MUTED, fontsize=7.5, va="center")
ax.errorbar(order["CV_RMSE"], ypos + 0.13, xerr=order["CV_RMSE_SD"], fmt="o",
            color=BLUE, ecolor=BLUE, elinewidth=1, capsize=3, markersize=6,
            label="Chronological CV RMSE (mean ± fold SD)", zorder=3)
ax.scatter(order["Test_RMSE"], ypos - 0.13, marker="D", s=36, color=ORANGE,
           edgecolor="white", linewidth=1, label="Holdout RMSE (final 21 matches)", zorder=3)
labels = [f"{m}  [linear]" if f == "Linear regression" else m
          for m, f in zip(order["Model"], order["Family"])]
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8)
for tick, fam, name in zip(ax.get_yticklabels(), order["Family"], order["Model"]):
    if name == best_linear:
        tick.set_fontweight("bold")
ax.set_ylim(-0.5, len(order) - 0.2)
ax.set_xlabel("RMSE in goals (lower is better)")
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
handles, labs = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labs[::-1], loc="upper center", bbox_to_anchor=(0.4, -0.18),
          ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out_folder / "report_fig2_model_comparison.png", dpi=220)
plt.close(fig)

# Figure 3: (a) permutation importance, (b) actual vs predicted, linear model.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
p_plot = perm_table.sort_values("RMSE_increase")
ax1.barh(p_plot["Feature"], p_plot["RMSE_increase"], xerr=p_plot["SD"],
         color=[BLUE if g == "Shared" else ORANGE for g in p_plot["Group"]],
         height=0.6, error_kw={"elinewidth": 0.8, "ecolor": MUTED})
ax1.axvline(0, color=GREY, linewidth=0.8)
ax1.set_xlabel("Increase in holdout RMSE when shuffled")
ax1.set_title("(a) Permutation importance", loc="left", fontsize=9, color=INK)
ax1.tick_params(axis="y", labelsize=7)
ax1.scatter([], [], marker="s", color=BLUE, label="Shared (rank/Elo)")
ax1.scatter([], [], marker="s", color=ORANGE, label="Regression 2.2 only")
ax1.legend(loc="lower right", frameon=False, fontsize=7)
ax1.grid(axis="x", color=GRID, linewidth=0.8)
ax1.set_axisbelow(True)

jitter = np.random.default_rng(1).uniform(-0.12, 0.12, len(actual))
ax2.scatter(actual + jitter, linear_pred, s=24, color=BLUE, alpha=0.85,
            edgecolor="white", linewidth=0.6)
lim = [-0.4, max(actual.max(), linear_pred.max()) + 0.3]
ax2.plot(lim, lim, "--", color=GREY, linewidth=1)
ax2.text(4.1, 4.6, "perfect\nprediction", ha="right", fontsize=7.5, color=MUTED)
ax2.set_xlim(lim)
ax2.set_ylim(lim)
ax2.set_xlabel("Actual goals (jittered)")
ax2.set_ylabel("Predicted goals")
ax2.set_title("(b) Holdout: actual vs predicted", loc="left", fontsize=9, color=INK)
ax2.grid(color=GRID, linewidth=0.8)
ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out_folder / "report_fig3_importance_and_fit.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary = {
    "rows": int(len(data)), "matches": int(data["Match_ID"].nunique()),
    "teams": int(data["Team"].nunique()), "predictors": len(features),
    "missing_predictor_cells": int(data[features].isna().sum().sum()),
    "eda": {k: float(v) for k, v in eda.items()},
    "log_team_rank_corr": float(log_rank_corr),
    "team_and_opponent_matches_played_identical": same_matches_played,
    "team_opponent_rest_days_corr": float(rest_day_corr),
    "matches_played_range_train": [int(train_data["Team_PreMatch_Matches_Played"].min()),
                                   int(train_data["Team_PreMatch_Matches_Played"].max())],
    "matches_played_range_holdout": [int(test_data["Team_PreMatch_Matches_Played"].min()),
                                     int(test_data["Team_PreMatch_Matches_Played"].max())],
    "reported_linear_model": best_linear,
    "reported_linear_params": str(linear_search.best_params_),
    "best_overall_model": best_overall,
    "linear_prediction_range": [float(linear_pred.min()), float(linear_pred.max())],
    "ols_adj_r2_train": float(ols_robust.rsquared_adj),
    "strength_joint_F": float(strength_test.fvalue), "strength_joint_p": float(strength_test.pvalue),
    "extra_vars_partial_F": float(extra_F), "extra_vars_partial_p": float(extra_p),
    "gb_wide_grid_best": str(wide_search.best_params_),
}
with open(out_folder / "summary.json", "w") as fh:
    import json
    json.dump(summary, fh, indent=2)

print("REGRESSION 2.2 - DATA CHECK")
print("Rows:", len(data), "| Matches:", data["Match_ID"].nunique(),
      "| Teams:", data["Team"].nunique(), "| Predictors:", len(features))
print("Missing predictor cells before imputation:", summary["missing_predictor_cells"])
print("\nEDA:", {k: round(float(v), 3) for k, v in eda.items()})
print("Team and opponent matches played identical in every row:", same_matches_played)
print("\nMODEL RESULTS (lower CV RMSE is better)")
print(results[["Model", "Family", "CV_RMSE", "CV_RMSE_SD", "Test_MAE", "Test_RMSE", "Test_R2"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nReported linear regression model:", best_linear, linear_search.best_params_)
print("Best overall model:", best_overall)
print("\nOLS inference (development period, HC3 robust SEs):")
print(coef_table.round(3).to_string())
print(f"Joint test, rank and Elo: F = {strength_test.fvalue:.2f}, p = {strength_test.pvalue:.2g}")
print(f"Partial F-test, Regression 2.2 variables: F = {extra_F:.2f}, p = {extra_p:.3f}")
print("\nFeature-set ablation:")
print(ablation.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nHoldout bootstrap (5,000 match resamples):")
print(bootstrap.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nGradient Boosting wider grid best:", wide_search.best_params_)
print("\nResults and figures saved in:", out_folder)
