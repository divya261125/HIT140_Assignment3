
# HIT140 Assessment 3 - Regression 2.1
# Cross-validation and model optimisation

# 02_regression_model.py tested the model on ONE random 80/20 split
# (21 test matches). With only 104 matches, one split can be lucky or
# unlucky, so this script:
#   1. re-checks the original result on the same split,
#   2. cross-validates the original model on 50 different splits,
#   3. tries several ways to improve (optimise) the model,
#   4. picks the best version and checks its assumptions.

# Run it from the project folder (the folder that contains Data/).


# --------------------------------------------------
# STEP 1: Import the libraries we need
# --------------------------------------------------

import os

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

# statsmodels gives a full regression table with p-values
import statsmodels.api as sm

from sklearn import set_config
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.linear_model import ElasticNetCV, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

# Keep column names when data passes through each step of a pipeline
set_config(transform_output="pandas")

os.makedirs("Figures", exist_ok=True)
os.makedirs("Results", exist_ok=True)

# Same chart style as 03_eda_visualisations.py
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
LIGHT_BLUE = "#cde2fb"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK_2,
    "axes.edgecolor": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
})

nice_names = {
    "team_fifa_rank": "Team FIFA rank",
    "opponent_fifa_rank": "Opponent FIFA rank",
    "team_elo": "Team Elo",
    "opponent_elo": "Opponent Elo",
    "host_advantage": "Host advantage",
    "team_recent_goal_difference_avg": "Team recent goal diff (avg)",
    "opponent_recent_goals_conceded_avg": "Opponent recent goals conceded (avg)",
    "team_recent_win_percentage": "Team recent win %",
}


# --------------------------------------------------
# STEP 2: Load the final dataset
# --------------------------------------------------

matches = pd.read_csv(
    "Data/processed/regression21_final_dataset.csv",
    encoding="utf-8-sig"
)

variables = [
    "team_fifa_rank",
    "opponent_fifa_rank",
    "team_elo",
    "opponent_elo",
    "host_advantage",
    "team_recent_goal_difference_avg",
    "opponent_recent_goals_conceded_avg",
    "team_recent_win_percentage"
]

X = matches[variables]
y = matches["goal_difference"]

print("Matches:", len(X), "| Explanatory variables:", X.shape[1])


# Helper: the four accuracy measures used in this script
# NRMSE = RMSE divided by the range of goal difference (max - min)
def get_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": rmse,
        "NRMSE": rmse / (y.max() - y.min()),
        "R-squared": r2_score(actual, predicted),
    }


# --------------------------------------------------
# STEP 3: Re-check the original single-split result
# --------------------------------------------------

# Exactly the same split as 02_regression_model.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

original_model = LinearRegression().fit(X_train, y_train)
single_split_original = get_metrics(y_test, original_model.predict(X_test))

baseline_single = DummyRegressor(strategy="mean").fit(X_train, y_train)
single_split_baseline = get_metrics(y_test, baseline_single.predict(X_test))

print("\nOriginal model on the single 80/20 split (should match 02 script):")
for name, value in single_split_original.items():
    print(f"  {name}: {value:.3f}")
print(f"  Baseline MAE: {single_split_baseline['MAE']:.3f}")


# --------------------------------------------------
# STEP 4: Set up cross-validation
# --------------------------------------------------

# 5-fold cross-validation: the 104 matches are split into 5 groups of
# about 21 matches. Each group takes a turn as the test set while the
# model trains on the other 4 groups (about 83 matches).
# 5 folds keeps each test set the same size as the original 80/20 split.

# We repeat this 10 times with different random groupings,
# giving 5 x 10 = 50 test results for every model.
# random_state=42 means every model is tested on exactly the same
# 50 splits, so the comparison between models is fair.

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R-squared": "r2",
}


def cross_validate_model(model):
    """Return a table with one row per test fold (50 rows)."""
    # n_jobs=-1 uses every processor core to run folds in parallel
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return pd.DataFrame({
        # scikit-learn reports errors as negative numbers, so flip them
        "MAE": -scores["test_MAE"],
        "RMSE": -scores["test_RMSE"],
        "R-squared": scores["test_R-squared"],
    })


# --------------------------------------------------
# STEP 5: Cross-validate the original model
# --------------------------------------------------

original_cv = cross_validate_model(LinearRegression())
baseline_cv = cross_validate_model(DummyRegressor(strategy="mean"))

print("\nCross-validation, original model (mean +/- SD over 50 folds):")
for metric in ["MAE", "RMSE", "R-squared"]:
    print(f"  {metric}: {original_cv[metric].mean():.3f} "
          f"+/- {original_cv[metric].std():.3f}   "
          f"(fold range {original_cv[metric].min():.3f} "
          f"to {original_cv[metric].max():.3f})")
print(f"  Baseline MAE: {baseline_cv['MAE'].mean():.3f}")

# Where does the single split sit among the 50 folds?
r2_percentile = stats.percentileofscore(
    original_cv["R-squared"], single_split_original["R-squared"])
mae_percentile = stats.percentileofscore(
    original_cv["MAE"], single_split_original["MAE"])
print(f"\nThe single split's R-squared ({single_split_original['R-squared']:.3f})"
      f" is higher than {r2_percentile:.0f}% of the 50 CV folds.")
print(f"The single split's MAE ({single_split_original['MAE']:.3f})"
      f" is worse than {mae_percentile:.0f}% of the 50 CV folds.")

# Figure 7: the 50 fold results vs the one single-split result
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

panels = [
    (axes[0], "R-squared", "R-squared on the test fold"),
    (axes[1], "MAE", "MAE on the test fold (goals)"),
]

for ax, metric, xlabel in panels:
    ax.hist(original_cv[metric], bins=12, color=BLUE,
            edgecolor="white", linewidth=1)
    cv_mean = original_cv[metric].mean()
    single = single_split_original[metric]
    ax.axvline(cv_mean, color=INK, linewidth=2, linestyle="--",
               label=f"Cross-validation mean = {cv_mean:.3f}")
    ax.axvline(single, color=ORANGE, linewidth=2.5,
               label=f"Single 80/20 split = {single:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of folds")
    ax.set_title(metric)
    ax.legend(loc="upper left", fontsize=9, framealpha=1,
              facecolor="white", edgecolor="none")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)

fig.suptitle("Original model: 50 cross-validation test folds "
             "vs the single 80/20 split",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig("Figures/fig07_cv_vs_single_split.png")
plt.close(fig)


# --------------------------------------------------
# STEP 6: Build the optimisation candidates
# --------------------------------------------------

# Each candidate is a "pipeline": the data-preparation steps and the
# regression model are bundled together. During cross-validation every
# step is fitted on the training folds only, so no information from
# the test fold leaks into training.


# Option A: log-transform the FIFA ranks.
# Moving from rank 1 to rank 10 matters more than moving from 60 to 70,
# and the rank variables are right-skewed. log() compresses large ranks.
def log_fifa_ranks(data):
    data = data.copy()
    data["team_fifa_rank"] = np.log(data["team_fifa_rank"])
    data["opponent_fifa_rank"] = np.log(data["opponent_fifa_rank"])
    return data


log_ranks = FunctionTransformer(log_fifa_ranks)

print("\nSkewness before and after the log transform:")
for var in ["team_fifa_rank", "opponent_fifa_rank"]:
    print(f"  {var}: {stats.skew(X[var]):.2f} -> "
          f"{stats.skew(np.log(X[var])):.2f}")


print("\nVIF check (removing collinear variables):")


# Option B: remove collinear variables using VIF.
# Drop the variable with the highest VIF, recalculate, and repeat
# until every VIF is 5 or lower. (VIF only uses the explanatory
# variables, not goal difference, so it does not leak the answer.)
def calculate_vif(data, columns_to_check):
    vif_values = {}
    for var in columns_to_check:
        others = [c for c in columns_to_check if c != var]
        fit = LinearRegression().fit(data[others], data[var])
        vif_values[var] = 1 / (1 - fit.score(data[others], data[var]))
    return pd.Series(vif_values)


low_vif_variables = variables.copy()
while True:
    vif = calculate_vif(X, low_vif_variables)
    if vif.max() <= 5:
        break
    print(f"  Removing {vif.idxmax()} (VIF = {vif.max():.2f})")
    low_vif_variables.remove(vif.idxmax())

print("Variables kept after the VIF check:", low_vif_variables)


def keep_low_vif_variables(data):
    return data[low_vif_variables]


low_vif_only = FunctionTransformer(keep_low_vif_variables)


# Option C: competing algorithms (stage 2 below).
# Ridge, Lasso and Elastic Net shrink the coefficients towards zero, which
# makes the model more stable when variables overlap (like FIFA rank and
# Elo). Huber regression is a linear model that gives less weight to very
# large errors, such as the blowout wins seen in the data.
# K-nearest neighbours, random forest and gradient boosting are NOT linear
# regression. They are included only as benchmarks: if one of them were
# clearly better, a straight-line model would be missing something.
#
# Every setting that needs tuning (alpha, epsilon, number of neighbours,
# tree depth...) is chosen using ONLY the training folds: RidgeCV, LassoCV,
# ElasticNetCV and GridSearchCV run their own inner cross-validation.
alphas = np.logspace(-3, 3, 61)


# A fair test for comparing two models across repeated k-fold CV.
# The 50 folds share training data, so an ordinary paired t-test would be
# far too confident. The corrected resampled t-test (Nadeau and Bengio,
# 2003) inflates the variance to allow for that overlap.
n_test = len(X) / 5
n_train = len(X) - n_test


def corrected_t_test(model_mae, reference_mae):
    differences = (model_mae - reference_mae).round(6).values
    k = len(differences)
    mean_difference = differences.mean()
    variance = differences.var(ddof=1)
    if variance == 0:
        return mean_difference, 0.0, 0.0, 1.0
    standard_error = np.sqrt((1 / k + n_test / n_train) * variance)
    t_value = mean_difference / standard_error
    p_value = 2 * stats.t.sf(abs(t_value), df=k - 1)
    margin = stats.t.ppf(0.975, df=k - 1) * standard_error
    return (mean_difference, mean_difference - margin,
            mean_difference + margin, p_value)


def as_pipeline(fitted_model):
    """GridSearchCV wraps its best pipeline; this unwraps it."""
    return getattr(fitted_model, "best_estimator_", fitted_model)


# --------------------------------------------------
# STEP 7: Stage 1 - which version of the inputs works best?
# --------------------------------------------------

# All stage 1 models use plain linear regression, so the only thing
# that changes is how the eight variables are prepared.
input_options = {
    "original variables": [],
    "standardised": [StandardScaler()],
    "log FIFA ranks": [clone(log_ranks)],
    "Yeo-Johnson transform": [PowerTransformer(method="yeo-johnson")],
    "collinear variables removed": [clone(low_vif_only)],
}

models = {}
model_stage = {}
model_family = {}

for input_name, steps in input_options.items():
    name = f"Linear regression: {input_name}"
    models[name] = make_pipeline(*[clone(s) for s in steps],
                                 LinearRegression())
    model_stage[name] = "Stage 1: inputs"
    model_family[name] = "Linear"

original_name = "Linear regression: original variables"

fold_scores = {"Baseline (predict the mean)": baseline_cv}
for name in models:
    fold_scores[name] = cross_validate_model(models[name])

stage1_names = list(models)
best_input_name = min(stage1_names,
                      key=lambda n: fold_scores[n]["MAE"].mean())
best_input = best_input_name.replace("Linear regression: ", "")
best_steps = input_options[best_input]

print(f"\nStage 1 winner: {best_input_name} "
      f"(CV MAE {fold_scores[best_input_name]['MAE'].mean():.3f})")


# --------------------------------------------------
# STEP 8: Stage 2 - competing algorithms on the best inputs
# --------------------------------------------------

def with_best_inputs(*final_steps):
    return make_pipeline(*[clone(s) for s in best_steps], *final_steps)


stage2 = {
    "Linear regression (least squares)": (
        "Linear", None),
    "Ridge regression": (
        "Linear", with_best_inputs(StandardScaler(), RidgeCV(alphas=alphas))),
    "Lasso regression": (
        "Linear", with_best_inputs(StandardScaler(),
                                   LassoCV(cv=5, max_iter=50000))),
    "Elastic Net": (
        "Linear", with_best_inputs(
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5,
                         max_iter=50000))),
    "Huber robust regression": (
        "Linear", GridSearchCV(
            with_best_inputs(StandardScaler(),
                             HuberRegressor(max_iter=1000)),
            {"huberregressor__epsilon": [1.35, 1.75, 2.5],
             "huberregressor__alpha": [0.0001, 10]},
            cv=5, scoring="neg_mean_absolute_error")),
    "K-nearest neighbours": (
        "Non-linear benchmark", GridSearchCV(
            with_best_inputs(StandardScaler(), KNeighborsRegressor()),
            {"kneighborsregressor__n_neighbors": [5, 10, 15, 20, 30]},
            cv=5, scoring="neg_mean_absolute_error")),
    "Random forest": (
        "Non-linear benchmark", GridSearchCV(
            RandomForestRegressor(n_estimators=150, random_state=42),
            {"max_depth": [2, 4, None], "min_samples_leaf": [5, 10]},
            cv=5, scoring="neg_mean_absolute_error")),
    "Gradient boosting": (
        "Non-linear benchmark", GridSearchCV(
            GradientBoostingRegressor(learning_rate=0.05, random_state=42),
            {"max_depth": [1, 2], "n_estimators": [100, 300]},
            cv=5, scoring="neg_mean_absolute_error")),
}

print("\nStage 2: cross-validating the competing algorithms "
      "(this takes a few minutes)...")

for name, (family, model) in stage2.items():
    if model is None:
        # Same model as the stage 1 winner, so reuse its scores
        models[name] = models[best_input_name]
        fold_scores[name] = fold_scores[best_input_name]
    else:
        models[name] = model
        fold_scores[name] = cross_validate_model(model)
    model_stage[name] = f"Stage 2: algorithms ({best_input})"
    model_family[name] = family
    print(f"  {name}: CV MAE {fold_scores[name]['MAE'].mean():.3f}")


# --------------------------------------------------
# STEP 9: Compare everything and choose the final model
# --------------------------------------------------

original_folds = fold_scores[original_name]

rows = []
for name, scores in fold_scores.items():
    diff, ci_low, ci_high, p_value = corrected_t_test(
        scores["MAE"], original_folds["MAE"])
    is_baseline = name.startswith("Baseline")
    rows.append({
        "Stage": "Reference" if is_baseline else model_stage[name],
        "Family": "Reference" if is_baseline else model_family[name],
        "Model": name,
        "CV MAE": scores["MAE"].mean(),
        "CV MAE SD": scores["MAE"].std(),
        "CV RMSE": scores["RMSE"].mean(),
        "CV RMSE SD": scores["RMSE"].std(),
        "CV R-squared": scores["R-squared"].mean(),
        "CV R-squared SD": scores["R-squared"].std(),
        "MAE change vs original": diff,
        "95% CI low": ci_low,
        "95% CI high": ci_high,
        "p-value (corrected t-test)": p_value,
        "% folds better than original (MAE)":
            np.nan if name == original_name
            else ((scores["MAE"] - original_folds["MAE"]).round(6) < 0)
            .mean() * 100,
    })

comparison = pd.DataFrame(rows)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("\nModel comparison (5-fold CV repeated 10 times):")
print(comparison.drop(columns=["CV RMSE SD"]).round(3).to_string(index=False))

comparison.round(4).to_csv("Results/cv_model_comparison.csv", index=False)

all_folds = pd.concat(fold_scores, names=["Model", "Fold"]).reset_index()
all_folds.round(4).to_csv("Results/cv_fold_scores_all_models.csv",
                          index=False)

# Rule, fixed before looking at the results: the final model is the
# LINEAR model with the lowest mean cross-validated MAE (the brief asks
# for linear regression, and MAE is the original script's main metric).
# Note: because the winner is chosen using these same folds, its CV score
# is slightly optimistic.
linear_rows = comparison[(comparison["Family"] == "Linear")]
final_name = linear_rows.sort_values("CV MAE").iloc[0]["Model"]
final_row = comparison.set_index("Model").loc[final_name]

best_benchmark = comparison[comparison["Family"] == "Non-linear benchmark"] \
    .sort_values("CV MAE").iloc[0]

print("\nFinal model chosen:", final_name, f"with {best_input}")
print(f"  CV MAE {final_row['CV MAE']:.3f} vs original "
      f"{original_folds['MAE'].mean():.3f}; better in "
      f"{final_row['% folds better than original (MAE)']:.0f}% of folds; "
      f"corrected t-test p = {final_row['p-value (corrected t-test)']:.3f}")
print(f"  Best non-linear benchmark: {best_benchmark['Model']} "
      f"(CV MAE {best_benchmark['CV MAE']:.3f})")

# Figure 8: change in MAE vs the original model, with 95% intervals
GREY = "#9a9993"
plot_rows = comparison[comparison["Stage"] != "Reference"]
stages = list(dict.fromkeys(plot_rows["Stage"]))

fig, axes = plt.subplots(
    len(stages), 1, figsize=(11, 7.5), sharex=True,
    gridspec_kw={"height_ratios": [
        (plot_rows["Stage"] == s).sum() for s in stages]})

for ax, stage in zip(axes, stages):
    part = plot_rows[plot_rows["Stage"] == stage].iloc[::-1]
    for i, (_, r) in enumerate(part.iterrows()):
        if r["Model"] == final_name:
            colour = ORANGE
        elif r["Family"] == "Linear":
            colour = BLUE
        else:
            colour = GREY
        if r["Model"] != original_name:
            ax.plot([r["95% CI low"], r["95% CI high"]], [i, i],
                    color=colour, linewidth=2.5, solid_capstyle="round")
        ax.scatter(r["MAE change vs original"], i, s=70, color=colour,
                   zorder=3, edgecolor="white", linewidth=1)
        ax.text(0.33, i, f"MAE {r['CV MAE']:.3f}", va="center",
                fontsize=9, color=INK_2)
    ax.set_yticks(range(len(part)))
    ax.set_yticklabels(part["Model"])
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_title(stage[0].upper() + stage[1:], loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(part) - 0.4)

axes[-1].set_xlim(-0.3, 0.42)
axes[-1].set_xlabel("Change in cross-validated MAE vs the original model "
                    "(goals; negative = better), with 95% interval")

fig.suptitle("Every candidate compared on the same 50 folds "
             "(orange = final model, grey = non-linear benchmark)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig("Figures/fig08_model_comparison.png")
plt.close(fig)




# --------------------------------------------------
# STEP 10: Final model - like-for-like results table
# --------------------------------------------------

final_label = f"{final_name} + {best_input}"

final_model = clone(models[final_name])

# (a) the same single 80/20 split as 02_regression_model.py
final_model.fit(X_train, y_train)
single_split_final = get_metrics(y_test, final_model.predict(X_test))


# (b) cross-validation averages
def cv_summary(scores):
    return {
        "MAE": scores["MAE"].mean(),
        "RMSE": scores["RMSE"].mean(),
        "NRMSE": scores["RMSE"].mean() / (y.max() - y.min()),
        "R-squared": scores["R-squared"].mean(),
    }


summary_rows = [
    ("Single 80/20 split", "Baseline (predict the mean)",
     single_split_baseline),
    ("Single 80/20 split", "Original linear regression",
     single_split_original),
    ("Single 80/20 split", final_label, single_split_final),
    ("5-fold CV x 10 (mean)", "Baseline (predict the mean)",
     cv_summary(baseline_cv)),
    ("5-fold CV x 10 (mean)", "Original linear regression",
     cv_summary(original_cv)),
    ("5-fold CV x 10 (mean)", final_label,
     cv_summary(fold_scores[final_name])),
]

final_summary = pd.DataFrame(
    [{"Evaluation": e, "Model": m, **v} for e, m, v in summary_rows]
)

print("\nFinal results table:")
print(final_summary.round(3).to_string(index=False))

final_summary.round(4).to_csv("Results/final_model_summary.csv",
                              index=False)


# --------------------------------------------------
# STEP 11: Fit the final model on all 104 matches
# --------------------------------------------------

final_model = as_pipeline(clone(models[final_name]).fit(X, y))

if hasattr(final_model[-1], "alpha_"):
    print(f"\nPenalty strength chosen on all 104 matches: "
          f"alpha = {final_model[-1].alpha_:.3f}")


def standardised_coefficients(fitted_model):
    """Coefficient for a one-standard-deviation change in each variable,
    so variables measured on different scales can be compared."""
    fitted_pipeline = as_pipeline(fitted_model)
    if len(fitted_pipeline.steps) > 1:
        transformed = fitted_pipeline[:-1].transform(X)
    else:
        transformed = X
    coefs = fitted_pipeline[-1].coef_ * transformed.std(ddof=0).values
    return pd.Series(coefs, index=transformed.columns)


original_full = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)

coefficients = pd.DataFrame({
    "Original linear regression (standardised)":
        standardised_coefficients(original_full),
    f"{final_label} (standardised)":
        standardised_coefficients(final_model),
}).reindex(variables)

print("\nStandardised coefficients (goals per 1 SD change):")
print(coefficients.round(3))

coefficients.round(4).to_csv("Results/final_model_coefficients.csv",
                             index_label="Variable")


# Coefficient stability: refit both models on each of the 50 training
# sets and record every coefficient.
def refit_coefficients(model):
    fitted_coefs = []
    for train_index, _ in cv.split(X):
        fitted = clone(model).fit(X.iloc[train_index], y.iloc[train_index])
        fitted_coefs.append(standardised_coefficients(fitted))
    return pd.DataFrame(fitted_coefs)[variables]


refits = {
    "Original": refit_coefficients(
        make_pipeline(StandardScaler(), LinearRegression())),
    "Final": refit_coefficients(models[final_name]),
}

stability = pd.concat({
    name: pd.DataFrame({
        "Mean": table.mean(),
        "SD": table.std(),
        "% folds positive": (table > 0).mean() * 100,
    })
    for name, table in refits.items()
}, axis=1)

print("\nCoefficient stability across the 50 training sets:")
print(stability.round(3).to_string())

stability.round(4).to_csv("Results/coefficient_stability.csv",
                          index_label="Variable")

# Figure 9: standardised coefficients, original vs final
fig, ax = plt.subplots(figsize=(10, 5.5))

positions = np.arange(len(variables))
bar_height = 0.36

ax.barh(positions - bar_height / 2, coefficients.iloc[:, 0],
        height=bar_height, color=BLUE, label="Original linear regression")
ax.barh(positions + bar_height / 2, coefficients.iloc[:, 1],
        height=bar_height, color=ORANGE, label=final_label)

ax.set_yticks(positions)
ax.set_yticklabels([nice_names[v] for v in variables])
ax.invert_yaxis()
ax.axvline(0, color=INK_2, linewidth=1)
ax.set_xlabel("Change in predicted goal difference for a "
              "1 SD increase in the variable")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
          ncol=2)
ax.grid(axis="y", visible=False)
ax.set_title("Standardised coefficients: original vs final "
             "model\n(the final model uses the log of each FIFA rank)",
             loc="left", pad=30)

fig.tight_layout()
fig.savefig("Figures/fig09_coefficients.png")
plt.close(fig)

# Figure 10: every coefficient from the 50 refits
fig, ax = plt.subplots(figsize=(10, 6.5))
rng = np.random.default_rng(42)

for i, var in enumerate(variables):
    for offset, name, colour in [(-0.17, "Original", BLUE),
                                 (0.17, "Final", ORANGE)]:
        values = refits[name][var]
        ax.scatter(values, i + offset + rng.uniform(-0.07, 0.07, len(values)),
                   s=14, color=colour, alpha=0.55, edgecolor="none",
                   label=name if i == 0 else None)

ax.set_yticks(range(len(variables)))
ax.set_yticklabels([nice_names[v] for v in variables])
ax.invert_yaxis()
ax.axvline(0, color=INK_2, linewidth=1)
ax.set_xlabel("Standardised coefficient in each of the 50 refits")
ax.grid(axis="y", visible=False)
legend = ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
                   ncol=2, markerscale=1.8,
                   labels=["Original linear regression", final_label])
ax.set_title("Coefficients refitted on 50 training sets: dots that cross "
             "zero mean the sign flips", loc="left", pad=30)

fig.tight_layout()
fig.savefig("Figures/fig10_coefficient_stability.png")
plt.close(fig)


# --------------------------------------------------
# STEP 12: Check the final model's assumptions (residuals)
# --------------------------------------------------

# Out-of-fold predictions: each match is predicted by a model that
# never saw that match during training.
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = cross_val_predict(clone(models[final_name]),
                                    X, y, cv=kfold)
residuals = y - oof_predictions

oof = pd.DataFrame({
    "match_id": matches["match_id"],
    "team": matches["team"],
    "opponent": matches["opponent"],
    "round": matches["round"],
    "actual_goal_difference": y,
    "predicted_goal_difference": oof_predictions,
    "residual": residuals,
})
oof.round(3).to_csv("Results/final_model_oof_predictions.csv",
                    index=False)

oof_metrics = get_metrics(y, oof_predictions)
print("\nOut-of-fold predictions for all 104 matches:")
for name, value in oof_metrics.items():
    print(f"  {name}: {value:.3f}")

# Did the model at least pick the right winner?
predicted_sign = np.sign(np.round(oof_predictions))
direction_correct = (predicted_sign == np.sign(y)).mean() * 100
print(f"  Correct result (win/draw/loss, prediction rounded): "
      f"{direction_correct:.0f}%")
print(f"  For comparison, always predicting a win for the first-listed "
      f"team is correct {(y > 0).mean() * 100:.0f}% of the time")

# Shapiro-Wilk test: are the residuals roughly normal?
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"  Shapiro-Wilk test on residuals: W = {shapiro_stat:.3f}, "
      f"p = {shapiro_p:.3f}")

largest_errors = oof.reindex(
    oof["residual"].abs().sort_values(ascending=False).index).head(6)
print("\nLargest prediction errors:")
print(largest_errors.round(2).to_string(index=False))

# Figure 11: actual vs predicted, with the biggest misses named
fig, ax = plt.subplots(figsize=(8, 7))
y_jittered = y + rng.uniform(-0.12, 0.12, len(y))
ax.scatter(oof_predictions, y_jittered, s=30, color=BLUE, alpha=0.65,
           edgecolor="white", linewidth=0.5)
limits = [-5, 7.5]
ax.plot(limits, limits, color=INK_2, linewidth=1.2, linestyle="--",
        label="Perfect prediction")

label_offsets = {
    27: (-0.6, 0.6, "right"),   # Canada 6-0 Qatar
    9: (0.5, 0.6, "left"),      # Germany 7-1 Curacao
    62: (-0.5, 0.5, "right"),   # Senegal 5-0 Iraq
    45: (0.7, -0.1, "left"),    # Portugal 5-0 Uzbekistan
    14: (0.1, -0.9, "center"),  # Spain 0-0 Cabo Verde
    94: (0.5, -0.6, "left"),    # USA 1-4 Belgium
}

for index, r in largest_errors.iterrows():
    score = (f"{matches.loc[index, 'team_goals']}–"
             f"{matches.loc[index, 'opponent_goals']}")
    label = f"{r['team']} {score} {r['opponent']}"
    x_point = r["predicted_goal_difference"]
    y_point = y_jittered[index]
    # Label positions chosen by hand so the names do not overlap
    # (match_id: x shift, y shift, alignment); others go up and right
    dx, dy, align = label_offsets.get(r["match_id"], (0.4, 0.4, "left"))
    ax.scatter(x_point, y_point, s=46, color=ORANGE, zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.annotate(label, (x_point, y_point),
                xytext=(x_point + dx, y_point + dy),
                ha=align, va="center", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))

ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_xlabel("Predicted goal difference (out-of-fold)")
ax.set_ylabel("Actual goal difference")
ax.legend(frameon=False, loc="upper left")
ax.text(7.2, -4.7,
        f"One 5-fold run:\n"
        f"MAE = {oof_metrics['MAE']:.2f}\n"
        f"RMSE = {oof_metrics['RMSE']:.2f}",
        ha="right", va="bottom", fontsize=9.5, color=INK_2)
ax.set_title("Final model: actual vs predicted, with the six biggest "
             "misses named\n(each match predicted by a model that did not "
             "see it)", loc="left")
fig.tight_layout()
fig.savefig("Figures/fig11_actual_vs_predicted.png")
plt.close(fig)

# Figure 12: residual diagnostics
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

axes[0].scatter(oof_predictions, residuals, s=24, color=BLUE, alpha=0.7,
                edgecolor="white", linewidth=0.5)
axes[0].axhline(0, color=ORANGE, linewidth=2)
axes[0].set_xlabel("Predicted goal difference")
axes[0].set_ylabel("Residual (actual - predicted)")
axes[0].set_title("Residuals vs predicted", loc="left")

axes[1].hist(residuals, bins=14, color=BLUE, edgecolor="white",
             linewidth=1)
axes[1].axvline(0, color=ORANGE, linewidth=2)
axes[1].set_xlabel("Residual (goals)")
axes[1].set_ylabel("Number of matches")
axes[1].set_title("Distribution of residuals", loc="left")
axes[1].grid(axis="x", visible=False)

(theoretical, ordered), (slope, intercept, _) = stats.probplot(residuals)
axes[2].scatter(theoretical, ordered, s=22, color=BLUE, alpha=0.75,
                edgecolor="white", linewidth=0.5)
axes[2].plot(theoretical, intercept + slope * theoretical, color=ORANGE,
             linewidth=2)
axes[2].set_xlabel("Theoretical normal quantiles")
axes[2].set_ylabel("Residual quantiles")
axes[2].set_title(f"Normal Q-Q plot (Shapiro-Wilk p = {shapiro_p:.2f})",
                  loc="left")

fig.suptitle("Residual checks for the final model "
             "(out-of-fold predictions)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig("Figures/fig12_residual_diagnostics.png")
plt.close(fig)


# --------------------------------------------------
# STEP 13: Would more matches help? (learning curve)
# --------------------------------------------------

# Train each model on growing numbers of matches and test it on the
# held-out fold, using the same 50 splits.
train_sizes = [25, 35, 45, 55, 65, 75, 83]

curve_models = {
    "Baseline (predict the mean)": (DummyRegressor(strategy="mean"), GREY),
    "Original linear regression": (LinearRegression(), BLUE),
    final_label: (models[final_name], ORANGE),
}

curve_rows = []
fig, ax = plt.subplots(figsize=(9, 5.2))

for label, (model, colour) in curve_models.items():
    sizes, _, test_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv,
        scoring="neg_mean_absolute_error", n_jobs=-1)
    mae = -test_scores
    means, sds = mae.mean(axis=1), mae.std(axis=1)
    for size, m, sd in zip(sizes, means, sds):
        curve_rows.append({"Model": label, "Training matches": size,
                           "CV MAE": m, "CV MAE SD": sd})
    ax.plot(sizes, means, marker="o", color=colour, linewidth=2,
            label=label,
            linestyle="--" if label.startswith("Baseline") else "-")

learning = pd.DataFrame(curve_rows)
learning.round(4).to_csv("Results/learning_curve.csv", index=False)
print("\nLearning curve (mean CV MAE by number of training matches):")
print(learning.pivot(index="Training matches", columns="Model",
                     values="CV MAE").round(3))

ax.set_xlabel("Number of matches used for training")
ax.set_ylabel("Cross-validated MAE (goals)")
ax.set_ylim(1.1, min(3.0, learning["CV MAE"].max() + 0.1))
ax.legend(frameon=False)
ax.set_title("Learning curve: error on unseen matches as the training set "
             "grows (mean of 50 folds)", loc="left")
fig.tight_layout()
fig.savefig("Figures/fig13_learning_curve.png")
plt.close(fig)


# --------------------------------------------------
# STEP 14: Full statistical summary of the original OLS model
# --------------------------------------------------

# statsmodels fits the same linear regression on all 104 matches and
# reports p-values and adjusted R-squared. This describes the fit on
# the data the model was trained on, so it is more optimistic than the
# cross-validated results above.
ols = sm.OLS(y, sm.add_constant(X)).fit()

print(f"\nOLS on all 104 matches: R-squared = {ols.rsquared:.3f}, "
      f"adjusted R-squared = {ols.rsquared_adj:.3f}, "
      f"F-test p = {ols.f_pvalue:.2e}")

with open("Results/ols_full_sample_summary.txt", "w") as f:
    f.write(ols.summary().as_text())

print("\nAll figures saved in Figures/ and tables saved in Results/")
