# HIT140 Assessment 3 - Regression 2.2
# Predict the goals scored by a team in one World Cup match.

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


# Build each model. Missing feature values are filled using the training data.
mean_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DummyRegressor(strategy="mean")),
])

linear_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])

ridge_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge()),
])

forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
])

boost_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GradientBoostingRegressor(random_state=42)),
])

models = {
    "Mean baseline": (mean_model, [{}]),
    "Linear regression": (linear_model, [{}]),
    "Ridge regression": (ridge_model, [{"model__alpha": [0.1, 1, 10, 100]}]),
    "Random forest": (
        forest_model,
        [{
            "model__n_estimators": [100, 200],
            "model__max_depth": [2, 4, None],
            "model__min_samples_leaf": [3, 5],
            "model__max_features": [0.7, 1.0],
        }],
    ),
    "Gradient boosting": (
        boost_model,
        [{
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.03, 0.08],
            "model__max_depth": [1, 2],
        }],
    ),
}


# Train and compare the models.
all_results = []
model_results = {}

for model_name, (model, param_grid) in models.items():
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
        refit=True,
    )

    # Ignore non-critical fitting warnings so the result table stays readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(train_data[features], train_data[target_col])

    predicted = search.predict(test_data[features])

    all_results.append({
        "Model": model_name,
        "CV_RMSE": -search.best_score_,
        "CV_RMSE_SD": search.cv_results_["std_test_score"][search.best_index_],
        "Test_MAE": mean_absolute_error(test_data[target_col], predicted),
        "Test_RMSE": np.sqrt(mean_squared_error(test_data[target_col], predicted)),
        "Test_R2": r2_score(test_data[target_col], predicted),
        "Best_Params": str(search.best_params_),
    })
    model_results[model_name] = (search, predicted)

results = pd.DataFrame(all_results)
results = results.sort_values("CV_RMSE").reset_index(drop=True)
results.to_csv(out_folder / "model_comparison.csv", index=False)

# Choose the model with the lowest cross-validation RMSE.
best_name = results.loc[0, "Model"]
best_search, best_predictions = model_results[best_name]

pred_data = test_data[id_cols + [target_col]].copy()
pred_data["Predicted_Goals"] = best_predictions
pred_data["Residual"] = pred_data[target_col] - pred_data["Predicted_Goals"]
pred_data.to_csv(out_folder / "holdout_predictions.csv", index=False)


# Figure 1: goal totals in the dataset.
goal_counts = data[target_col].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(goal_counts.index.astype(str), goal_counts.values)
ax.set_xlabel("Goals scored in a match")
ax.set_ylabel("Number of team-match rows")
ax.set_title("Distribution of goals scored")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(out_folder / "target_distribution.png", dpi=180)
plt.close(fig)

# Figure 2: two of the extra predictors compared with goals scored.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
axes[0].scatter(data["Team_PreMatch_Rest_Days"], data[target_col], alpha=0.65)
axes[0].set_xlabel("Team rest days before match")
axes[0].set_ylabel("Goals scored")
axes[0].set_title("Rest days and goals")
axes[0].grid(alpha=0.2)

axes[1].scatter(data["Team_PreMatch_Matches_Played"], data[target_col], alpha=0.65)
axes[1].set_xlabel("Team's earlier World Cup matches")
axes[1].set_ylabel("Goals scored")
axes[1].set_title("Earlier matches and goals")
axes[1].grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out_folder / "feature_relationships.png", dpi=180)
plt.close(fig)

# Figure 3: compare the models using cross-validation RMSE.
plot_data = results.sort_values("CV_RMSE")
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(plot_data["Model"], plot_data["CV_RMSE"])
ax.invert_yaxis()
ax.set_xlabel("Mean chronological CV RMSE (lower is better)")
ax.set_title("Comparison of regression models")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(out_folder / "model_comparison.png", dpi=180)
plt.close(fig)

# Figure 4: compare the predictions with the actual goals.
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(pred_data[target_col], pred_data["Predicted_Goals"], alpha=0.75)
low = min(pred_data[target_col].min(), pred_data["Predicted_Goals"].min())
high = max(pred_data[target_col].max(), pred_data["Predicted_Goals"].max())
ax.plot([low, high], [low, high], "--", linewidth=1)
ax.set_xlabel("Actual goals")
ax.set_ylabel("Predicted goals")
ax.set_title("Actual vs predicted: " + best_name)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(out_folder / "actual_vs_predicted.png", dpi=180)
plt.close(fig)

# Figure 5: residuals are the actual goals minus the predicted goals.
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(pred_data["Predicted_Goals"], pred_data["Residual"], alpha=0.75)
ax.axhline(0, linestyle="--", linewidth=1)
ax.set_xlabel("Predicted goals")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residual check on the holdout set")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(out_folder / "residuals.png", dpi=180)
plt.close(fig)

# Figure 6: feature importance, when the selected model provides it.
selected_model = best_search.best_estimator_.named_steps["model"]
feature_importance = None

if hasattr(selected_model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": selected_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    feature_importance.to_csv(out_folder / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_data = feature_importance.sort_values("Importance")
    ax.barh(plot_data["Feature"], plot_data["Importance"])
    ax.set_xlabel("Relative feature importance")
    ax.set_title("Predictor importance in the selected model")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_folder / "feature_importance.png", dpi=180)
    plt.close(fig)


# Save a short summary with the main results.
summary = {
    "rows": int(len(data)),
    "matches": int(data["Match_ID"].nunique()),
    "teams": int(data["Team"].nunique()),
    "predictors": len(features),
    "missing_predictor_cells": int(data[features].isna().sum().sum()),
    "selected_model": best_name,
    "cv_rmse": float(results.loc[0, "CV_RMSE"]),
    "test_mae": float(results.loc[0, "Test_MAE"]),
    "test_rmse": float(results.loc[0, "Test_RMSE"]),
    "test_r2": float(results.loc[0, "Test_R2"]),
    "shared_predictors": shared_feats,
    "regression_2_2_specific_predictors": extra_feats,
    "feature_importance_available": feature_importance is not None,
}
pd.Series(summary).to_json(out_folder / "summary.json", indent=2)

print("REGRESSION 2.2 - DATA CHECK")
print("Rows:", len(data))
print("Matches:", data["Match_ID"].nunique())
print("Teams:", data["Team"].nunique())
print("Predictors:", len(features))
print("Target:", target_col)
print("Missing predictor cells before imputation:", summary["missing_predictor_cells"])

print("\nMODEL RESULTS (lower CV RMSE is better)")
print(results[["Model", "CV_RMSE", "Test_MAE", "Test_RMSE", "Test_R2"]].to_string(
    index=False, float_format=lambda value: f"{value:.3f}"
))
print("\nSelected model:", best_name)

if feature_importance is not None:
    print("\nTop feature importance values:")
    print(feature_importance.head(4).to_string(
        index=False, float_format=lambda value: f"{value:.3f}"
    ))

print("\nResults and figures saved in:", out_folder)
