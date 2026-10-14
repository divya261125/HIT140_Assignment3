
# HIT140 Assessment 3 - Regression 2.1
# Exploratory Data Analysis (EDA) and visualisations

# This script uses the final dataset prepared in 01_prepare_dataset.py
# so every group member works from exactly the same 104 matches.

# Run it from the project folder (the folder that contains Data/),
# the same way as 01_prepare_dataset.py and 02_regression_model.py.


# --------------------------------------------------
# STEP 1: Import the libraries we need
# --------------------------------------------------

import os
from pathlib import Path

# Pandas and NumPy for working with the data
import pandas as pd
import numpy as np

# Matplotlib for the charts
import matplotlib
matplotlib.use("Agg")  # saves charts to files without opening windows
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# SciPy for correlation p-values and skewness
from scipy import stats

# Scikit-learn for the VIF (multicollinearity) calculation
from sklearn.linear_model import LinearRegression


# Make sure the output folders exist
os.makedirs("Figures", exist_ok=True)
os.makedirs("Results", exist_ok=True)


# One shared chart style so every figure in the report looks the same
BLUE = "#2a78d6"      # main data colour
ORANGE = "#eb6834"    # highlight colour (used sparingly)
RED = "#e34948"       # negative end of the correlation scale
INK = "#0b0b0b"       # main text
INK_2 = "#52514e"     # secondary text and axes
GRID = "#e4e3df"      # light grid lines

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


# Short, readable names for the charts
nice_names = {
    "team_fifa_rank": "Team FIFA rank",
    "opponent_fifa_rank": "Opponent FIFA rank",
    "team_elo": "Team Elo",
    "opponent_elo": "Opponent Elo",
    "host_advantage": "Host advantage",
    "team_recent_goal_difference_avg": "Team recent goal diff (avg)",
    "opponent_recent_goals_conceded_avg": "Opponent recent goals conceded (avg)",
    "team_recent_win_percentage": "Team recent win %",
    "goal_difference": "Goal difference",
}


# --------------------------------------------------
# STEP 2: Load the final dataset
# --------------------------------------------------

# encoding="utf-8-sig" removes the hidden character that
# 01_prepare_dataset.py writes at the start of the file
from pathlib import Path

# Find the folder where this Python script is saved
BASE_DIR = Path(__file__).resolve().parent

# Read the final dataset from the same folder
matches = pd.read_csv(
    BASE_DIR / "regression21_final_dataset.csv",
    encoding="utf-8-sig"
)

# The same eight explanatory variables used in 02_regression_model.py
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

target = "goal_difference"

print("Rows (matches):", len(matches))
print("Explanatory variables:", len(variables))
print("Missing values:", matches[variables + [target]].isnull().sum().sum())
print("Duplicate match IDs:", matches["match_id"].duplicated().sum())

# Check that goal difference really equals team goals minus opponent goals
check = (matches["team_goals"] - matches["opponent_goals"]
         == matches[target]).all()
print("Goal difference calculated correctly:", check)


# --------------------------------------------------
# STEP 3: Descriptive statistics
# --------------------------------------------------

columns = variables + [target]

summary = matches[columns].describe().T
summary["skewness"] = matches[columns].apply(stats.skew)
summary = summary.round(3)

print("\nDescriptive statistics:")
print(summary)

summary.to_csv("Results/eda_descriptive_statistics.csv")


# --------------------------------------------------
# STEP 4: Distribution of the target (goal difference)
# --------------------------------------------------

gd_counts = matches[target].value_counts().sort_index()

wins = (matches[target] > 0).sum()
draws = (matches[target] == 0).sum()
losses = (matches[target] < 0).sum()

print("\nGoal difference from the first-listed team's view:")
print("Wins:", wins, "| Draws:", draws, "| Losses:", losses)
print("Mean goal difference:", round(matches[target].mean(), 3))
print("Median goal difference:", matches[target].median())

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(gd_counts.index, gd_counts.values, color=BLUE, width=0.7)

# Label each bar with its count
for x, count in zip(gd_counts.index, gd_counts.values):
    ax.text(x, count + 0.4, str(count), ha="center", va="bottom",
            fontsize=9, color=INK_2)

ax.axvline(matches[target].mean(), color=ORANGE, linewidth=2,
           linestyle="--", label=f"Mean = {matches[target].mean():.2f}")

ax.set_xticks(gd_counts.index)
ax.set_ylim(0, gd_counts.max() + 4)
ax.set_xlabel("Goal difference (team goals - opponent goals)")
ax.set_ylabel("Number of matches")
ax.set_title("Distribution of goal difference (104 matches)",
             loc="left")
ax.legend(frameon=False)
ax.grid(axis="x", visible=False)

fig.tight_layout()
fig.savefig("Figures/fig01_goal_difference_distribution.png")
plt.close(fig)


# --------------------------------------------------
# STEP 5: Distribution of each explanatory variable
# --------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(13, 6))

for ax, var in zip(axes.flat, variables):

    if var == "host_advantage":
        # Host advantage is 0 or 1, so a bar chart is clearer
        counts = matches[var].value_counts().sort_index()
        ax.bar(["Not host (0)", "Host (1)"], counts.values,
               color=BLUE, width=0.6)
        for i, c in enumerate(counts.values):
            ax.text(i, c + 1, str(c), ha="center", va="bottom",
                    fontsize=9, color=INK_2)
    else:
        ax.hist(matches[var], bins=12, color=BLUE,
                edgecolor="white", linewidth=1)

    ax.set_title(nice_names[var], fontsize=10)
    ax.set_ylabel("Matches")
    ax.grid(axis="x", visible=False)

fig.suptitle("Distributions of the eight explanatory variables",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig("Figures/fig02_variable_distributions.png")
plt.close(fig)


# --------------------------------------------------
# STEP 6: Correlation analysis
# --------------------------------------------------

correlation = matches[columns].corr()
correlation.round(3).to_csv("Results/eda_correlation_matrix.csv")

# Correlation of each variable with goal difference, with p-values
rows = []
for var in variables:
    r, p = stats.pearsonr(matches[var], matches[target])
    rows.append({"Variable": var, "Pearson_r": r, "p_value": p})

target_corr = pd.DataFrame(rows)
target_corr["abs_r"] = target_corr["Pearson_r"].abs()
target_corr = target_corr.sort_values("abs_r", ascending=False)
target_corr = target_corr.drop(columns="abs_r").round(4)

print("\nCorrelation with goal difference:")
print(target_corr.to_string(index=False))

target_corr.to_csv("Results/eda_correlation_with_target.csv", index=False)

# Heatmap: red = negative, grey = none, blue = positive
diverging = LinearSegmentedColormap.from_list(
    "red_grey_blue", [RED, "#f0efec", BLUE]
)

labels = [nice_names[c] for c in columns]

fig, ax = plt.subplots(figsize=(9, 7.5))
image = ax.imshow(correlation.values, cmap=diverging, vmin=-1, vmax=1)

ax.set_xticks(range(len(columns)))
ax.set_yticks(range(len(columns)))
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_yticklabels(labels)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Write the correlation value inside each square
for i in range(len(columns)):
    for j in range(len(columns)):
        value = correlation.values[i, j]
        colour = "white" if abs(value) > 0.6 else INK
        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                fontsize=8.5, color=colour)

colourbar = fig.colorbar(image, ax=ax, shrink=0.8)
colourbar.set_label("Pearson correlation (r)")
colourbar.outline.set_visible(False)

ax.set_title("Correlation matrix of the variables",
             loc="left")
fig.tight_layout()
fig.savefig("Figures/fig03_correlation_heatmap.png")
plt.close(fig)


# --------------------------------------------------
# STEP 7: Each variable plotted against goal difference
# --------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)

# A small random shift stops points with the same value
# from sitting exactly on top of each other (display only)
rng = np.random.default_rng(42)
y_jitter = matches[target] + rng.uniform(-0.15, 0.15, len(matches))

for ax, var in zip(axes.flat, variables):

    x = matches[var]

    if var == "host_advantage":
        x_plot = x + rng.uniform(-0.08, 0.08, len(matches))
    else:
        x_plot = x

    ax.scatter(x_plot, y_jitter, s=22, color=BLUE, alpha=0.65,
               edgecolor="white", linewidth=0.5)

    # Straight line of best fit for this one variable
    slope, intercept = np.polyfit(x, matches[target], 1)
    x_line = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_line, intercept + slope * x_line, color=ORANGE,
            linewidth=2)

    r = correlation.loc[var, target]
    ax.set_title(f"{nice_names[var]}\nr = {r:.2f}", fontsize=10)

    if var == "host_advantage":
        ax.set_xticks([0, 1])

    ax.axhline(0, color=INK_2, linewidth=0.8)

for ax in axes[:, 0]:
    ax.set_ylabel("Goal difference")

fig.suptitle(
    "Each explanatory variable against goal difference "
    "(orange = simple line of best fit)",
    x=0.01, ha="left", fontweight="bold"
)
fig.tight_layout()
fig.savefig("Figures/fig04_variables_vs_goal_difference.png")
plt.close(fig)


# --------------------------------------------------
# STEP 8: Host advantage and tournament stage
# --------------------------------------------------

# Group stage vs knockout rounds
matches["stage"] = np.where(matches["round"] == "Group stage",
                            "Group stage", "Knockout")

host_table = matches.groupby("host_advantage")[target].agg(
    ["count", "mean", "std"]).round(3)
stage_table = matches.groupby("stage")[target].agg(
    ["count", "mean", "std"]).round(3)

print("\nGoal difference by host advantage:")
print(host_table)
print("\nGoal difference by stage:")
print(stage_table)

# Welch's t-test: do host-nation teams have a higher mean goal difference?
host_gd = matches.loc[matches["host_advantage"] == 1, target]
other_gd = matches.loc[matches["host_advantage"] == 0, target]
t_stat, p_value = stats.ttest_ind(host_gd, other_gd, equal_var=False)
print("\nWelch t-test (host vs not host): t =", round(t_stat, 3),
      " p =", round(p_value, 4))

group_tables = pd.concat(
    [host_table.rename(index={0: "Not host", 1: "Host"}),
     stage_table],
    keys=["Host advantage", "Stage"]
)
group_tables.to_csv("Results/eda_goal_difference_by_group.csv")

# Share of draws in each stage
# (in knockouts, a penalty shoot-out is recorded as a 0 goal difference,
# and extra-time goals are included in the score)
print("\nShare of draws by stage:")
print(matches.groupby("stage")[target].apply(
    lambda s: round((s == 0).mean() * 100, 1)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

panels = [
    (axes[0], "host_advantage", [0, 1], ["Not host (0)", "Host (1)"],
     "Host advantage"),
    (axes[1], "stage", ["Group stage", "Knockout"],
     ["Group stage", "Knockout"], "Tournament stage"),
]

for ax, column, groups, tick_labels, title in panels:

    data = [matches.loc[matches[column] == g, target] for g in groups]

    ax.boxplot(
        data,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE),
        medianprops=dict(color=BLUE, linewidth=2),
        whiskerprops=dict(color=BLUE),
        capprops=dict(color=BLUE),
    )

    for i, values in enumerate(data, start=1):
        x_points = i + rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(x_points, values + rng.uniform(-0.1, 0.1, len(values)),
                   s=16, color=BLUE, alpha=0.55, edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.scatter(i, values.mean(), marker="D", s=60, color=ORANGE,
                   zorder=4)
        ax.text(i + 0.3, values.mean(),
                f"mean {values.mean():.2f}\n(n = {len(values)})",
                va="center", fontsize=9, color=INK_2,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

    ax.set_xlim(0.5, 2.85)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(tick_labels)
    ax.set_title(title)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.grid(axis="x", visible=False)

axes[0].set_ylabel("Goal difference")

fig.suptitle("Goal difference by host advantage and by stage "
             "(orange diamond = mean)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig("Figures/fig05_host_and_stage.png")
plt.close(fig)


# --------------------------------------------------
# STEP 9: Multicollinearity check (VIF)
# --------------------------------------------------

# VIF = 1 / (1 - R squared), where R squared comes from predicting
# one explanatory variable using the other seven.
# VIF above 5 suggests a variable overlaps strongly with the others.

def calculate_vif(data, columns_to_check):
    results = []
    for var in columns_to_check:
        others = [c for c in columns_to_check if c != var]
        model = LinearRegression().fit(data[others], data[var])
        r_squared = model.score(data[others], data[var])
        results.append({"Variable": var, "VIF": 1 / (1 - r_squared)})
    return pd.DataFrame(results)


vif = calculate_vif(matches, variables).sort_values("VIF",
                                                    ascending=False)
vif["VIF"] = vif["VIF"].round(2)

print("\nVariance Inflation Factors:")
print(vif.to_string(index=False))

vif.to_csv("Results/eda_vif.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))

vif_plot = vif.sort_values("VIF")
bar_colours = [ORANGE if v > 5 else BLUE for v in vif_plot["VIF"]]

ax.barh([nice_names[v] for v in vif_plot["Variable"]], vif_plot["VIF"],
        color=bar_colours, height=0.6)

for y, v in enumerate(vif_plot["VIF"]):
    ax.text(v + 0.1, y, f"{v:.1f}", va="center", fontsize=9, color=INK_2)

ax.axvline(5, color=INK_2, linestyle="--", linewidth=1)
ax.text(5.1, len(vif_plot) - 0.45, "VIF = 5", color=INK_2, fontsize=9)
ax.set_ylim(-0.6, len(vif_plot) - 0.1)

ax.set_xlabel("Variance Inflation Factor")
ax.set_title("Multicollinearity check (orange = VIF above 5)",
             loc="left")
ax.grid(axis="y", visible=False)

fig.tight_layout()
fig.savefig("Figures/fig06_vif.png")
plt.close(fig)


# --------------------------------------------------
# STEP 10: Summary of the main EDA findings
# --------------------------------------------------

strongest = target_corr.iloc[0]

print("\n----- EDA summary -----")
print(f"Goal difference ranges from {matches[target].min()} "
      f"to {matches[target].max()}, mean {matches[target].mean():.2f}.")
print(f"Strongest single predictor: {strongest['Variable']} "
      f"(r = {strongest['Pearson_r']:.2f}).")
print("Pairs of variables with |r| above 0.8:")
for i, a in enumerate(variables):
    for b in variables[i + 1:]:
        if abs(correlation.loc[a, b]) > 0.8:
            print(f"  {a} & {b}: r = {correlation.loc[a, b]:.2f}")
print("Variables with VIF above 5:",
      ", ".join(vif.loc[vif["VIF"] > 5, "Variable"]))

print("\nFigures saved in Figures/ and tables saved in Results/")
