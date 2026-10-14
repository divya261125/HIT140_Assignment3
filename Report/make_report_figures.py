"""
Builds Figures 1-3 of the Objective 2 report from the group's datasets and the
CSV outputs of the Regression 2.1 and 2.2 scripts (run those first, including
Report/regression_2_1_supplement.py and the Regression 2.2 supplement).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
R21 = ROOT / "Regression_2.1"
R22 = ROOT / "Regeration2.2"
OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID = "#a9a8a3", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 7.8,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 7.2,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 6.6, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# ---------------------------------------------------------------- Figure 1
d = pd.read_csv(R21 / "regression21_final_dataset.csv", encoding="utf-8-sig")
y = d["goal_difference"]
d["elo_gap"] = d.team_elo - d.opponent_elo
d["log_rank_ratio"] = np.log(d.opponent_fifa_rank) - np.log(d.team_fifa_rank)
labels = {
    "opponent_elo": "Opponent Elo", "opponent_fifa_rank": "Opponent FIFA rank",
    "team_fifa_rank": "Team FIFA rank", "team_elo": "Team Elo",
    "opponent_recent_goals_conceded_avg": "Opp. recent conceded",
    "team_recent_goal_difference_avg": "Team recent GD",
    "team_recent_win_percentage": "Team recent win %", "host_advantage": "Host advantage",
    "elo_gap": "Elo gap (engineered)", "log_rank_ratio": "Log rank ratio (engineered)",
}
rows = []
for c in labels:
    r, p = stats.pearsonr(d[c], y)
    rows.append((labels[c], r, p, c in ("elo_gap", "log_rank_ratio")))
cor = pd.DataFrame(rows, columns=["v", "r", "p", "eng"])
cor = cor.reindex(cor.r.abs().sort_values().index)

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [1.25, 1.05, 0.9]})
cols = [ORANGE if e else (BLUE if p < 0.05 else GREY) for e, p in zip(cor.eng, cor.p)]
ax[0].barh(cor.v, cor.r, color=cols, height=0.62)
ax[0].axvline(0, color=INK2, lw=0.6)
for i, (r, p) in enumerate(zip(cor.r, cor.p)):
    txt = f"{r:+.2f}" + ("" if p < 0.05 else " (ns)")
    ax[0].text(r + (0.02 if r >= 0 else -0.02), i, txt, va="center",
               ha="left" if r >= 0 else "right", fontsize=6.2, color=INK)
ax[0].set_xlim(-0.9, 0.98)
ax[0].set_xlabel("Pearson r with goal difference")
ax[0].set_title("(a) Gaps beat levels")
ax[0].grid(axis="y", visible=False)
ax[0].tick_params(axis="y", labelsize=6.4)
ax[0].legend(handles=[Line2D([], [], color=BLUE, lw=5, label="Raw, p < .05"),
                      Line2D([], [], color=GREY, lw=5, label="Raw, not sig."),
                      Line2D([], [], color=ORANGE, lw=5, label="Engineered")],
             loc="lower left", fontsize=5.8, handlelength=1, borderaxespad=0.2)

rng = np.random.default_rng(1)
ko = ~d["round"].str.contains("Group")
for mask, col, lab in [(~ko, BLUE, "Group stage"), (ko, ORANGE, "Knockout")]:
    ax[1].scatter(d.elo_gap[mask], y[mask] + rng.uniform(-0.18, 0.18, mask.sum()), s=9,
                  color=col, alpha=0.75, edgecolor="white", linewidth=0.4, label=lab)
b, a = np.polyfit(d.elo_gap, y, 1)
xs = np.linspace(d.elo_gap.min(), d.elo_gap.max(), 50)
ax[1].plot(xs, a + b * xs, color=INK, lw=1.2)
ax[1].text(0.03, 0.95, f"r = {np.corrcoef(d.elo_gap, y)[0,1]:.2f}\n+{b*100:.2f} goals per 100 Elo",
           transform=ax[1].transAxes, va="top", fontsize=6.4, color=INK)
ax[1].set_xlabel("Elo gap (team − opponent)")
ax[1].set_ylabel("Goal difference")
ax[1].set_title("(b) Strength gap vs result")
ax[1].legend(loc="lower right", handletextpad=0.2, markerscale=1.3)

vals, cnt = np.unique(y, return_counts=True)
ax[2].bar(vals, cnt, color=BLUE, width=0.78)
m, s, n = y.mean(), y.std(ddof=1), len(y)
lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=s / np.sqrt(n))
ax[2].axvspan(lo, hi, color=ORANGE, alpha=0.22, lw=0)
ax[2].axvline(m, color=ORANGE, lw=1.2)
ax[2].axvline(0, color=INK2, lw=0.6, ls=":")
ax[2].text(0.98, 0.97, f"mean {m:.2f}\n95% CI\n{lo:.2f} to {hi:.2f}\n(t-test vs 0:\np = 0.017)",
           transform=ax[2].transAxes, fontsize=6.0, va="top", ha="right", color=INK)
ax[2].set_xlabel("Goal difference")
ax[2].set_ylim(0, 33)
ax[2].set_ylabel("Matches")
ax[2].set_title("(c) Target")
ax[2].grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT / "fig1_r21_eda.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
cvc = pd.read_csv(R21 / "Results" / "cv_model_comparison.csv")
s3 = pd.read_csv(OUT / "results_2_1" / "stage3_feature_engineering.csv")
short = {
    "Linear regression: log FIFA ranks": "OLS + log ranks",
    "Linear regression: Yeo-Johnson transform": "OLS + Yeo-Johnson",
    "Linear regression: collinear variables removed": "OLS, high-VIF removed",
    "Ridge regression": "Ridge + log ranks",
    "Lasso regression": "Lasso + log ranks",
    "Elastic Net": "Elastic Net + log ranks",
    "Huber robust regression": "Huber + log ranks",
    "K-nearest neighbours": "k-NN (benchmark)",
    "Random forest": "Random forest (benchmark)",
    "Gradient boosting": "Gradient boosting (benchmark)",
}
rows = []
for _, r in cvc.iterrows():
    if r.Model in short and not (r.Stage.startswith("Stage 2") and r.Model == "Linear regression (least squares)"):
        rows.append((short[r.Model], r["MAE change vs original"], r["95% CI low"], r["95% CI high"], "s12"))
for _, r in s3.iterrows():
    if "contrasts" in r.Model:
        nm = "OLS + strength contrasts" if r.Model.startswith("Linear") else "Ridge + strength contrasts"
        rows.append((nm, r["MAE change vs original"], r["95% CI low"], r["95% CI high"], "s3"))
cmp_ = pd.DataFrame(rows, columns=["m", "d", "lo", "hi", "stage"]).sort_values("d", ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1.15, 0.85]})
for i, r in enumerate(cmp_.itertuples()):
    col = ORANGE if r.m == "Ridge + strength contrasts" else (AQUA if r.stage == "s3" else (GREY if "benchmark" in r.m else BLUE))
    ax[0].plot([r.lo, r.hi], [i, i], color=col, lw=1.4, solid_capstyle="round")
    ax[0].plot(r.d, i, "o", color=col, ms=4.2, mec="white", mew=0.6)
    ax[0].text(1.01, i, f"{r.d:+.3f}", va="center", ha="left", fontsize=6.0, color=INK,
               transform=ax[0].get_yaxis_transform())
ax[0].set_yticks(range(len(cmp_)), cmp_.m, fontsize=6.3)
ax[0].axvline(0, color=INK2, lw=0.7)
ax[0].set_xlim(-0.2, 0.25)
ax[0].set_xlabel("Change in CV MAE vs original OLS, goals (corrected 95% CI)\n← better            worse →")
ax[0].set_title("(a) 14 candidates on the same 50 folds")
ax[0].grid(axis="y", visible=False)
ax[0].legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=1.4, ms=4, label="Linear (stages 1-2)"),
                      Line2D([], [], color=AQUA, marker="o", lw=1.4, ms=4, label="Feature engineering (stage 3)"),
                      Line2D([], [], color=GREY, marker="o", lw=1.4, ms=4, label="Non-linear benchmark"),
                      Line2D([], [], color=ORANGE, marker="o", lw=1.4, ms=4, label="Selected")],
             loc="upper center", bbox_to_anchor=(0.42, -0.36), ncol=4, fontsize=5.8, handlelength=1.4,
             columnspacing=0.8)

stab = pd.read_csv(R21 / "Results" / "coefficient_stability.csv", header=[0, 1], index_col=0)
expected = {"team_fifa_rank": -1, "opponent_fifa_rank": 1, "team_elo": 1, "opponent_elo": -1,
            "host_advantage": 1, "team_recent_goal_difference_avg": 1,
            "opponent_recent_goals_conceded_avg": 1, "team_recent_win_percentage": 1}
names = list(expected)
def pct(model):
    pos = stab[(model, "% folds positive")].astype(float)
    return np.array([pos[v] if expected[v] > 0 else 100 - pos[v] for v in names])
po, pf = pct("Original"), pct("Final")
order = np.argsort(po)
for j, k in enumerate(order):
    ax[1].plot([po[k], pf[k]], [j, j], color=GRID, lw=2.2, zorder=1)
    ax[1].plot(po[k], j, "o", color=GREY, ms=4.6, zorder=2, mec="white", mew=0.6)
    ax[1].plot(pf[k], j, "o", color=ORANGE, ms=4.6, zorder=3, mec="white", mew=0.6)
ax[1].set_yticks(range(8), [labels[names[k]] for k in order], fontsize=6.3)
ax[1].set_xlim(-5, 105)
ax[1].set_xlabel("% of 50 fits with expected sign")
ax[1].set_title("(b) Ridge stabilises signs")
ax[1].grid(axis="y", visible=False)
ax[1].legend(handles=[Line2D([], [], color=GREY, marker="o", lw=0, ms=4.5, label="Original OLS"),
                      Line2D([], [], color=ORANGE, marker="o", lw=0, ms=4.5, label="Ridge + log ranks")],
             loc="upper left", fontsize=5.8, borderaxespad=0.2)
fig.tight_layout(w_pad=2.2)
fig.savefig(OUT / "fig2_r21_models.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
g = pd.read_csv(ROOT / "Data" / "Regression_2_2_Dataset.csv")
goals = g.Goals_Scored
fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [0.9, 1.0, 1.15]})
k = np.arange(0, 8)
obs = np.array([(goals == i).sum() for i in k])
exp = stats.poisson.pmf(k, goals.mean()) * len(goals)
exp[-1] = (1 - stats.poisson.cdf(6, goals.mean())) * len(goals)
ax[0].bar(k, obs, color=BLUE, width=0.75, label="Observed")
ax[0].plot(k, exp, "o-", color=ORANGE, ms=3.5, lw=1.1, label="Poisson fit")
ax[0].set_xticks(k, [str(i) for i in k[:-1]] + ["7+"])
ax[0].set_xlabel("Goals scored by a team")
ax[0].set_ylabel("Team-matches")
ax[0].set_title("(a) Skewed count target")
ax[0].text(0.97, 0.52, f"variance {goals.var():.2f}\n> mean {goals.mean():.2f}\n{(goals==0).mean():.0%} zeros",
           transform=ax[0].transAxes, ha="right", fontsize=6.2, color=INK)
ax[0].grid(axis="x", visible=False)
ax[0].legend(loc="upper right", borderaxespad=0.1, handlelength=1.2, fontsize=5.8, labelspacing=0.3)

bands = pd.cut(g.Team_FIFA_Rank, [0, 10, 20, 40, 85], labels=["1–10", "11–20", "21–40", "41–85"])
gb = goals.groupby(bands, observed=True).agg(["mean", "std", "size"])
ci = stats.t.ppf(0.975, gb["size"] - 1) * gb["std"] / np.sqrt(gb["size"])
ax[1].bar(range(4), gb["mean"], color=BLUE, width=0.62)
ax[1].errorbar(range(4), gb["mean"], yerr=ci, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
ax[1].set_xticks(range(4), [f"{b}\nn={nn}" for b, nn in zip(gb.index, gb["size"])], fontsize=5.6)
ax[1].set_xlabel("Team FIFA rank band")
ax[1].set_ylabel("Mean goals (95% CI)")
ax[1].set_title("(b) Strength drives scoring")
ax[1].grid(axis="x", visible=False)

bs_main = pd.read_csv(R22 / "Regression_2_2_outputs" / "holdout_bootstrap.csv").set_index("Comparison")
bs_fe = pd.read_csv(R22 / "Regression_2_2_outputs" / "feature_engineering_bootstrap.csv")
rows = [("Ridge + log rank (CV pick)", *bs_fe.iloc[0, 1:4], BLUE),
        ("Ridge + contrasts", *bs_fe.iloc[1, 1:4], AQUA),
        ("Ridge + contr. + level", *bs_fe.iloc[2, 1:4], AQUA),
        ("Poisson + contrasts", *bs_fe.iloc[3, 1:4], AQUA),
        ("Poisson + contr. + level", *bs_fe.iloc[4, 1:4], ORANGE),
        ("Gradient boosting", bs_main.loc["best_vs_baseline", "RMSE_gain"],
         bs_main.loc["best_vs_baseline", "CI_2.5"], bs_main.loc["best_vs_baseline", "CI_97.5"], GREY)]
for i, (nm, gval, lo, hi, col) in enumerate(rows[::-1]):
    ax[2].plot([lo, hi], [i, i], color=col, lw=1.4, solid_capstyle="round")
    ax[2].plot(gval, i, "o", color=col, ms=4.2, mec="white", mew=0.6)
ax[2].set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=6.2)
ax[2].axvline(0, color=INK2, lw=0.7)
ax[2].set_xlabel("RMSE gain vs baseline (goals)\nbootstrap 95% CI")
ax[2].set_title("(c) Holdout gain")
ax[2].grid(axis="y", visible=False)
fig.tight_layout(w_pad=0.9)
fig.savefig(OUT / "fig3_r22.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
abl = pd.read_csv(R22 / "Regression_2_2_outputs" / "feature_set_ablation.csv")
perm = pd.read_csv(R22 / "Regression_2_2_outputs" / "permutation_importance.csv")
fig, ax = plt.subplots(1, 2, figsize=(6.9, 1.75), gridspec_kw={"width_ratios": [1, 1.15]})
sets = ["All eight", "Shared four only", "Regression 2.2 four only"]
setlab = ["All eight", "Strength four only", "Timing four only"]
w = 0.36
for j, (mdl, col) in enumerate([("Ridge + log FIFA rank", BLUE), ("Gradient boosting", GREY)]):
    v = [abl[(abl.Model == mdl) & (abl.Feature_set == s_)].CV_RMSE.iloc[0] for s_ in sets]
    ax[0].barh(np.arange(3) + (j - 0.5) * w, v, height=w * 0.92, color=col,
               label="Ridge" if j == 0 else "Boosting")
    for i, vv in enumerate(v):
        ax[0].text(vv + 0.004, i + (j - 0.5) * w, f"{vv:.3f}", va="center", fontsize=5.8, color=INK)
ax[0].axvline(1.2688, color=ORANGE, lw=1.1, ls="--")
ax[0].text(1.262, -0.78, "mean baseline 1.269", color=INK, fontsize=5.8, ha="right", va="center")
ax[0].set_yticks(range(3), setlab)
ax[0].set_ylim(2.55, -1.05)
ax[0].set_xlim(0.98, 1.36)
ax[0].set_xlabel("CV RMSE (goals; lower is better)")
ax[0].set_title("(a) Feature-set ablation")
ax[0].grid(axis="y", visible=False)
ax[0].legend(loc="center right", fontsize=5.8, handlelength=1, bbox_to_anchor=(1.0, 0.6))

nice = {"Team_FIFA_Rank": "Team FIFA rank", "Opponent_PreMatch_Rest_Days": "Opponent rest days",
        "Opponent_Elo_PreMatch": "Opponent Elo", "Team_Elo_PreMatch": "Team Elo",
        "Opponent_FIFA_Rank": "Opponent FIFA rank", "Team_PreMatch_Rest_Days": "Team rest days",
        "Opponent_PreMatch_Matches_Played": "Opp. matches played", "Team_PreMatch_Matches_Played": "Team matches played"}
perm = perm.iloc[::-1]
cols = [BLUE if g_ == "Shared" else AQUA for g_ in perm.Group]
ax[1].barh([nice[f] for f in perm.Feature], perm.RMSE_increase, xerr=perm.SD, color=cols, height=0.62,
           error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 1.5})
ax[1].axvline(0, color=INK2, lw=0.7)
ax[1].set_xlabel("Holdout RMSE increase when shuffled (± SD)")
ax[1].set_title("(b) Permutation importance (Ridge)")
ax[1].grid(axis="y", visible=False)
ax[1].tick_params(axis="y", labelsize=6.3)
ax[1].legend(handles=[Line2D([], [], color=BLUE, lw=5, label="Shared strength"),
                      Line2D([], [], color=AQUA, lw=5, label="2.2 timing")], loc="lower right", fontsize=5.8, handlelength=1)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig4_r22_variables.png")
plt.close(fig)
print("figures written to", OUT)
