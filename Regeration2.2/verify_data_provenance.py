"""
Regression 2.2 - data provenance and integrity checks.

1. Match results: every one of the 208 team rows is compared with an independent
   public record, Jurisoo's international football results dataset
   (github.com/martj42/international_results), pinned to commit
   a16ff3edb297bda95d0ed02d5fc0c1ecb5b1c1cf (20 July 2026, the day after the final).
   Penalty shoot-outs are checked against the same repository's shootouts.csv.
2. Pre-match features: rest days and matches already played are recomputed from
   the match dates and compared with the dataset.
3. FIFA ranks: every team whose position is stated in FIFA's 11 June 2026 ranking
   release is checked.
4. Elo: eight teams spanning the full rating range are checked against
   World Football Elo Ratings (eloratings.net), each team's rating after its last
   match before 11 June 2026 (accessed 15 October 2026).

Writes Regression_2_2_outputs/provenance_checks.csv and exits non-zero on any failure.
Needs internet access to raw.githubusercontent.com for step 1.
"""
from io import StringIO
from pathlib import Path
import sys
import urllib.request

import pandas as pd

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder.parent / "Data" / "Regression_2_2_Dataset.csv")
data["Date"] = pd.to_datetime(data["Date"])
COMMIT = "a16ff3edb297bda95d0ed02d5fc0c1ecb5b1c1cf"
BASE = f"https://raw.githubusercontent.com/martj42/international_results/{COMMIT}/"

# Dataset names -> names used in the international results dataset
NAMES = {"USA": "United States", "IR Iran": "Iran", "Czechia": "Czech Republic",
         "Cabo Verde": "Cape Verde", "Congo DR": "DR Congo", "Türkiye": "Turkey",
         "Côte d'Ivoire": "Ivory Coast"}
checks = []


def record(name, passed, detail):
    checks.append({"Check": name, "Result": "PASS" if passed else "FAIL", "Detail": detail})


def fetch(file):
    with urllib.request.urlopen(BASE + file, timeout=60) as resp:
        return pd.read_csv(StringIO(resp.read().decode("utf-8")))


# 1. Results against the independent record
res = fetch("results.csv")
res = res[(res.tournament == "FIFA World Cup") & (res.date >= "2026-06-11") & (res.date <= "2026-07-19")]
lookup = {}
for r in res.itertuples():
    lookup[(r.date, r.home_team, r.away_team)] = (r.home_score, r.away_score)
    lookup[(r.date, r.away_team, r.home_team)] = (r.away_score, r.home_score)
bad = []
for r in data.itertuples():
    key = (r.Date.strftime("%Y-%m-%d"), NAMES.get(r.Team, r.Team), NAMES.get(r.Opponent, r.Opponent))
    score = lookup.get(key)
    if score is None or score[0] != r.Goals_Scored:
        bad.append(key)
record("Independent source has 104 World Cup 2026 matches", len(res) == 104, f"{len(res)} matches found")
record("All 208 team rows match the independent goals record", not bad,
       f"{208 - len(bad)}/208 rows match" + (f"; mismatches: {bad[:5]}" if bad else ""))
record("Total goals agree", int(data.Goals_Scored.sum()) == int(res.home_score.sum() + res.away_score.sum()),
       f"dataset {int(data.Goals_Scored.sum())}, source {int(res.home_score.sum() + res.away_score.sum())}")

so = fetch("shootouts.csv")
so = so[(so.date >= "2026-06-11") & (so.date <= "2026-07-19")]
draws = 0
for s in so.itertuples():
    key = (s.date, s.home_team, s.away_team)
    draws += lookup.get(key, (0, 1))[0] == lookup.get(key, (0, 1))[1]
record("Shoot-out ties are stored as draws (shoot-out goals excluded)", draws == len(so),
       f"{draws}/{len(so)} shoot-out matches stored as draws")

# 2. Pre-match features recomputed from match dates (earlier dates only)
ok_rest = ok_played = 0
for r in data.itertuples():
    earlier = data[(data.Team == r.Team) & (data.Date < r.Date)]
    played = len(earlier)
    rest = (r.Date - earlier.Date.max()).days if played else None
    ok_played += played == r.Team_PreMatch_Matches_Played
    ok_rest += (rest is None and pd.isna(r.Team_PreMatch_Rest_Days)) or rest == r.Team_PreMatch_Rest_Days
record("Matches played recomputed from earlier dates", ok_played == 208, f"{ok_played}/208 rows match")
record("Rest days recomputed from earlier dates", ok_rest == 208, f"{ok_rest}/208 rows match")
pairs = data.merge(data, left_on=["Match_ID", "Team"], right_on=["Match_ID", "Opponent"], suffixes=("", "_o"))
mirror = ((pairs.Opponent_FIFA_Rank == pairs.Team_FIFA_Rank_o) & (pairs.Opponent_Elo_PreMatch == pairs.Team_Elo_PreMatch_o)).all()
record("Opponent columns mirror the paired row", bool(mirror), "rank and Elo checked for all 208 rows")
const = (data.groupby("Team")[["Team_FIFA_Rank", "Team_Elo_PreMatch"]].nunique() == 1).all().all()
record("Rank and Elo fixed for each team throughout the tournament", bool(const), "one value per team")

# 3. FIFA ranks stated in FIFA's 11 June 2026 release
fifa = {"Argentina": 1, "Spain": 2, "France": 3, "Morocco": 7, "Netherlands": 8, "Mexico": 14,
        "Senegal": 15, "Uruguay": 16, "USA": 17, "IR Iran": 20}
ranks = data.groupby("Team").Team_FIFA_Rank.first()
wrong = {t: (ranks[t], v) for t, v in fifa.items() if ranks[t] != v}
record("FIFA ranks match the 11 June 2026 release", not wrong,
       f"{len(fifa) - len(wrong)}/{len(fifa)} stated positions match" + (f"; {wrong}" if wrong else ""))

# 4. Elo spot check against eloratings.net (rating after last match before 11 June 2026)
elo = {"Spain": 2157, "Ecuador": 1938, "Mexico": 1875, "Cabo Verde": 1578, "Haiti": 1548,
       "South Africa": 1518, "Curaçao": 1434, "Qatar": 1421}
elos = data.groupby("Team").Team_Elo_PreMatch.first()
wrong = {t: (elos[t], v) for t, v in elo.items() if elos[t] != v}
record("Elo matches eloratings.net for 8 teams spanning the range", not wrong,
       f"{len(elo) - len(wrong)}/{len(elo)} teams match" + (f"; {wrong}" if wrong else ""))

out = pd.DataFrame(checks)
out.to_csv(folder / "Regression_2_2_outputs" / "provenance_checks.csv", index=False)
print(out.to_string(index=False))
sys.exit(0 if (out.Result == "PASS").all() else 1)
