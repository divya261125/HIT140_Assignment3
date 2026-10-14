
# HIT140 Assessment 3 - Regression 2.1
# Preparing the 2026 FIFA World Cup dataset

# Import pandas to work with datasets
import pandas as pd
from pathlib import Path

# Use the folder containing this script, not the terminal location.
project_folder = Path(__file__).resolve().parent


# --------------------------------------------------
# STEP 1: Load the World Cup dataset
# --------------------------------------------------

# Read the dataset containing 104 World Cup matches
matches = pd.read_csv(
    project_folder / "Data" / "processed" / "worldcup2026_regression21_with_fifa_and_elo.csv"
)

# Check the number of matches
print("Total World Cup matches:", len(matches))

# Check for repeated match IDs
print("Duplicate match IDs:",
      matches["match_id"].duplicated().sum())


# --------------------------------------------------
# STEP 2: Calculate Host Advantage
# --------------------------------------------------

# Start with 0 for every match
# 1 means the first-listed team is playing in its own country
matches["host_advantage"] = 0

# Stadiums located in Mexico
mexico_stadiums = [
    "Estadio Banorte",
    "Estadio Akron",
    "Estadio BBVA"
]

# Stadiums located in Canada
canada_stadiums = [
    "BMO Field",
    "BC Place"
]

# Stadiums located in the USA
usa_stadiums = [
    "SoFi Stadium",
    "MetLife Stadium",
    "Gillette Stadium",
    "Lumen Field",
    "Levi's Stadium",
    "Lincoln Financial Field",
    "Mercedes-Benz Stadium",
    "Hard Rock Stadium",
    "Reliant Stadium",
    "AT&T Stadium",
    "GEHA Field"
]

# Go through each match
for index, match in matches.iterrows():

    team = match["team"]
    venue = match["venue"]

    # Check whether Mexico is playing in Mexico
    if team == "Mexico" and any(
        stadium in venue for stadium in mexico_stadiums
    ):
        matches.loc[index, "host_advantage"] = 1

    # Check whether Canada is playing in Canada
    elif team == "Canada" and any(
        stadium in venue for stadium in canada_stadiums
    ):
        matches.loc[index, "host_advantage"] = 1

    # Check whether USA is playing in the USA
    elif team == "USA" and any(
        stadium in venue for stadium in usa_stadiums
    ):
        matches.loc[index, "host_advantage"] = 1

# Display the results
print("\nHost Advantage:")
print(matches["host_advantage"].value_counts())


# --------------------------------------------------
# STEP 3: Load historical international match results
# --------------------------------------------------

# Read the historical football dataset
historical_matches = pd.read_csv(
    project_folder / "Data" / "raw" / "results.csv"
)

# Convert text dates into Python dates
historical_matches["date"] = pd.to_datetime(
    historical_matches["date"]
)

# Set the World Cup starting date
world_cup_start = pd.to_datetime("2026-06-11")

# Keep only matches played before the World Cup
# This prevents data leakage from future results
historical_matches = historical_matches[
    historical_matches["date"] < world_cup_start
].copy()

# Sort the historical matches by date
historical_matches = historical_matches.sort_values("date")

print("\nHistorical matches before the World Cup:")
print(len(historical_matches))


# --------------------------------------------------
# STEP 4: Match country names between datasets
# --------------------------------------------------

# Some countries have different names in the two files.
# This dictionary connects the World Cup names to
# the names used in the historical dataset.

team_name_changes = {
    "Korea Republic": "South Korea",
    "USA": "United States",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo"
}

# Get the unique teams in the World Cup dataset
world_cup_teams = matches["team"].unique()

# Fix names that contain unusual or damaged characters
for team in world_cup_teams:

    if "Ivoire" in team:
        team_name_changes[team] = "Ivory Coast"

    elif "rkiye" in team:
        team_name_changes[team] = "Turkey"

    elif team.startswith("Bosnia"):
        team_name_changes[team] = "Bosnia and Herzegovina"


# --------------------------------------------------
# STEP 5: Calculate recent performance for each team
# --------------------------------------------------

# Create dictionaries to store the results
recent_goal_difference = {}
recent_goals_conceded = {}
recent_win_percentage = {}

# Check each World Cup team one at a time
for team in world_cup_teams:

    # Find the name used in the historical dataset
    historical_team_name = team_name_changes.get(team, team)

    # Find matches where this team played at home
    home_matches = historical_matches[
        historical_matches["home_team"] == historical_team_name
    ].copy()

    # Calculate goal difference for home matches
    home_matches["goal_difference"] = (
        home_matches["home_score"] - home_matches["away_score"]
    )

    # Goals conceded when playing at home
    home_matches["goals_conceded"] = home_matches["away_score"]

    # Find matches where this team played away
    away_matches = historical_matches[
        historical_matches["away_team"] == historical_team_name
    ].copy()

    # Calculate goal difference for away matches
    away_matches["goal_difference"] = (
        away_matches["away_score"] - away_matches["home_score"]
    )

    # Goals conceded when playing away
    away_matches["goals_conceded"] = away_matches["home_score"]

    # Combine home and away matches
    team_matches = pd.concat([
        home_matches,
        away_matches
    ])

    # Arrange matches from newest to oldest
    team_matches = team_matches.sort_values(
        "date",
        ascending=False
    )

    # Select the last five international matches
    last_five_matches = team_matches.head(5)

    # Check whether the team has five matches
    if len(last_five_matches) < 5:
        print("Warning: Fewer than five matches for", team)

    # Calculate average goal difference
    recent_goal_difference[team] = (
        last_five_matches["goal_difference"].mean()
    )

    # Calculate average goals conceded
    recent_goals_conceded[team] = (
        last_five_matches["goals_conceded"].mean()
    )

    # Count how many of the last five matches were wins
    wins = (
        last_five_matches["goal_difference"] > 0
    ).sum()

    # Calculate win percentage
    if len(last_five_matches) > 0:
        recent_win_percentage[team] = (
            wins / len(last_five_matches) * 100
        )
    else:
        recent_win_percentage[team] = float("nan")


# --------------------------------------------------
# STEP 6: Add recent-performance variables
# --------------------------------------------------

# Variable 6: First-listed team's recent goal difference
matches["team_recent_goal_difference_avg"] = (
    matches["team"].map(recent_goal_difference)
)

# Variable 7: Opponent's recent goals conceded
matches["opponent_recent_goals_conceded_avg"] = (
    matches["opponent"].map(recent_goals_conceded)
)

# Variable 8: First-listed team's recent win percentage
matches["team_recent_win_percentage"] = (
    matches["team"].map(recent_win_percentage)
)


# --------------------------------------------------
# STEP 7: Check the eight explanatory variables
# --------------------------------------------------

# List the eight variables used for Regression 2.1
explanatory_variables = [
    "team_fifa_rank",
    "opponent_fifa_rank",
    "team_elo",
    "opponent_elo",
    "host_advantage",
    "team_recent_goal_difference_avg",
    "opponent_recent_goals_conceded_avg",
    "team_recent_win_percentage"
]

# Display missing values in these variables
print("\nMissing values in explanatory variables:")
print(matches[explanatory_variables].isnull().sum())

# Show the first five rows
print("\nFirst five matches with explanatory variables:")
print(matches[["team", "opponent"] + explanatory_variables].head())


# --------------------------------------------------
# STEP 8: Save the prepared dataset
# --------------------------------------------------

# Check whether any explanatory variables are missing
missing_values = matches[explanatory_variables].isnull().sum().sum()

# Only save when all eight variables are complete
if missing_values == 0 and len(matches) == 104:

    matches.to_csv(
        project_folder / "regression21_final_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDataset saved successfully!")
    print("Total matches:", len(matches))
    print("Explanatory variables:", len(explanatory_variables))

else:
    print("\nDataset not saved.")
    print("Please fix the missing values first.")
