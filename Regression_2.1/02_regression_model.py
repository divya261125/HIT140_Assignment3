
# HIT140 Assessment 3 - Regression 2.1
# Predicting goal difference in the 2026 FIFA World Cup

# Goal difference = Team goals - Opponent goals


# STEP 1: Import the libraries we need

# Pandas helps us read and work with CSV files
import pandas as pd

# NumPy helps us calculate the square root
import numpy as np

# Path helps Python find files and folders
from pathlib import Path

# This splits our dataset into training and testing groups
from sklearn.model_selection import train_test_split

# This is the regression model we will use
from sklearn.linear_model import LinearRegression

# These calculate how accurate our predictions are
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# STEP 2: Load our dataset

# Find the folder where this Python script is saved
project_folder = Path(__file__).resolve().parent

# Read the CSV file from the same folder
matches = pd.read_csv(
    project_folder / "regression21_final_dataset.csv"
)

# Check how many matches we have
print("Total matches:", len(matches))


# STEP 3: Choose the variables

# These eight columns will help us predict goal difference
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

# X contains the information used to make predictions
X = matches[variables]

# y contains the actual goal differences
y = matches["goal_difference"]

# Check that our dataset has the correct size
print("\nNumber of explanatory variables:", X.shape[1])
print("Number of matches:", X.shape[0])

# Check that none of the model values are missing
print("\nMissing values in X:", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum())


# STEP 4: Split the dataset

# 80% of the matches will be used for training
# 20% will be used to test the model
# random_state=42 gives us the same split each time

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining matches:", len(X_train))
print("Testing matches:", len(X_test))


# STEP 5: Train the regression model

# Create a Linear Regression model
model = LinearRegression()

# Teach the model using the training matches
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# STEP 6: Predict goal differences

# Predict goal differences for the testing matches
predictions = model.predict(X_test)

# Compare the actual results with our predictions
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

# Show the first five predictions
print("\nActual vs Predicted:")
print(comparison.head())


# STEP 7: Check how well the model performed

# MAE is the average size of our prediction errors
mae = mean_absolute_error(y_test, predictions)

# RMSE also measures errors but gives more weight
# to predictions that are far from the actual result
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# R-squared measures how much of the variation
# in goal difference our model can explain
r2 = r2_score(y_test, predictions)

print("\nLinear Regression Results:")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R-squared:", round(r2, 3))


# STEP 8: Compare with a simple baseline

# A baseline gives us a simple prediction to compare with.
# We use the average goal difference from training matches.

average_goal_difference = y_train.mean()

# Predict this same average for every testing match
baseline_predictions = np.full(
    len(y_test),
    average_goal_difference
)

# Calculate the baseline MAE
baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("\nBaseline MAE:", round(baseline_mae, 3))


# STEP 9: Check the regression coefficients

# Each coefficient shows how a variable is related
# to predicted goal difference, with other variables fixed.

coefficients = pd.DataFrame({
    "Variable": variables,
    "Coefficient": model.coef_
})

print("\nRegression Coefficients:")
print(coefficients)

# The intercept is the model's starting value
print("\nIntercept:", round(model.intercept_, 3))


# STEP 10: Save our results

# Create the Results folder if it does not already exist
results_folder = project_folder / "Results"
results_folder.mkdir(parents=True, exist_ok=True)

# Save the actual and predicted goal differences
comparison.to_csv(
    results_folder / "regression21_predictions.csv",
    index=False
)

# Save the regression coefficients
coefficients.to_csv(
    results_folder / "regression21_coefficients.csv",
    index=False
)

# Create a table with the model results
results = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R-squared", "Baseline MAE"],
    "Value": [mae, rmse, r2, baseline_mae]
})

# Save the model results
results.to_csv(
    results_folder / "regression21_model_results.csv",
    index=False
)

print("\nResults saved successfully!")
