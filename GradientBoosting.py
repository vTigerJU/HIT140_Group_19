#NOTE: THIS WAS ONLY USED FOR TESTING THUS SHOULD NOT BE ASSESSED. THANK YOU

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline

# Reading the data sets
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

##### DATA WRANGLING #####
# Merging the datasets based on the 'ID' column
merged_df = df1.merge(df2, on='ID', how='inner').merge(df3, on='ID', how='inner')

# Creating total well-being variable
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
merged_df['total_wellbeing_score'] = merged_df[wellbeing_columns].sum(axis=1)

# Checking for missing values and dropping rows with missing values
merged_df = merged_df.dropna(subset=wellbeing_columns + ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk'])

# Feature Engineering (Create total screen time for each device by summing weekend and weekday times)
merged_df['total_computer_time'] = merged_df['C_we'] + merged_df['C_wk']
merged_df['total_game_time'] = merged_df['G_we'] + merged_df['G_wk']
merged_df['total_smartphone_time'] = merged_df['S_we'] + merged_df['S_wk']
merged_df['total_tv_time'] = merged_df['T_we'] + merged_df['T_wk']
merged_df['total_screen_time'] = merged_df['total_computer_time'] + merged_df['total_game_time'] + merged_df['total_smartphone_time'] + merged_df['total_tv_time']

# Handling Outliers
z_scores = np.abs(stats.zscore(merged_df[['total_screen_time']]))
merged_df = merged_df[(z_scores < 3).all(axis=1)]  # Keep rows with Z-scores less than 3

##### SCALING FEATURES #####
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(merged_df[['total_screen_time', 'total_computer_time', 'total_game_time', 'total_smartphone_time', 'total_tv_time']])

# Concatenate the scaled columns to the original dataframe
scaled_df = pd.DataFrame(scaled_features, columns=['scaled_screen_time', 'scaled_computer_time', 'scaled_game_time', 'scaled_smartphone_time', 'scaled_tv_time'])
merged_df = pd.concat([merged_df, scaled_df], axis=1)

# Adding interaction terms
merged_df['screen_time_x_gender'] = merged_df['total_screen_time'] * merged_df['gender']
merged_df['screen_time_x_minority'] = merged_df['total_screen_time'] * merged_df['minority']
merged_df['screen_time_x_deprived'] = merged_df['total_screen_time'] * merged_df['deprived']

# Define the feature set and target variable
x = merged_df[['scaled_screen_time', 'screen_time_x_gender', 'screen_time_x_minority', 'screen_time_x_deprived', 
               'gender', 'minority', 'deprived', 'scaled_computer_time', 'scaled_game_time', 
               'scaled_smartphone_time', 'scaled_tv_time']]
y = merged_df['total_wellbeing_score']

# Drop any remaining rows with missing values in both x and y
x = x.dropna()
y = y[x.index]

##### FEATURE TRANSFORMATION #####
# Using PolynomialFeatures to add polynomial interaction terms
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

##### GRADIENT BOOSTING REGRESSOR #####
# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # Number of boosting stages (trees)
    'max_depth': [3, 5],         # Maximum depth of the trees
    'learning_rate': [0.01, 0.1]  # Learning rate (step size)
}

# Initialize Gradient Boosting Regressor
gbr = GradientBoostingRegressor()

# Set up the grid search with cross-validation
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=42)

# Train the model using GridSearchCV
grid_search.fit(x_train, y_train)

# Best parameters from GridSearchCV
print(f"Best Parameters: {grid_search.best_params_}")

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Making predictions on the test set
y_pred = best_model.predict(x_test)

##### MODEL EVALUATION #####
# Mean Squared Error, R-squared, and RMSE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Root Mean Squared Error: {rmse}")

##### CROSS-VALIDATION #####
# Cross-validation on the training set
cross_val_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Average RMSE from cross-validation
cross_val_rmse = np.sqrt(-cross_val_scores).mean()
print(f"Cross-Validation RMSE: {cross_val_rmse}")

