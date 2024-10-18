#NOTE: THIS WAS ONLY USED FOR TESTING THUS SHOULD NOT BE ASSESSED. THANK YOU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

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

##### ADDING INTERACTION TERMS #####

merged_df['screen_time_x_gender'] = merged_df['total_screen_time'] * merged_df['gender']
merged_df['screen_time_x_minority'] = merged_df['total_screen_time'] * merged_df['minority']
merged_df['screen_time_x_deprived'] = merged_df['total_screen_time'] * merged_df['deprived']

##### XGBOOST REGRESSION #####

# Adding the interaction terms to your feature set
x = merged_df[['scaled_screen_time', 'screen_time_x_gender', 'screen_time_x_minority', 'screen_time_x_deprived', 
               'gender', 'minority', 'deprived', 'scaled_computer_time', 'scaled_game_time', 
               'scaled_smartphone_time', 'scaled_tv_time']]

y = merged_df['total_wellbeing_score']

# Drop any remaining rows with missing values in both x and y
x = x.dropna()
y = y[x.index]  # Ensure y is aligned with x after dropping NaNs in x

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initializing and fitting the XGBoost Regressor model
xgbr = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgbr.fit(x_train, y_train)

# Making predictions
y_pred = xgbr.predict(x_test)

##### MODEL EVALUATION #####

# Mean Squared Error, R-squared, Root Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Root Mean Squared Error: {rmse}")

# Optionally, feature importance plot
importances = xgbr.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices], align='center')
plt.xticks(range(x.shape[1]), [x.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()


