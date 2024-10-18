#NOTE: THIS WAS ONLY USED FOR TESTING THUS SHOULD NOT BE ASSESSED. THANK YOU

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import mord
from sklearn.metrics import mean_squared_error, r2_score  # <- ADD THIS

# Reading the data sets
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

##### DATA WRANGLING #####

# Merging the datasets based on the 'ID' column
merged_df = df1.merge(df2, on='ID', how='inner').merge(df3, on='ID', how='inner')
print(merged_df.head())

# Creating total well-being variable
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
merged_df['total_wellbeing_score'] = merged_df[wellbeing_columns].sum(axis=1)
print(merged_df[['ID', 'total_wellbeing_score']].head())

# Checking for missing values and dropping rows with missing values
print(merged_df.isnull().sum())
merged_df = merged_df.dropna()

# Feature Engineering (Create total screen time for each device by summing weekend and weekday times)
merged_df['total_computer_time'] = merged_df['C_we'] + merged_df['C_wk']
merged_df['total_game_time'] = merged_df['G_we'] + merged_df['G_wk']
merged_df['total_smartphone_time'] = merged_df['S_we'] + merged_df['S_wk']
merged_df['total_tv_time'] = merged_df['T_we'] + merged_df['T_wk']
merged_df['total_screen_time'] = merged_df['total_computer_time'] + merged_df['total_game_time'] + merged_df['total_smartphone_time'] + merged_df['total_tv_time']

# Track number of rows before removing outliers
rows_before = merged_df.shape[0]

# Handling Outliers
z_scores = np.abs(stats.zscore(merged_df[['total_screen_time']]))
merged_df = merged_df[(z_scores < 3).all(axis=1)]  # Keep rows with Z-scores less than 3

# Calculate how many rows were removed
rows_after = merged_df.shape[0]
rows_removed = rows_before - rows_after
print(f"\nROWS REMOVED: {rows_removed}")

##### SCALING FEATURES #####

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(merged_df[['total_screen_time', 'total_computer_time', 'total_game_time', 'total_smartphone_time', 'total_tv_time']])

# Concatenate the scaled columns to the original dataframe
scaled_df = pd.DataFrame(scaled_features, columns=['scaled_screen_time', 'scaled_computer_time', 'scaled_game_time', 'scaled_smartphone_time', 'scaled_tv_time'])
merged_df = pd.concat([merged_df, scaled_df], axis=1)
print(merged_df.head())

##### EXPLORATORY DATA ANALYSIS #####

# Summary statistics
print(merged_df.describe())

# Define feature columns
feature_columns = ['total_screen_time', 'total_computer_time', 'total_game_time', 'total_smartphone_time', 'total_tv_time']

# Plot histograms for each feature with labels
merged_df[feature_columns].hist(bins=20, figsize=(12, 8), edgecolor='black', layout=(2, 3))
plt.suptitle("Distributions of Features", fontsize=16)
plt.xlabel('Hours')
plt.ylabel('Count')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Correlation matrix
correlation_matrix = merged_df[['total_screen_time', 'total_computer_time', 'total_game_time', 'total_smartphone_time', 'total_tv_time']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Screen Time Features')
plt.show()

# Scatter plot: total screen time vs well-being score
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['total_screen_time'], merged_df['total_wellbeing_score'], alpha=0.5)
plt.title('Total Screen Time vs Well-being Score')
plt.xlabel('Total Screen Time')
plt.ylabel('Well-being Score')
plt.show()

# Adding interaction terms
merged_df['screen_time_x_gender'] = merged_df['total_screen_time'] * merged_df['gender']
merged_df['screen_time_x_minority'] = merged_df['total_screen_time'] * merged_df['minority']
merged_df['screen_time_x_deprived'] = merged_df['total_screen_time'] * merged_df['deprived']

# Check the head of the dataframe to see the new interaction columns
print(merged_df[['total_screen_time', 'gender', 'screen_time_x_gender', 'screen_time_x_minority', 'screen_time_x_deprived']].head())

# Adding the interaction terms to your feature set
x = merged_df[['scaled_screen_time', 'screen_time_x_gender', 'screen_time_x_minority', 'screen_time_x_deprived', 
               'gender', 'minority', 'deprived', 'scaled_computer_time', 'scaled_game_time', 
               'scaled_smartphone_time', 'scaled_tv_time']]

# Converting the well-being score into categories ('low', 'medium', 'high') for ordinal regression
bins = [0, 30, 60, 90]  # You can adjust these bin edges based on your data
labels = ['low', 'medium', 'high']
merged_df['wellbeing_category'] = pd.cut(merged_df['total_wellbeing_score'], bins=bins, labels=labels)

y = merged_df[['wellbeing_category']]

# Drop any remaining rows with missing values
x = x.dropna()
y = y.dropna()

##### SYNCHRONIZING INDICES #####

# Ensure the indices of `x` and `y` match
x, y = x.align(y, join='inner', axis=0)

##### CONVERTING CATEGORIES TO NUMERIC #####

# Convert 'low', 'medium', 'high' to 1, 2, 3 for ordinal regression
y_numeric = y['wellbeing_category'].map({'low': 1, 'medium': 2, 'high': 3}).astype(int)

##### ORDINAL REGRESSION MODEL #####

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y_numeric, test_size=0.2, random_state=42)

# Fit the ordinal regression model using mord LogisticIT
model = mord.LogisticIT()
model.fit(x_train, y_train)

# Making predictions on the test set
y_pred = model.predict(x_test)

##### EVALUATION #####
# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
# R-squared
r2 = r2_score(y_test, y_pred)
# Root Mean Squared Error
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Root Mean Squared Error: {rmse}")

# Optionally, to check the coefficients (feature importance)
print("Coefficients:", model.coef_)
