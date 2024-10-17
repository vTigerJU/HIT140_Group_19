from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# Load the cleaned dataset
merged_df = pd.read_csv('cleaned_data.csv')

##### DATA WRANGLING #####

# Create total well-being score
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
merged_df['total_wellbeing_score'] = merged_df[wellbeing_columns].sum(axis=1)

# Drop rows with missing values in the well-being and screen time columns
screen_time_columns = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
merged_df.dropna(subset=wellbeing_columns + screen_time_columns, inplace=True)

# Create total screen time variables
merged_df['total_computer_time'] = merged_df['C_we'] + merged_df['C_wk']
merged_df['total_game_time'] = merged_df['G_we'] + merged_df['G_wk']
merged_df['total_smartphone_time'] = merged_df['S_we'] + merged_df['S_wk']
merged_df['total_tv_time'] = merged_df['T_we'] + merged_df['T_wk']
merged_df['total_screen_time'] = merged_df[['total_computer_time', 'total_game_time', 'total_smartphone_time', 'total_tv_time']].sum(axis=1)

# Handle infinite values and NaNs
merged_df.replace([np.inf, -np.inf], 0, inplace=True)
merged_df.dropna(inplace=True)

##### OUTLIER DETECTION AND REMOVAL #####
# Using Z-score to detect and remove outliers
z_scores = np.abs(stats.zscore(merged_df[['total_wellbeing_score', 'total_screen_time', 'total_computer_time', 
                                          'total_game_time', 'total_smartphone_time', 'total_tv_time']]))
merged_df = merged_df[(z_scores < 3).all(axis=1)]  # Retain rows where z-score is less than 3

##### SCALING FEATURES #####

# Scale screen time features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(merged_df[['total_screen_time', 'total_computer_time', 'total_game_time', 'total_smartphone_time', 'total_tv_time']])
scaled_df = pd.DataFrame(scaled_features, columns=['scaled_screen_time', 'scaled_computer_time', 'scaled_game_time', 'scaled_smartphone_time', 'scaled_tv_time'])
merged_df = pd.concat([merged_df, scaled_df], axis=1)

##### FEATURE ENGINEERING #####

# Create interaction terms and ratios
merged_df['screen_time_x_gender'] = merged_df['scaled_screen_time'] * merged_df['gender']
merged_df['screen_time_x_minority'] = merged_df['scaled_screen_time'] * merged_df['minority']
merged_df['screen_time_x_deprived'] = merged_df['scaled_screen_time'] * merged_df['deprived']

# Ratios with gender interactions
merged_df['computer_to_total_ratio'] = merged_df['total_computer_time'] / merged_df['total_screen_time']
merged_df['game_to_total_ratio'] = merged_df['total_game_time'] / merged_df['total_screen_time']
merged_df['smartphone_to_total_ratio'] = merged_df['total_smartphone_time'] / merged_df['total_screen_time']
merged_df['tv_to_total_ratio'] = merged_df['total_tv_time'] / merged_df['total_screen_time']

merged_df['computer_time_x_gender'] = merged_df['computer_to_total_ratio'] * merged_df['gender']
merged_df['game_time_x_gender'] = merged_df['game_to_total_ratio'] * merged_df['gender']
merged_df['smartphone_time_x_gender'] = merged_df['smartphone_to_total_ratio'] * merged_df['gender']
merged_df['tv_time_x_gender'] = merged_df['tv_to_total_ratio'] * merged_df['gender']

# Adding interaction terms with minority and deprived status
merged_df['computer_time_x_minority'] = merged_df['computer_to_total_ratio'] * merged_df['minority']
merged_df['smartphone_time_x_minority'] = merged_df['smartphone_to_total_ratio'] * merged_df['minority']
merged_df['game_time_x_minority'] = merged_df['game_to_total_ratio'] * merged_df['minority']
merged_df['tv_time_x_minority'] = merged_df['tv_to_total_ratio'] * merged_df['minority']

merged_df['computer_time_x_deprived'] = merged_df['computer_to_total_ratio'] * merged_df['deprived']
merged_df['smartphone_time_x_deprived'] = merged_df['smartphone_to_total_ratio'] * merged_df['deprived']
merged_df['game_time_x_deprived'] = merged_df['game_to_total_ratio'] * merged_df['deprived']
merged_df['tv_time_x_deprived'] = merged_df['tv_to_total_ratio'] * merged_df['deprived']

##### RECURSIVE FEATURE ELIMINATION (RFE) #####

x = merged_df[['screen_time_x_gender', 'screen_time_x_minority', 'screen_time_x_deprived', 'scaled_computer_time',
               'scaled_game_time', 'scaled_smartphone_time', 'scaled_tv_time', 'computer_to_total_ratio', 'game_to_total_ratio',
               'smartphone_to_total_ratio', 'tv_to_total_ratio', 'computer_time_x_gender',
               'game_time_x_gender', 'smartphone_time_x_gender', 'tv_time_x_gender',
               'computer_time_x_minority', 'smartphone_time_x_minority', 'game_time_x_minority', 'tv_time_x_minority',
               'computer_time_x_deprived', 'smartphone_time_x_deprived', 'game_time_x_deprived', 'tv_time_x_deprived']]

y = merged_df['total_wellbeing_score']

# Drop rows from both `x` and `y` where `x` contains NaN values
x = x.dropna()
y = y.loc[x.index]  # Align y with the filtered x

# Recursive feature elimination
linear_model = LinearRegression()
rfe = RFE(estimator=linear_model, n_features_to_select=15)
rfe.fit(x, y)
selected_features = x.columns[rfe.support_]
print(f"Selected Features: {selected_features}")

# Use only selected features
x = merged_df[selected_features]

# Ensure that x and y remain aligned
x = x.dropna()
y = y.loc[x.index]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

##### MODEL EVALUATION #####

# Fit the Linear Regression model
linear_model.fit(x_train, y_train)

# Making predictions
y_pred = linear_model.predict(x_test)

# Mean Squared Error, R-squared, Root Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

# Adjusted R2
n = len(y_test)
p = x_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# NRMSE
nrmse = rmse / (y_test.max() - y_test.min())

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Adjusted R-squared: {adjusted_r2}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Normalized Root Mean Squared Error: {nrmse}")

##### BASELINE MODEL #####

# Baseline model: predict the mean of the training data
baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=np.float64)

# Evaluate the baseline model
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)
baseline_rmse = np.sqrt(baseline_mse)

# NRMSE for baseline model
baseline_nrmse = baseline_rmse / (y_test.max() - y_test.min())

print("\nBaseline Model Evaluation:")
print(f"Mean Squared Error (Baseline): {baseline_mse}")
print(f"R-squared (Baseline): {baseline_r2}")
print(f"Root Mean Squared Error (Baseline): {baseline_rmse}")
print(f"Normalized Root Mean Squared Error (Baseline): {baseline_nrmse}")

# NRMSE for baseline model
baseline_nrmse = baseline_rmse / (y_test.max() - y_test.min())

##### EDA and VISUALIZATIONS #####

# Correlation heatmap for key features
key_features = ['total_wellbeing_score', 'total_computer_time', 'total_game_time', 'total_smartphone_time', 'total_tv_time',
                'scaled_screen_time', 'scaled_computer_time', 'scaled_game_time', 'scaled_smartphone_time', 'scaled_tv_time']

filtered_corr = merged_df[key_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Key Features')
plt.show()

# Scatter plots of screen time vs well-being score
plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_computer_time', y='total_wellbeing_score', data=merged_df)
plt.title('Total Computer Time vs Well-being Score')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_smartphone_time', y='total_wellbeing_score', data=merged_df)
plt.title('Total Smartphone Time vs Well-being Score')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_tv_time', y='total_wellbeing_score', data=merged_df)
plt.title('Total TV Time vs Well-being Score')
plt.show()

# Bar plots for categorical variables with replaced labels

# Gender (0 = Not Male, 1 = Male)
merged_df['gender_label'] = merged_df['gender'].replace({0: 'Not Male', 1: 'Male'})
plt.figure(figsize=(8, 6))
sns.barplot(x='gender_label', y='total_wellbeing_score', data=merged_df, estimator=np.mean)
plt.title('Mean Well-being Score by Gender')
plt.ylabel('Mean Well-being Score')
plt.show()

# Deprivation Status (0 = Not Deprived, 1 = Deprived)
merged_df['deprived_label'] = merged_df['deprived'].replace({0: 'Not Deprived', 1: 'Deprived'})
plt.figure(figsize=(8, 6))
sns.barplot(x='deprived_label', y='total_wellbeing_score', data=merged_df, estimator=np.mean)
plt.title('Mean Well-being Score by Deprivation Status')
plt.ylabel('Mean Well-being Score')
plt.show()

# Minority Status (0 = Majority, 1 = Minority)
merged_df['minority_label'] = merged_df['minority'].replace({0: 'Majority', 1: 'Minority'})
plt.figure(figsize=(8, 6))
sns.barplot(x='minority_label', y='total_wellbeing_score', data=merged_df, estimator=np.mean)
plt.title('Mean Well-being Score by Ethnic Group')
plt.ylabel('Mean Well-being Score')
plt.show()

##### REAL VS PREDICTED VALUES PLOTS #####

# Real vs predicted for the linear regression model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Real vs Predicted Well-being Scores (Linear Regression)')
plt.xlabel('Real Well-being Score')
plt.ylabel('Predicted Well-being Score')
plt.grid(True)
plt.show()

# Real vs predicted for the baseline model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, baseline_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Real vs Predicted Well-being Scores (Baseline Model)')
plt.xlabel('Real Well-being Score')
plt.ylabel('Predicted Well-being Score')
plt.grid(True)
plt.show()

# Residuals plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_test - y_pred, alpha=0.5)
plt.axhline(y=0, color='red', lw=2)
plt.title('Residuals Plot (Linear Regression)')
plt.xlabel('Predicted Well-being Score')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Fit the model using statsmodels for detailed summary
x_with_const = sm.add_constant(x)
model = sm.OLS(y, x_with_const).fit()
print(model.summary())