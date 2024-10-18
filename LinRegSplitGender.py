#NOTE: THIS WAS ONLY USED FOR TESTING THUS SHOULD NOT BE ASSESSED. THANK YOU

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

##### IMPUTING MISSING VALUES #####
# Use SimpleImputer to fill missing values in feature columns
imputer = SimpleImputer(strategy='mean')
merged_df[['scaled_screen_time', 'scaled_computer_time', 'scaled_game_time', 'scaled_smartphone_time', 'scaled_tv_time']] = imputer.fit_transform(merged_df[['scaled_screen_time', 'scaled_computer_time', 'scaled_game_time', 'scaled_smartphone_time', 'scaled_tv_time']])

##### CHECKING FOR MULTICOLLINEARITY USING VIF #####
# Calculate VIF for features
features = ['scaled_screen_time', 'scaled_computer_time', 'scaled_game_time', 'scaled_smartphone_time', 'scaled_tv_time']
X_vif = merged_df[features]
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print("\nVIF Values:\n", vif_data)

##### SPLITTING BY GENDER #####

# Split the data into male and female
male_df = merged_df[merged_df['gender'] == 1]  # Subset where gender is male (assuming 1 is male)
female_df = merged_df[merged_df['gender'] == 0]  # Subset where gender is female (assuming 0 is female)

##### DEFINE A FUNCTION TO RUN RIDGE REGRESSION #####
def run_ridge_regression(df, gender_label):
    print(f"\nRunning Ridge Regression for {gender_label}:")
    
    # Define feature columns
    x = df[['scaled_screen_time', 'scaled_computer_time', 'scaled_game_time', 'scaled_smartphone_time', 'scaled_tv_time']]
    y = df['total_wellbeing_score']
    
    # Splitting the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Initialize and fit the Ridge Regression model (alpha is the regularization parameter)
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    
    # Making predictions
    y_pred = model.predict(x_test)
    
    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print(f"Root Mean Squared Error: {rmse}")
    
    # Optionally, to check the coefficients (feature importance)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

##### RUN RIDGE REGRESSION FOR MALE AND FEMALE #####

run_ridge_regression(male_df, "Male")
run_ridge_regression(female_df, "Female")
