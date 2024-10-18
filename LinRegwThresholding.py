#NOTE: THIS WAS ONLY USED FOR TESTING THUS SHOULD NOT BE ASSESSED. THANK YOU

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

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

##### APPLYING THRESHOLDING #####

# Define threshold values for well-being scores (you can adjust these based on your data distribution)
def categorize_wellbeing(score):
    if score < 30:
        return 'low'
    elif 30 <= score <= 50:
        return 'medium'
    else:
        return 'high'

# Apply the threshold function to create a new categorical column for well-being
merged_df['wellbeing_category'] = merged_df['total_wellbeing_score'].apply(categorize_wellbeing)

# Adding interaction terms
merged_df['screen_time_x_gender'] = merged_df['total_screen_time'] * merged_df['gender']
merged_df['screen_time_x_minority'] = merged_df['total_screen_time'] * merged_df['minority']
merged_df['screen_time_x_deprived'] = merged_df['total_screen_time'] * merged_df['deprived']

# Check the head of the dataframe to see the new interaction columns
print(merged_df[['total_screen_time', 'gender', 'screen_time_x_gender', 'screen_time_x_minority', 'screen_time_x_deprived']].head())

##### MODELING #####

# Adding the interaction terms to your feature set
x = merged_df[['scaled_screen_time', 'screen_time_x_gender', 'screen_time_x_minority', 'screen_time_x_deprived', 
               'gender', 'minority', 'deprived', 'scaled_computer_time', 'scaled_game_time', 
               'scaled_smartphone_time', 'scaled_tv_time']]

y = merged_df['wellbeing_category']  # Use the new well-being category as the target variable


##### LINEAR REGRESSION MODEL #####
# Drop any remaining rows with missing values in both x and y
x = x.dropna()
y = y[x.index]  # Ensure y is aligned with x after dropping NaNs in x


# Encoding the target variable (y) using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Proceed with train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
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

##### APPLYING THRESHOLDING TO PREDICTIONS #####

# Applying thresholding to predictions (map continuous predictions to categories)
y_pred_class = ['low' if pred < 30 else 'medium' if pred <= 50 else 'high' for pred in y_pred]

##### MODEL EVALUATION #####

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_class))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred_class)
print(f"\nAccuracy: {accuracy}")

##### ADDITIONAL VISUALIZATIONS #####

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['low', 'medium', 'high'], yticklabels=['low', 'medium', 'high'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Well-being Category')
plt.show()
