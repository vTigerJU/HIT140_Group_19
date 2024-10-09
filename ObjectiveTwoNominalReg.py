import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Function to calculate Root Mean Squared Error
def rmse(y_test, y_pred):
    rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # Normalised Root Mean Square Error
    y_max = y_test.max()
    y_min = y_test.min()
    rmse_norm = rmse / (y_max - y_min)
    return rmse_norm

# Function to calculate Adjusted R²
def adjustedR2(y_test, y_pred, sample_size, p):
    r2 = metrics.r2_score(y_true=y_test, y_pred=y_pred)
    return round(1 - (1 - r2) * ((sample_size - 1) / (sample_size - p - 1)), 4)

# Function to run linear regression and show results
def linRegResult(x, y, test_size):
    # Scaling the features using StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)  # Scale the independent variables

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=test_size, random_state=0)

    # Linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Print out the feature coefficients
    print("Intercept:", model.intercept_)
    print("Feature Coefficients:")
    for feature, coef_val in zip(x.columns, model.coef_.flatten()):  # Flattening the coefficients if needed
        print(f"  {feature}: {coef_val}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()

    # Calculate metrics
    predictionR2 = adjustedR2(y_test, y_pred, len(y_test), x.shape[1])
    rmse_pred = rmse(y_test, y_pred)

    # Store results in a dictionary and return as a DataFrame
    result = {
        'predicted value': "compositeScore",
        'intercept': model.intercept_,
        'coef': model.coef_,
        'predictionR2': predictionR2,
        'rmseNorm pred': rmse_pred
    }

    return pd.DataFrame([result])

# Function to calculate total screen time by summing weekday and weekend times for each device
def totalScreenTime(watchTime):
    # Sum Computer screen time (C_we + C_wk)
    watchTime['computerTime'] = watchTime['C_we'] + watchTime['C_wk']
    
    # Sum Games screen time (G_we + G_wk)
    watchTime['gameTime'] = watchTime['G_we'] + watchTime['G_wk']
    
    # Sum Smartphone screen time (S_we + S_wk)
    watchTime['smartphoneTime'] = watchTime['S_we'] + watchTime['S_wk']
    
    # Sum TV screen time (T_we + T_wk)
    watchTime['tvTime'] = watchTime['T_we'] + watchTime['T_wk']
    
    return watchTime

# Function to compute composite well-being score (mean of all well-being columns except 'ID')
def compositeWellBeingScore(wellBeing):
    wellBeing['compositeScore'] = wellBeing.drop(columns=['ID']).mean(axis=1)
    return wellBeing

### MAIN CODE ###

# 1. Read in datasets (Make sure the paths to your datasets are correct)
subjectDescription = pd.read_csv("dataset1.csv")
watchTime = pd.read_csv("dataset2.csv")
wellBeing = pd.read_csv("dataset3.csv")

# 2. Create total screen time for each type by summing weekday and weekend times
watchTime = totalScreenTime(watchTime)

# 3. Create composite well-being score
wellBeing = compositeWellBeingScore(wellBeing)

# 4. Merge datasets on 'ID'
df = subjectDescription.merge(wellBeing, on='ID').merge(watchTime, on='ID')

# Check if 'compositeScore' exists in the DataFrame
print(df.columns)  # Ensure 'compositeScore' is in the list of columns
print(df.head())   # Preview the first few rows to verify data

# 5. Select independent and dependent variables (use the new screen time columns)
x = df[['computerTime', 'gameTime', 'smartphoneTime', 'tvTime', 'gender', 'deprived', 'minority']]  # New screen time variables
y = df[['compositeScore']]  # Use 'compositeScore' as the target variable

# 6. Run linear regression and show results
result = linRegResult(x, y, test_size=0.2)
print(result)
