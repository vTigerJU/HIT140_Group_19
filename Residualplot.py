import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import math
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def linRegResult(x, y, test_size):
    '''Returns a row with all important information from linReg and includes scatter plot for predictions vs actuals and residuals plot'''
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    
    # Baseline model and linear regression model
    base = baseLineModel(y_train, y_test)
    prediction, intercept, coef = linReg(x_train, x_test, y_train)
    
    # Flatten y_test and prediction to ensure they're 1D arrays
    y_test_flat = y_test.values.flatten()  # Convert y_test to a 1D array
    prediction_flat = prediction.flatten()  # Ensure prediction is a 1D array

    # Plotting Actual vs Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_flat, prediction_flat, color='blue', label='Predicted vs Actual')
    plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 'r--', label='Perfect Fit Line')  # 45-degree line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()
    
    # Residuals calculation
    residuals = y_test_flat - prediction_flat
    
    # Check that prediction and residuals have the same length
    print(f"Prediction size: {len(prediction_flat)}, Residuals size: {len(residuals)}")
    
    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(prediction_flat, residuals, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
    
    # Calculate metrics
    baseR2 = adjustedR2(y_test_flat, base, len(y_test_flat), 0)
    rmse_base = rmse(y_test_flat, base)
    predictionR2 = adjustedR2(y_test_flat, prediction_flat, len(y_test_flat), x.shape[1])
    rmse_pred = rmse(y_test_flat, prediction_flat)
    
    # Store results in a dictionary and return as a DataFrame
    result = {
        'predicted value': "null",
        'intercept': intercept,
        'coef': coef,
        'predictionR2': predictionR2,
        'baseR2': baseR2,
        'rmseNorm pred': rmse_pred,
        'rmseNorm base': rmse_base
    }
    
    return pd.DataFrame([result])


# Select independent and dependent variables
x = df[['totalTime']]  # Example: using totalTime as the independent variable
y = df[['compositeScore']]  # Example: using compositeScore as the dependent variable

# Call linRegResult to perform regression and display results along with residuals plot
result = linRegResult(x, y, test_size=0.2)

# Print the result DataFrame with regression details
print(result)
