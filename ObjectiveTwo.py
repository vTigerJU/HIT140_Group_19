import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import math
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#----------------Not used------------------#
def scatterPlot(x, y, name, df): 
    plt.figure(figsize=(64, 48))
    plt.scatter(df[x], df[y], color='blue')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title(name)
    plt.show()

def histogram(df, col, title):
    plt.hist(df[col], color='blue', edgecolor='black',bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], align="mid",width=0.8)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Population")
    plt.show()
#----------------End------------------#

def linRegResult(x,y, test_size):
    '''Returns a row with all important information from linReg'''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    base = baseLineModel(y_train, y_test)
    prediction, intercept, coef = linReg(x_train, x_test, y_train)
    baseR2 = adjustedR2(y_test, base, len(y_test), 0)
    rmse_base = rmse(y_test, base)
    predictionR2 = adjustedR2(y_test, prediction, len(y_test),x.shape[1])
    rmse_pred = rmse(y_test,prediction)
    result = {
    'predicted value': "null",
    'intercept': intercept,
    'coef': coef,
    'predictionR2': predictionR2,
    'baseR2': baseR2,
    'rmseNorm pred':rmse_pred,
    'rmseNorm base':rmse_base
    }
    return pd.DataFrame([result])

def linReg(x_train, x_test, y_train):      
    '''Linear regression with 1 or more explanatory variables'''
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, float(model.intercept_[0]), model.coef_ 
    #standardMetrics(y_test, y_pred, len(y_test),x.shape[1])

def baseLineModel(y_train, y_test):
    '''BaselinModel uses the mean as prediction'''
    y_base = np.mean(y_train)
    y_pred = [y_base] * len(y_test)
    return y_pred
    #standardMetrics(y_test,y_pred, len(y_test))

def rmse(y_test, y_pred):
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # Normalised Root Mean Square Error
    y_max = y_test.max()
    y_min = y_test.min()
    rmse_norm = rmse / (y_max - y_min)
    return rmse_norm.iloc[0]

def adjustedR2(y_test, y_pred, sample_size, p):
    '''Adjusted R2 takes into account if multiple explanatory variables'''
    r2 = metrics.r2_score(y_true= y_test,y_pred= y_pred)
    return round(1 - (1 - r2)*((sample_size-1)/(sample_size-p-1)), 4) # result from 0 -> 1 . Close to 1 is better

def standardMetrics(y_test, y_pred, sample_size, p = 0):   
    '''Commonly used metrics for evaluating a linear regression'''  
    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_test, y_pred)
    # Mean Squared Error
    mse = metrics.mean_squared_error(y_test, y_pred)
    # Root Mean Square Error
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # Normalised Root Mean Square Error
    y_max = y_test.max()
    y_min = y_test.min()
    rmse_norm = rmse / (y_max - y_min)
    adjusted_r2 = adjustedR2(y_test, y_pred, sample_size, p)

    print("RMSE (Normalised): ", round(rmse_norm,4))
    print("Adjusted R2: ", adjusted_r2 )

def totalScreenTime(watchTime):
    #New feature
    #Finds the total screen time per individual
    colList = list(watchTime)
    colList.remove("ID")
    watchTime["totalTime"] = watchTime[colList].sum(axis=1)
    #watchTime.drop(watchTime.iloc[:,1:9], inplace=True, axis=1) #Drops all except ID and total watch time
    return watchTime

def wellBeingScore(wellBeing): 
    #New feature
    #Well being score
    #Mode of all other ratings
    colList = list(wellBeing)
    colList.remove("ID")
    wellBeingMode = wellBeing[colList].mode(axis=1)
    wellBeing["wellBeingScore"] = wellBeingMode[0]
    return wellBeing

def investigation(x_col, y_col, df, test_size):
    result_df = None
    x = df[x_col]
    for col in y_col: 
        result = linRegResult(x,df[[col]],test_size)
        result['predicted value'] = col
        if result_df is None:
            result_df = result
        else:
            result_df = pd.concat([result_df, result ], ignore_index=True)
    return result_df

def multipleInvestigations(wellBeingColList, df, test_size):
    totalTime_df = investigation(["totalTime"], wellBeingColList, df, test_size)
    timeGender_df = investigation(["totalTime","gender"], wellBeingColList, df, test_size)
    timeDeprived_df = investigation(["totalTime","deprived"], wellBeingColList, df, test_size)
    timeMinority_df = investigation(["totalTime","minority"], wellBeingColList, df, test_size)
    timeAll_df = investigation(["totalTime","gender","deprived","minority"], wellBeingColList, df, test_size)

    print("Screen v WellBeing")
    print(totalTime_df)
    print("Screen, gender v WellBeing")
    print(timeGender_df)
    print("Screen, deprived v WellBeing")
    print(timeDeprived_df)
    print("Screen, minority v WellBeing")
    print(timeMinority_df)
    print("Screen, all v WellBeing")
    print(timeAll_df)

def screenSourceInvestigations(x_col, wellBeingColList, df, test_size):
    for x in x_col:
        result_df = investigation([x], wellBeingColList,df,test_size)
        print(x)
        print(result_df)
        #Create graphic here

subjectDescription = pd.read_csv("dataset1.csv")
watchTime = pd.read_csv("dataset2.csv")
wellBeing = pd.read_csv("dataset3.csv")
watchTime = totalScreenTime(watchTime)
wellBeing = wellBeingScore(wellBeing)
df = subjectDescription.merge(wellBeing, on="ID").merge(watchTime, on="ID")
print(df.head())

wellBeingColList = list(wellBeing)
wellBeingColList.remove("ID")
timeColList = list(watchTime)
timeColList.remove("ID")
multipleInvestigations(wellBeingColList, df, 0.2)
screenSourceInvestigations(timeColList, wellBeingColList, df, 0.2) #Uses different ScreenUsages as explanatory variable

def linRegResult(x, y, test_size):
    '''Returns a row with all important information from linReg and includes scatter plot for predictions vs actuals'''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    
    # Baseline model and linear regression model
    base = baseLineModel(y_train, y_test)
    prediction, intercept, coef = linReg(x_train, x_test, y_train)
    
    # Plotting Actual vs Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, prediction, color='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit Line')  # 45-degree line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()
    
    # Calculate metrics
    baseR2 = adjustedR2(y_test, base, len(y_test), 0)
    rmse_base = rmse(y_test, base)
    predictionR2 = adjustedR2(y_test, prediction, len(y_test), x.shape[1])
    rmse_pred = rmse(y_test, prediction)
    
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
y = df[['wellBeingScore']]  # Example: using wellBeingScore as the dependent variable

# Call linRegResult to perform regression and display results
result = linRegResult(x, y, test_size=0.2)

# Print the result DataFrame with regression details
print(result)



def compositeWellBeingScore(wellBeing): 
    # Calculate the mean of all well-being columns to create a composite score
    colList = list(wellBeing)
    colList.remove("ID")
    wellBeing['compositeScore'] = wellBeing[colList].mean(axis=1)
    return wellBeing

# Update the well-being score with the composite score
wellBeing = compositeWellBeingScore(wellBeing)

# Now merge the updated wellBeing DataFrame with subjectDescription and watchTime
df = subjectDescription.merge(wellBeing, on="ID").merge(watchTime, on="ID")

# Now, the 'compositeScore' column should be available in df
print(df.head())  # Check if 'compositeScore' is included in the DataFrame

