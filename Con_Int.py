import pandas as pd
import numpy as np
from scipy import stats

data1 = pd.read_csv('dataset1.csv')
data2 = pd.read_csv('dataset2.csv')
data3 = pd.read_csv('dataset3.csv')

merged_data = pd.merge(data2, data3, on='ID')

variables = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thkclr', 'Goodme', 
             'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    margin = stats.sem(data) * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean, mean - margin, mean + margin

ci_results = {var: confidence_interval(merged_data[var].dropna()) for var in variables if var in merged_data.columns}

ci_df = pd.DataFrame(ci_results, index=['Mean', 'Lower CI', 'Upper CI']).T

print(ci_df)

ci_df.to_csv('confidence_intervals_results.csv')
