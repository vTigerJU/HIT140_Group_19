import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

def plot_pie_chart(df, column, labels=None):
    if labels:
        labeled_data = df[column].replace(labels)
    else:
        labeled_data = df[column]
        
    counts = labeled_data.value_counts()
    percentages = (counts / len(df[column])) * 100

    plt.figure(figsize=(6, 6))
    plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution of {column.capitalize()}')
    plt.show()

print("First few rows of df1:")
print(df1.head())

print("\nDistribution of status variables with percentages:")

status_columns = ['gender', 'minority', 'deprived']
labels = {
    'gender': {0: 'Not Male', 1: 'Male'},
    'minority': {0: 'Minority', 1: 'Not Minority'},
    'deprived': {0: 'Not Deprived', 1: 'Deprived'}
}

for col in status_columns:
    print(f"\nDistribution for {col}:")
    
    if col in labels:
        labeled_data = df1[col].replace(labels[col])
        counts = labeled_data.value_counts()
        percentages = (counts / len(df1[col])) * 100
        distribution = pd.DataFrame({'Count': counts, 'Percentage': percentages})
        print(distribution)
        plot_pie_chart(df1, col, labels.get(col))
    else:
        counts = df1[col].value_counts()
        percentages = (counts / len(df1[col])) * 100
        distribution = pd.DataFrame({'Count': counts, 'Percentage': percentages})
        print(distribution)

print("First few rows of df2:")
print(df2.head())
