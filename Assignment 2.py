import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

def plot_pie_chart(percentages, column):
    plt.figure(figsize=(6, 6))
    plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%')
    plt.title(f'Distribution of {column.capitalize()}')
    plt.show()

print("First few rows of df1:")
print(df1.head())

print("\nDistribution of status variables with percentages:")
status_columns = ['gender', 'minority', 'deprived']
labels = {
    'gender': {0: 'Not Male', 1: 'Male'},
    'minority': {0: 'Not Minority', 1: 'Minority'},
    'deprived': {0: 'Not Deprived', 1: 'Deprived'}
}

for col in status_columns:
    print(f"\nDistribution for {col}:")
    
    if col in labels:
        labeled_data = df1[col].replace(labels[col])
    else:
        labeled_data = df1[col]
    
    counts = labeled_data.value_counts()
    percentages = (counts / len(df1[col])) * 100
    distribution = pd.DataFrame({'Count': counts, 'Percentage': percentages})
    print(distribution)
    
    plot_pie_chart(percentages, col)

df2_df3_merged = pd.merge(df2, df3, on='ID', how='inner')

screen_time_columns = ['C', 'G', 'S', 'T']
for col in screen_time_columns:
    df2_df3_merged[f'total_{col}_time'] = df2_df3_merged[f'{col}_we'] + df2_df3_merged[f'{col}_wk']

wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

df2_df3_merged['composite_wellbeing'] = np.mean(df2_df3_merged[wellbeing_columns].values, axis=1)

df2_df3_merged = df2_df3_merged.dropna()

df2_df3_merged['total_screen_time'] = np.sum(df2_df3_merged[[f'total_{col}_time' for col in screen_time_columns]].values, axis=1)
df2_df3_merged['screen_time_category'] = pd.qcut(df2_df3_merged['total_screen_time'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

grouped = df2_df3_merged.groupby('screen_time_category', observed=True)['composite_wellbeing']
mean_wellbeing = grouped.apply(np.mean)
std_wellbeing = grouped.apply(np.std)

screen_time_stats = df2_df3_merged.groupby('screen_time_category', observed=True)['total_screen_time'].agg(['mean', 'std'])

print("\nMean and Standard Deviation of Total Screen Time for each category:")
print(screen_time_stats)

plt.figure(figsize=(9, 7))
plt.errorbar(mean_wellbeing.index, mean_wellbeing, yerr=std_wellbeing, fmt='-o', capsize=5)
plt.title('Composite Well-being Score vs Screen Time Category')
plt.xlabel('Screen Time Category')
plt.ylabel('Mean Composite Well-being Score')
plt.grid(True)
plt.show()
