import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

# Merge df2 and df3 on the 'ID' column using an inner join
df2_df3_merged = pd.merge(df2, df3, on='ID', how='inner')

# Display the first few rows of the merged dataset
print("First few rows of the merged dataset:")
print(df2_df3_merged.head())

# List of screen time columns (assuming C = Phone, G = Video Games, S = Laptop, T = TV)
screen_time_columns = ['C', 'G', 'S', 'T']

# Calculate total screen time for each activity
for col in screen_time_columns:
    df2_df3_merged[f'total_{col}_time'] = df2_df3_merged[f'{col}_we'] + df2_df3_merged[f'{col}_wk']

# Display the first few rows to check the new columns
print("\nFirst few rows with total screen time for each activity calculated:")
print(df2_df3_merged[[f'total_{col}_time' for col in screen_time_columns]].head())

# (Optional) Create bins for total screen time for one activity as an example (e.g., Phone time)
# You can create similar bins for other activities if needed.
df2_df3_merged['screen_time_category'] = pd.cut(df2_df3_merged['total_C_time'], bins=3, labels=['Low', 'Medium', 'High'])

# List of well-being indices
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Create a composite well-being score by averaging all indices
df2_df3_merged['composite_wellbeing'] = df2_df3_merged[wellbeing_columns].mean(axis=1)

# Dropping rows with any null values
df2_df3_merged = df2_df3_merged.dropna()

# Display the composite well-being score for the first few rows
print("\nComposite Well-being Scores:")
print(df2_df3_merged['composite_wellbeing'].head())

# Group data by screen time categories or by quantiles
df2_df3_merged['total_screen_time'] = df2_df3_merged['total_C_time'] + df2_df3_merged['total_G_time'] + df2_df3_merged['total_S_time'] + df2_df3_merged['total_T_time']

# For example, categorize total screen time into bins
df2_df3_merged['screen_time_category'] = pd.qcut(df2_df3_merged['total_screen_time'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Analyze the average composite well-being score for each screen time category
wellbeing_by_screen_time = df2_df3_merged.groupby('screen_time_category')['composite_wellbeing'].describe()

print(wellbeing_by_screen_time)

# Calculate the mean and standard deviation for each screen time category
wellbeing_by_screen_time = df2_df3_merged.groupby('screen_time_category')['composite_wellbeing'].agg(['mean', 'std'])

# Plot the mean composite well-being score with error bars
plt.figure(figsize=(8, 6))
plt.errorbar(wellbeing_by_screen_time.index, wellbeing_by_screen_time['mean'], yerr=wellbeing_by_screen_time['std'], fmt='-o', capsize=5)
plt.title('Composite Well-being Score vs Screen Time Category')
plt.xlabel('Screen Time Category')
plt.ylabel('Mean Composite Well-being Score')
plt.grid(True)
plt.show()