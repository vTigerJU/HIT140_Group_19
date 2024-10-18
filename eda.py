#NOTE: THIS IS ONE OF THE MAIN ASSIGNMENT FILES. IT SHOULD BE RUN BEFORE THE MAIN.PY FILE

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets and merge them on 'ID' column
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")
merged_df = df1.merge(df2, on='ID').merge(df3, on='ID')

# Well-being indicator columns with simplified titles
wellbeing_columns = {
    'Optm': 'Optimistic',
    'Usef': 'Useful',
    'Relx': 'Relaxed',
    'Intp': 'Interested in People',
    'Engs': 'Energetic',
    'Dealpr': 'Dealing with Problems',
    'Thcklr': 'Thinking Clearly',
    'Goodme': 'Feeling Good about Myself',
    'Clsep': 'Close to Others',
    'Conf': 'Confident',
    'Mkmind': 'Making Decisions',
    'Loved': 'Loved',
    'Intthg': 'Interested in New Things',
    'Cheer': 'Cheerful'
}

# Function to create bar charts for well-being indicators
def plot_barcharts(df, columns, title):
    plt.figure(figsize=(16, 10))
    for i, (col, description) in enumerate(columns.items(), 1):
        if col in df.columns:
            plt.subplot(2, 3, i)
            sns.countplot(x=col, data=df)
            plt.title(description)
            plt.xlabel('Response (1-5)')
            plt.ylabel('Frequency')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Split well-being indicators for display into three sets
wellbeing_set_1 = dict(list(wellbeing_columns.items())[:5])
wellbeing_set_2 = dict(list(wellbeing_columns.items())[5:10])
wellbeing_set_3 = dict(list(wellbeing_columns.items())[10:])

# Plot bar charts for well-being indicators in three batches
plot_barcharts(merged_df, wellbeing_set_1, "Well-being Indicators (Set 1)")
plot_barcharts(merged_df, wellbeing_set_2, "Well-being Indicators (Set 2)")
plot_barcharts(merged_df, wellbeing_set_3, "Well-being Indicators (Set 3)")

# Create total screen time column
merged_df['total_screen_time'] = (merged_df['C_we'] + merged_df['C_wk'] +
                                  merged_df['G_we'] + merged_df['G_wk'] +
                                  merged_df['S_we'] + merged_df['S_wk'] +
                                  merged_df['T_we'] + merged_df['T_wk'])

##### GROUP-BASED ANALYSIS #####

# Screen time and well-being by gender
merged_df['gender_label'] = merged_df['gender'].replace({0: 'Not Male', 1: 'Male'})
gender_screen_time = merged_df.groupby('gender_label')['total_screen_time'].mean()
gender_wellbeing = merged_df.groupby('gender_label')[list(wellbeing_columns.keys())].mean()

# Screen time and well-being by minority status
merged_df['minority_label'] = merged_df['minority'].replace({0: 'Majority', 1: 'Minority'})
minority_screen_time = merged_df.groupby('minority_label')['total_screen_time'].mean()
minority_wellbeing = merged_df.groupby('minority_label')[list(wellbeing_columns.keys())].mean()

# Screen time and well-being by deprivation status
merged_df['deprived_label'] = merged_df['deprived'].replace({0: 'Not Deprived', 1: 'Deprived'})
deprivation_screen_time = merged_df.groupby('deprived_label')['total_screen_time'].mean()
deprivation_wellbeing = merged_df.groupby('deprived_label')[list(wellbeing_columns.keys())].mean()

##### VISUALIZATIONS #####

# Gender-based analysis
plt.figure(figsize=(12, 10))

# Screen time by gender
plt.subplot(2, 2, 1)
sns.barplot(x=gender_screen_time.index, y=gender_screen_time.values)
plt.title('Average Total Screen Time by Gender')
plt.xlabel('Gender')
plt.ylabel('Avg. Total Screen Time (Hours)')

# Well-being by gender
plt.subplot(2, 2, 2)
gender_wellbeing.T.plot(kind='bar', ax=plt.gca())
plt.title('Average Well-being Scores by Gender')
plt.xlabel('Well-being Indicators')
plt.ylabel('Avg. Well-being Score')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Minority-based analysis
plt.figure(figsize=(12, 10))

# Screen time by minority status
plt.subplot(2, 2, 1)
sns.barplot(x=minority_screen_time.index, y=minority_screen_time.values)
plt.title('Average Total Screen Time by Ethnic Group')
plt.xlabel('Ethnic Group')
plt.ylabel('Avg. Total Screen Time (Hours)')

# Well-being by minority status
plt.subplot(2, 2, 2)
minority_wellbeing.T.plot(kind='bar', ax=plt.gca())
plt.title('Average Well-being Scores by Ethnic Group')
plt.xlabel('Well-being Indicators')
plt.ylabel('Avg. Well-being Score')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Deprivation-based analysis
plt.figure(figsize=(12, 8))

# Screen time by deprivation status
plt.subplot(2, 2, 1)
sns.barplot(x=deprivation_screen_time.index, y=deprivation_screen_time.values)
plt.title('Avg. Total Screen Time by Deprivation Status')
plt.xlabel('Deprivation Status')
plt.ylabel('Avg. Total Screen Time (Hours)')

# Well-being by deprivation status
plt.subplot(2, 2, 2)
deprivation_wellbeing.T.plot(kind='bar', ax=plt.gca())
plt.title('Avg. Well-being Scores by Deprivation Status')
plt.xlabel('Well-being Indicators')
plt.ylabel('Avg. Well-being Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

##### CORRELATION HEATMAP #####

# Select only numerical columns for the correlation matrix
numeric_columns = merged_df.select_dtypes(include=['float64', 'int64'])

# Correlation heatmap between numerical variables
plt.figure(figsize=(14, 10))
correlation_matrix = numeric_columns.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix: Screen Time and Well-being Indicators')
plt.show()

# Save the cleaned and merged dataset
merged_df.to_csv('cleaned_data.csv', index=False)
print("\nCleaned and merged dataset saved as 'cleaned_data.csv'.")
