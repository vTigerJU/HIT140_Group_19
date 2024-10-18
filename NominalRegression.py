import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Function to calculate total screen time by summing weekday and weekend times for each device
def totalScreenTime(watchTime):
    watchTime['computerTime'] = watchTime['C_we'] + watchTime['C_wk']
    watchTime['gameTime'] = watchTime['G_we'] + watchTime['G_wk']
    watchTime['smartphoneTime'] = watchTime['S_we'] + watchTime['S_wk']
    watchTime['tvTime'] = watchTime['T_we'] + watchTime['T_wk']
    return watchTime

# Function to bin compositeScore into categories: low, medium, high
def categorizeWellBeing(df):
    # Define the bins for the categories
    bins = [0, 2.5, 3.5, 5]  # Example thresholds for well-being categories
    labels = ['low', 'medium', 'high']  # Categories
    df['wellBeingCategory'] = pd.cut(df['compositeScore'], bins=bins, labels=labels)
    return df

### MAIN CODE ###

# 1. Read in datasets (Make sure the paths to your datasets are correct)
subjectDescription = pd.read_csv("dataset1.csv")
watchTime = pd.read_csv("dataset2.csv")
wellBeing = pd.read_csv("dataset3.csv")

# 2. Create total screen time for each type by summing weekday and weekend times
watchTime = totalScreenTime(watchTime)

# 3. Create composite well-being score and bin the categories
wellBeing['compositeScore'] = wellBeing.drop(columns=['ID']).mean(axis=1)
df = subjectDescription.merge(wellBeing, on='ID').merge(watchTime, on='ID')
df = categorizeWellBeing(df)  # Add the well-being categories

# 4. Select independent and dependent variables (use the new screen time columns)
x = df[['computerTime', 'gameTime', 'smartphoneTime', 'tvTime', 'gender', 'deprived', 'minority']]
y = df['wellBeingCategory']  # Use the well-being categories as the target variable

# 5. Scaling the features using StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 6. Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)

# 7. Apply Multinomial Logistic Regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(x_train, y_train)

# 8. Make predictions
y_pred = model.predict(x_test)

# 9. Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Visualize the confusion matrix as a heatmap
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

