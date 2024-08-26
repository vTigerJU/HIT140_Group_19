import pandas as pd

df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

#innerJoin = df1.merge(df2, on="ID").merge(df3, on="ID")
innerJoin = df1.merge(df3, on="ID")
print(innerJoin.head())
print(innerJoin.describe(include="all"))