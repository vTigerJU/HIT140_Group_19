import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

subjectDescription = pd.read_csv("dataset1.csv")
wellBeing = pd.read_csv("dataset3.csv")

def totalScreenTime():
    #New feature
    #Finds the total screen time per individual
    watchTime = pd.read_csv("dataset2.csv")
    colList = list(watchTime)
    colList.remove("ID")
    watchTime["totalTime"] = watchTime[colList].sum(axis=1)
    watchTime.drop(watchTime.iloc[:,1:9], inplace=True, axis=1) #Remaining ID and total watch time

def twoSampleTest(df1, df2, col):
    x_bar1 = st.tmean(df1[col])
    s1 = st.tstd(df1[col])
    n1 = len(df1)

    x_bar2 = st.tmean(df2[col])
    s2 = st.tstd(df2[col])
    n2 = len(df2)

    t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1, x_bar2, s2, n2, equal_var=False, alternative='two-sided')
    print("\t t-statistic (t*): %.2f" % t_stats) # -6.81
    print("\t p-value: %.6f" % p_val) #0.0000

def plot(df, col, title):
    plt.hist(df[col], color='blue', edgecolor='black',bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], align="mid",width=0.8)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Population")
    plt.show()

#New feature
#Well being score
#Mode of all other ratings
colList = list(wellBeing)
colList.remove("ID")
wellBeingMode = wellBeing[colList].mode(axis=1)
wellBeing["wellBeingScore"] = wellBeingMode[0]
wellBeing.drop(wellBeing.iloc[:,1:15], inplace=True, axis=1)

#Merge of dataframes on ID
dfCombined = subjectDescription.merge(wellBeing, on="ID")#.merge(watchTime, on="ID")

dfNonMinority = dfCombined.loc[dfCombined['minority'] == 0]
dfMinority = dfCombined.loc[dfCombined['minority'] == 1]
dfNonMale = dfCombined.loc[dfCombined['gender'] == 0]
dfMale = dfCombined.loc[dfCombined['gender'] == 1]
dfNonDeprived = dfCombined.loc[dfCombined['deprived'] == 0]
dfDeprived = dfCombined.loc[dfCombined['deprived'] == 1]

#two sample t-test

#Null hypotheses: Living in a deprived area doesnt affect well being
#T-stat -6.81
#P 0.000000
print("Deprived area effect on happiness")
twoSampleTest(dfDeprived, dfNonDeprived, "wellBeingScore")
plot(dfDeprived,"wellBeingScore", "Deprived")
plot(dfNonDeprived,"wellBeingScore", "Non-deprived")

#Null hypotheses: Being a minority doesnt affect well being
#T-stat 4.49
#P 0.000007
print("Minority effect on happiness")
twoSampleTest(dfMinority, dfNonMinority, "wellBeingScore")
plot(dfMinority,"wellBeingScore", "Minority")
plot(dfNonMinority,"wellBeingScore", "Non-minority")

#Null hyporheses: Gender doesn't affect well being
#T-Stat 78.36
#P 0.000000
print("Gender effect on happiness")
twoSampleTest(dfMale, dfNonMale, "wellBeingScore")
plot(dfMale,"wellBeingScore", "Male")
plot(dfNonMale,"wellBeingScore", "Non-male")


