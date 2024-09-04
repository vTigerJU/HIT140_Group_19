import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

#ALTERNATIVE HYPOTHESES -likely true
#Living in a deprived area lowers wellbeing

#NULL HYPOTHESES
#Living in a deprived area doesn't affect wellbeing

subjectDescription = pd.read_csv("dataset1.csv")
watchTime = pd.read_csv("dataset2.csv")
wellBeing = pd.read_csv("dataset3.csv")

def totalScreenTime():
    #New feature
    #Finds the total screen time per individual
    colList = list(watchTime)
    colList.remove("ID")
    watchTime["totalTime"] = watchTime[colList].sum(axis=1)
    watchTime.drop(watchTime.iloc[:,1:9], inplace=True, axis=1) #Remaining ID and total watch time

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
dfNonDeprived = dfCombined.loc[dfCombined['deprived'] == 0]
dfDeprived = dfCombined.loc[dfCombined['deprived'] == 1]
print(dfCombined.describe(include="all"))

#two sample t-test
x_barDeprived = st.tmean(dfDeprived["wellBeingScore"])
sDeprived = st.tstd(dfDeprived["wellBeingScore"])
nDeprived = len(dfDeprived)

x_barNonDeprived = st.tmean(dfNonDeprived["wellBeingScore"])
sNonDeprived = st.tstd(dfNonDeprived["wellBeingScore"])
nNonDeprived = len(dfNonDeprived)

t_stats, p_val = st.ttest_ind_from_stats(x_barDeprived, sDeprived, nDeprived, x_barNonDeprived, sNonDeprived, nNonDeprived, equal_var=False, alternative='two-sided')
print("\t t-statistic (t*): %.2f" % t_stats) # -6.81
print("\t p-value: %.4f" % p_val) #0.0000


def plot():
    plt.hist(dfCombined["wellBeingScore"].loc[dfCombined['deprived'] == 1], color='blue', edgecolor='black',bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], align="mid",width=0.8)
    plt.title("Well Being")
    plt.xlabel("Well Being")
    plt.ylabel("Amount people")
    plt.show()

