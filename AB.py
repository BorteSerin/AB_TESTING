

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df_Test=pd.read_excel("C:/Users/caspe/Desktop/VBO/week5/measurement_problems/ab_testing/ab_testing.xlsx", sheet_name="Test Group")
df_Test.head()


df_Control=pd.read_excel("C:/Users/caspe/Desktop/VBO/week5/measurement_problems/ab_testing/ab_testing.xlsx", sheet_name="Control Group")
df_Control.head()

df_Test.shape
df_Control.shape
df_Test.isnull().sum()
df_Control.isnull().sum()
df_Test.describe().T
df_Control.describe().T
df_Test.rename(columns={"Purchase":"Test_Purchase"},inplace=True)
df_Control.rename(columns={"Purchase":"Control_Purchase"},inplace=True)
df_Control.head()
df_Test.head()
df_=pd.concat([df_Test["Test_Purchase"],df_Control["Control_Purchase"]],axis=1)
df_.head()
df_.info()
sns.histplot(x="Test_Purchase",data=df_,bins=5)
plt.show(block=True)
sns.histplot(x="Control_Purchase",data=df_,bins=5)
plt.show(block=True)
df_.plot(kind="hist")#ikisi bir arada
plt.show(block=True)


test_stat, pvalue = shapiro([df_["Control_Purchase"]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro([df_["Test_Purchase"]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



test_stat, pvalue = levene(df_["Test_Purchase"],
                           df_["Control_Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = ttest_ind(df_["Test_Purchase"] ,
                              df_["Control_Purchase"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
