# -*- coding: utf-8 -*-
"""Demo206_CrashCourseML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-VXVpt25spcJ4GReJ_LtrXNva4CJ4mog
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/gdrive')

data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")

data.head()

data.columns

data.index

data.loc[3, ['PassengerId']]

data.iloc[3, [0]]

data.loc[:, :]

print("Total entries: {}".format(len(data)))

len(data)

data.values.shape

data.shape

"""##  NAN"""

for i in range(len(data.columns)):
  print("{:20} has {:5} NaN values with {:4.2f}%".format(data.columns[i], data.iloc[:,i].isna().sum(),\
                                                        100*data.iloc[:,i].isna().sum()/len(data)))

data.Cabin.isna().sum()

data.Age.isna().sum()

dataAge = data.loc[:, ['Age']]

type(dataAge)

import seaborn as sns

sns.distplot(dataAge['Age'])

from sklearn.impute import SimpleImputer
obj = SimpleImputer(missing_values = np.nan, strategy= 'constant',  fill_value = -10)
data_t = obj.fit_transform(dataAge)

type(dataAge)

type(data_t)

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.distplot(dataAge['Age'], ax = ax[0], color='blue')
sns.distplot(data_t[:,0], ax = ax[1], color='red')

from sklearn.impute import SimpleImputer
obj = SimpleImputer(missing_values = np.nan, strategy= 'mean')
data_t = obj.fit_transform(dataAge)

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.distplot(dataAge['Age'], ax = ax[0], color='blue')
sns.distplot(data_t[:,0], ax = ax[1], color='red')

"""## UNIQUE"""

for i in range(len(data.columns)):
  print("{:4} has {:5} unique values with {:4.2f}%".format(i, len(data.iloc[:,i].dropna().unique()),\
                                                           100*len(data.iloc[:,i].dropna().unique())/len(data.iloc[:,i].dropna())))

discrete = []
for i in range(len(data.columns)):
  percentage = 100*len(data.iloc[:,i].dropna().unique())/len(data.iloc[:,i].dropna())
  if percentage < 10:
    discrete.append(i)
    print("{:20} has {:5} unique values with {:4.2f}%".format(data.columns[i], len(data.iloc[:,i].dropna().unique()), percentage))

discrete

data.columns[discrete]

data.Pclass.value_counts()

for i in discrete:
  print(data.columns[i])

for i in discrete:
  plt.figure()
  data.iloc[:,i].value_counts().plot.bar(color='blue')

  plt.xlabel(i)
  plt.title(data.columns[i])
  plt.show()

"""# OUTLIERS"""

data.columns

sns.catplot(x="Survived", y="Age", kind="box", data=data)
sns.despine(left=False, right=False, top=False)

data.Age.hist(bins=50, color='green')

data['Age'].describe()

data_processed = data.copy()

_temp = np.ceil(data['Age'].mean() + 1.5 * data['Age'].std())
data_processed.loc[data_processed.Age >= _temp, 'Age'] = _temp

_temp = np.ceil(data['Age'].mean() - 1.5 * data['Age'].std())
data_processed.loc[data_processed.Age <= _temp, 'Age'] = _temp

data.Age.hist(bins=50, color='green')

data_processed.Age.hist(bins=50, color='green')

"""# Feature Scaling"""

dataAge.head()

dataAge.Age.hist()

from sklearn.preprocessing import MinMaxScaler
obj = MinMaxScaler()
dataAge = obj.fit_transform(dataAge)

dataAge = pd.DataFrame(dataAge, columns = ['Age'])
type(dataAge)

dataAge.Age.hist()

"""# Label Encoding"""

data_processed = data.copy()

data_processed.head()

idx = list(data_processed.columns).index('Sex')

idx

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
data_processed.iloc[:, idx] = labelencoder_X.fit_transform(data_processed.iloc[:, idx])

data_processed.head()

"""# One Hot Encoding"""

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(handle_unknown = 'ignore')
temp = onehotencoder.fit_transform(data_processed.iloc[:, [idx]]).toarray()

type(temp)

data_processed.values.shape

temp.shape

data_processed = data_processed.drop(columns = [data.columns[idx]])

data_processed.head()

data_processed = pd.concat([pd.DataFrame(temp), data_processed], axis=1)

data_processed.head()

