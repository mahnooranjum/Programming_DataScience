# -*- coding: utf-8 -*-
"""Demo146_NumericalFeatures.ipynb

"""

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/gdrive')

data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")

data.head()

"""### Continuous Variables"""

data.columns

print("Total entries: {}".format(len(data)))

data.values.shape

for i in range(len(data.columns)):
  percentage = 100*len(data.iloc[:,i].unique())/len(data.iloc[:,i])
  uniques = len(data.iloc[:,i].unique())
  print("{:4} has {:5} unique values with {:4.2f}%".format(i, uniques, percentage))

for i in range(len(data.columns)):
  print("{:4} has {:5} unique values with {:4.2f}%".format(i, len(data.iloc[:,i].dropna().unique()), 100*len(data.iloc[:,i].dropna().unique())/len(data.iloc[:,i].dropna())))

for i in range(len(data.columns)):
  print("{:4} has {:5} NaN values with {:4.2f}%".format(i, data.iloc[:,i].isna().sum(), 100*data.iloc[:,i].isna().sum()/len(data)))

i = 68
df = pd.concat([data.iloc[:,i], data.y], axis=1).dropna()
print(len(df))
df.head()

#!mkdir scatterPlots
row, col = 40, 2
fig, ax = plt.subplots(row, col, figsize=(20,160))
j = -1
for i in range(len(data.columns)):
  if i%row==0:
    j+=1
  df = pd.concat([data.iloc[:,i], data.y], axis=1).dropna()
  ax[i%row][j].scatter(df.iloc[:,0], df.iloc[:,1],color = 'blue')
  ax[i%row][j].set_xticks([])
  ax[i%row][j].set_yticks([])
  ax[i%row][j].set_xlabel(data.columns[i])
  ax[i%row][j].set_ylabel('y')
  
plt.savefig('ScatterPlotMatrix')
plt.show()

!mkdir histPlots
for i in range(len(data.columns)):
  plt.figure()
  df = pd.concat([data.iloc[:,i], data.y], axis=1).dropna()
  plt.hist(df.iloc[:,0],color = 'blue')
  plt.xlabel(i)
  plt.ylabel('value')
  plt.xticks([])
  plt.yticks([])
  plt.grid(True)
  plt.savefig('histPlots/column_' + str(i) + '.png')
  plt.show()

!zip -r histPlots.zip histPlots
from google.colab import files
files.download("/content/histPlots.zip")

!mkdir histPlots50
for i in range(len(data.columns)):
  plt.figure()
  df = pd.concat([data.iloc[:,i], data.y], axis=1).dropna()
  plt.hist(df.iloc[:,0],color = 'blue', bins=50)
  plt.xlabel(i)
  plt.ylabel('value')
  plt.xticks([])
  plt.yticks([])
  plt.grid(True)
  plt.savefig('histPlots50/column_' + str(i) + '.png')
  #plt.show()
!zip -r histPlots50.zip histPlots50
from google.colab import files
files.download("/content/histPlots50.zip")

"""### Discrete Variables"""

discrete = []
for i in range(len(data.columns)):
  percentage = 100*len(data.iloc[:,i].dropna().unique())/len(data.iloc[:,i].dropna())
  if percentage < 30:
    discrete.append(i)
    print("{:4} has {:5} unique values with {:4.2f}%".format(i, len(data.iloc[:,i].dropna().unique()), percentage))

#!mkdir histPlotsDiscrete
for i in discrete:
  plt.figure()
  df = pd.concat([data.iloc[:,i], data.y], axis=1).dropna()
  plt.hist(df.iloc[:,0],color = 'blue', bins=100)
  plt.xlabel(i)
  plt.ylabel('value')
  plt.xticks([])
  plt.yticks([])
  plt.grid(True)
  #plt.savefig('histPlots50/column_' + str(i) + '.png')
  plt.show()
# !zip -r histPlots50.zip histPlots50
# from google.colab import files
# files.download("/content/histPlots50.zip")

data.iloc[:,1].unique()

X = data.iloc[:,2:len(data.columns)-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2)

data_test = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/test.csv")
X_test = data_test.iloc[:,2:len(data.columns)-1].values

print(data.values.shape)
print(X.shape)
print(y.shape)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)

from sklearn.preprocessing import StandardScaler
obj = StandardScaler()
X_train = obj.fit_transform(X_train)
X_test = obj.transform(X_test)
X_val = obj.transform(X_val)

from sklearn.impute import SimpleImputer
obj = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train = obj.fit_transform(X_train)
X_test = obj.transform(X_test)
X_val = obj.transform(X_val)

from sklearn.svm import SVR
model = SVR(kernel='poly')
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_val, y_pred)

print(mse)