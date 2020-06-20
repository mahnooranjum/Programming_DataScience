# -*- coding: utf-8 -*-
"""Demo181_RareCategories_SomeCategories.ipynb

## Rare Labels 

- Values present for a small percentage 

- Usually present less than 5% 

- Concept of cardinality 

## Rare label consequences 

- May add information in low cardinality 

- May add noise is high cardinality 


### Engineering Rare Labels 

- Replacing by most frequent label
- Grouping all rare labels together 

Categorical variables can have:

- One predominant category
- A small number of categories
- High cardinality
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/trainh.csv")

data.head()

data.columns

# get number of categories in variables 
categoricals = []
for col in data.columns:
    if data[col].dtypes =='O':
      print('{} categories : {} '.format(col, len(data[col].unique())))
      categoricals.append(col)

# Get variables with more than n categories 
n = 8
cats = []
for col in data.columns:
    if data[col].dtypes =='O': 
        if len(data[col].unique())>n: 
            print('{} categories : {} '.format(col, len(data[col].unique())))
            cats.append(col)

for col in cats:
    if data[col].dtypes =='O': # if the variable is categorical
      print(100*data.groupby(col)[col].count()/np.float(len(data)))
      print()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[cats], data.SalePrice,
                                                    test_size=0.2)
X_train.shape, X_test.shape

def label_encoder(X_train, X_test, columns, na_flag = False):
  import random
  for col in columns:
      mapper = {k:i for i, k in enumerate(X_train[col].unique(), 0)}
      if na_flag:
        mapper[np.nan] = np.nan
      X_train.loc[:, col] = X_train.loc[:, col].map(mapper)
      X_test.loc[:, col] = X_test.loc[:, col].map(mapper)
      X_test[col] = X_test[col].fillna(random.choice(list(mapper.values())))

label_encoder(X_train, X_test, cats)

X_train.isnull().sum()

X_test.isnull().sum()

sns.set()
for i in cats:
  plt.figure()
  sns.distplot(X_train[i], kde=False)

def new_label_imputation(Xtrain, Xtest, threshold, cats):
  X_train, X_test = Xtrain.copy(), Xtest.copy()
  for col in cats:
      rows = len(X_train)
      temp_df = pd.Series(100*X_train[col].value_counts() / rows)
      nonrares = temp_df[temp_df>=threshold].index # non-rare labels
              
      X_train[col] = np.where(Xtrain[col].isin(nonrares), Xtrain[col], 'rare')
      X_test[col] = np.where(Xtest[col].isin(nonrares), Xtest[col], 'rare')
  return X_train, X_test

X_train_rare, X_test_rare = new_label_imputation(X_train, X_test, 10, cats)

label_encoder(X_train_rare, X_test_rare, cats)

sns.set()
for i in cats:
  fig, ax = plt.subplots(1,2, figsize=(10,5))
  sns.distplot(X_train[i], kde=False, ax=ax[0])
  sns.distplot(X_train_rare[i], kde=False, ax=ax[1])

def frequent_imputation(Xtrain, Xtest, threshold, cats):
  X_train, X_test = Xtrain.copy(), Xtest.copy()
  for col in cats:
      rows = len(X_train)
      temp_df = pd.Series(100*X_train[col].value_counts() / rows)
      nonrares = temp_df[temp_df>=threshold].index # non-rare labels
      
      frequent_cat = X_train.groupby(col)[col].count().sort_values().tail(1).index.values[0]
        
      X_train[col] = np.where(Xtrain[col].isin(nonrares), Xtrain[col], frequent_cat)
      X_test[col] = np.where(Xtest[col].isin(nonrares), Xtest[col], frequent_cat)

  return X_train, X_test

X_train_freq, X_test_freq = frequent_imputation(X_train, X_test, 10, cats)

sns.set()
for i in cats:
  fig, ax = plt.subplots(1,2, figsize=(10,5))
  sns.distplot(X_train[i], kde=False, ax=ax[0])
  sns.distplot(X_train_freq[i], kde=False, ax=ax[1])

def regressor(X_train, y_train, X_test, y_test, cols, model):
  from sklearn.metrics import mean_squared_error
  model.fit(X_train[cols],y_train)
  y_pred = model.predict(X_test[cols])
  print(mean_squared_error(y_test, y_pred))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
regressor(X_train_rare, y_train, X_test_rare, y_test, cats, model)
regressor(X_train_freq, y_train, X_test_freq, y_test, cats, model)

from sklearn.linear_model import RidgeCV
model = RidgeCV()
regressor(X_train, y_train, X_test, y_test, cats, model)

from sklearn.linear_model import Ridge
model = RidgeCV()
regressor(X_train, y_train, X_test, y_test, cats, model)

from sklearn.svm import SVR
model = SVR()
regressor(X_train, y_train, X_test, y_test, cats, model)

from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
regressor(X_train, y_train, X_test, y_test, cats, model)

from sklearn.svm import LinearSVR
model = LinearSVR()
regressor(X_train, y_train, X_test, y_test, cats, model)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
regressor(X_train, y_train, X_test, y_test, cats, model)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
regressor(X_train, y_train, X_test, y_test, cats, model)

from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
regressor(X_train, y_train, X_test, y_test, cats, model)