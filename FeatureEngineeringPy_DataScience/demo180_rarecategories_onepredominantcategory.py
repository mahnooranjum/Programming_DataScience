# -*- coding: utf-8 -*-
"""Demo180_RareCategories_OnePredominantCategory.ipynb


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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[data.columns[:-1]], data.SalePrice,
                                                    test_size=0.2)
X_train.shape, X_test.shape

def label_encoder(X_train, X_test, columns, na_flag = False):
  for col in columns:
      mapper = {k:i for i, k in enumerate(X_train[col].unique(), 0)}
      if na_flag:
        mapper[np.nan] = np.nan
      X_train.loc[:, col] = X_train.loc[:, col].map(mapper)
      X_test.loc[:, col] = X_test.loc[:, col].map(mapper)

X_train['Street'].head()

print(X_train['Street'].value_counts())

X_train['Street'].isnull().sum()

label_encoder(X_train, X_test, ['Street'])

X_train['Street'].head()

def regressor(X_train, y_train, X_test, y_test, cols, model):
  from sklearn.metrics import mean_squared_error
  model.fit(X_train[cols],y_train)
  y_pred = model.predict(X_test[cols])
  print(mean_squared_error(y_test, y_pred))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
regressor(X_train, y_train, X_test, y_test, ['Street'], model)

from sklearn.linear_model import RidgeCV
model = RidgeCV()
regressor(X_train, y_train, X_test, y_test, ['Street'], model)

from sklearn.linear_model import Ridge
model = RidgeCV()
regressor(X_train, y_train, X_test, y_test, ['Street'], model)

from sklearn.svm import SVR
model = SVR()
regressor(X_train, y_train, X_test, y_test, ['Street'], model)

from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
regressor(X_train, y_train, X_test, y_test, ['Street'], model)

from sklearn.svm import LinearSVR
model = LinearSVR()
regressor(X_train, y_train, X_test, y_test, ['Street'], model)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
regressor(X_train, y_train, X_test, y_test, ['Street'], model)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
regressor(X_train, y_train, X_test, y_test, ['Street'], model)

from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
regressor(X_train, y_train, X_test, y_test, ['Street'], model)

# get number of categories in variables 
categoricals = []
for col in data.columns:
    if data[col].dtypes =='O':
      print('{} categories : {} '.format(col, len(data[col].unique())))
      categoricals.append(col)

# Get variables with less than n categories 
n = 4
low_cardinals = []
for col in data.columns:
    if data[col].dtypes =='O': 
        if len(data[col].unique())<n: 
            print('{} categories : {} '.format(col, len(data[col].unique())))
            low_cardinals.append(col)

for col in low_cardinals:
    if data[col].dtypes =='O': # if the variable is categorical
      print(100*data.groupby(col)[col].count()/np.float(len(data)))
      print()

label_encoder(X_train, X_test, low_cardinals)

for col in low_cardinals:
  print(100*X_train.groupby(col)[col].count()/np.float(len(X_train)))
  print()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
regressor(X_train, y_train, X_test, y_test, low_cardinals, model)

from sklearn.linear_model import RidgeCV
model = RidgeCV()
regressor(X_train, y_train, X_test, y_test, low_cardinals, model)

from sklearn.linear_model import Ridge
model = RidgeCV()
regressor(X_train, y_train, X_test, y_test, low_cardinals, model)

from sklearn.svm import SVR
model = SVR()
regressor(X_train, y_train, X_test, y_test, low_cardinals, model)

from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
regressor(X_train, y_train, X_test, y_test, low_cardinals, model)

from sklearn.svm import LinearSVR
model = LinearSVR()
regressor(X_train, y_train, X_test, y_test, low_cardinals, model)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
regressor(X_train, y_train, X_test, y_test, low_cardinals, model)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
regressor(X_train, y_train, X_test, y_test, low_cardinals, model)

from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
regressor(X_train, y_train, X_test, y_test, low_cardinals, model)