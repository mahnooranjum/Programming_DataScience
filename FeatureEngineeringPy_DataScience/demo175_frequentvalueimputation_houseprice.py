# -*- coding: utf-8 -*-
"""Demo175_FrequentValueImputation_HousePrice.ipynb


## Frequent Value imputation

- Works on categorical data 

- Equivalent to the mean/median imputation in numerical data 

### When to use it? 

- Data is MCAR

### Pros

- Easy
- Completes dataset

### Cons

- Distorts the variable and its relations with other variables
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/trainh.csv")

data.head()

data.select_dtypes(include=['object'])

data = data[ ['BsmtQual', 'FireplaceQu', 'GarageType', 'Utilities',	'LotConfig','SalePrice'] ]

data.isnull().mean()

data.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[['BsmtQual', 'FireplaceQu', 'GarageType', 'Utilities',	'LotConfig']], data['SalePrice'], test_size=0.2)
X_train.shape, X_test.shape

data.columns

X_train.columns

X_test.columns

for i in X_train.columns:
  mapper = {k:i for i, k in enumerate(X_train[i].unique(), 0)} 
  mapper[np.nan] = np.nan
  X_train.loc[:, i] = X_train.loc[:, i].map(mapper)
  X_test.loc[:, i] = X_test.loc[:, i].map(mapper)

X_train.isnull().mean()

X_train.head()

sns.distplot(X_train['FireplaceQu'])

from sklearn.impute import SimpleImputer
obj = SimpleImputer(missing_values = np.nan, strategy= 'constant',  fill_value = 0)
X_train_0 = obj.fit_transform(X_train)
X_test_0 = obj.transform(X_test)

fig, ax = plt.subplots(1,2, figsize=(15,5))
sns.distplot(X_train['FireplaceQu'], ax = ax[0], color='blue')
sns.distplot(X_train_0[:,0], ax = ax[1], color='red')

from sklearn.impute import SimpleImputer
obj = SimpleImputer(missing_values = np.nan, strategy= 'most_frequent')
X_train_mode = obj.fit_transform(X_train)
X_test_mode = obj.transform(X_test)

fig, ax = plt.subplots(1,2, figsize=(15,5))
sns.distplot(X_train['FireplaceQu'], ax = ax[0], color='blue')
sns.distplot(X_train_mode[:,0], ax = ax[1], color='red')

print('Std original: ', X_train['FireplaceQu'].std())
print('Std 0: ', X_train_0[:,0].std())
print('Std mode: ', X_train_mode[:,0].std())

"""### Model performance"""

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_0,y_train)
y_pred = model.predict(X_test_0)
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_mode,y_train)
y_pred = model.predict(X_test_mode)
print(mean_squared_error(y_test, y_pred))

from sklearn.linear_model import RidgeCV
model = RidgeCV()
model.fit(X_train_0,y_train)
y_pred = model.predict(X_test_0)
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_mode,y_train)
y_pred = model.predict(X_test_mode)
print(mean_squared_error(y_test, y_pred))

from sklearn.linear_model import Ridge
model = RidgeCV()
model.fit(X_train_0,y_train)
y_pred = model.predict(X_test_0)
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_mode,y_train)
y_pred = model.predict(X_test_mode)
print(mean_squared_error(y_test, y_pred))

from sklearn.svm import SVR
model = SVR()
model.fit(X_train_0,y_train)
y_pred = model.predict(X_test_0)
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_mode,y_train)
y_pred = model.predict(X_test_mode)
print(mean_squared_error(y_test, y_pred))

from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
model.fit(X_train_0,y_train)
y_pred = model.predict(X_test_0)
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_mode,y_train)
y_pred = model.predict(X_test_mode)
print(mean_squared_error(y_test, y_pred))

from sklearn.svm import LinearSVR
model = LinearSVR()
model.fit(X_train_0,y_train)
y_pred = model.predict(X_test_0)
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_mode,y_train)
y_pred = model.predict(X_test_mode)
print(mean_squared_error(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train_0,y_train)
y_pred = model.predict(X_test_0)
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_mode,y_train)
y_pred = model.predict(X_test_mode)
print(mean_squared_error(y_test, y_pred))

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train_0,y_train)
y_pred = model.predict(X_test_0)
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_mode,y_train)
y_pred = model.predict(X_test_mode)
print(mean_squared_error(y_test, y_pred))

from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
model.fit(X_train_0,y_train)
y_pred = model.predict(X_test_0)
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_mode,y_train)
y_pred = model.predict(X_test_mode)
print(mean_squared_error(y_test, y_pred))

