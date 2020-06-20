# -*- coding: utf-8 -*-
"""Demo161_LinearRelationshipAssumption.ipynb
## Linear Relationship Assumption 

- Linear regression assumes that there is a linear relationship between X and Y

- Y ≈ α0 + α1X1 + α2X2 + ... + αnXn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set()
n = 500
x = np.linspace(1, 100, n)
y = x * 5 + np.random.randn(n)*100
data = pd.DataFrame({'X':x, 'y':y})
sns.scatterplot(x='X', y='y', data=data)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data[['X']], data['y'])
y_pred = model.predict(data[['X']])


fig, ax = plt.subplots()
sns.scatterplot(x='X', y='y', data=data, ax=ax,  edgecolor=None)
sns.scatterplot(x=data['X'], y=y_pred, ax=ax, color='r',  edgecolor=None)

"""### Logistic Regression [ Simple Classification ]"""

X = np.linspace(-6, 6, 100)
y = 1 / (1 + np.exp(-X))
sns.set_style("whitegrid")
with sns.color_palette('husl'):
  sns.lineplot(x=X, y=y)
plt.legend(labels=['Sigmoid'])

"""## Algorithms with Linear Assumptions

- Linear Regression
- Logistic Regression
- Linear Discriminant Analysis
- Principal Component Analysis

## Key Takeaways

- Linear models are easy to interpret and understand

- Linear models prove useful when data is linearly related 

- Non-linear models usually have trouble predicting outside the range of the training dataset
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

X[1,:]

X.shape

sns.set()
for i in range(X.shape[1]):
    plt.scatter(x=X[:,i], y=y)
    plt.title(i)
    plt.show()

L = [5,6,7,12]
NL = [10,9,4,0]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_train.shape, X_test.shape

from sklearn.linear_model import LinearRegression
for i in range(X.shape[1]):
  print(i)
  model = LinearRegression()
  model.fit(X_train[:,[i]], y_train)
  print('Train set')
  y_pred = model.predict(X_train[:,[i]])
  print('Linear Regression Train mse: {}'.format(mean_squared_error(y_train, y_pred)))
  print('Test set')
  y_pred = model.predict(X_test[:,[i]])
  print('Linear Regression Test mse: {}'.format(mean_squared_error(y_test, y_pred)))
  print()

for i in range(X.shape[1]):
  fig, ax = plt.subplots()
  plt.hist(X[:,i],bins=50)
  ax.set_xlabel(i)
  ax.set_ylabel('y')
  plt.show()

# let's normalise 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train[:,L+NL])
X_test = scaler.transform(X_test[:,L+NL])

X_train.shape

from sklearn.linear_model import LinearRegression
for i in range(X_train.shape[1]):
  print(i)
  model = LinearRegression()
  model.fit(X_train[:,[i]], y_train)
  print('Train set')
  y_pred = model.predict(X_train[:,[i]])
  print('Linear Regression Train mse: {}'.format(mean_squared_error(y_train, y_pred)))
  print('Test set')
  y_pred = model.predict(X_test[:,[i]])
  print('Linear Regression Test mse: {}'.format(mean_squared_error(y_test, y_pred)))
  print()

from sklearn.svm import SVR
for i in range(X_train.shape[1]):
  print(i)
  model = SVR()
  model.fit(X_train[:,[i]], y_train)
  print('Train set')
  y_pred = model.predict(X_train[:,[i]])
  print('SVR Train mse: {}'.format(mean_squared_error(y_train, y_pred)))
  print('Test set')
  y_pred = model.predict(X_test[:,[i]])
  print('SVR Test mse: {}'.format(mean_squared_error(y_test, y_pred)))
  print()

from sklearn.tree import DecisionTreeRegressor
for i in range(X_train.shape[1]):
  print(i)
  model = DecisionTreeRegressor()
  model.fit(X_train[:,[i]], y_train)
  print('Train set')
  y_pred = model.predict(X_train[:,[i]])
  print('DT Train mse: {}'.format(mean_squared_error(y_train, y_pred)))
  print('Test set')
  y_pred = model.predict(X_test[:,[i]])
  print('DT Test mse: {}'.format(mean_squared_error(y_test, y_pred)))
  print()