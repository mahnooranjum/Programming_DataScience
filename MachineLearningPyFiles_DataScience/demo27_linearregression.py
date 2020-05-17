
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
# Regression Dataset
n = 100
X, y = make_regression(n_samples=n, n_features=1, noise=80)
X = X.reshape(n)
datadict = {'data': X, 'target': y}
data = pd.DataFrame(data=datadict)
plt.scatter(X,y)
plt.show()

X = data.iloc[:,[0]]
X.shape

y = data.target.values

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X,y)
plt.plot(X_test, y_pred, 'r')
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)

from sklearn.datasets import load_boston
boston_data = load_boston()
X = boston_data['data']
y = boston_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler 
scaleObj = StandardScaler()
X_train = scaleObj.fit_transform(X_train)
X_test = scaleObj.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)

for i in range(0, X.shape[1]):
  plt.scatter(X_train[:,[i]], y_train)
  plt.scatter(X_test[:,[i]], y_pred)
  plt.show()