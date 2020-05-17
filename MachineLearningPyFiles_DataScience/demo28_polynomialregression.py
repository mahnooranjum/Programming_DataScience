
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from random import randint
# Regression Dataset
n = 10000
X, y = make_regression(n_samples=n, n_features=1)
X = X.reshape(n)
y = 3*X + 2*X**3
for i in range(0,len(y)):
  y[i] = y[i] + randint(0,100)
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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly = PolynomialFeatures(degree = 2)
model = make_pipeline(poly, LinearRegression())
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

plt.scatter(X,y)
plt.scatter(X_test, y_pred)
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly = PolynomialFeatures(degree = 3)
model = make_pipeline(poly, LinearRegression())
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

plt.scatter(X,y)
plt.scatter(X_test, y_pred)
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly = PolynomialFeatures(degree = 5)
model = make_pipeline(poly, LinearRegression())
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

plt.scatter(X,y)
plt.scatter(X_test, y_pred)
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)