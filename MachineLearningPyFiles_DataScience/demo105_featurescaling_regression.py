# -*- coding: utf-8 -*-
"""Demo105_FeatureScaling_Regression.ipynb



# **Spit some [tensor] flow**

Let's see some models in action

`Leggo`
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Visualising the results
def evaluate(y_test, y_pred):
  from sklearn.metrics import mean_squared_error
  print("MSE :")
  print(mean_squared_error(y_test, y_pred))
  return

from sklearn.datasets import load_diabetes
data = load_diabetes()

X = data.data
y = data.target

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import Lasso
model = Lasso()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import Lasso
model = Lasso()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import ElasticNetCV 
model = ElasticNetCV()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import ElasticNetCV
model = ElasticNetCV()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import BayesianRidge
model = BayesianRidge()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import BayesianRidge
model = BayesianRidge()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.svm import SVR
model = SVR()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVR
model = SVR()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.svm import NuSVR
model = NuSVR()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import NuSVR
model = NuSVR()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.svm import LinearSVR
model = LinearSVR()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import LinearSVR
model = LinearSVR()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.gaussian_process import GaussianProcessRegressor
model = GaussianProcessRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.gaussian_process import GaussianProcessRegressor
model = GaussianProcessRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)