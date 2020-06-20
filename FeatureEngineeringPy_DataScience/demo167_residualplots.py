# -*- coding: utf-8 -*-
"""Demo167_ResidualPlots.ipynb

## Variable distributions and their effects on Models

Reference 
[https://www.statisticssolutions.com/homoscedasticity/]

### Linear Regression Assumptions

- Linear relationship with the outcome Y
- Homoscedasticity
- Normality
- No Multicollinearity 

## Linear Assumption

- The X variable is linearly related to the dataset 
- Pearson correlation coefficient can determine the linearity magnitude  between variables 

## Normality Assumption

- The variable X follows a normal or gaussian distribution

## Homoscedasticity Assumption

- Homogeneity of variance

- Homoscedasticity describes a situation in which the error term (that is, the “noise” or random disturbance in the relationship between the independent variables and the dependent variable) is the same across all values of the independent variables


### Unaffected models

- Neural Networks
- Support Vector Machines
- Trees
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
sns.set()
import pandas as pd

from sklearn.datasets import load_boston
dataset = load_boston()

dataset.data.shape

dataset.target.shape

X = dataset.data
y = dataset.target

def residual_plot(y_pred, y_test, X):
  y_res = (y_test.reshape(-1) - y_pred.reshape(-1))
  # y_res = (y_test.reshape(-1) - y_pred.reshape(-1))**2
  # y_res = np.sqrt(y_res)
  for i in range(X.shape[1]):
    plt.figure()
    sns.residplot(x = y_res, y = X[:,i], lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
    plt.xlabel("Fitted values " + str(i))
    plt.xticks([])
    plt.title('Residual plot')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train_processed, X_test_processed, y_train_processed, y_test_processed = train_test_split(data_processed,
                                                    y,
                                                    test_size=0.2)

X_train_processed.shape, X_test_processed.shape,  y_train_processed.shape, y_test_processed.shape

from sklearn.preprocessing import StandardScaler
obj = StandardScaler()
X_train = obj.fit_transform(X_train)
X_test = obj.transform(X_test)

from sklearn.preprocessing import StandardScaler
obj = StandardScaler()
X_train_processed = obj.fit_transform(X_train_processed)
X_test_processed = obj.transform(X_test_processed)

from sklearn.metrics import mean_squared_error
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model = TransformedTargetRegressor( regressor=model,
                                    transformer=QuantileTransformer(n_quantiles=300,
                                                                    output_distribution='normal'))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_processed,y_train_processed)
y_pred_processed = model.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(mean_squared_error(y_test_processed, y_pred_processed))


residual_plot(y_pred, y_test, X_test)

from sklearn.linear_model import RidgeCV
model = RidgeCV()
model = TransformedTargetRegressor( regressor=model,
                                    transformer=QuantileTransformer(n_quantiles=300,
                                                                    output_distribution='normal'))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_processed,y_train_processed)
y_pred_processed = model.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(mean_squared_error(y_test_processed, y_pred_processed))

residual_plot(y_pred, y_test, X_test)

from sklearn.linear_model import Ridge
model = RidgeCV()
model = TransformedTargetRegressor( regressor=model,
                                    transformer=QuantileTransformer(n_quantiles=300,
                                                                    output_distribution='normal'))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_processed,y_train_processed)
y_pred_processed = model.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(mean_squared_error(y_test_processed, y_pred_processed))
residual_plot(y_pred, y_test, X_test)

from sklearn.svm import SVR
model = SVR()
model = TransformedTargetRegressor( regressor=model,
                                    transformer=QuantileTransformer(n_quantiles=300,
                                                                    output_distribution='normal'))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_processed,y_train_processed)
y_pred_processed = model.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(mean_squared_error(y_test_processed, y_pred_processed))

residual_plot(y_pred, y_test, X_test)

from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
model = TransformedTargetRegressor( regressor=model,
                                    transformer=QuantileTransformer(n_quantiles=300,
                                                                    output_distribution='normal'))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_processed,y_train_processed)
y_pred_processed = model.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(mean_squared_error(y_test_processed, y_pred_processed))

residual_plot(y_pred, y_test, X_test)

from sklearn.svm import LinearSVR
model = LinearSVR()
model = TransformedTargetRegressor( regressor=model,
                                    transformer=QuantileTransformer(n_quantiles=300,
                                                                    output_distribution='normal'))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_processed,y_train_processed)
y_pred_processed = model.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(mean_squared_error(y_test_processed, y_pred_processed))

residual_plot(y_pred, y_test, X_test)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model = TransformedTargetRegressor( regressor=model,
                                    transformer=QuantileTransformer(n_quantiles=300,
                                                                    output_distribution='normal'))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_processed,y_train_processed)
y_pred_processed = model.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(mean_squared_error(y_test_processed, y_pred_processed))

residual_plot(y_pred, y_test, X_test)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model = TransformedTargetRegressor( regressor=model,
                                    transformer=QuantileTransformer(n_quantiles=300,
                                                                    output_distribution='normal'))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_processed,y_train_processed)
y_pred_processed = model.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(mean_squared_error(y_test_processed, y_pred_processed))

residual_plot(y_pred, y_test, X_test)

from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
model = TransformedTargetRegressor( regressor=model,
                                    transformer=QuantileTransformer(n_quantiles=300,
                                                                    output_distribution='normal'))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(mean_squared_error(y_test, y_pred))

model.fit(X_train_processed,y_train_processed)
y_pred_processed = model.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(mean_squared_error(y_test_processed, y_pred_processed))


residual_plot(y_pred, y_test, X_test)

