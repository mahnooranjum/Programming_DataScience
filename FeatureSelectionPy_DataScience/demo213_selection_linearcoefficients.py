# -*- coding: utf-8 -*-
"""Demo213_Selection_LinearCoefficients.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q0Bho_p1Q1G1t0Ttqx4n1DHUJLUwvDPo

# **Survival of the FITtest**
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""## Get  Dataset"""

from sklearn.datasets import load_boston
data = load_boston()

data.keys()

X = data.data
y = data.target

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
importance = model.coef_
# summarize importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# plot importance
fig, ax = plt.subplots(figsize=(10,8))
sns.set_theme(style="whitegrid", palette="pastel")
g = sns.barplot([x for x in range(len(importance))], importance, ax=ax)
g.set_xticklabels([])
sns.set_style("whitegrid")
g.set_title("Feature Importance by Linear Regression")

type(importance)

n = 5
indices = (-abs(importance)).argsort()[:n]
print(indices)

y_pred = model.predict(X_test)

X_train.shape

from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test, y_pred)
print(X_train.shape, error)

model = LinearRegression()
model.fit(X_train[:, indices], y_train)
y_pred = model.predict(X_test[:, indices])
error = mean_squared_error(y_test, y_pred)
print(X_train[:, indices].shape, error)

