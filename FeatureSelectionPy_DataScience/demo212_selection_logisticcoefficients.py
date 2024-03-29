# -*- coding: utf-8 -*-
"""Demo212_Selection_LogisticCoefficients.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15iteQeB055Gswm7BWlSAewUza9AbNjTs

# **Survival of the FITtest**
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""## Get  Dataset"""

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

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

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
importance = classifier.coef_[0]
# summarize importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# plot importance
fig, ax = plt.subplots(figsize=(10,8))
sns.set_theme(style="whitegrid", palette="pastel")
g = sns.barplot([x for x in range(len(importance))], importance, ax=ax)
g.set_xticklabels([])
sns.set_style("whitegrid")
g.set_title("Feature Importance by Logistic Regression")

"""The positive scores indicate a feature that predicts class 1, whereas the negative scores indicate a feature that predicts class 0."""

type(importance)

n = 5
indices = (-abs(importance)).argsort()[:n]
print(indices)

y_pred = classifier.predict(X_test)

X_train.shape

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print(X_train.shape, acc_score)

classifier = LogisticRegression()
classifier.fit(X_train[:, indices], y_train)
y_pred = classifier.predict(X_test[:, indices])
acc_score = accuracy_score(y_test, y_pred)
print(X_train[:, indices].shape, acc_score)

