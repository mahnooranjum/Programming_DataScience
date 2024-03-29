# -*- coding: utf-8 -*-
"""Demo221_Filter_Nature_Quasi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1msIeb9RL_vGKY1Wheupr98uzbwcjqGXc

## Quasi-constant features


*   show the same value for the great majority of the observations of the dataset
*   provide little information
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from google.colab import drive
drive.mount("/content/gdrive")
data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/FeatureSelection/train_santander.csv', nrows = 5000)
y = data.TARGET
X = data.drop(columns=['TARGET'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape, X_test.shape

[col for col in data.columns if data[col].isnull().sum() > 0]

"""## Removing quasi-constant features

Variance threshold 
"""

obj = VarianceThreshold(threshold=0.01) 
obj.fit(X_train)

# not quasi-constant
sum(obj.get_support())

# print the quasi-constant features
len([x for x in X_train.columns if x not in X_train.columns[obj.get_support()]])

# [x for x in X_train.columns if x not in X_train.columns[obj.get_support()]]

X_train['imp_amort_var18_hace3'].value_counts() / np.float(len(X_train))

# Remove
X_train = obj.transform(X_train)
X_test = obj.transform(X_test)

X_train.shape, X_test.shape

"""### From scratch"""

data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/FeatureSelection/train_santander.csv', nrows = 5000)
y = data.TARGET
X = data.drop(columns=['TARGET'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape, X_test.shape

quasi_features = []
for col in X_train.columns:
    doms = (X_train[col].value_counts() / np.float(
        len(X_train))).sort_values(ascending=False).values[0]
    if doms > 0.998:
        quasi_features.append(col)

len(quasi_features)

quasi_features[0]

X_train['imp_op_var40_efect_ult1'].value_counts() / np.float(len(X_train))