# -*- coding: utf-8 -*-
"""Demo223_Filter_Metric.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14kHwKexfEt-hf5GpK_SCuz9sDHWkAqne

## Metric based selection

- Use one model per feature 
- Predict 
- Evaluate 
- Rank 
- Select !
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount("/content/gdrive")

data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/FeatureSelection/train_titanic.csv')

data.keys()

y = data.Survived
X = data.drop(columns=['Survived'])

X.head()

y.head()

"""# Analyze"""

X = X.drop(columns=['Name', 'Ticket'])

X.head()

X.isnull().sum()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
temp = encoder.fit_transform(X['Sex'].values.reshape(-1,1))
X['Sex'] = temp
X.head()

def impute(df, columns, dft):
    df_temp = df.copy()
    for column in columns:
      df_temp[column] = df_temp[column].apply(lambda x: np.random.choice(dft[column].dropna().values) if pd.isnull(x) else x)
    return df_temp

X['Embarked'].unique()

X = impute(X, ['Embarked'], X)
encoder = LabelEncoder()
temp = encoder.fit_transform(X['Embarked'].values.reshape(-1,1))
X['Embarked'] = temp
X.head()

mapper = {k:i for i, k in enumerate(X['Cabin'].unique(), 0)} 
# mapper[np.nan] = 'M'
X['Cabin'] = X['Cabin'].map(mapper)
X.head()

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

from sklearn.impute import SimpleImputer
obj = SimpleImputer(missing_values = np.nan, strategy= 'most_frequent')
X_train = obj.fit_transform(X_train)
X_test = obj.transform(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y.head()

X.columns

X_train = pd.DataFrame(X_train, columns = X.columns)
X_test = pd.DataFrame(X_test, columns = X.columns)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
classifiers = [
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    SVC(),
    LogisticRegression()
    ]
texts = [    "DecisionTreeClassifier",
             "KNeighborsClassifier",
             "SVC",
             "LogisticRegression"]

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score
metrics_cl = [accuracy_score, balanced_accuracy_score, recall_score, precision_score]

clf = classifiers[0]
metric = metrics_cl[1]
scores = []
for feature in X_train.columns:
    clf.fit(X_train[feature].to_frame(), y_train)
    y_pred = clf.predict(X_test[feature].to_frame())
    scores.append(metric(y_test, y_pred))

# let's add the variable names and order it for clearer visualisation
scores = pd.Series(scores)
scores.index = X_train.columns
scores.sort_values(ascending=False)

# let's plot
scores.sort_values(ascending=False).plot.bar(figsize=(20, 8))

scores[scores > 0.6]

"""### Regression"""

data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/FeatureSelection/train_house.csv')

data.keys()

y = data.SalePrice
X = data.drop(columns=['SalePrice'])

X.dtypes

X['SaleCondition'].dtype

objs = []
nums = []
for i in X.columns:
  if X[i].dtype == 'O':
    objs.append(i)
  else:
    nums.append(i)

na_objs = []
na_nums = []
for i in X.columns:
  if (X[i].isnull().sum() > 0):
    print(i, " ", X[i].isnull().sum())
    if X[i].dtype == 'O':
      na_objs.append(i)
    else:
      na_nums.append(i)

na_nums

na_objs

def impute(df, columns, dft):
    df_temp = df.copy()
    for column in columns:
      df_temp[column] = df_temp[column].apply(lambda x: np.random.choice(dft[column].dropna().values) if pd.isnull(x) else x)
    return df_temp

X = impute(X,na_nums + na_objs , X)

X.isnull().sum()

X.head()

for col in objs:
  mapper = {k:i for i, k in enumerate(X[col].unique(), 0)} 
  X[col] = X[col].map(mapper)

X.head()

objs_oh = []
for col in objs:
  if len(X[col].unique())>2:
    objs_oh.append(col)

objs_oh

len(X.columns)

for i in objs_oh:
  X = pd.concat([X, pd.get_dummies(X[i], prefix = i, drop_first=True)], axis=1)

X = X.drop(columns=objs_oh)

len(X.columns)

# from sklearn.decomposition import PCA
# obj = PCA()
# X   = obj.fit_transform(X)

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = pd.DataFrame(X_train, columns = X.columns)
X_test = pd.DataFrame(X_test, columns = X.columns)

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
regressors = [
    DecisionTreeRegressor(),
    KNeighborsRegressor(),
    SVR(),
    RandomForestRegressor()
    ]
texts = [    "DecisionTreeRegressor",
              "KNeighborsRegressor",
              "SVR",
              "RandomForestRegressor"]


from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

metrics_r = [mean_squared_error, r2_score, explained_variance_score]



mdl = regressors[0]
metric = metrics_r[0]
scores = []
for feature in X_train.columns:
    mdl.fit(X_train[feature].to_frame(), y_train)
    y_pred = mdl.predict(X_test[feature].to_frame())
    scores.append(metric(y_test, y_pred))

# let's add the variable names and order it for clearer visualisation
scores = pd.Series(scores)
scores.index = X_train.columns
scores.sort_values(ascending=False)

# let's plot
scores.sort_values(ascending=False).plot.bar(figsize=(20, 8))

len(scores[scores < (sum(scores)/(len(scores)))])

len(scores)

scores.sort_values(ascending=True)

