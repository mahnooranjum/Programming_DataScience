# -*- coding: utf-8 -*-
"""Demo190_TargetGuidedEncoding_WOE.ipynb

## Target Guided Encoding

- Capture valuable information while encoding 
- Ordering according to the target variable 
- Imputing using the probability of target variable 
- Using weight of evidence 

### Monotonicity and its implications in Target Guided Encoding

A monotonic relationship either:

- Increases the value of one variable with the increase in the value of the other variable 

- Decreases the value of one variable with the increase in the value of the other variable 

The target guided encoding methods assume a monotonic relationship.

### Pros

- Capture information 
- Do not expand feature space

### Cons

- Might cause overfitting


### Weight of Evidence

- Reference https://medium.com/@sundarstyles89/

- log (event / non-event)

- Handles missing values

- Handles outliers

- By using proper binning technique, it can establish monotonic relationship between the independent and dependent variable

- We obtain logistic scale of categories 

- The transformed variables all have the same scale so can be compared 

- Sometimes has loss of information 

- We assume there is no correlation between independent variables
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")

data.head()

data.columns

data = data.drop(['Name'], axis=1)

# Get variables with more than n categories 
n = 10
cats = ['Sex', 'Cabin']

data = data[cats+['Survived']]

for col in cats:
  print("{} unique categories : {}".format(col,len(data[col].unique())))

for col in cats:
  print("{} Missing : {}".format(col,data[col].isnull().sum()))

data.head()

for col in cats:
  data[col].fillna('MissingData', inplace=True)

data.head()

data['Cabin'] = data['Cabin'].astype(str).str[0]
data.head()

for col in cats:
  print("{} unique categories : {}".format(col,len(data[col].unique())))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[cats], data['Survived'],
                                                    test_size=0.2)
X_train.shape, X_test.shape

def tg_woe(Xtrain, Xtest, y_train, columns):
  X_train, X_test = Xtrain.copy(), Xtest.copy()
  _temp = pd.concat([X_train, y_train], axis=1)
  for col in columns:
    p = pd.DataFrame(_temp.groupby([col])[y_train.name].mean())
    p[y_train.name + "!"] =  1-p[y_train.name]
    p.loc[p[y_train.name]==0, y_train.name] = 0.000001
    p['woe'] = np.log(p[y_train.name]/p[y_train.name + "!"])
    mapper = p['woe'].to_dict()
    X_train[col] = X_train[col].map(mapper)
    X_test[col] = X_test[col].map(mapper)
  return X_train, X_test, mapper

X_train_encoded, X_test_encoded, mapper = tg_woe(X_train, X_test, y_train, cats)

mapper

X_train.head()

X_train_encoded.head()

sns.set()
# Explore Monotony
def monotony_plot(X_train, X_train_encoded, y_train, columns):
  for col in columns:
    fig = plt.figure()
    _temp = pd.concat([X_train, y_train], axis=1)
    fig = _temp.groupby([col])[y_train.name].mean().plot()
    fig.set_title('Unprocessed')
    fig = plt.figure()
    _temp = pd.concat([X_train_encoded, y_train], axis=1)
    fig = _temp.groupby([col])[y_train.name].mean().plot()
    fig.set_title('Processed')

monotony_plot(X_train, X_train_encoded, y_train, cats)