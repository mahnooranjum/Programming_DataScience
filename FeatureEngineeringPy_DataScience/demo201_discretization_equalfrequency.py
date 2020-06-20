# -*- coding: utf-8 -*-
"""Demo201_Discretization_EqualFrequency.ipynb

## Discretisation

- Transforming continuous variables to discrete ones 

- Ouliers transform 

- Resolve skewness 

- Can be target guided

## Target independent methods 

- Equal width 
- Equal frequency 


#### Target guided methods 

- Trees


## Equal frequency discretisation

- Each bin has same number of observations 

## Equal frequency discretisation

- Each bin has same range
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")

cats = ['Age', 'Fare', 'Survived']

data = data[cats]
data.head()

sns.set()
def distro(data, columns):
  import scipy.stats as stats
  for col in columns:
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    stats.probplot(data[col].dropna(), dist="norm", plot=ax[0])
    ax[0].set_title("QQPlot")
    sns.distplot(data[col], ax=ax[1])
    ax[1].set_title("Distribution")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[['Age', 'Fare']].fillna(data.mean()),
                                                    data['Survived'], test_size=0.2)
X_train.shape, X_test.shape

cols = cats[:-1]

distro(X_train, cols)

distro(X_train, ['Fare'])

X_train.head()

def equal_frequency(X_train, X_test, cols, n):
  for col in cols:
    X_train[col], bins = pd.qcut(X_train[col], q = n, retbins=True, precision=3)
    X_test[col] = pd.cut(X_test[col], bins = bins)

equal_frequency(X_train, X_test, ['Fare'], 5)

i = 'Fare'
mapper = {k:i for i, k in enumerate(X_train[i].unique(), 0)} 
#mapper[np.nan] = np.nan
X_train.loc[:, i] = X_train.loc[:, i].map(mapper)
X_test.loc[:, i] = X_test.loc[:, i].map(mapper)

X_train.head()
sns.distplot(X_train['Fare'])

# Explore Monotony
def monotony_plot(X_train,y_train, columns):
  for col in columns:
    fig = plt.figure()
    _temp = pd.concat([X_train, y_train], axis=1)
    fig = _temp.groupby([col])[y_train.name].mean().plot()
    fig.set_title('Processed')

monotony_plot(X_train, y_train, cols)

