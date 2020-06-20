# -*- coding: utf-8 -*-
"""Demo199_NormalTransformations_Exponential.ipynb

## Normal transformations 

- Some models assume that the data is normally distributed 

- We can transform variables to show a normal distribution 


## Examples 

- Reciprocal or inverse transformations

- Logarithmic

- Square root transformation 

- Exponential 

- Box-Cox
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

def exp_transform(X_train, X_test, cols):
  for col in cols:
    X_train.loc[X_train[col]==0, col] = 0.0001
    X_train[col] = X_train[col]**(1/1.8)
    X_test[col] = X_train[col]**(1/1.8)

X_train.describe()

exp_transform(X_train, X_test, ['Fare'])
X_train.describe()

distro(X_train, ['Fare'])

