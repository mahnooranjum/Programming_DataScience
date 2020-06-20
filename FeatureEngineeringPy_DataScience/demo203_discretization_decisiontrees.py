# -*- coding: utf-8 -*-
"""Demo203_Discretization_DecisionTrees.ipynb


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

def dt_binning(X_train, X_test, y_train, cols):
  from sklearn.tree import DecisionTreeClassifier
  for col in cols:
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X_train[col].to_frame(), y_train)
    X_train[col] = model.predict_proba(X_train[col].to_frame())[:,1]
    X_test[col] = model.predict_proba(X_test[col].to_frame())[:,1]

dt_binning(X_train, X_test, y_train, ['Fare'])

i = 'Fare'
mapper = {k:i for i, k in enumerate(X_train[i].unique(), 0)} 
#mapper[np.nan] = np.nan
X_train.loc[:, i] = X_train.loc[:, i].map(mapper)
X_test.loc[:, i] = X_test.loc[:, i].map(mapper)

X_train.head()
sns.distplot(X_train['Fare'], kde=False)

# Explore Monotony
def monotony_plot(X_train,y_train, columns):
  for col in columns:
    fig = plt.figure()
    _temp = pd.concat([X_train, y_train], axis=1)
    fig = _temp.groupby([col])[y_train.name].mean().plot()
    fig.set_title('Processed')

monotony_plot(X_train, y_train, cols)

