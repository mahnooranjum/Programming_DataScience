# -*- coding: utf-8 -*-
"""Demo168_CompleteCaseAnalysis.ipynb


## Complete Case Analysis

- Also called listwise deletion

- Complete case analysis (CCA) is the term used to describe a statistical analysis that only includes participants for which we have no missing data on the variables of interest.

- CCA can be applied to both categorical and numerical variables.

### When to use it? 

- CCA works well when the data are missing completely at random. 

### Pros

- Easy
- Preserves variable distribution 

### Cons

- It can discard a large population of the original sample
- Loss of information
- CCA will be biased if data is not MCAR


CCA is an acceptable method when the amount of missing information is small
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")

data.head()

data.isnull().mean()

print('Complete Rows: ', data.dropna().shape[0])
print('Total Rows: ', data.shape[0])
print('% of data that is complete: ', 100 * data.dropna().shape[0]/ np.float(data.shape[0]))

data[data['Embarked'].isnull()]

sns.set()
sns.distplot(data['Age'], color='blue')
print(len(data['Age']))

sns.distplot(data['Age'].dropna(), color='red')
print(len(data['Age'].dropna()))

fig, ax = plt.subplots(figsize=(12,5))
sns.distplot(data['Fare'], color='purple', ax=ax)
sns.distplot(data.dropna(axis=0, subset=['Age'])['Fare'], color='green', ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('# of observations')

fig, ax = plt.subplots(figsize=(12,5))
sns.distplot(data['Fare'], color='purple', ax=ax)
sns.distplot(data.dropna(axis=0)['Fare'], color='green', ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('# of observations')

fig, ax = plt.subplots(figsize=(12,5))
sns.distplot(data['Pclass'], color='purple', ax=ax)
sns.distplot(data.dropna(axis=0, subset=['Age'])['Pclass'], color='green', ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('# of observations')