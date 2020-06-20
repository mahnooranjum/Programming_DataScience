# -*- coding: utf-8 -*-
"""Demo153_RareCategories.ipynb

## Rare Categories

- Labels 

- The number of labels in the dataset are different 

- __high cardinality__ refers to uniqueness of data values 

- The lower the cardinality, the more duplicated elements in a column

-  A column with the lowest possible cardinality would have the same value for every row

- Highly cardinal variables dominate tree based algorithms

- Labels may only be present in the training data set, but not in the test data set

- Labels may appear in the test set that were not present in the training set


__Tree methods are biased towards variables with many labels__
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")

cat_cols = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

for i in cat_cols:
  print('Number of categories in the variable {}: {}'.format(i,len(data[i].unique())))

print('Total rows: {}'.format(len(data)))

data['Sex'].value_counts()

data['Cabin_processed'] = data['Cabin'].astype(str).str[0]
data['Cabin_processed_X'] = data['Cabin'].astype(str).str[1]
cat_cols = [ 'Sex', 'Embarked', 'Cabin_processed']

for i in cat_cols:
  sns.catplot(x=i, kind='count', data=data)

data['Cabin_processed'].value_counts() / len(data)



for i in cat_cols:
  sns.catplot(x=i,data=data, hue='Survived', kind='count', palette="ch:.25")

"""### Transform Rare Labels"""

_temp = pd.Series(data['Cabin_processed'].value_counts() / len(data))
_temp.sort_values(ascending=False)
_temp

_temp = pd.Series(data['Cabin_processed'].value_counts() / len(data))
_temp

for i in _labels:
  data['Cabin_processed'].replace(i, 'rare', inplace=True)

_temp = pd.Series(data['Cabin_processed'].value_counts() / len(data))
_temp