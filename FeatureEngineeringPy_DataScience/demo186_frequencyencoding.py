# -*- coding: utf-8 -*-
"""Demo186_FrequencyEncoding.ipynb

## Frequency Encoding

- For high cardinality 

- Replace each label by its frequency of occurence in the dataset

- Highly used in Kaggle competitions 

### Pros

- Easy
- Does not expand the feature space

### Disadvantages

- Variables having the same frequency will be lost
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/trainh.csv")

data.head()

data.columns

# Get variables with more than n categories 
n = 10
cats = []
for col in data.columns:
    if data[col].dtypes =='O': 
        if len(data[col].unique())>n: 
            print('{} categories : {} '.format(col, len(data[col].unique())))
            cats.append(col)

for col in cats:
    if data[col].dtypes =='O': # if the variable is categorical
      print(100*data.groupby(col)[col].count()/np.float(len(data)))
      print()

for col in cats:
    if data[col].dtypes =='O': # if the variable is categorical
      print(data.groupby(col)[col].count())
      print()

data[cats].isnull().sum()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[cats], data.SalePrice,
                                                    test_size=0.2)
X_train.shape, X_test.shape

def frequency_encoding(X_train, X_test, cols):
  import random
  for col in cols:
    mapper = X_train[col].value_counts().to_dict()
    X_train[col] = X_train[col].map(mapper)
    X_test[col] = X_test[col].map(mapper)
    #X_test[col] = X_test[col].fillna(random.choice(list(mapper.values())))

X_train.head()

frequency_encoding(X_train, X_test, cats)

X_train.head()