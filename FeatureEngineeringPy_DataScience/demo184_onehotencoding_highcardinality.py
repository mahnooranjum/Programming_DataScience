# -*- coding: utf-8 -*-
"""Demo184_OneHotEncoding_HighCardinality.ipynb
## One Hot Encoding

- Replacing categorical variables by a matrix of boolean variables 

- Each variable is called a dummy variable 

- For gender, we can have variables such as; Male, Female and Non-Binary 

## Number of Dummies 

- Pandas and sklearn provide K dummy variables; where K is the number of unique labels in the variable 

- When K=2, drop one dummy variable 

- When K!=2, drop one dummy variable if the underlying variables provide complete information even without K variables

- Should always use K-1 dummies for linear regression models because it **looks** at all the variables while fitting to the train set 


# OHE of the top most common labels 
__When we have a highly cardinal variable, we can use the top most common categories and encode them only to prevent the exponential expansion of feature space__

### Pros

- Easy
- Does not expand the feature space exponentially

### Cons 

- Loss of information
- No information of the less common variables
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

# get number of categories in variables 
categoricals = []
for col in data.columns:
    if data[col].dtypes =='O':
      print('{} categories : {} '.format(col, len(data[col].unique())))
      categoricals.append(col)

# Get variables with more than n categories 
n = 8
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

data_raw = data.copy()

data = data_raw[cats + ['SalePrice']]

data.columns

def get_top_variables(data, column, n):
  frame = [x for x in data[column].value_counts().sort_values(ascending=False).head(n).index]
  for label in frame:
    data[label] = np.where(data[column]==label, 1, 0)
  data.drop(column, axis = 1, inplace=True)

data.head()

get_top_variables(data, 'SaleType', 5)

data.head()

for i in cats:
  if i in data.columns:
    get_top_variables(data, i, 5)

data.head()

