# -*- coding: utf-8 -*-
"""Demo204_MixedVariables.ipynb
## Numbers and Labels? Why? 

- Mixed variables have numeric and labelled data

- If values are either numerical or categorical, we can just make two separate variables 

- If values have a label and a number, we will have to choose which part to keep and which part to remove

- We'll carry out stat analysis of these variables
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
cats = cats[1:]

for col in cats:
    if data[col].dtypes =='O': # if the variable is categorical
      print(100*data.groupby(col)[col].count()/np.float(len(data)))
      print()

data['Ticket'].unique()

data = data[cats+['Survived']]

data[cats[0]].head()

for col in ['Ticket']:
  data[col+'_numericals'] = np.where(data[col].str.isdigit(), data[col], np.nan)
  data[col+'_categoricals'] = np.where(data[col].str.isdigit(), np.nan, data[col],)

data.head()

data.dropna(subset = ['Ticket_numericals'], axis=0)

data.dropna(subset = ['Ticket_categoricals'], axis=0)

data

col = 'Cabin'
data[col] = data[col].str.replace(" ", "")
data[col + '_numerical'] = data[col].str.extract('(\d+)')
data[col + '_categorical'] = data[col].str.extract('([a-zA-Z]+)')

data.head(20)

