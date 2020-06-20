# -*- coding: utf-8 -*-
"""Demo183_OneHotEncoding.ipynb


## One Hot Encoding

- Replacing categorical variables by a matrix of boolean variables 

- Each variable is called a dummy variable 

- For gender, we can have variables such as; Male, Female and Non-Binary 

## Number of Dummies 

- Pandas and sklearn provide K dummy variables; where K is the number of unique labels in the variable 

- When K=2, drop one dummy variable 

- When K!=2, drop one dummy variable if the underlying variables provide complete information even without K variables

- Should always use K-1 dummies for linear regression models because it **looks** at all the variables while fitting to the train set
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

data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]

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
        if len(data[col].unique())<n: 
            print('{} categories : {} '.format(col, len(data[col].unique())))
            cats.append(col)

for col in cats:
    if data[col].dtypes =='O': # if the variable is categorical
      print(100*data.groupby(col)[col].count()/np.float(len(data)))
      print()

pd.get_dummies(data['Sex']).head()

pd.concat([data, pd.get_dummies(data['Sex'])], axis=1).head()

pd.get_dummies(data['Sex'], drop_first=True).head()

pd.concat([data, pd.get_dummies(data['Sex'], drop_first=True)], axis=1).head()

pd.get_dummies(data['Embarked']).head()

pd.concat([data, pd.get_dummies(data['Embarked'])], axis=1).head()

pd.get_dummies(data['Embarked'], drop_first=True).head()

pd.concat([data, pd.get_dummies(data['Embarked'], drop_first=True)], axis=1).head()

"""## Doing this in Sklearn"""

from sklearn.preprocessing import LabelEncoder
data_t = data.copy()
obj = LabelEncoder()
data_t['Sex'] = obj.fit_transform(data['Sex'])

data_t.head()

data_save = data.copy()

data.columns

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(handle_unknown = 'ignore')
temp = onehotencoder.fit_transform(data.iloc[:, [2]]).toarray()
#temp_test = onehotencoder.transform(data_test.iloc[:, [1]]).toarray()

data = data.drop(columns = [data.columns[2]])
#data_test = data_test.drop(columns = [data_test.columns[1]])

data = pd.concat([pd.DataFrame(temp), data], axis=1)
#data_test = pd.concat([pd.DataFrame(temp_test), data_test], axis=1)

data.head()

from sklearn.preprocessing import OneHotEncoder
data = data_save.copy()
onehotencoder = OneHotEncoder(drop = 'first')
temp = onehotencoder.fit_transform(data.iloc[:, [2]]).toarray()
#temp_test = onehotencoder.transform(data_test.iloc[:, [1]]).toarray()

data = data.drop(columns = [data.columns[2]])
#data_test = data_test.drop(columns = [data_test.columns[1]])

data = pd.concat([pd.DataFrame(temp), data], axis=1)
#data_test = pd.concat([pd.DataFrame(temp_test), data_test], axis=1)

data.head()

