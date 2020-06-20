# -*- coding: utf-8 -*-
"""Demo196_Scaling_Power.ipynb


## Scaling 

- To avoid biasing the input variables we scale them to be in the same range of values 

## Scalers affected by outliers 

- Min-Max

- Standard

- MaxAbs

- Quantile Transformer gaussian

## Scalers robust to outliers 

- Robust Scaler 

- Quantile Transformer uniform 

- Normalizer
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")

data = data[['Age', 'Fare', 'Survived']]
data.head()

data.describe()

data.isnull().sum()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[['Age', 'Fare']],
                                                    data['Survived'], test_size=0.2)
X_train.shape, X_test.shape

X_train['Age'].fillna(X_train['Age'].median(), inplace=True)
X_test['Age'].fillna(X_train['Age'].median(), inplace=True)

from sklearn.preprocessing import PowerTransformer
obj = PowerTransformer() 
X_train_scaled = obj.fit_transform(X_train) 
X_test_scaled = obj.transform(X_test) 

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train.describe()

X_train_scaled.describe()

sns.set()
sns.jointplot(X_train['Age'],X_train_scaled['Age'], kind='kde')

