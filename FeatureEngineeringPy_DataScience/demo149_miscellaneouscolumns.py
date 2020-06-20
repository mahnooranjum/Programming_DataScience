# -*- coding: utf-8 -*-
"""Demo149_MiscellaneousColumns.ipynb

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train_date.csv")
data.head()

data['Age'].unique()

data['Age'].isnull().sum()

data['Age'] = data['Age'].replace(np.nan, 'M')

data['Age'].isnull().sum()

data['Age'].unique()

data.shape

data['Age'].value_counts().plot(kind='barh', figsize=(12,20))