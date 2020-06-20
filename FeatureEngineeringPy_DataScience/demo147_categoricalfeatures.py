# -*- coding: utf-8 -*-
"""Demo147_CategoricalFeatures.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")
data.head()

discrete = []
for i in range(len(data.columns)):
  percentage = 100*len(data.iloc[:,i].dropna().unique())/len(data.iloc[:,i].dropna())
  if percentage < 30:
    discrete.append(i)
    print("{:4} has {:5} unique values with {:4.2f}%".format(i, len(data.iloc[:,i].dropna().unique()), percentage))

data.iloc[:,discrete[0]].unique()

data.iloc[:,discrete[1]].unique()

for i in discrete:
  plt.figure()
  data.iloc[:,i].value_counts().plot.bar(color='blue')
  plt.xticks([])
  plt.xlabel(i)
  plt.show()

data.iloc[:,discrete[0]].value_counts()

data.iloc[:,discrete[1]].value_counts()