# -*- coding: utf-8 -*-
"""Demo103_FeatureScaling_MinMax.ipynb
# **Tame Your Python**

Let's see how we can classify emails based on their contents

`Leggo`
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

"""## Get the dataset"""

def plot_dataset(X):
  plt.scatter(X[:,0], X[:,1], color = 'red')
  plt.show()

n = 1000
from sklearn.datasets import make_moons, make_blobs, make_circles, make_s_curve
X_moons, y_moons = make_moons(n_samples = n, noise=0.1)

X_moons = X_moons + 10

plot_dataset(X_moons)

def minmax(X):
  X_scaled = np.zeros(X.shape)
  for i in range(X.shape[1]):
    x_min = np.min(X[i])
    x_max = np.max(X[i])
    for j in range(X.shape[0]):
      X_scaled[j, i] = (X[j,i] - x_min)/(x_max - x_min)
  
  return X_scaled

X_moons_scaled = minmax(X_moons)

plot_dataset(X_moons_scaled)