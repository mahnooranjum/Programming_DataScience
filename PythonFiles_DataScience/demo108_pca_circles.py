# -*- coding: utf-8 -*-
"""Demo108_PCA_Circles.ipynb

# **Tame Your Python**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

from sklearn.datasets.samples_generator import make_circles
n = 100
# generate 2d classification dataset
X, y = make_circles(n_samples=n)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

datadict = {'X1': X[:,0],'X2' : X[:,1], 'target': y}
data = pd.DataFrame(data=datadict)

X = data.iloc[:,[0, 1]].values
type(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], color='red')
plt.show()

from sklearn.decomposition import PCA
decomposer = PCA(n_components = 2)
decomposer.fit(X)
X_pca = decomposer.transform(X)

pca1 = decomposer.components_[0]
pca2 = decomposer.components_[1]

print(pca1.shape)

print(pca2.shape)

for i, j in zip(X_pca, X):
  plt.scatter(pca1[0] * i[0], pca1[1] * i[0], color = 'blue')
  plt.scatter(pca2[0] * i[1], pca2[1] * i[1], color = 'green')
  plt.scatter(j[0], j[1], color='red')

plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

