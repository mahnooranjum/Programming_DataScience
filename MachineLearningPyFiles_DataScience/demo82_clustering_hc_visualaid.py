# -*- coding: utf-8 -*-
"""Demo82_Clustering_HC_VisualAid.ipynb


# **Tame Your Python**

Let's see how we can classify emails based on their contents

`Leggo`
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

def plot_dataset(X):
  plt.scatter(X[:,0], X[:,1])
  plt.show()

def visual(c, X):
  from sklearn.cluster import AgglomerativeClustering
  cluster_object = AgglomerativeClustering(n_clusters = c, affinity = 'euclidean', linkage = 'ward')
  y_pred = cluster_object.fit_predict(X)
  colors = ['red', 'green', 'blue', 'cyan', 'black', 'yellow', 'magenta', 'brown', 'orange', 'silver', 'goldenrod', 'olive', 'dodgerblue', 'turqoise']
  clusters = np.unique(y_pred)
  print(clusters)
  for i in np.unique(y_pred):
    plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], s = 10, c = colors[i], label = 'Cluster' + str(i))
  plt.title('Clusters')
  plt.xlabel('X1')
  plt.ylabel('X2')
  plt.legend()
  plt.show()

"""## Get the dataset"""

n = 1000
from sklearn.datasets import make_moons, make_blobs, make_circles, make_s_curve
X_moons, y_moons = make_moons(n_samples = n, noise=0.1)
X_blobs, y_blobs = make_blobs(n_samples = n, n_features = 2)
X_circles, y_circles = make_circles(n_samples=n, noise=0.1, factor = 0.5)
X_scurve, y_scurve = make_s_curve(n_samples=n, noise = 0.1)
X_random = np.random.random([n, 2])
transformation = [[0.80834549, -0.83667341], [-0.20887718, 0.85253229]]
X_aniso = np.dot(X_blobs, transformation)

plot_dataset(X_moons)

visual(2, X_moons)

plot_dataset(X_blobs)

visual(3, X_blobs)

plot_dataset(X_circles)

visual(2, X_circles)

plot_dataset(X_scurve[:,0:2])

visual(3, X_scurve)

plot_dataset(X_random)

visual(3, X_random)

plot_dataset(X_aniso)

visual(3, X_aniso)

