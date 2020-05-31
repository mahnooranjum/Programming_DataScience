# -*- coding: utf-8 -*-
"""Demo94_ClusteringEvaluation_HC_SingleLink_VisualAid.ipynb



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

def evaluation_labels(y, y_pred):
  from sklearn import metrics
  adj_rand = metrics.adjusted_rand_score(y, y_pred)
  print("Adjusted Rand Score = " + str(adj_rand))

  adj_mi = metrics.adjusted_mutual_info_score(y, y_pred)
  print("Adjusted Mutual Information = " + str(adj_mi))

  h = metrics.homogeneity_score(y, y_pred)
  print("Homogeneity = " + str(h))

  v = metrics.v_measure_score(y, y_pred)
  print("V-measure = " + str(v))

  c = metrics.completeness_score(y, y_pred)
  print("Completeness = " + str(c))

  f = metrics.fowlkes_mallows_score(y, y_pred)
  print("Fowlkes-Mallows = " + str(f))

  return

def evaluation(X, y_pred):
  from sklearn import metrics  
  s = metrics.silhouette_score(X, y_pred, metric='euclidean')
  print("Silhouette Coefficient = " + str(s))

  c = metrics.calinski_harabasz_score(X, y_pred)
  print("Calinski-Harabasz  = " + str(c))

  d = metrics.davies_bouldin_score(X, y_pred)
  print("Davies-Bouldin  = " + str(d))

  return

def visual(c, X, y):
  from sklearn.cluster import AgglomerativeClustering
  cluster_object = AgglomerativeClustering(n_clusters = c, affinity = 'euclidean', linkage = 'single')
  y_pred = cluster_object.fit_predict(X)
  colors = ['red', 'green', 'blue', 'cyan', 'black', 'yellow', 'magenta', 'brown', 'orange', 'silver', 'goldenrod', 'olive', 'dodgerblue', 'turqoise']
  clusters = np.unique(y_pred)
  print("Cluster Labels")
  print(clusters)
  print("Evaluation")
  evaluation_labels(y, y_pred)
  evaluation(X, y_pred)
  for cluster in clusters:
    row_idx = np.where(y== cluster)
    plt.scatter(X[row_idx, 0], X[row_idx, 1])
  plt.title('Dataset')
  plt.xlabel('X1')
  plt.ylabel('X2')
  plt.legend()
  plt.show()

  plt.figure()
  for cluster in clusters:
    row_idx = np.where(y_pred == cluster)
    plt.scatter(X[row_idx, 0], X[row_idx, 1])
  plt.title('Cluster')
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
y_random = np.random.randint(0,3,size = [n])
transformation = [[0.80834549, -0.83667341], [-0.20887718, 0.85253229]]
X_aniso = np.dot(X_blobs, transformation)
y_aniso = y_blobs

plot_dataset(X_moons)

visual(2, X_moons, y_moons)

plot_dataset(X_blobs)

visual(3, X_blobs, y_blobs)

plot_dataset(X_circles)

visual(2, X_circles, y_circles)

plot_dataset(X_random)

visual(3, X_random, y_random)

plot_dataset(X_aniso)

visual(3, X_aniso, y_aniso)

