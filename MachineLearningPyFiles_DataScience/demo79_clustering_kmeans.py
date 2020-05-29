# -*- coding: utf-8 -*-
"""Demo79_Clustering_KMeans.ipynb

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

!wget http://cs.joensuu.fi/sipu/datasets/s1.txt
!ls

!head s4.txt

X = pd.read_csv('s1.txt', delimiter='    ', header=None)
print(X.shape)

X.head()
print(X.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

plt.scatter(X[:,0], X[:,1])
plt.show()

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.scatter(range(1, 20), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
plt.clf()

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size = 0.1)

# Fitting K-Means to the dataset
cluster_object = KMeans(n_clusters = 8, init = 'k-means++')
y_pred = cluster_object.fit_predict(X_train)

clusters = np.unique(y_pred)
print(clusters)

# Visualising the clusters
colors = ['red', 'green', 'blue', 'cyan', 'black', 'yellow', 'magenta', 'brown', 'orange']
for i in np.unique(y_pred):
  plt.scatter(X_train[y_pred == i, 0], X_train[y_pred == i, 1], s = 10, c = colors[i], label = 'Cluster' + str(i))
  #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 10, c = 'yellow', label = 'Centroids')
plt.title('Clusters')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

