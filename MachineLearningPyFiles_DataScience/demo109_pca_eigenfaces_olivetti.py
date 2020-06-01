# -*- coding: utf-8 -*-
"""Demo109_PCA_Eigenfaces_Olivetti.ipynb


# **Tame Your Python**

Reference: https://pythonmachinelearning.pro/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Load data
dataset = fetch_olivetti_faces()
 
N, H, W = dataset.images.shape
X = dataset.data
y = dataset.target

print(np.unique(y))

print(dataset.images.shape)
print(dataset.data.shape)
print(dataset.target.shape)

print(H*W)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

from sklearn.decomposition import PCA
n_components = 80
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier (hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True)
model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

idx = np.random.randint(0,len(y_pred))
plt.figure()
plt.imshow(X_test[idx].reshape((H,W)), cmap = 'gray')
plt.title("Real = " + str(y_test[idx]) + " Predicted = " + str(y_pred[idx]))
plt.show()

idx = 1
plt.figure()
plt.imshow(pca.components_[idx].reshape((H,W)), cmap = 'gray')
plt.show()

idx = 79
plt.figure()
plt.imshow(pca.components_[idx].reshape((H,W)), cmap = 'gray')
plt.show()