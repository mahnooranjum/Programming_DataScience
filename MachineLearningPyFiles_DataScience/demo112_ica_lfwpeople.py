# -*- coding: utf-8 -*-
"""Demo112_ICA_LFWPeople.ipynb


# **Tame Your Python**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# Load data
dataset = fetch_lfw_people(min_faces_per_person=100)
 
N, H, W = dataset.images.shape
X = dataset.data
y = dataset.target
target_names = dataset.target_names

print(target_names)

print(dataset.images.shape)
print(dataset.data.shape)
print(dataset.target.shape)

print(H*W)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

from sklearn.decomposition import FastICA
n_components = 80
decomposer = FastICA(n_components=n_components).fit(X_train)

X_train_d = decomposer.transform(X_train)
X_test_d = decomposer.transform(X_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier (hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True)
model.fit(X_train_d, y_train)

y_pred = model.predict(X_test_d)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=target_names))

idx = np.random.randint(0,len(y_pred))
plt.figure()
plt.imshow(X_test[idx].reshape((H,W)), cmap = 'gray')
plt.title("Real = " + str(target_names[y_test[idx]]) + " Predicted = " + str(target_names[y_pred[idx]]))
plt.show()