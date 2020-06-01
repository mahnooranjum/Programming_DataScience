# -*- coding: utf-8 -*-
"""Demo104_FeatureScaling_Classification.ipynb

# **Spit some [tensor] flow**

Let's see some classifiers in action

`Leggo`
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Visualising the results
def plot_model(classifier, X_set, y_set, y_test, y_pred, text):
  from sklearn.metrics import accuracy_score
  print("Accuracy Score :")
  print(accuracy_score(y_test, y_pred))
  return

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

data.keys()

X = data.data
y = data.target

data.target_names

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "Logistic Regression")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "Logistic Regression Feature Scaled")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import RidgeClassifierCV
classifier = RidgeClassifierCV()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "RidgeClassifierCV")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import RidgeClassifierCV
classifier = RidgeClassifierCV()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "RidgeClassifierCV")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "SVC")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "SVC")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "SVC")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "SVC")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "LinearSVC")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "LinearSVC")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "RandomForestClassifier")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "RandomForestClassifier")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "DecisionTreeClassifier")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "DecisionTreeClassifier")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "GradientBoostingClassifier")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "GradientBoostingClassifier")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "SGDClassifier")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "SGDClassifier")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.linear_model import Perceptron
classifier = GradientBoostingClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "Perceptron")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import Perceptron
classifier = GradientBoostingClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "Perceptron")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "GaussianNB")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "GaussianNB")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "KNeighborsClassifier")

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X_train, y_train, y_test, y_pred, "KNeighborsClassifier")