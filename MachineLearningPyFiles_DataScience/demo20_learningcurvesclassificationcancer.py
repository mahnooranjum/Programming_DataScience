

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluate(y_test, y_pred):
  from sklearn.metrics import accuracy_score
  print("===== Accuracy Score =====")
  print(accuracy_score(y_test, y_pred))

  from sklearn.metrics import classification_report
  print("===== Accuracy Score =====")
  class_report = classification_report(y_test, y_pred)
  print(class_report)
  return

# Visualising the results
def plot_model(classifier, X_set, y_set, y_test, y_pred, text):
  from matplotlib.colors import ListedColormap
  X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
  plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('pink', 'cyan')))
  plt.xlim(X1.min(), X1.max())
  plt.ylim(X2.min(), X2.max())
  for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)
  plt.title(text)
  plt.xlabel('X')
  plt.ylabel('y')
  plt.legend()
  plt.show()

def preprocess(X_train, X_test):
  from sklearn.decomposition import PCA
  pca = PCA(n_components = 2)
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)
  # Feature Scaling
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  return X_train, X_test

"""## Get Breast Cancer Dataset"""

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

def draw_learning_curves(X, y, classifier):
  from sklearn.model_selection import learning_curve
  train_sizes, train_scores, test_scores = learning_curve(classifier, X, y)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  plt.grid()
  plt.title("Learning Curves")
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  plt.plot(train_scores_mean, 'o-', color="b", label="Training Score")
  plt.plot(test_scores_mean, 'o-', color="r", label="Cross Validation Score")
  plt.legend()
  plt.show()

data.keys()

X = data.data
y = data.target

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

def logistic_regression(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.linear_model import LogisticRegression
  classifier = LogisticRegression()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "Logistic Regression")
  draw_learning_curves(X_train, y_train, classifier)

logistic_regression(X_train, X_test, y_train, y_test)

def ridge_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.linear_model import RidgeClassifierCV
  classifier = RidgeClassifierCV()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "RidgeClassifierCV")
  draw_learning_curves(X_train, y_train, classifier)

ridge_classification(X_train, X_test, y_train, y_test)

def svm_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.svm import SVC
  classifier = SVC()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "SVC")
  draw_learning_curves(X_train, y_train, classifier)

svm_classification(X_train, X_test, y_train, y_test)

def mlp_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.neural_network import MLPClassifier
  classifier = MLPClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "MLP")
  draw_learning_curves(X_train, y_train, classifier)

mlp_classification(X_train, X_test, y_train, y_test)

def linearsvm_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.svm import LinearSVC
  classifier = LinearSVC()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "LinearSVC")
  draw_learning_curves(X_train, y_train, classifier)

linearsvm_classification(X_train, X_test, y_train, y_test)

def rf_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.ensemble import RandomForestClassifier
  classifier = RandomForestClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "RandomForestClassifier")
  draw_learning_curves(X_train, y_train, classifier)

rf_classification(X_train, X_test, y_train, y_test)

def dt_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.tree import DecisionTreeClassifier
  classifier = DecisionTreeClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "DecisionTreeClassifier")
  draw_learning_curves(X_train, y_train, classifier)

dt_classification(X_train, X_test, y_train, y_test)

def gb_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.ensemble import GradientBoostingClassifier
  classifier = GradientBoostingClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "GradientBoostingClassifier")
  draw_learning_curves(X_train, y_train, classifier)

gb_classification(X_train, X_test, y_train, y_test)

def sgd_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.linear_model import SGDClassifier
  classifier = SGDClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "SGDClassifier")
  draw_learning_curves(X_train, y_train, classifier)

sgd_classification(X_train, X_test, y_train, y_test)

def perceptron_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.linear_model import Perceptron
  classifier = Perceptron()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "Perceptron")
  draw_learning_curves(X_train, y_train, classifier)

perceptron_classification(X_train, X_test, y_train, y_test)

def nb_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.naive_bayes import GaussianNB
  classifier = GaussianNB()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "GaussianNB")
  draw_learning_curves(X_train, y_train, classifier)

nb_classification(X_train, X_test, y_train, y_test)

def knn_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.neighbors import KNeighborsClassifier
  classifier = KNeighborsClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "KNeighborsClassifier")
  draw_learning_curves(X_train, y_train, classifier)

knn_classification(X_train, X_test, y_train, y_test)