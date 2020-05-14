
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
            alpha = 0.75, cmap = ListedColormap(('pink', 'cyan', 'lightgreen')))
  plt.xlim(X1.min(), X1.max())
  plt.ylim(X2.min(), X2.max())
  for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue', 'green'))(i), label = j)
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

"""## Get Wine Dataset"""

from sklearn.datasets import load_wine
data = load_wine()

data.keys()

X = data.data
y = data.target

from sklearn.model_selection import KFold
kf = KFold(n_splits=4)
kf.get_n_splits(X)
print(kf)

def logistic_regression(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.linear_model import LogisticRegression
  classifier = LogisticRegression()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "Logistic Regression")

def ridge_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.linear_model import RidgeClassifierCV
  classifier = RidgeClassifierCV()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "RidgeClassifierCV")

def svm_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.svm import SVC
  classifier = SVC()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "SVC")

def mlp_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.neural_network import MLPClassifier
  classifier = MLPClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "MLP")

def linearsvm_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.svm import LinearSVC
  classifier = LinearSVC()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "LinearSVC")

def rf_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.ensemble import RandomForestClassifier
  classifier = RandomForestClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "RandomForestClassifier")

def dt_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.tree import DecisionTreeClassifier
  classifier = DecisionTreeClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "DecisionTreeClassifier")

def gb_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.ensemble import GradientBoostingClassifier
  classifier = GradientBoostingClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "GradientBoostingClassifier")

def sgd_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.linear_model import SGDClassifier
  classifier = SGDClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "SGDClassifier")

def perceptron_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.linear_model import Perceptron
  classifier = Perceptron()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "Perceptron")

def nb_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.naive_bayes import GaussianNB
  classifier = GaussianNB()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "GaussianNB")

def knn_classification(X_train, X_test, y_train, y_test):
  X_train, X_test = preprocess(X_train, X_test)
  from sklearn.neighbors import KNeighborsClassifier
  classifier = KNeighborsClassifier()
  classifier.fit(X_train,y_train)
  y_pred = classifier.predict(X_test)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X_train, y_train, y_test, y_pred, "KNeighborsClassifier")

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  logistic_regression(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  ridge_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  svm_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  mlp_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  linearsvm_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  rf_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  dt_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  gb_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  sgd_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  perceptron_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  knn_classification(X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  nb_classification(X_train, X_test, y_train, y_test)