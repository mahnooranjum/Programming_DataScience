
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Visualising the results
def plot_model(classifier, X_set, y_set, text):
  from matplotlib.colors import ListedColormap
  X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
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

def plot_gate(X, y, title):
  from matplotlib.colors import ListedColormap
  for i, j in enumerate(np.unique(y)):
    plt.scatter(X[y == j, 0], X[y == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)
  plt.title(title)
  plt.xlabel('X')
  plt.ylabel('y')
  plt.legend()
  plt.show()

"""## Make AND dataset"""

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,0,0,1])
print(X.shape)
print(y.shape)
plot_gate(X,y,"AND")

from sklearn.linear_model import Perceptron
classifier = Perceptron()
classifier.fit(X,y)
y_pred = classifier.predict(X)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X, y, "Perceptron")

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,1])
print(X.shape)
print(y.shape)
plot_gate(X,y,"OR")

from sklearn.linear_model import Perceptron
classifier = Perceptron()
classifier.fit(X,y)
y_pred = classifier.predict(X)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X, y, "Perceptron")

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])
print(X.shape)
print(y.shape)
plot_gate(X,y,"XOR")

from sklearn.linear_model import Perceptron
classifier = Perceptron()
classifier.fit(X,y)
y_pred = classifier.predict(X)
y_pred = np.round(y_pred).flatten()
plot_model(classifier, X, y, "Perceptron")

"""## This perceptron failed ! Let's try the multi-layer perceptron"""

from sklearn.neural_network import MLPClassifier
for i in range(0,10):
  classifier = MLPClassifier(hidden_layer_sizes=(i+1,))
  classifier.fit(X,y)
  y_pred = classifier.predict(X)
  y_pred = np.round(y_pred).flatten()
  plot_model(classifier, X, y, "Multi Layered Perceptron H1-Units = " + str(i+1))

from sklearn.neural_network import MLPClassifier
for i in range(0,10):
  for j in range(0,10):
    classifier = MLPClassifier(hidden_layer_sizes=(i+1,j+1))
    classifier.fit(X,y)
    y_pred = classifier.predict(X)
    y_pred = np.round(y_pred).flatten()
    plot_model(classifier, X, y, "Multi Layered Perceptron H1-Units = " + str(i+1) + " H2-Units = " + str(j+1))