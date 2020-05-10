
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

# Regression Dataset
n = 10000
X, y = make_regression(n_samples=n, n_features=1, noise=70)
X = X.reshape(n)
datadict = {'X': X, 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('G1.csv')
plt.scatter(X,y)
plt.show()