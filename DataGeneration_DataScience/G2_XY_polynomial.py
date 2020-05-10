import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

# the equation 2 * x^3 - 40 * x^2 + 9 * x + 24
n = 10000
X = np.random.randn(n)
randomize = np.random.randint(-500,500, size = n)
y = []
for i in range(n):
    y.append((2*X[i]**3)-(40*X[i]**2)+(9*X[i])+24)

for i in range(n):
    y[i] = y[i] + randomize[i]
    
datadict = {'X': X, 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('G2.csv')
plt.scatter(X,y)
plt.show()
