
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression


#MULTIPLE LINEAR REGRESSION
n = 100000
X = np.random.randn(n)
Z = np.random.randn(n)
randomize = np.random.randint(-50,50, size = n)
y = []
for i in range(n):
    y.append((2*X[i])-(40*Z[i])+24)

for i in range(n):
    y[i] = y[i] + randomize[i]
    
datadict = {'X': X, 'Z': Z, 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('G3.csv')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Z, y, zdir='z', color="green", s=20, c=None, depthshade=True)
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y") 
