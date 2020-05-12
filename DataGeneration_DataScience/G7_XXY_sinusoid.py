import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import pandas as pd
import numpy as np

n = 10000
std = 6
X = np.random.random((n,2)) * (2*std) - (std)
y = (np.cos(X[:,0]) + np.sin(X[:,1]))

# Visualising the dataset
plt.scatter(X[:,0], y, color = 'red')
plt.title('X1 Plot')
plt.xlabel('X1')
plt.ylabel('y')
plt.show()

# Visualising the dataset
plt.scatter(X[:,1], y, color = 'red')
plt.title('X2 Plot')
plt.xlabel('X2')
plt.ylabel('y')
plt.show()

# Visualising the dataset with the target variable
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.scatter(X[:,0], X[:,1], y)
plt.title('Scatter Plot')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

datadict = {'X1': X[:,0],'X2' : X[:,1], 'target': y}
df = pd.DataFrame(data=datadict)
df.to_csv('G7.csv')