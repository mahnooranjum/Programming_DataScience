from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


###### CIRCLES #########
# generate 2d classification dataset
n = 10000
X, y = make_circles(n_samples=n, noise=0.05)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()
datadict = {'X1': X[:,0],'X2' : X[:,1], 'target': y}
df = pd.DataFrame(data=datadict)
df.to_csv('G6.csv')