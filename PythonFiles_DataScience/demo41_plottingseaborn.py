# -*- coding: utf-8 -*-
"""Demo41_PlottingSeaborn.ipynb

# SEABORN FOR VISUALIZATION

Welcome to the seaborn tutorial. Seaborn gives us more "types" of plots vs matplotlib. Seaborn has built in themes to create the most attractive plot. 

As data scientists, visualization is not the end of our journey, presentation is. Presenting our findings to our peers or supervisors in a readable format is the key difference between a hobbyist and a professional. I promised an in-depth zero-to-all course of Data Science and it can not be complete without a thorough knowledge of Seaborn. 

- Python Basics
- Object Oriented Python
- Python for Data Science
- NumPy
- Pandas
- **Plotting** 
    - Matplotlib
    - **Seaborn**

    
Let's get visualizing!

## A- DISTRIBUTION PLOTS
"""

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
tips = sns.load_dataset('tips')
tips.head()

sns.distplot(tips['tip'])

sns.distplot(tips['total_bill'], kde=False, bins = 50)

sns.jointplot(x='total_bill', y = 'tip', data = tips, kind='scatter')

sns.jointplot(x='total_bill', y = 'tip', data = tips, kind='hex')

sns.jointplot(x='total_bill', y = 'tip', data = tips, kind='reg')

sns.jointplot(x='total_bill', y = 'tip', data = tips, kind='kde')

sns.pairplot(tips) # We get a pair plot for every numerical column

sns.pairplot(tips, hue='sex')

sns.rugplot(tips['total_bill'])
sns.distplot(tips['total_bill'])

sns.kdeplot(tips['total_bill'])

"""## B- CATEGORICAL PLOTS"""

sns.factorplot(x='day', y='tip', data=tips, kind='bar')

sns.factorplot(x='day', y='tip', data=tips, kind='violin')

sns.factorplot(x='day', y='tip', data=tips, kind='swarm')

sns.factorplot(x='day', y='tip', data=tips, kind='strip')

sns.barplot(x='sex', y='tip', data=tips)

import numpy as np
sns.barplot(x='sex', y='tip', data=tips, estimator = np.std)

import numpy as np
sns.barplot(x='sex', y='tip', data=tips, estimator = np.median)

sns.countplot(data=tips, x='sex')

sns.boxplot(x='day',y='tip', data=tips)

sns.boxplot(x='day',y='tip', data=tips, hue='sex')

sns.violinplot(x='day',y='tip', data=tips)

sns.violinplot(x='day',y='tip', data=tips, hue='smoker', split=True)

sns.stripplot(x='day',y='tip', data=tips)

sns.stripplot(x='day',y='tip', data=tips, hue = 'sex', dodge = True)

sns.stripplot(x='day',y='tip', data=tips, jitter = True)

sns.swarmplot(x='day', y='tip', data=tips, color='black')
sns.violinplot(x='day', y='tip', data=tips)

"""## B- MATRIX PLOTS"""

planets = sns.load_dataset('planets')
planets

corrt = tips.corr()
corrp = planets.corr()

sns.heatmap(corrt)

sns.heatmap(corrp, annot=True, cmap = "coolwarm")

flights = sns.load_dataset('flights')
p = flights.pivot(index = 'year', columns = 'month', values = 'passengers' )
flights.head()

sns.heatmap(p, cmap='coolwarm')

sns.clustermap(p, cmap = 'coolwarm')

sns.clustermap(p, cmap = 'coolwarm', standard_scale=1)

"""## REGRESSION PLOTS"""

sns.lmplot(x='tip', y='total_bill',data=tips, hue='sex', markers=['o','v'])

sns.lmplot(x='tip', y='total_bill',data=tips, col='sex', row='smoker')

sns.lmplot(x='tip', y='total_bill',data=tips, col='sex', aspect =1, size= 5) # ASPECT * SIZE = WIDTH

"""## REGRESSION PLOTS"""

iris = sns.load_dataset('iris')
iris.head()

sns.pairplot(iris)

grids = sns.PairGrid(iris)

grids = sns.PairGrid(iris)
grids.map_diag(sns.distplot)
grids.map_upper(plt.scatter)
grids.map_lower(sns.kdeplot)

f = sns.FacetGrid(data=tips, col='day', row='sex')
f.map(sns.distplot, 'total_bill')

f = sns.FacetGrid(data=tips, col='day', row='sex')
f.map(plt.scatter, 'total_bill', 'tip')

"""### WELCOME TO THE END OF THE TUTORIAL
You made it! As always, Hope you enjoyed taking this tutorial as much as I enjoyed making it. From the next tutorial, we will be starting data visualizations with Plotly. 
Until next time folks, Happy visualizing.

---------------------------------------------------------------------------------
Copyrights © 2018, All Rights Reserved.
- Author: Mahnoor Anjum.
- Course: The Complete Hands-On Machine Learning Course
- Date Created: 2018-07-06
- Date Modified: -
"""