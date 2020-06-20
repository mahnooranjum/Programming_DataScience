# -*- coding: utf-8 -*-
"""Demo160_Distributions.ipynb


## Variable distributions and their effects on Models

Reference 
[https://www.statisticssolutions.com/homoscedasticity/]

### Linear Regression Assumptions

- Linear relationship with the outcome Y
- Homoscedasticity
- Normality
- No Multicollinearity 

## Linear Assumption

- The X variable is linearly related to the dataset 
- Pearson correlation coefficient can determine the linearity magnitude  between variables 

## Normality Assumption

- The variable X follows a normal or gaussian distribution

## Homoscedasticity Assumption

- Homogeneity of variance

- Homoscedasticity describes a situation in which the error term (that is, the “noise” or random disturbance in the relationship between the independent variables and the dependent variable) is the same across all values of the independent variables


### Unaffected models

- Neural Networks
- Support Vector Machines
- Trees
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
sns.set()
import pandas as pd

import scipy.stats as stats
mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
alpha = 0
y = stats.skewnorm.pdf(x, alpha)
sns.lineplot(x,y)

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
alpha = 2
y = stats.skewnorm.pdf(x, alpha)
sns.lineplot(x,y)

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
alpha = -2
y = stats.skewnorm.pdf(x, alpha)
sns.lineplot(x,y)

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
fig, ax = plt.subplots()
y = stats.skewnorm.pdf(x, 0)
sns.lineplot(x,y, ax=ax)
y = stats.skewnorm.pdf(x, 3)
sns.lineplot(x,y, ax=ax)
y = stats.skewnorm.pdf(x, -3)
sns.lineplot(x,y, ax=ax)
ax.legend(["Normal", "Negative Skew", "Positive Skew"])

"""## Let's explore the distributions of the variables"""

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train_date.csv")

data.head()

"""#### Age"""

sns.set_color_codes()
sns.distplot(data['Age'], color='r')

"""Make a QQ-plot:

- Quantiles on vertical axis
- Quantiles of distributin on horizontal axis 
- The points on the graph will form a 45 degree line if gaussian
"""

fig, ax = plt.subplots()
stats.probplot(data['Age'].dropna(), dist="norm", plot=ax)
ax.set_title("QQPlot")

sns.set_color_codes()
sns.distplot(data['Age']**(1/1.5), color='g')

fig, ax = plt.subplots()
stats.probplot(data['Age'].dropna()**(1/1.5), dist="norm", plot=ax)
ax.set_title("QQPlot")

"""#### Fare"""

sns.set_color_codes()
sns.distplot(data['Fare'], color='r')

"""Make a QQ-plot:

- Quantiles on vertical axis
- Quantiles of distributin on horizontal axis 
- The points on the graph will form a 45 degree line if gaussian
"""

fig, ax = plt.subplots()
stats.probplot(data['Fare'].dropna(), dist="norm", plot=ax)
ax.set_title("QQPlot")

sns.set_color_codes()
sns.distplot(data['Fare']**(1/3), color='g')

fig, ax = plt.subplots()
stats.probplot(data['Fare'].dropna()**(1/1.5), dist="norm", plot=ax)
ax.set_title("QQPlot")