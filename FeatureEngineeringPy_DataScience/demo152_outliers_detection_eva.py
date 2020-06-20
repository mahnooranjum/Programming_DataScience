# -*- coding: utf-8 -*-
"""Demo152_Outliers_Detection_EVA.ipynb


## Outliers

An outlier is an observation that lies outside the overall pattern of a distribution __[Moore and McCabe, 1999]__.

- Outliers can either be treated special completely ignored 

- E.g., Fraudulant transactions are outliers, but since we want to avoid them, they must be paid special attention 

- If we think that the outliers are errors, we should remove them 


## Which of the ML models care about Outliers?

Affected models: 

- AdaBoost 
- Linear models
- Linear regression
- Neural Networks (if the number is high)
- Logistic regression 
- KMeans
- Heirarchical Clustering 
- PCA

Unaffected models:

- Decision trees
- Naive bayes
- SVMs
- Random forest 
- Gradient boosted trees 
- K-Nearest Neighbors


### Identification

- Extreme Value Analysis
  - IQR = 75th quantile - 25th quantile
  - Upper boundary = 75th quantile + (IQR * 1.5)
  - Lower boundary = 25th quantile - (IQR * 1.5)
  - Upper Extreme boundary = 75th quantile + (IQR * 3)
  - Lower Extreme boundary = 25th quantile - (IQR * 3)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train_date.csv")

data.head()

data['Age'].min()

data['Age'].max()

sns.distplot(data['Age'].fillna(100))

fig = data['Fare'].hist(bins=50, color='blue')
fig.set_title('Distribution')
fig.set_xlabel('X variable')
fig.set_ylabel('#')

# Outliers according to the quantiles + 1.5 IQR 
sns.catplot(x="Survived", y="Fare", kind="box", data=data)
sns.despine(left=False, right=False, top=False)

sns.set_style("whitegrid")
sns.catplot(x="Survived", y="Fare", kind="swarm", data=data)
sns.despine(left=False, right=False, top=False)

sns.set_style("whitegrid")
sns.catplot(x="Survived", y="Fare",  data=data);
sns.despine(left=False, right=False, top=False)

fig = sns.catplot(x="Survived", y="Fare",
            kind="point", data=data)

sns.set_context("talk")
sns.set_palette("bright")
sns.catplot(x="Survived", y="Fare", jitter = False, data=data);

data['Fare'].describe()

# Identify outliers according to IQR
IQR = data['Fare'].quantile(0.75) - data['Fare'].quantile(0.25)

lower_bnd = data['Fare'].quantile(0.25) - (IQR * 1.5)
upper_bnd = data['Fare'].quantile(0.75) + (IQR * 1.5)

upper_bnd, lower_bnd, IQR

# Identify outliers according to extreme IQR
IQR = data['Fare'].quantile(0.75) - data['Fare'].quantile(0.25)

ex_lower_bnd = data['Fare'].quantile(0.25) - (IQR * 3)
ex_upper_bnd = data['Fare'].quantile(0.75) + (IQR * 3)

ex_upper_bnd, ex_lower_bnd, IQR

print('total rows: {}'.format(data.shape[0]))
print('rows above IQR boundary: {}'.format(
    data[data['Fare'] > upper_bnd].shape[0]))
print('rows above extreme IQR boundary: {}'.format(
    data[data['Fare'] > ex_upper_bnd].shape[0]))

print('rows above IQR boundary: {}'.format(
    100* data[data['Fare'] > upper_bnd].shape[0]/data.shape[0]))
print('rows above extreme IQR boundary: {}'.format(
    100* data[data['Fare'] > ex_upper_bnd].shape[0]/data.shape[0]))

# Get outliers
outliers = data[data['Fare']>ex_upper_bnd]
outliers.groupby('Survived')['Fare'].count()

outliers = data[data['Fare']>ex_upper_bnd]
outliers.groupby('Cabin')['Fare'].count()

outliers = data[data['Fare']>ex_upper_bnd]
outliers.groupby('Ticket')['Fare'].count()

"""Age exploration"""

# First let's plot the histogram to get an idea of the distribution
fig = data.Age.hist(bins=50, color='green')
fig.set_title('Distribution')
fig.set_xlabel('X')
fig.set_ylabel('#')

# Outliers according to the quantiles + 1.5 IQR 
sns.catplot(x="Survived", y="Age", kind="box", data=data)
sns.despine(left=False, right=False, top=False)

data['Age'].describe()

upper_bnd = data['Age'].mean() + 1.5* data['Age'].std()
lower_bnd = data['Age'].mean() - 1.5* data['Age'].std()

upper_bnd, lower_bnd

ex_upper_bnd = data['Age'].mean() + 3* data['Age'].std()
ex_lower_bnd = data['Age'].mean() - 3* data['Age'].std()

ex_upper_bnd, ex_lower_bnd

print('total rows: {}'.format(data.shape[0]))
print('rows above IQR boundary: {}'.format(
    data[data['Age'] > upper_bnd].shape[0]))
print('rows above extreme IQR boundary: {}'.format(
    data[data['Age'] > ex_upper_bnd].shape[0]))

print('total rows: {}'.format(data.shape[0]))
print('rows below IQR boundary: {}'.format(
    data[data['Age'] < lower_bnd].shape[0]))
print('rows below extreme IQR boundary: {}'.format(
    data[data['Age'] < ex_lower_bnd].shape[0]))

print('rows above IQR boundary: {}'.format(
    100* data[data['Age'] > upper_bnd].shape[0]/data.shape[0]))
print('rows above extreme IQR boundary: {}'.format(
    100* data[data['Age'] > ex_upper_bnd].shape[0]/data.shape[0]))

print('rows lower IQR boundary: {}'.format(
    100* data[data['Age'] < lower_bnd].shape[0]/data.shape[0]))
print('rows lower extreme IQR boundary: {}'.format(
    100* data[data['Age'] < ex_lower_bnd].shape[0]/data.shape[0]))

"""## Get Out, LIARS!"""

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data.columns

data = data.drop(['Date'], axis=1)

data.columns

data[['Fare','Age']].isnull().mean()

X_train, X_test, y_train, y_test = train_test_split(data[['Age', 'Fare']].fillna(0),
                                                    data['Survived'],
                                                    test_size=0.2)
X_train.shape, X_test.shape

# We will cap the values of outliers 
data_processed = data.copy()

_temp = np.ceil(data['Age'].mean() + 3* data['Age'].std())
data_processed.loc[data_processed.Age >= _temp, 'Age'] = _temp

_temp = np.ceil(data['Fare'].quantile(0.75) + (IQR * 3))

data_processed.loc[data_processed.Fare > _temp, 'Fare'] = _temp

X_train_processed, X_test_processed, y_train_processed, y_test_processed = train_test_split(
    data_processed[['Age', 'Fare']].fillna(0),
    data_processed['Survived'],
    test_size=0.2)

print(X_train.shape)
print(X_train_processed.shape)
print(X_test.shape)
print(X_test_processed.shape)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.linear_model import RidgeClassifierCV
classifier = RidgeClassifierCV()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.linear_model import RidgeClassifierCV
classifier = RidgeClassifierCV()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.linear_model import Perceptron
classifier = Perceptron()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_processed,y_train_processed)
y_pred_processed = classifier.predict(X_test_processed)
y_pred_processed = np.round(y_pred_processed).flatten()
print(accuracy_score(y_test_processed, y_pred_processed))