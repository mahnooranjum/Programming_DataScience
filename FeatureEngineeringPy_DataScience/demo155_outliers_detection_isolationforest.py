# -*- coding: utf-8 -*-
"""Demo155_Outliers_Detection_IsolationForest.ipynb

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

# Outliers according to the quantiles + 1.5 IQR 
sns.catplot(x="Survived", y="Fare", kind="box", data=data)
sns.despine(left=False, right=False, top=False)

data['Fare'].describe()

# Get outliers
IQR = data['Fare'].quantile(0.75) - data['Fare'].quantile(0.25)
ub = data['Fare'].quantile(0.75) + (IQR * 3)
lb = data['Fare'].quantile(0.25) - (IQR * 3)
data[(data['Fare']>ub) | (data['Fare']<lb)].groupby('Survived')['Fare'].count()

# First let's plot the histogram to get an idea of the distribution
fig = data.Age.hist(bins=50, color='green')
fig.set_title('Distribution')
fig.set_xlabel('X')
fig.set_ylabel('#')

# Get outliers
IQR = data['Age'].quantile(0.75) - data['Age'].quantile(0.25)
ub = data['Age'].mean() + data['Age'].std()
lb = data['Age'].mean() - data['Age'].std()
data[(data['Age']>ub) | (data['Age']<lb)].groupby('Survived')['Age'].count()

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = data.drop(['Date'], axis=1)

data.columns

data[['Fare','Age']].isnull().mean()

X_train, X_test, y_train, y_test = train_test_split(data[['Age', 'Fare']].fillna(0),
                                                    data['Survived'],
                                                    test_size=0.2)
X_train.shape, X_test.shape

# We will cap the values of outliers 
data_processed = data.copy()

_temp = np.ceil(data['Age'].mean() + data['Age'].std())
data_processed.loc[data_processed.Age >= _temp, 'Age'] = _temp

IQR = data['Fare'].quantile(0.75) - data['Fare'].quantile(0.25)
_temp = np.ceil(data['Fare'].quantile(0.75) + (IQR * 3))
data_processed.loc[data_processed.Fare > _temp, 'Fare'] = _temp

X_train_processed, X_test_processed, y_train_processed, y_test_processed = train_test_split(
    data_processed[['Age', 'Fare']].fillna(0),
    data_processed['Survived'],
    test_size=0.2)

from sklearn.ensemble import IsolationForest

df_outliers = data.copy()
df_outliers = df_outliers.fillna(0)

column_name = 'Fare'
obj = IsolationForest().fit(df_outliers[[column_name]])
_temp = obj.predict(df_outliers[[column_name]])
print(np.unique(_temp, return_counts=True))
central = df_outliers[_temp==1][column_name].mean()
max_val = df_outliers[_temp==1][column_name].max()
min_val = df_outliers[_temp==1][column_name].min()
df_outliers.loc[_temp==-1,[column_name]] = df_outliers.loc[_temp==-1,[column_name]].apply(lambda x: [max_val if y > central else y for y in x])
df_outliers.loc[_temp==-1,[column_name]] = df_outliers.loc[_temp==-1,[column_name]].apply(lambda x: [min_val if y < central else y for y in x])
print(data.shape)
print(df_outliers.shape)

column_name = 'Age'
obj = IsolationForest().fit(df_outliers[[column_name]])
_temp = obj.predict(df_outliers[[column_name]])
print(np.unique(_temp, return_counts=True))
central = df_outliers[_temp==1][column_name].mean()
max_val = df_outliers[_temp==1][column_name].max()
min_val = df_outliers[_temp==1][column_name].min()
df_outliers.loc[_temp==-1,[column_name]] = df_outliers.loc[_temp==-1,[column_name]].apply(lambda x: [max_val if y > central else y for y in x])
df_outliers.loc[_temp==-1,[column_name]] = df_outliers.loc[_temp==-1,[column_name]].apply(lambda x: [min_val if y < central else y for y in x])
print(data.shape)
print(df_outliers.shape)

X_train_outliers, X_test_outliers, y_train_outliers, y_test_outliers = train_test_split(
    df_outliers[['Age', 'Fare']],
    df_outliers['Survived'],
    test_size=0.2)

print(X_train.shape)
print(X_train_processed.shape)
print(X_train_outliers.shape)
print(X_test.shape)
print(X_test_processed.shape)
print(X_test_outliers.shape)

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))

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

classifier.fit(X_train_outliers,y_train_outliers)
y_pred_outliers = classifier.predict(X_test_outliers)
y_pred_outliers = np.round(y_pred_outliers).flatten()
print(accuracy_score(y_test_outliers, y_pred_outliers))