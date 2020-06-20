# -*- coding: utf-8 -*-
"""Demo154_Categories.ipynb

## Categories

- Labels 

- The number of labels in the dataset are different 

- __high cardinality__ refers to uniqueness of data values 

- The lower the cardinality, the more duplicated elements in a column

-  A column with the lowest possible cardinality would have the same value for every row

- Highly cardinal variables dominate tree based algorithms

- Labels may only be present in the training data set, but not in the test data set

- Labels may appear in the test set that were not present in the training set


__Tree methods are biased towards variables with many labels__
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train_date.csv")

cat_cols = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

for i in cat_cols:
  print('Number of categories in the variable {}: {}'.format(i,len(data[i].unique())))

print('Total rows: {}'.format(len(data)))

data['Embarked'].unique()

data['Cabin'].unique()

data['Cabin_processed'] = data['Cabin'].astype(str).str[0]
data[['Cabin', 'Cabin_processed']].head()

cat_cols = ['Cabin_processed', 'Cabin']

for i in cat_cols:
  print('Number of categories in the variable {}: {}'.format(i,len(data[i].unique())))

from sklearn.model_selection import train_test_split
use_cols = ['Cabin', 'Sex', 'Cabin_processed']
X_train, X_test, y_train, y_test = train_test_split(data[use_cols], 
                                                    data['Survived'],  
                                                    test_size=0.2)

X_train.shape, X_test.shape

# Labels in training set that are not in testing set
len([x for x in X_train.Cabin.unique() if x not in X_test['Cabin'].unique()])

# Labels in testing set that are not in training set
len([x for x in X_test.Cabin.unique() if x not in X_train['Cabin'].unique()])

type(X_train)

mapper = {k:i for i, k in enumerate(X_train['Cabin'].unique(), 0)}

# replace the labels in Cabin, using the dic created above
X_train.loc[:, 'Cabin_mapped'] = X_train.loc[:, 'Cabin'].map(mapper)
X_test.loc[:, 'Cabin_mapped'] = X_test.loc[:, 'Cabin'].map(mapper)

X_train[['Cabin_mapped', 'Cabin']].head(10)

mapper = {k: i for i, k in enumerate(X_train['Cabin_processed'].unique(), 0)}

# replace labels by numbers with dictionary
X_train.loc[:, 'Cabin_processed'] = X_train.loc[:, 'Cabin_processed'].map(mapper)
X_test.loc[:, 'Cabin_processed'] = X_test.loc[:, 'Cabin_processed'].map(mapper)

X_train[['Cabin_processed', 'Cabin', 'Cabin_mapped']].head(10)

X_train.loc[:, 'Sex'] = X_train.loc[:, 'Sex'].map({'male': 0, 'female': 1})
X_test.loc[:, 'Sex'] = X_test.loc[:, 'Sex'].map({'male': 0, 'female': 1})

X_train.Sex.head()

X_train[['Cabin_mapped','Cabin_processed', 'Sex']].isnull().sum()

X_test[['Cabin_mapped','Cabin_processed', 'Sex']].isnull().sum()

len(X_train['Cabin_mapped'].unique()), len(X_train['Cabin_processed'].unique())

X_train.columns

X_test = X_test.fillna(0)

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.linear_model import RidgeClassifierCV
classifier = RidgeClassifierCV()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.linear_model import RidgeClassifierCV
classifier = RidgeClassifierCV()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.linear_model import Perceptron
classifier = Perceptron()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train[['Sex', 'Cabin_mapped']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_mapped']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train[['Sex', 'Cabin_processed']],y_train)
y_pred = classifier.predict(X_test[['Sex', 'Cabin_processed']])
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))