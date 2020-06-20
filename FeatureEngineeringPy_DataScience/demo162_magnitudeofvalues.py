# -*- coding: utf-8 -*-
"""Demo162_MagnitudeOfValues.ipynb

## Scale of variables

Reference https://machinelearningmastery.com/

- In Linear Regression models models **y = w x + b**, the scale of the X variable matters 

- The value of **w** is partly affected by the magnitude of **x**

- Changing the scale from mm to km will cause a change in the magnitude of the **w**

- Unscaled input variables can result in a slow or unstable learning process

- Unscaled target variables on regression problems can result in exploding gradients

- Input variables with larger values may dominate the learning curves

- Gradient descent converges faster when the input features are scaled 

- SVMs perform better with scaled features 

- Methods that require distance metrics, e.g., KNN, KMeans, are usually affected by the scale of input features 

### Affected Models

- KNN
- K-means clustering
- Linear Discriminant Analysis 
- Principal Component Analysis 
- Linear and Logistic Regression
- Neural Networks
- Support Vector Machines


### Unaffected Models

- Trees
- Random Forests
- Gradient Boosted Trees
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")

data.head()

num_cols = ['Survived', 'Pclass', 'Age', 'Fare']
data = data[num_cols]

data.describe()

for i in ['Pclass', 'Age', 'Fare']:
    print(i,': ', data[i].max()-data[i].min())

data.isnull().sum()

X_train, X_test, y_train, y_test = train_test_split(
          data[['Age', 'Fare', 'Pclass',]].fillna(data.mean()),
          data['Survived'],
          test_size=0.2)

X_train.shape, X_test.shape

"""### Feature Scaling"""

for i in ['Pclass', 'Age', 'Fare']:
    print(i,'Min: ', X_train[i].min())
    print(i,'Max: ', X_train[i].max())
    print(i,'Range: ', X_train[i].max()-X_train[i].min())
    print(i,'Mean: ', X_train[i].mean())
    print(i,'Std: ', X_train[i].std())

obj = StandardScaler()
X_train_scaled = obj.fit_transform(X_train)
X_test_scaled = obj.transform(X_test)

for i in range(3):
    print(i,'Min: ', X_train_scaled[i].min())
    print(i,'Max: ', X_train_scaled[i].max())
    print(i,'Range: ', X_train_scaled[i].max()-X_train_scaled[i].min())
    print(i,'Mean: ', X_train_scaled[i].mean())
    print(i,'Std: ', X_train_scaled[i].std())

obj = MinMaxScaler()
X_train_scaled = obj.fit_transform(X_train)
X_test_scaled = obj.transform(X_test)

for i in range(3):
    print(i,'Min: ', X_train_scaled[i].min())
    print(i,'Max: ', X_train_scaled[i].max())
    print(i,'Range: ', X_train_scaled[i].max()-X_train_scaled[i].min())
    print(i,'Mean: ', X_train_scaled[i].mean())
    print(i,'Std: ', X_train_scaled[i].std())



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.linear_model import RidgeClassifierCV
classifier = RidgeClassifierCV()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.linear_model import RidgeClassifierCV
classifier = RidgeClassifierCV()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.linear_model import Perceptron
classifier = Perceptron()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))

classifier.fit(X_train_scaled,y_train)
y_pred = classifier.predict(X_test_scaled)
y_pred = np.round(y_pred).flatten()
print(accuracy_score(y_test, y_pred))