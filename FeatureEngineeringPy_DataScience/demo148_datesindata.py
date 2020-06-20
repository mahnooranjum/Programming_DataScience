# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train_date.csv")
data.head()

data.dtypes

data['Date'] = pd.to_datetime(data['Date'])

data.head()

data.dtypes

fig = data.groupby(['Date'])['Survived'].sum().plot(
                   figsize=(5, 8), linewidth=1)

fig.set_title('Distribution')
fig.set_ylabel('y')

data['year'] = pd.DatetimeIndex(data['Date']).year

data.dtypes

data[['Date', 'year']].head()

data['month'] = pd.DatetimeIndex(data['Date']).month
data[['Date', 'year', 'month']].tail()

data['day'] = pd.DatetimeIndex(data['Date']).day
data[['Date', 'year', 'month', 'day']].tail()

data['week'] = pd.DatetimeIndex(data['Date']).week
data[['Date', 'year', 'month', 'day', 'week']].tail()

data['weekday'] = pd.DatetimeIndex(data['Date']).weekday
data[['Date', 'year', 'month', 'day', 'week', 'weekday']].head()

sns.set_palette(sns.light_palette("purple"))
sns.catplot(x = 'year' , y='Survived', kind ='bar', data = data)

sns.set_palette(sns.diverging_palette(10, 220, n=12))
sns.catplot(x = 'month' , y='Survived', kind ='bar', data = data)

sns.set_palette(sns.diverging_palette(145, 280, n=7, center='dark'))
sns.catplot(x = 'weekday' , y='Survived', kind ='bar', data = data)