# -*- coding: utf-8 -*-
"""Demo185_OrdinalEncoding.ipynb

## Ordinal encoding

- The labels can be ordered 
- Assign numbers to categories 

### Pros

- Keeps variables semantic 
- Easy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train_date.csv")

data.head()

data.columns

data['Date'] = pd.to_datetime(data['Date'])
data.head()

type(data['Date'][0])

data['Date'][0].weekday

data['year'] = pd.DatetimeIndex(data['Date']).year

data.head()

data['day'] = pd.DatetimeIndex(data['Date']).day

data.head()

