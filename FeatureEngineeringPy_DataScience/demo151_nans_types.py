# -*- coding: utf-8 -*-
"""Demo151_NANs_Types.ipynb

## Missing values

Reference:

https://www.ncbi.nlm.nih.gov/books/NBK493614/


Missing data is encountered when __no data__ / __no value__ is stored for a variable in an observation. 

### Types of Missing Data

Missing data can be of the following types:

#### Missing Completely at Random [MCAR]:

- A variable has data missing completely at random if the probability of being missing is the same for all the observations

- The fact that the data are missing is independent of the observed and unobserved data

- No systematic differences exist between participants with missing data and those with complete data

- In these instances, the missing data reduce the analyzable population of the study and consequently, the statistical power

- Removing them does not introduce bias, i.e. when data are MCAR, the data which remain can be considered a simple random sample of the full data set of interest

- MCAR is generally regarded as a strong and often unrealistic assumption


#### Missing at Random [MAR]: 

- The fact that the data are missing is systematically related to the observed but not the unobserved data

- The probability of an an observation being missing depends only on available information 

- For example, if women are less likely to disclose their age than men, age is MAR

- If we decide to use the MAR variable with missing values, we will have to include the correlated variables (e.g., gender) to control the bias in MAR variable (e.g., age) for the missing observations

#### Missing Not at Random [MNAR]: 

- When data are MNAR, the fact that the data are missing is systematically related to the unobserved data, that is, the missingness is related to events or factors which are not measured by the researcher.

- MNAR would occur if people failed to fill in a depression survey because of their level of depression. 


### Rules of thumb


- The complete case analysis will be unbiased due to missing data if the missingness is independent of the outcome under study, a condition that can be present whether the data are MAR or MNAR

- However, if the missingness is not independent of outcome, it can be made so through analytic means only if the missingness is MAR.

__The MAR vs. MNAR distinction is therefore not to indicate that there definitively will or will not be bias in a complete case analysis, but instead to indicate – if the complete case analysis is biased – whether that bias can be fully removed in analysis (see below sections for analytic strategies)__
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv("gdrive/My Drive/Colab Notebooks/FeatureEngineering/train.csv")

data.head()

len(data)

data.isnull().sum()

data.isnull().mean()*100

data['Cabin_nulls'] = np.where(data.Cabin.isnull(), 1, 0)

data['Cabin_nulls'].mean()

# Group data by Survived vs Non-Survived
data.groupby(['Survived'])['Cabin_nulls'].mean()

data['Age_nulls'] = np.where(data['Age'].isnull(), 1, 0)
data.groupby(['Survived'])['Age_nulls'].mean()

## Missing Completely at Random
data['Embarked_nulls'] = np.where(data['Embarked'].isnull(), 1, 0)
data.groupby(['Survived'])['Embarked_nulls'].mean()

data['Embarked_nulls'].sum()