# -*- coding: utf-8 -*-
"""Demo39_Pandas.ipynb

# MUNG - FU PANDA 

Welcome to the Pandas tutorial. Pandas is an excellent tool for data wrangling also known as data munging.
It refers to the cleaning and preperation of data from Raw format to a usable and suitable format for our use.

- Python Basics
- Object Oriented Python
- Python for Data Science
- NumPy
- **Pandas**
- Plotting 
    - Matplotlib
    - Seaborn

    
Let's get coding !

### SERIES AND DATAFRAMES
Series and dataframes are the main data types Pandas introduces.
"""

import numpy as np
import pandas as pd
Student_ID = list(range(10,20))
Grades = ['A','B','A','A','F','C','F','F','D','A']
arr_Grades = np.array(Grades)

print(pd.Series(data = Grades))

print(pd.Series(data = Grades, index = Student_ID))

print(pd.Series(arr_Grades))

d = {'Pakistan':11, 'Germany':4, 'Brazil':5, 'Argentina':6}
S = pd.Series(d)

print(S['Pakistan'])

print(S + S)

print(S-S)

print(S**S)

arr = np.random.randint(1,10,5)
df = pd.DataFrame(arr, ['A','B','C','D','E'])

print(df)

df = pd.DataFrame(np.random.randint(1,10,(5,3)), ['A','B','C','D','E'],['Q','W','E'])
print(df)

print(df['W'])

print(df.W)

print(type(df.W))
print(type(df['W']))
print(df[['Q','W']])

df['New Column'] = 0
print(df)

df.drop('New Column', axis = 1, inplace = True)
print(df)

print(df.loc['C'])
print(type(df.loc['C']))

print(df.iloc[2])
print(type(df.iloc[2]))

print(df.iloc[1:4,1:3])

print(df.loc[['A','D'],['Q','E']])

print(df>5)

print(df[df>5])

print(df[df['Q']>5])

print(df[(df['Q']>5) & (df['E']>5)])

print(df[(df['Q']>5) | (df['E']>5)])

df = df.reset_index()
print(df)

print(df['index'])

print(df.set_index('index'))

"""### DATA MANIPULATIONS"""

col1 = [1,2,np.nan,np.nan,5]
col2 = [5,6,7,np.nan,8]
col3 = [12,np.nan,13,14,15]
d = {'A':col1, 'B':col2, 'C':col3}
df = pd.DataFrame(d)
print(df)

print(df.dropna())
print(df.dropna(thresh=2))

print(df.fillna(value="VALUE"))

print(df['A'].fillna(value=df['A'].mean()))

"""### MERGE
Merging refers to a "Glue-ing" technique that does not care about the index.
"""

left = pd.DataFrame({'Key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'Key': ['K0', 'K1', 'K2', 'K4'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})  
print(left)
print(right)

#Inner join only looks at the intersection
print(pd.merge(left, right, on="Key", how="inner"))

#Left join gives us the left df + the intersection
print(pd.merge(left, right, on="Key", how="left"))

#Right join gives us the right df + the intersection
print(pd.merge(left, right, on="Key", how="right"))

#Outer join gives us the right df + the intersection + the left df
print(pd.merge(left, right, on="Key", how="outer"))

"""### JOIN
Joining refers to a "Glue-ing" technique that does care about the index
"""

left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']},
                   index = ['K0', 'K1', 'K2', 'K3'])
   
right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                    index =  ['K0', 'K1', 'K2', 'K4'],)  
print(left)
print(right)

print(left.join(right))

print(left.join(right, how="inner"))

print(left.join(right, how="outer"))

print(right.join(left))

"""### CONCATENATE"""

df1 = pd.DataFrame({'C0': ['00', '10','20', '30'],
                        'C1': ['01', '11', '21', '31'],
                        'C2': ['02', '12', '22', '32'],
                        'C3': ['03', '13', '23', '33']},
                        index=[0, 1, 2, 3])
print(df1)

df2 = pd.DataFrame({'C0': ['40', '50','60', '70'],
                        'C1': ['41', '51', '61', '71'],
                        'C2': ['42', '52', '62', '72'],
                        'C3': ['43', '53', '63', '73']},
                        index=[4, 5, 6, 7])
print(df2)

df3 = pd.DataFrame({'C0': ['80', '90','10,0', '11,0'],
                        'C1': ['81', '91', '10,1', '11,1'],
                        'C2': ['82', '92', '10,2', '11,2'],
                        'C3': ['83', '93', '10,3', '11,3']},
                        index=[8, 9, 10, 11])
print(df3)

print(pd.concat([df1,df2,df3]))

print(pd.concat([df1,df2,df3], axis=1))

"""### GROUPBY"""

import pandas as pd
school = ['SEECS','SEECS','SMME','SEECS','SCME','SMME','SADA']
student = ['Mahnoor','Usman','Mustafa','Abdullah','Mahum','Armughan','Ayesha']
cgpa = [3.12,4,3.17,4,3.14,3.04,3.04]
data = {'School':school, 'Student':student,'CGPA':cgpa}
df = pd.DataFrame(data)
print(df)

bySchool = df.groupby('School')
bySchool

print(bySchool.mean())
print(bySchool.std())
print(bySchool.sum())

print(df.groupby('School').std().loc['SMME'])

print(df.groupby('School').describe())

"""### OPERATIONS IN PANDAS"""

school = ['SEECS','SEECS','SMME','SMME','SCME','SMME','SADA']
student = ['Mahnoor','Usman','Mustafa','Abdullah','Mahum','Armughan','Ayesha']
cgpa = [3.12,4,4,3.13,3.14,3.04,3.04]
age = [21,18,22,21,20,21,24]
df= pd.DataFrame({'Student':student, 'School':school, "CGPA":cgpa, 'Age':age})
print(df)

print(df[(df['School']=='SEECS') & (df['CGPA']==4)])

(df['School'].value_counts())

print(df.sort_values(by='CGPA'))

print(df['School'].unique())
print(df['School'].nunique())

print(df.drop('School', axis=1))

print(df.columns)
print(df.index)

print(df.isnull())

"""### APPLYING CUSTOM FUNCTIONS"""

def square(x): return x*x
# lambda x : x*x

print(df.CGPA.apply(square))

print(df.CGPA.apply(lambda x:x*x))

"""### DATA I/O
You need to install sqlalchemy and lxml for this section of the tutorial
"""

import pandas as pd
df = pd.read_csv('../Datasets/CustomerList.csv')
print(df[1:10])

df.to_csv('Output',index=False)

df = pd.read_excel('../Datasets/Churn-Modelling.xlsx')
print(df[1:10])

"""### WELCOME TO THE END OF THE TUTORIAL
You made it!! As always, Hope you enjoyed taking this tutorial as much as I enjoyed making it. From the next tutorial, we will be starting data visualizations, enough with the boring mumbo jumbo, let's put some colors in our lives. Until then, enjoy data wrangling. 

---------------------------------------------------------------------------------
Copyrights © 2018, All Rights Reserved.
- Author: Mahnoor Anjum.
- Course: The Complete Hands-On Machine Learning Course
- Date Created: 2018-06-28
- Date Modified: -
"""