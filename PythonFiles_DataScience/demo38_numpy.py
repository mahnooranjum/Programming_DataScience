# -*- coding: utf-8 -*-
"""Demo38_Numpy.ipynb



# NUMPY TO THE RESCUE

Welcome to the NumPy tutorial. I am super excited that you're here and I sincerely hope you have enjoyed the course thus far. We will heavily use the slicing notation and basic python concepts in this section. If you have not completed the previous tutorials, I strongly suggest you run through them before you proceed.

- Python Basics
- Object Oriented Python
- Python for Data Science
- **NumPy**
- Pandas
- Plotting 
    - Matplotlib
    - Seaborn

    
Let's get coding !!

### NUMPY ARRAYS
We have two types of NumPy arrays. 
- Vectors
- Matrices

Vectors are one dimensional arrays where as Matrices are two dimensional.
"""

l = [1,2,3,4,5]
import numpy as np
# np is an alias for our ease
arr = np.array(l)
arr

m = [[1,2,3],
     [4,5,6],
     [7,8,9]]
np.array(m)

np.arange(0,5)

np.arange(0,51,5)

np.zeros((3,2))

np.ones((3,4))

np.ones(12)

np.linspace(0,1,10)

np.eye(3)

np.random.rand(3)

np.random.randn(3)

np.random.randint(1,100,5)#1-100 range

arr = np.random.randint(1,100,(3,4))
arr

arr.reshape(4,3)

arr.reshape(-1,1)

arr.reshape(1,-1)

arr.max()

arr.min()

arr.argmin()

arr.argmax()

len(arr)

"""### INDEXING, SLICING AND SELECTION IN NUMPY"""

arr = np.arange(11)
arr

arr[1:4]

arr[:6]

arr[5:]

arr[0:5] = 0
arr

#Slicing is exactly what it sounds like
#Changing a slice means changing the entire array
arr = np.arange(11)
arr

_slice = arr[2:5]
_slice

_slice[:] = 200
arr

#Hence it is advised that you create copies.
arr_copy = arr.copy()
arr_copy

_2d = np.random.randint(1,1000,(3,3))
_2d

_2d[1][2]

_2d[1,2]

_2d[:2,1:]

_2d[:2]

bool_2d = _2d<500
bool_2d

_2d[bool_2d]

_2d[_2d<500]

"""### OPERATIONS ON NUMPY ARRAYS"""

arr = np.arange(11)
arr

arr+arr

arr+100

arr/arr

arr**3

np.max(arr)

_2d = np.arange(9).reshape(3,3)
_2d

_2d.sum(axis=1)

_2d.sum(axis=0)

"""### WELCOME TO THE END OF THE TUTORIAL
You made it!! Hope you enjoyed taking this tutorial as much as I enjoyed coding it. From the next tutorial, we will be starting data manipulations in Pandas. Until then, happy coding. 

---------------------------------------------------------------------------------
Copyrights © 2018, All Rights Reserved.
- Author: Mahnoor Anjum.
- Course: The Complete Hands-On Machine Learning Course
- Date Created: 2018-06-27
- Date Modified: -
"""