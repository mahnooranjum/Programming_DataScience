# -*- coding: utf-8 -*-
"""Demo37_PythonforDataScience.ipynb

# PYTHON FOR DATA SCIENCE
We will take our python programming skills a step further and process large data in it. Python is an excellent language for deployment. Hence we will be using open source data during the learning process!!

This will make sure we understand the challenges a Data Scientist can face and how to deal with them. In my experience, Data Preprocessing takes 70% of the time in any project. Hence it is crucial for any Data Scientist to know what it is and how it is done. 

This may be the boring portion of the course but I assure you, you will feel accomplished by the end of this tutorial. 

- Python Basics
- Object Oriented Python
- **Python for Data Science**
- NumPy
- Pandas
- Plotting 
    - Matplotlib
    - Seaborn
    
Let's get coding !!
"""

#Variables can not start with a number
12var = 1

_13var = 1

name = "Mahnoor"
surname = "Anjum"
age = 21
print("I'm {} {} and I am {} years old.".format(name, surname, age))

name = "Mahnoor"
surname = "Anjum"
age = 21
print("I'm {_1} {_2} and I am {_3} years old.".format(_1 = name, _2= surname, _3 = age))

"""### INDEXING AND SLICING
One of the most important Python concept for data scientists is the slicing operator ':'
"""

str = "ONE TWO THREE FOUR FIVE"
print(str[0])
print(str[5])
print(str[len(str)-1])

str[:5]

str[5:]

str[1]="a"

nested = [1,2,3,['_1','_2','_3',['__1']]]
nested[0]

nested[3][0]

len(nested)

len(nested[3])

nested[3][3]

nested[3][3][0]

dict = {'key1':'value1', \
        'key2': 'value2', \
        'key3':'value3'}
dict['key1']

T = True
F = False
var = 10
for i in range(var):
    print(i)

for i in range(var):
    bool = (i==2)
    if bool:
        break
    print(i)

[1,2,3,1,1,2,3,4]

(1,2,3,1,1,2,3,4)

{1,2,3,1,1,2,3,4}

new_set = set([1,2,3,1,1,2,3,4])
new_set.add(5)
new_set

for item in new_set:
    print(item)

list(range(4))

my_list = list(range(5,10))

output = []
for number in my_list:
    output.append(number**3)
output

output = [num**3 for num in my_list]
output

"""### FUNCTIONS"""

def my_function(parameter):
    print(parameter)
my_function("Jalebi (Hungry okay?)")

def my_function(parameter="Default"):
    print(parameter)
my_function()

num = 4
def change(par):
    par =5
    return par 
change(num)
num

num = 4
def change(par):
    par =5
    return par 
change(num)
num

num = [4]
def change(par):
    par.append(5)
    del par[0]
    return par 
change(num)
num

my_list

"""### LAMBDA EXPRESSIONS"""

def square(x): return x*x
list(map(square, my_list))

list(map(lambda x:x*x, my_list))

"""### BUILT-IN FUNCTIONS"""

s = "We have a hulk !!!"

s.lower()

s.upper()

s.split()

dict = {'key1':1,'key2':2}
dict.keys()

dict.values()

dict.items()

my_list.pop()

my_list

"""### TUPLE UNPACKING"""

list_of_tuples  =[(1,2),(3,4),(5,6)]
for (a,b) in list_of_tuples:
    print (a)
    print (b)

"""### WELCOME TO THE END OF THE TUTORIAL
You made it!! Hope you enjoyed taking this tutorial as much as I enjoyed coding it. From the next tutorial, we will be starting our first Data Science Library called NumPy. Until then, happy coding. 

---------------------------------------------------------------------------------
Copyrights © 2018, All Rights Reserved.
- Author: Mahnoor Anjum.
- Course: The Complete Hands-On Machine Learning Course
- Date Created: 2018-06-27
- Date Modified: -
"""