# -*- coding: utf-8 -*-
"""Demo36_ObjectOrientedPython.ipynb


# OBJECT ORIENTED PROGRAMMING IN PYTHON 

Welcome to the second tutorial in this section. Object Oriented programming is a style of programming made available by high level languages such as Python, C++ and Java. It utilizes the concept of real world OBJECTS belonging to different CLASSES. 

Some Examples are presented below:

--------------------------------------
### < OBJECT , CLASS >
- < Mahnoor , Human >
- < Mazda , Car > 
- < FC Barcelona, Football Club > 
- < Electrical Engineering, Degree >
- < Nemo , Fish >


Most of our work in the Deep Learning section will heavily rely on OOP concepts hence we will develop a sound foundation in this tutorial. 

- Python Basics
- **Object Oriented Python**
- Python for Data Science
- NumPy
- Pandas
- Plotting 
    - Matplotlib
    - Seaborn

    
Let's get coding !

### CLASSES AND OBJECTS
"""

#This is a class declaration
class Student:
    pass

"""Here, Mahnoor Usman and Abdullah are all Instances or Objects (Yes Python just objectified people) of the class Student."""

#The class is a category which can contain objects or instances.
Mahnoor = Student()
Usman = Student()
Abdullah = Student()
print(Mahnoor)

"""Now that our objects, I mean Students(dang it) are ready, let us create some instance variables for them. 

Instance variables are basically attributes assigned to each object. 

They are specific to the object created and have no correlation with any other object of the same class.
"""

Mahnoor.LastName = 'Anjum'
Mahnoor.Hobby = 'Sleeping'
Mahnoor.Age = 21
Mahnoor.School = 'SEECS'

Usman.LastName = 'Muhammad'
Usman.Hobby = 'Gaming'
Usman.Age = 18
Usman.School = 'School of Thought'

Abdullah.LastName= 'Muhammad'
Abdullah.Hobby = 'Anime'
Abdullah.Age = 19
Abdullah.School = 'Master Academy'
print(Mahnoor.Hobby)
print(Abdullah.Hobby)
print(Usman.Hobby)

"""#### BUT THAT IS SO TEDIOUS
YES, YES IT IS. Which is why we should never create variables or ATTRIBUTES as described above. We should instantiate them in the CLASS DEFINITION and SET them while we are creating those OBJECTS !!!
"""

#Smart Class Definition
class Student:
    def __init__(self, LastName, Hobby, Age, School):
        self.LastName = LastName
        self.Hobby = Hobby
        self.Age = Age
        self.School = School
    def print_object(self, FirstName):
        print("---------------------------------")
        print("Last Name : {} {}".format(FirstName, self.LastName))
        print("Age : {}".format(self.Age))
        print("Hobby : {}".format(self.Hobby))
        print("School : {}".format(self.School))
        print("---------------------------------")

"""Okay let's go over these arguments one by one. 
- def __init__(self, LastName, Hobby, Age, School):
We are definiting an initialization method for the class Student. Initialization is basically the "creation" of an "object" of the "class". 

So why do we need a method to initialize it? We need a method when we want to specify some attributes of the object. For example, if we have a class of "CAR" and we want to enter different cars in the database. We will have a few "CAR" variables (or object variables, since car is the class and instances are objects) that define the object uniquely. 

It's model, company, manufacturer, horse power, speed, type et cetera.
"""

Mahnoor = Student("Anjum", "Sleeping", 21,"SEECS")
Mahnoor.print_object("Mahnoor")

Student.print_object(Mahnoor,"Mahnoor")

"""### CLASS VARIABLES
Class variables are the variables that are shared by ALL objects of ONE CLASS. These variables are not specific to ONE object. Consider the venn diagram below. 

![venn.PNG](attachment:venn.PNG)

The intersection of the two objects i.e students are the class variables.
"""

#Smarter Class Definition
class Student:
    Tax = 1.02
    Count = 0
    def __init__(self, LastName, Hobby, Age, School, Fee):
        self.LastName = LastName
        self.Hobby = Hobby
        self.Age = Age
        self.School = School
        self.Fee = Fee
        Student.Count = Student.Count+1
    def print_object(self, FirstName):
        print("---------------------------------")
        print("Last Name : {} {}".format(FirstName, self.LastName))
        print("Age : {}".format(self.Age))
        print("Hobby : {}".format(self.Hobby))
        print("School : {}".format(self.School))
        print("Fee : {}Rs".format(self.Fee))
        print("---------------------------------")
    def update_fee(self):
        self.Fee = int(self.Fee * Student.Tax)

Mahnoor = Student("Anjum", "Sleeping", 21,"SEECS", 98000)
Mahnoor.print_object("Mahnoor")
Usman = Student("Muhammad","Gaming", 18, "School of Thought", 78000)
Usman.print_object("Usman")

Mahnoor.update_fee()
Mahnoor.print_object("Mahnoor")

print(Mahnoor.Tax)
print(Usman.Tax)

print(Mahnoor.__dict__)
print(Usman.__dict__)

print(Student.__dict__)

Student.Tax = 1.03 #Class variable updated
print(Mahnoor.Tax) #Variable changed for all objects
print(Usman.Tax)
Student.Tax = 1.02 #Class variable updated

Mahnoor.Tax=1.03 #Class variable updated for ONE object effectively
print(Mahnoor.Tax)#creating another object variable named Tax. So,
print(Usman.Tax)#the object Mahnoor has two variables called Tax. 
print("===================================")
print(Mahnoor.__dict__)
print("===================================")
print(Usman.__dict__)
print("===================================")
print(Student.__dict__)
Mahnoor.Tax=1.02

Mahnoor.Count

"""### SUBCLASSES AND INHERITANCE"""

#Smarter Class Definition
class Student:
    Tax = 1.02
    Count = 0
    def __init__(self, LastName, Hobby, Age, School, Fee):
        self.LastName = LastName
        self.Hobby = Hobby
        self.Age = Age
        self.School = School
        self.Fee = Fee
        Student.Count = Student.Count+1
    def print_object(self, FirstName):
        print("---------------------------------")
        print("Last Name : {} {}".format(FirstName, self.LastName))
        print("Age : {}".format(self.Age))
        print("Hobby : {}".format(self.Hobby))
        print("School : {}".format(self.School))
        print("Fee : {}Rs".format(self.Fee))
        print("---------------------------------")
    def update_fee(self):
        self.Fee = int(self.Fee * Student.Tax)

class SEECS(Student):
    Tax = 1.18
    def __init__(self, LastName, Hobby, Age, School, Fee, FYP):
        super().__init__(LastName, Hobby, Age, School, Fee)
        self.FYP = FYP

class SADA(Student):
    Tax = 1.1
    def __init__(self, LastName, Hobby, Age, School, Fee, Thesis):
        super().__init__(LastName, Hobby, Age, School, Fee)
        self.Thesis = Thesis

Jane = SADA("Doe","Yelling",21,"SADA",10200, "Convertable Table")
Mahnoor =SEECS("Anjum", "Sleeping", 21,"SEECS", 98000, 'LPWAN')
print(Mahnoor.Tax)
print(Jane.Tax)
#help(SADA)

print(Mahnoor.FYP)
print(Jane.Thesis)

"""### CLASS AND STATIC METHODS"""

#Smarter-er Class Definition
class Student:
    Tax = 1.02
    Count = 0
    def __init__(self, LastName, Hobby, Age, School, Fee):
        self.LastName = LastName
        self.Hobby = Hobby
        self.Age = Age
        self.School = School
        self.Fee = Fee
        Student.Count = Student.Count+1
    def print_object(self, FirstName):
        print("---------------------------------")
        print("Last Name : {} {}".format(FirstName, self.LastName))
        print("Age : {}".format(self.Age))
        print("Hobby : {}".format(self.Hobby))
        print("School : {}".format(self.School))
        print("Fee : {}Rs".format(self.Fee))
        print("---------------------------------")
    def update_fee(self):
        self.Fee = int(self.Fee * Student.Tax)
    @classmethod #This method will effect the entire class
    def set_tax(cls, amount):
        cls.Tax = amount
    @classmethod 
    def from_str(cls, str):
        LastName, Hobby, Age, School, Fee = str.split(',')
        return cls(LastName, Hobby, Age, School, Fee)

Mahnoor = Student("Anjum", "Sleeping", 21,"SEECS", 98000)
Mahnoor.print_object("Mahnoor")
Usman = Student("Muhammad","Gaming", 18, "School of Thought", 78000)
Usman.print_object("Usman")

print(Mahnoor.Tax)
print(Usman.Tax)
Mahnoor.set_tax(1.07) 
#This method, called by ANY object, updates the variables for ALL
#objects. Class methods can only contain class variables.
print(Mahnoor.Tax)
print(Mahnoor.Hobby)
print(Usman.Tax)

New_Mahnoor = Student.from_str("Anjum,Sleeping,21,SEECS,98000")
Student.print_object(New_Mahnoor,"Mahnoor")

"""### SPECIAL METHODS OR DUNDER METHODS
Special methods also known as dunder methods are always surrounded by double underscores. Could you recall a dunder we have already seen?
"""

#Smarter-er-er Class Definition
class Student:
    Tax = 1.02
    Count = 0
    def __init__(self, LastName, Hobby, Age, School, Fee):
        self.LastName = LastName
        self.Hobby = Hobby
        self.Age = Age
        self.School = School
        self.Fee = Fee
        Student.Count = Student.Count+1
    def update_fee(self):
        self.Fee = int(self.Fee * Student.Tax)
    @classmethod
    def set_tax(cls, amount):
        cls.Tax = amount
    @classmethod
    def from_str(cls, str):
        LastName, Hobby, Age, School, Fee = str.split(',')
        return cls(LastName, Hobby, Age, School, Fee)
    def __repr__(self):
        return "Student({},{},{},{},{})".format(self.LastName, self.Hobby, self.Age, self.School, self.Fee)
        
    def __str__(self):
        print("---------------------------------")
        print("Last Name : {}".format(self.LastName))
        print("Age : {}".format(self.Age))
        print("Hobby : {}".format(self.Hobby))
        print("School : {}".format(self.School))
        print("Fee : {}Rs".format(self.Fee))
        print("---------------------------------")
        return ""
    def __add__(self, other):
        return self.Fee + other.Fee

Mahnoor = Student("Anjum", "Sleeping", 21,"SEECS", 98000) #WHY not from_str
Usman = Student("Muhammad","Gaming", 18, "School of Thought", 78000)
str(Mahnoor)
repr(Mahnoor)

print(Mahnoor + Usman)

"""### WELCOME TO THE END OF THE TUTORIAL
You made it!! Hope you enjoyed taking this tutorial as much as I enjoyed coding it. From the next tutorial, we will be jumping right into the Data Science ascpect of Python Programming Language. Until then, happy coding. 

---------------------------------------------------------------------------------
Copyrights © 2018, All Rights Reserved.
- Author: Mahnoor Anjum.
- Course: The Complete Hands-On Machine Learning Course
- Date Created: 2018-06-28
- Date Modified: -
"""