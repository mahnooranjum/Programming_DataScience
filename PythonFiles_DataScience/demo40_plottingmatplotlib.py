# -*- coding: utf-8 -*-
"""Demo40_PlottingMatplotlib.ipynb


# MATPLOTLIB FOR VISUALIZATION

Welcome to the first exciting colorful data visualization tutorial. Matplotlib is the default library most data scientists use for plotting. It is an excellent tool and a must-know for any python proficient data scientist.

- Python Basics
- Object Oriented Python
- Python for Data Science
- NumPy
- Pandas
- **Plotting** 
    - **Matplotlib**
    - Seaborn

    
Let's get visualizing!
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import math
# %matplotlib inline
import numpy as np

x = np.linspace(-1,1,100)
y = x*x

plt.plot(x,y)
plt.show()
plt.plot(x,y,'k')
plt.ylabel('y label')
plt.xlabel('x label')
plt.title('Title')
plt.show()

"""### MULTIPLE PLOTS"""

x = np.linspace(-10,10,100)
y = []
for i in x: 
    y.append(math.sin(i))
plt.subplot(1,3,1)
plt.plot(x,x*x, 'r')
plt.subplot(1,3,2)
plt.plot(x,x*x*x, 'k')
plt.subplot(1,3,3)
plt.plot(x,y,'b')

"""### OBJECT ORIENTED PLOTTING
So remember the classes and attributes and objects and instances and all that jazz?

This is where we use it for the very first time. Object oriented plotting is one of the most important features matplotlib has to offer. we instantiate a figure which is basically a plot, then we graph on it by calling different methods on it. 

Sounds complicated Mahnoor, let's break it down
"""

figure = plt.figure()
left, bottom, width, height = 0, 0, 0.4, 0.9
ax = figure.add_axes([left, bottom, width, height])
ax.plot(x,y)
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_title("Title")
ax.set_xlim([-10,10])
ax.set_ylim([-1,1])

figure = plt.figure() 
left, bottom, width, height = 0, 0, 0.9, 0.9
left1, bottom1, width1, height1 = 0.3, 0.5, 0.2, 0.3

ax = figure.add_axes([left, bottom, width, height])
ax1 = figure.add_axes([left1, bottom1, width1, height1])
ax.plot(x,y)
ax1.plot(x,y)

"""#### WHAT. JUST. HAPPENED
If you're wondering what just happened. You. are. not. alone. 

It took me a while too. 
So here's how I visualize the above code:
- **paper = plt.figure()** - creates a canvas. Think of it as the paper you're drawing the graph on.
- **box = paper.add_axes()** - basically creates the box in which you want to draw your graph.
- **box.plot()** - creates the plot !
"""

figure,axes = plt.subplots(nrows=1, ncols=5)
plt.tight_layout()

figure,axes = plt.subplots(nrows=1, ncols=2)
for ax in axes:
    ax.plot(x,y)
plt.tight_layout()
axes[0].set_title('Title graph 0')
axes[1].set_title('Title graph 1')

# [8.0, 6.0] is the default figsize
figure = plt.figure(figsize=(3,4))
ax = figure.add_axes([0,0,1,1])
ax.plot(x,y)

fig, axes = plt.subplots(nrows=3, ncols=1,figsize=(10,4))
axes[1].plot(x,x*x*x, label="x*x*x")
axes[1].plot(x,x*x, label="x*x")
plt.tight_layout()
axes[1].legend()
fig.savefig("Figure1", dpi=300)

"""### CUSTOMIZE THE APPEARANCE"""

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y,color='purple')#Can use RGB hex codes

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y,color='purple', linewidth= 10, alpha=.4)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y,color='purple', linestyle='-.', marker='o',markersize=10)

"""### STYLIZE 
I have covered a few basic techniques of plot appearance customization. You can visit the matplotlib documentation for more details. 
A few more options I would personally encourage you to try are as follows:
- linewidth = 2 or lw = 2
- linestyle = '-.' or ls = '-.'
- marker = 'o' 
- markersize = 10
- markerfacecolor = 'green'
- markeredgewidth = 2
- markeredgecolor = 'blue'

### WELCOME TO THE END OF THE TUTORIAL
You made it! As always, Hope you enjoyed taking this tutorial as much as I enjoyed making it. From the next tutorial, we will be starting data visualizations with Seaborn. 
Until next time folks, Happy visualizing.

---------------------------------------------------------------------------------
Copyrights © 2018, All Rights Reserved.
- Author: Mahnoor Anjum.
- Course: The Complete Hands-On Machine Learning Course
- Date Created: 2018-06-29
- Date Modified: -
"""