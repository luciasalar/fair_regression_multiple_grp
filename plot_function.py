import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy import linalg as LA

"""Here we plot the functions and see if they are convex problems """


# generate data
y_true  = random.normal(size=(100, 1))
noise = random.normal(size=(100, 1))
y_pred = y_true + noise

x = random.normal(size=(100, 1))
beta = [0.1, 0.2, 0.3, 0.4]

#residuals of multiple groups
g1  = random.normal(size=(100, 4))
g2 = random.normal(size=(100, 1))

c1  = random.normal(size=(100, 1))
c2 = random.normal(size=(100, 1))

d1  = random.normal(size=(100, 1))
d2 = random.normal(size=(100, 1))

# this algorithm is convex
#y = x**2 + ((g1-g2) + (c1-c2) + (d1-d2))**2 

# if there are multiple means (5 age levels), we probably want a distance function between 3 means
y = x**2 + (y_pred.size * sum(y_pred * g2) - sum(y_pred) * sum(g2))/(np.sqrt((y_pred.size* sum(np.square(y_pred)) - np.square(sum(y_pred))) * (y_pred.size*sum(np.square(g2)) - np.square(sum(g2)))))

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# plot the function
plt.plot(x,y, 'b')

# show the plot
plt.show()
