#!/usr/bin/env python3
# Marcos del Cueto
# Import libraries
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import numpy as np
# Initialize lists
list_x = []
list_y = []
# Generate dataset as 10 points from x=5 to x=6.8, with y=exp(x)
for x in np.arange(5, 7, 0.2):
    y = math.exp(x)
    list_x.append(x)
    list_y.append(y)
    print("%.2f, %.2f" %(x, y))
# Transform lists to numpy arrays
list_x = np.array(list_x).reshape(-1, 1)
list_y = np.array(list_y)
# Create arrays with function y=exp(x)
function_x = np.arange(4.9, 7.0, 0.01)
function_y = [math.exp(x) for x in function_x]
# Plot points in dataset plus dashed line with function
plt.plot(function_x,function_y,color='C0',linestyle='dashed',linewidth=1)
plt.scatter(list_x, list_y,color='C0')
# Set axis labels
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$y$',fontsize=15)
# Set axis ticks and limits
plt.xticks(np.arange(5,7,0.2))
plt.xlim(4.92,6.88)
plt.ylim(100,1000)
# Set minor ticks
axes = plt.gca()
axes.xaxis.set_minor_locator(MultipleLocator(0.05))
# Save plot into Figure_1.png
file_name='Figure_1.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()
