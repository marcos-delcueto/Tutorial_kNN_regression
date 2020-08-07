#!/usr/bin/env python3
# Marcos del Cueto
# Import libraries
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from matplotlib.ticker import (MultipleLocator)
# Set optimum k optimized previously
best_k = 2
# Initialize lists
list_x = []
list_y = []
# Generate dataset as 10 points from x=5 to x=6.8, with y=exp(x)
for x in np.arange(5, 7, 0.2):
    y = math.exp(x)
    list_x.append(x)
    list_y.append(y)
# Transform lists to numpy arrays
list_x = np.array(list_x).reshape(-1, 1)
list_y = np.array(list_y)
# Create arrays with function y=exp(x)
function_x = np.arange(3.9, 7.81, 0.01)
function_y = [math.exp(x) for x in function_x]
# Assign train data as all 10 points in dataset
X_train = np.array(list_x).reshape(-1, 1)
y_train = np.array(list_y)
# Assign prediction X to intermediate x values 
X_pred = np.arange(4.0, 7.81, 0.01).reshape(-1, 1)
# Scale data (not needed here, since just 1 descriptor, but it is good practice)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_pred_scaled = scaler.transform(X_pred)
# kNN regressor
knn = neighbors.KNeighborsRegressor(n_neighbors=best_k, weights='uniform')
label_kNN = "k-NN regression (uniform)"
# Train kNN model and predict values for X_pred_scaled
y_pred = knn.fit(X_train_scaled, y_train).predict(X_pred_scaled)
# Plot points in dataset plus dashed line with function
plt.plot(function_x,function_y,color='C0',linestyle='dashed',linewidth=1)
plt.scatter(list_x, list_y,color='C0',label='Training points')
# Plot predicted values with kNN regressor
plt.plot(X_pred, y_pred,color='C1',label=label_kNN)
# Plot legend
plt.legend()
# Set axis labels
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$y$',fontsize=15)
# Set axis ticks and limits
plt.xticks(np.arange(4,8,0.4))
plt.xlim(3.92,7.88)
plt.ylim(0,1200)
# Set minor ticks
axes = plt.gca()
axes.xaxis.set_minor_locator(MultipleLocator(0.05))
# Save plot into png
file_name='Figure6_1.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()
