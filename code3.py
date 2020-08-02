#!/usr/bin/env python3
# Marcos del Cueto
import math
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from matplotlib.ticker import (MultipleLocator)

### 1) Generate data
list_x = []
list_y = []
random.seed(19)
for i in np.arange(5, 7, 0.2):
    x = i
    y = math.exp(x)
    list_x.append(x)
    list_y.append(y)
list_x = np.array(list_x).reshape(-1, 1)
list_y = np.array(list_y)
#basic_y = [math.exp(x) for x in list_x]
basic_x = np.arange(4.9, 7.0, 0.01)
basic_y = [math.exp(x) for x in basic_x]

best_k = 2

### 3) Do kNN regression
X_train = np.array(list_x).reshape(-1, 1)
y_train = np.array(list_y)
X_test = np.arange(5.0, 6.81, 0.01).reshape(-1, 1)
#print('X_train', X_train)
#print('y_train', y_train)
#print('X_test', X_test)
# scale data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
################################################################################
############# UNIFORM #############
knn = neighbors.KNeighborsRegressor(n_neighbors=best_k, weights='uniform')
file_name='Fig3_1.png'
label_kNN = "k-NN regression (uniform)"
############# DISTANCE #############
#knn = neighbors.KNeighborsRegressor(n_neighbors=best_k, weights='distance')
#file_name='Fig3_2.png'
#label_kNN = "k-NN regression (distance)"
################################################################################
# kNN regression
y_test = knn.fit(X_train_scaled, y_train).predict(X_test_scaled)
 # Plot graph
plt.plot(basic_x,basic_y,color='C0',linestyle='dashed',linewidth=1)
plt.scatter(list_x, list_y,color='C0',label='$x_n$')
plt.plot(X_test, y_test,color='C1',label=label_kNN)
plt.legend()
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$f(x)$',fontsize=15)
plt.xticks(np.arange(5,7,0.2))
plt.xlim(4.92,6.88)
plt.ylim(100,1000)
axes = plt.gca()
axes.xaxis.set_minor_locator(MultipleLocator(0.05))
# Save plot into png
plt.savefig(file_name,format='png',dpi=600)
plt.close()

