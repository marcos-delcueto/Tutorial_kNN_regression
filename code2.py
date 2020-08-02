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
basic_x = np.arange(4.9, 7.0, 0.01)
basic_y = [math.exp(x) for x in basic_x]
### 2) Leave one out
best_rmse = 0
best_k = 0
possible_k = [2]
for k in possible_k:
    y_pred = []
    x_pred = []
    X = np.array(list_x).reshape(-1, 1)
    validation=LeaveOneOut().split(X)
    for train_index, test_index in validation:
        X_train, X_test = list_x[train_index], list_x[test_index]
        y_train, y_test = list_y[train_index], list_y[test_index]
        # scale data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        knn = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform')
        pred = knn.fit(X_train_scaled, y_train).predict(X_test_scaled)
        #print('NEW y_test,pred,error', pred, y_test,abs(pred-y_test))
        y_pred.append(pred)
        x_pred.append(X_test)
    #print('x_pred', x_pred)
    #print('y_pred', y_pred)
    mse = mean_squared_error(y_pred,list_y)
    rmse = np.sqrt(mse)
    print('k: %i, RMSE: %f' %(k, rmse))
    if rmse < best_rmse or k==possible_k[0]:
        best_rmse = rmse
        best_k = k
print('BEST:')
print("kNN, k=%i, RMSE: %f" %(best_k,best_rmse))
# Plot graph
plt.plot(basic_x,basic_y,color='C0',linestyle='dashed',linewidth=1)
plt.scatter(list_x, list_y,color='C0')
plt.scatter(x_pred[0:3], y_pred[0:3],color='C1')
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$f(x)$',fontsize=15)
plt.xticks(np.arange(5,7,0.2))
plt.xlim(4.92,6.88)
plt.ylim(100,1000)
axes = plt.gca()
axes.xaxis.set_minor_locator(MultipleLocator(0.05))
# Save plot into png
file_name='Fig2.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()
