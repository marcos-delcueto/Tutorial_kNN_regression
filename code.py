#!/usr/bin/env python3
# Marcos del Cueto
### 1 imports
import math
import matplotlib.pyplot as plt
import random
### 2 imports
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
import numpy as np
from matplotlib.ticker import (MultipleLocator)

### 1) Generate data
list_x = []
list_y = []
random.seed(19)
#for i in range(10,57,3):
for i in np.arange(5, 7, 0.2):
    #x = i*0.13
    x = i
    y = math.exp(x)

    #delta_y = random.uniform(-y*0.1,y*0.1)
    #y = y+delta_y

    list_x.append(x)
    list_y.append(y)
    #print("%.2f, %.6f" %(x, y))
list_x = np.array(list_x).reshape(-1, 1)
list_y = np.array(list_y)
basic_y = [math.exp(x) for x in list_x]
plt.plot(list_x,basic_y,color='C0',linestyle='dashed',linewidth=1)
plt.scatter(list_x, list_y,color='C0')
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$f(x)$',fontsize=15)

file_name='points.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()

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
        # scale data MISSING
        knn = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform')
        pred = knn.fit(X_train, y_train).predict(X_test)
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
plt.plot(list_x,basic_y,color='C0',linestyle='dashed',linewidth=1)
plt.scatter(list_x, list_y,color='C0')
plt.scatter(x_pred, y_pred,color='C1')
file_name='points_loo.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()



### 3) Do kNN regression
X_train = np.array(list_x).reshape(-1, 1)
y_train = np.array(list_y)
X_test = np.arange(5.0, 7.0, 0.01).reshape(-1, 1)
#print('X_train', X_train)
#print('y_train', y_train)
#print('X_test', X_test)

#knn = neighbors.KNeighborsRegressor(n_neighbors=best_k, weights='uniform')
#file_name='points2.png'
knn = neighbors.KNeighborsRegressor(n_neighbors=best_k, weights='distance')
file_name='points3.png'
y_test = knn.fit(X_train, y_train).predict(X_test)

axes = plt.gca()
axes.xaxis.set_minor_locator(MultipleLocator(0.1))
basic_y = [math.exp(x) for x in X_test]
plt.plot(X_test,basic_y,color='C0',linestyle='dashed',linewidth=1)
plt.scatter(list_x, list_y,color='C0')
plt.plot(X_test, y_test,color='C1')
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$f(x)$',fontsize=15)

plt.savefig(file_name,format='png',dpi=600)
plt.close()

