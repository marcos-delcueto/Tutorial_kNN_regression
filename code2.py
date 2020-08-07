#!/usr/bin/env python3
# Marcos del Cueto
# Import libraries
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from matplotlib.ticker import (MultipleLocator)
# Initialize list
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
# Initialize grid search values
best_rmse = 0
best_k = 0
# Select which k neighbors will be studied in grid
possible_k = [1,2,3,4,5,6,7,8,9]
# Start loop for grid search
for k in possible_k:
    # Initialize lists with predicted values
    y_pred = []
    x_pred = []
    # Start LOO loop 
    for train_index, test_index in LeaveOneOut().split(list_x):
        # Assign train and test data
        X_train, X_test = list_x[train_index], list_x[test_index]
        y_train, y_test = list_y[train_index], list_y[test_index]
        # Scale data (not needed here, since just 1 descriptor, but it is good practice)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # kNN regressor
        knn = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance')
        # Train kNN model and predict values for X_test_scaled
        pred = knn.fit(X_train_scaled, y_train).predict(X_test_scaled)
        # Assign predicted value in each LOO step
        y_pred.append(pred)
        x_pred.append(X_test)
    # Calculate rmse between actual y values and predicted values
    mse = mean_squared_error(y_pred,list_y)
    rmse = np.sqrt(mse)
    # Print k and rmse of each grid point
    print('k: %i, rmse: %.2f' %(k, rmse))
    # If this value is better than previous one (or it is first step), update best value
    if rmse < best_rmse or k==possible_k[0]:
        best_rmse = rmse
        best_k = k
# Print final best k value and rmse
print("Optimum: kNN, k=%i, rmse: %.2f" %(best_k,best_rmse))
