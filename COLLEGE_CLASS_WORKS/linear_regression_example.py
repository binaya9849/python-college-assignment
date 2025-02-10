import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model
from sklearn import metrics
my_data = np.genfromtxt("linreg_data.csv",delimiter=",")
print(my_data)
xp = my_data[:,0]
yp = my_data[:,1]
print(xp)
xp = xp.reshape(-1,1)
yp = yp.reshape(-1,1)
print("xp =",xp)

regr = linear_model.LinearRegression()

# training/fitting model with training data
regr.fit(xp,yp)
print("slope b =", regr.coef_)
print("y-intercept a=",regr.intercept_)

#calculating prediction
xval = np.full((1,1),0.5)
yval = regr.predict(xval)
print(yval)
yhat = regr.predict(xp)
print("yhat =",yhat)

#Evaluation
print("mean Absolute Error (MAE):",metrics.mean_absolute_error(yp, yhat))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(yp, yhat))
print("Root mean squared Error (RMSE) :",np.sqrt(metrics.mean_squared_error(yp, yhat)))
print("R2 value:", metrics.r2_score(yp,yhat))