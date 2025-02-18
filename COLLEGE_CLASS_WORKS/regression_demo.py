import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#load dataset
data = load_diabetes(as_frame=True)
print (data.keys())
print (data.DESCR)
df = data["frame"]
print (df)

#Visualizing the traget variable distribution
plt.hist(df["target"],25)
plt.xlabel ("target")
plt. show()

#Correlation heatmap
sns.heatmap(data=df.corr().round(2),annot=True)
plt.show()

#Scatter plots
plt .subplot (1,2,1)
plt.scatter(df['bmi'], df["target"])
plt.xlabel ( 'bmi')
plt.ylabel ('target')
plt.subplot (1,2,2)
plt.scatter(df['s5'],df['target'])
plt.xlabel ('s5')
plt.ylabel ('target')
plt. show()

#Selecting feature for regression
X = pd.DataFrame(df[['bmi', 's5']], columns=['bmi', 's5'])
y = df[ 'target' ]
print (X)
print (y)

#Split data into train and train sets(80% training , testing 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.2) #random_state = 5 : just randoms but every tim it uses it is consistent meaning same output for every time, which is important 
print(X_train.shape)
print (X_test.shape)

#Train the regression model
lm = LinearRegression()
lm.fit(X_train,y_train)

#Predictions on test and performance ebulaltion
y_train_predict = lm.predict(X_train)   #lm.predict(X_train) : predicts values for training
rmse = (np.sqrt(mean_squared_error(y_train,y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
print (f"RMSE = {rmse}, R2 = {r2}")

#Predictions on test and performance ebulaltion
y_test_predict = lm.predict(X_test) #lm.predict(X_test) : predicts values for training
rmse_test = (np.sqrt(mean_squared_error(y_test,y_test_predict)))
r2_test = r2_score(y_test,y_test_predict)
print(f'RMSE (test) = {rmse_test},R2 (test) = {r2_test}')

#Display prediction
print(X_test,y_test_predict)