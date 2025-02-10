import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('weight-height.csv')
print("Sample data from the dataset:")
print(df.head())
plt.scatter(df['Height'], df['Weight'], alpha=0.5 , color='red')
plt.title('Height vs Weight Scatter Plot')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.show()
X = df[['Height']]
y = df['Weight']  
print("Simple linear regression model")
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(df['Height'], df['Weight'], alpha=0.5, label='Actual Data')
plt.plot(df['Height'], y_pred, color='red', label='Regression Line')
plt.title('Height and Weight with Regression Line')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.show()
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
print("Both visually (through the regression model) and through the data itself, there is a strong relationship between height and weight. As we can see from the scatter plot along with the regression line, there is a clear positive linear trend present. The red line that we have plotted over our points indicates that the data points are being followed quite closely by a linear line, which suggests a good fit.")
print("The overall R2 value is 0.855, meaning that ~85.5% of the variance in weight can be accounted for by height, a very good outcome. We have an RMSE of 12.22 pounds, which means that on average the predictions made by the model are off by approximately 12 pounds, which is acceptable given the variance of the data. The model seems to capture the variation in weight to a good extent but there may yet be other variables affecting weight as height alone can't explain it all.")