import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('weight-height.csv')

# Display first few rows of the data
print(df.head())

# Scatter plot of Height vs Weight
plt.scatter(df['Height'], df['Weight'], alpha=0.5)
plt.title('Height vs Weight')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.show()

# Prepare data for polynomial regression
X = df[['Height']]  # Independent variable (Height)
y = df['Weight']    # Dependent variable (Weight)

# Step 1: Transform the data to allow polynomial regression (degree 2 for quadratic)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Step 2: Fit a linear regression model to the transformed data
model = LinearRegression()
model.fit(X_poly, y)

# Step 3: Make predictions using the polynomial regression model
y_pred = model.predict(X_poly)

# Step 4: Plot the results
plt.scatter(df['Height'], df['Weight'], alpha=0.5, label='Actual Data')
plt.plot(df['Height'], y_pred, color='red', label='Polynomial Regression Line')
plt.title('Height vs Weight with Polynomial Regression Line')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.legend()
plt.show()

# Step 5: Calculate RMSE and R2 score
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Print the results
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
