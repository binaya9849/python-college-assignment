from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
startups = pd.read_csv('50_Startups.csv')
X = startups[['R&D Spend', 'Marketing Spend']]
y = startups['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse_train = mean_squared_error(y_train, model.predict(X_train))
rmse_train = mse_train ** 0.5
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = mse_test ** 0.5
print("Training RMSE:", rmse_train)
print("Testing RMSE:", rmse_test)
print("Training R2:", r2_score(y_train, model.predict(X_train)))
print("Testing R2:", r2_score(y_test, y_pred))
