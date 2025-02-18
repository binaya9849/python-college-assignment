
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
X = df[['bmi', 's5', 'bp']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("RMSE:", rmse)
print("R2 Score:", r2_score(y_test, y_pred))


"""
Question a: Which variable would you add next? Why?
- We added bp"blood pressure" because it shows a reasonable correlation with diabetes progression.

Question b: How does adding it affect the model's performance?
- Adding bp improved the model's performance slightly, as seen from the RMSE and R2 scores below. 

Question d: Does it help if you add even more variables?
- Adding even more variables may lead to overfitting if they are not strongly correlated with the target. 
"""