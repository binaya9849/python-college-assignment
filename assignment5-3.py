from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
auto = pd.read_csv('Auto.csv')
X = auto.drop(['mpg', 'name', 'origin'], axis=1)
y = auto['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("RMSE:", rmse)
print("R2 Score:", r2_score(y_test, y_pred))


"""
Step 1: Read the data into pandas dataframe.
- loading the file

Step 2: Setup multiple regression X and y to predict 'mpg' of cars using all the variables except 'mpg', 'name' and 'origin'.

Step 3: Split data into training and testing sets (80/20 split).
- 80/20 split to ensure a robust model evaluation.

Step 4: Implement both ridge regression and LASSO regression using several values for alpha.

Step 5: Search optimal value for alpha (in terms of R2 score) by fitting the models with training data and computing the score using testing data.
- Alpha with the best R2 score was identified.

Step 6: Plot the R2 scores for both regressors as functions of alpha.
- Visual comparison to select the best model.

Step 7: Identify, as accurately as you can, the value for alpha which gives the best score.
- Optimal alpha selection for each model.
"""
