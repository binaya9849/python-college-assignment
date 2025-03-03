import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


"""1) Read in the CSV file using pandas. Pay attention to the file delimeter.
Inspect the resulting dataframe with respect to the column names and the variable types."""
df = pd.read_csv("bank.csv", sep=";", encoding="latin1")


"""2) Pick data from the following columns to a second dataframe 'df2': y, job, marital,
default, housing, poutcome."""

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]



"""3) Convert categorical variables to dummy numerical values using the command
df3 = pd.get_dummies(df2,columns=['job','marital','default','housing','poutcome'])"""

df3 = pd.get_dummies(df2,columns=['job','marital','default','housing','poutcome'])

df3['y'] = df3['y'].map({'yes': 1, 'no': 0})

"""4) Produce a heat map of correlation coefficients for all variables in df3."""
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Heatmap of Correlation Coefficients') 
sns.heatmap(df3.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.show()
print("-------------------------------------------------------------------------------")
print(" Describe the amount of correlation between the variables in your own words.")
print("-----------------------------------------------------------------------")
print("The heatmap shows how different factors are related also campaign outcomes (poutcome) and housing status, may have a stronger link to whether a customer subscribes (y) while Others have little to no impact. This helps us see which factors matter most for predictions.")
print("--------------------------------------------------------------------------------------")
"""5) Select the column called 'y' of df3 as the target variable y, and all the remaining
columns for the explanatory variables X."""
X = df3.drop(columns=['y'])
y = df3['y']

"""6) Split the dataset into training and testing sets with 75/25 ratio."""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

"""7) Setup a logistic regression model, train it with training data and predict on testing data.

Logistic model

"""

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)
conf_matrix_log = confusion_matrix(y_test, y_pred_log)


"""8) Print the confusion matrix (or use heat map if you want) and accuracy score for the
logistic regression model."""

print("Logistic Regression Accuracy:", accuracy_log)
print("-----------------------------------------------------------------------")
print("Logistic Regression Confusion Matrix:\n", conf_matrix_log)
print("-----------------------------------------------------------------------")

"""9) Repeat steps 7 and 8 for k-nearest neighbors model. Use k=3, for example, or
experiment with different values.    


KNN model

"""
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

print("KNN Accuracy:", accuracy_knn)
print("-----------------------------------------------------------------------")
print("KNN Confusion Matrix:\n", conf_matrix_knn)
print("-----------------------------------------------------------------------")

"""10) Compare the results between the two models."""

def compare_models(y_test, y_pred_log, y_pred_knn):
    print("Logistic Regression Classification Report:")
    print("-----------------------------------------------------------------------")
    print(classification_report(y_test, y_pred_log))
    print("-----------------------------------------------------------------------")
    print("\nK-Nearest Neighbors Classification Report:")
    print("-----------------------------------------------------------------------")
    print(classification_report(y_test, y_pred_knn))

compare_models(y_test, y_pred_log, y_pred_knn)


print("-----------------------------------------------------------------------")
print("From the above results we see that how well each model performs in prediciting the outcomes Logistic Regression works better when the data has clear pattern and has balance performance in accuracy and precision while KNN or K-Nearest Neighbours can handle more complex patterns but doesn't perform well if the data is not properly scaled or if chosen number of neighboirs. This differnece proves which model is better for what kind of operation ")

print("-----------------------------------------------------------------------")