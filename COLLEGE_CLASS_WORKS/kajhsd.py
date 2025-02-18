from sklearn.model_selection import train_test_split

# Example dataset
X = [1, 2, 3, 4, 5, 6]
y = [10, 20, 30, 40, 50, 60]

# Split without random_state

X_train, X_test = train_test_split(X,  test_size=0.2, random_state=5)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)
