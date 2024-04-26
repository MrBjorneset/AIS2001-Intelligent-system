import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset from CSV file
df = pd.read_csv('Part_E\\data\\solar_weather.csv')

# Assuming the last column contains the target variable
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Trees
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
train_pred_dt = dt_classifier.predict(X_train)
test_pred_dt = dt_classifier.predict(X_test)

# SVM
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
train_pred_svm = svm_classifier.predict(X_train)
test_pred_svm = svm_classifier.predict(X_test)

# Evaluate Decision Trees
train_accuracy_dt = accuracy_score(y_train, train_pred_dt)
test_accuracy_dt = accuracy_score(y_test, test_pred_dt)
print("Decision Trees - Training Accuracy:", train_accuracy_dt)
print("Decision Trees - Test Accuracy:", test_accuracy_dt)

# Evaluate SVM
train_accuracy_svm = accuracy_score(y_train, train_pred_svm)
test_accuracy_svm = accuracy_score(y_test, test_pred_svm)
print("SVM - Training Accuracy:", train_accuracy_svm)
print("SVM - Test Accuracy:", test_accuracy_svm)
