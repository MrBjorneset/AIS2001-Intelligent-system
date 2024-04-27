import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset
data = pd.read_csv('Part_E\\data\\solar_weather.csv')

# Exclude the first column since it's a string
data = data.iloc[:, 1:]

# Separate features (X) and target variable (y)
X = data.drop('weather_type', axis=1)  
y = data['weather_type']

# Split the dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate total number of data objects
total_samples = len(data)
num_train_samples = len(X_train)
num_test_samples = len(X_test)

# Count occurrences of each class in the training and test sets
class_counts_train = y_train.value_counts()
class_counts_test = y_test.value_counts()

# Define function to perform experiments for Logistic Regression
def logistic_regression_experiments(X_train, X_test, y_train, y_test):
    # Experiment 1
    lr_model_1 = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')  
    lr_model_1.fit(X_train, y_train)
    y_pred_lr_1 = lr_model_1.predict(X_test)
    acc_lr_1 = accuracy_score(y_test, y_pred_lr_1)

    # Experiment 2
    lr_model_2 = LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs')  
    lr_model_2.fit(X_train, y_train)
    y_pred_lr_2 = lr_model_2.predict(X_test)
    acc_lr_2 = accuracy_score(y_test, y_pred_lr_2)

    # Experiment 3
    lr_model_3 = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs') 
    lr_model_3.fit(X_train, y_train)
    y_pred_lr_3 = lr_model_3.predict(X_test)
    acc_lr_3 = accuracy_score(y_test, y_pred_lr_3)

    return [acc_lr_1, acc_lr_2, acc_lr_3]

# Define function to perform experiments for Decision Trees
def decision_tree_experiments(X_train, X_test, y_train, y_test):
    # Experiment 1
    dt_model_1 = DecisionTreeClassifier(max_depth=1)  
    dt_model_1.fit(X_train, y_train)
    y_pred_dt_1 = dt_model_1.predict(X_test)
    acc_dt_1 = accuracy_score(y_test, y_pred_dt_1)

    # Experiment 2
    dt_model_2 = DecisionTreeClassifier(max_depth=2)  
    dt_model_2.fit(X_train, y_train)
    y_pred_dt_2 = dt_model_2.predict(X_test)
    acc_dt_2 = accuracy_score(y_test, y_pred_dt_2)

    # Experiment 3
    dt_model_3 = DecisionTreeClassifier(max_depth=5)  
    dt_model_3.fit(X_train, y_train)
    y_pred_dt_3 = dt_model_3.predict(X_test)
    acc_dt_3 = accuracy_score(y_test, y_pred_dt_3)

    return [acc_dt_1, acc_dt_2, acc_dt_3]


def main():
    print("Total number of data objects:")
    print("Training set: {} samples ({}%)".format(num_train_samples, (num_train_samples / total_samples) * 100))
    print("Test set: {} samples ({}%)".format(num_test_samples, (num_test_samples / total_samples) * 100))

    # Print class distribution in training and test sets
    print("\nClass Distribution:")
    print("Training Set:")
    print(class_counts_train)
    print("Percentage:")
    print((class_counts_train / num_train_samples) * 100)

    print("\nTest Set:")
    print(class_counts_test)
    print("Percentage:")
    print((class_counts_test / num_test_samples) * 100)

    # Logistic Regression experiments
    lr_results = logistic_regression_experiments(X_train, X_test, y_train, y_test)
    print("\nLogistic Regression Results:")
    for i, acc in enumerate(lr_results):
        print(f"Experiment {i+1}: Accuracy = {acc}")

    # Decision Tree experiments
    dt_results = decision_tree_experiments(X_train, X_test, y_train, y_test)
    print("\nDecision Tree Results:")
    for i, acc in enumerate(dt_results):
        print(f"Experiment {i+1}: Accuracy = {acc}")

if __name__ == "__main__":
    main()
