import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

# Step 1: Read the CSV file
data = pd.read_csv('data/netflix.csv')

# Step 2: Preprocessing
# Assuming the last column is the target variable
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Handle categorical data (if any)
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Choose 2 supervised learning methods and an ANN
# Example 1: Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Example 2: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Example 3: Artificial Neural Network (ANN)
ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
ann_model.fit(X_train, y_train)
ann_predictions = ann_model.predict(X_test)

# Step 4: Evaluate the models
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

svm_scores = evaluate_model(y_test, svm_predictions)
rf_scores = evaluate_model(y_test, rf_predictions)
ann_scores = evaluate_model(y_test, ann_predictions)

print("Support Vector Machine (SVM) Performance:")
print("Accuracy:", svm_scores[0])
print("Precision:", svm_scores[1])
print("Recall:", svm_scores[2])
print("F1 Score:", svm_scores[3])
print()

print("Random Forest Classifier Performance:")
print("Accuracy:", rf_scores[0])
print("Precision:", rf_scores[1])
print("Recall:", rf_scores[2])
print("F1 Score:", rf_scores[3])
print()

print("Artificial Neural Network (ANN) Performance:")
print("Accuracy:", ann_scores[0])
print("Precision:", ann_scores[1])
print("Recall:", ann_scores[2])
print("F1 Score:", ann_scores[3])
