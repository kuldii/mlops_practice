import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

test_data = pd.read_csv("datasets/test/preprocessed_test_data.csv")
y_test = test_data["Survived"]
X_test = test_data.drop(columns="Survived")

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Test the model
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates of the positive class

# Predicted classes based on the probability threshold of 0.5
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Print classification metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"AUC: {auc}")