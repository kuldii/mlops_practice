import pandas as pd

train_data = pd.read_csv("datasets/train/preprocessed_train_data.csv")
test_data = pd.read_csv("datasets/test/preprocessed_test_data.csv")

# Split features and target
y_train = train_data["Survived"]
X_train = train_data.drop(columns="Survived")
y_test = test_data["Survived"]
X_test = test_data.drop(columns="Survived")