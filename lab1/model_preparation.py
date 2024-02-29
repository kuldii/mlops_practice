import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv("datasets/train/preprocessed_train_data.csv")

# Split features and target
y_train = train_data["Survived"]
X_train = train_data.drop(columns="Survived")

# Training the model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(lr, f)