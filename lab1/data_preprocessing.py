import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("datasets/train/train_data.csv")
test_data = pd.read_csv("datasets/test/test_data.csv")

# Remove uneccesary column
train_data.drop(columns="PassengerId", inplace=True)
test_data.drop(columns="PassengerId", inplace=True)

# Set into numerical
train_data["Sex"] = train_data["Sex"].map(lambda x: 1 if x == "male" else 0)
test_data["Sex"] = test_data["Sex"].map(lambda x: 1 if x == "male" else 0)
train_data['Ticket'] = train_data['Ticket'].map(lambda x: len(x))
test_data['Ticket'] = test_data['Ticket'].map(lambda x: len(x))

# Handle null value
train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
train_data["Embarked"].fillna(train_data["Embarked"].mode()[0], inplace=True)
train_data["Cabin"] = train_data["Cabin"].map(lambda x: "-" if pd.isna(x) else x[0])
test_data["Cabin"] = test_data["Cabin"].map(lambda x: "-" if pd.isna(x) else x[0])

# Drop null in test dataset
test_data.dropna(subset="Age", inplace=True)

# Categorize the name by prefix
def process_name(name):
  return name.split(".")[0].split(",")[1].strip()
train_data['Name'] = train_data['Name'].map(process_name)
test_data['Name'] = test_data['Name'].map(process_name)

# Factorize
train_data[["Name", "Cabin", "Embarked"]] = train_data[["Name", "Cabin", "Embarked"]].apply(lambda x: pd.factorize(x)[0])
test_data[["Name", "Cabin", "Embarked"]] = test_data[["Name", "Cabin", "Embarked"]].apply(lambda x: pd.factorize(x)[0])

# Scaling
scale = StandardScaler()
train_data[["Age", "SibSp", "Parch", "Fare"]] = scale.fit_transform(train_data[["Age", "SibSp", "Parch", "Fare"]])
test_data[["Age", "SibSp", "Parch", "Fare"]] = scale.fit_transform(test_data[["Age", "SibSp", "Parch", "Fare"]])

# Save dataset
train_data.to_csv('datasets/train/preprocessed_train_data.csv', index=False)
test_data.to_csv('datasets/test/preprocessed_test_data.csv', index=False)