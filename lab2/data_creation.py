import os
import pandas as pd
from sklearn.model_selection import train_test_split

if not os.path.exists('datasets/train'):
    os.makedirs('datasets/train')
if not os.path.exists('datasets/test'):
    os.makedirs('datasets/test')
    

# Load the data from internet -> Titanic
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Survived']), df['Survived'], test_size=0.2)

train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('datasets/train/train_data.csv', index=False)

test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('datasets/test/test_data.csv', index=False)