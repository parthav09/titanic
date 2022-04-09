import numpy as np
import pandas as pd
import config
from sklearn import preprocessing

train = pd.read_csv(config.TRAIN)
test = pd.read_csv(config.TEST)

print(train.shape)
print(test.shape)
dataset = [train, test]


def replaceNan(cols):
    Age = cols[0]
    Pclass = cols[1]
    if (pd.isnull(Age)):
        if Pclass == 1:
            return 37
        if Pclass == 2:
            return 31
        if Pclass == 3:
            return 25
    else:
        return Age


test['Fare'] = test["Fare"].replace(np.nan, test['Fare'].median())

for data in dataset:
    data['Age'] = data[['Age', 'Pclass']].apply(replaceNan, axis=1).reset_index(drop=True)
    data.drop(columns=["Cabin"], axis=1, inplace=True)

sex = pd.get_dummies(train['Sex'], drop_first=True)
embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embarked], axis=1).reset_index(drop=True)
train.drop(['Sex', 'Name', 'Embarked', 'Ticket'], inplace=True, axis=1)

sex = pd.get_dummies(test['Sex'], drop_first=True)
embarked = pd.get_dummies(test['Embarked'], drop_first=True)
test = pd.concat([test, sex, embarked], axis=1).reset_index(drop=True)
test.drop(['Sex', 'Name', 'Embarked', 'Ticket'], inplace=True, axis=1)

print(train.shape)
print(test.shape)

train.to_csv('../input/train_cleaned.csv',  index=False)
test.to_csv('../input/test_clearned.csv', index=False)
