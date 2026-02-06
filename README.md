# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.The dataset is loaded and preprocessed using encoding and feature scaling.

 2.The features and target variable are separated and split into training and testing sets.

 3.Logistic Regression is implemented from scratch and trained using Gradient Descent.
 
 4.The trained model is evaluated using accuracy, confusion matrix, and classification report. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: NITHYASRI T
RegisterNumber:  212225220068
*/

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:/Users/acer/Downloads/Placement_Data (1).csv")
data.drop("sl_no", axis=1, inplace=True)

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
data = pd.get_dummies(data, drop_first=True)

X = data.drop('status', axis=1).values
y = data['status'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test  = (X_test  - X_test.mean(axis=0))  / X_test.std(axis=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.1      
epochs = 3000
for _ in range(epochs):
    linear = np.dot(X_train, weights) + bias
    y_pred = sigmoid(linear)

    dw = (1 / len(y_train)) * np.dot(X_train.T, (y_pred - y_train))
    db = (1 / len(y_train)) * np.sum(y_pred - y_train)

    weights -= learning_rate * dw
    bias -= learning_rate * db
def predict(X):
    linear = np.dot(X, weights) + bias
    return np.where(sigmoid(linear) >= 0.5, 1, 0)

y_predicted = predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predicted) * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predicted))

print("\nClassification Report:")
print(classification_report(y_test, y_predicted))
```

## Output:
<img width="750" height="343" alt="Screenshot 2026-02-06 112907" src="https://github.com/user-attachments/assets/81ee0838-0117-41dc-bb32-94e2e5fbcd80" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

