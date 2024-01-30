import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

for i in range(len(conf_matrix)):
    tp = conf_matrix[i, i]
    fp = sum(conf_matrix[:, i]) - tp
    fn = sum(conf_matrix[i, :]) - tp
    tn = sum(sum(conf_matrix)) - tp - fp - fn
    print(f"\nClass {i}:\nTrue Positives: {tp}\nTrue Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\n")
