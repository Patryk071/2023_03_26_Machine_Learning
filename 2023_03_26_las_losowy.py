import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("iris.csv")
print(df["class"].value_counts())

species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)
print(df)

X = df[ ['sepallength', 'sepalwidth'] ]    #najpierw te kontrowersje
y = df.class_value

X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_test, y_test)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test)) ))