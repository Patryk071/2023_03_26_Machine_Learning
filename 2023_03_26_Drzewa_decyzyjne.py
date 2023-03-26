import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
print(["class"].value_counts())
species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}

df["class_value"] = df["class"].map(species)