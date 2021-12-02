# Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score 
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


df = pd.read_excel("merc.xlsx")

print(df)

y = df["price"].values
X = df.drop(["price", "transmission"], axis=1).values

# train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train", X_train.shape)
print("X_test", X_test.shape)

#Create Model
pipe = Pipeline([("scaler", StandardScaler()), ("rf_regr", RandomForestRegressor())])

# Train Model
pipe.fit(X_train, y_train)

scores = cross_val_score(pipe, X, y)

print(scores)
print("mean", scores.mean())
print("std", scores.std())

#Save model
pickle.dump(pipe, open("model.pkl", "wb"))

# Load Model
model = pickle.load(open("model.pkl", "rb"))

