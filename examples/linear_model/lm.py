import numpy as np
import pandas as pd
from yara.linear_model import LinearRegression
from yara.metrics import mean_squared_error

data = pd.read_csv("data/auto.csv")
X = data[["cylinders", "horsepower"]]
y = data["mpg"]
print(X.shape)

clf = LinearRegression()

clf.fit(X, y)
y_pred = clf.predict(X)
mse = mean_squared_error(y_pred=y_pred, y_actual=y)
print(f"MSE {mse}")
print(clf.coef_)