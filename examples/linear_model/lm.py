import numpy as np
from yara.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

clf = LinearRegression()

clf.fit(X, y)
print(clf.coef_)
