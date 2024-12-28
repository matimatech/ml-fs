import numpy as np

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

n = len(X)

from linear_model import LinearRegression as MLR

clf = MLR()
clf.fit(X, y)
print(f"beta 0 : {clf.intercept_} coef = {clf.coef_}")


#
#
# X_ = np.column_stack((np.ones(n), X))
# print(X_)
#
# beta = np.linalg.inv(X_.T @ X_) @ X_.T @ y
# print(beta)
