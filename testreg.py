import numpy as np

from ml_from_scratch.linear_model import Lasso
from ml_from_scratch.linear_model import LinearRegression as MLR
from ml_from_scratch.linear_model import LogisticRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
print(y)

n = len(X)


# clf = MLR()
# clf.fit(X, y)
# print(f"beta 0 : {clf.intercept_} coef = {clf.coef_}")

X = np.array([[0, 0], [1, 1], [0, 1]])
y = np.array([0, 1, 1])

mdl = Lasso()
mdl.fit(X, y)
mdl.intercept_


mdl = LogisticRegression()

mdl.fit(X, y)
print(mdl.intercept_, mdl.coef_)

print(mdl.predict_proba([0, 0]))


#
#
# X_ = np.column_stack((np.ones(n), X))
# print(X_)
#
# beta = np.linalg.inv(X_.T @ X_) @ X_.T @ y
# print(beta)
