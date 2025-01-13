import numpy as np
from ml_from_scratch.linear_model import Lasso
from ml_from_scratch.linear_model import LinearRegression as MLR
from ml_from_scratch.linear_model import LogisticRegression, Ridge

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
print(y)

mdl1 = Ridge()
mdl1.fit(X, y)
print(mdl1.coef_)
print(mdl1.intercept_)
print(mdl1.predict([1, 2]))

