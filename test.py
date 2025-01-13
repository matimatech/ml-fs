import numpy as np

from ml_from_scratch.svm import SVC

# from neighbors import KNeighborsRegressor as KNR


# from neighbors._base import NearestNeighbor

X = [[0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
y = [0, 0, 1, 1]

clf = SVC()
clf.fit(X, y)
print(clf.coef_)


# clf = KNR()
# clf.fit(X, y)
# res = clf.predict(X[:2])
# print(res)
