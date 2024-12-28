import numpy as np

from neighbors import KNeighborsRegressor as KNR

# from neighbors._base import NearestNeighbor


X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

clf = KNR()
clf.fit(X, y)
res = clf.predict(X[:2])
print(res
      
