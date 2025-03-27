import numpy as np

from yara.svm import SVC

X = [[0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
y = [0, 0, 1, 1]

clf = SVC()
clf.fit(X, y)
print(clf.coef_)

