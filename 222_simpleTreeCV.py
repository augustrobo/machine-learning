##modify the code from Machine Learning in Python
##choose the best tree depth
##10-fold cv

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import KFold

n_points = 1000
x = np.arange(- 0.5, 0.5, 1/n_points)[:, np.newaxis]

##y (labels) has random noises
y = x.ravel() + np.random.normal(0, 0.1, n_points)

##fit trees with different values of depth
depth = np.arange(1,8)

MSE = []
##ten-fold cross validation
kf = KFold(n_points, n_folds = 10)

##outer loop: tree depth
##inner loop: cross-validation
for d in depth:
    mse = 0
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tree_md = DecisionTreeRegressor(max_depth = d)
        tree_md.fit(x_train, y_train)
        y_hat = tree_md.predict(x_test)
        error = np.array(y_test) - np.array(y_hat)
        mse += sum(error ** 2)
    mse /= n_points
    MSE.append(mse)

plt.figure()        
plt.plot(depth, MSE)
plt.xlabel('Tree Depth')
plt.ylabel('MSE')
plt.show()
