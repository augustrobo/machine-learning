##modify the code from Machine Learning in Python
##simple binary decision trees with different depths


import numpy as np
import pandas as pd
from urllib.request import urlopen
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

n_points = 100
x = np.arange(- 0.5, 0.5, 1/n_points)[:, np.newaxis]
y = x.ravel() + np.random.normal(0, 0.1, n_points)

simpleTree1 = DecisionTreeRegressor(max_depth = 1)
simpleTree1.fit(x, y)
yHat_1 = simpleTree1.predict(x)

simpleTree2 = DecisionTreeRegressor(max_depth = 3)
simpleTree2.fit(x, y)
yHat_2 = simpleTree2.predict(x)

##overfitting
simpleTree3 = DecisionTreeRegressor(max_depth = 6)
simpleTree3.fit(x, y)
yHat_3 = simpleTree3.predict(x)

plt.figure()
plt.plot(x, y, label = 'True y')
plt.plot(x, yHat_1, c = 'g', label = "max_depth = 1" , linewidth = 2)
plt.plot(x, yHat_2, c = 'r', label = "max_depth = 3" , linewidth = 2)
plt.plot(x, yHat_3, c = 'k', label = "max_depth = 6" , linewidth = 2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Binary Decision Tree Regression")
plt.legend()
plt.show()


