##modify the code from Machine Learning in Python
##a simple binary decision tree for wine data

import numpy as np
import pandas as pd
from urllib.request import urlopen
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import matplotlib.pyplot as plt
import graphviz as gv

##read data
##url address
target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = pd.read_table(urlopen(target_url), sep = ";")

##print the first 5 lines
##print(data.head())

labels = data['quality']
x = data.drop('quality', axis = 1)
wineTree = DecisionTreeRegressor(max_depth = 3)
wineTree.fit(x, labels)

export_graphviz(wineTree, out_file = "wineTreeBDT.dot", feature_names = x.columns)
with open("wineTreeBDT.dot") as f:
    dot_graph = f.read()
    gv.Source(dot_graph)
