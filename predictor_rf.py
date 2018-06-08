#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/4 下午8:51
# @Author  : LeonHardt
# @File    : predictor_rf.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

x_train = pd.read_csv(os.getcwd() + "/data/x_train.csv")
x_test= pd.read_csv(os.getcwd() + "/data/x_test.csv")
y_label = x_train["Survived"]
print(y_label.shape)
x_train.drop("Survived", axis=1, inplace=True)
print(x_train.shape)
rf = RandomForestClassifier( n_estimators=7)
rf.fit(x_train, y_label)
y_test = rf.predict(x_test)

# print(y_test)
# print(y_test.shape)n

# index_test = x_test.index
result = pd.DataFrame(data=y_test, columns=["Survived"], index=range(892, 1310, 1), dtype=int).rename_axis("PassengerId", axis=0)
result = result.reset_index()
print(result)
# result.astype(int)
# result = result.reset_index()rarad
result.to_csv("result_rf.csv", sep=",", index=False)