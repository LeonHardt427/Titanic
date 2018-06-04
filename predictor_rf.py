#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/4 下午8:51
# @Author  : LeonHardt
# @File    : predictor_rf.py

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Imputer

x_train = pd.read_csv(os.getcwd()+"/data/x_train.csv")
y_label = np.loadtxt(os.getcwd()+"/data/y_train.txt", delimiter=',')
x_test = pd.read_csv(os.getcwd()+"/data/x_test.csv")

im = Imputer(strategy="mean")
x_train = im.fit_transform(x_train)
x_test = im.transform(x_test)
ex = ExtraTreesClassifier(n_estimators=70)
ex.fit(x_train, y_label)
y_test = ex.predict(x_test)

# print(y_test)
# print(y_test.shape)n

# index_test = x_test.index
result = pd.DataFrame(data=y_test, columns=["Survived"], index=range(892, 1310, 1), dtype=int).rename_axis("PassengerId", axis=0)
result = result.reset_index()
print(result)
# result.astype(int)
# result = result.reset_index()rarad
result.to_csv(os.getcwd()+"/update/result_ex70.csv", sep=",", index=False)