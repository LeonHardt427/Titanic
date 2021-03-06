#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/3 下午4:08
# @Author  : LeonHardt
# @File    : predictor_lgb.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

x_train = pd.read_csv(os.getcwd() + "/data/x_train.csv")
x_test= pd.read_csv(os.getcwd() + "/data/x_test.csv")
y_label = np.loadtxt(os.getcwd() + "/data/y_label.txt", delimiter=',')

# print(y_label.shape)
# x_train.drop("Survived", axis=1, inplace=True)
# print(x_train.shape)
gbm = lgb.LGBMClassifier(n_estimators=52, num_leaves=11, learning_rate=0.05)
gbm.fit(x_train, y_label)
y_test = gbm.predict(x_test)

# print(y_test)
# print(y_test.shape)n

# index_test = x_test.index
result = pd.DataFrame(data=y_test, columns=["Survived"], index=range(892, 1310, 1), dtype=int).rename_axis("PassengerId", axis=0)
result = result.reset_index()
print(result)
# result.astype(int)
# result = result.reset_index()rarad
result.to_csv("result_lgb.csv", sep=",", index=False)