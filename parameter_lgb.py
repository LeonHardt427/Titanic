#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/3 下午4:08
# @Author  : LeonHardt
# @File    : parameter_lgb.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

x_train = pd.read_csv(os.getcwd()+"/data/x_train.csv")
y_label = np.loadtxt(os.getcwd()+"/data/y_train.txt", delimiter=',')

parameters = {}
parameters["n_estimators"] = range(50, 150, 20)
parameters["num_leaves"] = range(10, 50, 5)
parameters["learning_rate"] = [0.05]
# parameters["g/"]
# im = Imputer(strategy="mean")
# x_train = im.fit_transform(x_train)
# x_test = im.transform(x_test)

grid = GridSearchCV(estimator=lgb.LGBMClassifier(random_state=1),
                    param_grid=parameters,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                    )
grid.fit(x_train, y_label)
print(grid.best_params_)
print(grid.best_score_)