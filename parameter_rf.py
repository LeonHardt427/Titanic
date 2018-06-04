#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/4 下午8:52
# @Author  : LeonHardt
# @File    : parameter_rf.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

x_train = pd.read_csv(os.getcwd()+"/data/x_train.csv")
y_label = np.loadtxt(os.getcwd()+"/data/y_train.txt", delimiter=',')

im = Imputer(strategy="mean")
x_train = im.fit_transform(x_train)
parameters = {}
parameters["n_estimators"] = range(50, 200, 10)
# parameters["num_leaves"] = range(10, 50, 5)
# parameters["learning_rate"] = [0.05]
# parameters["g/"]

grid = GridSearchCV(estimator=ExtraTreesClassifier(random_state=1),
                    param_grid=parameters,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                    )
grid.fit(x_train, y_label)
print(grid.best_params_)