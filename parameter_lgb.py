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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
if __name__ == '__main__':

    x_train = pd.read_csv(os.getcwd()+"/data/x_train.csv")
    y_label = np.loadtxt(os.getcwd() + "/data/y_label.txt", delimiter=',')
    # print(y_label.shape)
    # x_train.drop("Survived", axis=1, inplace=True)
    # x_train = x_train.loc[:, :].values
    # print(x_train.shape)
    parameters = {}
    parameters["n_estimators"] = range(47, 100, 5)
    parameters["num_leaves"] = range(5, 15, 2)
    parameters["learning_rate"] = [0.05]
    # parameters["g/"]
    # im = Imputer(strategy="mean")
    # x_train = im.fit_transform(x_train)
    # x_test = im.transform(x_test)
    grid = GridSearchCV(estimator=lgb.LGBMClassifier(),
                        param_grid=parameters,
                        cv=5,
                        scoring='accuracy',
                        n_jobs=-1
                        )
    grid.fit(x_train, y_label)
    print(grid.best_params_)
    print(grid.best_score_)