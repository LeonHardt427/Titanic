#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/3 下午2:34
# @Author  : LeonHardt
# @File    : nlp.py

import os
import numpy as np
import pandas as pd


data_train = pd.read_csv(os.getcwd()+"/data/train.csv")
# print(data_train)
y_train = data_train.loc[:, "Survived"].values
np.savetxt(os.getcwd()+"/data/y_train.txt", y_train, delimiter=',')
data_train.drop(["Survived", "PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# data_train.fillna(-1, inplace=True)
for ind in data_train.index:
    # Sex
    if data_train.loc[ind, "Sex"] == "female":
        data_train.loc[ind, "Sex"] = 0
    else:
        data_train.loc[ind, "Sex"] = 1

    # Cabin
    data_train.loc[ind, "Cabin"] = str(data_train.loc[ind, "Cabin"])
    if "A" in data_train.loc[ind, "Cabin"]:
        data_train.loc[ind, "Cabin"] = 1
    elif "B" in data_train.loc[ind, "Cabin"]:
        data_train.loc[ind, "Cabin"] = 2
    elif "C" in data_train.loc[ind, "Cabin"]:
        data_train.loc[ind, "Cabin"] = 3
    elif "D" in data_train.loc[ind, "Cabin"]:
        data_train.loc[ind, "Cabin"] = 4
    elif "E" in data_train.loc[ind, "Cabin"]:
        data_train.loc[ind, "Cabin"] = 5
    elif "F" in data_train.loc[ind, "Cabin"]:
        data_train.loc[ind, "Cabin"] = 6
    elif "G" in data_train.loc[ind, "Cabin"]:
        data_train.loc[ind, "Cabin"] = 7
    elif "T" in data_train.loc[ind, "Cabin"]:
        data_train.loc[ind, "Cabin"] = 8
    data_train.loc[ind, "Cabin"] = float(data_train.loc[ind, "Cabin"])


    # Embarked
    data_train.loc[ind, "Embarked"] = str(data_train.loc[ind, "Embarked"])
    if "S" in data_train.loc[ind, "Embarked"]:
        data_train.loc[ind, "Embarked"] = 1
    elif "C" in data_train.loc[ind, "Embarked"]:
        data_train.loc[ind, "Embarked"] = 2
    elif "Q" in data_train.loc[ind, "Embarked"]:
        data_train.loc[ind, "Embarked"] = 3
    data_train.loc[ind, "Embarked"] = float(data_train.loc[ind, "Embarked"])

print(data_train)
data_train.to_csv(os.getcwd()+"/data/x_train.csv", sep=",", index=False)


# ------------------- test ---------------------

data_test = pd.read_csv(os.getcwd()+"/data/test.csv")
#
data_test.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
# data_test.fillna(0)
for ind in data_test.index:
    # Sex
    if data_test.loc[ind, "Sex"] == "female":
        data_test.loc[ind, "Sex"] = 0
    else:
        data_test.loc[ind, "Sex"] = 1

    # Cabin
    # print(type(type(data_test.loc[ind, "Cabin"])))
    data_test.loc[ind, "Cabin"] = str(data_test.loc[ind, "Cabin"])
    if "A" in data_test.loc[ind, "Cabin"]:
        data_test.loc[ind, "Cabin"] = 1
    elif "B" in data_test.loc[ind, "Cabin"]:
        data_test.loc[ind, "Cabin"] = 2
    elif "C" in data_test.loc[ind, "Cabin"]:
        data_test.loc[ind, "Cabin"] = 3
    elif "D" in data_test.loc[ind, "Cabin"]:
        data_test.loc[ind, "Cabin"] = 4
    elif "E" in data_test.loc[ind, "Cabin"]:
        data_test.loc[ind, "Cabin"] = 5
    elif "F" in data_test.loc[ind, "Cabin"]:
        data_test.loc[ind, "Cabin"] = 6
    elif "G" in data_test.loc[ind, "Cabin"]:
        data_test.loc[ind, "Cabin"] = 7
    data_test.loc[ind, "Cabin"] = float(data_test.loc[ind, "Cabin"])

    # Embarked
    data_test.loc[ind, "Embarked"] = str(data_test.loc[ind, "Embarked"])
    if "S" in data_test.loc[ind, "Embarked"]:
        data_test.loc[ind, "Embarked"] = 1
    elif "C" in data_test.loc[ind, "Embarked"]:
        data_test.loc[ind, "Embarked"] = 2
    elif "Q" in data_test.loc[ind, "Embarked"]:
        data_test.loc[ind, "Embarked"] = 3
    data_test.loc[ind, "Embarked"] = float(data_test.loc[ind, "Embarked"])

print(data_test)
data_test.to_csv(os.getcwd()+"/data/x_test.csv", sep=",", index=False)
