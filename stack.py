#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: stack.py 
@time: 2019-05-01 23:18
@description:
"""
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso,SGDClassifier
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from scipy.stats import skew
from collections import OrderedDict

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 设置
no_features = ['ts_code', 'trade_date', 'y']
features = [fea for fea in train.columns if fea not in no_features]  # 11

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
# 8.得到输入X ，输出y
train_id = train['ts_code'].values
y = train['y'].values.astype(int)
X = train[features].values
print("X shape:", X.shape)
print("y shape:", y.shape)

test_id = test['ts_code'].values
test_data = test[features].values
print("test shape", test_data.shape)

xg_reg = xgboost.XGBClassifier(colsample_bytree=0.4,
                               gamma=0,
                               learning_rate=0.07,
                               max_depth=3,
                               # min_child_weight=1.5,
                               n_estimators=2000,
                               reg_alpha=0.75,
                               reg_lambda=0.45,
                               subsample=0.6,
                               seed=42)

xg_reg.fit(X, y, verbose=True)

sgd=SGDClassifier(loss='log')
sgd.fit(X, y)

GBoost = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05,
                                    max_depth=4,
                                    min_samples_leaf=15, min_samples_split=10,
                                    random_state=5)
GBoost.fit(X, y)

YpredictedonTrain = xg_reg.predict_proba(X)[:,1]
Ypredicted2onTrain = GBoost.predict_proba(X)[:,1]
Ypredicted3onTrain = sgd.predict_proba(X)[:,1]
print(YpredictedonTrain.shape)
print(Ypredicted2onTrain.shape)
print(Ypredicted3onTrain.shape)

dfinal = pd.DataFrame({'a': YpredictedonTrain, 'b': Ypredicted2onTrain, 'c': Ypredicted3onTrain})
dfinal.head()
model_lgb = lgb.LGBMClassifier(num_leaves=200,
                               learning_rate=0.05, n_estimators=720,
                               max_bin=55, bagging_fraction=0.8,
                               bagging_freq=5, feature_fraction=0.2319,
                               feature_fraction_seed=9, bagging_seed=9,
                               min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
model_lgb.fit(dfinal, y)

a = xg_reg.predict_proba(test_data)[:,1]
b = GBoost.predict_proba(test_data)[:,1]
c = sgd.predict_proba(test_data)[:,1]
dStack = pd.DataFrame({'a': a, 'b': b, 'c': c})
dStack.head()

output = model_lgb.predict_proba(dStack)[:,1]

print('result shape:', output.shape)
result = pd.DataFrame()
result['ts_code'] = test_id
result['trade_date'] = test['trade_date']
result['p'] = output
result.to_csv('stack.csv', index=False, sep=",")
