#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: 03_xgb.py
@time: 2019-04-20 00:04
@description:
"""
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 设置
no_features = ['ts_code', 'trade_date', 'y']
features = [fea for fea in train.columns if fea not in no_features]  # 11

# 8.得到输入X ，输出y
train_id = train['ts_code'].values
y = train['y'].values.astype(int)
X = train[features].values
print("X shape:", X.shape)
print("y shape:", y.shape)

test_id = test['ts_code'].values
test_data = test[features].values
print("test shape", test_data.shape)
print("start：********************************")
start = time.time()

auc_list = []
pred_list = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 参数设置
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eta': 0.05,
              # 'max_depth':4,
              'min_child_weight': 6,
              'colsample_bytree': 0.7,
              'subsample': 0.7,
              # 'gamma':0,
              # 'lambda':1,
              # 'alpha ':0，
              'silent': 1
              }
    params['eval_metric'] = ['auc']
    # 数据结构
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtrain, 'train'), (dvali, 'valid')]  # 'valid-auc' will be used for early stopping
    # 模型train
    model = xgb.train(params, dtrain,
                      num_boost_round=1000,
                      evals=evallist,
                      early_stopping_rounds=50,
                      verbose_eval=100)
    # 预测验证
    pred = model.predict(dvali, ntree_limit=model.best_ntree_limit)
    # 评估
    auc = roc_auc_score(y_test, pred)
    print('...........................auc value:', auc)
    auc_list.append(auc)
    # 预测
    dtest = xgb.DMatrix(test_data)
    pre = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    pred_list.append(pre)

print('......................validate result mean :', np.mean(auc_list))

end = time.time()
print("......................run with time: ", (end - start) / 60.0)

print("over:*********************************")

# 11.5折结果均值融合，并保存文件
mean_auc = np.mean(auc_list)
print("mean auc:", mean_auc)
filepath = 'output/xgb_' + str(mean_auc) + '.csv'  # 线下平均分数
# 转为array
res = np.array(pred_list)
print("总的结果：", res.shape)
# 最后结果平均，mean
r = res.mean(axis=0)
print('result shape:', r.shape)
result = pd.DataFrame()
result['ts_code'] = test_id
result['trade_date'] = test['trade_date']
result['p'] = r
result.to_csv(filepath, index=False, sep=",")
