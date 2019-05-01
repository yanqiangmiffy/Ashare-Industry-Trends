#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 02_lgb.py 
@time: 2019-04-19 15:34
@description:
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
train=pd.read_csv('input/train.csv')
test=pd.read_csv('input/test.csv')

# 设置
no_features = ['ts_code', 'trade_date', 'y']
features = [fea for fea in train.columns if fea not in no_features]  # 11

# 8.得到输入X ，输出y
train_id = train['ts_code'].values
y = train['y'].values.astype(int)
X = train[features].values
print("X shape:",X.shape)
print("y shape:",y.shape)

test_id = test['ts_code'].values
test_data = test[features].values
print("test shape",test_data.shape)

# 9.开始训练
# 采取分层采样
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score

print("start：********************************")
start = time.time()

N = 5
skf = KFold(n_splits=N,shuffle=True,random_state=2018)

auc_cv = []
pred_cv = []
for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], \
                                       y[train_in], y[test_in]

    # 数据结构
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 设置参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        # 'max_depth': 4,
        # 'min_child_weight': 6,
        'num_leaves': 128,
        'learning_rate': 0.1,  # 0.05
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'reg_alpha':0.3,
        'reg_lambda':0.3,
        'min_data_in_leaf' :18,
        'min_sum_hessian_in_leaf' :0.001,
        'n_jobs' :-1,
        'num_threads':8,

    }

    print('................Start training {} fold..........................'.format(k+1))
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    verbose_eval=100,feature_name=features)
    lgb.plot_importance(gbm,max_num_features=20)
    plt.show()
    print('................Start predict .........................')
    # 预测
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # 评估
    tmp_auc = roc_auc_score(y_test, y_pred)
    auc_cv.append(tmp_auc)
    print("valid auc:", tmp_auc)
    # test
    pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
    pred_cv.append(pred)

    # K交叉验证的平均分数
print('the cv information:')
print(auc_cv)
print('cv mean score', np.mean(auc_cv))

end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 10.5折交叉验证结果均值融合，保存文件
mean_auc = np.mean(auc_cv)
print("mean auc:", mean_auc)
filepath = 'output/lgb_' + str(mean_auc) + '.csv'  # 线下平均分数

# 转为array
res = np.array(pred_cv)
print("总的结果：", res.shape)
# 最后结果平均，mean
r = res.mean(axis=0)
print('result shape:', r.shape)
result = pd.DataFrame()
result['ts_code'] = test_id
result['trade_date']=test['trade_date']
result['p'] = r
result.to_csv(filepath, index=False, sep=",")
