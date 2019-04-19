#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: data_process.py
@time: 2019-04-18 01:03
@description:
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
train_stock = pd.read_csv('input/TRAINSET_STOCK.csv')
# train_stock=pd.get_dummies(train_stock,columns=['name'])
train_stock['name']=lb.fit_transform(train_stock.name.values)
# 设置
train_stock['ts_code']=train_stock['ts_code'].astype('int32')
train_stock['trade_date']=train_stock['trade_date'].astype('int32')
train_stock['y']=train_stock['y'].astype('int32')

no_features = ['ts_code', 'trade_date']
features = [fea for fea in train_stock.columns if fea not in no_features]  # 11
period = 40
featurenum = len(features) * period
future_date = [20190402, 20190403, 20190404, 20190408, 20190409]
all_train,all_test=pd.DataFrame(),pd.DataFrame()
for index,group in tqdm(train_stock.groupby(by='ts_code')):
    # 生成训练集
    for _ in future_date:
        group = group.append(pd.Series(), ignore_index=True)
    group['ts_code'].iloc[-5:] = [group['ts_code'][0]] * 5
    group['trade_date'].iloc[-5:] = future_date

    cols, names = list(), list()
    for i in range(period, 0, -1):
        cols.append(group[features].shift(i))
        names += [col + '(day-%d)' % i for col in features]

    df = pd.concat(cols, axis=1)
    df.columns = names
    df = pd.concat([group[['ts_code', 'trade_date', 'y']], df], axis=1)

    train, test = df.iloc[:df.shape[0] - 5], df.iloc[-5:]
    train=train.dropna(subset=['pb(day-1)','pe(day-1)'],how="all")
    all_train=all_train.append(train)
    all_test=all_test.append(test)

all_train['ts_code']=all_train['ts_code'].astype('int32')
all_train['trade_date']=all_train['trade_date'].astype('int32')
all_train['y']=all_train['y'].astype('int32')


all_test['ts_code']=all_test['ts_code'].astype('int32')
all_test['trade_date']=all_test['trade_date'].astype('int32')

print(all_test.shape)
all_train.to_csv('input/train.csv', index=False)
all_test.to_csv('input/test.csv', index=False)
# .to_csv('input/test.csv', index=False)
