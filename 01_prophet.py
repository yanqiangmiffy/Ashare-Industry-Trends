#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 01_prophet.py 
@time: 2019-04-18 01:16
@description:使用facebook先知进行简单预测
"""
import pandas as pd
train_stock=pd.read_csv('input/TRAINSET_STOCK.csv')
# print(train_stock.head())
from dateutil.parser import parse
# print(parse('20140101'))
#
# from datetime import datetime
#
# print(datetime.strptime("24052017", '%d%m%Y'))
train_stock.rename(columns={'trade_date':'ds'},inplace=True)
train_stock.ds=train_stock.ds.apply(lambda x:parse(str(x)))
print(train_stock.ds[0])
# print(train_stock.tail())
# from fbprophet import Prophet
# # Python
# m = Prophet()
# m.fit(train_stock)
# # Python
# future = m.make_future_dataframe(periods=30)
# print(future.tail())
print(pd.to_datetime('20170401',format="%Y-%m-%d"))